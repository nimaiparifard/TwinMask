"""HeaRT-style hard negatives for link prediction (Li et al., "Evaluating Graph Neural Networks for Link
Prediction: Current Pitfalls and New Benchmarking", NeurIPS 2023 D&B).

For each test positive (a, b) we corrupt one endpoint: K/2 negatives (a, c) and K/2 negatives (c, b), where
the candidates c are the nodes that a combination of heuristics ranks highest
(Resource Allocation, Personalized PageRank, text-embedding cosine; aggregated by mean rank), excluding
every true edge of the full graph and cold-start nodes. Heuristics are computed on the *training* graph only.
Each positive is ranked against its own K negatives -> MRR, Hits@K.

Deviations from the paper (documented): K=100 instead of 500 (cost of the token-level model), at most 1000
test positives per dataset, PPR by 30 power iterations (alpha=0.15).
"""
import argparse
import os

import numpy as np
import torch

from .data import PROC, load


def _norm_adj(ei, n, dev):
    ei = torch.cat([ei, ei.flip(0)], 1).to(dev)
    deg = torch.bincount(ei[0], minlength=n).float()
    return ei, deg


def build(name, K=100, max_pos=1000, seed=0, dev="cuda"):
    d = load(name)
    n = d["num_nodes"]
    rng = np.random.default_rng(seed)
    tp = d["test_pos"]
    if tp.shape[1] > max_pos:
        tp = tp[:, torch.from_numpy(np.sort(rng.choice(tp.shape[1], max_pos, replace=False)))]
    ei, deg = _norm_adj(d["train_pos"], n, dev)
    A = torch.sparse_coo_tensor(ei, torch.ones(ei.shape[1], device=dev), (n, n)).coalesce()
    dinv = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))
    P = torch.sparse_coo_tensor(ei, dinv[ei[1]], (n, n)).coalesce()          # A D^-1 (column-normalised)
    emb = d["emb"].to(dev)
    # forbidden: all true edges of the full graph (train/val/test/cold) + cold nodes
    allE = torch.cat([d["train_pos"], d["val_pos"], d["test_pos"], d["cold_pos"]], 1)
    nbr = [set() for _ in range(n)]
    for u, v in allE.t().tolist():
        nbr[u].add(v); nbr[v].add(u)
    cold = torch.zeros(n, dtype=torch.bool, device=dev); cold[d["cold_nodes"].to(dev)] = True

    def candidates(src, other):  # src: (B,) node ids. returns (B, K//2) top candidates c for pairs (src, c)
        B = len(src)
        S = torch.zeros(n, B, device=dev); S[src, torch.arange(B, device=dev)] = 1.0
        ra = torch.sparse.mm(A, dinv[:, None] * torch.sparse.mm(A, S))       # RA(src, c) = sum_z A_sz A_zc / d_z
        x = S.clone()
        for _ in range(30):
            x = 0.85 * torch.sparse.mm(P, x) + 0.15 * S
        ppr = x
        cos = emb @ emb[src].t()
        scores = [ra.t(), ppr.t(), cos.t()]                                   # each (B, n)
        rank = torch.zeros(B, n, device=dev)
        for s in scores:
            s = s + 1e-9 * torch.rand_like(s)                                 # random tie-breaking
            rank += torch.argsort(torch.argsort(-s, 1), 1).float()
        rank[:, cold] = float("inf")
        rank[torch.arange(B, device=dev), src] = float("inf")
        out = []
        for i, (s_, o_) in enumerate(zip(src.tolist(), other.tolist())):
            bad = list(nbr[s_] | {o_})
            rank[i, bad] = float("inf")
        top = torch.topk(-rank, K // 2, dim=1).indices
        return top

    negs = []
    bs = 64
    for s in range(0, tp.shape[1], bs):
        a, b = tp[0, s:s + bs].to(dev), tp[1, s:s + bs].to(dev)
        ca = candidates(a, b)                    # pairs (a, c)
        cb = candidates(b, a)                    # pairs (c, b)
        na = torch.stack([a[:, None].expand_as(ca), ca], -1)
        nb = torch.stack([cb, b[:, None].expand_as(cb)], -1)
        negs.append(torch.cat([na, nb], 1).cpu())
    negs = torch.cat(negs)                        # (P, K, 2)
    out = dict(pos=tp, neg=negs, K=K)
    torch.save(out, os.path.join(PROC, f"{name}_heart.pt"))
    print(f"[{name}] HeaRT: {tp.shape[1]} positives x {K} hard negatives")
    return out


def load_heart(name):
    p = os.path.join(PROC, f"{name}_heart.pt")
    return torch.load(p) if os.path.exists(p) else None


def rank_metrics(pos_score, neg_score):
    """pos_score (P,), neg_score (P,K). OGB-style rank with ties split between optimistic and pessimistic."""
    pos_score, neg_score = torch.as_tensor(pos_score).float(), torch.as_tensor(neg_score).float()
    opt = (neg_score > pos_score[:, None]).sum(1).float()
    pes = (neg_score >= pos_score[:, None]).sum(1).float()
    rank = 0.5 * (opt + pes) + 1
    res = dict(mrr=float((1.0 / rank).mean()))
    for k in (1, 10, 20):
        res[f"hits@{k}"] = float((rank <= k).float().mean())
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cora", "citeseer", "pubmed", "wikics", "instagram", "reddit", "arxiv"])
    a = ap.parse_args()
    for nm in a.datasets:
        build(nm)
