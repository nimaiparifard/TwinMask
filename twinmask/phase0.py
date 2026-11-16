"""Phase 0 — embedding-level joint diffusion (the original raw idea, made rigorous):
Gaussian diffusion on MiniLM text embeddings (per-node time t_x) + absorbing-state masking of
edges (time t_a), shared GCN denoiser. Variants differ only in how (t_x, t_a) are sampled.

variants:
  joint      independent t_x (per node) and t_a (per step)     -> learns p(X|A), p(A|X), joint
  tied       t_x = t_a for all nodes
  text_only  t_a = 0 (all train edges visible, no edge loss)    ~ GraphMAE-style diffusion
  edge_only  t_x = 0 (clean text)                               ~ MaskGAE-style with text features
  mlp_lp     t_x = 0, t_a = 1 (no graph ever visible)           text-only link predictor
  gae        no masking at all, GCN-GAE on all train edges       classic GAE baseline
"""
import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected

from .common import TimeEmb, lp_metrics, probe_nc, sample_t, save_json, set_seed
from .data import load
from .heart import load_heart, rank_metrics


def alpha_bar(t):  # cosine schedule
    return torch.cos(0.5 * math.pi * t).clamp(min=1e-4) ** 2


class Denoiser(nn.Module):
    def __init__(self, d_in, hid=256, layers=2):
        super().__init__()
        self.inp = nn.Linear(d_in, hid)
        self.tx, self.ta = TimeEmb(hid), TimeEmb(hid)
        self.convs = nn.ModuleList([GCNConv(hid, hid) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])
        self.x_head = nn.Sequential(nn.Linear(2 * hid, hid), nn.SiLU(), nn.Linear(hid, d_in))
        self.e_head = nn.Sequential(nn.Linear(2 * hid, hid), nn.SiLU(), nn.Linear(hid, 1))

    def encode(self, x_t, t_x, t_a, ei):
        h0 = self.inp(x_t) + self.tx(t_x) + self.ta(t_a).expand(x_t.shape[0], -1)
        h = h0
        for conv, ln in zip(self.convs, self.norms):
            h = h + F.silu(conv(ln(h), ei))
        return torch.cat([h0, h], -1), h

    def x0(self, hc, x_t, t_x):
        # skip-parameterisation: exact at t=0, pure prediction at t=1
        ab = alpha_bar(t_x).unsqueeze(-1)
        return ab.sqrt() * x_t + (1 - ab).sqrt() * self.x_head(hc)

    def edge_logit(self, h, pairs):
        a, b = h[pairs[0]], h[pairs[1]]
        return self.e_head(torch.cat([a * b, (a - b).abs()], -1)).squeeze(-1)


def train_return_model(name, variant, seed=0, epochs=1000, data=None, **kw):
    return run(name, variant, seed, epochs=epochs, quiet=True, return_model=True, data=data, **kw)


def run(name, variant, seed=0, epochs=1500, lr=1e-3, wd=1e-5, lam_e=1.0, dev="cuda", quiet=False, return_model=False,
        data=None, hid=256, p_bound=0.1, select_lp=True):
    """data: optional override of the processed dataset dict (e.g. other text features in 'emb', other edges)."""
    set_seed(seed)
    d = load(name) if data is None else data
    n = d["num_nodes"]
    emb = d["emb"].to(dev)
    mu, sd = emb.mean(0), emb.std(0) + 1e-6
    x0 = (emb - mu) / sd
    tr = d["train_pos"].to(dev)
    is_cold = torch.zeros(n, dtype=torch.bool, device=dev); is_cold[d["cold_nodes"].to(dev)] = True
    warm = torch.nonzero(~is_cold).view(-1)
    model = Denoiser(x0.shape[1], hid=hid).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    E = tr.shape[1]
    best, best_state, t0 = -1, None, time.time()

    def full_embed(x_in=None, t_x=None, edges=tr):
        x_in = x0 if x_in is None else x_in
        t_x = torch.zeros(n, device=dev) if t_x is None else t_x
        ei = to_undirected(edges, num_nodes=n) if edges.numel() else edges
        return model.encode(x_in, t_x, torch.zeros(1, device=dev), ei)

    for ep in range(1, epochs + 1):
        model.train()
        # ---- sample noise levels per variant
        if variant in ("edge_only", "mlp_lp", "gae"):
            t_x = torch.zeros(n, device=dev)
        else:
            t_x = sample_t(n, dev, p_bound, p_bound)
        if variant == "text_only" or variant == "gae":
            t_a = torch.zeros(1, device=dev)
        elif variant == "mlp_lp":
            t_a = torch.ones(1, device=dev)
        elif variant == "tied":
            t_a = sample_t(1, dev, p_bound, p_bound); t_x = t_a.expand(n).clone()
        else:
            t_a = sample_t(1, dev, p_bound, p_bound)
        # ---- corrupt
        ab = alpha_bar(t_x).unsqueeze(-1)
        x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * torch.randn_like(x0)
        if variant == "gae":
            vis, msk = tr, tr  # reconstruct all (visible) edges
        else:
            m = torch.rand(E, device=dev) < t_a
            vis, msk = tr[:, ~m], tr[:, m]
        ei = to_undirected(vis, num_nodes=n) if vis.numel() else vis
        hc, h = model.encode(x_t, t_x, t_a, ei)
        loss = torch.zeros((), device=dev)
        # text loss: x0-prediction MSE on noised nodes
        if variant not in ("edge_only", "mlp_lp", "gae"):
            w = (t_x > 0).float()
            mse = ((model.x0(hc, x_t, t_x) - x0) ** 2).mean(-1)
            loss = loss + (w * mse).sum() / w.sum().clamp(min=1)
        # edge loss on masked positives vs random warm negatives
        if variant != "text_only" and msk.shape[1] > 0:
            k = min(msk.shape[1], 20000)
            pos = msk[:, torch.randperm(msk.shape[1], device=dev)[:k]]
            neg = warm[torch.randint(len(warm), (2, k), device=dev)]
            lp, ln = model.edge_logit(h, pos), model.edge_logit(h, neg)
            loss = loss + lam_e * (F.binary_cross_entropy_with_logits(lp, torch.ones_like(lp)) +
                                   F.binary_cross_entropy_with_logits(ln, torch.zeros_like(ln)))
        if loss.requires_grad:
            opt.zero_grad(); loss.backward(); opt.step()

        if select_lp and ep % 50 == 0 and variant != "text_only":
            model.eval()
            with torch.no_grad():
                _, h = full_embed()
                vp = torch.sigmoid(model.edge_logit(h, d["val_pos"].to(dev))).cpu()
                vn = torch.sigmoid(model.edge_logit(h, d["val_neg"].to(dev))).cpu()
                auc = lp_metrics(vp, vn)["auc"]
            if auc > best:
                best, best_state = auc, {k: v.clone() for k, v in model.state_dict().items()}
            if not quiet and ep % 250 == 0:
                print(f"  [{name}/{variant}] ep{ep} loss={loss.item():.4f} valAUC={auc:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    if return_model:
        return model, x0, mu, sd
    res = dict(dataset=name, variant=variant, seed=seed, train_sec=time.time() - t0)
    with torch.no_grad():
        hc, h = full_embed()
        if variant != "text_only":
            sc = lambda p: torch.sigmoid(model.edge_logit(h, p.to(dev))).cpu()
            res["lp"] = lp_metrics(sc(d["test_pos"]), sc(d["test_neg"]))
            res["cold"] = lp_metrics(sc(d["cold_pos"]), sc(d["cold_neg"]))
            # LP from text alone (graph hidden): how much does the edge head rely on structure?
            _, h_ng = full_embed(edges=tr[:, :0])
            sc2 = lambda p: torch.sigmoid(model.edge_logit(h_ng, p.to(dev))).cpu()
            res["lp_nograph"] = lp_metrics(sc2(d["test_pos"]), sc2(d["test_neg"]))
            hr = load_heart(name)
            if hr is not None:
                P, K = hr["neg"].shape[:2]
                res["heart"] = rank_metrics(sc(hr["pos"]), sc(hr["neg"].reshape(-1, 2).t()).view(P, K))
        res["nc"] = probe_nc(h, d, seed)
        # text-embedding denoising: noise only test nodes, compare graph-visible vs graph-hidden
        if variant not in ("edge_only", "mlp_lp", "gae"):
            te = d["test_mask"].to(dev)
            den = {}
            for tl in (0.3, 0.5, 0.7, 0.9):
                g = torch.Generator(device=dev).manual_seed(seed)
                t_x = torch.where(te, torch.full((n,), tl, device=dev), torch.zeros(n, device=dev))
                ab = alpha_bar(t_x).unsqueeze(-1)
                x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * torch.randn(x0.shape, device=dev, generator=g)
                out = {}
                for tag, edges in (("graph", tr), ("nograph", tr[:, :0])):
                    hc2, _ = full_embed(x_t, t_x, edges)
                    xh = model.x0(hc2, x_t, t_x)[te] * sd + mu
                    out[tag] = F.cosine_similarity(xh, emb[te], dim=-1).mean().item()
                noisy = (x_t[te] * sd + mu)
                out["noisy_input"] = F.cosine_similarity(noisy, emb[te], dim=-1).mean().item()
                den[str(tl)] = out
            res["denoise_cos"] = den
    return res


def baselines(name, seed=0):
    """Training-free / supervised reference points."""
    d = load(name)
    emb = d["emb"]
    cos = lambda p: (emb[p[0]] * emb[p[1]]).sum(-1)
    hr = load_heart(name)
    extra = {}
    if hr is not None:
        P, K = hr["neg"].shape[:2]
        extra["heart"] = rank_metrics(cos(hr["pos"]), cos(hr["neg"].reshape(-1, 2).t()).view(P, K))
    return dict(dataset=name, variant="text_cosine", seed=seed, **extra,
                lp=lp_metrics(cos(d["test_pos"]), cos(d["test_neg"])),
                cold=lp_metrics(cos(d["cold_pos"]), cos(d["cold_neg"])),
                nc=probe_nc(emb, d, seed))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cora", "citeseer", "pubmed", "wikics", "instagram", "reddit"])
    ap.add_argument("--variants", nargs="+", default=["joint", "tied", "text_only", "edge_only", "mlp_lp", "gae"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=1500)
    a = ap.parse_args()
    allres = []
    for nm in a.datasets:
        for s in a.seeds:
            allres.append(baselines(nm, s))
            for v in a.variants:
                r = run(nm, v, s, epochs=a.epochs)
                print({k: r[k] for k in r if k in ("dataset", "variant", "seed", "lp", "cold", "nc", "train_sec")}, flush=True)
                allres.append(r)
        save_json(allres, "phase0", "phase0_all.json")
