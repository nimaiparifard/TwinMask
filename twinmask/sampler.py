"""Subgraph sampling over the *training* graph (validation/test/cold edges are never visible)."""
import numpy as np
import scipy.sparse as sp
import torch

from .model import ABSENT, MASKED, PRESENT, SELF


class SubgraphSampler:
    def __init__(self, d, k=16, seed=0):
        self.n, self.k = d["num_nodes"], k
        tp = d["train_pos"].numpy()
        A = sp.coo_matrix((np.ones(tp.shape[1] * 2, dtype=np.int8),
                           (np.r_[tp[0], tp[1]], np.r_[tp[1], tp[0]])), shape=(self.n, self.n)).tocsr()
        A.data[:] = 1
        self.A = A
        self.tp = tp
        self.tokens = d["tokens"]
        self.rng = np.random.default_rng(seed)
        cold = np.zeros(self.n, bool); cold[d["cold_nodes"].numpy()] = True
        self.cold = cold
        self.warm = np.nonzero(~cold)[0]

    def nbrs(self, u):
        return self.A.indices[self.A.indptr[u]:self.A.indptr[u + 1]]

    def gather(self, anchors, k=None):
        """anchors first, then shuffled 1-hop neighbours of anchors (round-robin), then 2-hop."""
        k = k or self.k
        rng = self.rng
        out = list(dict.fromkeys(int(a) for a in anchors))
        seen = set(out)
        frontier = out[:]
        for _hop in range(2):
            lists = [rng.permutation(self.nbrs(u)) for u in frontier]
            nxt = []
            i = 0
            while len(out) < k and any(i < len(l) for l in lists):
                for l in lists:
                    if i < len(l) and len(out) < k:
                        v = int(l[i])
                        if v not in seen:
                            seen.add(v); out.append(v); nxt.append(v)
                i += 1
            frontier = nxt
            if len(out) >= k:
                break
        return out

    def build(self, node_lists, mask_pairs=None, unknown_nodes=None):
        """Pack subgraphs into tensors. mask_pairs[b]: list of (local i, local j) to force [M];
        unknown_nodes[b]: local indices whose every edge is [M] (cold-start)."""
        B, k = len(node_lists), max(len(x) for x in node_lists)
        L = self.tokens.shape[1]
        ids = torch.full((B, k, L), 1, dtype=torch.int32)  # roberta pad id = 1
        valid = torch.zeros(B, k, dtype=torch.bool)
        adj = torch.zeros(B, k, k, dtype=torch.long)
        for b, nodes in enumerate(node_lists):
            m = len(nodes)
            ids[b, :m] = self.tokens[nodes]
            valid[b, :m] = True
            adj[b, :m, :m] = torch.from_numpy(self.A[nodes][:, nodes].toarray().astype(np.int64))
        state = torch.where(adj > 0, PRESENT, ABSENT)
        eye = torch.eye(k, dtype=torch.bool)[None].expand(B, k, k)
        state = torch.where(eye, torch.full_like(state, SELF), state)
        forced = torch.zeros(B, k, k, dtype=torch.bool)
        for b in range(B):
            for (i, j) in (mask_pairs[b] if mask_pairs else []):
                forced[b, i, j] = forced[b, j, i] = True
            for i in (unknown_nodes[b] if unknown_nodes else []):
                forced[b, i, :] = forced[b, :, i] = True
        forced &= ~eye
        attn = ids != 1
        return dict(ids=ids, attn=attn, valid=valid, adj=adj, state=state, forced=forced)

    # ---------------- training batches
    def train_batch(self, B):
        lists = []
        for _ in range(B):
            r = self.rng.random()
            if r < 0.5:  # anchored on a training edge
                e = self.rng.integers(self.tp.shape[1])
                anchors = [self.tp[0, e], self.tp[1, e]]
            elif r < 0.75:  # anchored on a random warm pair (mostly non-edges)
                anchors = list(self.rng.choice(self.warm, 2, replace=False))
            else:  # node-centric
                anchors = [self.rng.choice(self.warm)]
            lists.append(self.gather(anchors))
        return self.build(lists)
