from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix

from twinmask.data import ROOT, load, undirected_unique


def edge_proposals(adj, text, seed=721):
    """Choose up to two positive-cosine, two-hop nonedges per node."""
    rng = np.random.default_rng(seed)
    neighbors = np.split(adj.indices, adj.indptr[1:-1])
    proposals = np.full((len(neighbors), 2), -1, dtype=np.int64)
    for node, direct in enumerate(neighbors):
        if not len(direct):
            continue
        intermediates = rng.choice(direct, min(len(direct), 8), replace=False)
        candidates = np.unique(np.concatenate([
            rng.choice(neighbors[mid], min(len(neighbors[mid]), 12), replace=False)
            for mid in intermediates
        ]))
        candidates = np.setdiff1d(candidates, np.append(direct, node))
        similarity = (text[candidates] * text[node]).sum(1)
        order = np.argsort(-similarity, kind="stable")
        best = candidates[order[similarity[order] > 0][:2]]
        proposals[node, :len(best)] = best
    return proposals


def prepare(name, device):
    data = load(name)
    n = data["num_nodes"]
    pairs = undirected_unique(torch.cat([data[key] for key in
        ("train_pos", "val_pos", "test_pos", "cold_pos")], dim=1), n)
    src, dst = torch.cat([pairs, pairs.flip(0)], dim=1).numpy()
    adj = csr_matrix((np.ones(len(src), dtype=np.float32), (src, dst)), shape=(n, n))
    degree = np.diff(adj.indptr)
    text = F.normalize(data["emb"].float()).numpy()
    proposals = edge_proposals(adj, text)
    # Fixed unlabeled semantic anchors for coverage, shared by all selectors.
    anchors = np.random.default_rng(38).choice(n, 24, replace=False)
    group = (text @ text[anchors].T).argmax(1)
    cache = Path(ROOT, "checkpoints", f"emb_{name}_joint_s0.pt")
    diffusion = torch.load(cache, map_location="cpu", weights_only=True).float()
    views = []
    for x in (data["emb"].float(), diffusion):
        x = (x - x.mean(0)) / x.std(0).clamp_min(1e-5)
        sums = torch.from_numpy(adj @ x.numpy()).to(device)
        deg = torch.tensor(degree, device=device)[:, None]
        views.append((x.to(device), sums, sums / deg.clamp_min(1)))
    return views, deg, torch.tensor(proposals, device=device), group, data["y"].flatten(), len(data["label_names"])
