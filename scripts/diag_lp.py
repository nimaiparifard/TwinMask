"""Diagnose TwinMask edge learning: train-edge vs test-edge AUC under the same protocol, and subgraph edge density."""
import sys

import numpy as np
import torch

from twinmask.common import lp_metrics
from twinmask.data import load
from twinmask.sampler import SubgraphSampler
from twinmask.tm import load_model, score_pairs

name = sys.argv[1] if len(sys.argv) > 1 else "cora"
d = load(name)
m = load_model(name, "joint", 0)
S = SubgraphSampler(d, k=12, seed=5)
rng = np.random.default_rng(0)
tr = d["train_pos"][:, torch.from_numpy(rng.choice(d["train_pos"].shape[1], 400, replace=False))]
neg = d["test_neg"][:, :400]
print("train-edge AUC (edge masked, rest visible):", lp_metrics(score_pairs(m, S, tr, "full", k=12), score_pairs(m, S, neg, "full", k=12)))
print("test-edge  AUC:", lp_metrics(score_pairs(m, S, d["test_pos"][:, :400], "full", k=12), score_pairs(m, S, neg, "full", k=12)))
print("pair-only (text only) train AUC:", lp_metrics(score_pairs(m, S, tr, "paironly"), score_pairs(m, S, neg, "paironly")))
dens = []
for _ in range(50):
    b = S.train_batch(16)
    v = b["valid"]; iu = torch.triu(torch.ones(v.shape[1], v.shape[1], dtype=torch.bool), 1)
    p = iu[None] & v[:, :, None] & v[:, None, :]
    dens.append((b["adj"][p].float().mean()).item())
dd = float(np.mean(dens))
print(f"edge density in training subgraphs: {dd:.3f}  -> base-rate BCE = {-(dd*np.log(dd)+(1-dd)*np.log(1-dd)):.3f}")
