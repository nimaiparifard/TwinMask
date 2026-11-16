import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def lp_metrics(pos, neg):
    pos, neg = np.asarray(pos, dtype=np.float64), np.asarray(neg, dtype=np.float64)
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[pos, neg]
    return dict(auc=float(roc_auc_score(y, s)), ap=float(average_precision_score(y, s)))


def _lr(h, y, tr, te, seed):
    sc = StandardScaler().fit(h[tr])
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit(sc.transform(h[tr]), y[tr])
    p = clf.predict(sc.transform(h[te]))
    return float(accuracy_score(y[te], p)), float(f1_score(y[te], p, average="macro"))


def probe_nc(h, d, seed=0, n_splits=10, nodes=None):
    """Linear probe (logistic regression) on frozen, label-free node representations. Two settings:
      low-label  : public semi-supervised split (e.g. Planetoid 20 labels/class)      -> acc, f1
      supervised : 60/20/20 random splits over `nodes` (default all), mean over n_splits -> sup_acc, sup_f1 (+std)
    """
    h = h.detach().float().cpu().numpy() if torch.is_tensor(h) else h
    y = d["y"].numpy()
    tr, te = d["train_mask"].numpy(), d["test_mask"].numpy()
    acc, f1 = _lr(h, y, tr, te, seed)
    nodes = np.arange(len(y)) if nodes is None else np.asarray(nodes)
    accs, f1s = [], []
    for s in range(n_splits):
        p = np.random.default_rng(1000 + s).permutation(nodes)
        a_, f_ = _lr(h, y, p[:int(.6 * len(p))], p[int(.8 * len(p)):], seed)
        accs.append(a_); f1s.append(f_)
    return dict(acc=acc, f1=f1, sup_acc=float(np.mean(accs)), sup_acc_std=float(np.std(accs)),
                sup_f1=float(np.mean(f1s)), sup_f1_std=float(np.std(f1s)))


class TimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):  # t in [0,1], any shape -> (..., dim)
        half = self.dim // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t.float().unsqueeze(-1) * 1000 * f
        return self.mlp(torch.cat([a.sin(), a.cos()], -1))


def sample_t(n, device, p_zero=0.1, p_one=0.1, eps=1e-3):
    """Antithetic (low-discrepancy) uniform times with boundary modes.
    t=0 -> modality fully observed (conditional), t=1 -> fully masked (marginal)."""
    u = (torch.rand(1, device=device) + torch.arange(n, device=device) / n) % 1.0
    t = eps + (1 - eps) * u[torch.randperm(n, device=device)]
    r = torch.rand(n, device=device)
    t = torch.where(r < p_zero, torch.zeros_like(t), t)
    t = torch.where((r >= p_zero) & (r < p_zero + p_one), torch.ones_like(t), t)
    return t


def save_json(obj, *path):
    p = os.path.join(RESULTS, *path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return p
