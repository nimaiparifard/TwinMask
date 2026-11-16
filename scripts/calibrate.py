"""Where does the NC gap to published numbers come from? Controlled checks on the same data/features.

A. Supervised GCN on our MiniLM features (public low-label split)  -> is the data/feature pipeline sane?
     A1 full graph   A2 LP-split graph (train edges only, as our SSL models see it)
B. Phase-0 joint SSL model + linear probe
     B1 as reported (LP graph, C=1)   B2 full graph at inference   B3 full graph + C tuned on validation
     B4 = B3 on [text emb ‖ h] (concat raw text features)
C. Same probes with an MLP on raw MiniLM (no graph) for reference.
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected

sys.path.insert(0, ".")
from twinmask.common import set_seed  # noqa: E402
from twinmask.data import DIVERGE_DIR, load, undirected_unique  # noqa: E402
from twinmask import phase0  # noqa: E402

dev = "cuda"


def full_edges(name, n):
    g = torch.load(f"{DIVERGE_DIR}/{name}.pt", weights_only=False, map_location="cpu")
    return undirected_unique(g.edge_index, n)


def sup_gcn(x, ei, y, tr, va, te, seed):
    set_seed(seed)
    c1, c2 = GCNConv(x.shape[1], 256).to(dev), GCNConv(256, int(y.max()) + 1).to(dev)
    opt = torch.optim.Adam(list(c1.parameters()) + list(c2.parameters()), lr=0.01, weight_decay=5e-4)
    best, bt = 0, 0
    for _ in range(300):
        c1.train(); c2.train()
        out = c2(F.dropout(F.relu(c1(F.dropout(x, 0.5, True), ei)), 0.5, True), ei)
        loss = F.cross_entropy(out[tr], y[tr]); opt.zero_grad(); loss.backward(); opt.step()
        c1.eval(); c2.eval()
        with torch.no_grad():
            p = c2(F.relu(c1(x, ei)), ei).argmax(1)
            va_acc = (p[va] == y[va]).float().mean().item()
            if va_acc >= best:
                best, bt = va_acc, (p[te] == y[te]).float().mean().item()
    return bt


def probe(h, y, tr, va, te, tune):
    h = h.cpu().numpy(); y = y.cpu().numpy(); tr, va, te = tr.cpu().numpy(), va.cpu().numpy(), te.cpu().numpy()
    sc = StandardScaler().fit(h[tr])
    Cs = [0.001, 0.01, 0.1, 1.0, 10.0] if tune else [1.0]
    best = (-1, None)
    for C in Cs:
        clf = LogisticRegression(max_iter=3000, C=C).fit(sc.transform(h[tr]), y[tr])
        v = accuracy_score(y[va], clf.predict(sc.transform(h[va])))
        if v > best[0]:
            best = (v, clf)
    return accuracy_score(y[te], best[1].predict(sc.transform(h[te])))


for name in sys.argv[1:] or ["cora", "citeseer", "pubmed"]:
    d = load(name); n = d["num_nodes"]
    x = d["emb"].to(dev); y = d["y"].to(dev)
    tr, va, te = d["train_mask"].to(dev), d["val_mask"].to(dev), d["test_mask"].to(dev)
    E_full = full_edges(name, n)
    ei_full = to_undirected(E_full, num_nodes=n).to(dev)
    ei_lp = to_undirected(d["train_pos"], num_nodes=n).to(dev)
    print(f"\n== {name}: full edges {E_full.shape[1]}, LP-graph edges {d['train_pos'].shape[1]} "
          f"({100*d['train_pos'].shape[1]/E_full.shape[1]:.0f}%)")
    A1 = np.mean([sup_gcn(x, ei_full, y, tr, va, te, s) for s in range(3)])
    A2 = np.mean([sup_gcn(x, ei_lp, y, tr, va, te, s) for s in range(3)])
    print(f"A1 supervised GCN, full graph : {100*A1:.2f}")
    print(f"A2 supervised GCN, LP graph   : {100*A2:.2f}")
    C1 = probe(x, y, tr, va, te, False); C2 = probe(x, y, tr, va, te, True)
    print(f"C  LR on raw MiniLM (C=1 / tuned): {100*C1:.2f} / {100*C2:.2f}")
    # B: retrain phase-0 joint (seed 0) and probe under different conditions
    res = {"B1": [], "B2": [], "B3": [], "B4": []}
    for s in range(3):
        set_seed(s)
        m, x0, mu, sd = phase0.train_return_model(name, "joint", s, epochs=1000)
        with torch.no_grad():
            z = torch.zeros(n, device=dev); za = torch.zeros(1, device=dev)
            _, h_lp = m.encode(x0, z, za, ei_lp)
            _, h_full = m.encode(x0, z, za, ei_full)
        res["B1"].append(probe(h_lp, y, tr, va, te, False))
        res["B2"].append(probe(h_full, y, tr, va, te, False))
        res["B3"].append(probe(h_full, y, tr, va, te, True))
        res["B4"].append(probe(torch.cat([h_full, x], 1), y, tr, va, te, True))
    for k, lab in (("B1", "as reported (LP graph, C=1)"), ("B2", "full graph, C=1"),
                   ("B3", "full graph, C tuned on val"), ("B4", "full graph, tuned, [h | text]")):
        print(f"{k} Phase-0 joint SSL probe, {lab}: {100*np.mean(res[k]):.2f} ± {100*np.std(res[k]):.2f}")
