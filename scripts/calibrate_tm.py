"""TwinMask NC gap: probe tuning, full graph at inference, neighbourhood size, and text truncation."""
import sys

import numpy as np
import scipy.sparse as sp
import torch
from transformers import AutoTokenizer

sys.path.insert(0, ".")
from twinmask.data import load  # noqa: E402
from twinmask.sampler import SubgraphSampler  # noqa: E402
from twinmask.tm import load_model, node_embeddings  # noqa: E402
from scripts.calibrate import full_edges, probe  # noqa: E402

tok = AutoTokenizer.from_pretrained("distilroberta-base")
for name in sys.argv[1:] or ["cora", "pubmed"]:
    d = load(name); n = d["num_nodes"]
    lens = np.array([len(tok(t)["input_ids"]) for t in d["texts"][:2000]])
    print(f"\n== {name}: tokens per node median {int(np.median(lens))}, "
          f"truncated at 64: {100*(lens > 64).mean():.0f}% of nodes, mean fraction of text kept {np.mean(np.minimum(lens, 64) / lens):.2f}")
    y, tr, va, te = d["y"], d["train_mask"], d["val_mask"], d["test_mask"]
    m = load_model(name, "joint", 0)
    h_lp = torch.load(f"checkpoints/emb_{name}_joint_s0.pt").float()
    print(f"  stored emb (LP graph, k=12), C=1 / tuned: {100*probe(h_lp, y, tr, va, te, False):.2f} / {100*probe(h_lp, y, tr, va, te, True):.2f}")
    # full graph at inference
    dd = dict(d); dd["train_pos"] = full_edges(name, n)
    for k in (12, 24):
        S = SubgraphSampler(dd, k=k, seed=1234)
        h = node_embeddings(m, S, np.arange(n), k=k)
        print(f"  full graph, k={k}, tuned: {100*probe(h, y, tr, va, te, True):.2f}")
