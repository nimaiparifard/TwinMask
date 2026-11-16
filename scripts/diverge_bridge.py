"""Bridge experiment: DIVERGE (label-fine-tuned LLM embeddings + GNN + ensemble) x TwinMask phase-0 joint diffusion.

Protocol = DIVERGE's: random 60/20/20 split per seed (42..45, DIVERGE's own re_split_data), full graph, SAGE from
DIVERGE's GNNEncoder with DIVERGE's tuned hyperparameters, checkpoint chosen on validation accuracy.
Text views: MiniLM (frozen, ours) and Gemma-2-2B LoRA-fine-tuned caches from DIVERGE (5 inits, fine-tuned on the
*same* split -> no test label leakage). Diffusion views: our label-free phase-0 model trained on top of a text view
with different noise schedules (joint / tied / text_only / edge_only), node states h taken on the full graph.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected

DIV = r"C:/Users/Nima/Desktop/Python Project/DIVERGE"
sys.path.insert(0, DIV); sys.path.insert(0, ".")
from common.dataloader import re_split_data  # noqa: E402
from common.gnn import GNNEncoder  # noqa: E402
from twinmask import phase0  # noqa: E402
from twinmask.common import save_json, set_seed  # noqa: E402
from twinmask.data import DIVERGE_DIR, load, undirected_unique  # noqa: E402

dev = "cuda"
INITS = ["pissa", "gaussian", "eva", "loftq", "orthogonal"]
VARIANTS = ["joint", "tied", "text_only", "edge_only"]


def hparams(ds, seed):
    for f in (f"{ds}_gemma2_2B_SAGE_seed{seed}.json", f"{ds}.json"):
        p = os.path.join(DIV, "gnn_hyperparameters", f)
        if os.path.exists(p):
            return json.load(open(p))


def sage(x, ei, y, masks, hp, seed):
    """DIVERGE GNN training: Adam, early stopping on val acc, returns (test acc, test softmax, val acc)."""
    set_seed(seed)
    tr, va, te = masks
    m = GNNEncoder(x.shape[1], hp["hidden_dim"], int(y.max()) + 1, hp["num_layers"], "SAGE", hp["dropout"],
                   batch_norm=hp.get("batch_norm", 0)).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=hp["lr"], weight_decay=hp.get("weight_decay", 0.0))
    best, bad, out = -1, 0, None
    for _ in range(hp["epochs"]):
        m.train(); opt.zero_grad()
        F.cross_entropy(m(x, ei)[tr], y[tr]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            lo = m(x, ei); p = lo.argmax(1)
            v = (p[va] == y[va]).float().mean().item()
        if v > best:
            best, bad, out = v, 0, (lo.softmax(1), (p[te] == y[te]).float().mean().item())
        else:
            bad += 1
            if bad >= hp["early_stop"]:
                break
    return out[1], out[0], best


def zs(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-6)


def main(ds, seeds=(42, 43, 44, 45)):
    d = load(ds); n = d["num_nodes"]
    g = torch.load(f"{DIVERGE_DIR}/{ds}.pt", weights_only=False, map_location="cpu")
    E = undirected_unique(g.edge_index, n)
    ei = to_undirected(E, num_nodes=n).to(dev)
    y = d["y"].to(dev)
    rows = []
    for s in seeds:
        masks = re_split_data(n, device=dev, seed=s)
        hp = hparams(ds, s)
        views = {"minilm": d["emb"]}
        for ini in INITS:
            f = f"{DIV}/artifacts/cache/gemma2_2B_{ds}_seqcls_lora_init-{ini}_pool-mean_seed{s}_semi_supervised.pt"
            if os.path.exists(f):
                views[f"gemma_{ini}"] = torch.load(f, weights_only=False, map_location="cpu")["embeddings"].float()
        # label-free diffusion views on top of MiniLM and of one Gemma view (PiSSA)
        for base in ("minilm", "gemma_pissa"):
            if base not in views:
                continue
            dd = dict(d); dd["emb"] = views[base]; dd["train_pos"] = E  # transductive NC: all edges known
            for v in VARIANTS:
                model, x0, _, _ = phase0.train_return_model(ds, v, s, epochs=1000, data=dd)
                with torch.no_grad():
                    z = torch.zeros(n, device=dev)
                    hc, h = model.encode(x0, z, torch.zeros(1, device=dev), ei)
                views[f"{base}+diff_{v}"] = hc.cpu()      # [input proj | propagated] node state
        probs, res = {}, {"dataset": ds, "seed": s}
        for name, x in views.items():
            acc, pr, va = sage(zs(x.to(dev)), ei, y, masks, hp, s)
            probs[name] = pr; res[name] = acc
        te = masks[2]
        ens = lambda keys: (torch.stack([probs[k] for k in keys]).mean(0).argmax(1)[te] == y[te]).float().mean().item()
        gk = [k for k in probs if k.startswith("gemma_") and "+" not in k]
        res["ENS_diverge5 (5 LoRA inits)"] = ens(gk) if len(gk) > 1 else None
        for base in ("minilm", "gemma_pissa"):
            dk = [f"{base}+diff_{v}" for v in VARIANTS]
            if all(k in probs for k in dk):
                res[f"ENS_{base}+4 diffusion views"] = ens(dk)
                res[f"ENS_{base} raw + 4 diffusion views"] = ens([base] + dk)
        if len(gk) > 1 and all(f"gemma_pissa+diff_{v}" in probs for v in VARIANTS):
            res["ENS_diverge5 + gemma diffusion views"] = ens(gk + [f"gemma_pissa+diff_{v}" for v in VARIANTS])
        print({k: (round(100 * v, 2) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)
        rows.append(res)
    save_json(rows, "diverge_bridge", f"{ds}.json")
    keys = [k for k in rows[0] if k not in ("dataset", "seed")]
    print(f"\n=== {ds}: mean ± std over seeds {list(seeds)} (test acc %)")
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals:
            print(f"  {k:45s} {100*np.mean(vals):6.2f} ± {100*np.std(vals):.2f}")


if __name__ == "__main__":
    for ds in sys.argv[1:] or ["cora", "citeseer"]:
        main(ds)
