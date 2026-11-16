"""Fair tuning for the DIVERGE x TwinMask bridge (all selection on VALIDATION accuracy only).

Stage A  diffusion hyper-parameters per (dataset, variant): grid hid x epochs x lambda_e x p_bound (16 configs), tuned
         on the seed-42 split with a neutral SAGE (DIVERGE default json), then frozen for all seeds.
Stage B  SAGE hyper-parameters per (dataset, seed, view) for OUR views (MiniLM raw + 4 diffusion views) with DIVERGE's
         search space: 64 random candidates, top-6 re-run with 3 GNN seeds, best mean val acc wins.
         DIVERGE's Gemma views keep DIVERGE's own tuned configs (512-candidate search).
Stage C  ensembles: uniform, and DIVERGE's learned WeightLearner (method 1, its own grid), fitted on train-node logits,
         epoch/config chosen on validation; for DIVERGE-5 and for MiniLM + 4 diffusion views.
"""
import itertools
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected

DIV = r"C:/Users/Nima/Desktop/Python Project/DIVERGE"
sys.path.insert(0, DIV); sys.path.insert(0, ".")
from common.dataloader import re_split_data  # noqa: E402
from common.gnn import GNNEncoder  # noqa: E402
from ensemble.weight_learner_ensemble import WeightLearner  # noqa: E402
from twinmask import phase0  # noqa: E402
from twinmask.common import save_json, set_seed  # noqa: E402
from twinmask.data import DIVERGE_DIR, load, undirected_unique  # noqa: E402

dev = "cuda"
INITS = ["pissa", "gaussian", "eva", "loftq", "orthogonal"]
VARIANTS = ["joint", "tied", "text_only", "edge_only"]
SPACE = dict(lr=[0.01, 0.005, 0.001, 0.0001], hidden_dim=[128, 256, 512, 1024], num_layers=[2, 3],
             dropout=[0.0, 0.1, 0.01, 0.5], weight_decay=[0.0, 0.01], batch_norm=[0, 1])
DIFF_GRID = list(itertools.product([256, 512], [500, 1500], [0.5, 2.0], [0.1, 0.3]))  # hid, epochs, lam_e, p_bound
OUT = "results/diverge_bridge_tuned"


def zs(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-6)


def sage(x, ei, y, masks, hp, seed):
    set_seed(seed)
    tr, va, te = masks
    m = GNNEncoder(x.shape[1], hp["hidden_dim"], int(y.max()) + 1, hp["num_layers"], "SAGE", hp["dropout"],
                   batch_norm=hp.get("batch_norm", 0)).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=hp["lr"], weight_decay=hp.get("weight_decay", 0.0))
    best, bad, out = -1, 0, None
    for _ in range(hp.get("epochs", 200)):
        m.train(); opt.zero_grad()
        F.cross_entropy(m(x, ei)[tr], y[tr]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            lo = m(x, ei); p = lo.argmax(1)
            v = (p[va] == y[va]).float().mean().item()
        if v > best:
            best, bad, out = v, 0, (lo.detach(), (p[te] == y[te]).float().mean().item())
        else:
            bad += 1
            if bad >= hp.get("early_stop", 20):
                break
    return dict(val=best, test=out[1], logits=out[0])


def div_hp(ds, seed):
    for f in (f"{ds}_gemma2_2B_SAGE_seed{seed}.json", f"{ds}.json"):
        p = os.path.join(DIV, "gnn_hyperparameters", f)
        if os.path.exists(p):
            h = json.load(open(p)); h.setdefault("epochs", 200); h.setdefault("early_stop", 20)
            return h


def tune_sage(x, ei, y, masks, seed, n_cand=64, top=6):
    rng = random.Random(seed)
    all_cfg = [dict(zip(SPACE, v)) for v in itertools.product(*SPACE.values())]
    cands = rng.sample(all_cfg, n_cand)
    scored = []
    for c in cands:
        c = dict(c, epochs=200, early_stop=20)
        scored.append((sage(x, ei, y, masks, c, seed)["val"], c))
    scored.sort(key=lambda t: -t[0])
    best = None
    for _, c in scored[:top]:
        v = np.mean([sage(x, ei, y, masks, c, seed + 1000 * k)["val"] for k in range(3)])
        if best is None or v > best[0]:
            best = (v, c)
    return best[1]


def diffusion_view(ds, d, base_emb, E, ei, variant, seed, cfg):
    hid, ep, lam, pb = cfg
    dd = dict(d); dd["emb"] = base_emb; dd["train_pos"] = E
    model, x0, _, _ = phase0.train_return_model(ds, variant, seed, epochs=ep, data=dd, hid=hid, lam_e=lam,
                                                p_bound=pb, select_lp=False)
    with torch.no_grad():
        n = x0.shape[0]
        hc, _ = model.encode(x0, torch.zeros(n, device=dev), torch.zeros(1, device=dev), ei)
    return hc.cpu()


def learned_ensemble(logits, y, masks):
    """DIVERGE method 1 (WeightLearner) with DIVERGE's grid; fit on train logits, select on val."""
    tr, va, te = masks
    C = int(y.max()) + 1
    best = (-1, None)
    for L, Hd, dr, lr in itertools.product([2, 3], [64, 128, 256], [0.3, 0.5], [0.001, 0.01, 0.0001]):
        torch.manual_seed(0)
        wl = WeightLearner(len(logits), C, Hd, L, dr).to(dev)
        opt = torch.optim.Adam(wl.parameters(), lr=lr)
        for _ in range(200):
            wl.train(); opt.zero_grad()
            out, _ = wl(*[l[tr] for l in logits])
            F.cross_entropy(out, y[tr]).backward(); opt.step()
            wl.eval()
            with torch.no_grad():
                out, _ = wl(*logits); p = out.argmax(1)
                v = (p[va] == y[va]).float().mean().item()
                if v > best[0]:
                    best = (v, (p[te] == y[te]).float().mean().item())
    return best[1]


def uniform(logits, y, te):
    p = torch.stack([l.softmax(1) for l in logits]).mean(0).argmax(1)
    return (p[te] == y[te]).float().mean().item()


def main(ds, seeds=(42, 43, 44, 45)):
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    d = load(ds); n = d["num_nodes"]
    g = torch.load(f"{DIVERGE_DIR}/{ds}.pt", weights_only=False, map_location="cpu")
    E = undirected_unique(g.edge_index, n); ei = to_undirected(E, num_nodes=n).to(dev)
    y = d["y"].to(dev)
    neutral = json.load(open(os.path.join(DIV, "gnn_hyperparameters", f"{ds}.json")))
    neutral.setdefault("epochs", 200); neutral.setdefault("early_stop", 20)
    # ---------------- Stage A: diffusion configs (seed-42 split, validation only)
    pA = f"{OUT}/{ds}_stageA.json"
    if os.path.exists(pA):
        best_cfg = {k: tuple(v) for k, v in json.load(open(pA))["best"].items()}
    else:
        masks = re_split_data(n, device=dev, seed=42)
        trials, best_cfg = [], {}
        for v in VARIANTS:
            sc = []
            for cfg in DIFF_GRID:
                h = diffusion_view(ds, d, d["emb"], E, ei, v, 42, cfg)
                r = sage(zs(h.to(dev)), ei, y, masks, neutral, 42)
                sc.append((r["val"], cfg)); trials.append(dict(variant=v, cfg=cfg, val=r["val"]))
            best_cfg[v] = max(sc, key=lambda t: t[0])[1]
            print(f"[{ds}] stage A {v}: best {best_cfg[v]} val={max(s[0] for s in sc):.4f} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(dict(best=best_cfg, trials=trials), open(pA, "w"), indent=1)
    # ---------------- Stage B + C per seed
    rows = []
    for s in seeds:
        masks = re_split_data(n, device=dev, seed=s); te = masks[2]
        res = {"dataset": ds, "seed": s}; logits = {}
        ours = {"minilm": d["emb"]}
        for v in VARIANTS:
            ours[f"minilm+diff_{v}"] = diffusion_view(ds, d, d["emb"], E, ei, v, s, best_cfg[v])
        for name, x in ours.items():
            xx = zs(x.to(dev))
            r0 = sage(xx, ei, y, masks, div_hp(ds, s), s)           # untuned (DIVERGE's Gemma config)
            hp = tune_sage(xx, ei, y, masks, s)
            r = sage(xx, ei, y, masks, hp, s)
            res[name + " [untuned]"] = r0["test"]; res[name + " [tuned]"] = r["test"]; logits[name] = r["logits"]
            res[name + " cfg"] = hp
        for ini in INITS:
            f = f"{DIV}/artifacts/cache/gemma2_2B_{ds}_seqcls_lora_init-{ini}_pool-mean_seed{s}_semi_supervised.pt"
            if os.path.exists(f):
                x = torch.load(f, weights_only=False, map_location="cpu")["embeddings"].float()
                r = sage(zs(x.to(dev)), ei, y, masks, div_hp(ds, s), s)
                res[f"gemma_{ini} [DIVERGE-tuned]"] = r["test"]; logits[f"gemma_{ini}"] = r["logits"]
        gk = [k for k in logits if k.startswith("gemma_")]
        ok = ["minilm"] + [f"minilm+diff_{v}" for v in VARIANTS]
        res["ENS uniform  DIVERGE-5"] = uniform([logits[k] for k in gk], y, te)
        res["ENS learned  DIVERGE-5"] = learned_ensemble([logits[k] for k in gk], y, masks)
        res["ENS uniform  MiniLM+4 diffusion (tuned)"] = uniform([logits[k] for k in ok], y, te)
        res["ENS learned  MiniLM+4 diffusion (tuned)"] = learned_ensemble([logits[k] for k in ok], y, masks)
        res["ENS learned  DIVERGE-5 + our 5 views"] = learned_ensemble([logits[k] for k in gk + ok], y, masks)
        print({k: round(100 * v, 2) for k, v in res.items() if isinstance(v, float)}, f"({time.time()-t0:.0f}s)", flush=True)
        rows.append(res)
        save_json(rows, "diverge_bridge_tuned", f"{ds}.json")
    keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
    print(f"\n=== {ds} (test acc %, mean ± std over seeds {list(seeds)})")
    for k in keys:
        vals = [r[k] for r in rows]
        print(f"  {k:48s} {100*np.mean(vals):6.2f} ± {100*np.std(vals):.2f}")


if __name__ == "__main__":
    for ds in sys.argv[1:] or ["cora", "citeseer"]:
        main(ds)
