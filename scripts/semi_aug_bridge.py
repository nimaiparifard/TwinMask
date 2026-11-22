"""DIVERGE semi-supervised augmentation (tags_data_augmentation, public low-label split) x TwinMask phase-0 diffusion views.

Arms (DIVERGE SemiTask):  base  = plain semi-supervised adapters, original graph
                          aug   = adapters fine-tuned on base+augmented nodes, augmented nodes added to the graph (kNN edges)
                                  as pseudo-labelled TRAINING nodes (ref_plain)
Views per arm: 5 DIVERGE members (SAGE on each LoRA-init embedding, DIVERGE's semi hp)  +
               label-free diffusion views trained on the same LLM embeddings over the SAME graph (incl. augmented nodes):
               joint/tied/text_only/edge_only on PiSSA, joint on the other 4 inits.
Selection: validation only (public val).  Ensembles: uniform softmax mean.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.utils import to_undirected

DIV = os.environ.get("TWINMASK_DIVERGE_DIR", r"C:/Users/Nima/Desktop/Python Project/DIVERGE")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.append(DIV)
from diverge2 import core  # noqa: E402
from tags_data_augmentation.semi_study.semi_task import SemiTask  # noqa: E402
from twinmask import phase0  # noqa: E402
from twinmask.common import save_json  # noqa: E402

dev = "cuda"
VARIANTS = ["joint", "tied", "text_only", "edge_only"]


def zs(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-6)


def member(t, x, ei, seed):
    hp = t.hp("SAGE")
    core.set_all_seeds(seed)
    m = core.build_encoder("SAGE", x.shape[1], t.C, hp).to(dev)
    r = core.train_loop(m, lambda mm: mm(x, ei), t.y, t.masks, hp, seed)
    return r


def diffusion_view(t, emb, ei_u, variant, seed, cfg=(256, 1000, 1.0, 0.1), *, device=dev):
    hid, ep, lam, pb = cfg
    n = emb.shape[0]
    d = dict(num_nodes=n, emb=emb.cpu(), train_pos=t.und.cpu(), cold_nodes=torch.zeros(0, dtype=torch.long))
    model, x0, _, _ = phase0.train_return_model("semi", variant, seed, epochs=ep, data=d, hid=hid, lam_e=lam,
                                                p_bound=pb, select_lp=False, dev=device)
    with torch.no_grad():
        hc, _ = model.encode(x0, torch.zeros(n, device=device), torch.zeros(1, device=device), ei_u)
    return hc


def acc(p, t):
    return core.acc(p, t.y, t.masks[2])


def ens(t, probs):
    p = torch.stack(probs).mean(0).argmax(1)
    return core.acc(p, t.y, t.masks[2])


SEL = os.path.join(DIV, "tags_data_augmentation", "semi_study", "results")


def make_task(ds, arm, s, selection_dir=None):
    """DIVERGE's own validation-selected configuration (ref_selected): edge rule, pseudo-label handling, lambda, tuned hp."""
    selection_dir = SEL if selection_dir is None else selection_dir
    with open(f"{selection_dir}/edge_selection.json", encoding="utf8") as f:
        edge = json.load(f)[ds]
    with open(f"{selection_dir}/mode_selection.json", encoding="utf8") as f:
        mode = json.load(f)[ds]
    hp = f"{selection_dir}/hp_{ds}_{arm}.json"
    kw = dict(hp_file=hp) if os.path.exists(hp) else {}
    if arm == "base":
        return SemiTask(ds, arm="base", repeat=s, **kw)
    ek = {k: edge[k] for k in ("edge_rule", "cos_thr", "ep_thr", "top_k")}
    mk = {k: mode[k] for k in ("mode", "conf_min", "gamma", "aug_lambda")}
    return SemiTask(ds, arm="augFull", variant="curated_raw", repeat=s, **kw, **ek, **mk)


def main(ds, seeds=(0, 1, 2)):
    rows = []
    for arm in ("base", "augFull"):
        for s in seeds:
            t = make_task(ds, arm, s)
            ei = to_undirected(t.und, num_nodes=t.n).to(dev)
            res = dict(dataset=ds, arm=arm, seed=s, n_nodes=t.n, n_train=int(t.masks[0].sum()))
            P, V = {}, {}
            for i in t.inits:
                r = member(t, t.embs[i].to(dev), ei, s)
                P[f"div_{i}"] = r["logits"].softmax(1); V[f"div_{i}"] = r["val"]; res[f"div_{i}"] = r["test"]
            for v in VARIANTS:
                h = diffusion_view(t, t.embs["pissa"], ei, v, s)
                r = member(t, zs(h), ei, s)
                P[f"diff_pissa_{v}"] = r["logits"].softmax(1); V[f"diff_pissa_{v}"] = r["val"]; res[f"diff_pissa_{v}"] = r["test"]
            for i in t.inits:
                if i == "pissa":
                    continue
                h = diffusion_view(t, t.embs[i], ei, "joint", s)
                r = member(t, zs(h), ei, s)
                P[f"diff_{i}_joint"] = r["logits"].softmax(1); V[f"diff_{i}_joint"] = r["val"]; res[f"diff_{i}_joint"] = r["test"]
            P = {k: v.cpu() for k, v in P.items()}
            dk = [k for k in P if k.startswith("div_")]
            fk = [k for k in P if k.startswith("diff_")]
            te = t.masks[2].cpu(); y = t.y.cpu()
            e = lambda keys: (torch.stack([P[k] for k in keys]).mean(0).argmax(1)[te] == y[te]).float().mean().item()
            res["ENS DIVERGE-1 (5 inits)"] = e(dk)
            res["ENS diffusion (8 views)"] = e(fk)
            res["ENS DIVERGE-1 + diffusion"] = e(dk + fk)
            # validation-screened (DIVERGE rule: within 5 points of best val)
            vb = max(V.values())
            sk = [k for k in P if V[k] >= vb - 0.05]
            res["ENS screened all views"] = e(sk)
            print({k: (round(100 * v, 2) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)
            rows.append(res)
            save_json(rows, "semi_aug_bridge", f"{ds}.json")
    keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
    for arm in ("base", "augFull"):
        rr = [r for r in rows if r["arm"] == arm]
        print(f"\n=== {ds} / {arm}  (test acc %, mean ± std over {len(rr)} repeats)")
        for k in keys:
            v = [r[k] for r in rr]
            print(f"  {k:32s} {100*np.mean(v):6.2f} ± {100*np.std(v):.2f}")


if __name__ == "__main__":
    for ds in sys.argv[1:] or ["cora", "citeseer"]:
        main(ds)
