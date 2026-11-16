"""DIVERGE augmentation METHODS (DIVA family, own_attn, own_viewdis, literature ideas) x TwinMask diffusion views.
Each method is built with DIVERGE's own code (diva_ideas.build -> SemiTask with its filter/weights/soft labels/edges and patched
train loop). For every method: 5 DIVERGE-1 members + 8 label-free diffusion views on the same graph; uniform ensembles.
Resumable; a failing method is logged and skipped."""
import json, os, sys, time, traceback
import numpy as np, torch
from torch_geometric.utils import to_undirected
DIV = r"C:/Users/Nima/Desktop/Python Project/DIVERGE"
sys.path.insert(0, DIV); sys.path.insert(0, ".")
from tags_data_augmentation.semi_study import diva_ideas as D  # noqa: E402
from scripts.semi_aug_bridge import member, diffusion_view, zs, VARIANTS  # noqa: E402
from twinmask.common import save_json  # noqa: E402

dev = "cuda"
METHODS = ["ref_plain", "ref_selected", "com_vote", "own_attn", "own_viewdis", "lit_conf", "diva2_ens", "diva_mi", "diva_full"]
OUT = "results/semi_aug_bridge/methods_{}.json"


def one(ds, name, s):
    t = D.build(ds, name, repeat=s)
    ei = to_undirected(t.und, num_nodes=t.n).to(dev)
    P, V, res = {}, {}, dict(dataset=ds, method=name, seed=s, n_aug_kept=int(t.n_aug_kept))
    def add(k, r):
        P[k] = r["logits"].softmax(1).cpu(); V[k] = r["val"]; res[k] = r["test"] / 100
    for i in t.inits:
        add(f"div_{i}", member(t, t.embs[i].to(dev), ei, s))
    for v in VARIANTS:
        add(f"diff_pissa_{v}", member(t, zs(diffusion_view(t, t.embs["pissa"], ei, v, s)), ei, s))
    for i in t.inits:
        if i != "pissa":
            add(f"diff_{i}_joint", member(t, zs(diffusion_view(t, t.embs[i], ei, "joint", s)), ei, s))
    te, y = t.masks[2].cpu(), t.y.cpu()
    e = lambda ks: (torch.stack([P[k] for k in ks]).mean(0).argmax(1)[te] == y[te]).float().mean().item()
    dk = [k for k in P if k.startswith("div_")]; fk = [k for k in P if k.startswith("diff_")]
    res["ENS DIVERGE-1"] = e(dk); res["ENS diffusion"] = e(fk); res["ENS both"] = e(dk + fk)
    return res


def main(ds, seeds=(0, 1, 2)):
    path = OUT.format(ds); rows = json.load(open(path)) if os.path.exists(path) else []
    for name in METHODS:
        for s in seeds:
            if any(r["method"] == name and r["seed"] == s for r in rows):
                continue
            t0 = time.time()
            try:
                r = one(ds, name, s)
            except Exception:
                print(f"[FAILED] {ds} {name} {s}: {traceback.format_exc()[-400:]}", flush=True); break
            rows.append(r); os.makedirs("results/semi_aug_bridge", exist_ok=True); json.dump(rows, open(path, "w"), indent=1)
            print(ds, name, s, {k: round(100 * v, 2) for k, v in r.items() if k.startswith("ENS")}, f"{time.time()-t0:.0f}s", flush=True)
    print(f"\n=== {ds}: mean ± std test acc % (3 repeats)")
    for name in METHODS:
        rr = [r for r in rows if r["method"] == name]
        if rr:
            print(f"{name:14s} " + "  ".join(f"{k[4:]:10s} {100*np.mean([r[k] for r in rr]):.2f}±{100*np.std([r[k] for r in rr]):.2f}" for k in ("ENS DIVERGE-1", "ENS diffusion", "ENS both")))


if __name__ == "__main__":
    for ds in sys.argv[1:] or ["cora", "citeseer"]:
        main(ds)
