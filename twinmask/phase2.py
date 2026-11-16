"""Phase 2 — analysis of a trained joint TwinMask model.

(1) (t_x, t_a) heatmap: edge AUC on masked pairs and text NLL on masked tokens, for every cell of a
    noise grid — one model, all conditionals.
(2) TSMI (Text–Structure Mutual Information) estimates from differences of NELBOs:
      TSMI_A   = [NELBO(A | X masked) - NELBO(A | X)] / #pairs        (nats per edge variable)
      TSMI_X   = [NELBO(X | A masked) - NELBO(X | A)] / #tokens       (nats per token; topology only)
      CTX_X    = [NELBO(X_anchor | alone) - NELBO(X_anchor | subgraph)] / #tokens  (neighbour text + topology)
    Each NELBO is a Monte-Carlo estimate of the MDLM continuous-time bound with a stratified t-grid.
    These are differences of *upper bounds* on -log p, i.e. estimates, not exact MI values.
"""
import argparse

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .common import save_json, set_seed
from .data import load
from .sampler import SubgraphSampler
from .tm import DEV, forward_losses, load_model, to_dev


def eval_batches(S, d, n_sub, bs, k, rng):
    """Held-out style subgraphs: half anchored on validation edges, half on random warm pairs."""
    vp = d["val_pos"].numpy()
    out = []
    for s in range(0, n_sub, bs):
        lists = []
        for i in range(min(bs, n_sub - s)):
            if i % 2 == 0:
                e = rng.integers(vp.shape[1]); anchors = [vp[0, e], vp[1, e]]
            else:
                anchors = list(rng.choice(S.warm, 2, replace=False))
            lists.append(S.gather(anchors, k))
        out.append(lists)
    return out


@torch.no_grad()
def heatmap(model, batches, grid, seed=0):
    res = {}
    for tx in grid:
        for ta in grid:
            gen = torch.Generator(device=DEV).manual_seed(seed)
            nll = cnt = 0.0
            ys, ss = [], []
            for b in batches:
                b = to_dev(b); B = b["ids"].shape[0]
                r = forward_losses(model, b, torch.full((B,), tx, device=DEV), torch.full((B,), ta, device=DEV),
                                   gen=gen, reduce=False)
                nll += r["tok_nll"].sum().item(); cnt += r["tok_cnt"].sum().item()
                m = r["emask"]
                ys.append(b["adj"][m].float().cpu()); ss.append(r["elog"][m].cpu())
            y, s_ = torch.cat(ys).numpy(), torch.cat(ss).numpy()
            auc = float(roc_auc_score(y, s_)) if 0 < y.sum() < len(y) else None
            res[f"{tx}|{ta}"] = dict(text_nll=nll / cnt if cnt else None, edge_auc=auc)
            print(f"  t_x={tx} t_a={ta}: nll={res[f'{tx}|{ta}']['text_nll']} auc={auc}", flush=True)
    return res


@torch.no_grad()
def nelbo(model, batches, which, fix_other, n_t=10, seed=0, anchor_only=False):
    """MC estimate of the MDLM NELBO of modality `which` ('x' or 'a') while the other modality is
    held at noise level fix_other (0 = observed, 1 = fully masked). Returns nats per item."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    tot, items = 0.0, 0.0
    ts = (np.arange(n_t) + 0.5) / n_t  # stratified grid on (0,1)
    for b in batches:
        b = to_dev(b); B, k = b["valid"].shape
        rows = None
        if anchor_only:
            rows = torch.zeros(B, k, dtype=torch.bool, device=DEV); rows[:, 0] = True
        for t in ts:
            tt = torch.full((B,), float(t), device=DEV); ff = torch.full((B,), float(fix_other), device=DEV)
            if which == "x":
                r = forward_losses(model, b, tt, ff, gen=gen, text_rows=rows, reduce=False)
                tot += (r["tok_nll"] / t).sum().item() / n_t
                if t == ts[0]:
                    items += r["n_tok"].sum().item()
            else:
                r = forward_losses(model, b, ff, tt, gen=gen, reduce=False)
                tot += (r["e_nll"] / t).sum().item() / n_t
                if t == ts[0]:
                    items += r["n_pair"].sum().item()
    return tot / items


def analyse(name, variant="joint", seed=0, n_sub=256, bs=32, k=16, grid=(0.0, 0.25, 0.5, 0.75, 1.0), do_heat=True):
    set_seed(seed)
    d = load(name)
    model = load_model(name, variant, seed)
    S = SubgraphSampler(d, k=k, seed=777)
    rng = np.random.default_rng(777)
    groups = eval_batches(S, d, n_sub, bs, k, rng)
    batches = [S.build(g) for g in groups]
    res = dict(dataset=name, variant=variant, seed=seed, homophily=d["homophily"])
    if do_heat:
        res["heatmap"] = heatmap(model, batches, grid, seed)
    nA_x, nA_m = nelbo(model, batches, "a", 0.0), nelbo(model, batches, "a", 1.0)
    nX_a, nX_m = nelbo(model, batches, "x", 0.0), nelbo(model, batches, "x", 1.0)
    # context: anchor text alone vs inside its subgraph (anchor = node 0)
    alone = [S.build([[l[0]] for l in g]) for g in groups]
    nX_ctx = nelbo(model, batches, "x", 0.0, anchor_only=True)
    nX_alone = nelbo(model, alone, "x", 0.0, anchor_only=True)
    res["nelbo"] = dict(A_given_X=nA_x, A_given_Xmasked=nA_m, X_given_A=nX_a, X_given_Amasked=nX_m,
                        Xanchor_ctx=nX_ctx, Xanchor_alone=nX_alone)
    res["tsmi"] = dict(TSMI_A=nA_m - nA_x, TSMI_X_topology=nX_m - nX_a, CTX_X=nX_alone - nX_ctx)
    print(f"[{name}] {res['tsmi']}", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cora"])
    ap.add_argument("--variant", default="joint")
    ap.add_argument("--no_heat", action="store_true")
    a = ap.parse_args()
    for nm in a.datasets:
        r = analyse(nm, a.variant, do_heat=not a.no_heat)
        save_json(r, "phase2", f"{nm}_{a.variant}_analysis.json")
