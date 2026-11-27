"""TwinMask-Path: which path through the (t_x, t_a) noise square should the reverse process take?

Add-on experiment on top of a trained, FROZEN TwinMask checkpoint (`tm.py` is unchanged; the original
one-shot evaluation is the N=1 row of the step sweep).

Stages (each one decides whether the next is worth doing):
  1. fixed paths       : diagonal, text-first, edge-first, power-law leads, random order, joint confidence
  2. tuned fixed path  : the best fixed path selected on a disjoint set of cold nodes (= non-contextual bandit)
  3. learned policy    : a small MLP chooses, at every step, how the step's budget is split between text and
                         edges, from the current denoising state. Trained with group-relative policy gradients
                         (GRPO-style: G rollouts of the same episode, advantage = group z-score; no KL / clipping
                         because each batch is used for exactly one update). Cross-modal credit: the text and
                         edge rewards are z-scored separately inside the group and then mixed with weight w.
Reports are on held-out cold nodes, with paired differences to the diagonal (same episodes).

  python -m twinmask.path_rl --dataset cora --steps 8 --rho 0.5
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from .common import lp_metrics, save_json, set_seed
from .data import load
from .denoise import PATHS, ColdEpisodes, _counts_from_ratio, make_controller, repeat, run
from .model import PRESENT
from .tm import DEV, load_model

RATIOS = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])  # share of the step budget given to text
N_FEAT = 11


# ------------------------------------------------------------------ learned controller
def features(ro, obs, s, N):
    rt, re = ro.remaining()
    nt, ne = ro.n_tok0.float().clamp(min=1), ro.n_edge0.float().clamp(min=1)
    tm, em = ro.tok_m.float(), ro.edge_m.float()
    mc_t = (obs["tok_conf"] * tm).sum(1) / tm.sum(1).clamp(min=1)
    mc_e = (obs["edge_conf"] * em).sum(1) / em.sum(1).clamp(min=1)
    md_e = (obs["edge_p"] * em).sum(1) / em.sum(1).clamp(min=1)
    vis = (ro.state == PRESENT).float().sum((1, 2)) / 2 / ro.n_pair
    one = torch.ones_like(nt)
    return torch.stack([rt / nt, re / ne, one * s / N, one * (N - s) / N, mc_t, mc_e, md_e, vis,
                        ro.n_tok0 / ro.n_maskable, torch.log1p(nt) / 4, torch.log1p(ne) / 3], 1)


class RatioPolicy(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_FEAT, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, len(RATIOS)))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)  # starts uniform
        self.sample = True
        self.reset()

    def reset(self):
        self.actions, self.ents = [], []

    def step(self, ro, obs, s, N):
        f = features(ro, obs, s, N).detach()
        dist = torch.distributions.Categorical(logits=self.net(f))
        a = dist.sample() if self.sample else dist.probs.argmax(-1)
        self.actions.append(a.detach()); self.ents.append(dist.entropy())
        rt, re = ro.remaining()
        n_t, n_e = _counts_from_ratio(RATIOS.to(DEV)[a], rt, re, N - s)
        ro.commit_counts(obs, n_t, n_e)
        return dist.log_prob(a)


# ------------------------------------------------------------------ evaluation
def evaluate(model, E, nodes, controller, N, rho, mode, reps, bs=32):
    """Per-episode metrics over `reps` resamplings of the episodes (same seeds for every controller)."""
    per = {kk: [] for kk in ("text_nll", "text_acc", "edge_nll", "edge_acc")}
    p_all, y_all, acts = [], [], []
    for r in range(reps):
        for i, s in enumerate(range(0, len(nodes), bs)):
            ep = E.make(nodes[s:s + bs], rho, seed=10_000 * (r + 1) + i)
            if hasattr(controller, "reset"):
                controller.reset()
            with torch.no_grad():
                ro, _ = run(model, ep, controller, N, mode)
            m = ro.episode_metrics()
            for kk in per:
                per[kk].append(m[kk].cpu())
            em = m["edge_mask"]
            p_all.append(ro.rec_edge_p[em].cpu()); y_all.append(ro.true_e[em].cpu())
            if hasattr(controller, "actions"):
                acts.append(RATIOS[torch.stack(controller.actions, 1).cpu()])
    per = {kk: torch.cat(v).numpy() for kk, v in per.items()}
    p, y = torch.cat(p_all).numpy(), torch.cat(y_all).numpy()
    out = {kk: float(v.mean()) for kk, v in per.items()}
    out.update({f"edge_{kk}": v for kk, v in lp_metrics(p[y == 1], p[y == 0]).items()})
    pred = p > 0.5
    tp = float((pred & (y == 1)).sum()); out["edge_f1"] = 2 * tp / max(1.0, float(pred.sum() + (y == 1).sum()))
    out["forwards"] = N
    if acts:
        out["text_share_per_step"] = torch.cat(acts).mean(0).tolist()
    return out, per


def paired(per, base):
    """mean difference to the baseline on the same episodes, with a 95% normal interval."""
    res = {}
    for kk in ("text_nll", "edge_nll", "text_acc", "edge_acc"):
        d = per[kk] - base[kk]
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        res[kk] = dict(mean=float(d.mean()), ci95=float(1.96 * se))
    return res


def rel_score(m, base):
    """lower is better: text and edge NLL relative to the diagonal path (scale-free sum)."""
    return m["text_nll"] / max(base["text_nll"], 1e-9) + m["edge_nll"] / max(base["edge_nll"], 1e-9)


# ------------------------------------------------------------------ GRPO training
def train_policy(model, E, nodes, N, rho, mode, iters=300, n_ep=8, G=8, lr=3e-3, beta=0.01, w_text=0.5,
                 adv="zsum", seed=0, log_every=25):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pol = RatioPolicy().to(DEV)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    hist = []
    t0 = time.time()
    for it in range(1, iters + 1):
        ep = repeat(E.make(rng.choice(nodes, n_ep, replace=len(nodes) < n_ep), rho, seed=2_000_000 + it), G)
        pol.sample = True; pol.reset()
        ro, logp = run(model, ep, pol, N, mode)
        m = ro.episode_metrics()
        rt, re = -m["text_nll"].view(n_ep, G), -m["edge_nll"].view(n_ep, G)

        def z(x):
            return (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-6)
        if adv == "zsum":  # per-modality credit: each modality's reward is normalised inside the group
            a = w_text * z(rt) + (1 - w_text) * z(re)
        else:  # plain scalar reward with a group-mean baseline
            r = w_text * rt + (1 - w_text) * re
            a = r - r.mean(1, keepdim=True)
        a = a.reshape(-1).detach()
        ent = torch.stack(pol.ents, 1).mean()
        loss = -(a * logp.sum(1)).mean() - beta * ent
        opt.zero_grad(); loss.backward(); opt.step()
        if it % log_every == 0 or it == iters:
            share = RATIOS[torch.stack(pol.actions, 1).cpu()].mean(0)
            h = dict(it=it, text_nll=float(-rt.mean()), edge_nll=float(-re.mean()), entropy=float(ent),
                     text_share_per_step=[round(float(v), 2) for v in share], sec=round(time.time() - t0))
            hist.append(h)
            print(f"    [policy] it {it} text_nll {h['text_nll']:.3f} edge_nll {h['edge_nll']:.4f} "
                  f"H {h['entropy']:.2f} share {h['text_share_per_step']} ({h['sec']}s)", flush=True)
    pol.sample = False
    return pol, hist


# ------------------------------------------------------------------ main experiment
def experiment(ds, steps=8, rho=0.5, mode="generate", k=12, reps=3, iters=300, seed=0, ckpt_seed=0,
               max_nodes=400, sweep=(1, 2, 4, 8, 16), adv="zsum", w_text=0.5, no_policy=False):
    set_seed(seed)
    d = load(ds)
    model = load_model(ds, "joint", ckpt_seed)
    for p in model.parameters():
        p.requires_grad_(False)
    E = ColdEpisodes(d, k=k)
    fit, test = E.split(seed)
    fit, test = fit[:max_nodes], test[:max_nodes]
    print(f"[{ds}] hom={d['homophily']:.3f} cold nodes fit={len(fit)} test={len(test)} N={steps} rho={rho} mode={mode}",
          flush=True)
    res = dict(dataset=ds, homophily=d["homophily"], steps=steps, rho=rho, mode=mode, k=k, reps=reps, seed=seed,
               n_fit=len(fit), n_test=len(test), paths={}, fit_paths={}, sweep={})
    names = ["diagonal", "diagonal_random", "text_first", "edge_first", "text_lead2", "text_lead1",
             "edge_lead1", "edge_lead2", "joint_conf"]
    per_test = {}
    for nm in names:
        res["fit_paths"][nm], _ = evaluate(model, E, fit, make_controller(nm), steps, rho, mode, 1)
        res["paths"][nm], per_test[nm] = evaluate(model, E, test, make_controller(nm), steps, rho, mode, reps)
        m = res["paths"][nm]
        print(f"  {nm:16s} text nll {m['text_nll']:.3f} acc {m['text_acc']:.3f} | edge nll {m['edge_nll']:.4f} "
              f"auc {m['edge_auc']:.3f} f1 {m['edge_f1']:.3f}", flush=True)
    base = res["fit_paths"]["diagonal"]
    fixed = [nm for nm in names if nm != "joint_conf"]  # 'tuned' = non-adaptive path selection
    tuned = min(fixed, key=lambda nm: rel_score(res["fit_paths"][nm], base))
    res["tuned_path"] = tuned
    print(f"  tuned fixed path (selected on fit nodes): {tuned}", flush=True)
    for nm in names:
        res["paths"][nm]["vs_diagonal"] = paired(per_test[nm], per_test["diagonal"])

    if not no_policy:
        pol, hist = train_policy(model, E, fit, steps, rho, mode, iters=iters, seed=seed, adv=adv, w_text=w_text)
        res["policy_hist"] = hist
        m, per = evaluate(model, E, test, pol, steps, rho, mode, reps)
        m["vs_diagonal"] = paired(per, per_test["diagonal"])
        m["vs_tuned"] = paired(per, per_test[tuned])
        res["paths"]["policy_grpo"] = m
        print(f"  {'policy_grpo':16s} text nll {m['text_nll']:.3f} acc {m['text_acc']:.3f} | edge nll "
              f"{m['edge_nll']:.4f} auc {m['edge_auc']:.3f} f1 {m['edge_f1']:.3f} share {np.round(m['text_share_per_step'], 2).tolist()}",
              flush=True)

    for N in sweep:  # compute-aware view: quality vs number of denoiser calls
        for nm in ("diagonal", "joint_conf"):
            m, _ = evaluate(model, E, test, make_controller(nm), N, rho, mode, reps)
            res["sweep"][f"{nm}@{N}"] = m
            print(f"  sweep {nm}@{N:<3d} text nll {m['text_nll']:.3f} edge nll {m['edge_nll']:.4f} auc {m['edge_auc']:.3f}",
                  flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cora")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--rho", type=float, default=0.5, help="fraction of the cold node's tokens that is masked")
    ap.add_argument("--mode", default="generate", choices=["generate", "teacher"])
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt_seed", type=int, default=0)
    ap.add_argument("--max_nodes", type=int, default=400)
    ap.add_argument("--adv", default="zsum", choices=["zsum", "sum"])
    ap.add_argument("--w_text", type=float, default=0.5)
    ap.add_argument("--no_policy", action="store_true")
    ap.add_argument("--sweep", type=int, nargs="*", default=[1, 2, 4, 8, 16])
    ap.add_argument("--output-tag", default="path_rl", help="result subdirectory under results")
    a = ap.parse_args()
    r = experiment(a.dataset, a.steps, a.rho, a.mode, a.k, a.reps, a.iters, a.seed, a.ckpt_seed, a.max_nodes,
                   tuple(a.sweep), a.adv, a.w_text, a.no_policy)
    print(save_json(r, a.output_tag, f"{a.dataset}_rho{a.rho}_N{a.steps}_{a.mode}_s{a.seed}.json"))
