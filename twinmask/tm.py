"""Phase 1/3 — train and evaluate TwinMask-S (token-level joint masked diffusion)."""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .common import ROOT, lp_metrics, probe_nc, sample_t, save_json, set_seed
from .data import load
from .heart import load_heart, rank_metrics
from .model import MASKED, TwinMask
from .sampler import SubgraphSampler

CKPT = os.path.join(ROOT, "checkpoints")
DEV = "cuda"
MASK_ID = AutoTokenizer.from_pretrained("distilroberta-base").mask_token_id


def to_dev(b):
    return {k: v.to(DEV, non_blocking=True) for k, v in b.items()}


def corrupt(b, t_x, t_a, gen=None, text_rows=None):
    """Absorbing-state corruption. text_rows (B,k) bool: restrict text masking to these nodes."""
    ids, attn, valid, state = b["ids"], b["attn"], b["valid"], b["state"]
    B, k, L = ids.shape
    pos = torch.arange(L, device=ids.device)[None, None, :]
    maskable = attn & (pos > 0)
    if text_rows is not None:
        maskable &= text_rows[:, :, None]
    r = torch.rand(ids.shape, device=ids.device, generator=gen)
    tmask = maskable & (r < t_x[:, None, None])
    ids_in = torch.where(tmask, torch.full_like(ids, MASK_ID), ids)
    iu = torch.triu(torch.ones(k, k, dtype=torch.bool, device=ids.device), 1)[None]
    pairs = iu & valid[:, :, None] & valid[:, None, :]
    re = torch.rand(B, k, k, device=ids.device, generator=gen)
    emask = pairs & (re < t_a[:, None, None])
    emask = emask | (b["forced"] & pairs)
    emask = emask | emask.transpose(1, 2)
    state_in = torch.where(emask, torch.full_like(state, MASKED), state)
    return ids_in, tmask, state_in, emask & iu, pairs, maskable


def forward_losses(model, b, t_x, t_a, gen=None, text_rows=None, reduce=True):
    ids_in, tmask, state_in, emask, pairs, maskable = corrupt(b, t_x, t_a, gen, text_rows)
    B, k, L = ids_in.shape
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h, x, elog = model(ids_in, b["attn"], b["valid"], state_in, t_x, t_a)
    sel = tmask.view(B * k, L)
    tok_nll = torch.zeros(B, device=DEV); tok_cnt = torch.zeros(B, device=DEV); tok_hit = torch.zeros(B, device=DEV)
    if sel.any():
        xs = x[sel]
        tgt = b["ids"].view(B * k, L)[sel].long()
        bidx = (torch.arange(B * k, device=DEV) // k)[:, None].expand(B * k, L)[sel]
        for c in range(0, xs.shape[0], 2048):  # chunked LM head keeps the 50k-vocab logits small
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model.token_logits(xs[c:c + 2048])
            nll = F.cross_entropy(lg.float(), tgt[c:c + 2048], reduction="none")
            tok_nll = tok_nll.index_add(0, bidx[c:c + 2048], nll)
            tok_cnt.index_add_(0, bidx[c:c + 2048], torch.ones_like(nll))
            tok_hit.index_add_(0, bidx[c:c + 2048], (lg.argmax(-1) == tgt[c:c + 2048]).float())
    bce = F.binary_cross_entropy_with_logits(elog.float(), b["adj"].float(), reduction="none")
    e_nll = (bce * emask).sum((1, 2))
    e_cnt = emask.sum((1, 2)).float()
    if not reduce:
        return dict(tok_nll=tok_nll, tok_cnt=tok_cnt, tok_hit=tok_hit, e_nll=e_nll, e_cnt=e_cnt,
                    elog=elog.float(), emask=emask, h=h.float(),
                    n_tok=maskable.sum((1, 2)).float(), n_pair=pairs.sum((1, 2)).float())
    # Low-variance training objective: mean CE over masked items (pooled over the batch).
    # E[#masked] = t*N, so this matches the 1/t-weighted ELBO up to per-sample reweighting, without the
    # huge-weight outliers that small t produces on short sequences / few edge variables.
    # (The exact 1/t-weighted NELBO is used for evaluation in phase2.nelbo.)
    Lx = tok_nll.sum() / tok_cnt.sum().clamp(min=1)
    La = e_nll.sum() / e_cnt.sum().clamp(min=1)
    return Lx, La


def sample_times(variant, B):
    if variant in ("edge_only",):
        return torch.zeros(B, device=DEV), sample_t(B, DEV)
    if variant == "text_only":
        return sample_t(B, DEV), torch.zeros(B, device=DEV)
    t_x = sample_t(B, DEV)
    if variant == "tied":
        return t_x, t_x.clone()
    return t_x, sample_t(B, DEV)


def topo_of(variant):
    return {"static": "static", "notopo": "none"}.get(variant, "dynamic")


def train(name, variant="joint", steps=3000, bs=16, k=16, lam=1.0, seed=0, log_every=100, lm_lr=2e-5):
    set_seed(seed)
    d = load(name)
    S = SubgraphSampler(d, k=k, seed=seed)
    model = TwinMask(topo=topo_of(variant)).to(DEV)
    new = [p for n_, p in model.named_parameters() if not n_.startswith("lm.")]
    old = [p for n_, p in model.named_parameters() if n_.startswith("lm.")]
    opt = torch.optim.AdamW([{"params": old, "lr": lm_lr}, {"params": new, "lr": 3e-4}], weight_decay=0.01)
    warm = 200
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * max(0.05, 1 - s / steps))
    t0 = time.time(); hist = []
    # model selection on held-out validation edges (never visible in training) -> guards against the
    # memorisation of small graphs by the 82M-parameter backbone
    vrng = np.random.default_rng(99)
    vpos = subsample(d["val_pos"], 300, vrng); vneg = subsample(d["val_neg"], 300, vrng)
    Sv = SubgraphSampler(d, k=k, seed=99)
    best, best_state, best_step = -1.0, None, 0
    model.train()
    for step in range(1, steps + 1):
        b = to_dev(S.train_batch(bs))
        t_x, t_a = sample_times(variant, bs)
        Lx, La = forward_losses(model, b, t_x, t_a)
        loss = Lx + lam * La
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % log_every == 0:
            hist.append(dict(step=step, Lx=Lx.item(), La=La.item(), sec=time.time() - t0))
            print(f"  [{name}/{variant}] step {step} Lx={Lx.item():.3f} La={La.item():.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if variant != "text_only" and (step % 250 == 0 or step == steps):
            model.eval()
            vauc = lp_metrics(score_pairs(model, Sv, vpos, "full", k=k), score_pairs(model, Sv, vneg, "full", k=k))["auc"]
            model.train()
            hist.append(dict(step=step, val_auc=vauc))
            print(f"    val AUC {vauc:.4f}{' *' if vauc > best else ''}", flush=True)
            if vauc > best:
                best, best_step = vauc, step
                best_state = {k_: v.detach().to("cpu", copy=True) for k_, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
        hist.append(dict(selected_step=best_step, best_val_auc=best))
    os.makedirs(CKPT, exist_ok=True)
    path = os.path.join(CKPT, f"{name}_{variant}_s{seed}.pt")
    torch.save({k_: v for k_, v in model.state_dict().items()}, path)
    return model, hist, time.time() - t0


def load_model(name, variant, seed=0):
    m = TwinMask(topo=topo_of(variant)).to(DEV)
    m.load_state_dict(torch.load(os.path.join(CKPT, f"{name}_{variant}_s{seed}.pt"), map_location=DEV))
    return m.eval()


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def score_pairs(model, S, pairs, mode="full", cold=False, bs=32, k=16):
    """pairs: (2,P). Returns P(edge) for each pair under different conditioning modes."""
    out = []
    P = pairs.shape[1]
    for s in range(0, P, bs):
        chunk = pairs[:, s:s + bs].t().tolist()
        lists, unk = [], []
        for u, v in chunk:
            if cold:  # put the cold endpoint first
                if not S.cold[u]:
                    u, v = v, u
            if mode == "paironly":
                lists.append([u, v])
            else:
                lists.append(S.gather([u, v], k))
            unk.append([0] if cold else [])
        b = to_dev(S.build(lists, mask_pairs=[[(0, 1)]] * len(lists), unknown_nodes=unk))
        B = len(lists)
        if mode == "textmasked":
            t_x = torch.ones(B, device=DEV)
        else:
            t_x = torch.zeros(B, device=DEV)
        # time input for edges: actual fraction of masked pairs
        npair = (b["valid"].sum(1).float() * (b["valid"].sum(1).float() - 1) / 2).clamp(min=1)
        t_a = ((b["forced"].sum((1, 2)) / 2).float() / npair).clamp(max=1)
        ids_in, _, state_in, _, _, _ = corrupt(b, t_x, torch.zeros(B, device=DEV))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, _, elog = model(ids_in, b["attn"], b["valid"], state_in, t_x, t_a)
        out.append(torch.sigmoid(elog[:, 0, 1].float()).cpu())
    return torch.cat(out)


@torch.no_grad()
def node_embeddings(model, S, nodes, bs=32, k=16):
    hs = []
    for s in range(0, len(nodes), bs):
        lists = [S.gather([int(u)], k) for u in nodes[s:s + bs]]
        b = to_dev(S.build(lists))
        B = len(lists)
        z = torch.zeros(B, device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h, _, _ = model(b["ids"], b["attn"], b["valid"], b["state"], z, z)
        hs.append(h[:, 0].float().cpu())
    return torch.cat(hs)


@torch.no_grad()
def text_completion(model, S, nodes, ratio, mode, bs=32, k=16, seed=0):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    tot = dict(nll=0.0, hit=0.0, cnt=0.0)
    for s in range(0, len(nodes), bs):
        lists = [[int(u)] if mode == "alone" else S.gather([int(u)], k) for u in nodes[s:s + bs]]
        b = to_dev(S.build(lists))
        B, kk = b["valid"].shape
        rows = torch.zeros(B, kk, dtype=torch.bool, device=DEV); rows[:, 0] = True
        t_x = torch.full((B,), ratio, device=DEV)
        t_a = torch.ones(B, device=DEV) if mode == "edges_masked" else torch.zeros(B, device=DEV)
        r = forward_losses(model, b, t_x, t_a, gen=gen, text_rows=rows, reduce=False)
        tot["nll"] += r["tok_nll"].sum().item(); tot["hit"] += r["tok_hit"].sum().item(); tot["cnt"] += r["tok_cnt"].sum().item()
    return dict(nll=tot["nll"] / tot["cnt"], acc=tot["hit"] / tot["cnt"])


def subsample(pairs, m, rng):
    if pairs.shape[1] <= m:
        return pairs
    return pairs[:, torch.from_numpy(rng.choice(pairs.shape[1], m, replace=False))]


def evaluate(model, name, variant, seed=0, k=16, max_pairs=2000, max_nc=None, n_text=500):
    d = load(name)
    S = SubgraphSampler(d, k=k, seed=1234)
    rng = np.random.default_rng(1234)
    res = dict(dataset=name, variant=variant, seed=seed)
    tp, tn = subsample(d["test_pos"], max_pairs, rng), subsample(d["test_neg"], max_pairs, rng)
    cp, cn = subsample(d["cold_pos"], max_pairs, rng), subsample(d["cold_neg"], max_pairs, rng)
    if variant != "text_only":
        for mode in ("full", "textmasked", "paironly"):
            res[f"lp_{mode}"] = lp_metrics(score_pairs(model, S, tp, mode, k=k), score_pairs(model, S, tn, mode, k=k))
        res["cold"] = lp_metrics(score_pairs(model, S, cp, "full", cold=True, k=k),
                                 score_pairs(model, S, cn, "full", cold=True, k=k))
        print(f"  LP {res['lp_full']}  cold {res['cold']}", flush=True)
        hr = load_heart(name)
        if hr is not None:
            P, K = hr["neg"].shape[:2]
            ps = score_pairs(model, S, hr["pos"], "full", k=k)
            ns = score_pairs(model, S, hr["neg"].reshape(-1, 2).t(), "full", k=k, bs=64).view(P, K)
            res["heart"] = rank_metrics(ps, ns)
            print(f"  HeaRT {res['heart']}", flush=True)
    # node classification (linear probe on node states of clean subgraphs)
    n = d["num_nodes"]
    if max_nc is None or n <= max_nc:
        h = node_embeddings(model, S, np.arange(n), k=k)
        torch.save(h.half(), os.path.join(CKPT, f"emb_{name}_{variant}_s{seed}.pt"))
        res["nc"] = probe_nc(h, d, seed)
    else:  # large graph: probe on a subsample of train + test nodes
        tr = np.nonzero(d["train_mask"].numpy())[0]; te = np.nonzero(d["test_mask"].numpy())[0]
        tr = rng.choice(tr, min(len(tr), max_nc // 2), replace=False); te = rng.choice(te, min(len(te), max_nc // 2), replace=False)
        idx = np.r_[tr, te]
        h = torch.zeros(n, model.lm.config.hidden_size); h[idx] = node_embeddings(model, S, idx, k=k)
        dd = dict(d); m1 = torch.zeros(n, dtype=torch.bool); m1[tr] = True; m2 = torch.zeros(n, dtype=torch.bool); m2[te] = True
        dd["train_mask"], dd["test_mask"] = m1, m2
        res["nc"] = probe_nc(h, dd, seed, nodes=idx)
    print(f"  NC {res['nc']}", flush=True)
    if variant != "edge_only":
        te = np.nonzero(d["test_mask"].numpy())[0]
        te = rng.choice(te, min(n_text, len(te)), replace=False)
        res["text"] = {f"{mode}@{r}": text_completion(model, S, te, r, mode, k=k)
                       for r in (0.5, 1.0) for mode in ("ctx", "edges_masked", "alone")}
        print(f"  TEXT {res['text']}", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cora")
    ap.add_argument("--variant", default="joint")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_pairs", type=int, default=2000)
    ap.add_argument("--max_nc", type=int, default=None)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--lm_lr", type=float, default=2e-5)
    ap.add_argument("--tag", default="phase1")
    a = ap.parse_args()
    if a.eval_only:
        model, hist, sec = load_model(a.dataset, a.variant, a.seed), [], 0
    else:
        model, hist, sec = train(a.dataset, a.variant, a.steps, a.bs, a.k, seed=a.seed, lm_lr=a.lm_lr)
    model.eval()
    res = evaluate(model, a.dataset, a.variant, a.seed, a.k, a.max_pairs, a.max_nc)
    res.update(train_hist=hist, train_sec=sec, steps=a.steps)
    print(save_json(res, a.tag, f"{a.dataset}_{a.variant}_s{a.seed}.json"))
