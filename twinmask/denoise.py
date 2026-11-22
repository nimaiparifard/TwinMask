"""Multi-step joint reverse process for a trained (frozen) TwinMask model.

This module is an *add-on*: it only imports from the original code and does not change training or
the one-shot evaluation in `tm.py`.

Task (cold-start imputation, leakage-free):
  * an episode is a subgraph whose node 0 is a COLD node: none of its edges and none of its text were
    ever seen in training (cold nodes have no training edges, so the training sampler never reaches them);
  * a fraction `rho` of node 0's tokens is [M] and ALL of node 0's pairs are [M];
  * ground truth: the masked tokens, and node 0's held-out edges (`cold_pos`) to the other k-1 nodes.

Reverse process: in N steps the masked items of both modalities are committed. A *controller* decides
how many tokens and how many edge variables are committed at each step, so it chooses a path through
the (t_x, t_a) square, on which the denoiser was trained. Inside a modality, items are committed in
order of confidence (or at random, for the ablation).

Modes:
  generate : committed values are the model's own predictions (errors propagate).
  teacher  : committed values are the ground truth. The summed NLL is then an order-dependent
             (any-order autoregressive) likelihood of the masked items.
Each item is scored once, when it is committed: -log p(truth | everything committed before).
"""
import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from .model import ABSENT, MASKED, PRESENT
from .sampler import SubgraphSampler
from .tm import DEV, MASK_ID, to_dev

EPS = 1e-6


# ------------------------------------------------------------------ episodes
class ColdEpisodes:
    def __init__(self, d, k=12, n_pos=3, n_neg=3):
        self.S = SubgraphSampler(d, k=k, seed=0)
        self.k, self.n_pos, self.n_neg = k, n_pos, n_neg
        cold = self.S.cold
        nbr = defaultdict(set)
        for a, b in d["cold_pos"].t().tolist():
            c, w = (a, b) if cold[a] else (b, a)
            nbr[c].add(w)
        self.nbr = nbr
        self.nodes = np.array(sorted(nbr))  # cold nodes with >= 1 warm neighbour

    def split(self, seed=0, frac=0.5):
        """Disjoint cold-node sets: one to fit controllers / select paths, one to report on."""
        p = np.random.default_rng(seed).permutation(self.nodes)
        m = int(frac * len(p))
        return p[:m], p[m:]

    def make(self, cold_nodes, rho, seed):
        """Builds a batch of episodes (deterministic in `seed`)."""
        rng = np.random.default_rng(seed)
        self.S.rng = rng  # gather() draws from the sampler's rng
        lists, truth = [], []
        for c in cold_nodes:
            c = int(c)
            pos = list(self.nbr[c]); rng.shuffle(pos)
            neg = [int(v) for v in rng.choice(self.S.warm, 4 * self.n_neg) if int(v) not in self.nbr[c]][:self.n_neg]
            anchors = pos[:self.n_pos] + neg
            rng.shuffle(anchors)
            sub = self.S.gather(anchors, self.k - 1)  # warm nodes + their training neighbourhood
            lists.append([c] + sub)
            truth.append([0.0] + [1.0 if v in self.nbr[c] else 0.0 for v in sub])
        b = to_dev(self.S.build(lists))
        B, k, L = b["ids"].shape
        tr = torch.zeros(B, k, device=DEV)
        for i, t in enumerate(truth):
            tr[i, :len(t)] = torch.tensor(t, device=DEV)
        pos = torch.arange(L, device=DEV)[None]
        maskable = b["attn"][:, 0] & (pos > 0)
        r = torch.from_numpy(rng.random((B, L))).to(DEV)
        tok_m = maskable & (r < rho)
        edge_m = b["valid"].clone(); edge_m[:, 0] = False
        return dict(b=b, true_e=tr, tok_m=tok_m, edge_m=edge_m, maskable=maskable)


def repeat(ep, G):
    """Repeat every episode G times (for group-relative policy optimisation)."""
    def rep(x):
        return x.repeat_interleave(G, 0) if torch.is_tensor(x) else {kk: rep(v) for kk, v in x.items()}
    return {kk: rep(v) for kk, v in ep.items()}


# ------------------------------------------------------------------ one reverse trajectory
class Rollout:
    def __init__(self, model, ep, mode="generate"):
        self.m, self.mode = model, mode
        b = ep["b"]
        self.attn, self.valid = b["attn"], b["valid"]
        self.true_ids = b["ids"][:, 0].long()
        self.true_e = ep["true_e"]
        self.ids = b["ids"].clone()
        self.tok_m = ep["tok_m"].clone()
        self.ids[:, 0] = torch.where(self.tok_m, torch.full_like(self.ids[:, 0], MASK_ID), self.ids[:, 0])
        self.edge_m = ep["edge_m"].clone()
        self.edge_m0 = ep["edge_m"].clone()
        st =b["state"].clone()
        st[:, 0, :] = torch.where(self.edge_m, torch.full_like(st[:, 0], MASKED), st[:, 0])
        st[:, :, 0] = st[:, 0, :]
        st[:, 0, 0] = b["state"][:, 0, 0]
        self.state = st
        self.n_maskable = ep["maskable"].sum(1).float().clamp(min=1)
        self.n_tok0 = self.tok_m.sum(1)
        self.n_edge0 = self.edge_m.sum(1)
        nv = self.valid.sum(1).float()
        self.n_pair = (nv * (nv - 1) / 2).clamp(min=1)
        B, L = self.tok_m.shape
        self.rec_tok_lp = torch.zeros(B, L, device=DEV)
        self.rec_tok_hit = torch.zeros(B, L, device=DEV)
        self.rec_edge_p = torch.zeros_like(self.true_e)
        self.forwards = 0

    # -------------------------------------------------- denoiser call
    @torch.no_grad()
    def observe(self):
        B, k, L = self.ids.shape
        t_x = self.tok_m.sum(1).float() / self.n_maskable
        t_a = self.edge_m.sum(1).float() / self.n_pair
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, x, elog = self.m(self.ids, self.attn, self.valid, self.state, t_x, t_a)
        self.forwards += 1
        x0 = x.view(B, k, L, -1)[:, 0]
        conf = torch.zeros(B, L, device=DEV); pred = self.true_ids.clone(); lpt = torch.zeros(B, L, device=DEV)
        sel = self.tok_m
        if sel.any():
            xs, tgt = x0[sel], self.true_ids[sel]
            cs, ps, ls = [], [], []
            for c in range(0, xs.shape[0], 2048):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = self.m.token_logits(xs[c:c + 2048])
                lp = F.log_softmax(lg.float(), -1)
                mx, am = lp.max(-1)
                cs.append(mx.exp()); ps.append(am); ls.append(lp.gather(1, tgt[c:c + 2048, None]).squeeze(1))
            conf[sel] = torch.cat(cs); pred[sel] = torch.cat(ps); lpt[sel] = torch.cat(ls)
        ep = torch.sigmoid(elog[:, 0, :].float())
        return dict(tok_conf=conf, tok_pred=pred, tok_lpt=lpt, edge_p=ep, edge_conf=torch.maximum(ep, 1 - ep))

    # -------------------------------------------------- commit
    @staticmethod
    def _pick(score, avail, n):
        """Boolean (B,M): the n[b] highest-scoring available items of each row."""
        score = score.masked_fill(~avail, float("-inf"))
        order = score.argsort(1, descending=True)
        rank = torch.empty_like(order)
        rank.scatter_(1, order, torch.arange(score.shape[1], device=DEV).expand_as(order))
        return (rank < n[:, None]) & avail

    def _scores(self, obs, order):
        if order == "random":
            return torch.rand_like(obs["tok_conf"]), torch.rand_like(obs["edge_conf"])
        return obs["tok_conf"], obs["edge_conf"]

    def commit_counts(self, obs, n_t, n_e, order="confidence"):
        st, se = self._scores(obs, order)
        self._apply(obs, self._pick(st, self.tok_m, n_t), self._pick(se, self.edge_m, n_e))

    def commit_joint(self, obs, n_tot, order="confidence"):
        """Cross-modal confidence: one ranking over tokens and edge variables together."""
        st, se = self._scores(obs, order)
        L = st.shape[1]
        pick = self._pick(torch.cat([st, se], 1), torch.cat([self.tok_m, self.edge_m], 1), n_tot)
        self._apply(obs, pick[:, :L], pick[:, L:])

    def _apply(self, obs, pt, pe):
        # text
        val = obs["tok_pred"] if self.mode == "generate" else self.true_ids
        self.ids[:, 0] = torch.where(pt, val.to(self.ids.dtype), self.ids[:, 0])
        self.rec_tok_lp = torch.where(pt, obs["tok_lpt"], self.rec_tok_lp)
        self.rec_tok_hit = torch.where(pt, (obs["tok_pred"] == self.true_ids).float(), self.rec_tok_hit)
        self.tok_m = self.tok_m & ~pt
        # edges
        ev = (obs["edge_p"] > 0.5) if self.mode == "generate" else (self.true_e > 0.5)
        new = torch.where(ev, torch.full_like(self.state[:, 0], PRESENT), torch.full_like(self.state[:, 0], ABSENT))
        row = torch.where(pe, new, self.state[:, 0])
        self.state[:, 0, :] = row
        self.state[:, :, 0] = row
        self.rec_edge_p = torch.where(pe, obs["edge_p"], self.rec_edge_p)
        self.edge_m = self.edge_m & ~pe

    # -------------------------------------------------- summary
    def remaining(self):
        return self.tok_m.sum(1), self.edge_m.sum(1)

    def episode_metrics(self):
        nt = self.n_tok0.float().clamp(min=1)
        ne = self.n_edge0.float().clamp(min=1)
        em = self.edge_m0
        p =self.rec_edge_p.clamp(EPS, 1 - EPS)
        bce = -(self.true_e * p.log() + (1 - self.true_e) * (1 - p).log())
        hit_e = ((self.rec_edge_p > 0.5).float() == self.true_e).float()
        return dict(text_nll=-self.rec_tok_lp.sum(1) / nt, text_acc=self.rec_tok_hit.sum(1) / nt,
                    edge_nll=(bce * em).sum(1) / ne, edge_acc=(hit_e * em).sum(1) / ne, edge_mask=em)


# ------------------------------------------------------------------ controllers
def _counts_from_ratio(ratio, rem_t, rem_e, steps_left):
    """Split this step's budget ceil(remaining / steps_left) between the modalities."""
    budget = torch.ceil((rem_t + rem_e).float() / steps_left).long()
    n_t = torch.minimum(rem_t, torch.round(ratio * budget).long())
    n_e = torch.minimum(rem_e, budget - n_t)
    n_t = torch.minimum(rem_t, budget - n_e)  # if edges run out, the rest of the budget goes to text
    return n_t, n_e


class PathController:
    """Fixed path u -> (f_x(u), f_a(u)): cumulative fraction of the masked tokens / edges committed."""
    def __init__(self, fx, fa, order="confidence"):
        self.fx, self.fa, self.order = fx, fa, order

    def step(self, ro, obs, s, N):
        u = (s + 1) / N
        rt, re = ro.remaining()
        done_t, done_e = ro.n_tok0 - rt, ro.n_edge0 - re
        tgt_t = torch.round(self.fx(u) * ro.n_tok0.float()).long() if s < N - 1 else ro.n_tok0
        tgt_e = torch.round(self.fa(u) * ro.n_edge0.float()).long() if s < N - 1 else ro.n_edge0
        ro.commit_counts(obs, (tgt_t - done_t).clamp(min=0), (tgt_e - done_e).clamp(min=0), self.order)
        return None


class JointConfidenceController:
    """Same total budget per step; which modality gets it is decided by cross-modal confidence."""
    def __init__(self, order="confidence"):
        self.order = order

    def step(self, ro, obs, s, N):
        rt, re = ro.remaining()
        n = torch.ceil((rt + re).float() / (N - s)).long()
        ro.commit_joint(obs, n, self.order)
        return None


def _lead(g):
    return (lambda u: u ** (2.0 ** g)), (lambda u: u ** (2.0 ** -g))


PATHS = {
    "diagonal": (lambda u: u, lambda u: u),
    "text_first": (lambda u: min(1.0, 2 * u), lambda u: max(0.0, 2 * u - 1)),
    "edge_first": (lambda u: max(0.0, 2 * u - 1), lambda u: min(1.0, 2 * u)),
    "text_lead2": _lead(-2), "text_lead1": _lead(-1),  # f_x = u^0.25 / u^0.5 (text early)
    "edge_lead1": _lead(1), "edge_lead2": _lead(2),
}


def make_controller(name):
    if name == "joint_conf":
        return JointConfidenceController()
    if name == "diagonal_random":
        return PathController(*PATHS["diagonal"], order="random")
    return PathController(*PATHS[name])


def run(model, ep, controller, N, mode="generate"):
    """Runs one batched reverse trajectory; returns the Rollout (metrics) and the controller's log-probs."""
    ro = Rollout(model, ep, mode)
    logps = []
    for s in range(N):
        obs = ro.observe()
        lp = controller.step(ro, obs, s, N)
        if lp is not None:
            logps.append(lp)
    rt, re = ro.remaining()
    assert int(rt.sum()) == 0 and int(re.sum()) == 0, "controller left items masked"
    return ro, (torch.stack(logps, 1) if logps else None)
