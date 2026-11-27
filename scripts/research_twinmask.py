"""Sparse discrete TwinMask pilot with a free-running action-value controller."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv

from scripts.research_ensemble import tick
from twinmask.common import TimeEmb, save_json, set_seed
from twinmask.data import load, sample_negatives


MASK, PAD, OTHER = 0, 1, 2
UNKNOWN = 2
ACTIONS = ("joint", "text", "edge", "correct")


class SparseTwinMask(nn.Module):
    def __init__(self, vocab, length, width=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.position = nn.Parameter(torch.randn(length, width) * .02)
        self.local = nn.Conv1d(width, width, 3, padding=1)
        self.clock_x, self.clock_a = TimeEmb(width), TimeEmb(width)
        self.mix = nn.ModuleList(SAGEConv(width, width) for _ in range(2))
        self.norm = nn.LayerNorm(width)
        self.text = nn.Linear(width, vocab)
        self.pair_state = nn.Embedding(3, width)
        self.edge = nn.Sequential(nn.Linear(4*width+1, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, ids, pairs, state, base, token_query, edge_query, tx, ta):
        token = self.embed(ids) + self.position
        token = token + F.gelu(self.local(token.transpose(1, 2)).transpose(1, 2))
        valid = (ids != PAD).unsqueeze(-1)
        h = (token * valid).sum(1) / valid.sum(1).clamp_min(1)
        h = h + self.clock_x(h.new_tensor(tx)) + self.clock_a(h.new_tensor(ta))
        visible = torch.cat((base, pairs[:, state == 1]), 1)
        adjacency = torch.cat((visible, visible.flip(0)), 1)
        degree = torch.bincount(adjacency[0], minlength=len(h)).float().log1p()
        for layer in self.mix:
            h = h + F.gelu(layer(h, adjacency))
        i, j = pairs[:, edge_query]
        a, b = h[i], h[j]
        features = torch.cat((a+b, (a-b).abs(), a*b, self.pair_state(state[edge_query]),
                              (degree[i]+degree[j])[:, None]), 1)
        row, col = token_query
        return self.text(self.norm(token[row, col] + h[row])), self.edge(features).flatten()


def corruption(eligible, rate, rng):
    return eligible & (torch.rand(eligible.shape, device=eligible.device, generator=rng) < rate)


def token_loss(logits, truth):
    return F.cross_entropy(logits, truth) if truth.numel() else logits.sum()


def edge_loss(logits, truth):
    return F.binary_cross_entropy_with_logits(logits, truth.float()) if truth.numel() else logits.sum()


class Case:
    def __init__(self, context, ids, mask, pairs, labels, edge_mask, base):
        self.context, self.ids, self.mask = context, ids, mask
        self.pairs, self.labels, self.edge_mask, self.base = pairs, labels, edge_mask, base
        self.x_index = mask.nonzero(as_tuple=True)
        self.a_index = edge_mask.nonzero().flatten()
        self.x_truth = ids[self.x_index]
        self.a_truth = labels[self.a_index]

    def initial(self):
        return self.context.masked_fill(self.mask, MASK), self.labels.masked_fill(self.edge_mask, UNKNOWN)

    def predict(self, model, state, without=None, score=None):
        ids, edges = state
        tx = float((ids[self.x_index] == MASK).float().mean()) if self.x_truth.numel() else 0.
        ta = float((edges[self.a_index] == UNKNOWN).float().mean()) if self.a_truth.numel() else 0.
        x, a = score if score is not None else (slice(None), slice(None))
        base = self.base
        if without == "structure":
            base = base[:, :0]
            edges = torch.full_like(edges, UNKNOWN)
        if without == "text":
            ids = ids.masked_fill(ids != PAD, MASK)
        return model(ids, self.pairs, edges, base, tuple(i[x] for i in self.x_index), self.a_index[a], tx, ta)


def features(case, state, logits, remaining):
    x, a = state
    p = logits[0].softmax(-1)
    q = logits[1].sigmoid()
    mx = x[case.x_index] == MASK
    ma = a[case.a_index] == UNKNOWN
    ent_x = -(p * p.clamp_min(1e-9).log()).sum(1)
    ent_a = F.binary_cross_entropy_with_logits(logits[1], q, reduction="none")
    return [float(mx.float().mean()), float(ma.float().mean()), float(ent_x.mean()), float(ent_a.mean()),
            float((ent_x*mx).mean()), float((ent_a*ma).mean()), remaining]


def advance(case, state, logits, action, remaining, score=None):
    ids, edges = (x.clone() for x in state)
    px, pa = logits[0].softmax(-1), logits[1].sigmoid()
    px[:, :2] = 0
    confidence_x, predicted_x = px.max(1)
    confidence_a, predicted_a = (pa-.5).abs(), (pa >= .5).long()
    selections = (case.x_index, (case.a_index,))
    fill = (("text", ids, MASK, confidence_x, predicted_x), ("edge", edges, UNKNOWN, confidence_a, predicted_a))
    changes = []
    for modality, (name, values, unknown, confidence, prediction) in enumerate(fill):
        index = selections[modality]
        current = values[index]
        allowed = torch.ones_like(current, dtype=torch.bool)
        if score is not None:
            allowed[score[modality]] = False
        mutable = allowed & (current != unknown)
        missing = allowed & (current == unknown)
        remasked = 0
        if action == "correct" and remaining > 1:
            generated = mutable.nonzero().flatten()
            remasked = len(generated)//10
            low = generated[confidence[generated].argsort()[:remasked]]
            current[low] = unknown
        eligible = missing.nonzero().flatten()
        count = (len(eligible)+remaining-1)//remaining if remaining == 1 or action in ("joint", "correct", name) else 0
        high = eligible[confidence[eligible].argsort(descending=True)[:count]]
        current[high] = prediction[high]
        values[index] = current
        changes.append([count, remasked])
    return (ids, edges), changes


@torch.no_grad()
def rollout_values(model, cases, budget):
    rows, gains = [], []
    for case in cases:
        state = case.initial()
        score = tuple(torch.arange(0, size, 5, device=case.ids.device)
                      for size in (len(case.x_truth), len(case.a_truth)))
        for remaining in range(budget, 1, -1):
            logits = case.predict(model, state)
            x, a = score

            def loss(current):
                lx, la = case.predict(model, current, score=score)
                return float(token_loss(lx, case.x_truth[x]) + edge_loss(la, case.a_truth[a]))

            before = loss(state)
            candidates = [advance(case, state, logits, action, remaining, score)[0] for action in ACTIONS]
            rows.append(features(case, state, logits, remaining))
            gains.append([before-loss(candidate) for candidate in candidates])
            state = candidates[len(rows) % len(ACTIONS)]
    return np.array(rows), np.array(gains)


@torch.no_grad()
def reconstruct(model, case, policy, budget, controller):
    state = case.initial()
    trace = []
    edge_probs = torch.empty_like(case.a_truth, dtype=torch.float)
    for remaining in range(budget, 0, -1):
        logits = case.predict(model, state)
        pending = state[1][case.a_index] == UNKNOWN
        edge_probs[pending] = logits[1].sigmoid()[pending]
        action = policy
        if remaining == 1 or policy == "one_shot":
            action = "joint"
        elif policy == "adaptive":
            action = ACTIONS[controller.predict([features(case, state, logits, remaining)])[0].argmax()]
        elif policy == "entropy":
            tx, ta = features(case, state, logits, remaining)[2:4]
            action = "text" if tx / np.log(logits[0].shape[-1]) > ta / np.log(2) else "edge"
        state, changes = advance(case, state, logits, action, remaining)
        trace.append(dict(action=action, tokens=changes[0], pairs=changes[1]))
    ids, edges = state
    assert torch.equal(ids[~case.mask], case.context[~case.mask])
    assert torch.equal(edges[~case.edge_mask], case.labels[~case.edge_mask])
    assert (ids[case.x_index] != MASK).all() and (edges[case.a_index] != UNKNOWN).all()
    y, p = case.a_truth.cpu().numpy(), edge_probs.cpu().numpy()
    return dict(token_accuracy=float((ids[case.x_index] == case.x_truth).float().mean()),
                edge_accuracy=float((edges[case.a_index] == case.a_truth).float().mean()),
                edge_auc=roc_auc_score(y, p), edge_ap=average_precision_score(y, p),
                edge_brier=float(np.mean((y-p)**2)), calls=budget, trace=trace)


def run(name, seed, epochs, device, width, vocab_size, clock_training="mixed"):
    set_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rng = torch.Generator(device=device).manual_seed(seed)
    data = load(name)
    raw = data["tokens"][:, 1:17].long().to(device)
    valid = (raw > 2) & (raw != 50264)
    pairs = data["train_pos"].to(device)
    forbid = set()
    for key in ("train_pos", "val_pos", "test_pos", "cold_pos", "cold_neg", "val_neg", "test_neg"):
        a, b = data[key]
        forbid.update((a*len(raw)+b).tolist())
    perm = torch.randperm(pairs.shape[1], generator=rng, device=device)
    fit_edges, controller_edges, dev_edges, base_edges = perm.split(
        [len(perm)//5, len(perm)//10, len(perm)//10, len(perm)-len(perm)//5-2*(len(perm)//10)])
    count = len(fit_edges)+len(controller_edges)+len(dev_edges)
    negatives = sample_negatives(len(raw), count, forbid, np.random.default_rng(seed)).to(device)
    neg_fit, neg_controller, neg_dev = negatives.split([len(fit_edges), len(controller_edges), len(dev_edges)], dim=1)
    order = torch.randperm(len(raw), generator=rng, device=device)
    train_nodes, controller_nodes, dev_nodes, test_nodes = order.split(
        [len(raw)//2, len(raw)//5, len(raw)//10, len(raw)-len(raw)//2-len(raw)//5-len(raw)//10])
    counts = torch.bincount(raw[train_nodes][valid[train_nodes]], minlength=50265)
    vocabulary = counts.topk(min(vocab_size, int((counts > 0).sum()))).indices
    mapping = torch.full((50265,), OTHER, device=device, dtype=torch.long)
    mapping[vocabulary] = torch.arange(3, len(vocabulary)+3, device=device)
    ids = mapping[raw].masked_fill(~valid, PAD)
    eligible = ids > OTHER
    train_eligible = torch.zeros_like(eligible)
    train_eligible[train_nodes] = eligible[train_nodes]
    reserved = torch.zeros_like(eligible)
    reserved[controller_nodes] = eligible[controller_nodes]
    reserved[dev_nodes] = eligible[dev_nodes]
    reserved[test_nodes] = eligible[test_nodes]
    context = ids.masked_fill(reserved, MASK)

    def make_case(nodes, positives, negative, base, text_rate, edge_rate):
        candidates = torch.cat((positives, negative), 1)
        labels = torch.arange(candidates.shape[1], device=device) < positives.shape[1]
        shuffle = torch.randperm(len(labels), generator=rng, device=device)
        candidates, labels = candidates[:, shuffle], labels[shuffle].long()
        pool = torch.zeros_like(eligible)
        pool[nodes] = eligible[nodes]
        masked = corruption(pool, text_rate, rng)
        backdrop = context.clone()
        backdrop[nodes] = ids[nodes]
        hold = torch.zeros_like(eligible)
        hold[nodes] = True
        backdrop[(reserved & ~hold)] = MASK
        return Case(backdrop, ids, masked, candidates, labels, corruption(torch.ones_like(labels).bool(), edge_rate, rng), base)

    model = SparseTwinMask(len(vocabulary)+3, ids.shape[1], width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=.003)
    start = tick(device)
    base = pairs[:, base_edges]
    for epoch in range(epochs):
        tx, ta = (.25, .5, .75, .9)[epoch % 4], (.25, .5, .75, .9)[(epoch//4) % 4]
        if clock_training == "tied" or epoch % 5 == 0:
            ta = tx
        case = make_case(train_nodes, pairs[:, fit_edges], neg_fit, base, tx, ta)
        state = case.initial()
        dropout = {0: "structure", 1: "text"}.get(epoch % 10)
        lx, la = case.predict(model, state, without=dropout)
        loss = token_loss(lx, case.x_truth) + edge_loss(la, case.a_truth)
        opt.zero_grad()
        loss.backward()
        opt.step()
    train_seconds = tick(device)-start
    model.eval()
    budget = 4
    start = tick(device)
    # Context here contains fitting edges and never any controller/development candidate positives.
    richer_base = pairs[:, torch.cat((base_edges, fit_edges))]
    rates = [(.25,.75), (.5,.5), (.75,.25), (.9,.9)]
    controller_cases = [make_case(controller_nodes, pairs[:, controller_edges], neg_controller, richer_base, x, a)
                        for x, a in rates*3]
    dev_cases = [make_case(dev_nodes, pairs[:, dev_edges], neg_dev, richer_base, x, a) for x, a in rates]
    x, gain = rollout_values(model, controller_cases, budget)
    controller = Ridge(alpha=.01).fit(x, gain)
    dx, dgain = rollout_values(model, dev_cases, budget)
    predicted = controller.predict(dx)
    rho = spearmanr(predicted.ravel(), dgain.ravel()).statistic
    controller_seconds = tick(device)-start
    def fixed_score(policy):
        return np.mean([row["token_accuracy"]+row["edge_accuracy"] for row in
                        (reconstruct(model, case, policy, budget, controller) for case in dev_cases)])
    fixed = max(ACTIONS, key=fixed_score)
    base = pairs
    # Validation positives stay outside the message graph, including at test time.
    positives, negatives = (data[key].to(device) for key in ("test_pos", "test_neg"))
    rows = []
    masks = []
    for tx, ta in rates:
        case = make_case(test_nodes, positives, negatives, base, tx, ta)
        initial = case.initial()
        masks.append(dict(tokens=case.mask.cpu(), pairs=case.pairs.cpu(), edge_mask=case.edge_mask.cpu()))
        with torch.no_grad():
            lx, la = case.predict(model, initial)
            without_a, _ = case.predict(model, initial, without="structure")
            _, without_x = case.predict(model, initial, without="text")
            ce, bce = token_loss(lx, case.x_truth), edge_loss(la, case.a_truth)
            row = dict(text_rate=tx, edge_rate=ta, tokens=len(case.x_truth), pairs=len(case.a_truth),
                       initial_token_ce=float(ce), initial_edge_bce=float(bce),
                       structure_to_text=float(token_loss(without_a, case.x_truth)-ce),
                       text_to_edge=float(edge_loss(without_x, case.a_truth)-bce), metrics={})
            for policy in ("one_shot", *ACTIONS, "entropy", "adaptive"):
                start = tick(device)
                result = reconstruct(model, case, policy, 1 if policy == "one_shot" else budget, controller)
                result["seconds"] = tick(device)-start
                row["metrics"][policy] = result
        rows.append(row)
    directory = Path("checkpoints/research_twinmask" + ("_tied" if clock_training == "tied" else ""))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}_{seed}.pt"
    torch.save(dict(model=model.state_dict(), vocabulary=vocabulary.cpu(), masks=masks, node_split=order.cpu(),
                    edge_split=perm.cpu(), controller_coef=controller.coef_, controller_intercept=controller.intercept_), path)
    return dict(seed=seed, epochs=epochs, width=width, vocab=len(vocabulary), best_fixed=fixed,
                test_vocab_coverage=float(eligible[test_nodes].sum()/valid[test_nodes].sum()),
                dev_gain_spearman=float(rho) if np.isfinite(rho) else None,
                dev_negative_gain_accuracy=float(((predicted < 0) == (dgain < 0)).mean()),
                training_seconds=train_seconds, controller_seconds=controller_seconds,
                checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                peak_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else None, cases=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer", "pubmed"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--vocab", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clock-training", choices=("mixed", "tied"), default="mixed")
    args = parser.parse_args()
    for name in args.datasets:
        rows = []
        for seed in args.seeds:
            rows.append(run(name, seed, args.epochs, torch.device(args.device), args.width, args.vocab,
                            args.clock_training))
            suffix = "_tied" if args.clock_training == "tied" else ""
            save_json(rows, "research_twinmask", f"{name}{suffix}.json")
            print(name, seed, rows[-1]["best_fixed"], rows[-1]["dev_gain_spearman"], flush=True)


if __name__ == "__main__":
    main()
