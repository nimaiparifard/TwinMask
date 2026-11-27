"""Sparse discrete TwinMask pilot with a free-running action-value controller."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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
    def __init__(self, vocab, length, width=64, pretrained=None, vocabulary=None):
        super().__init__()
        self.encoder = None
        if pretrained is None:
            self.embed = nn.Embedding(vocab, width)
            self.position = nn.Parameter(torch.randn(length, width) * .02)
            self.local = nn.Conv1d(width, width, 3, padding=1)
        else:
            from transformers import AutoModel
            self.encoder = AutoModel.from_pretrained(pretrained, local_files_only=True).requires_grad_(False).eval()
            self.project = nn.Linear(self.encoder.config.hidden_size, width)
            self.register_buffer("original_ids", torch.cat((torch.tensor([50264, 1, 3]), vocabulary.cpu())))
        self.clock_x, self.clock_a = TimeEmb(width), TimeEmb(width)
        self.mix = nn.ModuleList(SAGEConv(width, width) for _ in range(2))
        self.norm = nn.LayerNorm(width)
        self.text = nn.Linear(width, vocab)
        self.pair_state = nn.Embedding(3, width)
        self.edge = nn.Sequential(nn.Linear(4*width+1, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, ids, pairs, state, base, token_query, edge_query, tx, ta, generated=None):
        row, col = token_query
        valid = (ids != PAD).unsqueeze(-1)
        if self.encoder is None:
            token = self.embed(ids) + self.position
            token = token + F.gelu(self.local(token.transpose(1, 2)).transpose(1, 2))
            h = (token * valid).sum(1) / valid.sum(1).clamp_min(1)
            selected = token[row, col]
        else:
            self.encoder.eval()
            pooled, queries, ordering = [], [], []
            for start in range(0, len(ids), 64):
                mask = valid[start:start+64]
                batch = self.original_ids[ids[start:start+64]]
                with torch.no_grad():
                    hidden = self.encoder(input_ids=batch, attention_mask=mask.squeeze(-1)).last_hidden_state
                in_batch = (row >= start) & (row < start+64)
                pooled.append((hidden * mask).sum(1) / mask.sum(1).clamp_min(1))
                queries.append(hidden[row[in_batch]-start, col[in_batch]])
                ordering.append(in_batch.nonzero().flatten())
            h = self.project(torch.cat(pooled).to(self.project.weight.dtype))
            selected = self.project(torch.cat(queries)[torch.cat(ordering).argsort()].to(self.project.weight.dtype))
        h = h + self.clock_x(h.new_tensor(tx)) + self.clock_a(h.new_tensor(ta))
        weights = (state == 1).float()
        if generated is not None:
            weights = torch.where(generated >= 0, generated, weights)
        active = weights > 0
        visible = torch.cat((base, pairs[:, active]), 1)
        adjacency = torch.cat((visible, visible.flip(0)), 1)
        weight = torch.cat((h.new_ones(base.shape[1]), weights[active])).repeat(2)
        source, target = adjacency
        mass = h.new_zeros(len(h)).index_add_(0, target, weight)
        degree = mass.log1p()
        for layer in self.mix:
            neighbors = torch.zeros_like(h).index_add_(0, target, h[source] * weight[:, None])
            h = h + F.gelu(layer.lin_l(neighbors / mass.clamp_min(1)[:, None]) + layer.lin_r(h))
        i, j = pairs[:, edge_query]
        a, b = h[i], h[j]
        features = torch.cat((a+b, (a-b).abs(), a*b, self.pair_state(state[edge_query]),
                              (degree[i]+degree[j])[:, None]), 1)
        return self.text(self.norm(selected + h[row])), self.edge(features).flatten()


def corruption(eligible, rate, rng):
    return eligible & (torch.rand(eligible.shape, device=eligible.device, generator=rng) < rate)


def token_loss(logits, truth):
    return F.cross_entropy(logits, truth) if truth.numel() else logits.sum()


def edge_loss(logits, truth):
    return F.binary_cross_entropy_with_logits(logits, truth.float()) if truth.numel() else logits.sum()


class Case:
    def __init__(self, context, ids, mask, pairs, labels, edge_mask, base, rates=(1., 1.), soft_edges=False,
                 unscored_tokens=0):
        self.context, self.ids, self.mask = context, ids, mask
        self.pairs, self.labels, self.edge_mask, self.base = pairs, labels, edge_mask, base
        self.x_index = mask.nonzero(as_tuple=True)
        self.a_index = edge_mask.nonzero().flatten()
        self.x_truth = ids[self.x_index]
        self.a_truth = labels[self.a_index]
        self.rates, self.soft_edges = rates, soft_edges
        self.unscored_tokens = unscored_tokens

    def initial(self):
        return (self.context.masked_fill(self.mask, MASK), self.labels.masked_fill(self.edge_mask, UNKNOWN),
                torch.full_like(self.labels, -1., dtype=torch.float))

    def clocks(self, state):
        ids, edges, _ = state
        pending = ids[self.x_index] == MASK
        if self.unscored_tokens:
            tx = self.rates[0] * (float(pending.sum()) + self.unscored_tokens) / (pending.numel() + self.unscored_tokens)
        else:
            tx = self.rates[0] * float(pending.float().mean()) if pending.numel() else 0.
        pending = edges[self.a_index] == UNKNOWN
        ta = self.rates[1] * float(pending.float().mean()) if pending.numel() else 0.
        return tx, ta

    def predict(self, model, state, without=None, score=None):
        ids, edges, generated = state
        tx, ta = self.clocks(state)
        x, a = score if score is not None else (slice(None), slice(None))
        base = self.base
        if without == "structure":
            base = base[:, :0]
            edges = torch.full_like(edges, UNKNOWN)
            generated = None
        if without == "text":
            ids = ids.masked_fill(ids != PAD, MASK)
        return model(ids, self.pairs, edges, base, tuple(i[x] for i in self.x_index), self.a_index[a], tx, ta,
                     generated if self.soft_edges else None)

    def without_generated(self, state):
        ids, edges, weights = (value.clone() for value in state)
        ids[self.x_index] = MASK
        edges[self.a_index] = UNKNOWN
        weights[self.a_index] = -1
        return ids, edges, weights


def features(case, state, logits, remaining):
    x, a, _ = state
    p = logits[0].softmax(-1)
    q = logits[1].sigmoid()
    mx = x[case.x_index] == MASK
    ma = a[case.a_index] == UNKNOWN
    ent_x = -(p * p.clamp_min(1e-9).log()).sum(1)
    ent_a = F.binary_cross_entropy_with_logits(logits[1], q, reduction="none")
    return [float(mx.float().mean()), float(ma.float().mean()), float(ent_x.mean()), float(ent_a.mean()),
            float((ent_x*mx).mean()), float((ent_a*ma).mean()), remaining]


def advance(case, state, logits, action, remaining, score=None, rescored=None):
    ids, edges, weights = (x.clone() for x in state)
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
        if action == "correct" and remaining > 1 and rescored is not None:
            generated = mutable.nonzero().flatten()
            remasked = len(generated)//10
            posterior = rescored[modality][generated]
            if modality == 0:
                support = posterior.softmax(-1).gather(1, current[generated, None]).flatten()
            else:
                probability = posterior.sigmoid()
                support = torch.where(current[generated] == 1, probability, 1-probability)
            low = generated[support.argsort()[:remasked]]
            current[low] = unknown
            if modality == 1:
                weights[case.a_index[low]] = -1
        eligible = missing.nonzero().flatten()
        count = (len(eligible)+remaining-1)//remaining if remaining == 1 or action in ("joint", "correct", name) else 0
        high = eligible[confidence[eligible].argsort(descending=True)[:count]]
        current[high] = prediction[high]
        if modality == 1:
            weights[case.a_index[high]] = pa[high]
        values[index] = current
        changes.append([count, remasked])
    return (ids, edges, weights), changes


def can_correct(case, state, remaining):
    return remaining >= 3 and (bool((state[0][case.x_index] != MASK).any()) or
                               bool((state[1][case.a_index] != UNKNOWN).any()))


@torch.no_grad()
def rollout_values(model, cases, budget):
    rows, gains = [], []
    for case in cases:
        state = case.initial()
        score = tuple(torch.arange(0, size, 5, device=case.ids.device)
                      for size in (len(case.x_truth), len(case.a_truth)))
        remaining = budget
        while remaining > 1:
            logits = case.predict(model, state)
            x, a = score

            def loss(current):
                lx, la = case.predict(model, current, score=score)
                return float(token_loss(lx, case.x_truth[x]) + edge_loss(la, case.a_truth[a]))

            before = loss(state)
            correction = can_correct(case, state, remaining)
            rescored = case.predict(model, case.without_generated(state)) if correction else None
            candidates = [advance(case, state, logits, action, remaining-int(action == "correct" and correction),
                                  score, rescored)[0] for action in ACTIONS]
            rows.append(features(case, state, logits, remaining))
            gains.append([(before-loss(candidate))/(1+int(action == "correct" and correction))
                          for action, candidate in zip(ACTIONS, candidates)])
            chosen = len(rows) % len(ACTIONS)
            state = candidates[chosen]
            remaining -= 1 + int(ACTIONS[chosen] == "correct" and correction)
    return np.array(rows), np.array(gains)


@torch.no_grad()
def gate_inputs(model, case):
    state = case.initial()
    joint = case.predict(model, state)
    text, _ = case.predict(model, state, without="structure")
    _, edge = case.predict(model, state, without="text")
    return features(case, state, joint, 1) + list(case.clocks(state)), (text, edge), joint


def gate_results(case, solo, joint, weights):
    px = torch.lerp(solo[0].softmax(-1), joint[0].softmax(-1), float(weights[0]))
    pa = torch.lerp(solo[1].sigmoid(), joint[1].sigmoid(), float(weights[1]))
    return dict(token_ce=F.nll_loss(px.clamp_min(1e-9).log(), case.x_truth).item(),
                token_accuracy=float((px[:, 2:].argmax(1)+2 == case.x_truth).float().mean()),
                edge_bce=F.binary_cross_entropy(pa, case.a_truth.float()).item(),
                edge_auc=roc_auc_score(case.a_truth.cpu(), pa.cpu()), weights=list(map(float, weights)))


def oracle_weights(case, solo, joint):
    rows = [gate_results(case, solo, joint, (w, w)) for w in np.linspace(0, 1, 5)]
    return [min(rows, key=lambda row: row[key])["weights"][i]
            for i, key in enumerate(("token_ce", "edge_bce"))]


@torch.no_grad()
def reconstruct(model, case, policy, budget, controller):
    state = case.initial()
    trace = []
    edge_probs = torch.empty_like(case.a_truth, dtype=torch.float)
    remaining = budget
    while remaining:
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
        correction = action == "correct" and can_correct(case, state, remaining)
        rescored = case.predict(model, case.without_generated(state)) if correction else None
        if correction:
            remaining -= 1
            trace.append(dict(action="rescore"))
        state, changes = advance(case, state, logits, action, remaining, rescored=rescored)
        trace.append(dict(action=action, tokens=changes[0], pairs=changes[1]))
        remaining -= 1
    ids, edges, _ = state
    assert torch.equal(ids[~case.mask], case.context[~case.mask])
    assert torch.equal(edges[~case.edge_mask], case.labels[~case.edge_mask])
    assert (ids[case.x_index] != MASK).all() and (edges[case.a_index] != UNKNOWN).all()
    y, p = case.a_truth.cpu().numpy(), edge_probs.cpu().numpy()
    return dict(token_accuracy=float((ids[case.x_index] == case.x_truth).float().mean()),
                edge_accuracy=float((edges[case.a_index] == case.a_truth).float().mean()),
                edge_auc=roc_auc_score(y, p), edge_ap=average_precision_score(y, p),
                edge_brier=float(np.mean((y-p)**2)), calls=budget, trace=trace)


SYNTHETIC = {
    "synthetic_reference": (.85, .8, 0.),
    "synthetic_low_homophily": (.30, .8, 0.),
    "synthetic_weak_text": (.85, .2, 0.),
    "synthetic_noisy_context": (.85, .8, .25),
}


def synthetic_graph(name):
    homophily, lexical, noise = SYNTHETIC[name]
    streams = [np.random.default_rng(seed) for seed in range(100, 105)]
    n = 480
    classes = streams[0].integers(4, size=n)
    u, v = np.triu_indices(n, k=1)
    same = classes[u] == classes[v]
    probability = 12/n * np.where(same, 4*homophily, 4*(1-homophily)/3)
    connected = streams[1].random(len(u)) < probability
    edges = torch.from_numpy(np.stack((u[connected], v[connected])))
    order = streams[2].permutation(edges.shape[1])
    test, val, train = np.split(order, [len(order)//8, len(order)//4])
    forbidden = set((edges[0]*n+edges[1]).tolist())
    negatives = sample_negatives(n, len(test)+len(val), forbidden, streams[3])
    forbidden.update((negatives[0]*n+negatives[1]).tolist())
    noise_edges = sample_negatives(n, int(len(train)*noise), forbidden, np.random.default_rng(105)) if noise else edges[:, :0]
    token_class = np.where(streams[4].random((n, 16)) < lexical, classes[:, None],
                           streams[4].integers(4, size=(n, 16)))
    tokens = torch.ones((n, 17), dtype=torch.long)
    tokens[:, 1:] = torch.from_numpy(4 + 8*token_class + streams[4].integers(8, size=(n, 16)))
    return dict(tokens=tokens, train_pos=edges[:, train], val_pos=edges[:, val], test_pos=edges[:, test],
                test_neg=negatives[:, :len(test)], val_neg=negatives[:, len(test):len(test)+len(val)],
                cold_pos=edges[:, :0], cold_neg=edges[:, :0],
                noise_edges=noise_edges)


def resplit_link_data(data, seed):
    rng = np.random.default_rng(seed)
    n = len(data["tokens"])
    edges = torch.cat(tuple(data[k] for k in ("train_pos", "val_pos", "test_pos")), 1)
    n_test, n_val = data["test_pos"].shape[1], data["val_pos"].shape[1]
    test, val, train = np.split(rng.permutation(edges.shape[1]), [n_test, n_test+n_val])
    positives = torch.cat((edges, data["cold_pos"]), 1)
    excluded = set((positives[0]*n+positives[1]).tolist())
    neg = sample_negatives(n, n_test+n_val, excluded, rng)
    return dict(data, train_pos=edges[:, train], val_pos=edges[:, val], test_pos=edges[:, test],
                test_neg=neg[:, :n_test], val_neg=neg[:, n_test:])


def variant_suffix(clock_training, rollout, soft_edges, encoder, resplit, full_vocab=False, directional_only=False):
    return "".join(suffix for enabled, suffix in (
        (clock_training == "tied", "_tied"), (not rollout, "_no_rollout"),
        (not soft_edges, "_hard"), (bool(encoder), "_pretrained"),
        (resplit, "_resplit"), (full_vocab, "_full_vocab"), (directional_only, "_directional")) if enabled)


def run(name, seed, epochs, device, width, vocab_size, clock_training="mixed", rollout=True, soft_edges=True,
        encoder=None, length=16, resplit=False, full_vocab=False, max_token_targets=512, directional_only=False):
    if name == "arxiv":
        raise ValueError("The cached arxiv split is undirected. A directed temporal split is required for ogbn-arxiv.")
    set_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rng = torch.Generator(device=device).manual_seed(seed)
    data = synthetic_graph(name) if name in SYNTHETIC else load(name)
    if resplit:
        data = resplit_link_data(data, seed)
    if data["tokens"].shape[1] <= length and name not in SYNTHETIC:
        if not encoder:
            raise ValueError("Longer text requires --encoder pointing to the local tokenizer")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(encoder, local_files_only=True)
        tokenized = tokenizer(data["texts"], max_length=length+1, truncation=True,
                              padding="max_length", return_tensors="pt")["input_ids"]
    else:
        tokenized = data["tokens"]
    raw = tokenized[:, 1:length+1].long().to(device)
    valid = (raw > 2) & (raw != 50264)
    pairs = data["train_pos"].to(device)
    forbid = set()
    for key in ("train_pos", "val_pos", "test_pos", "cold_pos", "cold_neg", "val_neg", "test_neg"):
        a, b = data[key]
        forbid.update((a*len(raw)+b).tolist())
    noise_edges = data.get("noise_edges", pairs[:, :0]).to(device)
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
    vocabulary = (torch.arange(3, 50264, device=device) if full_vocab else
                  counts.topk(min(vocab_size, int((counts > 0).sum()))).indices)
    mapping = torch.full((50265,), OTHER, device=device, dtype=torch.long)
    mapping[vocabulary] = torch.arange(3, len(vocabulary)+3, device=device)
    ids = mapping[raw].masked_fill(~valid, PAD)
    eligible = ids > OTHER
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
        unscored = 0
        if full_vocab and int(masked.sum()) > max_token_targets:
            masked_positions = masked.nonzero()
            keep = torch.randperm(len(masked_positions), generator=rng, device=device)[:max_token_targets]
            selected = masked_positions[keep]
            panel = torch.zeros_like(masked)
            panel[selected[:, 0], selected[:, 1]] = True
            backdrop[masked & ~panel] = MASK
            unscored = len(masked_positions)-len(selected)
            masked = panel
        return Case(backdrop, ids, masked, candidates, labels,
                    corruption(torch.ones_like(labels).bool(), edge_rate, rng), base,
                    (text_rate, edge_rate), soft_edges, unscored)

    model = SparseTwinMask(len(vocabulary)+3, ids.shape[1], width, encoder, vocabulary).to(device)
    if encoder and device.type == "cuda" and torch.cuda.is_bf16_supported():
        model.encoder.bfloat16()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.003)
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
        if rollout and dropout is None and epoch % 2 == 0:
            with torch.no_grad():
                generated, _ = advance(case, state, (lx.detach(), la.detach()), "joint", 2)
            pending_x = generated[0][case.x_index] == MASK
            pending_a = generated[1][case.a_index] == UNKNOWN
            rx, ra = case.predict(model, generated, score=(pending_x, pending_a))
            loss = (loss + token_loss(rx, case.x_truth[pending_x]) + edge_loss(ra, case.a_truth[pending_a])) / 2
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
                        for x, a in rates*(1 if directional_only else 3)]
    controller = fixed = rho = negative_gain_accuracy = None
    if not directional_only:
        dev_cases = [make_case(dev_nodes, pairs[:, dev_edges], neg_dev, richer_base, x, a) for x, a in rates]
        x, gain = rollout_values(model, controller_cases, budget)
        controller = Ridge(alpha=.01).fit(x, gain)
        dx, dgain = rollout_values(model, dev_cases, budget)
        predicted = controller.predict(dx)
        rho = spearmanr(predicted.ravel(), dgain.ravel()).statistic
        negative_gain_accuracy = float(((predicted < 0) == (dgain < 0)).mean())

        def fixed_score(policy):
            return np.mean([row["token_accuracy"]+row["edge_accuracy"] for row in
                            (reconstruct(model, case, policy, budget, controller) for case in dev_cases)])

        fixed = max(ACTIONS, key=fixed_score)
    gate_rows = [(features_, oracle_weights(case, solo, joint))
                 for case in controller_cases for features_, solo, joint in [gate_inputs(model, case)]]
    gate_x, gate_y = zip(*gate_rows)
    gate = make_pipeline(StandardScaler(), Ridge(alpha=1)).fit(gate_x, gate_y)
    controller_seconds = tick(device)-start
    base = torch.cat((pairs, noise_edges), 1)
    # Validation positives stay outside the message graph, including at test time.
    positives, negatives = (data[key].to(device) for key in ("test_pos", "test_neg"))
    rows = []
    masks = []
    for tx, ta in rates:
        case = make_case(test_nodes, positives, negatives, base, tx, ta)
        initial = case.initial()
        masks.append(dict(tokens=case.mask.cpu(), pairs=case.pairs.cpu(), edge_mask=case.edge_mask.cpu()))
        with torch.no_grad():
            gate_features, (without_a, without_x), (lx, la) = gate_inputs(model, case)
            ce, bce = token_loss(lx, case.x_truth), edge_loss(la, case.a_truth)
            row = dict(text_rate=tx, edge_rate=ta, initial_clocks=case.clocks(initial),
                       tokens=len(case.x_truth), pairs=len(case.a_truth),
                       initial_token_ce=float(ce), initial_edge_bce=float(bce),
                       structure_to_text=float(token_loss(without_a, case.x_truth)-ce),
                       text_to_edge=float(edge_loss(without_x, case.a_truth)-bce), metrics={})
            solo, joint = (without_a, without_x), (lx, la)
            row["gates"] = {
                key: gate_results(case, solo, joint, weight)
                for key, weight in {
                    "unimodal": (0, 0), "joint": (1, 1),
                    "learned": gate.predict([gate_features])[0].clip(0, 1),
                    "oracle_analysis_only": oracle_weights(case, solo, joint),
                }.items()
            }
            policies = ("one_shot",) if directional_only else ("one_shot", *ACTIONS, "entropy", "adaptive")
            for policy in policies:
                start = tick(device)
                result = reconstruct(model, case, policy, 1 if policy == "one_shot" else budget, controller)
                result["seconds"] = tick(device)-start
                row["metrics"][policy] = result
        rows.append(row)
    suffix = variant_suffix(clock_training, rollout, soft_edges, encoder, resplit, full_vocab, directional_only)
    directory = Path("checkpoints/research_twinmask_v2" + suffix)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}_{seed}.pt"
    torch.save(dict(model={k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")},
                    vocabulary=vocabulary.cpu(), masks=masks, node_split=order.cpu(),
                    edge_split=perm.cpu(), controller_coef=controller.coef_ if controller is not None else None,
                    controller_intercept=controller.intercept_ if controller is not None else None,
                    gate_mean=gate[0].mean_, gate_scale=gate[0].scale_, gate_coef=gate[1].coef_,
                    gate_intercept=gate[1].intercept_), path)
    return dict(seed=seed, epochs=epochs, width=width, vocab=len(vocabulary), best_fixed=fixed,
                rollout_training=rollout, soft_edges=soft_edges, clock_training=clock_training,
                encoder=str(encoder) if encoder else None, text_length=ids.shape[1], resplit_links=resplit,
                full_vocab=full_vocab, max_token_targets=max_token_targets if full_vocab else None,
                test_vocab_coverage=float(eligible[test_nodes].sum()/valid[test_nodes].sum()),
                dev_gain_spearman=float(rho) if rho is not None and np.isfinite(rho) else None,
                dev_negative_gain_accuracy=negative_gain_accuracy, directional_only=directional_only,
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
    parser.add_argument("--no-rollout", action="store_true")
    parser.add_argument("--hard-edges", action="store_true")
    parser.add_argument("--encoder", help="local pretrained DistilRoBERTa model and tokenizer")
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--resplit-links", action="store_true")
    parser.add_argument("--full-vocab", action="store_true", help="decode all RoBERTa lexical IDs on a random target panel")
    parser.add_argument("--max-token-targets", type=int, default=512, help="per-case query cap with --full-vocab")
    parser.add_argument("--directional-only", action="store_true", help="skip iterative evaluation and action-value fitting")
    args = parser.parse_args()
    if args.max_token_targets < 1:
        parser.error("--max-token-targets must be positive")
    for name in args.datasets:
        rows = []
        for seed in args.seeds:
            rows.append(run(name, seed, args.epochs, torch.device(args.device), args.width, args.vocab,
                            args.clock_training, not args.no_rollout, not args.hard_edges, args.encoder, args.length,
                            args.resplit_links, args.full_vocab, args.max_token_targets, args.directional_only))
            suffix = variant_suffix(args.clock_training, not args.no_rollout, not args.hard_edges, args.encoder,
                                    args.resplit_links, args.full_vocab, args.directional_only)
            save_json(rows, "research_twinmask_v2", f"{name}{suffix}.json")
            print(name, seed, rows[-1]["best_fixed"], rows[-1]["dev_gain_spearman"], flush=True)


if __name__ == "__main__":
    main()
