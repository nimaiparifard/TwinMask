"""TwinMask V5: isolated encoders, independent teachers, and locked transfer evaluation."""
import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, SAGEConv

from scripts.research_ensemble import tick
from scripts.research_twinmask import Case, MASK, OTHER, PAD, UNKNOWN, corruption
from scripts.research_twinmask_v3 import summarize
from twinmask.audit_protocol import graph_split, induced_raw_split, keys, verify
from twinmask.common import save_json, set_seed
from twinmask.data import sample_negatives
from twinmask.set_utility import SetUtility, choose, enumerate_subsets, subset_objective


# The run grid and endpoints are fixed before evaluation. Each split trains fresh parameters.
SPLIT_SEEDS = (2101, 2113)
MODEL_SEEDS = (71, 83, 97)
RATES = ((.25, .75), (.5, .5), (.75, .25), (.9, .9))
ARCHITECTURES = ("mlm", "structure", "text_link", "gae", "sage", "gat", "twinmask")


def expected_utility(log_probs, teacher, full):
    """Compute expected log-score gain in either a binary or categorical outcome space."""
    return ((log_probs-log_probs[full:full+1])*teacher).sum(-1)


def as_log_probs(logits, direction):
    if direction == "text":
        return logits.log_softmax(-1)
    return torch.stack((F.logsigmoid(-logits), F.logsigmoid(logits)), -1)


class IsolatedTwin(nn.Module):
    def __init__(self, vocab, width=48, architecture="sage", *, depth=2, length=16,
                 dropout=0., fusion=1., structural_stats=False, text_encoder="cnn"):
        super().__init__()
        self.architecture = architecture
        self.vocab = vocab
        self.emb = nn.Embedding(vocab, width)
        self.position = nn.Parameter(torch.randn(length, width)*.02)
        self.local = (nn.Conv1d(width, width, 3, padding=1) if text_encoder == "cnn" else
                      nn.TransformerEncoderLayer(width, 4, width*2, dropout, batch_first=True))
        self.structure = nn.Linear(6 if structural_stats else 3, width)
        layer = GATConv if architecture in ("gat", "structure_gat") else SAGEConv
        self.layers = nn.ModuleList(layer(width, width) for _ in range(depth))
        self.dropout, self.fusion, self.structural_stats = dropout, fusion, structural_stats
        self.text_encoder = text_encoder
        self.directions = (1., 1.)
        self.detach_cross = False
        self.register_parameter("gate_logits", None)
        self.t_fuse = nn.Linear(width, width, bias=False)
        self.e_fuse = nn.Linear(width, width, bias=False)
        self.token = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, vocab))
        self.link = nn.Sequential(nn.Linear(4*width, width), nn.GELU(), nn.Linear(width, 1))
        self.decode_gae = nn.Linear(width, width, bias=False)

    def encode_text(self, ids):
        local = self.emb(ids) + self.position[:ids.shape[1]]
        if self.text_encoder == "cnn":
            local = local + F.gelu(self.local(local.transpose(1, 2)).transpose(1, 2))
        else:
            padding = ids == PAD
            padding[:, 0] &= ~padding.all(1)
            local = self.local(local, src_key_padding_mask=padding)
        return F.dropout(local, self.dropout, self.training)

    def streams(self, ids, pairs, states, base, clocks):
        present = torch.cat((base, pairs[:, states == 1]), 1)
        adjacency = torch.cat((present, present.flip(0)), 1)
        degree = torch.bincount(adjacency[1], minlength=len(ids)).float().log1p()
        features = [degree, (degree == 0).float(), torch.full_like(degree, clocks[1])]
        if self.structural_stats:
            src, dst = adjacency
            count = torch.bincount(dst, minlength=len(ids)).clamp_min(1)
            one_hop = torch.zeros_like(degree).index_add_(0, dst, degree[src])/count
            two_hop = torch.zeros_like(degree).index_add_(0, dst, one_hop[src])/count
            variance = torch.zeros_like(degree).index_add_(0, dst, degree[src].square())/count-one_hop.square()
            features.extend((one_hop, two_hop, variance.clamp_min(0).sqrt()))
        structure = self.structure(torch.stack(features, 1))
        for layer in self.layers:
            structure = structure + F.gelu(layer(structure, adjacency))
        local = self.encode_text(ids)
        visible = (ids != PAD)[..., None]
        semantic = (local*visible).sum(1)/visible.sum(1).clamp_min(1)
        return local, semantic, structure, adjacency, degree

    def edge_scores(self, left, right, message):
        if self.architecture == "gae":
            a, b = self.decode_gae(left+message), self.decode_gae(right+message)
            return (a*b).sum(-1)/math.sqrt(a.shape[-1])
        return self.link(torch.cat((left+right, (left-right).abs(), left*right, self.e_fuse(message)), -1)).squeeze(-1)

    def predict(self, case, state=None, *, text_query=None, edge_query=None, clocks=None):
        state = case.initial() if state is None else state
        ids, edges, _ = state
        local, semantic, structure, adjacency, degree = self.streams(
            ids, case.pairs, edges, case.base, case.clocks(state) if clocks is None else clocks)
        tx = case.x_index if text_query is None else text_query
        ea = case.a_index if edge_query is None else edge_query
        row, col = tx
        a, b = case.pairs[:, ea]
        src, dst = adjacency
        neighbor = torch.zeros_like(structure).index_add_(0, dst, structure[src])
        count = torch.bincount(dst, minlength=len(ids)).clamp_min(1)
        text_joint = structure+neighbor/count[:, None]
        if self.detach_cross:
            text_joint = text_joint.detach()
        directions = self.directions if self.gate_logits is None else self.gate_logits.sigmoid()
        logits_text = (self.token(local[row, col]+self.fusion*directions[0]*self.t_fuse(text_joint[row])),
                       self.token(local[row, col]))
        neighbor_text = torch.zeros_like(semantic).index_add_(0, dst, semantic[src])/count[:, None]
        cross = (semantic[a]+semantic[b]+neighbor_text[a]+neighbor_text[b])/4
        if self.detach_cross:
            cross = cross.detach()
        cross = cross*self.fusion*directions[1]
        without = torch.zeros_like(cross)
        logits_edge = (self.edge_scores(structure[a], structure[b], cross),
                       self.edge_scores(structure[a], structure[b], without))
        if self.architecture == "text_link":
            logits_edge = (self.edge_scores(semantic[a], semantic[b], without),)*2
        if self.architecture in ("structure", "structure_gat", "gae"):
            logits_edge = (logits_edge[1],)*2
        if self.architecture == "mlm":
            logits_text = (logits_text[1],)*2
        return logits_text, logits_edge


def text_loss(logits, truth):
    return F.cross_entropy(logits, truth, reduction="none")


def edge_loss(logits, truth):
    return F.binary_cross_entropy_with_logits(logits, truth.float(), reduction="none")


def fit_backbones(make_case, ids, vocab, fit_nodes, rng, device, epochs, architectures=ARCHITECTURES, target_cap=None):
    models = {}
    for architecture in architectures:
        model = IsolatedTwin(vocab, architecture=architecture).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
        for step in range(epochs):
            rates = RATES[step % len(RATES)]
            case = make_case(0, rates, fit_nodes, rng)
            state = case.initial()
            if architecture == "twinmask" and step % 3 == 2:
                with torch.no_grad():
                    preliminary = model.predict(case, state)
                    state, _ = reveal(case, state, preliminary, case.mask, case.edge_mask, 2, "free", rng)
                clocks = clock_at(case, state)
                x_index = tuple(i[state[0][case.x_index] == MASK] for i in case.x_index)
                a_index = case.a_index[state[1][case.a_index] == UNKNOWN]
            else:
                clocks, x_index, a_index = case.rates, case.x_index, case.a_index
            if target_cap is not None:
                sample = torch.randperm(len(x_index[0]), generator=rng, device=device)[:target_cap]
                x_index = tuple(i[sample] for i in x_index)
            lx, la = model.predict(case, state, text_query=x_index, edge_query=a_index, clocks=clocks)
            mode = int(step % 5 == 0)
            losses = []
            if architecture in ("mlm", "sage", "gat", "twinmask"):
                losses.append(text_loss(lx[mode], ids[x_index]).mean())
            if architecture != "mlm":
                losses.append(edge_loss(la[mode], case.labels[a_index]).mean())
            optimizer.zero_grad()
            sum(losses).backward()
            optimizer.step()
        models[architecture] = model.eval()
    return models


def clock_at(case, state):
    remaining = (state[0][case.x_index] == MASK, state[1][case.a_index] == UNKNOWN)
    return tuple(rate*int(mask.sum())/max(1, mask.numel()) for rate, mask in zip(case.rates, remaining))


def reveal(case, state, predictions, text_feedback, edge_feedback, remaining, mode, rng):
    ids, edges, weights = (x.clone() for x in state)
    text_mask = text_feedback[case.x_index] & (ids[case.x_index] == MASK)
    edge_mask = edge_feedback[case.a_index] & (edges[case.a_index] == UNKNOWN)
    positions = []
    for mask in (text_mask, edge_mask):
        ix = mask.nonzero().flatten()
        ix = ix[torch.randperm(len(ix), device=ids.device, generator=rng)]
        positions.append(ix[:math.ceil(len(ix)/remaining)])
    x, a = positions
    nodes, tokens = case.x_index
    pred_x = predictions[0][0][x, 3:].argmax(-1)+3 if mode == "free" else case.x_truth[x]
    pred_a = (predictions[1][0][a] >= 0).long() if mode == "free" else case.a_truth[a]
    ids[nodes[x], tokens[x]] = pred_x
    edges[case.a_index[a]] = pred_a
    return (ids, edges, weights), (len(x), len(a))


@torch.no_grad()
def temporal(model, case, seed, steps=4):
    device = case.ids.device
    rng = torch.Generator(device=device).manual_seed(seed)
    x_audit = torch.randperm(len(case.x_truth), device=device, generator=rng)[::4]
    a_audit = torch.randperm(len(case.a_truth), device=device, generator=rng)[::4]
    x_score = tuple(i[x_audit] for i in case.x_index)
    a_score = case.a_index[a_audit]
    token_feedback, edge_feedback = case.mask.clone(), case.edge_mask.clone()
    token_feedback[x_score] = False
    edge_feedback[a_score] = False
    output = {}
    for mode in ("static", "free", "teacher"):
        matched = torch.Generator(device=device).manual_seed(seed+71)
        state = case.initial()
        rows = []
        reveal_count = [0, 0]
        total_feedback = [int(mask.sum()) for mask in (token_feedback, edge_feedback)]
        for step in range(steps):
            clocks = clock_at(case, state)
            x, a = model.predict(case, state, text_query=x_score, edge_query=a_score, clocks=clocks)
            tx, ta = case.x_truth[x_audit], case.a_truth[a_audit]
            rows.append(dict(clocks=clocks, text=float((text_loss(x[1], tx)-text_loss(x[0], tx)).mean()),
                             edge=float((edge_loss(a[1], ta)-edge_loss(a[0], ta)).mean()),
                             token_ce=float(text_loss(x[0], tx).mean()), edge_bce=float(edge_loss(a[0], ta).mean()),
                             token_accuracy=float((x[0][..., 3:].argmax(-1)+3 == tx).float().mean()),
                             edge_auc=summarize(edge_loss(a[0], ta), a[0].sigmoid(), ta, "edge")["auc"]
                             if ta.unique().numel() > 1 else None, counts=(len(tx), len(ta))))
            if step+1 == steps:
                continue
            if mode == "static":
                initial = case.initial()
                reveal_count = [done+math.ceil((total-done)/(steps-step))
                                for done, total in zip(reveal_count, total_feedback)]
                reveal_count_x, reveal_count_a = reveal_count
                static = torch.Generator(device=device).manual_seed(seed+423+step)
                x_order, a_order = token_feedback.nonzero(), edge_feedback.nonzero().flatten()
                xi = x_order[torch.randperm(len(x_order), device=device, generator=static)[:reveal_count_x]]
                ai = a_order[torch.randperm(len(a_order), device=device, generator=static)[:reveal_count_a]]
                initial[0][xi[:, 0], xi[:, 1]] = case.ids[xi[:, 0], xi[:, 1]]
                initial[1][ai] = case.labels[ai]
                state = initial
            else:
                prediction = model.predict(case, state, clocks=clocks) if mode == "free" else None
                state, _ = reveal(case, state, prediction, token_feedback, edge_feedback, steps-step, mode, matched)
            assert (state[0][x_score] == MASK).all() and (state[1][a_score] == UNKNOWN).all()
        output[mode] = rows
    return output


def prepare_cases(data, base, positives, negatives, split_seed, device, vocab_limit=1024, *, length=16,
                  fixed_vocab=None, train_fraction=.6, node_groups=None):
    raw = data["tokens"][:, 1:length+1].long().to(device)
    n = len(raw)
    seeded = torch.Generator().manual_seed(split_seed+17)
    order = torch.randperm(n, generator=seeded).to(device)
    nfit = int(n*train_fraction)
    fit, dev, test = order.split([nfit, n//5, n-nfit-n//5])
    if node_groups is not None:
        fit, dev, test = (block.to(device) for block in node_groups)
    valid = (raw > 2) & (raw < 50264)
    count = torch.bincount(raw[fit][valid[fit]], minlength=50265)
    vocab = (torch.arange(3, 50264, device=device) if vocab_limit is None else
             count.topk(min(vocab_limit, int((count > 0).sum()))).indices)
    if fixed_vocab is not None:
        vocab = fixed_vocab.to(device)
    lookup = torch.full((50265,), OTHER, dtype=torch.long, device=device)
    lookup[vocab] = torch.arange(3, len(vocab)+3, device=device)
    ids = lookup[raw].masked_fill(~valid, PAD)
    context = ids.masked_fill(ids > OTHER, MASK)
    context[fit] = ids[fit]

    def make(which, rates, nodes, rng):
        pairs = torch.cat((positives[which], negatives[which]), 1).to(device)
        labels = (torch.arange(pairs.shape[1], device=device) < positives[which].shape[1]).long()
        shuffle = torch.randperm(len(labels), device=device, generator=rng)
        backdrop, pool = context.clone(), torch.zeros_like(ids, dtype=torch.bool)
        backdrop[nodes] = ids[nodes]
        pool[nodes] = ids[nodes] > OTHER
        return Case(backdrop, ids, corruption(pool, rates[0], rng), pairs[:, shuffle], labels[shuffle],
                    corruption(torch.ones_like(labels).bool(), rates[1], rng), base.to(device), rates)

    make.vocabulary = vocab
    return make, ids, len(vocab)+3, (fit, dev, test)


@torch.no_grad()
@torch.no_grad()
def fusion_panel(model, case, direction, target_index):
    ids, states, _ = case.initial()
    local, semantic, structure, adjacency, degree = model.streams(ids, case.pairs, states, case.base, case.rates)
    source, destination = adjacency
    n = len(ids)
    count = torch.bincount(destination, minlength=n).clamp_min(1)
    if direction == "text":
        row, col = (i[target_index] for i in case.x_index)
        group = destination*3+source.remainder(3)
        bins = structure.new_zeros(n*3, structure.shape[-1]).index_add_(0, group, structure[source])
        incoming = bins.reshape(n, 3, -1)/count[:, None, None]
        flags = torch.bincount(group, minlength=n*3).reshape(n, 3) > 0
        message = torch.cat((structure[:, None], incoming), 1)[row]
        valid = torch.cat((torch.ones_like(flags[:, :1]), flags), 1)[row]
        ground = case.x_truth[target_index]
        scale = model.directions[0] if model.gate_logits is None else model.gate_logits[0].sigmoid()
        dec = lambda fused: model.token(local[row, col] + model.fusion*scale*model.t_fuse(fused))
        context = local[row, col]
        target_degree = degree[row]
    else:
        a, b = case.pairs[:, case.a_index[target_index]]
        incoming = torch.zeros_like(semantic).index_add_(0, destination, semantic[source])/count[:, None]
        message = torch.stack((semantic[a], semantic[b], incoming[a], incoming[b]), 1)/4
        valid = torch.stack((torch.ones_like(a).bool(), torch.ones_like(b).bool(), degree[a] > 0, degree[b] > 0), 1)
        ground = case.a_truth[target_index]
        scale = model.directions[1] if model.gate_logits is None else model.gate_logits[1].sigmoid()
        dec = lambda fused: model.edge_scores(structure[a].expand_as(fused), structure[b].expand_as(fused),
                                             model.fusion*scale*fused)
        context = (structure[a]+structure[b])/2
        target_degree = (degree[a]+degree[b])/2
    masks, allowed = enumerate_subsets(valid)
    inputs = torch.cat((message, context[:, None].expand_as(message)), -1)
    return message, valid, masks, allowed, ground, target_degree, dec, inputs


@torch.no_grad()
def subset_predictions(model, case, direction, target_index):
    message, _, masks, allowed, ground, target_degree, dec, _ = fusion_panel(
        model, case, direction, target_index)
    fused = torch.einsum("sni,nih->snh", masks, message)
    logits = dec(fused)
    return as_log_probs(logits, direction), masks, allowed, ground, target_degree


@torch.no_grad()
def teachers(models, case, direction, ix):
    if direction == "text":
        query = tuple(k[ix] for k in case.x_index)
        return {"mlm": models["mlm"].predict(case, text_query=query, edge_query=case.a_index[:0])[0][0]}
    query = case.a_index[ix]
    return {k: models[k].predict(case, edge_query=query, text_query=tuple(i[:0] for i in case.x_index))[1][0]
            for k in ("text_link", "structure", "gae")}


def probability_metrics(probability, labels, direction):
    loss = -probability.clamp_min(1e-9).log().gather(-1, labels[..., None]).squeeze(-1)
    prediction = probability[..., 3:].argmax(-1)+3 if direction == "text" else probability[..., 1]
    if direction == "edge" and labels.unique().numel() < 2:
        return dict(loss=float(loss.mean()), auc=None, ap=None)
    return summarize(loss, prediction, labels, direction)


def prepare_evaluation(model, models, cases, direction, seed, limit):
    dev = []
    for j, case in enumerate(cases):
        rng = torch.Generator(device=case.ids.device).manual_seed(seed+j)
        targets = case.x_truth if direction == "text" else case.a_truth
        ix = torch.randperm(len(targets), generator=rng, device=case.ids.device)[:limit]
        logp, mask, allowed, truth, degree = subset_predictions(model, case, direction, ix)
        dev.append((logp, mask, allowed, truth, teachers(models, case, direction, ix), degree))
    return dev


def calibrate_teachers(dev, direction):
    temperatures = {}
    for name in dev[0][4]:
        logits = torch.cat([row[4][name] for row in dev])
        truth = torch.cat([row[3] for row in dev])
        def loss(temp):
            p = as_log_probs(logits/temp, direction)
            return float(F.nll_loss(p, truth))
        temperatures[name] = min((.5, .75, 1., 1.25, 1.5, 2., 3.), key=loss)
    mixtures = (("mlm",),) if direction == "text" else (
        ("text_link",), ("structure",), ("gae",), ("text_link", "structure", "gae"), ("text_link", "structure"))
    def loss(names):
        p = torch.cat([teacher_prob(row[4], names, temperatures, direction) for row in dev])
        return float(F.nll_loss(p.clamp_min(1e-9).log(), torch.cat([row[3] for row in dev])))
    return temperatures, min(mixtures, key=loss), mixtures


def teacher_prob(logits, names, temperatures, direction):
    return torch.stack([as_log_probs(logits[name]/temperatures[name], direction).exp() for name in names]).mean(0)


def mixture_feasible(masks, allowed):
    all_kept = masks.sum(-1) == masks.sum(-1).amax(0)
    return allowed & ((masks.sum(-1) <= 2) | all_kept)


def fit_stacking(dev, steps=70, regularization=0.):
    # Shared nonnegative weights, fitted on the development targets only.
    weights = nn.Parameter(dev[0][0].new_zeros(dev[0][0].shape[0]))
    optimizer = torch.optim.Adam([weights], lr=.06)
    for _ in range(steps):
        loss = regularization*weights.square().mean()
        for logp, mask, allowed, truth, _, _ in dev:
            keep = mixture_feasible(mask, allowed)
            probability = weights[:, None].expand_as(keep).masked_fill(~keep, -torch.inf).softmax(0)
            correct = logp.gather(-1, truth[None, :, None].expand(logp.shape[0], -1, -1)).squeeze(-1)
            loss = loss-(probability*correct.exp()).sum(0).clamp_min(1e-9).log().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return weights.detach()


@torch.no_grad()
def evaluate_expected(rows, direction, temperatures, chosen, alternatives, weights, seed, mixture_temperature=.15):
    rng = torch.Generator(device=rows[0][0].device).manual_seed(seed)
    out = []
    for logp, mask, allowed, truth, logits, degree in rows:
        full = len(logp)-1
        ix = torch.arange(len(truth), device=truth.device)
        independent = teacher_prob(logits, chosen, temperatures, direction)
        gain = expected_utility(logp, independent, full)
        random = torch.rand(gain.shape, generator=rng, device=gain.device)
        oracle = (logp-logp[-1:]).gather(-1, truth[None, :, None].expand(len(logp), -1, -1)).squeeze(-1)
        selected = choose(gain, mask, allowed, 2)
        exact = choose(gain, mask, allowed, 2, exact=True)
        count = mask[selected, ix].sum(-1)
        matched = allowed & (mask.sum(-1) == count)
        picks = dict(expected_at_most=selected, expected_exact=exact,
                     random_matched=random.masked_fill(~matched, -torch.inf).argmax(0),
                     oracle_matched_analysis_only=oracle.masked_fill(~matched, -torch.inf).argmax(0))
        p = logp.exp()
        keep = mixture_feasible(mask, allowed)
        soft = (gain/mixture_temperature).masked_fill(~keep, -torch.inf).softmax(0)
        stack = weights[:, None].expand_as(gain).masked_fill(~keep, -torch.inf).softmax(0)
        estimates = dict(full=p[full], without=p[0], teacher=independent,
                         uniform=(p*keep[..., None]).sum(0)/keep.sum(0)[:, None],
                         utility_mixture=(p*soft[..., None]).sum(0),
                         stacking=(p*stack[..., None]).sum(0))
        estimates.update({key: p[pick, ix] for key, pick in picks.items()})
        estimates.update({"teacher_"+"_".join(group): teacher_prob(logits, group, temperatures, direction)
                          for group in alternatives})
        row = {key: probability_metrics(prob, truth, direction) for key, prob in estimates.items()}
        row["messages"] = {key: float(mask[pick, ix].sum(-1).mean()) for key, pick in picks.items()}
        local_gain = oracle[0]
        bins = torch.quantile(degree, degree.new_tensor([1/3, 2/3]))
        degree_bins = torch.bucketize(degree.contiguous(), bins)
        row["degree_conditioning_gain"] = [float((-local_gain[degree_bins == b]).mean())
                                            if (degree_bins == b).any() else None for b in range(3)]
        out.append(row)
    return out


@torch.no_grad()
def cross_architecture(models, cases, limit, seed):
    result = {}
    for case_num, case in enumerate(cases):
        device = case.ids.device
        rng = torch.Generator(device=device).manual_seed(seed+case_num)
        x = torch.randperm(len(case.x_truth), device=device, generator=rng)[:limit]
        a = torch.randperm(len(case.a_truth), device=device, generator=rng)[:limit]
        for key, model in models.items():
            tx, ta = case.x_truth[x], case.a_truth[a]
            logits = model.predict(case, text_query=tuple(i[x] for i in case.x_index), edge_query=case.a_index[a])
            scores = {}
            for d, pair, truth in zip(("text", "edge"), logits, (tx, ta)):
                if d == "text" and key in ("structure", "text_link", "gae") or d == "edge" and key == "mlm":
                    continue
                full, removed = [probability_metrics(as_log_probs(v, d).exp(), truth, d) for v in pair]
                fn = text_loss if d == "text" else edge_loss
                gains = fn(pair[1], truth)-fn(pair[0], truth)
                noise = float((gains < -.001).float().mean())
                scores[d] = dict(full=full, removed=removed, conditioning_gain=float(gains.mean()),
                                 harmful_fraction=noise)
            result.setdefault(key, []).append(scores)
    for name in ("sage", "gat", "twinmask"):
        for j, scores in enumerate(result[name]):
            for d, arm in (("text", "mlm"), ("edge", "structure")):
                scores[d]["joint_training_gain"] = result[arm][j][d]["full"]["loss"]-scores[d]["full"]["loss"]
    return result


def selector_study(model, models, fit, test, d, temperatures, chosen, device, seed, limit):
    training = []
    for j, case in enumerate(fit):
        rng = torch.Generator(device=device).manual_seed(seed+j)
        length = len(case.x_truth if d == "text" else case.a_truth)
        ix = torch.randperm(length, generator=rng, device=device)[:limit]
        message, valid, masks, allowed, truth, _, decode, features = fusion_panel(model, case, d, ix)
        with torch.no_grad():
            logits = decode(torch.einsum("sni,nih->snh", masks, message))
            logp = as_log_probs(logits, d)
            independently = teacher_prob(teachers(models, case, d, ix), chosen, temperatures, d)
            target = expected_utility(logp, independently, len(logp)-1)
            observed_train = logp.gather(-1, truth[None, :, None].expand(len(logp), -1, -1)).squeeze(-1)
            training.append((features, masks, allowed, valid, target, observed_train-observed_train[-1:]))
    x, v = [torch.cat([r[i] for r in training]) for i in (0, 3)]
    masks, allowed, expected, labeled = [torch.cat([r[i] for r in training], 1) for i in (1, 2, 4, 5)]
    models_ = {key: SetUtility(x.shape[-1], anchored=True).to(device) for key in ("distilled", "direct")}
    loo = nn.Sequential(nn.Linear(x.shape[-1], 48), nn.GELU(), nn.Linear(48, 1)).to(device)
    opt = torch.optim.AdamW([*loo.parameters(), *(p for model_ in models_.values() for p in model_.parameters())], lr=.003)
    rng = torch.Generator(device=device).manual_seed(seed+613)
    expected = expected/expected[allowed].std(unbiased=False).clamp_min(1e-6)
    labeled = labeled/labeled[allowed].std(unbiased=False).clamp_min(1e-6)
    for _ in range(70):
        ix = torch.randperm(len(x), generator=rng, device=device)[:128]
        loss = 0.
        for key, net in models_.items():
            score, spread = net(x[ix], masks[:, ix], v[ix])
            values = expected if key == "distilled" else labeled
            loss = loss+subset_objective(score, spread, values[:, ix], masks[:, ix], allowed[:, ix],
                                         decision_weight=.4 if key == "direct" else 0.)
        estimate = loo(x[ix]).squeeze(-1)
        effects = -labeled[[-2, -3, -5, -9]][:, ix].T
        valid = v[ix]
        loss = loss+F.smooth_l1_loss(estimate[valid], effects[valid])
        opt.zero_grad()
        loss.backward()
        opt.step()
    output = []
    with torch.no_grad():
        for j, case in enumerate(test):
            rng = torch.Generator(device=device).manual_seed(seed+j)
            n = len(case.x_truth if d == "text" else case.a_truth)
            ix = torch.randperm(n, generator=rng, device=device)[:limit]
            message, valid, masks, allowed, truth, _, decode, features = fusion_panel(model, case, d, ix)
            node = torch.arange(len(ix), device=device)
            scores = {key: net(features, masks, valid)[0] for key, net in models_.items()}
            scores["loo"] = (masks*loo(features).squeeze(-1)).sum(-1)
            logp = as_log_probs(decode(torch.einsum("sni,nih->snh", masks, message)), d)
            q = teacher_prob(teachers(models, case, d, ix), chosen, temperatures, d)
            scores["expected"] = expected_utility(logp, q, len(logp)-1)
            pick = {key: choose(score, masks, allowed, 2, exact=True) for key, score in scores.items()}
            row = {}
            for key, indices in pick.items():
                # Student deployment only needs the selected decoder and no prediction probes.
                selected = (masks[indices, node, :, None]*message).sum(1)
                probabilities = as_log_probs(decode(selected), d).exp()
                row[key] = probability_metrics(probabilities, truth, d)
            row["distilled_expected_agreement"] = float((pick["distilled"] == pick["expected"]).float().mean())
            if j == 1:
                timing = {}
                for key in ("distilled", "expected"):
                    seconds = []
                    for _ in range(4):
                        start = tick(device)
                        msg, available, candidates, eligible, _, _, decoder, f = fusion_panel(model, case, d, ix)
                        if key == "distilled":
                            val = models_[key](f, candidates, available)[0]
                        else:
                            prob = as_log_probs(decoder(torch.einsum("sni,nih->snh", candidates, msg)), d)
                            q = teacher_prob(teachers(models, case, d, ix), chosen, temperatures, d)
                            val = expected_utility(prob, q, len(prob)-1)
                        selected = choose(val, candidates, eligible, 2, exact=True)
                        decoder((candidates[selected, node, :, None]*msg).sum(1))
                        seconds.append(tick(device)-start)
                    timing[key] = 1000*float(np.median(seconds[1:]))
                row["timing_ms"] = timing
            output.append(row)
    return output


def run(name, split_seed, model_seed, device, epochs, limit, protocol=None, compare_selectors=False):
    set_seed(model_seed)
    started = tick(device)
    data, base, positives, negatives, audit = graph_split(name, split_seed) if protocol is None else protocol
    make, ids, vocab, nodes = prepare_cases(data, base, positives, negatives, split_seed, device)
    rng = torch.Generator(device=device).manual_seed(model_seed+split_seed)
    models = fit_backbones(make, ids, vocab, nodes[0], rng, device, epochs)
    split_rng = torch.Generator(device=device).manual_seed(split_seed+910)
    dev = [make(1, rates, nodes[1], split_rng) for rates in RATES]
    test = [make(2, rates, nodes[2], split_rng) for rates in RATES]
    model = models["twinmask"]
    rows = {}
    for d in ("text", "edge"):
        development = prepare_evaluation(model, models, dev, d, split_seed+541, limit)
        temperatures, chosen, alternatives = calibrate_teachers(development, d)
        stacking = fit_stacking(development)
        actual = prepare_evaluation(model, models, test, d, split_seed+541, limit)
        rows[d] = dict(temperatures=temperatures, teacher=chosen,
                       cases=evaluate_expected(actual, d, temperatures, chosen, alternatives, stacking, split_seed+99))
        if compare_selectors:
            fit_rng = torch.Generator(device=device).manual_seed(split_seed+377)
            fit = [make(0, rates, nodes[0], fit_rng) for rates in RATES]
            rows[d]["selector_controls"] = selector_study(model, models, fit, test, d, temperatures, chosen,
                                                         device, split_seed+541, limit)
    clocks = {str(horizon): temporal(model, test[1], split_seed+91, horizon) for horizon in (1, 2, 4, 8)}
    missingness = {}
    for target in (0, 1):
        key = ("text", "edge")[target]
        cases = []
        for rate in (.1, .3, .5, .7, .9):
            rates = [.5, .5]
            rates[target] = rate
            same = torch.Generator(device=device).manual_seed(split_seed+200)
            cases.append(make(2, rates, nodes[2], same))
        missingness[key] = [x for x in cross_architecture({"mlm": models["mlm"],
                            "structure": models["structure"], "sage": models["sage"], "gat": models["gat"],
                            "twinmask": model}, cases, limit, split_seed+541)["twinmask"]]
    return dict(dataset=name, split_seed=split_seed, model_seed=model_seed, epochs=epochs, vocabulary=vocab-3,
                audit=audit, seconds=tick(device)-started, architectures=cross_architecture(models, test, limit, split_seed+541),
                temporal=clocks, missingness=missingness, expected=rows)


def synthetic_protocol(seed, homophily, text_noise, edge_noise, vocabulary):
    n = 480
    rng = [np.random.default_rng(seed+i*11) for i in range(7)]
    cls = rng[0].integers(4, size=n)
    a, b = np.triu_indices(n, 1)
    probability = 12/n*np.where(cls[a] == cls[b], 4*homophily, 4*(1-homophily)/3)
    visible = rng[1].random(len(a)) < probability
    edges = torch.from_numpy(np.stack((a[visible], b[visible])))
    reference = keys(edges, n)
    count = edges.shape[1]//8
    ix = rng[2].permutation(edges.shape[1])
    positives = [edges[:, i] for i in np.split(ix, [count, 2*count, 3*count])[:3]]
    base = edges[:, ix[3*count:]]
    all_neg = sample_negatives(n, 3*count, reference, rng[3])
    negatives = all_neg.split(count, dim=1)
    noise = sample_negatives(n, int(base.shape[1]*edge_noise), reference | keys(all_neg, n), rng[4])
    verify(positives, negatives, reference, torch.arange(n), n)
    size = (50260 if vocabulary == "full" else vocabulary)//4
    alternate = rng[5].integers(4, size=(n, 16))
    noise_mask = rng[6].random((n, 16)) < text_noise
    classes = np.where(noise_mask, alternate, cls[:, None])
    tokens = torch.ones(n, 17, dtype=torch.long)
    tokens[:, 1:] = torch.from_numpy(classes*size + (rng[5].random((n, 16))*size).astype(np.int64) + 3)
    return {"tokens": tokens}, base, positives, negatives, noise


def mechanism_study(device):
    controls = (.7, .1, 0., 128)
    grids = {"homophily": (.1, .3, .5, .7, .9), "text_noise": (0., .1, .25, .5),
             "edge_noise": (0., .1, .25, .5), "vocabulary": (32, 128, 512, 1024, "full")}
    rows = []
    for index, (factor, levels) in enumerate(grids.items()):
        for level in levels:
            config = list(controls)
            config[index] = level
            for seed in (3101, 3103, 3109):
                set_seed(91)
                data, base, positives, negatives, noise = synthetic_protocol(seed, *config)
                make, ids, vocab, nodes = prepare_cases(data, base, positives, negatives, seed, device,
                                                       None if config[3] == "full" else config[3])
                rng = torch.Generator(device=device).manual_seed(seed+91)
                models = fit_backbones(make, ids, vocab, nodes[0], rng, device, 18, ("sage",), target_cap=96)
                test_case = make(2, (.5, .5), nodes[2], rng)
                test_case.base = torch.cat((test_case.base, noise.to(device)), 1)
                model = models["sage"]
                with torch.no_grad():
                    i = torch.randperm(len(test_case.x_truth), device=device, generator=rng)[:96]
                    j = torch.randperm(len(test_case.a_truth), device=device, generator=rng)[:96]
                    logits = model.predict(test_case, text_query=tuple(v[i] for v in test_case.x_index),
                                           edge_query=test_case.a_index[j])
                    row = dict(factor=factor, level=level, graph_seed=seed, vocab=vocab-3)
                    for d, pair, fn, gold in zip(("text", "edge"), logits, (text_loss, edge_loss),
                                                  (test_case.x_truth[i], test_case.a_truth[j])):
                        full, removed = (fn(k, gold) for k in pair)
                        gain = removed-full
                        row[d] = dict(full=float(full.mean()), conditioning_gain=float(gain.mean()),
                                      harmful_fraction=float((gain < -.001).float().mean()))
                rows.append(row)
            print("mechanism", factor, level, flush=True)
            save_json(rows, "research_twinmask_v5", "mechanisms.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer", "pubmed", "computer"])
    parser.add_argument("--split-seeds", nargs="+", type=int, default=SPLIT_SEEDS)
    parser.add_argument("--model-seeds", nargs="+", type=int, default=MODEL_SEEDS)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--limit", type=int, default=320)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mechanism-study", action="store_true")
    parser.add_argument("--selector-controls", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.mechanism_study:
        mechanism_study(device)
        return
    for name in args.datasets:
        rows = []
        for split in args.split_seeds:
            shared = induced_raw_split(name, split) if name == "computer" else graph_split(name, split)
            for seed in args.model_seeds:
                row = run(name, split, seed, device, args.epochs, args.limit, shared, args.selector_controls)
                if args.selector_controls:
                    row = {key: row[key] for key in ("dataset", "split_seed", "model_seed", "epochs", "vocabulary",
                                                     "audit", "seconds", "expected")}
                rows.append(row)
                print(name, split, seed, round(row["seconds"], 2), flush=True)
        suffix = "_selectors" if args.selector_controls else ""
        save_json(rows, "research_twinmask_v5", f"{name}{suffix}.json")


if __name__ == "__main__":
    main()
