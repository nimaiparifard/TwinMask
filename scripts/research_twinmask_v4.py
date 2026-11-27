"""Paired TwinMask V4 utility, decision, budget, and risk experiments."""
import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.nn import functional as F

from scripts.research_ensemble import tick
from scripts.research_twinmask_v3 import MessagePanel, contexts, score_network, summarize
from scripts.research_twinmask import Case, OTHER, PAD, SYNTHETIC, corruption, synthetic_graph
from twinmask.common import save_json, set_seed
from twinmask.set_utility import SetUtility, choose, enumerate_subsets, shapley_values, subset_objective


def target_features(panel):
    n, k = panel.valid.shape
    if panel.direction == "text":
        position = F.one_hot(panel.position, panel.model.position.shape[0]).float()
        detail = torch.cat((panel.selected, panel.model.position[panel.position], position), -1)
        detail = detail[:, None].expand(-1, k, -1)
    else:
        opposite = panel.h.flip(1).repeat_interleave(k//2, dim=1)
        detail = torch.cat((opposite, (panel.receiver_h-opposite).abs(), panel.receiver_h*opposite), -1)
    local_average = (panel.total / panel.mass[..., None].clamp_min(1)).repeat_interleave(k//panel.h.shape[1], 1)
    disagreement = 1-F.cosine_similarity(panel.sender_h, local_average, dim=-1)
    return torch.cat((panel.features, panel.sender_h, panel.receiver_h, panel.messages, detail,
                      disagreement[..., None]), -1).reshape(n, k, -1)


def prediction_features(panel):
    k = panel.valid.shape[1]
    weights = torch.cat((torch.ones_like(panel.valid)[None],
                         ~torch.eye(k, dtype=torch.bool, device=panel.valid.device)[:, None].expand(-1, len(panel.truth), -1)))
    result = []
    for start in range(0, len(panel.truth), 128):
        span = slice(start, start+128)
        logits = panel.predict(weights[:, span].float(), span)
        probability = (logits.softmax(-1) if panel.direction == "text" else
                       torch.stack(((-logits).sigmoid(), logits.sigmoid()), -1))
        full, withheld = probability[:1], probability[1:]
        mixture = (full+withheld)/2

        def entropy(p):
            return -(p*p.clamp_min(1e-9).log()).sum(-1)

        full_entropy, without_entropy = entropy(full), entropy(withheld)
        t = probability.topk(2, dim=-1)
        confidence = t.values[..., 0]
        margin = confidence-t.values[..., 1]
        shift = panel.messages[span].norm(dim=-1) / panel.mass[span].repeat_interleave(
            k//panel.mass.shape[-1], dim=-1).clamp_min(1)
        result.append(torch.stack((full_entropy.expand_as(without_entropy), without_entropy-full_entropy,
                                   entropy(mixture)-(full_entropy+without_entropy)/2,
                                   confidence[1:]-confidence[:1], margin[1:]-margin[:1],
                                   (t.indices[1:, :, 0] != t.indices[:1, :, 0]).float(), shift.T), -1).transpose(0, 1))
    return torch.cat(result)


def features_for(panel):
    target = target_features(panel)
    return {"base": panel.features, "target": target,
            "aware": torch.cat((target, prediction_features(panel)), -1)}


def bundle(panel):
    masks, allowed = enumerate_subsets(panel.valid)
    loss, predictions = panel.evaluate(masks)
    return panel, features_for(panel), masks, allowed, loss, predictions


def fit_messages(train, feature, target, steps, rng):
    x = torch.cat([row[1][feature] for row in train])
    valid = torch.cat([row[0].valid for row in train])
    width = valid.shape[1]
    y = torch.cat([((row[4][[-1-(1<<j) for j in range(width)]]-row[4][-1]).T
                    if target == "loo" else shapley_values(row[4], width)) for row in train])
    scale = y[valid].std(unbiased=False).clamp_min(1e-5)
    model = score_network(x)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    for _ in range(steps):
        ix = torch.randperm(len(x), generator=rng, device=x.device)[:256]
        values = model(x[ix]).squeeze(-1)
        truth, keep = y[ix]/scale, valid[ix]
        sign = (truth[:, :, None]-truth[:, None, :]).sign()
        pairs = keep[:, :, None] & keep[:, None, :] & (sign != 0)
        loss = F.smooth_l1_loss(values[keep], truth[keep])
        if pairs.any():
            loss += .15*F.softplus(-sign[pairs]*(values[:, :, None]-values[:, None, :])[pairs]).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, scale


def fit_sets(train, steps, rng, decision=False, *, attention=False, dropout=0.):
    valid = torch.cat([row[0].valid for row in train])
    features = torch.cat([row[1]["aware"] for row in train])
    masks = torch.cat([row[2] for row in train], dim=1)
    allowed = torch.cat([row[3] for row in train], dim=1)
    utility = torch.cat([row[4][-1:]-row[4] for row in train], dim=1)
    scale = utility[allowed].std(unbiased=False).clamp_min(1e-5)
    model = SetUtility(features.shape[-1], attention=attention, dropout=dropout).to(features.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    for _ in range(steps):
        ix = torch.randperm(len(features), generator=rng, device=features.device)[:128]
        mean, spread = model(features[ix], masks[:, ix], valid[ix])
        loss = subset_objective(mean, spread, utility[:, ix]/scale, masks[:, ix], allowed[:, ix],
                                decision_weight=.5 if decision else 0., uncertainty=decision)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, scale


class PredictiveMixture:
    def __init__(self, models, dropout=False):
        self.models, self.dropout = models, dropout

    def __call__(self, features, masks, valid):
        samples = [model(features, masks, valid, sample_dropout=self.dropout)
                   for model in self.models for _ in range(8 if self.dropout else 1)]
        means, spreads = map(torch.stack, zip(*samples))
        uncertainty = (spreads.square().mean(0)+means.var(0, unbiased=False)).sqrt()
        return means.mean(0), uncertainty


def fit_direct(train, steps, rng, attention=False):
    model = score_network(train[0][1]["aware"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    for step in range(steps):
        panel, features, *_ = train[step % len(train)]
        ix = torch.randperm(len(panel.truth), generator=rng, device=panel.truth.device)[:256]
        score = model(features["aware"][ix]).squeeze(-1)
        valid = panel.valid[ix]
        if attention:
            weight = score.masked_fill(~valid, -1e4).softmax(1)*valid
            weight = weight*valid.sum(1, keepdim=True)
        else:
            weight = score.sigmoid()*valid
        logits = panel.predict(weight, ix)
        loss = (F.cross_entropy(logits, panel.truth[ix]) if panel.direction == "text" else
                F.binary_cross_entropy_with_logits(logits, panel.truth[ix].float()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


class LocalGATScore(nn.Module):
    """Shared projection and additive sender/receiver attention on a fixed local panel."""

    def __init__(self, width):
        super().__init__()
        self.project = nn.Linear(width, 48, bias=False)
        self.source = nn.Linear(48, 1, bias=False)
        self.receiver = nn.Linear(48, 1, bias=False)

    def forward(self, panel, ix):
        return F.leaky_relu(self.source(self.project(panel.sender_h[ix])) +
                            self.receiver(self.project(panel.receiver_h[ix])), .2).squeeze(-1)


def fit_comparators(train, steps, rng):
    panel = train[0][0]
    gat = LocalGATScore(panel.sender_h.shape[-1]).to(panel.truth.device)
    sparse = score_network(train[0][1]["aware"])
    sgul = SetUtility(train[0][1]["aware"].shape[-1]).to(panel.truth.device)
    optimizer = torch.optim.AdamW([*gat.parameters(), *sparse.parameters(), *sgul.parameters()], lr=.002)
    for step in range(steps):
        panel, features, subsets, allowed, _, predictions = train[step % len(train)]
        ix = torch.randperm(len(panel.truth), generator=rng, device=panel.truth.device)[:128]
        x, valid = features["aware"][ix], panel.valid[ix]
        attention = gat(panel, ix).masked_fill(~valid, -1e4).softmax(-1)*valid
        attention = attention*valid.sum(1, keepdim=True)
        priority = sparse(x).squeeze(-1)
        uniform = torch.rand(priority.shape, device=priority.device, generator=rng).clamp(1e-5, 1-1e-5)
        noisy = priority+uniform.log()-(-uniform).log1p()
        chosen = noisy.masked_fill(~valid, -torch.inf).topk(2, dim=-1).indices
        hard = torch.zeros_like(priority).scatter_(1, chosen, 1.)*valid
        soft = noisy.sigmoid()*valid
        sparse_weight = hard+(soft-soft.detach())
        targets = panel.truth[ix]
        objectives = []
        for weights in (attention, sparse_weight):
            logits = panel.predict(weights, ix)
            objectives.append(F.cross_entropy(logits, targets) if panel.direction == "text" else
                              F.binary_cross_entropy_with_logits(logits, targets.float()))
        values, _ = sgul(x, subsets[:, ix], valid)
        predicted_classes = predictions[:, ix]
        if panel.direction == "edge":
            predicted_classes = (predicted_classes >= .5).long()
        correct = (predicted_classes == targets).float()
        keep = allowed[:, ix]
        objectives.append(F.binary_cross_entropy_with_logits(values[keep], correct[keep]))
        optimizer.zero_grad()
        sum(objectives).backward()
        optimizer.step()
    return gat, sparse, sgul


def shared_oracle(panel, masks, allowed, loss, budget):
    feasible = allowed & (masks.sum(-1) <= budget)
    grouped = panel.targets[:, 0].unique(return_inverse=True)[1]
    if panel.direction == "text":
        alternatives = loss.masked_fill(~feasible, torch.inf)
    else:
        n = 2**(panel.valid.shape[1]//2)
        alternatives = loss.masked_fill(~feasible, torch.inf).reshape(n, n, -1).min(0).values
    groups = alternatives.new_zeros(alternatives.shape[0], int(grouped.max())+1)
    groups.scatter_add_(1, grouped.expand_as(alternatives), alternatives)
    selected = groups.argmin(0)[grouped]
    return alternatives[selected, torch.arange(len(selected), device=selected.device)].mean()


def calibration(model, scale, dev):
    candidates = [(b, c, f) for b in (0., .5, 1., 2.) for c in (0., .05, .15) for f in (False, True)]
    risk = [[] for _ in candidates]
    for panel, features, masks, allowed, loss, _ in dev:
        mean, spread = model(features["aware"], masks, panel.valid)
        ix = torch.arange(len(panel.truth), device=loss.device)
        for row, (beta, cost, fallback) in zip(risk, candidates):
            score = mean-mean[-1:]-beta*spread
            pick = choose(score, masks, allowed, 2, cost=cost)
            if fallback:
                pick = torch.where(score[pick, ix] > 0, pick, allowed.float().mul(
                    torch.arange(len(masks), device=loss.device)[:, None]).argmax(0))
            row.append(float(loss[pick, ix].mean()))
    return min(zip(risk, candidates), key=lambda x: np.mean(x[0]))[1]


def fit_models(train, steps, rng):
    by_name = {"base_loo": ("base", "loo"), "target_loo": ("target", "loo"),
               "aware_loo": ("aware", "loo"), "shapley": ("aware", "shapley")}
    local = {key: (feature, *fit_messages(train, feature, goal, steps, rng))
             for key, (feature, goal) in by_name.items()}
    sets = {name: fit_sets(train, steps, rng, decision) for name, decision in
            (("deep_sets", False), ("decision", True))}
    direct = {name: fit_direct(train, steps, rng, attention) for name, attention in
              (("direct", False), ("attention_control", True))}
    return local, sets, direct


def evaluate(data, local, sets, direct, risk, rng, comparators=None, uncertainty=None):
    panel, features, masks, allowed, loss, predictions = data
    counts = masks.sum(-1)
    ix = torch.arange(len(panel.truth), device=loss.device)
    full = loss[-1]
    loo = (loss[[-1-(1<<j) for j in range(panel.valid.shape[1])]]-full).T
    valid = panel.valid
    stats, scores = {}, {}
    for name, (feature, model, scale) in local.items():
        priority = model(features[feature]).squeeze(-1)
        values = priority[valid]*scale
        truth = loo[valid]
        rho = spearmanr(values.cpu(), truth.cpu()).statistic
        stats[name] = dict(mae=float((values-truth).abs().mean()),
                           spearman=float(rho) if np.isfinite(rho) else None,
                           sign_accuracy=float(((values > 0) == (truth > 0)).float().mean()))
        scores[name] = (masks*priority).sum(-1)
    for key, model in direct.items():
        scores[key] = (masks*model(features["aware"]).squeeze(-1)).sum(-1)
    if comparators:
        gat, sparse, sgul = comparators
        value, _ = sgul(features["aware"], masks, valid)
        scores.update(gat_local=(masks*gat(panel, slice(None))).sum(-1),
                      neural_sparse_local=(masks*sparse(features["aware"]).squeeze(-1)).sum(-1),
                      sgul_local=(masks*shapley_values(-value.sigmoid(), valid.shape[1])).sum(-1))
    risk_score = None
    conservative_scores = {}
    for key, (model, scale) in sets.items():
        mean, spread = model(features["aware"], masks, valid)
        scores[key] = mean
        actual = (full-loss)[allowed]
        predicted = (mean-mean[-1:])[allowed]*scale
        rho = spearmanr(predicted.cpu(), actual.cpu()).statistic
        stats[key] = dict(mae=float((predicted-actual).abs().mean()),
                          spearman=float(rho) if np.isfinite(rho) else None,
                          sign_accuracy=float(((predicted > 0) == (actual > 0)).float().mean()))
        if key == "decision":
            beta, cost, fallback = risk
            risk_score = mean-mean[-1:]-beta*spread
            stats[key]["normal_90_coverage"] = float(((predicted-actual).abs() <= 1.645*spread[allowed]*scale).float().mean())
    for key, (predictor, scale, params) in (uncertainty or {}).items():
        mean, spread = predictor(features["aware"], masks, valid)
        beta_, cost_, fallback_ = params
        conservative_scores[key] = (mean-mean[-1:]-beta_*spread, cost_, fallback_)
        error = ((mean-mean[-1:])[allowed]*scale - (full-loss)[allowed]).abs()
        stats[key] = {"mae": float(error.mean()), "normal_90_coverage": float(
            (error <= 1.645*spread[allowed]*scale).float().mean())}
    results = {}
    random = torch.rand(loss.shape, device=loss.device, generator=rng)
    for budget in (1, 2, 3, 4):
        picked = {f"{key}_{mode}": choose(score, masks, allowed, budget, exact=mode == "exact")
                  for mode in ("exact", "at_most") for key, score in scores.items()}
        picked["random_exact"] = choose(random, masks, allowed, budget, exact=True)
        picked["oracle_cap_analysis_only"] = choose(-loss, masks, allowed, budget)
        picked["oracle_exact_analysis_only"] = choose(-loss, masks, allowed, budget, exact=True)
        threshold_score = scores["aware_loo"]
        picked["threshold"] = choose(threshold_score, masks, allowed, budget, cost=.1)
        picked["decision_cost"] = choose(scores["decision"], masks, allowed, budget, cost=cost)
        decision_count = counts[picked["decision_cost"], ix]
        for key, value in (("matched_random", random), ("matched_oracle_analysis_only", -loss)):
            picked[key] = value.masked_fill(~(allowed & (counts == decision_count)), -torch.inf).argmax(0)
        picked["risk"] = choose(risk_score, masks, allowed, budget, cost=cost)
        fallback_mask = risk_score[picked["risk"], ix] <= 0 if fallback else torch.zeros_like(ix, dtype=torch.bool)
        picked["risk_fallback_full"] = torch.where(fallback_mask, len(masks)-1, picked["risk"])
        for label, (score, cost_, fallback_) in conservative_scores.items():
            p = choose(score, masks, allowed, budget, cost=cost_)
            picked[label] = p
            if fallback_:
                picked[label+"_fallback_full"] = torch.where(score[p, ix] <= 0, len(masks)-1, p)
        results[budget] = {key: {**summarize(loss[p, ix], predictions[p, ix], panel.truth, panel.direction),
                                  "panel_messages": float(counts[p, ix].mean()),
                                  "zero_fraction": float((counts[p, ix] == 0).float().mean())}
                           for key, p in picked.items()}
        if hasattr(panel, "noisy") and panel.noisy.any():
            for key, p in picked.items():
                selected_mask = masks[p, ix]
                results[budget][key]["noise_retention"] = float(selected_mask[panel.noisy].mean())
                good = valid & ~panel.noisy
                results[budget][key]["clean_retention"] = float(selected_mask[good].mean()) if good.any() else None
    return dict(rates=panel.rates, full=summarize(full, predictions[-1], panel.truth, panel.direction),
                targets=len(ix), budget=results, scores=stats, unique_subsets=int(allowed.sum()),
                shared_oracle_k2=float(shared_oracle(panel, masks, allowed, loss, 2)),
                sign_majority=float(torch.maximum((loo[valid] > 0).float().mean(), (loo[valid] <= 0).float().mean())),
                harmful={str(e): float((loo[valid] < -e).float().mean()) for e in (0., .001, .005, .01)})


def fit_denoiser(backbone, cases, selector, direction, device, seed, mode):
    model, utility = copy.deepcopy(backbone), copy.deepcopy(selector)
    parameters = model.mix[-1].parameters() if mode == "partial" else model.parameters()
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": .0002},
        {"params": utility.parameters(), "lr": .001},
    ])
    rng = torch.Generator(device=device).manual_seed(seed+817)
    for step in range(12):
        case = cases[step % len(cases)][0]
        panel = MessagePanel(model, case, direction, 80, rng)
        if mode == "full_control":
            objective = panel.evaluate(panel.valid.float()[None])[0].mean()
        else:
            features = features_for(panel)["aware"].detach()
            masks, allowed = enumerate_subsets(panel.valid)
            loss, _ = panel.evaluate(masks)
            mean, spread = utility(features, masks, panel.valid)
            feasible = allowed & (masks.sum(-1) <= 2)
            probability = (mean/.35).masked_fill(~feasible, -torch.inf).softmax(0)
            current_utility = loss[-1:].detach()-loss.detach()
            scale = current_utility[allowed].std(unbiased=False).clamp_min(1e-5)
            objective = (probability*loss).sum(0).mean()
            objective = objective + .05*subset_objective(mean, spread, current_utility/scale, masks, allowed, .3)
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
    return model.eval(), utility.eval()


@torch.no_grad()
def deployment(panel, selector, use_selector):
    all_rows = slice(None)
    if use_selector:
        features = features_for(panel)["aware"]
        masks, allowed = enumerate_subsets(panel.valid)
        scores, _ = selector(features, masks, panel.valid)
        pick = choose(scores, masks, allowed, 2)
        weights = masks[pick, torch.arange(len(pick), device=pick.device)]
    else:
        weights = panel.valid.float()
    return panel.predict(weights, all_rows), weights


def score_logits(logits, panel):
    if panel.direction == "text":
        loss = F.cross_entropy(logits, panel.truth, reduction="none")
        prediction = logits[..., 2:].argmax(-1)+2
    else:
        loss = F.binary_cross_entropy_with_logits(logits, panel.truth.float(), reduction="none")
        prediction = logits.sigmoid()
    return summarize(loss, prediction, panel.truth, panel.direction)


@torch.no_grad()
def paired_timing(backbone, case, selector, direction, device, seed, limit):
    outcomes = {}
    for name, use in (("full", False), ("selected", True)):
        elapsed, kept = [], []
        for repeat in range(4):
            rng = torch.Generator(device=device).manual_seed(seed+913)
            start = tick(device)
            panel = MessagePanel(backbone, case, direction, limit, rng)
            logits, weights = deployment(panel, selector, use)
            elapsed.append(tick(device)-start)
            kept.append(float(weights.sum(1).mean()))
        outcomes[name] = dict(median_ms=1000*float(np.median(elapsed[1:])), panel_messages=kept[-1])
    return outcomes


@torch.no_grad()
def diffusion_case(model, case, selectors, device, seed, use_selector):
    ids, states, _ = case.initial()
    token_mask, edge_mask = case.mask.clone(), case.edge_mask.clone()
    only_tokens = case.mask.nonzero()
    rng = torch.Generator(device=device).manual_seed(seed+1776)
    chosen = only_tokens[torch.randperm(len(only_tokens), device=device, generator=rng)[:192]]
    token_mask.zero_()
    token_mask[chosen[:, 0], chosen[:, 1]] = True
    unknown = case.a_index[torch.randperm(len(case.a_index), device=device, generator=rng)[:192]]
    edge_mask.zero_()
    edge_mask[unknown] = True
    all_tokens, all_edges = token_mask.clone(), edge_mask.clone()
    edge_probabilities = torch.zeros_like(states, dtype=torch.float)
    traces = []
    for step in range(2):
        rates = (case.rates[0]*(1-.5*step), case.rates[1]*(1-.5*step))
        progress = Case(ids, case.ids, token_mask, case.pairs, states, edge_mask, case.base, rates)
        progress.a_truth = case.labels[progress.a_index]
        trace = {"clocks": rates}
        panels = {d: MessagePanel(model, progress, d, 192, rng) for d in ("text", "edge")}
        for direction, panel in panels.items():
            logits, weights = deployment(panel, selectors[direction], use_selector)
            count = (len(panel.truth)+1)//2 if step == 0 else len(panel.truth)
            if direction == "text":
                probabilities = logits.softmax(-1)
                confidence, prediction = probabilities[:, 2:].max(1)
                chosen = confidence.topk(count).indices
                indices = tuple(i[chosen] for i in panel.query)
                ids[indices] = prediction[chosen]+2
                token_mask[indices] = False
            else:
                probability = logits.sigmoid()
                chosen = (probability-.5).abs().topk(count).indices
                indices = panel.query[chosen]
                states[indices] = (probability[chosen] >= .5).long()
                edge_probabilities[indices] = probability[chosen]
                edge_mask[indices] = False
            trace[direction] = dict(revealed=count, panel_messages=float(weights.sum(1).mean()))
        traces.append(trace)
    assert not token_mask.any() and not edge_mask.any()
    assert torch.equal(ids[~case.mask], case.context[~case.mask])
    assert torch.equal(states[~case.edge_mask], case.labels[~case.edge_mask])
    p = edge_probabilities[all_edges]
    return dict(token_accuracy=float((ids[all_tokens] == case.ids[all_tokens]).float().mean()),
                edge=summarize(F.binary_cross_entropy(p, case.labels[all_edges].float(), reduction="none"),
                               p, case.labels[all_edges], "edge"), traces=traces)


@torch.no_grad()
def independent_synthetic(model, selectors, seed, device, limit):
    checkpoint = torch.load(Path("checkpoints/research_twinmask_v2") / f"synthetic_reference_{seed}.pt",
                            map_location="cpu", weights_only=False)
    vocabulary = checkpoint["vocabulary"].to(device)
    mapping = torch.full((50265,), OTHER, device=device, dtype=torch.long)
    mapping[vocabulary] = torch.arange(3, len(vocabulary)+3, device=device)
    results = {}
    for name in SYNTHETIC:
        data = synthetic_graph(name, seed_offset=1000+seed)
        raw = data["tokens"][:, 1:].to(device)
        ids = mapping[raw].masked_fill(raw <= 2, PAD)
        pairs = torch.cat((data["test_pos"], data["test_neg"]), 1).to(device)
        labels = (torch.arange(pairs.shape[1], device=device) < data["test_pos"].shape[1]).long()
        noise = data["noise_edges"].to(device)
        base = torch.cat((data["train_pos"].to(device), noise), 1)
        generator = torch.Generator(device=device).manual_seed(seed+912)
        result = {}
        for direction in ("text", "edge"):
            scores = []
            for tx, ta in ((.25, .75), (.5, .5), (.75, .25), (.9, .9)):
                case = Case(ids, ids, corruption(ids > OTHER, tx, generator), pairs, labels,
                            corruption(torch.ones_like(labels, dtype=torch.bool), ta, generator), base, (tx, ta))
                panel = MessagePanel(model, case, direction, limit, generator)
                features = features_for(panel)["aware"]
                subsets, allowed = enumerate_subsets(panel.valid)
                value, _ = selectors[direction](features, subsets, panel.valid)
                learned = choose(value, subsets, allowed, 2)
                ix = torch.arange(len(panel.truth), device=device)
                count = subsets[learned, ix].sum(-1)
                valid = allowed & (subsets.sum(-1) == count)
                random = torch.rand(value.shape, device=device, generator=generator)
                sampled = random.masked_fill(~valid, -torch.inf).argmax(0)
                losses, predictions = panel.evaluate(subsets)
                oracle = (-losses).masked_fill(~valid, -torch.inf).argmax(0)
                scores.append({key: summarize(losses[choice, ix], predictions[choice, ix], panel.truth, direction)
                               for key, choice in dict(full=len(subsets)-1, learned=learned, random=sampled,
                                                       oracle_analysis_only=oracle).items()})
                if noise.numel():
                    low, high = torch.minimum(panel.slots, panel.receivers), torch.maximum(panel.slots, panel.receivers)
                    marked = torch.isin(low*len(ids)+high, noise[0]*len(ids)+noise[1]) & panel.valid
                    if marked.any():
                        scores[-1]["noise_retention"] = {
                            key: float(subsets[choice, ix][marked].mean())
                            for key, choice in (("learned", learned), ("random", sampled), ("oracle", oracle))}
            result[direction] = scores
        results[name] = result
    return results


def run(dataset, seed, device, limit, steps, panel_size, resplit=False, layer="last", extra=False,
        comparators=False, transfer=False, tag=""):
    set_seed(seed)
    start = tick(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    groups = {direction: [[], [], []] for direction in ("text", "edge")}
    saved_cases = []
    noise = synthetic_graph(dataset)["noise_edges"].to(device) if dataset in SYNTHETIC else None
    with torch.no_grad():
        for model, fit, dev, test, rng in contexts(dataset, seed, device, include_dev=True, resplit=resplit):
            saved_cases.append((fit, dev, test))
            for direction, splits in groups.items():
                for case, rows in zip((fit, dev, test), splits):
                    panel = MessagePanel(model, case, direction, limit, rng, panel_size, layer)
                    panel.rates = case.rates
                    if noise is not None and noise.numel():
                        n = len(case.ids)
                        low = torch.minimum(panel.slots, panel.receivers)
                        high = torch.maximum(panel.slots, panel.receivers)
                        panel.noisy = torch.isin(low*n+high, noise[0]*n+noise[1]) & panel.valid
                    rows.append(bundle(panel))
    results = {}
    selectors, states = {}, {}
    for direction, (train, dev, test) in groups.items():
        local, sets, direct = fit_models(train, steps, rng)
        baseline_models = fit_comparators(train, steps, rng) if comparators else None
        uncertainty_models = {}
        if comparators:
            sets["set_attention"] = fit_sets(train, steps, rng, attention=True)
            dropout_model, scale = fit_sets(train, steps, rng, True, dropout=.1)
            ensemble_members = [sets["decision"][0], *(fit_sets(train, steps, rng, True)[0] for _ in range(2))]
            dropout_model.eval()
            uncertainty_models = {"mc_dropout": (PredictiveMixture([dropout_model], True), scale),
                                  "deep_ensemble": (PredictiveMixture(ensemble_members), sets["decision"][1])}
        selectors[direction] = sets["decision"][0]
        for _, estimator, _ in local.values():
            estimator.eval()
        with torch.no_grad():
            risk = calibration(*sets["decision"], dev)
            uncertainty = {key: (*value, calibration(*value, dev)) for key, value in uncertainty_models.items()}
            cases = [evaluate(row, local, sets, direct, risk, rng, baseline_models, uncertainty) for row in test]
        results[direction] = dict(risk_beta=risk[0], risk_cost_std_units=risk[1], risk_fallback=risk[2], cases=cases)
        states[direction] = dict(features=train[0][1]["aware"].shape[-1],
                                 decision=selectors[direction].state_dict(),
                                 scale=sets["decision"][1], risk=risk)
        if uncertainty:
            results[direction]["alternative_calibration"] = {key: value[-1] for key, value in uncertainty.items()}
        if extra:
            results[direction]["timing"] = paired_timing(model, saved_cases[1][2], selectors[direction], direction,
                                                        device, seed, limit)
            evaluated = {}
            annealed = {"frozen": (model, selectors[direction])}
            for mode in ("partial", "end_to_end", "full_control"):
                annealed[mode] = fit_denoiser(model, saved_cases, selectors[direction], direction, device, seed, mode)
                adapted, estimator = annealed[mode]
                states[direction][mode] = dict(denoiser=adapted.state_dict(), selector=estimator.state_dict())
            with torch.no_grad():
                for mode, (backbone, selector) in annealed.items():
                    evaluated[mode] = []
                    matched_rng = torch.Generator(device=device).manual_seed(seed+223)
                    for _, _, case in saved_cases:
                        panel = MessagePanel(backbone, case, direction, limit, matched_rng)
                        logits, weights = deployment(panel, selector, mode != "full_control")
                        evaluated[mode].append({**score_logits(logits, panel),
                                                "panel_messages": float(weights.sum(1).mean())})
            results[direction]["fine_tuning"] = evaluated
    diffusion = ({name: [diffusion_case(model, case, selectors, device, seed+j, use)
                         for j, (_, _, case) in enumerate(saved_cases)]
                  for name, use in (("full", False), ("selected", True))} if extra else None)
    directory = Path("checkpoints/research_twinmask_v4")
    directory.mkdir(parents=True, exist_ok=True)
    suffix = suffix_for(resplit, panel_size, layer, extra, comparators, transfer) + ("_"+tag if tag else "")
    torch.save(dict(dataset=dataset, seed=seed, layer=layer, panel=panel_size, resplit=resplit, directions=states),
               directory / f"{dataset}_{seed}{suffix}.pt")
    transfer_results = independent_synthetic(model, selectors, seed, device, limit) if transfer else None
    return dict(seed=seed, panel=panel_size, limit=limit, steps=steps, resplit=resplit, layer=layer, diffusion=diffusion,
                independent_synthetic=transfer_results,
                seconds=tick(device)-start, peak_mib=torch.cuda.max_memory_allocated(device)/2**20
                if device.type == "cuda" else None, directions=results)


def suffix_for(resplit, panel, layer, extra, comparators=False, transfer=False):
    return (("_resplit" if resplit else "") + ("_panel6" if panel == 6 else "") +
            ("_"+layer if layer != "last" else "") + ("_extra" if extra else "") +
            ("_baselines" if comparators else "") + ("_transfer" if transfer else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer", "pubmed"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--panel", type=int, choices=(4, 6), default=4)
    parser.add_argument("--limit", type=int, default=384)
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--resplit", action="store_true")
    parser.add_argument("--layer", choices=("first", "last", "both"), default="last")
    parser.add_argument("--extra", action="store_true", help="run fine-tuning, timing, and two-call diffusion")
    parser.add_argument("--comparators", action="store_true")
    parser.add_argument("--transfer", action="store_true", help="evaluate on independently generated synthetic graphs")
    parser.add_argument("--tag", default="", help="optional alphanumeric experiment label")
    args = parser.parse_args()
    if args.limit < 32 or args.steps < 1:
        parser.error("use at least 32 targets and one optimization step")
    if args.transfer and (args.datasets != ["synthetic_reference"] or args.panel != 4):
        parser.error("--transfer uses the four-slot synthetic_reference checkpoint and selector")
    if args.tag and not args.tag.replace("_", "").isalnum():
        parser.error("--tag must contain only letters, digits, and underscores")
    for dataset in args.datasets:
        rows = []
        for seed in args.seeds:
            rows.append(run(dataset, seed, torch.device(args.device), args.limit, args.steps, args.panel, args.resplit,
                            args.layer, args.extra, args.comparators, args.transfer, args.tag))
            suffix = suffix_for(args.resplit, args.panel, args.layer, args.extra, args.comparators, args.transfer)
            if args.tag:
                suffix += "_"+args.tag
            save_json(rows, "research_twinmask_v4", f"{dataset}{suffix}.json")
            print(dataset, seed, round(rows[-1]["seconds"], 2), flush=True)


if __name__ == "__main__":
    main()
