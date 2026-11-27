"""TwinMask-CUS pilot using frozen V2 checkpoints and local message interventions."""
import argparse
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F

from scripts.research_ensemble import tick
from scripts.research_twinmask import (Case, MASK, OTHER, PAD, SYNTHETIC, SparseTwinMask, corruption,
                                       resplit_link_data, synthetic_graph)
from twinmask.common import save_json, set_seed
from twinmask.data import load, sample_negatives

SLOT_PAIRS = [(i, j) for i in range(4) for j in range(i+1, 4)]


class MessagePanel:
    """Intervene in the final aggregation while preserving the observed graph and degree."""

    def __init__(self, model, case, direction, limit, generator, panel_size=4, layer="last"):
        self.model, self.direction = model, direction
        self.layer = layer
        ids, state, _ = case.initial()
        x = torch.randperm(len(case.x_truth), generator=generator, device=ids.device)[:limit]
        a = torch.randperm(len(case.a_truth), generator=generator, device=ids.device)[:limit]
        query = tuple(index[x] for index in case.x_index)
        selected, semantic, h, total, degree, adjacency = model.encode_messages(
            ids, case.pairs, state, case.base, query, *case.clocks(case.initial()))
        if direction == "text":
            targets = query[0][:, None]
            self.selected, self.truth = selected, case.x_truth[x]
            per_target = panel_size
        else:
            targets = case.pairs[:, case.a_index[a]].T
            self.state, self.truth = state[case.a_index[a]], case.a_truth[a]
            per_target = panel_size // 2
        neighborhoods = [[] for _ in range(len(ids))]
        for source, target in adjacency.T.cpu().tolist():
            neighborhoods[target].append(source)
        # Fixed seed and outcome-independent sampling of the optional message pool.
        rng = np.random.default_rng(int(torch.randint(2**31, (), device=ids.device, generator=generator)))
        panels = {node: rng.choice(neighborhoods[node], min(per_target, len(neighborhoods[node])),
                                   replace=False).tolist() for node in sorted(set(targets.flatten().tolist()))}
        slots = torch.tensor([panels[node]+[-1]*(per_target-len(panels[node]))
                              for node in targets.flatten().tolist()], device=ids.device).reshape(len(targets), -1)
        receivers = targets.repeat_interleave(per_target, dim=1)
        self.valid = slots >= 0
        safe = slots.clamp_min(0)
        self.targets, self.slots, self.receivers = targets, slots, receivers
        self.query = query if direction == "text" else case.a_index[a]
        self.sender_h, self.receiver_h = h[safe], h[receivers]
        self.position = query[1]
        self.h, self.total, self.mass = h[targets], total[targets], degree[targets]
        # Edge interventions remove the direct pooled text component of a message.
        # The structural residual and indirect text paths remain available.
        senders = h if direction == "text" else semantic
        self.messages = senders[safe] * self.valid[..., None]
        if layer != "last":
            start = semantic + model.clock_x(semantic.new_tensor(case.clocks(case.initial())[0]))
            start = start + model.clock_a(semantic.new_tensor(case.clocks(case.initial())[1]))
            source, destination = adjacency
            first_totals = torch.zeros_like(start).index_add_(0, destination, start[source])
            first_senders = start if direction == "text" else semantic
            self.first_h, self.first_total = start[targets], first_totals[targets]
            self.first_messages = first_senders[safe] * self.valid[..., None]
            if direction == "edge":
                n = len(ids)
                observed = source*n+destination
                assert not torch.isin(targets[:, 0]*n+targets[:, 1], observed).any(), "Candidate must be hidden"
        self.unit = F.normalize(semantic[safe]) * self.valid[..., None]
        sim = (self.unit * F.normalize(semantic[receivers])).sum(-1)
        text_fraction = ((ids > OTHER).sum(-1) / (ids != PAD).sum(-1).clamp_min(1))
        tx, ta = case.rates
        self.features = torch.stack((sim, degree[safe].log1p(), degree[receivers].log1p(),
                                     text_fraction[safe], text_fraction[receivers], senders[safe].norm(dim=-1),
                                     h[receivers].norm(dim=-1), torch.full_like(sim, tx), torch.full_like(sim, ta)), -1)

    def predict(self, weights, indices):
        def subtract(total, messages):
            removed = (1-weights)[..., None]*messages[indices]
            return total[indices]-removed.reshape(*removed.shape[:-2], self.h.shape[1], -1,
                                                   self.h.shape[-1]).sum(-2)

        h, mass = self.h[indices], self.mass[indices]
        if self.layer != "last":
            layer = self.model.mix[0]
            first = self.first_h[indices]
            aggregate = subtract(self.first_total, self.first_messages) / mass.clamp_min(1)[..., None]
            h = first + F.gelu(layer.lin_l(aggregate) + layer.lin_r(first))
        total = self.total[indices] if self.layer == "first" else subtract(self.total, self.messages)
        h = self.model.finish_messages(h, total, mass)
        if self.direction == "text":
            return self.model.text(self.model.norm(self.selected[indices] + h[..., 0, :]))
        return self.model.predict_pairs(h[..., 0, :], h[..., 1, :],
                                        self.mass[indices].log1p().sum(-1).expand(h.shape[:-2]),
                                        self.state[indices].expand(h.shape[:-2]))

    def evaluate(self, weights):
        losses, probabilities = [], []
        for start in range(0, len(self.truth), 128):
            end = min(start+128, len(self.truth))
            logits = self.predict(weights[:, start:end], slice(start, end))
            truth = self.truth[start:end].expand(len(weights), -1)
            if self.direction == "text":
                losses.append(-logits.log_softmax(-1).gather(-1, truth[..., None]).squeeze(-1))
                probabilities.append(logits[..., 2:].argmax(-1)+2)
            else:
                losses.append(F.binary_cross_entropy_with_logits(logits, truth.float(), reduction="none"))
                probabilities.append(logits.sigmoid())
        return torch.cat(losses, 1), torch.cat(probabilities, 1)


def contexts(name, seed, device, *, include_dev=False, resplit=False):
    data = synthetic_graph(name) if name in SYNTHETIC else load(name)
    if resplit:
        data = resplit_link_data(data, seed)
    directory = "checkpoints/research_twinmask_v2" + ("_resplit" if resplit else "")
    checkpoint = torch.load(Path(directory) / f"{name}_{seed}.pt",
                            map_location="cpu", weights_only=False)
    vocabulary = checkpoint["vocabulary"].to(device)
    model = SparseTwinMask(len(vocabulary)+3, 16, width=checkpoint["model"]["embed.weight"].shape[1]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.requires_grad_(False).eval()
    raw = data["tokens"][:, 1:17].long().to(device)
    mapping = torch.full((50265,), OTHER, device=device, dtype=torch.long)
    mapping[vocabulary] = torch.arange(3, len(vocabulary)+3, device=device)
    ids = mapping[raw].masked_fill((raw <= 2) | (raw == 50264), PAD)
    n = len(ids)
    generator = torch.Generator(device=device).manual_seed(seed)
    order = checkpoint["node_split"].to(device)
    train, fit, dev, test = order.split([n//2, n//5, n//10, n-n//2-n//5-n//10])
    context = ids.clone()
    context[order[len(train):]] = context[order[len(train):]].masked_fill(ids[order[len(train):]] > OTHER, MASK)
    pairs = data["train_pos"].to(device)
    perm = checkpoint["edge_split"].to(device)
    index = perm[len(perm)//5:len(perm)//5+len(perm)//10]
    keep = torch.ones(pairs.shape[1], dtype=torch.bool, device=device)
    keep[index] = False
    keys = torch.cat([data[key] for key in ("train_pos", "val_pos", "test_pos", "cold_pos",
                                            "val_neg", "test_neg", "cold_neg")], 1)
    forbidden = set((keys[0]*n+keys[1]).tolist())
    negatives = sample_negatives(n, len(index), forbidden, np.random.default_rng(seed+3000)).to(device)
    fit_pairs = torch.cat((pairs[:, index], negatives), 1)
    fit_labels = torch.arange(fit_pairs.shape[1], device=device) < len(index)
    evaluation = torch.cat([data[k] for k in ("test_pos", "test_neg")], 1)
    truth_table = dict(zip((evaluation[0]*n+evaluation[1]).tolist(),
                           [1]*data["test_pos"].shape[1]+[0]*data["test_neg"].shape[1]))
    dev_index = perm[len(perm)//5+len(perm)//10:len(perm)//5+2*(len(perm)//10)]
    if include_dev:
        keep[dev_index] = False
        forbidden.update((negatives[0]*n+negatives[1]).tolist())
        dev_neg = sample_negatives(n, len(dev_index), forbidden, np.random.default_rng(seed+4000)).to(device)
        dev_pairs = torch.cat((pairs[:, dev_index], dev_neg), 1)
        dev_labels = torch.arange(dev_pairs.shape[1], device=device) < len(dev_index)
    noisy = data.get("noise_edges", pairs[:, :0]).to(device)
    for rates, saved in zip(((.25, .75), (.5, .5), (.75, .25), (.9, .9)), checkpoint["masks"]):
        fit_context, test_context = context.clone(), context.clone()
        fit_context[fit], test_context[test] = ids[fit], ids[test]
        pool = torch.zeros_like(ids, dtype=torch.bool)
        pool[fit] = ids[fit] > OTHER
        fit_case = Case(fit_context, ids, corruption(pool, rates[0], generator), fit_pairs, fit_labels.long(),
                        corruption(torch.ones_like(fit_labels), rates[1], generator), pairs[:, keep], rates)
        candidates = saved["pairs"]
        test_labels = torch.tensor([truth_table[k] for k in (candidates[0]*n+candidates[1]).tolist()], device=device)
        test_case = Case(test_context, ids, saved["tokens"].to(device), candidates.to(device), test_labels,
                         saved["edge_mask"].to(device), torch.cat((pairs, noisy), 1), rates)
        if include_dev:
            dev_context = context.clone()
            dev_context[dev] = ids[dev]
            dev_pool = torch.zeros_like(pool)
            dev_pool[dev] = ids[dev] > OTHER
            dev_case = Case(dev_context, ids, corruption(dev_pool, rates[0], generator), dev_pairs, dev_labels.long(),
                            corruption(torch.ones_like(dev_labels), rates[1], generator), pairs[:, keep], rates)
            yield model, fit_case, dev_case, test_case, generator
        else:
            yield model, fit_case, test_case, generator


def subset_weights(panel):
    bits = torch.arange(16, device=panel.valid.device)[:, None] >> torch.arange(4, device=panel.valid.device)
    return ((bits % 2)[:, None, :] & panel.valid[None]).float()


def score_network(features):
    return nn.Sequential(nn.Linear(features.shape[-1], 48), nn.LayerNorm(48), nn.GELU(),
                         nn.Linear(48, 24), nn.GELU(), nn.Linear(24, 1)).to(features.device)


def fit_utility(features, utility, valid):
    estimator = score_network(features)
    scale = utility[valid].std(unbiased=False).clamp_min(1e-5)
    utility = utility / scale
    optimizer = torch.optim.AdamW(estimator.parameters(), lr=.003)
    pairs = valid[:, :, None] & valid[:, None, :]
    target = (utility[:, :, None]-utility[:, None, :]).sign()
    pairs &= target != 0
    for _ in range(120):
        prediction = estimator(features).squeeze(-1)
        delta = prediction[:, :, None]-prediction[:, None, :]
        regression = F.smooth_l1_loss(prediction[valid], utility[valid])
        ranking = F.softplus(-target*delta)[pairs].mean() if pairs.any() else regression.new_zeros(())
        optimizer.zero_grad()
        (regression+.15*ranking).backward()
        optimizer.step()
    return estimator, scale


def pair_features(panel):
    left, right = zip(*SLOT_PAIRS)
    a, b = panel.features[:, left], panel.features[:, right]
    cosine = (panel.unit[:, left]*panel.unit[:, right]).sum(-1, keepdim=True)
    return torch.cat((a+b, (a-b).abs(), cosine), -1), panel.valid[:, left] & panel.valid[:, right]


def interactions(loss):
    return torch.stack([-loss[(1<<i)|(1<<j)]+loss[1<<i]+loss[1<<j]-loss[0]
                        for i, j in SLOT_PAIRS], -1)


def fit_direct_gate(panels, generator):
    estimator = score_network(panels[0].features)
    optimizer = torch.optim.AdamW(estimator.parameters(), lr=.003)
    for epoch in range(120):
        panel = panels[epoch % len(panels)]
        selected = torch.randperm(len(panel.truth), generator=generator, device=panel.truth.device)[:256]
        weights = estimator(panel.features[selected]).squeeze(-1).sigmoid()*panel.valid[selected]
        logits = panel.predict(weights, selected)
        truth = panel.truth[selected]
        loss = (F.cross_entropy(logits, truth) if panel.direction == "text" else
                F.binary_cross_entropy_with_logits(logits, truth.float()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return estimator


def choice(mask, values):
    return values.masked_fill(~mask, -torch.inf).argmax(0)


def summarize(loss, pred, truth, direction):
    result = {"loss": float(loss.mean())}
    if direction == "text":
        result.update(accuracy=float((pred == truth).float().mean()), perplexity=math.exp(result["loss"]))
    else:
        result.update(auc=roc_auc_score(truth.cpu(), pred.cpu()), ap=average_precision_score(truth.cpu(), pred.cpu()))
    return result


def budget_choices(panel, subsets, utility, synergy, direct, generator):
    counts = subsets.sum(-1)
    similarity = (panel.unit @ panel.unit.transpose(-2, -1)).clamp_min(0).triu(1)
    redundancy = torch.einsum("sni,nij,snj->sn", subsets, similarity, subsets)
    objectives = dict(random=torch.rand(counts.shape, device=counts.device, generator=generator),
                      similarity=(subsets*panel.features[..., 0]).sum(-1),
                      top_utility=(subsets*utility).sum(-1), direct_top=(subsets*direct).sum(-1))
    objectives["complementary"] = objectives["top_utility"]-.25*redundancy
    left, right = zip(*SLOT_PAIRS)
    joint = subsets[..., left]*subsets[..., right]
    objectives["synergy"] = objectives["complementary"]+(joint*synergy).sum(-1)
    return counts, {budget: {key: choice(counts == panel.valid.sum(1).clamp(max=budget), value)
                            for key, value in objectives.items()} for budget in (1, 2, 3, 4)}


def phase(name, seed, device, limit):
    set_seed(seed)
    started = tick(device)
    fit, test = {k: [] for k in ("text", "edge")}, {k: [] for k in ("text", "edge")}
    for model, f, t, generator in contexts(name, seed, device):
        with torch.no_grad():
            for direction in fit:
                panel = MessagePanel(model, f, direction, limit, generator)
                weights = subset_weights(panel)
                loss, _ = panel.evaluate(weights)
                utility = (loss[[14, 13, 11, 7]]-loss[15]).T
                fit[direction].append((panel, utility, interactions(loss)))
                test[direction].append((t.rates, MessagePanel(model, t, direction, limit, generator)))
    rows = {}
    weights_to_save = {}
    for direction in fit:
        panels, targets, pair_targets = zip(*fit[direction])
        estimator, scale = fit_utility(torch.cat([p.features for p in panels]), torch.cat(targets),
                                       torch.cat([p.valid for p in panels]))
        pair_x, pair_valid = map(torch.cat, zip(*(pair_features(p) for p in panels)))
        synergy_model, synergy_scale = fit_utility(pair_x, torch.cat(pair_targets), pair_valid)
        direct_model = fit_direct_gate(panels, generator)
        weights_to_save[direction] = dict(utility=estimator.state_dict(), utility_scale=scale,
                                          synergy=synergy_model.state_dict(), synergy_scale=synergy_scale,
                                          direct=direct_model.state_dict())
        cases = []
        with torch.no_grad():
            for rates, panel in test[direction]:
                estimated = estimator(panel.features).squeeze(-1)
                subsets = subset_weights(panel)
                synergy = synergy_model(pair_features(panel)[0]).squeeze(-1)*synergy_scale/scale
                direct = direct_model(panel.features).squeeze(-1)
                counts, choices = budget_choices(panel, subsets, estimated, synergy, direct, generator)
                soft = (estimated.sigmoid()*panel.valid)[None]
                hard = ((estimated > 0) & panel.valid).float()[None]
                # Learned policies are selected before constructing label-based audit losses.
                losses, pred = panel.evaluate(torch.cat((subsets, soft, hard, (direct.sigmoid()*panel.valid)[None])))
                loo = (losses[[14, 13, 11, 7]]-losses[15]).T
                shapley = torch.zeros_like(loo)
                for slot in range(4):
                    bit = 1 << slot
                    for s in range(16):
                        if s & bit == 0:
                            size = s.bit_count()
                            shapley[:, slot] += (losses[s]-losses[s | bit]) / (4*math.comb(3, size))
                interaction = interactions(losses)
                both = pair_features(panel)[1]
                v = panel.valid
                predicted = estimated[v]*scale
                correlation = spearmanr(predicted.cpu(), loo[v].cpu()).statistic
                case = dict(rates=rates, targets=len(panel.truth), messages=int(v.sum()),
                            utility_mae=float((predicted-loo[v]).abs().mean()),
                            utility_sign_accuracy=float(((predicted > 0) == (loo[v] > 0)).float().mean()),
                            utility_spearman=float(correlation) if np.isfinite(correlation) else None,
                            negative_loo_fraction=float((loo[v] < 0).float().mean()),
                            loo_mean=float(loo[v].mean()), shapley_mean=float(shapley[v].mean()),
                            absolute_interaction_mean=float(interaction[both].abs().mean()) if both.any() else None,
                            full=summarize(losses[15], pred[15], panel.truth, direction),
                            none=summarize(losses[0], pred[0], panel.truth, direction),
                            soft=summarize(losses[16], pred[16], panel.truth, direction),
                            hard=summarize(losses[17], pred[17], panel.truth, direction),
                            direct_soft=summarize(losses[18], pred[18], panel.truth, direction), budgets={})
                ix = torch.arange(len(panel.truth), device=device)
                for budget, arms in choices.items():
                    arms["oracle_analysis_only"] = choice(counts <= budget, -losses[:16])
                    arms["oracle_matched_count_analysis_only"] = choice(
                        counts == panel.valid.sum(1).clamp(max=budget), -losses[:16])
                    arms["shapley_rank_analysis_only"] = choice(
                        counts == panel.valid.sum(1).clamp(max=budget), (subsets*shapley).sum(-1))
                    case["budgets"][budget] = {
                        arm: {**summarize(losses[pick, ix], pred[pick, ix], panel.truth, direction),
                              "messages_per_target": float(counts[pick, ix].mean())}
                        for arm, pick in arms.items()
                    }
                cases.append(case)
        rows[direction] = cases
    directory = Path("checkpoints/research_twinmask_v3")
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(weights_to_save, directory / f"{name}_{seed}.pt")
    return dict(seed=seed, seconds=tick(device)-started, target_limit=limit, directions=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer", "pubmed"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--limit", type=int, default=768)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.limit < 32:
        parser.error("--limit must be at least 32")
    for dataset in args.datasets:
        rows = []
        for seed in args.seeds:
            rows.append(phase(dataset, seed, torch.device(args.device), args.limit))
            print(dataset, seed, round(rows[-1]["seconds"], 2), flush=True)
        save_json(rows, "research_twinmask_v3", f"{dataset}.json")


if __name__ == "__main__":
    main()
