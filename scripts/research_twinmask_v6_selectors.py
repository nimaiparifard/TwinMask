"""Tune independent teachers, local utility, set/decision selectors, and mixture policies."""
import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import torch
from torch import nn
from torch.nn import functional as F

from scripts.research_twinmask_v5 import (as_log_probs, calibrate_teachers, evaluate_expected, expected_utility,
                                          fit_stacking, fusion_panel, teacher_prob, teachers, temporal)
from scripts.research_twinmask_v6 import Config, LOCKED, SEARCH_SPLIT, locked_protocol, protocol, stage_cases, train
from twinmask.common import save_json
from twinmask.set_utility import SetUtility, choose, subset_objective


FAMILIES = ("loo", "shapley", "set", "decision", "distilled")


def predictions(models, cases, direction, cap=160):
    outputs, input_rows = [], []
    model = models["twinmask"]
    for j, case in enumerate(cases):
        rng = torch.Generator(device=case.ids.device).manual_seed(192+j)
        count = len(case.x_truth if direction == "text" else case.a_truth)
        target = torch.randperm(count, generator=rng, device=case.ids.device)[:cap]
        message, valid, masks, allowed, y, degree, decode, features = fusion_panel(model, case, direction, target)
        with torch.no_grad():
            logprobs = as_log_probs(decode(torch.einsum("sni,nih->snh", masks, message)), direction)
        logits = teachers(models, case, direction, target)
        outputs.append((logprobs, masks, allowed, y, logits, degree))
        input_rows.append((features, valid))
    return outputs, input_rows


def tensors(results, inputs, temperature, chosen, direction):
    x, v = map(torch.cat, zip(*inputs))
    masks, allowed = (torch.cat([row[i] for row in results], 1) for i in (1, 2))
    losses, expected = [], []
    for logprob, _, _, y, logits, _ in results:
        n = len(logprob)
        losses.append(-logprob.gather(-1, y[None, :, None].expand(n, -1, -1)).squeeze(-1))
        expected.append(expected_utility(logprob, teacher_prob(logits, chosen, temperature, direction), n-1))
    return x, v, masks, allowed, torch.cat(losses, 1), torch.cat(expected, 1)


def scores(model, kind, x, masks, valid):
    if kind in ("loo", "shapley"):
        return (masks*model(x).squeeze(-1)).sum(-1), torch.zeros_like(masks[..., 0])
    return model(x, masks, valid)


@torch.no_grad()
def policy_loss(model, kind, data, cost):
    x, v, masks, available, loss, _ = data
    s, _ = scores(model, kind, x, masks, v)
    selected = choose(s, masks, available, 2, cost=cost)
    ix = torch.arange(len(x), device=x.device)
    count = masks[selected, ix].sum(-1)
    matched = available & (masks.sum(-1) == count)
    oracle = loss.masked_fill(~matched, torch.inf).amin(0)
    rng = torch.Generator(device=x.device).manual_seed(402)
    pick = torch.rand(loss.shape, device=x.device, generator=rng).masked_fill(~matched, -torch.inf).argmax(0)
    return dict(loss=float(loss[selected, ix].mean()), random=float(loss[pick, ix].mean()),
                oracle=float(oracle.mean()), full=float(loss[-1].mean()), messages=float(count.mean()))


def fit(kind, data, params, *, development=None, trial=None):
    torch.manual_seed(71)
    x, v, masks, allowed, loss, expected = data
    width = params["width"]
    model = (nn.Sequential(nn.Linear(x.shape[-1], width), nn.GELU(), nn.Linear(width, 1))
             if kind in ("loo", "shapley") else SetUtility(x.shape[-1], width, anchored=True)).to(x.device)
    optim = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["wd"])
    target = expected if kind == "distilled" else loss[-1:]-loss
    scale = target[allowed].std(unbiased=False).clamp_min(1e-6)
    target = target/scale
    rng = torch.Generator(device=x.device).manual_seed(498)
    best, best_state, best_step, history = float("inf"), None, params["updates"], []
    n = v.shape[1]
    if kind == "shapley":
        from twinmask.set_utility import shapley_values
        message_target = shapley_values(loss, n)/scale
    else:
        message_target = -target[[-1-(1<<j) for j in range(n)]].T
    for step in range(1, params["updates"]+1):
        ix = torch.randperm(len(x), device=x.device, generator=rng)[:128]
        if kind in ("loo", "shapley"):
            values = model(x[ix]).squeeze(-1)
            truth, keep = message_target[ix], v[ix]
            error = F.smooth_l1_loss(values[keep], truth[keep])
            differences = truth[:, :, None]-truth[:, None, :]
            pairs = keep[:, :, None] & keep[:, None, :] & (differences != 0)
            if pairs.any():
                error += params["rank"]*F.softplus(-differences[pairs].sign()*
                                                   (values[:, :, None]-values[:, None, :])[pairs]).mean()
        else:
            mean, sigma = model(x[ix], masks[:, ix], v[ix])
            error = subset_objective(mean, sigma, target[:, ix], masks[:, ix], allowed[:, ix],
                                     params["decision"] if kind == "decision" else 0.,
                                     rank_weight=params["rank"], temperature=params["temperature"])
        optim.zero_grad()
        error.backward()
        optim.step()
        if development is None or step not in {35, 70, 120, 200, 300, 500, params["updates"]}:
            continue
        val = policy_loss(model, kind, development, params["cost"])["loss"]
        history.append(dict(step=step, loss=val))
        if trial is not None:
            trial.set_user_attr("curve", history)
            trial.report(val, step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if val < best:
            best, best_step = val, step
            best_state = {k: p.detach().cpu().clone() for k, p in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model.eval(), best, best_step


def search(kind, fit_data, dev_data, trials):
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=41),
                               pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=70))

    def objective(t):
        params = dict(lr=t.suggest_float("lr", 1e-4, .01, log=True),
                      wd=t.suggest_float("wd", 1e-6, .03, log=True),
                      width=t.suggest_categorical("width", [32, 48, 96]),
                      updates=t.suggest_categorical("updates", [70, 200, 500]),
                      rank=t.suggest_categorical("rank", [0., .1, .15, .5, 1.]),
                      decision=t.suggest_categorical("decision", [.1, .3, .5, 1.]) if kind == "decision" else 0.,
                      temperature=t.suggest_categorical("temperature", [.15, .35, .7]) if kind == "decision" else .35,
                      cost=t.suggest_categorical("cost", [0., .01, .04, .1]))
        t.set_user_attr("config", params)
        _, best, step = fit(kind, fit_data, params, development=dev_data, trial=t)
        t.set_user_attr("best_step", step)
        return best

    study.optimize(objective, n_trials=trials)
    best = study.best_trial
    return dict(config=dict(best.user_attrs["config"], updates=best.user_attrs["best_step"]), objective=best.value,
                trials=[dict(state=t.state.name, value=t.value, **t.user_attrs) for t in study.trials])


def fit_context(shared, split, winners, device):
    models, cases = {}, None
    for family in ("twinmask", "mlm", "structure", "text_link", "gae"):
        selected = winners[family]
        model, (make, nodes, _), _ = train(shared, split, 97, family, Config(**selected["config"]),
                                          device, selected["steps"], fixed_steps=True)
        models[family] = model
        if cases is None:
            cases = make, nodes
    return models, cases


def mixture_search(dev, direction, temperatures, chosen, alternatives):
    trials = []
    for steps, reg in ((35, 0.), (70, .001), (150, .01)):
        stacking = fit_stacking(dev, steps, reg)
        for t in (.01, .05, .15, .5, 1.):
            result = evaluate_expected(dev, direction, temperatures, chosen, alternatives, stacking, 87, t)
            trials.append(dict(steps=steps, weight_decay=reg, temperature=t,
                               mixture=float(np.mean([r["utility_mixture"]["loss"] for r in result])),
                               stacking=float(np.mean([r["stacking"]["loss"] for r in result]))))
    return dict(mixture=min(trials, key=lambda x: x["mixture"])["temperature"],
                stacking=min(trials, key=lambda x: x["stacking"]), trials=trials)


@torch.no_grad()
def calibration_scores(test, names, temperatures, direction):
    truth = torch.cat([r[3] for r in test])
    output = {}
    for label, temps in (("calibrated", temperatures), ("unscaled", dict.fromkeys(temperatures, 1.))):
        p = torch.cat([teacher_prob(row[4], names, temps, direction) for row in test])
        if direction == "text":
            confidence, predicted = p.max(1)
            event = (predicted == truth).float()
            brier = (p-F.one_hot(truth, num_classes=p.shape[-1])).square().sum(1)
        else:
            confidence, event = p[:, 1], truth.float()
            brier = (confidence-event).square()
        bins = (confidence*10).long().clamp(max=9)
        error = sum(float(subset.float().mean()*(confidence[subset].mean()-event[subset].mean()).abs())
                    for k in range(10) if (subset := bins == k).any())
        output[label] = dict(nll=float(F.nll_loss(p.clamp_min(1e-9).log(), truth)), brier=float(brier.mean()), ece=error)
    return output


def calibration_only(name, device):
    winners = json.loads(Path(f"results/research_twinmask_v6/{name}_search.json").read_text())
    shared = protocol(name, SEARCH_SPLIT)
    results = []
    for split in LOCKED:
        models, (make, nodes) = fit_context(locked_protocol(shared, split), split, winners, device)
        dev_cases, test_cases = [stage_cases(make, nodes, split, device, k) for k in (1, 2)]
        result = {"split": split}
        for d in ("text", "edge"):
            dev, _ = predictions(models, dev_cases, d)
            test, _ = predictions(models, test_cases, d)
            temperatures, teacher, _ = calibrate_teachers(dev, d)
            result[d] = dict(teacher=teacher, temperature=temperatures,
                             scores=calibration_scores(test, teacher, temperatures, d))
        results.append(result)
    return results


def run(name, device, trials):
    winners = json.loads(Path(f"results/research_twinmask_v6/{name}_search.json").read_text())
    shared = protocol(name, SEARCH_SPLIT)
    models, (make, nodes) = fit_context(shared, SEARCH_SPLIT, winners, device)
    fit_cases, dev_cases = [stage_cases(make, nodes, SEARCH_SPLIT, device, k) for k in (0, 1)]
    # Schedule choice is made on development only, comparing the same audit targets.
    schedules = {h: temporal(models["twinmask"], dev_cases[1], 33, h)["free"][-1] for h in (1, 2, 4, 8)}
    chosen_horizon = min(schedules, key=lambda h: schedules[h]["token_ce"]/np.log(models["twinmask"].vocab)+
                         schedules[h]["edge_bce"]/np.log(2))
    output = dict(search_split=SEARCH_SPLIT, schedules=schedules, horizon=chosen_horizon, directions={}, locked=[])
    for d in ("text", "edge"):
        fit_result, fit_x = predictions(models, fit_cases, d)
        dev, dev_x = predictions(models, dev_cases, d)
        temperatures, teacher, alternatives = calibrate_teachers(dev, d)
        fit_data, dev_data = [tensors(rows, x, temperatures, teacher, d) for rows, x in ((fit_result, fit_x), (dev, dev_x))]
        output["directions"][d] = dict(policies={kind: search(kind, fit_data, dev_data, trials) for kind in FAMILIES},
                                       mixture=mixture_search(dev, d, temperatures, teacher, alternatives))
        print(name, d, "selector search complete", flush=True)
    for split in LOCKED:
        models, (make, nodes) = fit_context(locked_protocol(shared, split), split, winners, device)
        fit_cases, dev_cases, test_cases = [stage_cases(make, nodes, split, device, k) for k in (0, 1, 2)]
        item = dict(split=split, temporal=temporal(models["twinmask"], test_cases[1], 33, chosen_horizon), directions={})
        for d in ("text", "edge"):
            fit_result, fit_x = predictions(models, fit_cases, d)
            dev, _ = predictions(models, dev_cases, d)
            test, test_x = predictions(models, test_cases, d)
            temperatures, teacher, alternatives = calibrate_teachers(dev, d)
            train_data, test_data = [tensors(rows, x, temperatures, teacher, d) for rows, x in
                                     ((fit_result, fit_x), (test, test_x))]
            defaults = output["directions"][d]
            outcome = {}
            for kind, selected in defaults["policies"].items():
                config = selected["config"]
                model, _, _ = fit(kind, train_data, config)
                outcome[kind] = policy_loss(model, kind, test_data, config["cost"])
            mixture = defaults["mixture"]
            stack = fit_stacking(dev, mixture["stacking"]["steps"], mixture["stacking"]["weight_decay"])
            item["directions"][d] = dict(selectors=outcome, temperatures=temperatures, teacher=teacher,
                                         expected=evaluate_expected(test, d, temperatures, teacher, alternatives,
                                                                    stack, 87, mixture["mixture"]))
        output["locked"].append(item)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["cora", "pubmed"])
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calibration-only", action="store_true")
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for name in args.datasets:
        if args.calibration_only:
            save_json(calibration_only(name, torch.device(args.device)), "research_twinmask_v6", name+"_calibration.json")
            continue
        data = run(name, torch.device(args.device), args.trials)
        save_json(data, "research_twinmask_v6", name+"_selectors.json")


if __name__ == "__main__":
    main()
