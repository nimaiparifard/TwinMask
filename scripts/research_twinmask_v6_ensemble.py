"""Tune embedding diffusion and its matched classifier controls on development labels."""
import argparse

import optuna
import torch
from torch.nn import functional as F

from scripts.research_ensemble import SAGE, generate, metrics, noise_view, schedule
from twinmask.common import save_json, set_seed
from twinmask.data import load


def optimize(fn, trials):
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=413),
                               pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=20))
    study.optimize(fn, n_trials=trials)
    valid = [t for t in study.trials if t.state.name == "COMPLETE"]
    return min(valid, key=lambda t: t.value), [
        dict(parameters=t.params, state=t.state.name, value=t.value, intermediate=t.intermediate_values)
        for t in study.trials]


def run(name, device, trials):
    data = load(name)
    x, y, pairs = (data[k].to(device) for k in ("emb", "y", "train_pos"))
    edges = torch.cat((pairs, pairs.flip(0)), 1)
    order = torch.randperm(len(x), generator=torch.Generator(device=device).manual_seed(900), device=device)
    fit, dev, test = order.split([len(x)*3//5, len(x)//5, len(x)-len(x)*3//5-len(x)//5])
    x = (x-x[fit].mean(0))/x[fit].std(0).clamp_min(1e-6)
    alpha = schedule(device)

    def fit_noise(params, trial=None):
        set_seed(71)
        denoiser = SAGE(x.shape[1], x.shape[1], True, hidden=params["hidden"], dropout=params["dropout"]).to(device)
        optim = torch.optim.AdamW(denoiser.parameters(), lr=params["lr"], weight_decay=params["wd"])
        error = None
        for step in range(1, params["updates"]+1):
            t = int(torch.randint(1, 101, ()))
            eps = torch.randn_like(x)
            rate = params["dropedge"]
            observed = pairs[:, torch.rand(pairs.shape[1], device=device) >= rate]
            guess = denoiser(noise_view(x, alpha[t], eps), torch.cat((observed, observed.flip(0)), 1),
                             x.new_tensor(t), rate)
            loss = F.mse_loss(guess[fit], eps[fit])
            optim.zero_grad()
            loss.backward()
            optim.step()
            if step % 20 and step != params["updates"]:
                continue
            denoiser.eval()
            with torch.no_grad():
                rng = torch.Generator(device=device).manual_seed(99)
                eps = torch.randn(x.shape, device=device, generator=rng)
                error = sum(float(F.mse_loss(denoiser(noise_view(x, alpha[t], eps), edges,
                                                      x.new_tensor(t))[dev], eps[dev])) for t in (10, 30, 70))/3
            denoiser.train()
            if trial is not None:
                trial.report(error, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        return denoiser.eval(), error

    def noise_objective(t):
        params = dict(hidden=t.suggest_categorical("hidden", [64, 128, 256]),
                      dropout=t.suggest_categorical("dropout", [0., .1, .2]),
                      lr=t.suggest_float("lr", .0001, .005, log=True),
                      wd=t.suggest_float("wd", 1e-6, .01, log=True),
                      dropedge=t.suggest_categorical("dropedge", [0., .1, .3]),
                      updates=t.suggest_categorical("updates", [40, 80, 120]))
        return fit_noise(params, t)[1]

    winner, noise_trials = optimize(noise_objective, trials)
    diffusion = fit_noise(winner.params)[0]
    views = {}

    def inputs(arm, params):
        if arm == "raw":
            return [x]*3
        key = (arm, params["start"], params["ddim_steps"], params["dropedge"])
        if key not in views:
            pool = []
            for seed in (948, 949, 950):
                reconstructed, noise = generate(diffusion, x, pairs, alpha, key[1], key[2], key[3], seed)
                if arm == "gaussian":
                    target = float(F.cosine_similarity(reconstructed[fit], x[fit]).mean())
                    low, high = 0., 16.
                    for _ in range(14):
                        mid = (low+high)/2
                        if float(F.cosine_similarity(x[fit]+mid*noise[fit], x[fit]).mean()) > target:
                            low = mid
                        else:
                            high = mid
                    reconstructed = x+((low+high)/2)*noise
                pool.append(reconstructed)
            views[key] = pool
        return views[key]

    def fit_head(arm, params, seed, trial=None):
        set_seed(seed)
        members = inputs(arm, params)
        head = SAGE(x.shape[1], int(y.max())+1, hidden=params["hidden"], dropout=params["dropout"]).to(device)
        optim = torch.optim.AdamW(head.parameters(), lr=params["lr"], weight_decay=params["wd"])
        val = None
        for step in range(1, params["updates"]+1):
            head.train()
            logits = head(members[step % len(members)], edges)[fit]
            optim.zero_grad()
            F.cross_entropy(logits, y[fit]).backward()
            optim.step()
            if step % 20 and step != params["updates"]:
                continue
            head.eval()
            with torch.no_grad():
                p = torch.stack([head(view, edges)[dev].softmax(-1) for view in members]).mean(0)
                val = float(F.nll_loss(p.clamp_min(1e-9).log(), y[dev]))
            if trial is not None:
                trial.report(val, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        return head.eval(), val

    outcomes = {}
    for arm in ("raw", "diffusion", "gaussian"):
        def objective(t):
            params = dict(hidden=t.suggest_categorical("hidden", [64, 128, 256]),
                          dropout=t.suggest_categorical("dropout", [0., .1, .2, .3]),
                          lr=t.suggest_float("lr", .0003, .01, log=True),
                          wd=t.suggest_float("wd", 1e-6, .01, log=True),
                          updates=t.suggest_categorical("updates", [40, 80, 120]))
            if arm != "raw":
                params.update(start=t.suggest_categorical("start", [10, 30, 50]),
                              ddim_steps=t.suggest_categorical("ddim_steps", [1, 4, 8]),
                              dropedge=t.suggest_categorical("dropedge", [0., .1, .3]))
            return fit_head(arm, params, 71, t)[1]
        best, head_trials = optimize(objective, trials)
        evaluated = []
        for seed in (83, 97, 109):
            head, _ = fit_head(arm, best.params, seed)
            with torch.no_grad():
                prediction = torch.stack([head(view, edges)[test].softmax(-1) for view in inputs(arm, best.params)]).mean(0)
            evaluated.append(metrics(prediction, y[test]))
        outcomes[arm] = dict(config=best.params, trials=head_trials, test=evaluated)
    controlled = []
    for seed in (83, 97, 109):
        head, _ = fit_head("raw", outcomes["raw"]["config"], seed)
        with torch.no_grad():
            policy = outcomes["diffusion"]["config"]
            controlled.append({arm: metrics(torch.stack([head(view, edges)[test].softmax(-1)
                                                         for view in inputs(arm, policy)]).mean(0), y[test])
                               for arm in ("diffusion", "gaussian")})
    return dict(split_seed=900, denoiser=winner.params, denoiser_trials=noise_trials, arms=outcomes,
                matched_raw_head=controlled)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer"])
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for name in args.datasets:
        save_json(run(name, torch.device(args.device), args.trials), "research_twinmask_v6", name+"_ensemble.json")
        print(name, "embedding ensemble complete", flush=True)


if __name__ == "__main__":
    main()
