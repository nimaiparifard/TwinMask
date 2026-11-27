"""Tune the original V2 token/edge architecture and V3/V4.1 local selector families."""
import argparse
from dataclasses import asdict

import optuna
import torch

from scripts.research_twinmask import MASK, PAD, UNKNOWN, SparseTwinMask
from scripts.research_twinmask_v3 import MessagePanel
from scripts.research_twinmask_v4 import bundle
from scripts.research_twinmask_v41 import audited_contexts
from scripts.research_twinmask_v6 import Config, LOCKED, SEARCH_SPLIT, locked_protocol, measure, protocol, stage_cases, train
from scripts.research_twinmask_v6_selectors import fit, policy_loss, search
from twinmask.common import save_json


class OriginalTwin(SparseTwinMask):
    def __init__(self, vocab, config):
        super().__init__(vocab, config.length, config.width)
        self.vocab = vocab
        self.gate_logits = None

    def predict(self, case, state=None, *, text_query=None, edge_query=None, clocks=None):
        state = case.initial() if state is None else state
        ids, edges, _ = state
        query = case.x_index if text_query is None else text_query
        candidate = case.a_index if edge_query is None else edge_query
        tx, ta = case.clocks(state) if clocks is None else clocks
        both = self(ids, case.pairs, edges, case.base, query, candidate, tx, ta)
        x, _ = self(ids, case.pairs, torch.full_like(edges, UNKNOWN), case.base[:, :0], query, candidate[:0], tx, ta)
        _, a = self(ids.masked_fill(ids != PAD, MASK), case.pairs, edges, case.base,
                    tuple(i[:0] for i in query), candidate, tx, ta)
        return (both[0], x), (both[1], a)


def v2(name, trials, device):
    shared = protocol(name, SEARCH_SPLIT)
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=82, n_startup_trials=4),
                               pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=32))
    factory = lambda _make, vocab, cfg, _family: OriginalTwin(vocab, cfg).to(device)

    def objective(t):
        config = Config(lr=t.suggest_float("lr", .0002, .01, log=True),
                        wd=t.suggest_float("wd", 1e-6, .03, log=True),
                        width=t.suggest_categorical("width", [32, 48, 64, 96]),
                        edge_weight=t.suggest_float("edge_weight", .1, 10., log=True),
                        rollout=t.suggest_categorical("rollout", [0., .25, .5]),
                        modality_dropout=t.suggest_categorical("modality_dropout", [.0, .2, .4]))
        t.set_user_attr("config", asdict(config))
        _, _, trace = train(shared, SEARCH_SPLIT, 71, "twinmask", config, device, 64,
                            trial=t, model_factory=factory)
        t.set_user_attr("step", trace["best_step"])
        return min(row["score"] for row in trace["curve"])

    study.optimize(objective, n_trials=trials)
    best = study.best_trial
    results = []
    for split in LOCKED:
        locked = locked_protocol(shared, split)
        model, (make, nodes, _), stats = train(locked, split, 97, "twinmask", Config(**best.user_attrs["config"]),
                                              device, best.user_attrs["step"], fixed_steps=True, model_factory=factory)
        results.append(dict(split=split, test=measure(model, stage_cases(make, nodes, split, device, 2)),
                            history=stats))
    return dict(config=best.user_attrs["config"], steps=best.user_attrs["step"], test=results,
                trials=[dict(state=t.state.name, score=t.value, **t.user_attrs) for t in study.trials])


def legacy_selector(name, trials, device):
    directions = {direction: [[], [], []] for direction in ("text", "edge")}
    with torch.no_grad():
        for model, fit_case, dev, test, rng in audited_contexts(name, 42, device, True, {}):
            for direction, partitions in directions.items():
                for k, case in enumerate((fit_case, dev, test)):
                    panel = MessagePanel(model, case, direction, 160, rng)
                    result = bundle(panel)
                    partitions[k].append((result[1]["aware"], panel.valid, result[2], result[3], result[4]))
    output = {}
    for d, partitions in directions.items():
        arrays = []
        for rows in partitions:
            x, v, m, a, loss = [torch.cat([row[i] for row in rows], dim=0 if i < 2 else 1) for i in range(5)]
            arrays.append((x, v, m, a, loss, torch.zeros_like(loss)))
        fit_data, dev, test = arrays
        arms = {}
        for kind in ("loo", "shapley", "set", "decision"):
            best = search(kind, fit_data, dev, trials)
            fitted, _, _ = fit(kind, fit_data, best["config"])
            arms[kind] = dict(best=best, test=policy_loss(fitted, kind, test, best["config"]["cost"]))
        output[d] = arms
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["cora"])
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    device = torch.device(args.device)
    for name in args.datasets:
        save_json(dict(denoiser=v2(name, args.trials, device), local=legacy_selector(name, args.trials, device)),
                  "research_twinmask_v6", name+"_legacy.json")
        print(name, "legacy stage tuning complete", flush=True)


if __name__ == "__main__":
    main()
