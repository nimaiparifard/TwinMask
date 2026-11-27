"""Expand equal-budget TPE studies, verify longer training, and isolate loss weighting."""
import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import optuna
import torch

from scripts.research_twinmask_v6 import Config, SEARCH_SPLIT, locked_eval, optimize, protocol, train
from twinmask.common import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["cora", "pubmed"])
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    device = torch.device(args.device)
    for dataset in args.datasets:
        initial = json.loads(Path(f"results/research_twinmask_v6/{dataset}_search.json").read_text())
        shared = protocol(dataset, SEARCH_SPLIT)
        profiles = {}
        for family in ("mlm", "structure", "sage", "twinmask"):
            extended = optimize(shared, family, args.trials, 64, device, replay=initial[family]["trials"])
            top = sorted((r for r in extended["trials"] if "selection_score" in r),
                         key=lambda r: r["selection_score"])[:2]
            longer = []
            for trial in top:
                config = Config(**trial["config"])
                curves = [train(shared, SEARCH_SPLIT, seed, family, config, device, 300, fixed_steps=True)[2]["curve"]
                          for seed in (71, 83)]
                candidates = [(sum(row["score"] for row in rows)/len(rows), rows[0]["step"])
                              for rows in zip(*curves)]
                average, step = min(candidates)
                longer.append(dict(config=trial["config"], steps=step, score=average, curves=curves))
            selected = min(longer, key=lambda r: r["score"])
            profiles[family] = dict(config=selected["config"], steps=selected["steps"], screening=extended,
                                    confirmation=longer)
            print(dataset, family, "expanded", selected["steps"], flush=True)
            save_json(profiles, "research_twinmask_v6", dataset+"_deep_search.json")
        save_json(locked_eval(dataset, profiles, device, tag="_deep"), "research_twinmask_v6", dataset+"_deep_locked.json")
        settings = []
        for normalize in (False, True):
            for weight in (.1, .25, .5, 1., 2., 4., 10.):
                config = replace(Config(), edge_weight=weight, normalize=normalize)
                _, _, trace = train(shared, SEARCH_SPLIT, 71, "sage", config, device, 100,
                                    fixed_steps=True, diagnostic=True)
                settings.append(dict(config=asdict(config), history=trace))
        save_json(settings, "research_twinmask_v6", dataset+"_weight_sweep.json")


if __name__ == "__main__":
    main()
