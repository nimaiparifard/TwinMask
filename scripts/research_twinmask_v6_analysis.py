"""Paired convergence, optimization interventions and hard-negative diagnostics."""
import argparse
import json
from dataclasses import replace

import numpy as np
import torch

from scripts.research_twinmask_v6 import Config, LOCKED, SEARCH_SPLIT, locked_protocol, measure, protocol, stage_cases, train
from twinmask.audit_protocol import keys
from twinmask.common import save_json
from twinmask.data import sample_negatives


def alternate_negatives(shared, flavor, seed):
    data, base, ps, qs, audit, nodes = shared
    n = len(data["tokens"])
    warm = (torch.arange(n)[~torch.isin(torch.arange(n), data["cold_nodes"])] if "cold_nodes" in data else torch.arange(n))
    raw = data.get("all_true_edges")
    if raw is None and "name" in data:
        from twinmask.audit_protocol import reference
        raw = reference(data["name"], data)
    if raw is None:
        raw = torch.cat((base, *ps), 1)
    forbidden = keys(raw, n) | set.union(*(keys(q, n) for q in qs))
    rng = np.random.default_rng(seed)
    neighborhoods = [[] for _ in range(n)]
    for a, b in base.T.tolist():
        neighborhoods[a].append(b)
        neighborhoods[b].append(a)
    bins = np.array([int(np.log2(len(nbr)+1)) for nbr in neighborhoods])
    by_bin = {value: warm.numpy()[bins[warm] == value] for value in np.unique(bins)}
    candidates = []
    pivots = [i for i, nbr in enumerate(neighborhoods) if len(nbr) >= 2]
    for a, b in ps[-1].T.tolist():
        proposals = []
        if flavor == "degree":
            proposals = zip(rng.choice(by_bin[bins[a]], 128), rng.choice(by_bin[bins[b]], 128))
        else:
            proposals = (rng.choice(neighborhoods[root], 2, replace=False)
                         for root in rng.choice(pivots, 128))
        for u, v in proposals:
            u, v = sorted((int(u), int(v)))
            if u != v and u*n+v not in forbidden:
                candidates.append((u, v))
                forbidden.add(u*n+v)
                break
    specific = len(candidates)
    tail = sample_negatives(n, len(ps[-1].T)-specific, forbidden, rng, warm.numpy())
    output = torch.cat((torch.tensor(candidates, dtype=torch.long).reshape(-1, 2).T, tail), 1)
    assert len(keys(output, n)) == output.shape[1] and not (keys(output, n) & keys(raw, n))
    return (data, base, ps, (*qs[:2], output), audit, nodes), specific


def run(name, device, budget, convergence):
    search = protocol(name, SEARCH_SPLIT)
    locked = locked_protocol(search, LOCKED[0])
    base = Config()
    winners = json.loads(open(f"results/research_twinmask_v6/{name}_search.json").read())
    tuned = Config(**winners["sage"]["config"])
    # Keep the transfer interventions paired on capacity, initials, masks and update counts.
    choices = (
        ("independent", base, "weighted", (0., 0.)),
        ("structure_guided", base, "weighted", (1., 0.)),
        ("text_guided", base, "weighted", (0., 1.)),
        ("bidirectional", base, "weighted", (1., 1.)),
        ("stop_gradient", base, "stop_gradient", (1., 1.)),
        ("pcgrad", base, "pcgrad", (1., 1.)),
        ("gradnorm_proxy", base, "gradnorm", (1., 1.)),
        ("cagrad_grid", base, "cagrad", (1., 1.)),
        ("selected_loss_weight", replace(base, edge_weight=tuned.edge_weight, normalize=tuned.normalize), "weighted", (1., 1.)),
        ("structural_statistics", replace(base, structural_stats=True), "weighted", (1., 1.)),
        ("small_transformer", replace(base, text_encoder="transformer"), "weighted", (1., 1.)),
    )
    rows = {}
    for name_, config, method, direction in choices:
        model, (make, nodes, _), trace = train(locked, LOCKED[0], 97, "sage", config, device, budget,
                                              fixed_steps=True, method=method, directions=direction, diagnostic=True)
        row = dict(config=vars(config), history=trace, test=measure(model, stage_cases(make, nodes, LOCKED[0], device, 2)))
        if name_ == "bidirectional":
            for flavor in ("degree", "two_hop"):
                alternate, count = alternate_negatives(locked, flavor, LOCKED[0]+19+(flavor == "two_hop"))
                # The alternate only changes test negative pairs.
                from scripts.research_twinmask_v6 import build_cases
                test_make, _, _, test_nodes, _ = build_cases(alternate, LOCKED[0], config, device)
                row[flavor] = dict(count=count, metrics=measure(model, stage_cases(test_make, test_nodes,
                                                                                  LOCKED[0], device, 2)))
        rows[name_] = row
        print(name, name_, flush=True)
    traces = {}
    if convergence:
        for family in ("mlm", "sage", "structure"):
            _, _, traces[family] = train(search, SEARCH_SPLIT, 71, family, base, device, 1000,
                                         fixed_steps=True, convergence=True, diagnostic=True)
            print(name, "convergence", family, flush=True)
    return dict(split=LOCKED[0], arms=rows, convergence=traces)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["cora", "pubmed"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-convergence", action="store_true")
    parser.add_argument("--gates-only", action="store_true")
    args = parser.parse_args()
    for name in args.datasets:
        if args.gates_only:
            device = torch.device(args.device)
            shared = locked_protocol(protocol(name, SEARCH_SPLIT), LOCKED[0])
            cfg = replace(Config(), learned_gates=True)
            model, (make, nodes, _), history = train(shared, LOCKED[0], 97, "sage", cfg, device, args.steps,
                                                     fixed_steps=True, diagnostic=True)
            save_json(dict(history=history, test=measure(model, stage_cases(make, nodes, LOCKED[0], device, 2))),
                      "research_twinmask_v6", name+"_gates.json")
            continue
        outcome = run(name, torch.device(args.device), args.steps, not args.no_convergence)
        save_json(outcome, "research_twinmask_v6", name+"_analysis.json")


if __name__ == "__main__":
    main()
