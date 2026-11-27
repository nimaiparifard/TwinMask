"""Run the corrected V4 protocol on frozen legacy checkpoints."""
import argparse
from pathlib import Path

import torch

from scripts.research_twinmask import Case, MASK, OTHER, PAD, SparseTwinMask, corruption
from twinmask.audit_protocol import graph_split
from twinmask.common import save_json


RATES = ((.25, .75), (.5, .5), (.75, .25), (.9, .9))


def audited_contexts(name, seed, device, matched, audit):
    checkpoint = torch.load(Path("checkpoints/research_twinmask_v2")/f"{name}_{seed}.pt",
                            map_location="cpu", weights_only=False)
    order = checkpoint["edge_split"]
    n_edges = len(order)
    data, base, positives, negatives, details = graph_split(
        name, seed, legacy_order=order, original_fit_negative_count=n_edges//5+2*(n_edges//10))
    audit.update(details)
    audit["context"] = "matched" if matched else "growing"
    audit["test_base_edges"] = base.shape[1] if matched else n_edges
    vocabulary = checkpoint["vocabulary"].to(device)
    model = SparseTwinMask(len(vocabulary)+3, 16, checkpoint["model"]["embed.weight"].shape[1]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.requires_grad_(False).eval()
    raw = data["tokens"][:, 1:17].long().to(device)
    mapping = torch.full((50265,), OTHER, device=device, dtype=torch.long)
    mapping[vocabulary] = torch.arange(3, len(vocabulary)+3, device=device)
    ids = mapping[raw].masked_fill((raw <= 2) | (raw == 50264), PAD)
    n = len(ids)
    train, fit, dev, test = checkpoint["node_split"].to(device).split([n//2, n//5, n//10, n-n//2-n//5-n//10])
    context = ids.masked_fill(ids > OTHER, MASK)
    context[train] = ids[train]
    rng = torch.Generator(device=device).manual_seed(seed+702)
    bases = [base, base, base if matched else data["train_pos"]]
    for rates in RATES:
        cases = []
        for nodes, p, q, graph in zip((fit, dev, test), positives, negatives, bases):
            pairs = torch.cat((p, q), 1).to(device)
            labels = (torch.arange(pairs.shape[1], device=device) < p.shape[1]).long()
            ix = torch.randperm(len(labels), device=device, generator=rng)
            backdrop, eligible = context.clone(), torch.zeros_like(ids, dtype=torch.bool)
            backdrop[nodes] = ids[nodes]
            eligible[nodes] = ids[nodes] > OTHER
            cases.append(Case(backdrop, ids, corruption(eligible, rates[0], rng), pairs[:, ix], labels[ix],
                              corruption(torch.ones_like(labels).bool(), rates[1], rng), graph.to(device), rates))
        if matched:
            assert all(torch.equal(case.base, cases[0].base) for case in cases)
        yield model, *cases, rng


def main():
    from scripts.research_twinmask_v4 import run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer", "pubmed"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    for name in args.datasets:
        for matched in (True, False):
            rows = []
            context = "matched" if matched else "growing"
            for seed in args.seeds:
                audit = {}
                source = audited_contexts(name, seed, torch.device(args.device), matched, audit)
                row = run(name, seed, torch.device(args.device), args.limit, args.steps, 4, extra=True,
                          source=source, audit=True, tag="v41_"+context)
                row["audit"] = audit
                rows.append(row)
                print(name, context, seed, round(row["seconds"], 2), flush=True)
            save_json(rows, "research_twinmask_v41", f"{name}_{context}.json")


if __name__ == "__main__":
    main()
