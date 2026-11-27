"""Compare text lengths and vocabulary sizes on exactly the same held-out token positions."""
import argparse
from dataclasses import replace

import numpy as np
import torch

from scripts.research_twinmask import Case, MASK, OTHER, corruption
from scripts.research_twinmask_v5 import text_loss
from scripts.research_twinmask_v6 import (Config, LOCKED, SEARCH_SPLIT, build_cases, locked_protocol, protocol,
                                          stage_cases, train)
from twinmask.common import save_json


def evaluate(model, common, ids, nodes, device):
    values = []
    for j, initial in enumerate(common):
        rng = torch.Generator(device=device).manual_seed(2018+j)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        mask[:, :16] = initial.mask
        suffix = torch.zeros_like(mask)
        suffix[nodes[2], 16:] = ids[nodes[2], 16:] > OTHER
        mask |= corruption(suffix, initial.rates[0], rng)
        backdrop = ids.masked_fill(ids > OTHER, MASK)
        backdrop[nodes[0]], backdrop[nodes[2]] = ids[nodes[0]], ids[nodes[2]]
        case = Case(backdrop, ids, mask, initial.pairs, initial.labels, initial.edge_mask, initial.base, initial.rates)
        query = initial.x_index
        with torch.no_grad():
            logits, _ = model.predict(case, text_query=query, edge_query=initial.a_index[:0])
        values.append(dict(targets=len(query[0]), ce=float(text_loss(logits[0], ids[query]).mean()),
                           accuracy=float((logits[0][:, 3:].argmax(-1)+3 == ids[query]).float().mean())))
    return dict(ce=float(np.mean([v["ce"] for v in values])),
                accuracy=float(np.mean([v["accuracy"] for v in values])), panels=values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    from transformers import AutoTokenizer

    dataset = "cora"
    shared = locked_protocol(protocol(dataset, SEARCH_SPLIT), LOCKED[0])
    data, graph, positives, negatives, audit, splits = shared
    tok = AutoTokenizer.from_pretrained("hf_models/all-distilroberta-v1", local_files_only=True)
    tokens = tok(data["texts"], max_length=129, truncation=True, padding="max_length",
                 return_tensors="pt")["input_ids"]
    observed = tokens[splits[0], 1:17].flatten()
    counts = torch.bincount(observed[(observed > 2) & (observed < 50264)], minlength=50265)
    ranked = counts.argsort(descending=True, stable=True)
    fixed = ranked[counts[ranked] > 0][:4096]
    shared = (dict(data, tokens=tokens, capacity_vocabulary=fixed), graph, positives, negatives, audit, splits)
    make, _, _, nodes, _ = build_cases(shared, LOCKED[0], Config(vocab_limit=256), device)
    common_cases = stage_cases(make, nodes, LOCKED[0], device, 2)
    results = []
    for length, vocabulary in ((16, 256), (16, 1024), (16, 4096), (32, 1024), (64, 1024), (128, 1024)):
        config = replace(Config(), length=length, vocab_limit=vocabulary)
        for family in ("mlm", "sage"):
            model, (_, nodes, vocab), history = train(shared, LOCKED[0], 97, family, config, device, 100, fixed_steps=True)
            _, ids, _, _, _ = build_cases(shared, LOCKED[0], config, device)
            results.append(dict(family=family, length=length, vocabulary=vocab-3, config=vars(config),
                                test=evaluate(model, common_cases, ids, nodes, device), history=history))
        print("capacity", length, vocabulary, flush=True)
    save_json(results, "research_twinmask_v6", "cora_capacity.json")


if __name__ == "__main__":
    main()
