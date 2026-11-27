"""Reference-backed splits for the V4.1 audit and V5 evaluation."""
from hashlib import sha256
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from twinmask.data import DIVERGE_DIR, ROOT, build_split, load, sample_negatives, undirected_unique


def keys(edges, n):
    return set((edges[0]*n+edges[1]).tolist())


def reference(name, data):
    path = Path(DIVERGE_DIR) / f"{name}.pt"
    graph = torch.load(path, map_location="cpu", weights_only=False)
    texts = [t.replace("\n", " ").strip() for t in graph.raw_texts]
    if texts != data["texts"] or graph.num_nodes != len(data["tokens"]):
        raise ValueError(f"Raw and cached node order differ for {name}")
    edges = undirected_unique(graph.edge_index, graph.num_nodes)
    truth = keys(edges, graph.num_nodes)
    pos = ("train_pos", "val_pos", "test_pos", "cold_pos")
    if not all(keys(data[k], graph.num_nodes) <= truth for k in pos):
        raise ValueError(f"Raw graph disagrees with cached positive edges for {name}")
    directory = Path(ROOT)/"data"/"references"
    directory.mkdir(exist_ok=True)
    torch.save(edges, directory/f"{name}.pt")
    return edges


def audit_legacy(data, truth, n):
    negative = {k: keys(data[k], n) for k in ("val_neg", "test_neg", "cold_neg")}
    seen = set.union(*(keys(data[k], n) for k in ("train_pos", "val_pos", "test_pos", "cold_pos")))
    return dict(omitted_positive_edges=len(truth-seen),
                legacy_false_negatives={k: len(v & truth) for k, v in negative.items()},
                legacy_duplicates={k: data[k].shape[1]-len(v) for k, v in negative.items()},
                legacy_validation_test_overlap=len(negative["val_neg"] & negative["test_neg"]))


def verify(positives, negatives, reference_keys, warm, n):
    for groups in (positives, negatives):
        for a, b in combinations(groups, 2):
            assert not keys(a, n) & keys(b, n)
        for pairs in groups:
            assert pairs.shape[1] == len(keys(pairs, n)) and (pairs[0] < pairs[1]).all()
            assert torch.isin(pairs, warm).all()
    for pairs in negatives:
        assert not keys(pairs, n) & reference_keys
    for pairs in positives:
        assert keys(pairs, n) <= reference_keys


def graph_split(name, seed, *, legacy_order=None, original_fit_negative_count=0):
    try:
        data = load(name)
    except FileNotFoundError:
        from transformers import AutoTokenizer

        data = build_split(name)
        tokenizer = AutoTokenizer.from_pretrained(Path(ROOT)/"hf_models"/"all-distilroberta-v1",
                                                   local_files_only=True)
        data["tokens"] = tokenizer(data["texts"], padding="max_length", truncation=True, max_length=17,
                                   return_tensors="pt")["input_ids"]
    n = len(data["tokens"])
    edges = reference(name, data)
    truth = keys(edges, n)
    warm = torch.arange(n)[~torch.isin(torch.arange(n), data["cold_nodes"])]
    rng = np.random.default_rng(seed)
    previous = set.union(*(keys(data[k], n) for k in
                           ("train_pos", "val_pos", "test_pos", "cold_pos", "cold_neg", "val_neg", "test_neg")))
    excluded = truth | previous
    if original_fit_negative_count:
        old = sample_negatives(n, original_fit_negative_count, previous, np.random.default_rng(seed))
        excluded |= keys(old, n)
    if legacy_order is not None:
        train = data["train_pos"][:, legacy_order]
        count = train.shape[1]//10
        earlier, fit, dev, foundation = train.split([train.shape[1]//5, count, count,
                                                      train.shape[1]-train.shape[1]//5-2*count], 1)
        base = torch.cat((earlier, foundation), 1)
        positives = (fit, dev, data["test_pos"])
    else:
        eligible = edges[:, torch.isin(edges, warm).all(0)]
        # Reserve old evaluation positives from the new test before drawing splits.
        historical = keys(torch.cat((data["val_pos"], data["test_pos"]), 1), n)
        fresh = torch.tensor([k not in historical for k in (eligible[0]*n+eligible[1]).tolist()])
        new = eligible[:, fresh]
        order = rng.permutation(new.shape[1])
        count = max(1, new.shape[1]//10)
        fit, dev, test, base = torch.from_numpy(order).split([count]*3+[len(order)-3*count])
        positives = tuple(new[:, ix] for ix in (fit, dev, test))
        base = new[:, base]
    negatives = []
    for items in positives:
        pairs = sample_negatives(n, items.shape[1], excluded, rng, warm.numpy())
        excluded |= keys(pairs, n)
        negatives.append(pairs)
    verify(positives, negatives, truth, warm, n)
    assert all(not keys(base, n) & keys(p, n) for p in positives)
    signature = sha256()
    for items in (*positives, base, *negatives):
        signature.update(items.contiguous().numpy().tobytes())
    audit = dict(reference_edges=edges.shape[1], warm_nodes=len(warm), split_hash=signature.hexdigest(),
                 base_edges=base.shape[1], positive_counts=[p.shape[1] for p in positives], **audit_legacy(data, truth, n))
    return data, base, positives, negatives, audit


def induced_raw_split(name, seed, size=6000):
    """Uniform node sampling from a new large graph, before partitioning any links."""
    from transformers import AutoTokenizer

    graph = torch.load(Path(DIVERGE_DIR)/f"{name}.pt", map_location="cpu", weights_only=False, mmap=True)
    original = undirected_unique(graph.edge_index, graph.num_nodes)
    directory = Path(ROOT)/"data"/"references"
    directory.mkdir(exist_ok=True)
    torch.save(original, directory/f"{name}.pt")
    rng = np.random.default_rng(seed)
    sampled = torch.from_numpy(np.sort(rng.choice(graph.num_nodes, min(size, graph.num_nodes), replace=False)))
    n = len(sampled)
    lookup = torch.full((graph.num_nodes,), -1, dtype=torch.long)
    lookup[sampled] = torch.arange(n)
    pairs = lookup[original]
    edges = pairs[:, (pairs >= 0).all(0)]
    truth = keys(edges, n)
    order = torch.from_numpy(rng.permutation(edges.shape[1]))
    count = edges.shape[1]//10
    groups = order.split([count]*3+[len(order)-3*count])
    fit, dev, test, base = [edges[:, i] for i in groups]
    positives = (fit, dev, test)
    negatives = sample_negatives(n, 3*count, truth, rng).split(count, 1)
    verify(positives, negatives, truth, torch.arange(n), n)
    assert all(not keys(base, n) & keys(p, n) for p in positives)
    tokenizer = AutoTokenizer.from_pretrained(Path(ROOT)/"hf_models"/"all-distilroberta-v1",
                                               local_files_only=True)
    tokens = tokenizer([graph.raw_texts[i] for i in sampled.tolist()], max_length=17, truncation=True,
                       padding="max_length", return_tensors="pt")["input_ids"]
    signature = sha256()
    for block in (sampled, base, *positives, *negatives):
        signature.update(block.numpy().tobytes())
    audit = dict(reference_edges=int(original.shape[1]), induced_edges=int(edges.shape[1]),
                 reference_nodes=graph.num_nodes, induced_nodes=n, split_hash=signature.hexdigest(),
                 base_edges=base.shape[1], positive_counts=[p.shape[1] for p in positives])
    return {"tokens": tokens}, base, positives, negatives, audit
