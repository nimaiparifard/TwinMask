"""Compare frozen TwinMask-S probes and the CF-DIVE backbone on DIVERGE's label splits."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from twinmask.common import _lr
from twinmask.data import ROOT, load

from .experiment import fit
from .graph import prepare


def evaluate(dataset, split_data):
    data = load(dataset)
    views, _, _, _, y, classes = prepare(dataset, "cuda" if torch.cuda.is_available() else "cpu")
    hidden = torch.load(Path(ROOT, "checkpoints", f"emb_{dataset}_joint_s0.pt"),
                        map_location="cpu", weights_only=True).float().numpy()
    records = []
    for protocol, seeds in (("supervised", range(42, 46)), ("semi_public", range(3))):
        for seed in seeds:
            masks = (split_data(len(y), device="cpu", seed=seed) if protocol == "supervised"
                     else [data[k] for k in ("train_mask", "val_mask", "test_mask")])
            train, valid, test = [np.flatnonzero(m.numpy()) for m in masks]
            assert len(np.unique(np.r_[train, valid, test])) == len(train)+len(valid)+len(test)
            probs = [fit(v, y, train, classes, seed*100+i)[1] for i, v in enumerate(views)]
            pred = torch.stack(probs).mean(0)[test].argmax(1).cpu()
            acc, _ = _lr(hidden, y.numpy(), train, test, seed)
            records.append(dict(protocol=protocol, seed=seed, train=len(train), test=len(test),
                                cf_backbone=float((pred == y[test]).double().mean()), twinmask_probe=acc))
    return records


if __name__ == "__main__":
    import importlib.util
    parser = argparse.ArgumentParser()
    parser.add_argument("--diverge", type=Path, default=Path(ROOT).parent / "DIVERGE")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("diverge_split", args.diverge / "common/dataloader.py")
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        results = {ds: evaluate(ds, loader.re_split_data) for ds in ("cora", "citeseer")}
    path = Path(ROOT, "results/cf_dive/fixed_labels/accuracy.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf8")
    for ds, rows in results.items():
        for protocol in ("supervised", "semi_public"):
            subset = [r for r in rows if r["protocol"] == protocol]
            print(ds, protocol, subset[0]["train"], subset[0]["test"],
                  {key: np.round(100*np.mean([r[key] for r in subset]), 2)
                   for key in ("cf_backbone", "twinmask_probe")})
