import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import scipy
import torch

from .experiment import trial
from .graph import ROOT, prepare


def save(path, data):
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1, allow_nan=False), encoding="utf8")
    tmp.replace(path)


def run(name, out, device, repeat):
    begin = time.perf_counter()
    graph = prepare(name, device)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.json"
    package = Path(__file__).parent
    files = [Path(ROOT, "data", "processed", f"{name}.pt"),
             Path(ROOT, "checkpoints", f"emb_{name}_joint_s0.pt")]
    signature = dict(code=hashlib.sha256(b"".join((package / f).read_bytes() for f in
        ("graph.py", "experiment.py", "acquire.py", "__main__.py"))).hexdigest(),
        inputs=[[p.name, p.stat().st_size, p.stat().st_mtime_ns] for p in files],
        torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__,
        device=torch.cuda.get_device_name() if device == "cuda" else platform.processor())
    data = json.loads(path.read_text()) if path.exists() else dict(
        signature=signature, dataset=name, preparation_seconds=time.perf_counter()-begin, runs={})
    if data["signature"] != signature:
        raise ValueError(f"Inputs or implementation changed. Use a fresh --out directory for {name}.")
    rows = data["runs"]

    def series(stage, label, seeds, method="combined", **settings):
        values = []
        for seed in seeds:
            key = f"{stage}/{label}/{seed}"
            if repeat or key not in rows:
                rows[key] = trial(graph, seed, stage, method, **settings)
                save(path, data)
            values.append(rows[key]["aulc"])
        return np.mean(values)

    weights = (0., .1, .3, .6)
    validation = {(g, b): series("validation", f"gamma-{g}_bonus-{b}", (11, 12, 13), gamma=g, bonus=b)
                  for g in weights for b in (.05, .15, .3)}
    # Tune the positive-weight family, then independently compare to its nested null.
    gamma, bonus = max((key for key in validation if key[0] > 0), key=validation.get)
    _, null_bonus = max((key for key in validation if key[0] == 0), key=validation.get)
    data.update(gamma=gamma, bonus=bonus, null_bonus=null_bonus, validation_winner=max(validation, key=validation.get))
    for label, method, selector, weight, diversity, view in (
        ("random", "random", "top", 0, bonus, "both"), ("entropy", "entropy", "top", 0, bonus, "both"),
        ("disagreement", "disagreement", "top", 0, bonus, "both"),
        ("no_cf", "combined", "greedy", 0, bonus, "both"),
        ("tuned_no_cf", "combined", "greedy", 0, null_bonus, "both"),
        ("cf", "combined", "greedy", gamma, bonus, "both"),
        ("cf_top", "combined", "top", gamma, bonus, "both"),
        ("cf_milp", "combined", "milp", gamma, bonus, "both"),
        ("single_cf", "combined", "greedy", gamma, bonus, "single"),
        ("raw_ensemble_cf", "combined", "greedy", gamma, bonus, "raw_ensemble")):
        series("test", label, range(41, 46), method, gamma=weight, selector=selector, bonus=diversity, view_mode=view)
    save(path, data)
    print(f"{name} complete, selected positive gamma={gamma}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer", "pubmed", "wikics"])
    parser.add_argument("--out", type=Path, default=Path(ROOT, "results", "cf_dive"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--repeat", action="store_true", help="Recompute existing seeds in this output directory")
    args = parser.parse_args()
    torch.set_num_threads(2)
    for dataset in args.datasets:
        run(dataset, args.out, args.device, args.repeat)
