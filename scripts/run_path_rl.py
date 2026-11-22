"""Job queue for the TwinMask-Path experiments (needs the phase-1 `joint` checkpoints). Skips finished jobs.

  python -m scripts.run_path_rl            # all jobs
  python -m scripts.run_path_rl cora       # one dataset
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = dict(os.environ, PYTHONUTF8="1", TOKENIZERS_PARALLELISM="false")
DATASETS = ["cora", "citeseer", "pubmed", "wikics", "instagram"]  # homophilous -> low homophily
JOBS = [  # (mode, rho, extra args)
    ("generate", 0.5, ()),
    ("teacher", 0.5, ()),
    ("generate", 1.0, ("--no_policy",)),
]


def main(datasets):
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    for ds in datasets:
        for mode, rho, extra in JOBS:
            tag = f"{ds}_rho{rho}_N8_{mode}_s0"
            if os.path.exists(os.path.join(ROOT, "results", "path_rl", tag + ".json")):
                print("skip", tag, flush=True); continue
            print(f"=== {time.strftime('%X')} {tag}", flush=True)
            cmd = [sys.executable, "-m", "twinmask.path_rl", "--dataset", ds, "--mode", mode, "--rho", str(rho), *extra]
            with open(os.path.join(ROOT, "logs", f"path_rl_{tag}.log"), "w", encoding="utf-8") as f:
                rc = subprocess.call(cmd, cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT)
            print("  rc", rc, flush=True)
    print("ALL DONE", time.strftime("%X"), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or DATASETS)
