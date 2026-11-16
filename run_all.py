"""Sequential job queue for phases 1-3 (phase 0: python -m twinmask.phase0). Skips finished jobs."""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
ENV = dict(os.environ, PYTHONUTF8="1", TOKENIZERS_PARALLELISM="false")
STEPS = int(os.environ.get("STEPS", 2000))
os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)


def run(ds, v, s=0, steps=STEPS, tag="phase1", extra=()):
    out = os.path.join(ROOT, "results", tag, f"{ds}_{v}_s{s}.json")
    if os.path.exists(out):
        print("skip", ds, v, s, flush=True); return
    print(f"=== {time.strftime('%X')} {tag} {ds} {v} seed={s} steps={steps}", flush=True)
    log = os.path.join(ROOT, "logs", f"{tag}_{ds}_{v}_s{s}.log")
    cmd = [sys.executable, "-m", "twinmask.tm", "--dataset", ds, "--variant", v, "--seed", str(s),
           "--steps", str(steps), "--k", "12", "--max_pairs", "1000", "--tag", tag, *extra]
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.call(cmd, cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT)
    print("  rc", rc, flush=True)


if __name__ == "__main__":
    # Phase 1: main model + modality ablations
    for ds in ("cora", "citeseer", "pubmed"):
        for v in ("joint", "edge_only", "text_only", "tied"):
            run(ds, v)
    # Phase 2: topology ablations, extra seed, TSMI datasets, analysis
    for ds in ("cora", "pubmed"):
        for v in ("static", "notopo"):
            run(ds, v)
    run("cora", "joint", 1)
    for ds in ("wikics", "instagram", "reddit"):
        run(ds, "joint")
    with open(os.path.join(ROOT, "logs", "phase2.log"), "w", encoding="utf-8") as f:
        subprocess.call([sys.executable, "-m", "twinmask.phase2", "--datasets", "cora", "citeseer", "pubmed",
                         "wikics", "instagram", "reddit"], cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT)
    # Phase 3: ogbn-arxiv
    run("arxiv", "joint", 0, 8000, "phase3", ("--max_nc", "20000"))
    run("arxiv", "edge_only", 0, 8000, "phase3", ("--max_nc", "20000"))
    subprocess.call([sys.executable, "-m", "twinmask.report"], cwd=ROOT, env=ENV)
    print("ALL DONE", time.strftime("%X"), flush=True)
