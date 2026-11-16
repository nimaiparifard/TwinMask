"""Re-evaluate saved TwinMask checkpoints whose result JSON predates HeaRT / supervised-NC (no retraining)."""
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = dict(os.environ, PYTHONUTF8="1", TOKENIZERS_PARALLELISM="false")

for tag in ("phase1", "phase3"):
    for f in sorted(glob.glob(os.path.join(ROOT, "results", tag, "*.json"))):
        r = json.load(open(f))
        need = ("heart" not in r and r["variant"] != "text_only") or "sup_acc" not in r.get("nc", {})
        if not need:
            continue
        print("re-eval", tag, r["dataset"], r["variant"], r["seed"], flush=True)
        extra = ["--max_nc", "20000"] if r["dataset"] == "arxiv" else []
        log = os.path.join(ROOT, "logs", f"reeval_{tag}_{r['dataset']}_{r['variant']}_s{r['seed']}.log")
        with open(log, "w", encoding="utf-8") as lf:
            rc = subprocess.call([sys.executable, "-m", "twinmask.tm", "--dataset", r["dataset"], "--variant", r["variant"],
                                  "--seed", str(r["seed"]), "--k", "12", "--max_pairs", "1000", "--tag", tag + "_reeval",
                                  "--eval_only", *extra], cwd=ROOT, env=ENV, stdout=lf, stderr=subprocess.STDOUT)
        new = os.path.join(ROOT, "results", tag + "_reeval", os.path.basename(f))
        if rc == 0 and os.path.exists(new):
            n = json.load(open(new))
            n.update(train_hist=r.get("train_hist"), train_sec=r.get("train_sec"), steps=r.get("steps"))
            json.dump(n, open(f, "w"), indent=2)
            os.remove(new)
        print("  rc", rc, flush=True)
