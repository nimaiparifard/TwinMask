"""Additional paired seeds with hyperparameters locked to the primary study."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure
from scipy.stats import t

from .__main__ import save
from .experiment import trial
from .graph import ROOT, prepare
from .report import summarize, table


def replicate(folder):
    out = folder / "replication"
    out.mkdir(exist_ok=True)
    report = ["# Locked-configuration replication",
              "Exploratory follow-up after inspecting seeds 41–45. Seeds 51–70 vary initial labels "
              "and model initialization on the original split. Values are normalized accuracy AULC in "
              "percent. Differences and unadjusted paired 95% t intervals are in percentage points."]
    summary = []
    paths = sorted(folder.glob("*.json"))
    if not paths:
        raise ValueError("No primary study results found")
    figure = Figure(figsize=(3.3 * len(paths), 6.2), layout="constrained")
    axes = figure.subplots(2, len(paths), squeeze=False, sharey="col")
    for column, path in enumerate(paths):
        summarize(path)  # require complete and audited primary results
        primary = json.loads(path.read_text())
        device = "cuda" if torch.cuda.is_available() else "cpu"
        graph = None
        target = out / path.name
        source = hashlib.sha256(path.read_bytes()).hexdigest()
        result = json.loads(target.read_text()) if target.exists() else dict(source=source, runs={})
        if result["source"] != source:
            raise ValueError(f"Primary study changed for {path.name}; use a fresh replication directory")
        runs = result["runs"]
        for seed in range(51, 71):
            for name, method, g in (("random", "random", 0), ("no_cf", "combined", 0),
                                    ("cf", "combined", primary["gamma"])):
                key = f"{name}/{seed}"
                if key not in runs:
                    if graph is None:
                        graph = prepare(primary["dataset"], device)
                    runs[key] = trial(graph, seed, "test", method, gamma=g, bonus=primary["bonus"])
                    save(target, result)
            assert runs[f"cf/{seed}"]["initial"] == runs[f"no_cf/{seed}"]["initial"] == runs[f"random/{seed}"]["initial"]
        cf, null, random = [np.array([100*runs[f"{m}/{s}"]["aulc"] for s in range(51, 71)])
                            for m in ("cf", "no_cf", "random")]
        delta = cf-null
        half = t.ppf(.975, 19)*delta.std(ddof=1)/np.sqrt(20)
        summary.append([path.stem, *(f"{v.mean():.2f}" for v in (cf, null, random)),
                        f"{delta.mean():+.2f}", f"[{delta.mean()-half:+.2f}, {delta.mean()+half:+.2f}]",
                        int((delta>0).sum())])
        for m, label in (("cf", "CF"), ("no_cf", "Matched control"), ("random", "Random")):
            curves = [runs[f"{m}/{s}"]["curve"] for s in range(51, 71)]
            acc = np.mean([[p["accuracy"]*100 for p in c] for c in curves], axis=0)
            for ax, key in zip(axes[:, column], ("labels", "seconds")):
                x = np.mean([[p[key] for p in c] for c in curves], axis=0)
                ax.plot(x, acc, marker="o", markersize=4, label=label)
        axes[0, column].set(title=path.stem, xlabel="Acquired labels", xticks=[16, 24, 32, 40])
        axes[1, column].set_xlabel("Cumulative wall seconds on one GPU" if device == "cuda" else "Cumulative wall seconds on CPU")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean test accuracy (%)")
    axes[0, 0].legend(fontsize=8)
    figure.savefig(out / "curves.png", dpi=150)
    report.append(table(["Dataset", "CF", "Matched no CF", "Random", "Paired Δ", "95% interval", "CF wins / 20"], summary))
    report.append("![Label and time efficiency over 20 seeds](curves.png)")
    (out / "REPORT.md").write_text("\n\n".join(report)+"\n", encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(ROOT, "results", "cf_dive"))
    torch.set_num_threads(2)
    replicate(parser.parse_args().out)
