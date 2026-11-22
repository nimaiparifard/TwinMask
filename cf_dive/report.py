"""Audit complete experiment files and rebuild the Markdown results."""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from twinmask.data import ROOT, load


def table(headers, rows):
    return "\n".join("| " + " | ".join(map(str, row)) + " |" for row in
                     [headers, ["---"] * len(headers), *rows])


def summarize(path):
    data = json.loads(path.read_text())
    runs = data["runs"]
    n = load(data["dataset"])["num_nodes"]
    pool = set(np.random.default_rng(312).permutation(n)[:int(.6*n)])
    expected = {f"validation/gamma-{g}_bonus-{b}/{s}" for g in (0., .1, .3, .6)
                for b in (.05, .15, .3) for s in (11, 12, 13)}
    methods = ("random", "entropy", "disagreement", "no_cf", "tuned_no_cf", "cf",
               "cf_top", "cf_milp", "raw_ensemble_cf", "single_cf")
    expected |= {f"test/{m}/{s}" for m in methods for s in range(41, 46)}
    if set(runs) != expected:
        raise ValueError(f"Incomplete or unexpected trials in {path}")
    initial = {}
    for key, row in runs.items():
        seed = row["seed"]
        assert seed == int(key.split("/")[-1])
        assert initial.setdefault(seed, row["initial"]) == row["initial"]
        nodes = set(row["initial"])
        for point, budget in zip(row["curve"], (16, 24, 32, 40), strict=True):
            assert len(nodes) == budget == point["labels"] and nodes <= pool
            nodes.update(point.get("queried", []))
        assert len(nodes) == 40 and nodes <= pool

    def series(stage, method, seeds):
        return [runs[f"{stage}/{method}/{s}"] for s in seeds]

    tuning = []
    for g in (0., .1, .3, .6):
        cells = []
        for b in (.05, .15, .3):
            vs = [100*r["aulc"] for r in series("validation", f"gamma-{g}_bonus-{b}", (11, 12, 13))]
            cells.append(f"{mean(vs):.2f} ± {stdev(vs):.2f}")
        tuning.append([g, *cells])
    paired = []
    grid = []
    costs = []
    for m in methods:
        rows = series("test", m, range(41, 46))
        budgets = [100*mean(r["curve"][i]["accuracy"] for r in rows) for i in range(4)]
        scores = [100*r["aulc"] for r in rows]
        f1 = 100*mean(r["curve"][-1]["macro_f1"] for r in rows)
        grid.append([m, *(f"{x:.2f}" for x in budgets), f"{mean(scores):.2f} ± {stdev(scores):.2f}",
                     f"{f1:.2f}"])
        costs.append([m, *(f"{mean(r['curve'][i]['seconds'] for r in rows):.3f}" for i in range(4))])
    cf = series("test", "cf", range(41, 46))
    null = series("test", "no_cf", range(41, 46))
    paired = [100*(a["aulc"]-b["aulc"]) for a, b in zip(cf, null)]
    delta = mean(paired)
    half = 2.77645*stdev(paired)/np.sqrt(5)
    probes = [p["interventions"] for r in cf for p in r["curve"][:-1]]
    count = sum(p["probes"] for p in probes)
    flip = [100*sum(p[k]*p["probes"] for p in probes)/max(1,count)
            for k in ("target_flip", "other_endpoint_flip")]
    hours = sum(r["seconds"] for r in runs.values())/3600
    title = data["dataset"]
    sections = [f"## {title}", f"Positive-CF validation choice γ={data['gamma']}, coverage bonus={data['bonus']}. "
                f"Best unrestricted validation pair {data['validation_winner']}. Best zero-CF bonus {data['null_bonus']}.",
                "Validation AULC in percent, mean ± sample SD over three seeds.",
                table(["γ / bonus", ".05", ".15", ".30"], tuning),
                "Test accuracy by acquired labels, normalized accuracy AULC, and final macro-F1, all in percent.",
                table(["Method", "16", "24", "32", "40", "AULC ± SD", "F1 @ 40"], grid),
                f"Paired CF minus no_cf AULC in seed order 41–45 (percentage points): "
                + ", ".join(f"{x:+.2f}" for x in paired) + f". Mean {delta:+.2f}, "
                f"descriptive paired t interval [{delta-half:+.2f}, {delta+half:+.2f}].",
                f"Across {count:,} independent edge probes in the CF arm, the average member flipped "
                f"at the target in {flip[0]:.2f}% and at the other endpoint in {flip[1]:.2f}%.",
                "Mean cumulative wall seconds by label budget on one device. Divide by 3,600 for allocated device-hours.",
                table(["Method", "16", "24", "32", "40"], costs),
                f"Summed validation and test trial time {hours:.4f} device-hours. "
                f"One-time preparation {data['preparation_seconds']:.2f} seconds. Device: {data['signature']['device']}."]
    return sections


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(ROOT, "results", "cf_dive"))
    args = parser.parse_args()
    sections = ["# CF-DIVE experiment results",
                "Each dataset uses one fixed transductive split. The paired intervals describe acquisition "
                "and model seed variability on that split. Historical pretraining, cache creation, process startup "
                "and result serialization are excluded from the timing."]
    for path in sorted(args.out.glob("*.json")):
        sections.extend(summarize(path))
    if len(sections) == 2:
        raise ValueError("No experiment results found")
    (args.out / "REPORT.md").write_text("\n\n".join(sections)+"\n", encoding="utf8")
