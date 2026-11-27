"""Generate the V4.1 and V5 README sections from audited results."""
import json
from pathlib import Path
from statistics import mean, stdev

from scipy.stats import t


BENCHMARKS = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed", "computer": "Amazon Computers"}
DIRECTIONS = ("text", "edge")


def load(version, name, count=None):
    rows = json.loads(Path(f"results/research_twinmask_{version}/{name}.json").read_text(encoding="utf-8"))
    if count is not None and len(rows) != count:
        raise ValueError(f"{version}/{name} has {len(rows)} runs, expected {count}")
    return rows


def avg(values, digits=4):
    collected = [x for x in values if x is not None]
    return f"{mean(collected):.{digits}f}" if collected else "—"


def table(lines, headers, rows):
    lines.extend(["", "| " + " | ".join(headers) + " |", "|"+"---|"*len(headers)])
    lines.extend("| " + " | ".join(map(str, r)) + " |" for r in rows)
    lines.append("")


def link(version, name, label=None):
    return f"[{label or name}](results/research_twinmask_{version}/{name}.json)"


def cases41(rows, d):
    return [case for row in rows for case in row["directions"][d]["cases"]]


def v41_section():
    names = tuple(BENCHMARKS)[:3]
    sets = {(name, context): load("v41", f"{name}_{context}", 3)
            for name in names for context in ("matched", "growing")}
    lines = [
        "## TwinMask V4.1 · Correctness and protocol audit",
        "",
        "V4.1 aligns the local raw graphs to the cached node order by exact text comparison and confirms that every saved positive is present in the raw graph. "
        "The complete canonical edge references, including Cold–Cold links, are saved in `data/references/`. "
        "Warm negatives are generated against the full reference and are unique and disjoint across controller train, development, and test. "
        "The revised sampler also prevents validation/test negative overlap and duplicate cold negatives in newly prepared datasets. "
        "The old frozen backbones are retained so the matched/growing comparison isolates the selector context protocol. Those historical checkpoint weights were fitted under V2.",
        "",
        "The matched and growing arms share regenerated candidate pools, target node splits, masks, and seed IDs 42–44. "
        "Both arms train on the same reduced base graph. Growing adds the controller and development positive pools to the test-only base. "
        "Each episode separately adds its observed positive candidates. Thus the arms differ in base connectivity while evaluating identical target labels.",
    ]
    table(lines, ("Dataset", "Raw edges", "Omitted positives in legacy cache", "Legacy false negatives",
                  "Legacy val/test negative overlap", "Legacy duplicate cold negatives", "Matched / growing test base"),
          ((BENCHMARKS[n], a["reference_edges"], a["omitted_positive_edges"], sum(a["legacy_false_negatives"].values()),
            a["legacy_validation_test_overlap"], a["legacy_duplicates"]["cold_neg"],
            f'{a["test_base_edges"]} / {sets[n, "growing"][0]["audit"]["test_base_edges"]}')
           for n in names for a in [sets[n, "matched"][0]["audit"]]))
    lines.extend(["The audit of stored legacy validation, test, and cold negatives found zero true positives among those negatives. "
                  "This check concerns the stored pools and does not establish equivalence to the old training sampler.",
                  "", "Lower token CE and edge BCE are better. K=2 is an upper bound on optional messages. Random and oracle receive the decision policy's realized count on each target."])
    table(lines, ("Dataset", "Task", "Context", "Full", "Decision + cost", "Count-matched random", "Equal-count oracle regret",
                  "Expected regret", "Dev-scaled 90% coverage"),
          ((BENCHMARKS[n], d, protocol, avg(c["full"]["loss"] for c in v),
            avg(c["budget"]["2"]["decision_cost"]["loss"] for c in v),
            avg(c["budget"]["2"]["matched_random"]["loss"] for c in v),
            avg(c["budget"]["2"]["decision_cost"]["oracle_regret_equal_count"] for c in v),
            avg(c["scores"]["decision"]["expected_regret_k2"] for c in v),
            avg(c["scores"]["decision"]["dev_scaled_90_coverage_nonfull"] for c in v))
           for (n, protocol), rows in sets.items() for d in DIRECTIONS for v in [cases41(rows, d)]))
    lines.extend(["Risk models now learn utility relative to Full directly, with exact zero mean and uncertainty for Full. "
                  "The held-out interval uses the development 90th percentile of standardized non-full residuals. "
                  "Reported coverage is empirical because messages and subsets are dependent.",
                  "", "The next table uses the matched arm and the prediction-aware LOO estimator. The rank metrics exclude targets with no non-tied pair. "
                  "Harmful precision and recall use LOO utility below −0.001 and predicted utility at most zero."])
    metrics = ("spearman", "within_target_spearman", "ranking_coverage", "pairwise_accuracy", "top1_accuracy",
               "harmful_precision", "harmful_recall")
    table(lines, ("Dataset", "Task", "Global ρ", "Within-target ρ", "Rank coverage", "Pairwise accuracy",
                  "Top-1", "Harmful precision", "Harmful recall"),
          ((BENCHMARKS[n], d, *(avg(c["scores"]["aware_loo"].get(k) for c in v) for k in metrics))
           for n in names for d in DIRECTIONS for v in [cases41(sets[n, "matched"], d)]))
    lines.extend(["Corrected diffusion clocks scale each original rate by the fraction of the original hidden entries still pending. "
                  "Unqueried hidden entries count toward the clock. The paired replay below evaluates two passes."])
    table(lines, ("Matched context", "Full token accuracy / edge AUROC", "Selected token accuracy / edge AUROC",
                  "Selected second-call text / edge clock"),
          ((BENCHMARKS[n], *(" / ".join((avg(c["token_accuracy"] for c in v), avg(c["edge"]["auc"] for c in v)))
                             for k in ("full", "selected") for v in [[c for r in sets[n, "matched"]
                                                                     for c in r["diffusion"][k]]]),
            " / ".join(avg(c["traces"][1]["clocks"][i] for r in sets[n, "matched"] for c in r["diffusion"]["selected"])
                       for i in (0, 1))) for n in names))
    lines.extend(["V4.1 retains the V4 direct-component intervention for comparability. A test compares the cached two-layer local intervention to direct recomputation. "
                  "The latency remains that of the existing full graph aggregation and local probes, separate from the number of optional messages retained.", "",
                  "Per-seed results. " + " ".join(link("v41", f"{n}_{c}", f"{BENCHMARKS[n]} {c}")
                                               for n, c in sets), ""])
    return lines


def split_interval(rows, extract):
    splits = sorted({r["split_seed"] for r in rows})
    values = [mean(extract(r) for r in rows if r["split_seed"] == split) for split in splits]
    midpoint = mean(values)
    half = t.ppf(.975, len(values)-1)*stdev(values)/len(values)**.5
    return f"{midpoint:+.4f} [{midpoint-half:+.4f}, {midpoint+half:+.4f}]"


def v5_section():
    datasets = {k: load("v5", k, 6) for k in BENCHMARKS}
    controls = {k: load("v5", k+"_selectors", 2) for k in BENCHMARKS}
    lines = [
        "## TwinMask V5 · Expected utility and isolated cross-modal transfer",
        "",
        "V5 trains text and topology encoders independently in the forward computation and introduces cross-modal evidence only in fusion. "
        "With all optional groups suppressed, the text decoder depends on the local text stream and the edge decoder uses the topology stream. "
        "At full weights the subset decoder reproduces the ordinary joint logits. The four groups are a structural self representation and three "
        "deterministic incoming-structure groups for token targets, and two endpoint text representations plus their two aggregated text neighborhoods for edge targets.",
        "",
        "The locked run grid uses split seeds **2101, 2113** and model seeds **71, 83, 97**. Every split trains fresh weights. "
        "The same base edges support fitting, development, and evaluation. Reference-backed warm pools are disjoint and historical validation/test citation positives "
        "are excluded from these new pools. Histories on the citation graphs still make them exploratory as corpora. "
        "The new Amazon Computers experiment uses two deterministic uniform 6,000-node induced samples from the 87,229-node raw graph and independently splits the induced pairs. "
        "This scope allows comparable GPU runs and does not estimate performance on the full graph.",
        "",
        "**Main finding.** Structural conditioning worsened mean text CE in 11 of 12 joint architecture–dataset comparisons. "
        "Joint training also raised mean text CE relative to the separately trained MLM in all 12 comparisons. "
        "Text conditioning for edges varied by architecture and dataset. Expected-utility selection produced small average improvements on some comparisons, "
        "while all paired expected-versus-random loss intervals included zero.",
        "",
        "### Conditioning versus joint training",
        "",
        "Positive gain means the joint arm lowers loss. Conditioning gain switches opposite-modality fusion off in the same checkpoint. "
        "Joint-training gain compares the joint checkpoint with a separately trained text-only MLM or structure-only GNN. "
        "The GAT and GraphSAGE controls are one-shot, the GAE is an inner-product structural reconstruction baseline, and the TwinMask arm adds free-prediction feedback during training.",
    ]
    table(lines, ("Dataset", "Architecture", "Structure → text", "Text → edge", "Joint training text gain", "Joint training edge gain"),
          ((BENCHMARKS[n], model, *(avg(c[d][metric] for row in rows for c in row["architectures"][model])
                                     for metric in ("conditioning_gain", "joint_training_gain") for d in DIRECTIONS))
           for n, rows in datasets.items() for model in ("sage", "gat", "twinmask")))
    table(lines, ("Dataset", "Text-only MLM CE", "Structure-only GNN BCE", "Text-only link BCE", "GAE BCE"),
          ((BENCHMARKS[n], *(avg(c[d]["full"]["loss"] for row in rows for c in row["architectures"][model])
                             for model, d in (("mlm", "text"), ("structure", "edge"), ("text_link", "edge"), ("gae", "edge"))))
           for n, rows in datasets.items()))
    lines.extend(["### Fixed-target timestep analysis", "",
                  "One held-out audit set stays masked at every timestep. Static corruption redraws the known context for each step. "
                  "The free-running and teacher-forced arms reveal the same scheduled positions. Only teacher forcing inserts gold feedback, for diagnostic analysis. "
                  "All three arms follow identical actual remaining-mask clocks. The four-step table gives signed text CE and edge BCE conditioning gains in chronological order."])
    table(lines, ("Dataset", "Mode", "Structure → text by step", "Text → edge by step"),
          ((BENCHMARKS[n], mode, *(" / ".join(avg(r["temporal"]["4"][mode][i][d] for r in rows) for i in range(4))
                                  for d in DIRECTIONS))
           for n, rows in datasets.items() for mode in ("static", "free", "teacher")))
    lines.extend(["For the 1, 2, 4, and 8 step checks, the table reports final-step free-running conditional utility on the same audit targets."])
    table(lines, ("Dataset", "Task", "1 step", "2 steps", "4 steps", "8 steps"),
          ((BENCHMARKS[n], d, *(avg(r["temporal"][str(h)]["free"][-1][d] for r in rows) for h in (1, 2, 4, 8)))
           for n, rows in datasets.items() for d in DIRECTIONS))
    lines.extend(["### Expected utility, mixtures, and distillation", "",
                  "A standalone MLM supplies text targets. Edge teachers include a text-only link MLP, a structure-only GNN, a graph autoencoder, and two ensembles. "
                  "Teacher temperatures and the teacher family are chosen on development loss. The student backbone's full prediction is never its own teacher. "
                  "Expected gain is the teacher-weighted log-probability difference between a subset and Full. Hard selection uses at most two groups. "
                  "The three soft mixture controls share the exact candidate pool, including Full, and stacking fits nonnegative mixture weights on development only.",
                  "", "The paired interval uses two independent split means, with three model seeds averaged within each split. Its width reflects the small split count."])
    table(lines, ("Dataset", "Task", "Full", "Teacher", "Expected ≤2", "Matched random", "Expected − random [95% CI]",
                  "Uniform", "Expected mixture", "Stacking"),
          ((BENCHMARKS[n], d, *(avg(c[k]["loss"] for c in v) for k in ("full", "teacher", "expected_at_most", "random_matched")),
            split_interval(rows, lambda row: mean(c["expected_at_most"]["loss"]-c["random_matched"]["loss"]
                                                  for c in row["expected"][d]["cases"])),
            *(avg(c[k]["loss"] for c in v) for k in ("uniform", "utility_mixture", "stacking")))
           for n, rows in datasets.items() for d in DIRECTIONS for v in [[c for r in rows for c in r["expected"][d]["cases"]]]))
    table(lines, ("Dataset", "Metric", "Full", "Teacher", "Expected ≤2", "Expected mixture", "Stacking"),
          ((BENCHMARKS[n], m, *(avg(c[k][m] for c in v) for k in ("full", "teacher", "expected_at_most", "utility_mixture", "stacking")))
           for n, rows in datasets.items() for d, m in (("text", "accuracy"), ("edge", "auc"), ("edge", "ap"))
           for v in [[c for r in rows for c in r["expected"][d]["cases"]]]))
    lines.extend(["The distillation pilot uses one model seed per split. Direct and LOO controls use only fitting outcomes. "
                  "The distilled scorer learns from independent-teacher expected utility and decodes only the selected subset at inference, without teacher calls or prediction probes. "
                  "All four hard controls use exact K=2, capped by availability. Timing compares synchronization-warmed full selector pipelines. "
                  "The measured baseline is expected utility with teacher queries, while ordinary full-message inference is a separate baseline."])
    table(lines, ("Dataset", "Task", "Expected exact", "Direct", "LOO", "Distilled", "Agreement",
                  "Expected / distilled ms"),
          ((BENCHMARKS[n], d, *(avg(c[k]["loss"] for c in v) for k in ("expected", "direct", "loo", "distilled")),
            avg(c["distilled_expected_agreement"] for c in v),
            " / ".join(avg(c["timing_ms"][k] for c in v if "timing_ms" in c) for k in ("expected", "distilled")))
           for n, rows in controls.items() for d in DIRECTIONS
           for v in [[c for r in rows for c in r["expected"][d]["selector_controls"]]]))
    lines.extend(["### Mechanisms and sensitivity checks", "",
                  "Each synthetic setting uses three independently generated graphs, and each sweep changes one factor with the others fixed "
                  "at homophily 0.7, text noise 0.1, edge noise 0, and 128 tokens. Edge noise adds false edges only to test context. "
                  "Vocabulary varies the data-generating alphabet and decoder size. The full vocabulary run limits fitting and scoring query counts for memory. "
                  "Positive utility indicates improved loss with the opposite modality; harmful fraction uses per-target utility below −0.001."])
    mechanism = load("v5", "mechanisms", 54)
    order = list(dict.fromkeys((r["factor"], r["level"]) for r in mechanism))
    table(lines, ("Factor", "Level", "Text utility", "Edge utility", "Text harmful", "Edge harmful"),
          ((f, v, *(avg(r[d][k] for r in subset) for k in ("conditioning_gain", "harmful_fraction") for d in DIRECTIONS))
           for f, v in order for subset in [[r for r in mechanism if r["factor"] == f and r["level"] == v]]))
    lines.extend(["Real-data missingness sweeps change either text or edge missingness through 0.1, 0.3, 0.5, 0.7, and 0.9 with the other fixed at 0.5. "
                  "Degree bins stratify targets by observed degree at each masked case. Ties can produce empty bins. "
                  "The table lists direction-matched utilities for the two missingness sweeps and the low, medium, and high degree bins."])
    table(lines, ("Dataset", "Text gain over text-mask sweep", "Edge gain over edge-mask sweep",
                  "Text degree-bin gain", "Edge degree-bin gain"),
          ((BENCHMARKS[n], *(" / ".join(avg(r["missingness"][d][i][d]["conditioning_gain"] for r in rows)
                                         for i in range(5)) for d in DIRECTIONS),
            *(" / ".join(avg(c["degree_conditioning_gain"][i] for r in rows for c in r["expected"][d]["cases"])
                          for i in range(3)) for d in DIRECTIONS)) for n, rows in datasets.items()))
    lines.extend(["Full V5 runs. " + " ".join(link("v5", n, label) for n, label in BENCHMARKS.items()), "",
                  "Selector and timing controls. " + " ".join(link("v5", n+"_selectors", label)
                                                            for n, label in BENCHMARKS.items()), "",
                  "Independent synthetic graphs. " + link("v5", "mechanisms"), "",
                  "### Reproduction", "",
                  "Raw references default to the neighboring `DIVERGE/datasets` directory and can be supplied through `TWINMASK_RAW_GRAPHS`. "
                  "The audit requires the three local V2 checkpoints per citation dataset. The V5 experiments initialize all models independently.", "",
                  "```powershell", "python -m scripts.research_twinmask_v41",
                  "python -m scripts.research_twinmask_v5",
                  "python -m scripts.research_twinmask_v5 --model-seeds 71 --selector-controls",
                  "python -m scripts.research_twinmask_v5 --mechanism-study",
                  "python -m unittest scripts.test_research_pilots scripts.test_research_twinmask_v3 scripts.test_research_twinmask_v4 scripts.test_research_twinmask_v5",
                  "python -m scripts.report_research_v5", "```", ""])
    return lines


def main():
    path = Path("README.md")
    readme = path.read_text(encoding="utf-8")
    marker = "<!-- research-v41-v5 -->"
    next_section = "<!-- research-v4-results -->"
    before = readme.split(marker if marker in readme else next_section, 1)[0]
    tail = readme[readme.index(next_section):]
    updated = "\n".join((*v41_section(), *v5_section()))
    path.write_text(before+marker+"\n\n"+updated+"\n"+tail, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
