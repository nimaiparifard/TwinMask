"""Refresh the V2 tables in README.md from completed experiment JSON files."""
import json
from pathlib import Path
from statistics import mean, stdev

from scipy.stats import t


DATASETS = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed", "wikics": "WikiCS",
            "reddit": "Reddit", "instagram": "Instagram"}
BASE = ("cora", "citeseer", "pubmed")


def load(study, name, count):
    path = Path("results") / f"research_{study}_v2" / f"{name}.json"
    if not path.exists():
        raise ValueError(f"Missing results: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != count or [row["seed"] for row in rows] != list(range(42, 42 + count)):
        raise ValueError(f"Unexpected seed set: {path}")
    return rows


def table(lines, headers, rows):
    lines.extend(["", "| " + " | ".join(headers) + " |",
                  "|" + "---|" * len(headers)])
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    lines.append("")


def policy(rows, name):
    scores = [case["metrics"][row["best_fixed"] if name == "fixed" else name]
              for row in rows for case in row["cases"]]
    return f"{100*mean(s['token_accuracy'] for s in scores):.2f} / {mean(s['edge_auc'] for s in scores):.4f}"


def loss_pair(rows, name):
    scores = [c["gates"][name] for row in rows for c in row["cases"]]
    return " / ".join(f"{mean(s[k] for s in scores):.4f}" for k in ("token_ce", "edge_bce"))


def paired_ci(values):
    half = t.ppf(.975, len(values)-1)*stdev(values)/len(values)**.5
    return f"{mean(values):+.4f} [{mean(values)-half:+.4f}, {mean(values)+half:+.4f}]"


def links(study, entries):
    return " ".join(f"[{label}](results/research_{study}_v2/{key}.json)" for key, label in entries)


def main():
    lines = ["### Corrected experiment results", "",
             "TwinMask uses seeds 42–51 on the three citation datasets and seeds 42–44 for the ablations, extra datasets, and synthetic cases. Each seed has four jointly scored text/edge mask settings. The matched Ensemble comparison uses seeds 42–44.",
             "", "#### Ensemble classifier controls", "",
             "Test accuracy (%) uses the same representations, labeled split, classifier architecture, and training epochs within each comparison. The raw-only head sees only raw embeddings during training. Each matched head sees its own arm's representations and is selected using that arm's mean validation predictive distribution."]
    arms = ("diffusion_5", "gaussian_5", "diffusion_raw_plus_4", "independent_raw_plus_4")
    ensemble = []
    for name in BASE:
        rows = load("ensemble", name, 3)
        for head, prefix in (("Raw-only", "raw_head_"), ("Method-matched", "matched_head_")):
            ensemble.append((DATASETS[name], head, *(f"{100*mean(r['metrics'][prefix+k]['accuracy'] for r in rows):.2f}"
                                                     for k in arms)))
    table(lines, ("Dataset", "Classifier", "Diffusion-5", "Gaussian-5", "Raw + shared-4", "Raw + independent-4"), ensemble)
    lines.extend(["Full metrics, denoiser and classifier timing, and ensemble diversity. " +
                  links("ensemble", ((n, DATASETS[n]) for n in BASE)), "", "#### TwinMask clocks, rollout, and correction", "",
                  "Cells report mean token accuracy (%) / edge AUROC. One-shot uses one call. Joint, corrective, and adaptive schedules have four calls, including any rescoring. The fixed policy is selected on development cases."])
    results = {name: load("twinmask", name, 10 if name in BASE else 3) for name in DATASETS}
    table(lines, ("Dataset", "One-shot", "Joint", "Corrective", "Adaptive", "Selected fixed"),
          ((DATASETS[n], *(policy(rows, k) for k in ("one_shot", "joint", "correct", "adaptive", "fixed")))
           for n, rows in results.items()))
    differences = []
    for n in BASE:
        rows = results[n]
        for comparator in ("one_shot", "fixed"):
            deltas = []
            for metric, scale in (("token_accuracy", 100), ("edge_auc", 1)):
                deltas.append(paired_ci([
                    scale * mean(c["metrics"]["adaptive"][metric] -
                                 c["metrics"][r["best_fixed"] if comparator == "fixed" else comparator][metric]
                                 for c in r["cases"]) for r in rows
                ]))
            differences.append((DATASETS[n], comparator.replace("_", " "), *deltas))
    table(lines, ("Dataset", "Adaptive comparator", "Token accuracy change (pp) [95% CI]",
                  "Edge AUROC change [95% CI]"), differences)
    lines.extend(["Ablations use the same first three seed IDs. Mixed and tied refer to clock training. Hard-edge message passing and no-rollout training are separate ablations."])
    ablations = ("_tied", "_no_rollout", "_hard")
    table(lines, ("Dataset", "Policy", "Mixed", "Tied", "No rollout", "Hard edges"),
          ((DATASETS[n], policy_name.replace("_", " "), policy(results[n][:3], policy_name),
            *(policy(load("twinmask", n+suffix, 3), policy_name) for suffix in ablations))
           for n in BASE for policy_name in ("one_shot", "adaptive")))
    lines.extend(["Detailed corrected runs. " + links("twinmask", DATASETS.items()), "",
                  "Detailed ablations. " + links("twinmask", ((n+s, DATASETS[n]+s.replace("_", " "))
                            for n in BASE for s in ablations)), "",
                  "#### One-shot directional conditioning", "",
                  "Cells report token cross-entropy / edge binary cross-entropy. Lower is better. Each seed contributes the mean over its four cases. The 95% paired t intervals are for learned minus joint loss and treat seeds as the run unit within each graph."])
    gate_table = []
    for n, rows in results.items():
        diffs = [[mean(c["gates"]["learned"][metric]-c["gates"]["joint"][metric] for c in r["cases"])
                  for r in rows] for metric in ("token_ce", "edge_bce")]
        gate_table.append((DATASETS[n], *(loss_pair(rows, k) for k in
                          ("unimodal", "joint", "learned", "oracle_analysis_only")), *(paired_ci(d) for d in diffs)))
    table(lines, ("Dataset", "Unimodal", "Joint", "Learned", "Oracle", "Token CE paired change [95% CI]",
                  "Edge BCE paired change [95% CI]"), gate_table)
    synthetic = ("reference", "low_homophily", "weak_text", "noisy_context")
    syn_table = []
    for n in synthetic:
        rows = load("twinmask", "synthetic_"+n, 3)
        gains = [mean(c[k] for r in rows for c in r["cases"]) for k in ("structure_to_text", "text_to_edge")]
        syn_table.append((n.replace("_", " "), *(f"{g:+.4f}" for g in gains),
                          loss_pair(rows, "joint"), loss_pair(rows, "learned")))
    lines.extend(["Synthetic interventions change expected homophily (0.85 to 0.30), lexical signal (0.80 to 0.20), or add spurious test-context edges (25% of the number of real training edges), one factor at a time. The independently varied text and edge missingness cases are stored in the JSON. A positive evidence gain means that removing the other modality increased loss."])
    table(lines, ("Synthetic case", "Structure to text CE gain", "Text to edge BCE gain", "Joint CE / BCE",
                  "Learned CE / BCE"), syn_table)
    lines.extend(["Synthetic runs. " + links("twinmask", (("synthetic_"+n, n.replace("_", " ")) for n in synthetic)), "",
                  "#### Pretrained encoder, alternate split, and text audit", ""])
    pretrained = load("twinmask", "cora_pretrained_full_vocab_directional", 1)
    resplit = load("twinmask", "cora_resplit", 3)
    full_vocab = load("twinmask", "cora_full_vocab", 1)
    table(lines, ("Cora configuration", "Seeds", "One-shot accuracy / AUROC", "Adaptive accuracy / AUROC", "Learned CE / BCE"),
          ((label, len(rows), policy(rows, "one_shot"), "—" if rows[0].get("directional_only") else policy(rows, "adaptive"),
            loss_pair(rows, "learned"))
           for label, rows in (("Frozen DistilRoBERTa", pretrained), ("Full lexical vocabulary smoke test", full_vocab),
                               ("Seed-specific link re-splits, CPU", resplit))))
    lines.extend(["The pretrained directional feasibility run uses 128 token positions, a 256-dimensional graph mixer, five training epochs, all 50,261 RoBERTa lexical IDs, and up to 128 masked token targets per case. The frozen encoder runs on the currently visible text. The CPU vocabulary smoke test uses up to 64 masked token targets per case. Other masked positions stay concealed. The feasibility rows use different encoders and target panels. A multi-seed pretrained comparison remains open. The alternate Cora protocol re-splits warm pairs per seed and preserves held-out positives outside training adjacency.",
                  links("twinmask", (("cora_pretrained_full_vocab_directional", "Pretrained run"), ("cora_full_vocab", "Full vocabulary smoke test"),
                                     ("cora_resplit", "Link re-splits"))), "",
                  "The cached raw text audit counted blank strings and median whitespace-delimited words. WikiCS had 0 blank among 11,701 texts (median 219 words). Reddit had 0 among 33,434 (median 87). Instagram had 99 among 11,339 (median 16). These counts describe coverage and length. The ogbn-arxiv cache had 0 blanks among 169,343 texts (median 167). Its cached links are undirected and lack an evaluation-ready temporal split, so no ogbn-arxiv V2 result is reported.",
                  "", "#### Reproduction commands", "", "```powershell",
                  "python -m scripts.research_ensemble",
                  "python -m scripts.research_twinmask --seeds 42 43 44 45 46 47 48 49 50 51",
                  "python -m scripts.research_twinmask --clock-training tied",
                  "python -m scripts.research_twinmask --no-rollout",
                  "python -m scripts.research_twinmask --hard-edges",
                  "python -m scripts.research_twinmask --datasets wikics reddit instagram",
                  "python -m scripts.research_twinmask --datasets synthetic_reference synthetic_low_homophily synthetic_weak_text synthetic_noisy_context",
                  "python -m scripts.research_twinmask --datasets cora --seeds 42 --epochs 5 --encoder hf_models/all-distilroberta-v1 --width 256 --length 128 --full-vocab --max-token-targets 128 --directional-only",
                  "python -m scripts.research_twinmask --datasets cora --resplit-links --device cpu",
                  "python -m scripts.research_twinmask --datasets cora --seeds 42 --epochs 5 --device cpu --full-vocab --max-token-targets 64",
                  "python -m unittest scripts.test_research_pilots -v",
                  "python -m scripts.report_research_v2", "```", ""])
    readme = Path("README.md")
    original = readme.read_text(encoding="utf-8")
    marker = "<!-- research-v2-results -->"
    old_section = "## Research continuation pilots (2026-11-27)"
    before, after = original.split(marker, 1)
    _, archived = after.split(old_section, 1)
    readme.write_text(before + marker + "\n\n" + "\n".join(lines) + "\n" + old_section + archived, encoding="utf-8")


if __name__ == "__main__":
    main()
