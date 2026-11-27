"""Generate V5.1/V6 README tables and figures from saved trials."""
import json
from hashlib import sha256
from pathlib import Path
from statistics import mean
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("results/research_twinmask_v6")
LABELS = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed", "computer": "Amazon Computers"}
CORE = ("cora", "pubmed")
DIRECTIONS = ("text", "edge")
FAMILIES = ("mlm", "structure", "structure_gat", "text_link", "gae", "sage", "gat", "twinmask")


def load(name):
    return json.loads((ROOT/f"{name}.json").read_text(encoding="utf-8"))


def link(name, label=None):
    return f"[{label or name}]({ROOT.as_posix()}/{name}.json)"


def avg(items):
    return f"{mean(items):.4f}"


def table(lines, columns, rows):
    lines.extend(["", "| "+" | ".join(columns)+" |", "|"+"---|"*len(columns)])
    lines.extend("| "+" | ".join(map(str, row))+" |" for row in rows)
    lines.append("")


def plot(analysis, refined, ensemble):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"mlm": "#2f6f97", "sage": "#ce6a39", "structure": "#368450", "twinmask": "#865caa"}
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.8), layout="constrained")
    for i, name in enumerate(CORE):
        for j, (d, baseline) in enumerate((("text", "mlm"), ("edge", "structure"))):
            ax = axes[i, j]
            for model in (baseline, "sage"):
                trace = analysis[name]["convergence"][model]["curve"]
                x = [row["step"] for row in trace]
                ax.plot(x, [row["dev"][d] for row in trace], marker="o", label=f"{model} dev",
                        color=colors[model])
                ax.plot(x, [row["train"][j] for row in trace], linestyle="--", alpha=.55,
                        color=colors[model], label=f"{model} train")
            ax.set(xscale="log", title=f"{LABELS[name]} • {'Token CE' if j == 0 else 'Edge BCE'}",
                   xlabel="Updates", ylabel="Loss")
            ax.set_xticks([32, 100, 300, 600, 1000], ["32", "100", "300", "600", "1000"])
            ax.legend(fontsize=7, frameon=False, ncol=2)
            ax.grid(alpha=.15)
    fig.savefig(ROOT/"convergence.png", dpi=165)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9, 5.8), layout="constrained")
    for i, name in enumerate(CORE):
        ax = axes[i, 0]
        for family, values in refined[name].items():
            best = None
            ys = []
            for r in values["screening"]["trials"]:
                if r["status"] == "COMPLETE":
                    best = r["score"] if best is None else min(best, r["score"])
                ys.append(best)
            ax.step(range(1, len(ys)+1), ys, where="post", label=family, color=colors[family])
        ax.axvline(12, color="#777777", linestyle=":", linewidth=1)
        ax.set(title=f"{LABELS[name]} • Dev search history", xlabel="Trial count", ylabel="Incumbent objective")
        ax.legend(frameon=False, ncol=2)
        ax = axes[i, 1]
        sweep = load(name+"_weight_sweep")
        for normalized, marker in ((False, "o"), (True, "^")):
            records = [r for r in sweep if r["config"]["normalize"] == normalized]
            x = [r["history"]["curve"][-1]["dev"]["text"] for r in records]
            y = [r["history"]["curve"][-1]["dev"]["edge"] for r in records]
            weight = [r["config"]["edge_weight"] for r in records]
            points = ax.scatter(x, y, marker=marker, c=np.log10(weight), cmap="viridis", vmin=-1, vmax=1,
                                label="normalized" if normalized else "raw weights", s=34)
            front = [j for j in range(len(x)) if not any(a <= x[j] and b <= y[j] and
                     (a < x[j] or b < y[j]) for a, b in zip(x, y))]
            front.sort(key=lambda k: x[k])
            ax.plot([x[k] for k in front], [y[k] for k in front], alpha=.38)
        fig.colorbar(points, ax=ax, label="log₁₀ edge weight")
        ax.set(title=f"{LABELS[name]} • 100-update loss balance", xlabel="Dev token CE", ylabel="Dev edge BCE")
        ax.legend(frameon=False, fontsize=7)
    fig.savefig(ROOT/"search_and_pareto.png", dpi=165)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
    for ax, name in zip(axes, CORE):
        for arm, color in (("bidirectional", "#444444"), ("pcgrad", "#2f6f97"),
                           ("cagrad_grid", "#ce6a39"), ("gradnorm_proxy", "#368450")):
            trace = analysis[name]["arms"][arm]["history"]["curve"]
            ax.plot([r["step"] for r in trace], [r["gradients"]["shared"]["cosine"] for r in trace],
                    marker="o", label=arm.replace("_", " "), color=color)
        ax.axhline(0, color="#888888", linestyle=":")
        ax.set(title=LABELS[name], xlabel="Updates", ylabel="Shared task-gradient cosine")
        ax.legend(frameon=False, fontsize=7)
    fig.savefig(ROOT/"gradient_alignment.png", dpi=165)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.6), layout="constrained")
    for offset, arm in ((-.22, "raw"), (0., "gaussian"), (.22, "diffusion")):
        ax.bar(np.arange(len(ensemble))+offset,
               [mean(r["accuracy"] for r in d["arms"][arm]["test"])*100 for d in ensemble.values()],
               width=.2, label=arm)
    ax.set_xticks(range(len(ensemble)), [LABELS[n] for n in ensemble])
    ax.set(ylabel="Held-out accuracy (%)", ylim=(0, 100))
    ax.legend(frameon=False)
    fig.savefig(ROOT/"ensemble_controls.png", dpi=165)
    plt.close(fig)


def path_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    initial = {n: load(n+"_search") for n in LABELS}
    locked = {n: load(n+"_locked") for n in LABELS}
    defaults = {n: load(n+"_default") for n in LABELS}
    refined = {n: load(n+"_deep_search") for n in CORE}
    deep = {n: load(n+"_deep_locked") for n in CORE}
    analysis = {n: load(n+"_analysis") for n in CORE}
    ensemble = {n: load(n+"_ensemble") for n in ("cora", "citeseer")}
    plot(analysis, refined, ensemble)
    manifest = dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    note="Hashes of the current scripts and local artifacts at report generation.",
                    source={p.as_posix(): path_hash(p) for p in sorted(Path("scripts").glob("research_twinmask_v6*.py"))},
                    raw_references={p.as_posix(): path_hash(p) for p in sorted(Path("data/references").glob("*.pt"))},
                    checkpoints={p.name: path_hash(p) for p in sorted(Path("checkpoints/research_twinmask_v6").glob("*.pt"))})
    (ROOT/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = [
        "## TwinMask V5.1 and V6 · Hyperparameter optimization and transfer",
        "",
        "V5.1 removes the out-of-scope CLI-variable reference in the mechanism study. The CLI now regenerates all 54 synthetic rows. "
        "The [artifact manifest](results/research_twinmask_v6/manifest.json) records the repository base commit, current source hashes, graph reference hashes, and saved checkpoint hashes.",
        "",
        "**What the longer run changes.** The paired development sweep compares the joint-minus-MLM text CE gap at 32 and 1,000 updates. "
        + " ".join(f'On {LABELS[n]}, it was '
                   f'{analysis[n]["convergence"]["sage"]["curve"][0]["dev"]["text"]-analysis[n]["convergence"]["mlm"]["curve"][0]["dev"]["text"]:+.4f}'
                   f' and {analysis[n]["convergence"]["sage"]["curve"][-1]["dev"]["text"]-analysis[n]["convergence"]["mlm"]["curve"][-1]["dev"]["text"]:+.4f}.'
                   for n in CORE)+" The late Cora edge-loss curves also show overfitting, which matters when choosing the checkpoint.",
        "",
        "### Optimization protocol",
        "",
        "All eight reconstruction families get 12 Optuna TPE trials per dataset with median pruning and common 64-update screening caps. "
        "The best three completed configurations are checked on a second training seed using development only. "
        "For MLM, structure GNN, GraphSAGE, and TwinMask on Cora and PubMed, the study expands to 30 trials per family. "
        "The top two then train for up to 300 updates on two development seeds, and the configuration and common checkpoint step are frozen before locked evaluation. "
        "The main objectives are CE for MLM, BCE for edge-only models, and the equally weighted mean of CE/log(vocabulary) and BCE/log(2) for joint models. "
        "Family comparisons use loss for the applicable task only.",
        "",
        "The untouched search-test pool is partitioned into two disjoint sets of node and link targets. Search fit/development have no access to their target labels "
        "or positive edges as observations. Each locked fold uses a separate fit/dev repartition and fresh model parameters. "
        "Transductive observed context and repeated use of the same real graphs remain part of the protocol. "
        "Amazon Computers uses one 6,000-node induced sample, with two target folds. "
        "The default comparator uses 32 updates under the same V6 data and training protocol.",
        "",
        "Learning rate, weight decay, width, layers, dropout, branch LR, cosine scheduling, fusion scale, modality dropout, rollout frequency, "
        "and task weights are logged for all applicable families. The table shows the selected configurations and held-out scores. "
        "Cora/PubMed rows prefer the extended result where available. The last column is tuned minus default loss on the same folds.",
    ]
    entries = []
    for n in LABELS:
        for family in FAMILIES:
            extended = n in refined and family in refined[n]
            profile = (refined[n] if extended else initial[n])[family]
            conf = profile["config"]
            rows = [r["test"] for r in (deep[n] if extended else locked[n]) if r["family"] == family]
            control = [r["test"] for r in defaults[n] if r["family"] == family]
            tasks = ("text",) if family == "mlm" else DIRECTIONS if family in ("sage", "gat", "twinmask") else ("edge",)
            entries.append((LABELS[n], family, 30 if extended else 12, f'{conf["lr"]:.2g} / {conf["wd"]:.2g}',
                            f'{conf["width"]} / {conf["depth"]} / {conf["dropout"]:g}', profile["steps"],
                            " / ".join(avg(r[d] for r in rows) for d in tasks),
                            " / ".join(f"{mean(r[d] for r in rows)-mean(r[d] for r in control):+.4f}" for d in tasks)))
    table(lines, ("Dataset", "Model", "Trials", "LR / WD", "Width / layers / dropout", "Updates",
                  "Test CE or BCE", "Loss shift vs default"), entries)
    lines.extend([
        "Full profiles, intermediate objectives, completed/pruned trial counts, and per-fold outputs are in the JSON files below.",
        "",
        "![Development trial history and development loss-weight Pareto points](results/research_twinmask_v6/search_and_pareto.png)",
        "",
        "### Convergence, directional training, and gradient interventions",
        "",
        "Cora and PubMed convergence runs use paired training masks, matched updates from 32 to 1,000 and a fixed 48-wide model. "
        "The measured development loss is held out from weight fitting. Training losses are averages since the preceding checkpoint.",
        "",
        "![Paired training and development loss curves at 32, 100, 300, 600 and 1000 updates](results/research_twinmask_v6/convergence.png)",
        "",
        "The table uses the first locked fold, equal 100-update budgets, and a shared initialization for arms with the same architecture. "
        "Gradient cosine is recorded before gradient projection on parameters differentiable for both tasks. "
        "GradNorm denotes a closed-form balancing proxy and CAGrad uses an approximate 65-point two-task solver.",
    ])
    arows = []
    for n in CORE:
        entries = analysis[n]["arms"] | {"learned_gates": load(n+"_gates")}
        for key, result in entries.items():
            history = result["history"]
            cos = [r["gradients"]["shared"]["cosine"] for r in history["curve"] if "shared" in r["gradients"]]
            gates = history.get("learned_gates")
            arows.append((LABELS[n], key.replace("_", " "), *(f'{result["test"][k]:.4f}' for k in ("text", "edge", "edge_auc")),
                          avg(cos) if cos else "—", " / ".join(f"{g:.3f}" for g in gates) if gates else "—"))
    table(lines, ("Dataset", "Arm", "Token CE", "Edge BCE", "Edge AUROC", "Mean gradient cosine", "Learned text / edge gates"), arows)
    lines.extend(["![Shared gradient alignment under selected mitigation methods](results/research_twinmask_v6/gradient_alignment.png)",
                  "", "Hard-negative controls use only observed degrees or observed two-hop paths to propose negatives, then reject every true reference edge. "
                  "The candidate composition changes, so ranking results represent a stress test rather than interchangeable samples."])
    table(lines, ("Dataset", "Negatives", "Matched proposals", "Edge BCE", "Edge AUROC"),
          ((LABELS[n], flavor, row["count"] if flavor != "uniform" else "—",
            f'{values["edge"]:.4f}', f'{values["edge_auc"]:.4f}')
           for n in CORE for flavor in ("uniform", "degree", "two_hop")
           for arm in [analysis[n]["arms"]["bidirectional"]]
           for row in [arm if flavor == "uniform" else arm[flavor]]
           for values in [row["test"] if flavor == "uniform" else row["metrics"]]))
    lines.extend(["### Encoder and vocabulary controls", "",
                  "Capacity changes below share exact token positions and truth IDs from the common 256-token fitting-derived vocabulary. "
                  "Raw CE still scores each model's native alphabet. DistilRoBERTa uses cached contextual representations of the actual masked inputs "
                  "and a trained projection. Its short mask-bank feasibility study uses four development settings and a single locked fold."])
    table(lines, ("Cora encoder", "Task", "Positions", "Vocabulary", "Test CE", "Token accuracy"),
          (( "CNN", r["family"], r["length"], r["vocabulary"], f'{r["test"]["ce"]:.4f}', f'{r["test"]["accuracy"]:.4f}')
           for r in load("cora_capacity")))
    table(lines, ("Cora frozen encoder", "Selected LR", "Fusion", "Updates", "Test CE", "Test token accuracy"),
          ((family, r["winner"]["config"]["lr"], r["winner"]["config"]["fusion"], r["winner"]["steps"],
            f'{r["test"]["text"]:.4f}', f'{r["test"]["token_accuracy"]:.4f}')
           for family, r in load("cora_pretrained")["arms"].items()))
    lines.extend(["### Teacher calibration, local utility, and diffusion policies", "",
                  "Each selector family gets ten TPE trials per direction and dataset on Cora and PubMed. "
                  "The search covers LR, decay, width, 70/200/500-update caps, message cost, rank weight, and decision weight or temperature where applicable. "
                  "Teachers use independently tuned backbones and development-only temperature/family choice. "
                  "The mixture and stacking searches use development labels, as does the choice of rollout horizon from 1, 2, 4, and 8. "
                  "The selected configurations and evaluated stages use the same locked targets as the reconstruction study."])
    selection = {n: load(n+"_selectors") for n in CORE}
    table(lines, ("Dataset", "Task", "Stage", "LR / width / updates", "Rank / decision / cost", "Locked loss", "Matched random"),
          ((LABELS[n], d, branch, f'{p["lr"]:.2g} / {p["width"]} / {p["updates"]}',
            f'{p["rank"]:g} / {p["decision"]:g} / {p["cost"]:g}',
            avg(v["directions"][d]["selectors"][branch]["loss"] for v in selection[n]["locked"]),
            avg(v["directions"][d]["selectors"][branch]["random"] for v in selection[n]["locked"]))
           for n in CORE for d in DIRECTIONS for branch, rec in selection[n]["directions"][d]["policies"].items()
           for p in [rec["config"]]))
    table(lines, ("Dataset", "Task", "Horizon", "Mixture temperature", "Expected hard loss", "Uniform",
                  "Tuned mixture", "Tuned stacking"),
          ((LABELS[n], d, selection[n]["horizon"], selection[n]["directions"][d]["mixture"]["mixture"],
            *(avg(r[k]["loss"] for row in selection[n]["locked"] for r in row["directions"][d]["expected"])
              for k in ("expected_at_most", "uniform", "utility_mixture", "stacking")))
           for n in CORE for d in DIRECTIONS))
    table(lines, ("Dataset", "Task", "Unscaled / calibrated NLL", "Unscaled / calibrated Brier", "Unscaled / calibrated ECE"),
          ((LABELS[n], d, *(" / ".join(avg(r[d]["scores"][mode][m] for r in rows)
                                      for mode in ("unscaled", "calibrated")) for m in ("nll", "brier", "ece")))
           for n in CORE for rows in [load(n+"_calibration")] for d in DIRECTIONS))
    lines.extend(["### Embedding denoiser and ensemble controls", "",
                  "On Cora and CiteSeer, the shared diffusion denoiser uses a separate ten-trial search. "
                  "The raw, Gaussian and diffusion classifier arms each get ten development trials. "
                  "The matched raw-head comparison fixes the classifier, policy, seeds and Gaussian cosine distortion target."])
    table(lines, ("Dataset", "Arm", "Hidden / LR / updates", "Test accuracy", "Test NLL"),
          ((LABELS[n], k, f'{r["config"]["hidden"]} / {r["config"]["lr"]:.2g} / {r["config"]["updates"]}',
            avg(x["accuracy"] for x in r["test"]), avg(x["nll"] for x in r["test"]))
           for n, e in ensemble.items() for k, r in e["arms"].items()))
    lines.extend(["![Held-out classification accuracy from tuned ensemble arms](results/research_twinmask_v6/ensemble_controls.png)",
                  "", "### Original V2 backbone and V3/V4.1 selector audit", "",
                  "On Cora, the original SparseTwinMask decoder and graph mixer are trained from scratch in a ten-trial development study and evaluated on the two reserved V6 folds. "
                  "For the frozen original V2 checkpoints, we also tune the V3/V4.1 message-panel LOO, Shapley, set, and decision stages with ten trials for each direction and family. "
                  "Those selector results use the matched-context historical audit split and are exploratory. This pilot preserves the historical branch behavior for comparison."])
    legacy = load("cora_legacy")
    denoiser = legacy["denoiser"]
    cfg = denoiser["config"]
    table(lines, ("Original V2", "LR", "Width", "Updates", "Locked token CE", "Locked edge BCE"),
          [("Cora", f'{cfg["lr"]:.3g}', cfg["width"], denoiser["steps"],
            avg(r["test"]["text"] for r in denoiser["test"]), avg(r["test"]["edge"] for r in denoiser["test"]))])
    table(lines, ("Task", "Local selector", "LR / width / updates", "Held-out loss", "Matched random"),
          ((d, kind, f'{p["lr"]:.2g} / {p["width"]} / {p["updates"]}', f'{v["test"]["loss"]:.4f}',
            f'{v["test"]["random"]:.4f}') for d, arms in legacy["local"].items() for kind, v in arms.items()
           for p in [v["best"]["config"]]))
    lines.extend(["### Artifacts and commands", "",
                  "Scope. The expanded equal-budget search is limited to four families on Cora and PubMed. "
                  "The local legacy, pretrained, and orthogonal capacity studies are Cora feasibility runs. "
                  "There is no RoBERTa PEFT run, full-graph Amazon run, or re-tuning of every published competitor or archived script.",
                  "",
                  "Backbone trials and locked folds. " + " ".join(link(n+s, LABELS[n]+s.replace("_", " "))
                      for n in LABELS for s in ("_search", "_locked", "_default")),
                  "", "Expanded search, transfer diagnostics, and downstream stages. " + " ".join(
                      link(n+s) for n in CORE for s in ("_deep_search", "_deep_locked", "_analysis", "_weight_sweep",
                                                       "_gates", "_selectors", "_calibration")),
                  "", "Additional encoder/classifier controls. "+ " ".join(link(n) for n in
                  ("cora_capacity", "cora_pretrained", "cora_ensemble", "citeseer_ensemble", "cora_legacy")),
                  "", "```powershell",
                  "# Requires the existing TwinMask dependencies plus optuna and matplotlib.",
                  "python -m scripts.research_twinmask_v5 --mechanism-study",
                  "python -m scripts.research_twinmask_v6 --trials 12 --trial-updates 64",
                  "python -m scripts.research_twinmask_v6 --default-control",
                  "python -m scripts.research_twinmask_v6_refine --trials 30",
                  "python -m scripts.research_twinmask_v6_analysis",
                  "python -m scripts.research_twinmask_v6_analysis --gates-only",
                  "python -m scripts.research_twinmask_v6_selectors --trials 10",
                  "python -m scripts.research_twinmask_v6_selectors --calibration-only",
                  "python -m scripts.research_twinmask_v6_pretrained",
                  "python -m scripts.research_twinmask_v6_capacity",
                  "python -m scripts.research_twinmask_v6_ensemble",
                  "python -m scripts.research_twinmask_v6_legacy",
                  "python -m unittest scripts.test_research_pilots scripts.test_research_twinmask_v3 scripts.test_research_twinmask_v4 scripts.test_research_twinmask_v5 scripts.test_research_twinmask_v6",
                  "python -m scripts.report_research_twinmask_v6", "```", ""])
    readme = Path("README.md")
    old = readme.read_text(encoding="utf-8")
    marker, next_section = "<!-- research-v6-results -->", "<!-- research-v41-v5 -->"
    before = old.split(marker if marker in old else next_section, 1)[0]
    readme.write_text(before+marker+"\n\n"+"\n".join(lines)+"\n"+old[old.index(next_section):],
                      encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
