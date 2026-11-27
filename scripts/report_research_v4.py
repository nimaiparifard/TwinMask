"""Build the TwinMask V4 README section from measured result files."""
import json
from pathlib import Path
from statistics import mean, stdev

from scipy.stats import t


ROOT = Path("results/research_twinmask_v4")
NAMES = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}
DIRECTIONS = ("text", "edge")


def load(name, count):
    rows = json.loads((ROOT/f"{name}.json").read_text(encoding="utf-8"))
    assert [r["seed"] for r in rows] == list(range(42, 42+count)), name
    return rows


def avg(values):
    return f"{mean(x for x in values if x is not None):.4f}"


def cases(rows, direction):
    return [case for row in rows for case in row["directions"][direction]["cases"]]


def table(lines, header, rows):
    lines.extend(["", "| " + " | ".join(header) + " |", "|"+"---|"*len(header)])
    lines.extend("| " + " | ".join(map(str, r)) + " |" for r in rows)
    lines.append("")


def ci(values):
    estimate = mean(values)
    half = t.ppf(.975, len(values)-1)*stdev(values)/len(values)**.5
    return f"{estimate:+.4f} [{estimate-half:+.4f}, {estimate+half:+.4f}]"


def link(name, label=None):
    return f"[{label or name}]({ROOT.as_posix()}/{name}.json)"


def main():
    primary = {key: load(key, 10) for key in NAMES}
    extras = {key: load(key+"_extra", 3) for key in NAMES}
    controls = {key: load(key+"_baselines", 3) for key in NAMES}
    lines = [
        "## TwinMask V4 · Target-conditioned counterfactual set utility",
        "",
        "V4 evaluates whether observed target features and learned subset value recover the oracle gains in V3. "
        "The primary study uses the same V2 checkpoints and saved test masks across all selectors, four asymmetric mask cases per seed, "
        "and up to 384 targets per direction and case. It covers Cora, CiteSeer, and PubMed with seeds 42–51. "
        "Controller-training, development, and test cases have disjoint target nodes and held-out edge pools. "
        "Test targets enter metric and oracle computation only. A regression test modifies hidden labels and verifies unchanged selections and predictions.",
        "",
        "**What worked.** Target and prediction-change features improved PubMed edge LOO Spearman from 0.0633 to 0.2026. "
        "The decision selector reduced edge BCE versus equal-count random selection on Cora and PubMed. "
        "The highest matched-budget recovery for decision plus cost was 14.8%, below the proposed 30% goal. "
        "The audit also missed the proposed 0.25 LOO Spearman and 65% sign-accuracy targets for the PubMed edge arm. "
        "The selected inference path was slower than full-message decoding in each measured timing pair.",
        "",
        "### V4.0 · Oracle and utility audit",
        "",
        "Loss is token cross-entropy (CE) or edge binary cross-entropy (BCE). Token and edge oracles choose per target. "
        "The node oracle uses one panel subset for a node's sampled tokens. The shared-endpoint oracle shares a subset at "
        "the first endpoint of all the sampled candidates with that endpoint, while allowing the other endpoint to vary within the total budget. "
        "Both shared oracles still use the target labels, so they measure restrictions on decision granularity. The equal-count oracle and random control use "
        "the same number of optional messages per target as the decision score with message cost.",
    ]
    for d in DIRECTIONS:
        table(lines, ("Dataset", "Task", "Full", "Per-target oracle ≤2", "Shared oracle ≤2", "Matched-count oracle",
                      "Harmful fraction thresholds 0 / .001 / .005 / .01"),
              ((NAMES[n], d, avg(c["full"]["loss"] for c in v),
                avg(c["budget"]["2"]["oracle_cap_analysis_only"]["loss"] for c in v),
                avg(c["shared_oracle_k2"] for c in v),
                avg(c["budget"]["2"]["matched_oracle_analysis_only"]["loss"] for c in v),
                " / ".join(avg(c["harmful"][str(e)] for c in v) for e in (0., .001, .005, .01)))
               for n, rows in primary.items() for v in [cases(rows, d)]))
    lines.extend(["### V4.1–V4.2 · Target-conditioned and prediction-aware estimates", "",
                  "Text features include sender, receiver, message, masked-token representation, and position. Edge features include both endpoints, "
                  "their absolute difference and product, with the opposite endpoint defined relative to each message receiver. "
                  "Prediction-aware features add full uncertainty, changes in entropy, confidence and margin, JS divergence, top-answer disagreement, "
                  "and the magnitude of the removed contribution before the final nonlinearity. The position and all semantic features come from visible or masked inputs.",
                  "", "The first three columns below are LOO Spearman correlations. Sign accuracy is for the prediction-aware LOO estimator."])
    table(lines, ("Dataset", "Task", "V3 features", "+ Target", "+ Prediction changes", "Sign accuracy", "Majority sign"),
          ((NAMES[n], d, *(avg(c["scores"][k]["spearman"] for c in v) for k in ("base_loo", "target_loo", "aware_loo")),
            avg(c["scores"]["aware_loo"]["sign_accuracy"] for c in v), avg(c["sign_majority"] for c in v))
           for n, rows in primary.items() for d in DIRECTIONS for v in [cases(rows, d)]))
    lines.extend(["### V4.3–V4.5 · Set utility, decision loss, and flexible budgets", "",
                  "DeepSets assigns a value to each unique feasible subset using selected and full panel summaries. The target is full loss minus subset loss. "
                  "Decision training adds expected reconstruction regret under a masked softmax at K=1, 2, and 3. "
                  "Development tunes the cost per optional message in standardized utility units and the risk rule. The threshold arm uses a fixed 0.1 "
                  "utility threshold in standardized units, and the at-most arms can choose an empty set.",
                  "", "All entries in the first results table use exact K=2, capped by availability. The direct gate uses the same prediction-aware features "
                  "and scorer width as the message-utility MLP. The Shapley arm is trained on exact message Shapley values."])
    table(lines, ("Dataset", "Task", "Random", "Direct", "Aware LOO", "Shapley", "DeepSets", "Decision"),
          ((NAMES[n], d, *(avg(c["budget"]["2"][k]["loss"] for c in v) for k in
                          ("random_exact", "direct_exact", "aware_loo_exact", "shapley_exact", "deep_sets_exact", "decision_exact")))
           for n, rows in primary.items() for d in DIRECTIONS for v in [cases(rows, d)]))
    budget_rows = []
    for n, rows in primary.items():
        for d in DIRECTIONS:
            v = cases(rows, d)
            gains = [mean(c["budget"]["2"]["decision_cost"]["loss"] - c["budget"]["2"]["matched_random"]["loss"]
                          for c in row["directions"][d]["cases"]) for row in rows]
            dscore = mean(c["budget"]["2"]["decision_cost"]["loss"] for c in v)
            random = mean(c["budget"]["2"]["matched_random"]["loss"] for c in v)
            oracle = mean(c["budget"]["2"]["matched_oracle_analysis_only"]["loss"] for c in v)
            budget_rows.append((NAMES[n], d, avg(c["budget"]["2"]["decision_at_most"]["loss"] for c in v),
                                avg(c["budget"]["2"]["threshold"]["loss"] for c in v), f"{dscore:.4f}",
                                avg(c["budget"]["2"]["decision_cost"]["panel_messages"] for c in v),
                                avg(c["budget"]["2"]["decision_cost"]["zero_fraction"] for c in v), ci(gains),
                                f"{100*(random-dscore)/max(1e-9,random-oracle):.1f}%"))
    table(lines, ("Dataset", "Task", "At most 2", "Threshold", "Decision + cost", "Messages", "Empty fraction",
                  "Loss change vs matched random [paired 95% CI]", "Matched-random-to-oracle gap recovered"), budget_rows)
    lines.extend(["Paired intervals use ten seed means within each fixed graph. The recovery fraction uses the random-to-oracle loss gap "
                  "with the selector's exact realized message count. That quantity differs from the full-to-unconstrained-oracle gap.",
                  "", "Accuracy, AUROC, and AP use the same test targets as the loss comparisons. Direct, aware LOO, and decision use exact K=2 in this table."])
    recon = []
    for n, rows in primary.items():
        for d in DIRECTIONS:
            v = cases(rows, d)
            for metric in (("accuracy",) if d == "text" else ("auc", "ap")):
                recon.append((NAMES[n], metric.upper(), avg(c["full"][metric] for c in v),
                              *(avg(c["budget"]["2"][k][metric] for c in v) for k in
                                ("direct_exact", "aware_loo_exact", "decision_exact", "decision_cost", "matched_random"))))
    table(lines, ("Dataset", "Metric", "Full", "Direct exact", "Aware LOO exact", "Decision exact",
                  "Decision + cost", "Matched random"), recon)
    lines.extend([
                  "", "### V4.6 · Uncertainty and external-method controls", "",
                  "The decision model estimates a predictive spread and risk uses mean minus beta times spread, with a cost for each message. "
                  "The development-selected fallback may use the full panel and is reported separately from strict-cap policies. "
                  "The three-seed comparator runs include eight MC dropout draws and a three-member ensemble. The table reports nominal 90% empirical interval coverage "
                  "and strict-cap loss at K=2. Coverage is an empirical diagnostic and carries no calibration guarantee."])
    table(lines, ("Dataset", "Task", "Mean/variance coverage", "MC dropout coverage", "Ensemble coverage",
                  "Risk", "Risk + full fallback"),
          ((NAMES[n], d, *(avg(c["scores"][k]["normal_90_coverage"] for c in v) for k in
                           ("decision", "mc_dropout", "deep_ensemble")),
            *(avg(c["budget"]["2"][k]["loss"] for c in v) for k in ("risk", "risk_fallback_full")))
           for n, rows in controls.items() for d in DIRECTIONS for v in [cases(rows, d)]))
    lines.extend(["Equal-count local reconstructions below use additive GAT attention, a straight-through top-K sparsifier inspired by NeuralSparse, "
                  "and a correctness-predicting utility proxy with predicted Shapley scores inspired by SGUL. "
                  "These are adaptations to the frozen reconstruction model with paired panels and targets. Published end-to-end benchmark numbers use other tasks and protocols. "
                  "The set-attention ablation adds one permutation-equivariant self-attention block before DeepSets pooling."])
    table(lines, ("Dataset", "Task", "GAT local", "NeuralSparse local", "SGUL local", "Set attention", "Decision"),
          ((NAMES[n], d, *(avg(c["budget"]["2"][k+"_exact"]["loss"] for c in v) for k in
                           ("gat_local", "neural_sparse_local", "sgul_local", "set_attention", "decision")))
           for n, rows in controls.items() for d in DIRECTIONS for v in [cases(rows, d)]))
    lines.extend(["### Larger panels, layer interventions, and generalization", "",
                  "All interventions preserve the graph and degree denominator. The edge branch suppresses the direct text component of incoming messages. "
                  "Layer ablations include first-only and both layers. With hidden candidate edges and no self loops, the target roots are locally recomputed while untouched messages keep their cached values.",
                  "", "This Cora table uses three seeds and the same 192-target cap within each protocol. The four-slot both-layer arm and the six-slot layer comparison are exploratory. "
                  "The re-split arm loads independently re-split V2 warm-link checkpoints and regenerated node splits."])
    experimental = ("cora_panel4_last", "cora_both", "cora_panel6_first", "cora_panel6", "cora_panel6_both", "cora_resplit")
    table(lines, ("Cora variant", "Task", "Full", "Decision K=2", "Matched-count oracle"),
          ((n, d, avg(c["full"]["loss"] for c in v), *(avg(c["budget"]["2"][k]["loss"] for c in v)
           for k in ("decision_cost", "matched_oracle_analysis_only")))
           for n in experimental for d in DIRECTIONS for v in [cases(load(n, 3), d)]))
    synthetic = ("synthetic_reference", "synthetic_low_homophily", "synthetic_weak_text", "synthetic_noisy_context")
    transfer = load("synthetic_reference_transfer", 3)
    lines.extend(["Synthetic controls vary homophily, text signal, and target-time false edges. In the transfer test, the denoiser and selector are fitted "
                  "on the reference graph and evaluated on newly generated graphs without fitting on their labels. The objective is the same K≤2 decision score. "
                  "Random and oracle receive the learned selector's exact count per target."])
    table(lines, ("Unseen synthetic regime", "Task", "Full", "Decision", "Matched random", "Matched oracle", "False-edge retention decision / random"),
          ((n, d, *(avg(c[k]["loss"] for c in v) for k in ("full", "learned", "random", "oracle_analysis_only")),
            " / ".join(avg(c["noise_retention"][k] for c in v if "noise_retention" in c) for k in ("learned", "random"))
            if any("noise_retention" in c for c in v) else "—")
           for n in synthetic for d in DIRECTIONS for v in [[c for r in transfer for c in r["independent_synthetic"][n][d]]]))
    lines.extend(["### V4.7 · Fine-tuning and measured runtime", "",
                  "For each direction, partial fine-tuning unfreezes the last GraphSAGE layer. End-to-end fine-tuning updates the compact denoiser and selector "
                  "jointly with refreshed subset losses and differentiable subset choice. The full-message control fine-tunes the same denoiser for the same 12 updates. "
                  "All tuning uses controller-training cases, and each arm scores paired test panels.",
                  "", "Measured latency includes graph and panel preparation, prediction probes where applicable, scoring, and final decoding. "
                  "Each value averages three seed medians after warm-up. The graph prefix and its full aggregation are still computed in this implementation. "
                  "Optional-message counts describe the selected panel only."])
    table(lines, ("Dataset", "Task", "Frozen selected", "Partial selected", "End-to-end selected", "Fine-tuned full",
                  "Full / selected ms", "Full / selected panel messages"),
          ((NAMES[n], d, *(avg(c["loss"] for r in rows for c in r["directions"][d]["fine_tuning"][k])
                          for k in ("frozen", "partial", "end_to_end", "full_control")),
            *((" / ".join(avg(r["directions"][d]["timing"][k][metric] for r in rows) for k in ("full", "selected")))
              for metric in ("median_ms", "panel_messages")))
           for n, rows in extras.items() for d in DIRECTIONS))
    lines.extend(["### V4.8 · Diffusion integration", "",
                  "Two free-running reconstruction iterations use the same sampled panel of up to 192 held-out tokens and edges per case. "
                  "The first iteration confidence-fills half, and the second completes the panel at half the initial clocks using the previous predictions. "
                  "The selector is reapplied on both iterations. The paired full-message rollout uses the same masks, initial targets, and number of reconstruction iterations. "
                  "The selected arm also uses counterfactual prediction probes and thus incurs additional work."])
    table(lines, ("Dataset", "Full token accuracy / edge AUROC", "Selected token accuracy / edge AUROC"),
          ((NAMES[n], *(" / ".join((avg(c["token_accuracy"] for c in v), avg(c["edge"]["auc"] for c in v)))
                        for key in ("full", "selected") for v in [[c for r in rows for c in r["diffusion"][key]]]))
           for n, rows in extras.items()))
    lines.extend(["### Results and reproduction", "",
                  "Primary paired results. " + " ".join(link(n, NAMES[n]) for n in NAMES), "",
                  "Fine-tuning, timing, and diffusion. " + " ".join(link(n+"_extra", NAMES[n]) for n in NAMES), "",
                  "Local baselines and uncertainty controls. " + " ".join(link(n+"_baselines", NAMES[n]) for n in NAMES), "",
                  "Layer, split, and synthetic studies. " + " ".join(link(n) for n in (*experimental, *synthetic, "synthetic_reference_transfer")), "",
                  "DeepSets provides the invariant aggregation. Set Transformer motivates the attention ablation. The decision objective is a softmax "
                  "over this pilot's enumerated subsets, following the decision-focused training principle. "
                  "[Deep Sets](https://arxiv.org/abs/1703.06114) [Set Transformer](https://arxiv.org/abs/1810.00825) "
                  "[Decision-Focused Learning](https://mlanthology.org/jair/2024/mandi2024jair-decisionfocused/)", "",
                  "```powershell", "python -m scripts.research_twinmask_v4",
                  "python -m scripts.research_twinmask_v4 --seeds 42 43 44 --extra --steps 60 --limit 256",
                  "python -m scripts.research_twinmask_v4 --seeds 42 43 44 --comparators --steps 60 --limit 256",
                  "foreach ($layer in 'first','last','both') { python -m scripts.research_twinmask_v4 --datasets cora --seeds 42 43 44 --panel 6 --layer $layer --steps 60 --limit 192 }",
                  "python -m scripts.research_twinmask_v4 --datasets cora --seeds 42 43 44 --steps 60 --limit 192 --tag panel4_last",
                  "python -m scripts.research_twinmask_v4 --datasets cora --seeds 42 43 44 --layer both --steps 60 --limit 192",
                  "python -m scripts.research_twinmask_v4 --datasets cora --seeds 42 43 44 --resplit --steps 60 --limit 192",
                  "python -m scripts.research_twinmask_v4 --datasets synthetic_reference synthetic_low_homophily synthetic_weak_text synthetic_noisy_context --seeds 42 43 44 --steps 60 --limit 192",
                  "python -m scripts.research_twinmask_v4 --datasets synthetic_reference --transfer --seeds 42 43 44 --steps 60 --limit 192",
                  "python -m unittest scripts.test_research_pilots scripts.test_research_twinmask_v3 scripts.test_research_twinmask_v4",
                  "python -m scripts.report_research_v4", "```", ""])
    readme = Path("README.md")
    content = readme.read_text(encoding="utf-8")
    marker = "<!-- research-v4-results -->"
    prior = "## TwinMask V3"
    before, _, _ = content.partition(marker)
    if marker not in content:
        before = content.split(prior, 1)[0]
    v3 = content[content.index(prior):]
    readme.write_text(before+marker+"\n\n"+"\n".join(lines)+"\n"+v3, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
