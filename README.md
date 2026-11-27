# TwinMask — Joint Masked Diffusion over Text and Topology in Text-Attributed Graphs

> **Status: research prototype, work in progress (last updated 2026-11-16).**
> Phases 0–3 of the plan are implemented and have been run, and the DIVERGE-style ensemble study is complete for Cora and CiteSeer (4 seeds, fair tuning). Phases 1–3 use a single seed. Negative and mixed findings are reported on purpose; see [Limitations](#9-limitations-and-honest-assessment).

---

<!-- research-v6-results -->

## TwinMask V5.1 and V6 · Hyperparameter optimization and transfer

V5.1 removes the out-of-scope CLI-variable reference in the mechanism study. The CLI now regenerates all 54 synthetic rows. The [artifact manifest](results/research_twinmask_v6/manifest.json) records the repository base commit, current source hashes, graph reference hashes, and saved checkpoint hashes.

**What the longer run changes.** The paired development sweep compares the joint-minus-MLM text CE gap at 32 and 1,000 updates. On Cora, it was +0.0244 and +0.0346. On PubMed, it was +0.0603 and +0.0005. The late Cora edge-loss curves also show overfitting, which matters when choosing the checkpoint.

### Optimization protocol

All eight reconstruction families get 12 Optuna TPE trials per dataset with median pruning and common 64-update screening caps. The best three completed configurations are checked on a second training seed using development only. For MLM, structure GNN, GraphSAGE, and TwinMask on Cora and PubMed, the study expands to 30 trials per family. The top two then train for up to 300 updates on two development seeds, and the configuration and common checkpoint step are frozen before locked evaluation. The main objectives are CE for MLM, BCE for edge-only models, and the equally weighted mean of CE/log(vocabulary) and BCE/log(2) for joint models. Family comparisons use loss for the applicable task only.

The untouched search-test pool is partitioned into two disjoint sets of node and link targets. Search fit/development have no access to their target labels or positive edges as observations. Each locked fold uses a separate fit/dev repartition and fresh model parameters. Transductive observed context and repeated use of the same real graphs remain part of the protocol. Amazon Computers uses one 6,000-node induced sample, with two target folds. The default comparator uses 32 updates under the same V6 data and training protocol.

Learning rate, weight decay, width, layers, dropout, branch LR, cosine scheduling, fusion scale, modality dropout, rollout frequency, and task weights are logged for all applicable families. The table shows the selected configurations and held-out scores. Cora/PubMed rows prefer the extended result where available. The last column is tuned minus default loss on the same folds.

| Dataset | Model | Trials | LR / WD | Width / layers / dropout | Updates | Test CE or BCE | Loss shift vs default |
|---|---|---|---|---|---|---|---|
| Cora | mlm | 30 | 0.007 / 0.06 | 64 / 2 / 0 | 300 | 4.6103 | -0.9960 |
| Cora | structure | 30 | 0.00061 / 0.0068 | 256 / 3 / 0.3 | 64 | 0.5880 | -0.0120 |
| Cora | structure_gat | 12 | 0.00042 / 0.015 | 256 / 1 / 0 | 64 | 0.5915 | -0.0131 |
| Cora | text_link | 12 | 0.00014 / 1.5e-05 | 256 / 2 / 0 | 64 | 0.6848 | +0.0019 |
| Cora | gae | 12 | 0.0012 / 4.9e-06 | 256 / 3 / 0.3 | 64 | 0.6619 | -0.0137 |
| Cora | sage | 30 | 0.0057 / 1.5e-05 | 128 / 1 / 0 | 160 | 4.7076 / 0.5945 | -0.9663 / -0.0097 |
| Cora | gat | 12 | 0.0053 / 0.089 | 48 / 1 / 0 | 64 | 5.3720 / 0.6042 | -0.2919 / -0.0059 |
| Cora | twinmask | 30 | 0.0069 / 0.0063 | 64 / 2 / 0 | 100 | 4.9273 / 0.6029 | -0.7517 / -0.0012 |
| CiteSeer | mlm | 12 | 0.0081 / 7.1e-06 | 256 / 1 / 0 | 64 | 5.6836 | -0.2893 |
| CiteSeer | structure | 12 | 0.0047 / 0.00047 | 48 / 2 / 0 | 64 | 0.5701 | -0.0245 |
| CiteSeer | structure_gat | 12 | 0.00067 / 0.022 | 256 / 3 / 0 | 64 | 0.5775 | -0.0106 |
| CiteSeer | text_link | 12 | 0.0036 / 0.00018 | 64 / 2 / 0 | 32 | 0.7027 | +0.0060 |
| CiteSeer | gae | 12 | 0.003 / 0.01 | 48 / 2 / 0 | 32 | 0.6283 | +0.0000 |
| CiteSeer | sage | 12 | 0.0085 / 2.8e-05 | 64 / 2 / 0.3 | 64 | 5.7316 / 0.5882 | -0.2740 / -0.0201 |
| CiteSeer | gat | 12 | 0.0084 / 7.3e-05 | 48 / 2 / 0.3 | 64 | 5.8118 / 0.5922 | -0.1935 / -0.0030 |
| CiteSeer | twinmask | 12 | 0.0062 / 0.001 | 48 / 3 / 0 | 64 | 5.6025 / 0.5910 | -0.4147 / -0.0333 |
| PubMed | mlm | 30 | 0.0069 / 4.9e-06 | 256 / 3 / 0 | 300 | 3.6573 | -1.4372 |
| PubMed | structure | 30 | 0.0031 / 8.6e-05 | 256 / 3 / 0 | 300 | 0.3437 | -0.0628 |
| PubMed | structure_gat | 12 | 0.0064 / 1.2e-05 | 48 / 3 / 0 | 64 | 0.3628 | -0.0842 |
| PubMed | text_link | 12 | 0.00042 / 0.015 | 256 / 1 / 0 | 32 | 0.6931 | -0.0009 |
| PubMed | gae | 12 | 0.00088 / 2.8e-06 | 256 / 3 / 0.3 | 64 | 0.5805 | -0.0075 |
| PubMed | sage | 30 | 0.0069 / 0.0013 | 64 / 2 / 0.2 | 300 | 3.7942 / 0.3469 | -1.4470 / -0.0726 |
| PubMed | gat | 12 | 0.0084 / 2.9e-05 | 128 / 2 / 0.3 | 64 | 4.4748 / 0.3796 | -0.7579 / -0.0598 |
| PubMed | twinmask | 30 | 0.0077 / 0.056 | 64 / 3 / 0 | 300 | 3.6953 / 0.3644 | -1.5616 / -0.0624 |
| Amazon Computers | mlm | 12 | 0.0047 / 0.00047 | 48 / 2 / 0 | 64 | 5.5377 | -0.2859 |
| Amazon Computers | structure | 12 | 0.0047 / 0.00047 | 48 / 2 / 0 | 64 | 0.5028 | +0.0059 |
| Amazon Computers | structure_gat | 12 | 0.0017 / 1.4e-06 | 256 / 3 / 0 | 64 | 0.4991 | +0.0030 |
| Amazon Computers | text_link | 12 | 0.00042 / 0.015 | 256 / 1 / 0 | 64 | 0.6754 | -0.0054 |
| Amazon Computers | gae | 12 | 0.00014 / 1.5e-05 | 256 / 2 / 0 | 64 | 0.6322 | +0.0022 |
| Amazon Computers | sage | 12 | 0.0085 / 7.4e-05 | 48 / 2 / 0.3 | 64 | 5.6015 / 0.4978 | -0.2612 / +0.0017 |
| Amazon Computers | gat | 12 | 0.0045 / 0.00075 | 256 / 3 / 0 | 64 | 5.5232 / 0.4953 | -0.3182 / -0.0015 |
| Amazon Computers | twinmask | 12 | 0.007 / 5.8e-06 | 48 / 2 / 0 | 64 | 5.4975 / 0.5165 | -0.3727 / +0.0151 |

Full profiles, intermediate objectives, completed/pruned trial counts, and per-fold outputs are in the JSON files below.

![Development trial history and development loss-weight Pareto points](results/research_twinmask_v6/search_and_pareto.png)

### Convergence, directional training, and gradient interventions

Cora and PubMed convergence runs use paired training masks, matched updates from 32 to 1,000 and a fixed 48-wide model. The measured development loss is held out from weight fitting. Training losses are averages since the preceding checkpoint.

![Paired training and development loss curves at 32, 100, 300, 600 and 1000 updates](results/research_twinmask_v6/convergence.png)

The table uses the first locked fold, equal 100-update budgets, and a shared initialization for arms with the same architecture. Gradient cosine is recorded before gradient projection on parameters differentiable for both tasks. GradNorm denotes a closed-form balancing proxy and CAGrad uses an approximate 65-point two-task solver.

| Dataset | Arm | Token CE | Edge BCE | Edge AUROC | Mean gradient cosine | Learned text / edge gates |
|---|---|---|---|---|---|---|
| Cora | independent | 5.2331 | 0.5941 | 0.7585 | 0.0000 | — |
| Cora | structure guided | 5.2866 | 0.5952 | 0.7534 | -0.0644 | — |
| Cora | text guided | 5.2321 | 0.5925 | 0.7606 | -0.0202 | — |
| Cora | bidirectional | 5.2814 | 0.5946 | 0.7538 | 0.0392 | — |
| Cora | stop gradient | 5.2996 | 0.5923 | 0.7633 | — | — |
| Cora | pcgrad | 5.2924 | 0.5942 | 0.7544 | 0.0310 | — |
| Cora | gradnorm proxy | 5.3801 | 0.5890 | 0.7632 | 0.0435 | — |
| Cora | cagrad grid | 5.3122 | 0.5914 | 0.7570 | -0.0198 | — |
| Cora | selected loss weight | 5.2872 | 0.5936 | 0.7501 | -0.0053 | — |
| Cora | structural statistics | 5.2850 | 0.5881 | 0.7601 | 0.0702 | — |
| Cora | small transformer | 5.6872 | 0.5914 | 0.7520 | 0.0565 | — |
| Cora | learned gates | 5.2523 | 0.5929 | 0.7509 | -0.0147 | 0.466 / 0.490 |
| PubMed | independent | 4.4987 | 0.3940 | 0.8976 | 0.0000 | — |
| PubMed | structure guided | 4.5723 | 0.4057 | 0.8898 | 0.0172 | — |
| PubMed | text guided | 4.4958 | 0.3992 | 0.8960 | 0.0106 | — |
| PubMed | bidirectional | 4.5675 | 0.4085 | 0.8895 | -0.0165 | — |
| PubMed | stop gradient | 4.5932 | 0.3994 | 0.8962 | — | — |
| PubMed | pcgrad | 4.5713 | 0.4057 | 0.8901 | -0.0068 | — |
| PubMed | gradnorm proxy | 4.6168 | 0.4014 | 0.8938 | -0.0153 | — |
| PubMed | cagrad grid | 4.5946 | 0.4014 | 0.8934 | 0.0141 | — |
| PubMed | selected loss weight | 4.5744 | 0.4192 | 0.8844 | -0.0755 | — |
| PubMed | structural statistics | 4.5451 | 0.3918 | 0.8990 | 0.0294 | — |
| PubMed | small transformer | 4.9334 | 0.4000 | 0.8943 | -0.1041 | — |
| PubMed | learned gates | 4.5386 | 0.3991 | 0.8954 | -0.0059 | 0.458 / 0.496 |

![Shared gradient alignment under selected mitigation methods](results/research_twinmask_v6/gradient_alignment.png)

Hard-negative controls use only observed degrees or observed two-hop paths to propose negatives, then reject every true reference edge. The candidate composition changes, so ranking results represent a stress test rather than interchangeable samples.

| Dataset | Negatives | Matched proposals | Edge BCE | Edge AUROC |
|---|---|---|---|---|
| Cora | uniform | — | 0.5946 | 0.7538 |
| Cora | degree | 202 | 0.7848 | 0.5840 |
| Cora | two_hop | 202 | 0.8449 | 0.5283 |
| PubMed | uniform | — | 0.4085 | 0.8895 |
| PubMed | degree | 1700 | 0.8494 | 0.6142 |
| PubMed | two_hop | 1700 | 0.8920 | 0.5873 |

### Encoder and vocabulary controls

Capacity changes below share exact token positions and truth IDs from the common 256-token fitting-derived vocabulary. Raw CE still scores each model's native alphabet. DistilRoBERTa uses cached contextual representations of the actual masked inputs and a trained projection. Its short mask-bank feasibility study uses four development settings and a single locked fold.

| Cora encoder | Task | Positions | Vocabulary | Test CE | Token accuracy |
|---|---|---|---|---|---|
| CNN | mlm | 16 | 256 | 3.9292 | 0.1890 |
| CNN | sage | 16 | 256 | 3.9419 | 0.1893 |
| CNN | mlm | 16 | 1024 | 4.4129 | 0.1621 |
| CNN | sage | 16 | 1024 | 4.4723 | 0.1703 |
| CNN | mlm | 16 | 4096 | 4.9233 | 0.1427 |
| CNN | sage | 16 | 4096 | 5.0288 | 0.1429 |
| CNN | mlm | 32 | 1024 | 4.4828 | 0.1693 |
| CNN | sage | 32 | 1024 | 4.5801 | 0.1587 |
| CNN | mlm | 64 | 1024 | 4.7844 | 0.1428 |
| CNN | sage | 64 | 1024 | 4.8657 | 0.1314 |
| CNN | mlm | 128 | 1024 | 4.9991 | 0.1150 |
| CNN | sage | 128 | 1024 | 5.0532 | 0.1010 |


| Cora frozen encoder | Selected LR | Fusion | Updates | Test CE | Test token accuracy |
|---|---|---|---|---|---|
| mlm | 0.003 | 0.5 | 32 | 5.9587 | 0.0560 |
| sage | 0.001 | 1.0 | 64 | 5.9265 | 0.0671 |

### Teacher calibration, local utility, and diffusion policies

Each selector family gets ten TPE trials per direction and dataset on Cora and PubMed. The search covers LR, decay, width, 70/200/500-update caps, message cost, rank weight, and decision weight or temperature where applicable. Teachers use independently tuned backbones and development-only temperature/family choice. The mixture and stacking searches use development labels, as does the choice of rollout horizon from 1, 2, 4, and 8. The selected configurations and evaluated stages use the same locked targets as the reconstruction study.

| Dataset | Task | Stage | LR / width / updates | Rank / decision / cost | Locked loss | Matched random |
|---|---|---|---|---|---|---|
| Cora | text | loo | 0.0063 / 32 / 200 | 0.15 / 0 / 0.1 | 5.1785 | 5.1792 |
| Cora | text | shapley | 0.00024 / 48 / 70 | 0 / 0 / 0 | 5.1768 | 5.1765 |
| Cora | text | set | 0.00091 / 48 / 70 | 0.15 / 0 / 0.04 | 5.1763 | 5.1770 |
| Cora | text | decision | 0.0059 / 96 / 70 | 1 / 0.3 / 0 | 5.1771 | 5.1775 |
| Cora | text | distilled | 0.00024 / 48 / 35 | 0 / 0 / 0 | 5.1773 | 5.1775 |
| Cora | edge | loo | 0.0018 / 96 / 200 | 0.5 / 0 / 0.01 | 0.5875 | 0.5876 |
| Cora | edge | shapley | 0.0063 / 32 / 300 | 0.15 / 0 / 0.1 | 0.5881 | 0.5880 |
| Cora | edge | set | 0.01 / 48 / 500 | 1 / 0 / 0 | 0.5875 | 0.5875 |
| Cora | edge | decision | 0.0021 / 48 / 500 | 1 / 0.5 / 0 | 0.5878 | 0.5877 |
| Cora | edge | distilled | 0.01 / 48 / 300 | 1 / 0 / 0 | 0.5874 | 0.5876 |
| PubMed | text | loo | 0.00024 / 48 / 200 | 0 / 0 / 0 | 4.3095 | 4.3101 |
| PubMed | text | shapley | 0.0063 / 32 / 500 | 0.15 / 0 / 0.1 | 4.3101 | 4.3101 |
| PubMed | text | set | 0.01 / 48 / 35 | 1 / 0 / 0 | 4.3099 | 4.3103 |
| PubMed | text | decision | 0.0011 / 48 / 500 | 0.5 / 0.5 / 0 | 4.3101 | 4.3107 |
| PubMed | text | distilled | 0.00024 / 48 / 35 | 0 / 0 / 0 | 4.3122 | 4.3108 |
| PubMed | edge | loo | 0.0063 / 32 / 200 | 0.15 / 0 / 0.1 | 0.3818 | 0.3818 |
| PubMed | edge | shapley | 0.0063 / 32 / 200 | 0.15 / 0 / 0.1 | 0.3818 | 0.3818 |
| PubMed | edge | set | 0.00024 / 48 / 70 | 0 / 0 / 0 | 0.3820 | 0.3820 |
| PubMed | edge | decision | 0.00029 / 32 / 35 | 0.15 / 0.1 / 0.1 | 0.3819 | 0.3819 |
| PubMed | edge | distilled | 0.0063 / 32 / 200 | 0.15 / 0 / 0.1 | 0.3813 | 0.3814 |


| Dataset | Task | Horizon | Mixture temperature | Expected hard loss | Uniform | Tuned mixture | Tuned stacking |
|---|---|---|---|---|---|---|---|
| Cora | text | 1 | 0.01 | 5.1776 | 5.1785 | 5.1779 | 5.1777 |
| Cora | edge | 1 | 0.01 | 0.5867 | 0.5873 | 0.5873 | 0.5874 |
| PubMed | text | 1 | 1.0 | 4.3111 | 4.3109 | 4.3109 | 4.3108 |
| PubMed | edge | 1 | 0.01 | 0.3809 | 0.3819 | 0.3817 | 0.3819 |


| Dataset | Task | Unscaled / calibrated NLL | Unscaled / calibrated Brier | Unscaled / calibrated ECE |
|---|---|---|---|---|
| Cora | text | 5.1567 / 5.1567 | 0.9374 / 0.9374 | 0.0258 / 0.0258 |
| Cora | edge | 0.5802 / 0.5865 | 0.1986 / 0.2018 | 0.0804 / 0.0893 |
| PubMed | text | 4.4696 / 4.4696 | 0.8092 / 0.8092 | 0.0314 / 0.0314 |
| PubMed | edge | 0.3638 / 0.3700 | 0.1152 / 0.1161 | 0.0392 / 0.0432 |

### Embedding denoiser and ensemble controls

On Cora and CiteSeer, the shared diffusion denoiser uses a separate ten-trial search. The raw, Gaussian and diffusion classifier arms each get ten development trials. The matched raw-head comparison fixes the classifier, policy, seeds and Gaussian cosine distortion target.

| Dataset | Arm | Hidden / LR / updates | Test accuracy | Test NLL |
|---|---|---|---|---|
| Cora | raw | 256 / 0.00035 / 40 | 0.8398 | 0.4804 |
| Cora | diffusion | 64 / 0.0022 / 40 | 0.8465 | 0.5110 |
| Cora | gaussian | 64 / 0.0022 / 40 | 0.8471 | 0.5136 |
| CiteSeer | raw | 256 / 0.00035 / 40 | 0.7644 | 0.6821 |
| CiteSeer | diffusion | 64 / 0.0022 / 40 | 0.7727 | 0.6902 |
| CiteSeer | gaussian | 64 / 0.0022 / 40 | 0.7743 | 0.6937 |

![Held-out classification accuracy from tuned ensemble arms](results/research_twinmask_v6/ensemble_controls.png)

### Original V2 backbone and V3/V4.1 selector audit

On Cora, the original SparseTwinMask decoder and graph mixer are trained from scratch in a ten-trial development study and evaluated on the two reserved V6 folds. For the frozen original V2 checkpoints, we also tune the V3/V4.1 message-panel LOO, Shapley, set, and decision stages with ten trials for each direction and family. Those selector results use the matched-context historical audit split and are exploratory. This pilot preserves the historical branch behavior for comparison.

| Original V2 | LR | Width | Updates | Locked token CE | Locked edge BCE |
|---|---|---|---|---|---|
| Cora | 0.00292 | 96 | 64 | 5.4167 | 0.6848 |


| Task | Local selector | LR / width / updates | Held-out loss | Matched random |
|---|---|---|---|---|
| text | loo | 0.01 / 48 / 70 | 4.8668 | 4.8694 |
| text | shapley | 0.00024 / 48 / 70 | 4.8713 | 4.8718 |
| text | set | 0.01 / 48 / 300 | 4.8683 | 4.8693 |
| text | decision | 0.0059 / 96 / 120 | 4.8722 | 4.8733 |
| edge | loo | 0.0018 / 96 / 120 | 0.6839 | 0.6854 |
| edge | shapley | 0.00011 / 96 / 200 | 0.6856 | 0.6834 |
| edge | set | 0.01 / 48 / 500 | 0.6826 | 0.6830 |
| edge | decision | 0.00032 / 32 / 70 | 0.6839 | 0.6843 |

### Artifacts and commands

Scope. The expanded equal-budget search is limited to four families on Cora and PubMed. The local legacy, pretrained, and orthogonal capacity studies are Cora feasibility runs. There is no RoBERTa PEFT run, full-graph Amazon run, or re-tuning of every published competitor or archived script.

Backbone trials and locked folds. [Cora search](results/research_twinmask_v6/cora_search.json) [Cora locked](results/research_twinmask_v6/cora_locked.json) [Cora default](results/research_twinmask_v6/cora_default.json) [CiteSeer search](results/research_twinmask_v6/citeseer_search.json) [CiteSeer locked](results/research_twinmask_v6/citeseer_locked.json) [CiteSeer default](results/research_twinmask_v6/citeseer_default.json) [PubMed search](results/research_twinmask_v6/pubmed_search.json) [PubMed locked](results/research_twinmask_v6/pubmed_locked.json) [PubMed default](results/research_twinmask_v6/pubmed_default.json) [Amazon Computers search](results/research_twinmask_v6/computer_search.json) [Amazon Computers locked](results/research_twinmask_v6/computer_locked.json) [Amazon Computers default](results/research_twinmask_v6/computer_default.json)

Expanded search, transfer diagnostics, and downstream stages. [cora_deep_search](results/research_twinmask_v6/cora_deep_search.json) [cora_deep_locked](results/research_twinmask_v6/cora_deep_locked.json) [cora_analysis](results/research_twinmask_v6/cora_analysis.json) [cora_weight_sweep](results/research_twinmask_v6/cora_weight_sweep.json) [cora_gates](results/research_twinmask_v6/cora_gates.json) [cora_selectors](results/research_twinmask_v6/cora_selectors.json) [cora_calibration](results/research_twinmask_v6/cora_calibration.json) [pubmed_deep_search](results/research_twinmask_v6/pubmed_deep_search.json) [pubmed_deep_locked](results/research_twinmask_v6/pubmed_deep_locked.json) [pubmed_analysis](results/research_twinmask_v6/pubmed_analysis.json) [pubmed_weight_sweep](results/research_twinmask_v6/pubmed_weight_sweep.json) [pubmed_gates](results/research_twinmask_v6/pubmed_gates.json) [pubmed_selectors](results/research_twinmask_v6/pubmed_selectors.json) [pubmed_calibration](results/research_twinmask_v6/pubmed_calibration.json)

Additional encoder/classifier controls. [cora_capacity](results/research_twinmask_v6/cora_capacity.json) [cora_pretrained](results/research_twinmask_v6/cora_pretrained.json) [cora_ensemble](results/research_twinmask_v6/cora_ensemble.json) [citeseer_ensemble](results/research_twinmask_v6/citeseer_ensemble.json) [cora_legacy](results/research_twinmask_v6/cora_legacy.json)

```powershell
# Requires the existing TwinMask dependencies plus optuna and matplotlib.
python -m scripts.research_twinmask_v5 --mechanism-study
python -m scripts.research_twinmask_v6 --trials 12 --trial-updates 64
python -m scripts.research_twinmask_v6 --default-control
python -m scripts.research_twinmask_v6_refine --trials 30
python -m scripts.research_twinmask_v6_analysis
python -m scripts.research_twinmask_v6_analysis --gates-only
python -m scripts.research_twinmask_v6_selectors --trials 10
python -m scripts.research_twinmask_v6_selectors --calibration-only
python -m scripts.research_twinmask_v6_pretrained
python -m scripts.research_twinmask_v6_capacity
python -m scripts.research_twinmask_v6_ensemble
python -m scripts.research_twinmask_v6_legacy
python -m unittest scripts.test_research_pilots scripts.test_research_twinmask_v3 scripts.test_research_twinmask_v4 scripts.test_research_twinmask_v5 scripts.test_research_twinmask_v6
python -m scripts.report_research_twinmask_v6
```

<!-- research-v41-v5 -->

## TwinMask V4.1 · Correctness and protocol audit

V4.1 aligns the local raw graphs to the cached node order by exact text comparison and confirms that every saved positive is present in the raw graph. The complete canonical edge references, including Cold–Cold links, are saved in `data/references/`. Warm negatives are generated against the full reference and are unique and disjoint across controller train, development, and test. The revised sampler also prevents validation/test negative overlap and duplicate cold negatives in newly prepared datasets. The old frozen backbones are retained so the matched/growing comparison isolates the selector context protocol. Those historical checkpoint weights were fitted under V2.

The matched and growing arms share regenerated candidate pools, target node splits, masks, and seed IDs 42–44. Both arms train on the same reduced base graph. Growing adds the controller and development positive pools to the test-only base. Each episode separately adds its observed positive candidates. Thus the arms differ in base connectivity while evaluating identical target labels.

| Dataset | Raw edges | Omitted positives in legacy cache | Legacy false negatives | Legacy val/test negative overlap | Legacy duplicate cold negatives | Matched / growing test base |
|---|---|---|---|---|---|---|
| Cora | 5278 | 18 | 0 | 0 | 1 | 3237 / 4045 |
| CiteSeer | 4225 | 14 | 0 | 0 | 1 | 2557 / 3195 |
| PubMed | 44324 | 103 | 0 | 0 | 0 | 27199 / 33997 |

The audit of stored legacy validation, test, and cold negatives found zero true positives among those negatives. This check concerns the stored pools and does not establish equivalence to the old training sampler.

Lower token CE and edge BCE are better. K=2 is an upper bound on optional messages. Random and oracle receive the decision policy's realized count on each target.

| Dataset | Task | Context | Full | Decision + cost | Count-matched random | Equal-count oracle regret | Expected regret | Dev-scaled 90% coverage |
|---|---|---|---|---|---|---|---|---|
| Cora | text | matched | 4.9159 | 4.9044 | 4.9041 | 0.0081 | 0.0541 | 0.9051 |
| Cora | edge | matched | 0.6651 | 0.6645 | 0.6653 | 0.0174 | 0.0316 | 0.8977 |
| Cora | text | growing | 4.9142 | 4.9025 | 4.9023 | 0.0094 | 0.0541 | 0.9063 |
| Cora | edge | growing | 0.6558 | 0.6565 | 0.6568 | 0.0177 | 0.0305 | 0.9091 |
| CiteSeer | text | matched | 5.2378 | 5.2346 | 5.2347 | 0.0011 | 0.0591 | 0.8983 |
| CiteSeer | edge | matched | 0.6585 | 0.6594 | 0.6595 | 0.0144 | 0.0398 | 0.8975 |
| CiteSeer | text | growing | 5.2353 | 5.2311 | 5.2311 | 0.0015 | 0.0655 | 0.8951 |
| CiteSeer | edge | growing | 0.6482 | 0.6495 | 0.6495 | 0.0166 | 0.0420 | 0.8955 |
| PubMed | text | matched | 3.9883 | 3.9822 | 3.9826 | 0.0044 | 0.0389 | 0.8955 |
| PubMed | edge | matched | 0.4991 | 0.4954 | 0.4978 | 0.0111 | 0.0310 | 0.9098 |
| PubMed | text | growing | 3.9910 | 3.9865 | 3.9872 | 0.0049 | 0.0433 | 0.9011 |
| PubMed | edge | growing | 0.4922 | 0.4872 | 0.4904 | 0.0107 | 0.0305 | 0.9169 |

Risk models now learn utility relative to Full directly, with exact zero mean and uncertainty for Full. The held-out interval uses the development 90th percentile of standardized non-full residuals. Reported coverage is empirical because messages and subsets are dependent.

The next table uses the matched arm and the prediction-aware LOO estimator. The rank metrics exclude targets with no non-tied pair. Harmful precision and recall use LOO utility below −0.001 and predicted utility at most zero.

| Dataset | Task | Global ρ | Within-target ρ | Rank coverage | Pairwise accuracy | Top-1 | Harmful precision | Harmful recall |
|---|---|---|---|---|---|---|---|---|
| Cora | text | 0.0462 | 0.0352 | 0.6364 | 0.5067 | 0.3964 | 0.5439 | 0.5644 |
| Cora | edge | 0.0508 | 0.0301 | 0.9452 | 0.5179 | 0.3590 | 0.4794 | 0.4069 |
| CiteSeer | text | 0.0248 | -0.0081 | 0.3571 | 0.4986 | 0.3984 | 0.5041 | 0.5734 |
| CiteSeer | edge | 0.0531 | 0.0611 | 0.7824 | 0.5221 | 0.4117 | 0.4902 | 0.5992 |
| PubMed | text | 0.0949 | 0.1155 | 0.4017 | 0.5453 | 0.3996 | 0.5201 | 0.5732 |
| PubMed | edge | 0.1657 | 0.0857 | 0.9030 | 0.5217 | 0.4211 | 0.4337 | 0.6145 |

Corrected diffusion clocks scale each original rate by the fraction of the original hidden entries still pending. Unqueried hidden entries count toward the clock. The paired replay below evaluates two passes.

| Matched context | Full token accuracy / edge AUROC | Selected token accuracy / edge AUROC | Selected second-call text / edge clock |
|---|---|---|---|
| Cora | 0.1571 / 0.6587 | 0.1584 / 0.6562 | 0.5841 / 0.4977 |
| CiteSeer | 0.1215 / 0.6814 | 0.1215 / 0.6734 | 0.5870 / 0.4715 |
| PubMed | 0.3234 / 0.8351 | 0.3242 / 0.8406 | 0.5981 / 0.5880 |

V4.1 retains the V4 direct-component intervention for comparability. A test compares the cached two-layer local intervention to direct recomputation. The latency remains that of the existing full graph aggregation and local probes, separate from the number of optional messages retained.

Per-seed results. [Cora matched](results/research_twinmask_v41/cora_matched.json) [Cora growing](results/research_twinmask_v41/cora_growing.json) [CiteSeer matched](results/research_twinmask_v41/citeseer_matched.json) [CiteSeer growing](results/research_twinmask_v41/citeseer_growing.json) [PubMed matched](results/research_twinmask_v41/pubmed_matched.json) [PubMed growing](results/research_twinmask_v41/pubmed_growing.json)

## TwinMask V5 · Expected utility and isolated cross-modal transfer

V5 trains text and topology encoders independently in the forward computation and introduces cross-modal evidence only in fusion. With all optional groups suppressed, the text decoder depends on the local text stream and the edge decoder uses the topology stream. At full weights the subset decoder reproduces the ordinary joint logits. The four groups are a structural self representation and three deterministic incoming-structure groups for token targets, and two endpoint text representations plus their two aggregated text neighborhoods for edge targets.

The locked run grid uses split seeds **2101, 2113** and model seeds **71, 83, 97**. Every split trains fresh weights. The same base edges support fitting, development, and evaluation. Reference-backed warm pools are disjoint and historical validation/test citation positives are excluded from these new pools. Histories on the citation graphs still make them exploratory as corpora. The new Amazon Computers experiment uses two deterministic uniform 6,000-node induced samples from the 87,229-node raw graph and independently splits the induced pairs. This scope allows comparable GPU runs and does not estimate performance on the full graph.

**Main finding.** Structural conditioning worsened mean text CE in 11 of 12 joint architecture–dataset comparisons. Joint training also raised mean text CE relative to the separately trained MLM in all 12 comparisons. Text conditioning for edges varied by architecture and dataset. Expected-utility selection produced small average improvements on some comparisons, while all paired expected-versus-random loss intervals included zero.

### Conditioning versus joint training

Positive gain means the joint arm lowers loss. Conditioning gain switches opposite-modality fusion off in the same checkpoint. Joint-training gain compares the joint checkpoint with a separately trained text-only MLM or structure-only GNN. The GAT and GraphSAGE controls are one-shot, the GAE is an inner-product structural reconstruction baseline, and the TwinMask arm adds free-prediction feedback during training.

| Dataset | Architecture | Structure → text | Text → edge | Joint training text gain | Joint training edge gain |
|---|---|---|---|---|---|
| Cora | sage | -0.0047 | -0.0015 | -0.0758 | -0.0005 |
| Cora | gat | 0.0020 | 0.0018 | -0.0895 | 0.0045 |
| Cora | twinmask | -0.0107 | -0.0005 | -0.1081 | -0.0050 |
| CiteSeer | sage | -0.0035 | -0.0032 | -0.0712 | -0.0082 |
| CiteSeer | gat | -0.0009 | 0.0004 | -0.0925 | 0.0023 |
| CiteSeer | twinmask | -0.0097 | -0.0046 | -0.1039 | -0.0112 |
| PubMed | sage | -0.0302 | 0.0005 | -0.1171 | -0.0114 |
| PubMed | gat | -0.0321 | 0.0004 | -0.1680 | -0.0191 |
| PubMed | twinmask | -0.0206 | -0.0000 | -0.1728 | -0.0095 |
| Amazon Computers | sage | -0.0090 | 0.0003 | -0.0581 | 0.0011 |
| Amazon Computers | gat | -0.0028 | 0.0006 | -0.0642 | 0.0040 |
| Amazon Computers | twinmask | -0.0097 | 0.0012 | -0.0719 | 0.0039 |


| Dataset | Text-only MLM CE | Structure-only GNN BCE | Text-only link BCE | GAE BCE |
|---|---|---|---|---|
| Cora | 5.3886 | 0.6163 | 0.6884 | 0.6603 |
| CiteSeer | 5.8405 | 0.6163 | 0.6931 | 0.6495 |
| PubMed | 4.9665 | 0.3814 | 0.6905 | 0.5839 |
| Amazon Computers | 5.6514 | 0.4880 | 0.6902 | 0.5937 |

### Fixed-target timestep analysis

One held-out audit set stays masked at every timestep. Static corruption redraws the known context for each step. The free-running and teacher-forced arms reveal the same scheduled positions. Only teacher forcing inserts gold feedback, for diagnostic analysis. All three arms follow identical actual remaining-mask clocks. The four-step table gives signed text CE and edge BCE conditioning gains in chronological order.

| Dataset | Mode | Structure → text by step | Text → edge by step |
|---|---|---|---|
| Cora | static | -0.0124 / -0.0138 / -0.0156 / -0.0183 | -0.0048 / -0.0045 / -0.0043 / -0.0037 |
| Cora | free | -0.0124 / -0.0084 / -0.0051 / -0.0037 | -0.0048 / -0.0044 / -0.0042 / -0.0039 |
| Cora | teacher | -0.0124 / -0.0134 / -0.0154 / -0.0182 | -0.0048 / -0.0044 / -0.0042 / -0.0040 |
| CiteSeer | static | -0.0037 / -0.0066 / -0.0086 / -0.0108 | 0.0011 / 0.0013 / 0.0014 / 0.0021 |
| CiteSeer | free | -0.0037 / -0.0029 / -0.0020 / -0.0017 | 0.0011 / 0.0014 / 0.0015 / 0.0017 |
| CiteSeer | teacher | -0.0037 / -0.0064 / -0.0080 / -0.0109 | 0.0011 / 0.0011 / 0.0013 / 0.0017 |
| PubMed | static | -0.0259 / -0.0284 / -0.0307 / -0.0342 | -0.0003 / -0.0005 / -0.0006 / -0.0007 |
| PubMed | free | -0.0259 / -0.0217 / -0.0172 / -0.0129 | -0.0003 / -0.0005 / -0.0005 / -0.0006 |
| PubMed | teacher | -0.0259 / -0.0280 / -0.0305 / -0.0340 | -0.0003 / -0.0005 / -0.0006 / -0.0006 |
| Amazon Computers | static | -0.0142 / -0.0151 / -0.0163 / -0.0167 | 0.0002 / 0.0002 / -0.0001 / 0.0000 |
| Amazon Computers | free | -0.0142 / -0.0115 / -0.0091 / -0.0071 | 0.0002 / 0.0002 / 0.0002 / -0.0000 |
| Amazon Computers | teacher | -0.0142 / -0.0150 / -0.0157 / -0.0167 | 0.0002 / 0.0002 / 0.0002 / 0.0003 |

For the 1, 2, 4, and 8 step checks, the table reports final-step free-running conditional utility on the same audit targets.

| Dataset | Task | 1 step | 2 steps | 4 steps | 8 steps |
|---|---|---|---|---|---|
| Cora | text | -0.0124 | -0.0048 | -0.0037 | -0.0014 |
| Cora | edge | -0.0048 | -0.0040 | -0.0039 | -0.0035 |
| CiteSeer | text | -0.0037 | -0.0019 | -0.0017 | -0.0017 |
| CiteSeer | edge | 0.0011 | 0.0019 | 0.0017 | 0.0018 |
| PubMed | text | -0.0259 | -0.0178 | -0.0129 | -0.0103 |
| PubMed | edge | -0.0003 | -0.0006 | -0.0006 | -0.0006 |
| Amazon Computers | text | -0.0142 | -0.0090 | -0.0071 | -0.0065 |
| Amazon Computers | edge | 0.0002 | 0.0002 | -0.0000 | 0.0000 |

### Expected utility, mixtures, and distillation

A standalone MLM supplies text targets. Edge teachers include a text-only link MLP, a structure-only GNN, a graph autoencoder, and two ensembles. Teacher temperatures and the teacher family are chosen on development loss. The student backbone's full prediction is never its own teacher. Expected gain is the teacher-weighted log-probability difference between a subset and Full. Hard selection uses at most two groups. The three soft mixture controls share the exact candidate pool, including Full, and stacking fits nonnegative mixture weights on development only.

The paired interval uses two independent split means, with three model seeds averaged within each split. Its width reflects the small split count.

| Dataset | Task | Full | Teacher | Expected ≤2 | Matched random | Expected − random [95% CI] | Uniform | Expected mixture | Stacking |
|---|---|---|---|---|---|---|---|---|---|
| Cora | text | 5.4967 | 5.3886 | 5.4824 | 5.4834 | -0.0010 [-0.0037, +0.0017] | 5.4855 | 5.4853 | 5.4857 |
| Cora | edge | 0.6244 | 0.6253 | 0.6229 | 0.6238 | -0.0008 [-0.0027, +0.0010] | 0.6240 | 0.6240 | 0.6231 |
| CiteSeer | text | 5.9444 | 5.8405 | 5.9348 | 5.9345 | +0.0002 [-0.0043, +0.0048] | 5.9338 | 5.9337 | 5.9325 |
| CiteSeer | edge | 0.6245 | 0.6181 | 0.6198 | 0.6200 | -0.0002 [-0.0006, +0.0002] | 0.6211 | 0.6211 | 0.6199 |
| PubMed | text | 5.1394 | 4.9665 | 5.1079 | 5.1113 | -0.0034 [-0.0173, +0.0106] | 5.1191 | 5.1175 | 5.1156 |
| PubMed | edge | 0.3999 | 0.3883 | 0.3988 | 0.3995 | -0.0007 [-0.0033, +0.0018] | 0.3998 | 0.3998 | 0.3994 |
| Amazon Computers | text | 5.7233 | 5.6514 | 5.7140 | 5.7146 | -0.0005 [-0.0036, +0.0025] | 5.7143 | 5.7142 | 5.7130 |
| Amazon Computers | edge | 0.4842 | 0.4876 | 0.4847 | 0.4845 | +0.0002 [-0.0049, +0.0053] | 0.4846 | 0.4846 | 0.4839 |


| Dataset | Metric | Full | Teacher | Expected ≤2 | Expected mixture | Stacking |
|---|---|---|---|---|---|---|
| Cora | accuracy | 0.0895 | 0.0992 | 0.0917 | 0.0905 | 0.0913 |
| Cora | auc | 0.6926 | 0.6922 | 0.6925 | 0.6910 | 0.6907 |
| Cora | ap | 0.7405 | 0.7397 | 0.7401 | 0.7394 | 0.7394 |
| CiteSeer | accuracy | 0.0398 | 0.0576 | 0.0419 | 0.0418 | 0.0426 |
| CiteSeer | auc | 0.7070 | 0.7095 | 0.7083 | 0.7078 | 0.7082 |
| CiteSeer | ap | 0.7357 | 0.7385 | 0.7363 | 0.7367 | 0.7376 |
| PubMed | accuracy | 0.1839 | 0.1905 | 0.1814 | 0.1815 | 0.1813 |
| PubMed | auc | 0.8946 | 0.9016 | 0.8947 | 0.8944 | 0.8945 |
| PubMed | ap | 0.8799 | 0.8896 | 0.8801 | 0.8797 | 0.8799 |
| Amazon Computers | accuracy | 0.0551 | 0.0673 | 0.0586 | 0.0591 | 0.0604 |
| Amazon Computers | auc | 0.8474 | 0.8450 | 0.8456 | 0.8474 | 0.8498 |
| Amazon Computers | ap | 0.8518 | 0.8459 | 0.8506 | 0.8519 | 0.8526 |

The distillation pilot uses one model seed per split. Direct and LOO controls use only fitting outcomes. The distilled scorer learns from independent-teacher expected utility and decodes only the selected subset at inference, without teacher calls or prediction probes. All four hard controls use exact K=2, capped by availability. Timing compares synchronization-warmed full selector pipelines. The measured baseline is expected utility with teacher queries, while ordinary full-message inference is a separate baseline.

| Dataset | Task | Expected exact | Direct | LOO | Distilled | Agreement | Expected / distilled ms |
|---|---|---|---|---|---|---|---|
| Cora | text | 5.4934 | 5.4924 | 5.4917 | 5.4940 | 0.7473 | 6.1215 / 2.7457 |
| Cora | edge | 0.6264 | 0.6265 | 0.6267 | 0.6267 | 0.5074 | 10.3066 / 2.8951 |
| CiteSeer | text | 5.9308 | 5.9308 | 5.9301 | 5.9308 | 0.9031 | 4.5932 / 2.7948 |
| CiteSeer | edge | 0.6258 | 0.6272 | 0.6272 | 0.6266 | 0.6984 | 10.1592 / 3.7226 |
| PubMed | text | 5.1362 | 5.1341 | 5.1347 | 5.1368 | 0.8500 | 9.8753 / 5.2179 |
| PubMed | edge | 0.3948 | 0.3960 | 0.3953 | 0.3949 | 0.6988 | 21.6396 / 6.4402 |
| Amazon Computers | text | 5.7144 | 5.7150 | 5.7152 | 5.7151 | 0.9355 | 5.2318 / 3.1417 |
| Amazon Computers | edge | 0.4894 | 0.4894 | 0.4898 | 0.4895 | 0.7150 | 11.6436 / 3.7497 |

### Mechanisms and sensitivity checks

Each synthetic setting uses three independently generated graphs, and each sweep changes one factor with the others fixed at homophily 0.7, text noise 0.1, edge noise 0, and 128 tokens. Edge noise adds false edges only to test context. Vocabulary varies the data-generating alphabet and decoder size. The full vocabulary run limits fitting and scoring query counts for memory. Positive utility indicates improved loss with the opposite modality; harmful fraction uses per-target utility below −0.001.

| Factor | Level | Text utility | Edge utility | Text harmful | Edge harmful |
|---|---|---|---|---|---|
| homophily | 0.1 | 0.0231 | 0.0002 | 0.4653 | 0.4583 |
| homophily | 0.3 | 0.0105 | 0.0008 | 0.5035 | 0.4757 |
| homophily | 0.5 | -0.0513 | 0.0003 | 0.5833 | 0.4444 |
| homophily | 0.7 | -0.0275 | 0.0001 | 0.5104 | 0.4792 |
| homophily | 0.9 | -0.0326 | 0.0001 | 0.5556 | 0.4792 |
| text_noise | 0.0 | -0.0112 | -0.0004 | 0.4792 | 0.4722 |
| text_noise | 0.1 | -0.0275 | 0.0001 | 0.5104 | 0.4792 |
| text_noise | 0.25 | 0.0331 | -0.0000 | 0.4896 | 0.5069 |
| text_noise | 0.5 | 0.0437 | 0.0009 | 0.5000 | 0.4722 |
| edge_noise | 0.0 | -0.0275 | 0.0001 | 0.5104 | 0.4792 |
| edge_noise | 0.1 | -0.0281 | 0.0002 | 0.5174 | 0.4653 |
| edge_noise | 0.25 | -0.0290 | 0.0002 | 0.5278 | 0.4826 |
| edge_noise | 0.5 | -0.0293 | 0.0004 | 0.5278 | 0.4583 |
| vocabulary | 32 | -0.0222 | -0.0001 | 0.5660 | 0.4896 |
| vocabulary | 128 | -0.0275 | 0.0001 | 0.5104 | 0.4792 |
| vocabulary | 512 | 0.0329 | 0.0007 | 0.4861 | 0.4618 |
| vocabulary | 1024 | -0.0237 | 0.0017 | 0.5243 | 0.4271 |
| vocabulary | full | -0.0153 | 0.0004 | 0.5174 | 0.4479 |

Real-data missingness sweeps change either text or edge missingness through 0.1, 0.3, 0.5, 0.7, and 0.9 with the other fixed at 0.5. Degree bins stratify targets by observed degree at each masked case. Ties can produce empty bins. The table lists direction-matched utilities for the two missingness sweeps and the low, medium, and high degree bins.

| Dataset | Text gain over text-mask sweep | Edge gain over edge-mask sweep | Text degree-bin gain | Edge degree-bin gain |
|---|---|---|---|---|
| Cora | -0.0153 / -0.0158 / -0.0132 / -0.0197 / -0.0042 | 0.0003 / 0.0002 / -0.0019 / -0.0009 / -0.0017 | -0.0086 / -0.0149 / -0.0071 | 0.0009 / 0.0027 / -0.0036 |
| CiteSeer | -0.0238 / -0.0193 / 0.0001 / 0.0048 / -0.0092 | -0.0056 / -0.0036 / -0.0025 / -0.0046 / -0.0058 | -0.0105 / -0.0049 / -0.0082 | -0.0047 / -0.0083 / -0.0029 |
| PubMed | -0.0541 / -0.0367 / -0.0259 / -0.0150 / -0.0212 | -0.0014 / -0.0008 / -0.0003 / -0.0002 / 0.0004 | -0.0153 / -0.0368 / -0.0288 | 0.0008 / -0.0009 / 0.0001 |
| Amazon Computers | -0.0059 / -0.0061 / -0.0098 / -0.0099 / -0.0043 | -0.0015 / -0.0016 / 0.0003 / 0.0007 / -0.0001 | -0.0056 / -0.0268 / -0.0047 | 0.0004 / 0.0014 / 0.0011 |

Full V5 runs. [Cora](results/research_twinmask_v5/cora.json) [CiteSeer](results/research_twinmask_v5/citeseer.json) [PubMed](results/research_twinmask_v5/pubmed.json) [Amazon Computers](results/research_twinmask_v5/computer.json)

Selector and timing controls. [Cora](results/research_twinmask_v5/cora_selectors.json) [CiteSeer](results/research_twinmask_v5/citeseer_selectors.json) [PubMed](results/research_twinmask_v5/pubmed_selectors.json) [Amazon Computers](results/research_twinmask_v5/computer_selectors.json)

Independent synthetic graphs. [mechanisms](results/research_twinmask_v5/mechanisms.json)

### Reproduction

Raw references default to the neighboring `DIVERGE/datasets` directory and can be supplied through `TWINMASK_RAW_GRAPHS`. The audit requires the three local V2 checkpoints per citation dataset. The V5 experiments initialize all models independently.

```powershell
python -m scripts.research_twinmask_v41
python -m scripts.research_twinmask_v5
python -m scripts.research_twinmask_v5 --model-seeds 71 --selector-controls
python -m scripts.research_twinmask_v5 --mechanism-study
python -m unittest scripts.test_research_pilots scripts.test_research_twinmask_v3 scripts.test_research_twinmask_v4 scripts.test_research_twinmask_v5
python -m scripts.report_research_v5
```

<!-- research-v4-results -->

## TwinMask V4 · Target-conditioned counterfactual set utility

V4 evaluates whether observed target features and learned subset value recover the oracle gains in V3. The primary study uses the same V2 checkpoints and saved test masks across all selectors, four asymmetric mask cases per seed, and up to 384 targets per direction and case. It covers Cora, CiteSeer, and PubMed with seeds 42–51. Controller-training, development, and test cases have disjoint target nodes and held-out edge pools. Test targets enter metric and oracle computation only. A regression test modifies hidden labels and verifies unchanged selections and predictions.

**What worked.** Target and prediction-change features improved PubMed edge LOO Spearman from 0.0633 to 0.2026. The decision selector reduced edge BCE versus equal-count random selection on Cora and PubMed. The highest matched-budget recovery for decision plus cost was 14.8%, below the proposed 30% goal. The audit also missed the proposed 0.25 LOO Spearman and 65% sign-accuracy targets for the PubMed edge arm. The selected inference path was slower than full-message decoding in each measured timing pair.

### V4.0 · Oracle and utility audit

Loss is token cross-entropy (CE) or edge binary cross-entropy (BCE). Token and edge oracles choose per target. The node oracle uses one panel subset for a node's sampled tokens. The shared-endpoint oracle shares a subset at the first endpoint of all the sampled candidates with that endpoint, while allowing the other endpoint to vary within the total budget. Both shared oracles still use the target labels, so they measure restrictions on decision granularity. The equal-count oracle and random control use the same number of optional messages per target as the decision score with message cost.

| Dataset | Task | Full | Per-target oracle ≤2 | Shared oracle ≤2 | Matched-count oracle | Harmful fraction thresholds 0 / .001 / .005 / .01 |
|---|---|---|---|---|---|---|
| Cora | text | 4.8501 | 4.7873 | 4.7963 | 4.8369 | 0.5207 / 0.5076 / 0.4639 / 0.4175 |
| CiteSeer | text | 5.1609 | 5.0933 | 5.1019 | 5.1502 | 0.5191 / 0.5112 / 0.4807 / 0.4417 |
| PubMed | text | 4.0550 | 4.0112 | 4.0120 | 4.0491 | 0.5592 / 0.4816 / 0.3858 / 0.3211 |


| Dataset | Task | Full | Per-target oracle ≤2 | Shared oracle ≤2 | Matched-count oracle | Harmful fraction thresholds 0 / .001 / .005 / .01 |
|---|---|---|---|---|---|---|
| Cora | edge | 0.6483 | 0.6060 | 0.6074 | 0.6241 | 0.4852 / 0.4497 / 0.3663 / 0.2950 |
| CiteSeer | edge | 0.6436 | 0.5911 | 0.5919 | 0.6266 | 0.4871 / 0.4587 / 0.3815 / 0.3168 |
| PubMed | edge | 0.4909 | 0.4572 | 0.4573 | 0.4708 | 0.4940 / 0.3916 / 0.2778 / 0.2146 |

### V4.1–V4.2 · Target-conditioned and prediction-aware estimates

Text features include sender, receiver, message, masked-token representation, and position. Edge features include both endpoints, their absolute difference and product, with the opposite endpoint defined relative to each message receiver. Prediction-aware features add full uncertainty, changes in entropy, confidence and margin, JS divergence, top-answer disagreement, and the magnitude of the removed contribution before the final nonlinearity. The position and all semantic features come from visible or masked inputs.

The first three columns below are LOO Spearman correlations. Sign accuracy is for the prediction-aware LOO estimator.

| Dataset | Task | V3 features | + Target | + Prediction changes | Sign accuracy | Majority sign |
|---|---|---|---|---|---|---|
| Cora | text | 0.0238 | 0.0398 | 0.0469 | 0.5226 | 0.5300 |
| Cora | edge | 0.0183 | 0.0583 | 0.0686 | 0.5250 | 0.5187 |
| CiteSeer | text | 0.0421 | 0.0267 | 0.0329 | 0.5115 | 0.5316 |
| CiteSeer | edge | 0.0328 | 0.0686 | 0.0725 | 0.5217 | 0.5196 |
| PubMed | text | 0.0506 | 0.0480 | 0.0449 | 0.5186 | 0.5593 |
| PubMed | edge | 0.0633 | 0.1542 | 0.2026 | 0.5580 | 0.5175 |

### V4.3–V4.5 · Set utility, decision loss, and flexible budgets

DeepSets assigns a value to each unique feasible subset using selected and full panel summaries. The target is full loss minus subset loss. Decision training adds expected reconstruction regret under a masked softmax at K=1, 2, and 3. Development tunes the cost per optional message in standardized utility units and the risk rule. The threshold arm uses a fixed 0.1 utility threshold in standardized units, and the at-most arms can choose an empty set.

All entries in the first results table use exact K=2, capped by availability. The direct gate uses the same prediction-aware features and scorer width as the message-utility MLP. The Shapley arm is trained on exact message Shapley values.

| Dataset | Task | Random | Direct | Aware LOO | Shapley | DeepSets | Decision |
|---|---|---|---|---|---|---|---|
| Cora | text | 4.8487 | 4.8485 | 4.8486 | 4.8486 | 4.8487 | 4.8485 |
| Cora | edge | 0.6496 | 0.6477 | 0.6476 | 0.6481 | 0.6492 | 0.6476 |
| CiteSeer | text | 5.1606 | 5.1606 | 5.1606 | 5.1606 | 5.1607 | 5.1604 |
| CiteSeer | edge | 0.6459 | 0.6448 | 0.6438 | 0.6436 | 0.6445 | 0.6439 |
| PubMed | text | 4.0546 | 4.0544 | 4.0546 | 4.0546 | 4.0544 | 4.0544 |
| PubMed | edge | 0.4905 | 0.4888 | 0.4867 | 0.4867 | 0.4901 | 0.4880 |


| Dataset | Task | At most 2 | Threshold | Decision + cost | Messages | Empty fraction | Loss change vs matched random [paired 95% CI] | Matched-random-to-oracle gap recovered |
|---|---|---|---|---|---|---|---|---|
| Cora | text | 4.8450 | 4.8461 | 4.8457 | 0.6212 | 0.5670 | -0.0002 [-0.0006, +0.0002] | 2.0% |
| Cora | edge | 0.6487 | 0.6491 | 0.6490 | 1.0956 | 0.2827 | -0.0021 [-0.0036, -0.0005] | 7.7% |
| CiteSeer | text | 5.1551 | 5.1549 | 5.1547 | 0.4887 | 0.6857 | -0.0001 [-0.0004, +0.0002] | 1.9% |
| CiteSeer | edge | 0.6444 | 0.6444 | 0.6443 | 0.8130 | 0.4746 | -0.0019 [-0.0045, +0.0008] | 9.5% |
| PubMed | text | 4.0510 | 4.0517 | 4.0511 | 0.2275 | 0.8219 | -0.0002 [-0.0005, +0.0000] | 10.0% |
| PubMed | edge | 0.4849 | 0.4868 | 0.4853 | 0.9157 | 0.3851 | -0.0025 [-0.0044, -0.0006] | 14.8% |

Paired intervals use ten seed means within each fixed graph. The recovery fraction uses the random-to-oracle loss gap with the selector's exact realized message count. That quantity differs from the full-to-unconstrained-oracle gap.

Accuracy, AUROC, and AP use the same test targets as the loss comparisons. Direct, aware LOO, and decision use exact K=2 in this table.

| Dataset | Metric | Full | Direct exact | Aware LOO exact | Decision exact | Decision + cost | Matched random |
|---|---|---|---|---|---|---|---|
| Cora | ACCURACY | 0.1665 | 0.1666 | 0.1665 | 0.1671 | 0.1676 | 0.1678 |
| Cora | AUC | 0.6787 | 0.6800 | 0.6807 | 0.6804 | 0.6785 | 0.6759 |
| Cora | AP | 0.7017 | 0.7032 | 0.7042 | 0.7041 | 0.7023 | 0.7000 |
| CiteSeer | ACCURACY | 0.1359 | 0.1355 | 0.1355 | 0.1355 | 0.1348 | 0.1349 |
| CiteSeer | AUC | 0.6835 | 0.6803 | 0.6821 | 0.6811 | 0.6772 | 0.6745 |
| CiteSeer | AP | 0.7136 | 0.7113 | 0.7124 | 0.7115 | 0.7085 | 0.7071 |
| PubMed | ACCURACY | 0.3037 | 0.3037 | 0.3036 | 0.3039 | 0.3036 | 0.3036 |
| PubMed | AUC | 0.8590 | 0.8624 | 0.8649 | 0.8624 | 0.8637 | 0.8629 |
| PubMed | AP | 0.8565 | 0.8586 | 0.8606 | 0.8584 | 0.8593 | 0.8591 |


### V4.6 · Uncertainty and external-method controls

The decision model estimates a predictive spread and risk uses mean minus beta times spread, with a cost for each message. The development-selected fallback may use the full panel and is reported separately from strict-cap policies. The three-seed comparator runs include eight MC dropout draws and a three-member ensemble. The table reports nominal 90% empirical interval coverage and strict-cap loss at K=2. Coverage is an empirical diagnostic and carries no calibration guarantee.

| Dataset | Task | Mean/variance coverage | MC dropout coverage | Ensemble coverage | Risk | Risk + full fallback |
|---|---|---|---|---|---|---|
| Cora | text | 0.8891 | 0.9044 | 0.9186 | 4.8890 | 4.8898 |
| Cora | edge | 0.8659 | 0.8770 | 0.8970 | 0.6453 | 0.6446 |
| CiteSeer | text | 0.8805 | 0.8668 | 0.8948 | 5.1414 | 5.1424 |
| CiteSeer | edge | 0.8347 | 0.8803 | 0.8625 | 0.6439 | 0.6431 |
| PubMed | text | 0.8852 | 0.8895 | 0.9109 | 4.0280 | 4.0280 |
| PubMed | edge | 0.9055 | 0.8998 | 0.9080 | 0.4879 | 0.4879 |

Equal-count local reconstructions below use additive GAT attention, a straight-through top-K sparsifier inspired by NeuralSparse, and a correctness-predicting utility proxy with predicted Shapley scores inspired by SGUL. These are adaptations to the frozen reconstruction model with paired panels and targets. Published end-to-end benchmark numbers use other tasks and protocols. The set-attention ablation adds one permutation-equivariant self-attention block before DeepSets pooling.

| Dataset | Task | GAT local | NeuralSparse local | SGUL local | Set attention | Decision |
|---|---|---|---|---|---|---|
| Cora | text | 4.8966 | 4.8970 | 4.8970 | 4.8974 | 4.8963 |
| Cora | edge | 0.6435 | 0.6441 | 0.6467 | 0.6447 | 0.6438 |
| CiteSeer | text | 5.1412 | 5.1411 | 5.1412 | 5.1407 | 5.1409 |
| CiteSeer | edge | 0.6428 | 0.6422 | 0.6434 | 0.6433 | 0.6427 |
| PubMed | text | 4.0315 | 4.0322 | 4.0319 | 4.0316 | 4.0319 |
| PubMed | edge | 0.4945 | 0.4938 | 0.4926 | 0.4945 | 0.4907 |

### Larger panels, layer interventions, and generalization

All interventions preserve the graph and degree denominator. The edge branch suppresses the direct text component of incoming messages. Layer ablations include first-only and both layers. With hidden candidate edges and no self loops, the target roots are locally recomputed while untouched messages keep their cached values.

This Cora table uses three seeds and the same 192-target cap within each protocol. The four-slot both-layer arm and the six-slot layer comparison are exploratory. The re-split arm loads independently re-split V2 warm-link checkpoints and regenerated node splits.

| Cora variant | Task | Full | Decision K=2 | Matched-count oracle |
|---|---|---|---|---|
| cora_panel4_last | text | 4.8999 | 4.8891 | 4.8829 |
| cora_panel4_last | edge | 0.6409 | 0.6425 | 0.6217 |
| cora_both | text | 4.8999 | 4.8914 | 4.8688 |
| cora_both | edge | 0.6409 | 0.6443 | 0.6021 |
| cora_panel6_first | text | 4.8999 | 4.8994 | 4.8919 |
| cora_panel6_first | edge | 0.6409 | 0.6482 | 0.6216 |
| cora_panel6 | text | 4.8999 | 4.8886 | 4.8853 |
| cora_panel6 | edge | 0.6409 | 0.6431 | 0.6161 |
| cora_panel6_both | text | 4.8999 | 4.8935 | 4.8756 |
| cora_panel6_both | edge | 0.6409 | 0.6478 | 0.6020 |
| cora_resplit | text | 4.8555 | 4.8534 | 4.8468 |
| cora_resplit | edge | 0.6425 | 0.6444 | 0.6227 |

Synthetic controls vary homophily, text signal, and target-time false edges. In the transfer test, the denoiser and selector are fitted on the reference graph and evaluated on newly generated graphs without fitting on their labels. The objective is the same K≤2 decision score. Random and oracle receive the learned selector's exact count per target.

| Unseen synthetic regime | Task | Full | Decision | Matched random | Matched oracle | False-edge retention decision / random |
|---|---|---|---|---|---|---|
| synthetic_reference | text | 2.7900 | 2.7919 | 2.7943 | 2.7839 | — |
| synthetic_reference | edge | 0.7036 | 0.6889 | 0.6891 | 0.6809 | — |
| synthetic_low_homophily | text | 2.8926 | 2.8860 | 2.8874 | 2.8692 | — |
| synthetic_low_homophily | edge | 0.8335 | 0.8300 | 0.8292 | 0.8133 | — |
| synthetic_weak_text | text | 3.7182 | 3.7122 | 3.7132 | 3.6958 | — |
| synthetic_weak_text | edge | 0.6774 | 0.6791 | 0.6784 | 0.6687 | — |
| synthetic_noisy_context | text | 2.8028 | 2.8037 | 2.8055 | 2.7947 | 0.3064 / 0.3155 |
| synthetic_noisy_context | edge | 0.6753 | 0.6663 | 0.6663 | 0.6589 | 0.3749 / 0.3424 |

### V4.7 · Fine-tuning and measured runtime

For each direction, partial fine-tuning unfreezes the last GraphSAGE layer. End-to-end fine-tuning updates the compact denoiser and selector jointly with refreshed subset losses and differentiable subset choice. The full-message control fine-tunes the same denoiser for the same 12 updates. All tuning uses controller-training cases, and each arm scores paired test panels.

Measured latency includes graph and panel preparation, prediction probes where applicable, scoring, and final decoding. Each value averages three seed medians after warm-up. The graph prefix and its full aggregation are still computed in this implementation. Optional-message counts describe the selected panel only.

| Dataset | Task | Frozen selected | Partial selected | End-to-end selected | Fine-tuned full | Full / selected ms | Full / selected panel messages |
|---|---|---|---|---|---|---|---|
| Cora | text | 4.8714 | 4.8665 | 4.8408 | 4.8404 | 6.4919 / 9.1331 | 2.5117 / 0.6914 |
| Cora | edge | 0.6471 | 0.6450 | 0.6514 | 0.6513 | 8.9295 / 11.9824 | 3.3984 / 1.6536 |
| CiteSeer | text | 5.1473 | 5.1455 | 5.1317 | 5.1298 | 5.8951 / 8.4394 | 1.6602 / 0.9414 |
| CiteSeer | edge | 0.6399 | 0.6380 | 0.6375 | 0.6385 | 7.1049 / 10.5318 | 2.8854 / 1.0417 |
| PubMed | text | 4.0316 | 4.0294 | 4.0148 | 4.0160 | 25.3872 / 29.6589 | 1.9479 / 0.4727 |
| PubMed | edge | 0.4904 | 0.4874 | 0.4865 | 0.4928 | 27.8257 / 30.9140 | 3.0964 / 1.3763 |

### V4.8 · Diffusion integration

Two free-running reconstruction iterations use the same sampled panel of up to 192 held-out tokens and edges per case. The first iteration confidence-fills half, and the second completes the panel at half the initial clocks using the previous predictions. The selector is reapplied on both iterations. The paired full-message rollout uses the same masks, initial targets, and number of reconstruction iterations. The selected arm also uses counterfactual prediction probes and thus incurs additional work.

| Dataset | Full token accuracy / edge AUROC | Selected token accuracy / edge AUROC |
|---|---|---|
| Cora | 0.1606 / 0.6813 | 0.1597 / 0.6757 |
| CiteSeer | 0.1380 / 0.6827 | 0.1367 / 0.6766 |
| PubMed | 0.3181 / 0.8678 | 0.3147 / 0.8709 |

### Results and reproduction

Primary paired results. [Cora](results/research_twinmask_v4/cora.json) [CiteSeer](results/research_twinmask_v4/citeseer.json) [PubMed](results/research_twinmask_v4/pubmed.json)

Fine-tuning, timing, and diffusion. [Cora](results/research_twinmask_v4/cora_extra.json) [CiteSeer](results/research_twinmask_v4/citeseer_extra.json) [PubMed](results/research_twinmask_v4/pubmed_extra.json)

Local baselines and uncertainty controls. [Cora](results/research_twinmask_v4/cora_baselines.json) [CiteSeer](results/research_twinmask_v4/citeseer_baselines.json) [PubMed](results/research_twinmask_v4/pubmed_baselines.json)

Layer, split, and synthetic studies. [cora_panel4_last](results/research_twinmask_v4/cora_panel4_last.json) [cora_both](results/research_twinmask_v4/cora_both.json) [cora_panel6_first](results/research_twinmask_v4/cora_panel6_first.json) [cora_panel6](results/research_twinmask_v4/cora_panel6.json) [cora_panel6_both](results/research_twinmask_v4/cora_panel6_both.json) [cora_resplit](results/research_twinmask_v4/cora_resplit.json) [synthetic_reference](results/research_twinmask_v4/synthetic_reference.json) [synthetic_low_homophily](results/research_twinmask_v4/synthetic_low_homophily.json) [synthetic_weak_text](results/research_twinmask_v4/synthetic_weak_text.json) [synthetic_noisy_context](results/research_twinmask_v4/synthetic_noisy_context.json) [synthetic_reference_transfer](results/research_twinmask_v4/synthetic_reference_transfer.json)

DeepSets provides the invariant aggregation. Set Transformer motivates the attention ablation. The decision objective is a softmax over this pilot's enumerated subsets, following the decision-focused training principle. [Deep Sets](https://arxiv.org/abs/1703.06114) [Set Transformer](https://arxiv.org/abs/1810.00825) [Decision-Focused Learning](https://mlanthology.org/jair/2024/mandi2024jair-decisionfocused/)

```powershell
python -m scripts.research_twinmask_v4
python -m scripts.research_twinmask_v4 --seeds 42 43 44 --extra --steps 60 --limit 256
python -m scripts.research_twinmask_v4 --seeds 42 43 44 --comparators --steps 60 --limit 256
foreach ($layer in 'first','last','both') { python -m scripts.research_twinmask_v4 --datasets cora --seeds 42 43 44 --panel 6 --layer $layer --steps 60 --limit 192 }
python -m scripts.research_twinmask_v4 --datasets cora --seeds 42 43 44 --steps 60 --limit 192 --tag panel4_last
python -m scripts.research_twinmask_v4 --datasets cora --seeds 42 43 44 --layer both --steps 60 --limit 192
python -m scripts.research_twinmask_v4 --datasets cora --seeds 42 43 44 --resplit --steps 60 --limit 192
python -m scripts.research_twinmask_v4 --datasets synthetic_reference synthetic_low_homophily synthetic_weak_text synthetic_noisy_context --seeds 42 43 44 --steps 60 --limit 192
python -m scripts.research_twinmask_v4 --datasets synthetic_reference --transfer --seeds 42 43 44 --steps 60 --limit 192
python -m unittest scripts.test_research_pilots scripts.test_research_twinmask_v3 scripts.test_research_twinmask_v4
python -m scripts.report_research_v4
```

## TwinMask V3 · Counterfactual utility and selective message passing

V3 asks whether local evidence removal improves masked token and edge reconstruction, and how much of the oracle gain a selector can recover from observations alone. This pilot freezes the V2 denoiser and intervenes in its final GraphSAGE aggregation. Each token target has up to four sampled incoming messages. Each edge candidate has up to two sampled incoming messages at each endpoint, and the edge experiment gates the direct pooled-text term of those messages. Indirect text paths, degree normalization, the observed graph, and messages outside the sampled panel stay fixed. The budget applies to the panel.

The runner measures leave-one-out loss changes, exact Shapley values over the at-most-four-slot panel, and signed pair interactions. Separate lightweight utility networks for text and edge use Huber and ranking losses. A reconstruction-trained gate with the same scoring architecture serves as a control. The budgeted comparisons use matched counts for random, semantic similarity, predicted utility, redundancy-penalized utility, and predicted synergy. The unconstrained-count oracle can use any subset of size at most K. A second oracle uses the matched count. Oracle scores use target labels for diagnostics. Learned selectors are fitted on controller cases and choose test weights from observable features.

### Related work and overlap

| Primary source | What overlaps and what V3 measures |
|---|---|
| [SGUL, ICLR 2025](https://arxiv.org/html/2503.18195v1) | The closest overlap for learning neighbor value without test labels. SGUL values neighbor subsets using a learned utility tied to inference accuracy and Shapley features. The V3 pilot estimates signed loss changes for two masked reconstruction targets. |
| [Efficient Modality Selection, JMLR 2024](https://jmlr.org/papers/v25/23-0439.html) | Uses Shapley and marginal utility for cardinality-constrained modality selection, with guarantees under its assumptions. V3 enumerates small message panels, including potentially negative and interacting utility. |
| [GraphMask, ICLR 2021](https://arxiv.org/html/2010.00577) | Trains differentiable message masks to explain which messages a frozen GNN can omit while preserving behavior. V3 compares reconstruction loss changes under fixed-normalization message suppression. |
| [NeuralSparse, ICML 2020](https://proceedings.mlr.press/v119/zheng20d.html) and [PASS, KDD 2021](https://blog.ml.cmu.edu/2021/12/03/pass-performance-adaptive-sampling-strategy-for-graph-convolutional-networks/) | Learn task-relevant sparsification or neighbor sampling. The pilot's reconstruction-trained gate is a local control. Equal-protocol reproductions of the published methods remain open. |
| [UDPD, ICML 2026](https://proceedings.mlr.press/v306/yu26ab.html) | Already modulates cross-modal interaction by uncertainty in continuous embedding diffusion for node classification. V3 studies local counterfactual evidence for discrete reconstruction under separate masking rates. |

These overlaps leave the paper's candidate contribution in the **measurement and mitigation of direction-specific harmful evidence under incomplete text and structure**, with equal-budget controls. Establishing novelty and improvement requires stronger same-protocol comparisons.

<!-- research-v3-results -->

### Ten-seed results

Seeds 42–51, four asymmetric corruption cases per seed, and up to 768 sampled targets per task and case. The test cases use saved V2 masks. Lower token CE and edge BCE are better. Each cell is a mean over cases and seeds.

| Dataset | Direction | Full | Random K=2 | Direct gate top-2 | Learned top-2 | Learned redundancy + synergy K=2 | Oracle matched K=2 | Oracle at most 2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | text | 4.8499 | 4.8496 | 4.8490 | 4.8492 | 4.8493 | 4.8333 | 4.7882 |
| Cora | edge | 0.6465 | 0.6483 | 0.6480 | 0.6471 | 0.6477 | 0.6132 | 0.6044 |
| CiteSeer | text | 5.1818 | 5.1819 | 5.1814 | 5.1815 | 5.1816 | 5.1734 | 5.1151 |
| CiteSeer | edge | 0.6468 | 0.6489 | 0.6483 | 0.6479 | 0.6477 | 0.6191 | 0.5943 |
| PubMed | text | 3.9951 | 3.9945 | 3.9942 | 3.9942 | 3.9944 | 3.9884 | 3.9512 |
| PubMed | edge | 0.4940 | 0.4929 | 0.4925 | 0.4907 | 0.4915 | 0.4692 | 0.4599 |

Paired intervals use the ten seed means within each graph and quantify run variability on that graph.

| Dataset | Direction | Oracle at most 2 minus full [95% CI] | Learned top-2 minus direct gate top-2 [95% CI] | LOO Spearman |
|---|---|---:|---:|---:|
| Cora | text | -0.0617 [-0.0720, -0.0513] | +0.0001 [-0.0002, +0.0004] | 0.033 |
| Cora | edge | -0.0420 [-0.0522, -0.0319] | -0.0009 [-0.0031, +0.0013] | 0.023 |
| CiteSeer | text | -0.0667 [-0.0766, -0.0567] | +0.0000 [-0.0002, +0.0003] | 0.043 |
| CiteSeer | edge | -0.0524 [-0.0682, -0.0366] | -0.0003 [-0.0017, +0.0011] | 0.052 |
| PubMed | text | -0.0439 [-0.0507, -0.0371] | +0.0000 [-0.0001, +0.0002] | 0.083 |
| PubMed | edge | -0.0341 [-0.0414, -0.0267] | -0.0018 [-0.0028, -0.0008] | 0.115 |

The oracle with an up-to-two budget improved both losses on all three datasets. The learned selectors recovered only a small share of the diagnostic oracle gap. Weak positive rank correlation and the comparison with direct supervision temper any claim for the utility or synergy objective. The current audit scores all subsets of a small panel after full-graph encoding, so its runtime measures the research experiment. The selected message counts do not establish inference speedup.

![Change in token and edge reconstruction loss relative to full message use across local budgets](results/research_twinmask_v3/budget_curves.png)

Detailed curves for budgets 1–4, hard/soft gates, oracle Shapley ranking, interaction magnitude, accuracy, perplexity, AUROC, AP, and per-seed runtimes. Fitted selectors are stored locally under `checkpoints/research_twinmask_v3/`. [Cora](results/research_twinmask_v3/cora.json) [CiteSeer](results/research_twinmask_v3/citeseer.json) [PubMed](results/research_twinmask_v3/pubmed.json)

This pilot uses the compact V2 checkpoints and a one-shot final-layer intervention. Multi-step diffusion integration, larger message panels, node classification, GAT/GraphMask/NeuralSparse/PASS reproductions, synthetic-shift generalization, and deployment latency remain future tests.

```powershell
python -m scripts.research_twinmask_v3 --seeds 42 43 44 45 46 47 48 49 50 51
python -m unittest scripts.test_research_pilots scripts.test_research_twinmask_v3 -v
```

## Research correction and directional conditioning (V2)

**Working title.** When Text and Topology Disagree: Adaptive Cross-Modal Conditioning for Incomplete Text-Attributed Graphs.

The earlier sparse TwinMask run conditioned on the fraction of *initially masked targets* still pending. Both clocks therefore started at 1 for every nonempty modality, including in training. V2 starts each clock at its own nominal corruption rate and scales it by the fraction of its initial targets still pending. Tied training uses equal initial rates. Both training arms face the same independently masked evaluation cases.

The denoiser is trained on both initially corrupted inputs and short rollouts containing its own detached predictions. The correction policy re-masks the generated block in a separate scoring pass and ranks each generated value by support for that value. Observations remain clamped. The rescoring pass counts against the same four-call budget. In the soft-edge variant, a generated pair contributes to message passing with its predicted probability, while observed positives retain unit weight. The hard-edge control uses thresholded generated edges.

The primary V2 question is whether observed structure helps token recovery and observed text helps link recovery under each missingness pattern. One-shot directional conditioning compares a joint pass, text-only for token reconstruction, structure-only for link reconstruction, and a per-case learned mixture with a separate probability weight for each direction. The learned weights use corruption clocks and uncertainty summaries, with targets fitted on disjoint controller cases. An oracle selects weights against held-out labels for analysis only. One-shot here means no generated feedback. Learned conditioning requires three denoiser passes to produce its joint and unimodal probabilities.

Across ten seeds on Cora, CiteSeer, and PubMed, the learned gate reduced mean token cross-entropy by 0.008, 0.019, and 0.022 respectively. Mean edge binary cross-entropy increased on all three. On PubMed, four-call adaptive reconstruction raised mean token accuracy from 30.68% to 31.12% and edge AUROC from 0.8552 to 0.8589 relative to one-shot. The validation-selected fixed four-call policy reached 31.39% and 0.8601. Cora and CiteSeer had small declines in edge AUROC relative to one-shot. In the symmetric Ensemble comparisons, the accuracy difference between five diffusion views and five Gaussian views stayed within 0.25 percentage points under either classifier protocol.

The strongest literature overlap is UDPD §§3.3–4.1. It explicitly adjusts cross-branch interaction by node uncertainty while denoising two continuous embedding streams and evaluates downstream node classification. TAG-DLM injects topology into a masked diffusion language model and recovers a masked answer token for classification and candidate link decisions. V2 evaluates held-out content-token and candidate-edge reconstruction and directly measures whether removing cross-modal evidence lowers loss. A one-shot probability gate alone establishes no novelty claim. [UDPD](https://proceedings.mlr.press/v306/yu26ab.html) [TAG-DLM](https://arxiv.org/html/2606.31166v1)

<!-- research-v2-results -->

### Corrected experiment results

TwinMask uses seeds 42–51 on the three citation datasets and seeds 42–44 for the ablations, extra datasets, and synthetic cases. Each seed has four jointly scored text/edge mask settings. The matched Ensemble comparison uses seeds 42–44.

#### Ensemble classifier controls

Test accuracy (%) uses the same representations, labeled split, classifier architecture, and training epochs within each comparison. The raw-only head sees only raw embeddings during training. Each matched head sees its own arm's representations and is selected using that arm's mean validation predictive distribution.

| Dataset | Classifier | Diffusion-5 | Gaussian-5 | Raw + shared-4 | Raw + independent-4 |
|---|---|---|---|---|---|
| Cora | Raw-only | 84.81 | 84.87 | 85.24 | 85.06 |
| Cora | Method-matched | 84.93 | 84.69 | 85.06 | 85.06 |
| CiteSeer | Raw-only | 74.35 | 74.40 | 74.82 | 74.71 |
| CiteSeer | Method-matched | 74.76 | 74.76 | 74.56 | 74.71 |
| PubMed | Raw-only | 89.18 | 89.17 | 89.20 | 89.20 |
| PubMed | Method-matched | 89.66 | 89.51 | 89.60 | 89.59 |

Full metrics, denoiser and classifier timing, and ensemble diversity. [Cora](results/research_ensemble_v2/cora.json) [CiteSeer](results/research_ensemble_v2/citeseer.json) [PubMed](results/research_ensemble_v2/pubmed.json)

#### TwinMask clocks, rollout, and correction

Cells report mean token accuracy (%) / edge AUROC. One-shot uses one call. Joint, corrective, and adaptive schedules have four calls, including any rescoring. The fixed policy is selected on development cases.

| Dataset | One-shot | Joint | Corrective | Adaptive | Selected fixed |
|---|---|---|---|---|---|
| Cora | 16.71 / 0.6812 | 17.02 / 0.6798 | 17.05 / 0.6816 | 16.82 / 0.6797 | 16.88 / 0.6799 |
| CiteSeer | 13.29 / 0.6792 | 13.37 / 0.6776 | 13.40 / 0.6781 | 13.47 / 0.6783 | 13.34 / 0.6770 |
| PubMed | 30.68 / 0.8552 | 31.41 / 0.8600 | 31.37 / 0.8600 | 31.12 / 0.8589 | 31.39 / 0.8601 |
| WikiCS | 17.24 / 0.9063 | 16.80 / 0.9058 | 16.82 / 0.9070 | 17.17 / 0.9070 | 17.14 / 0.9066 |
| Reddit | 12.21 / 0.7391 | 12.11 / 0.7389 | 12.15 / 0.7396 | 12.23 / 0.7395 | 12.14 / 0.7394 |
| Instagram | 13.82 / 0.8743 | 14.05 / 0.8744 | 14.10 / 0.8755 | 13.89 / 0.8746 | 14.11 / 0.8748 |


| Dataset | Adaptive comparator | Token accuracy change (pp) [95% CI] | Edge AUROC change [95% CI] |
|---|---|---|---|
| Cora | one shot | +0.1026 [-0.0138, +0.2191] | -0.0015 [-0.0029, -0.0001] |
| Cora | fixed | -0.0665 [-0.3222, +0.1892] | -0.0002 [-0.0023, +0.0019] |
| CiteSeer | one shot | +0.1811 [+0.0113, +0.3508] | -0.0009 [-0.0034, +0.0017] |
| CiteSeer | fixed | +0.1226 [-0.0043, +0.2496] | +0.0013 [-0.0011, +0.0037] |
| PubMed | one shot | +0.4401 [+0.3466, +0.5337] | +0.0037 [+0.0020, +0.0054] |
| PubMed | fixed | -0.2673 [-0.3755, -0.1590] | -0.0011 [-0.0021, -0.0002] |

Ablations use the same first three seed IDs. Mixed and tied refer to clock training. Hard-edge message passing and no-rollout training are separate ablations.

| Dataset | Policy | Mixed | Tied | No rollout | Hard edges |
|---|---|---|---|---|---|
| Cora | one shot | 16.34 / 0.6813 | 16.52 / 0.6942 | 16.24 / 0.6770 | 16.37 / 0.6816 |
| Cora | adaptive | 16.47 / 0.6805 | 16.81 / 0.6924 | 16.01 / 0.6750 | 16.44 / 0.6823 |
| CiteSeer | one shot | 13.10 / 0.6835 | 13.12 / 0.6934 | 12.73 / 0.6712 | 13.12 / 0.6863 |
| CiteSeer | adaptive | 13.30 / 0.6855 | 13.38 / 0.6861 | 12.69 / 0.6623 | 13.25 / 0.6891 |
| PubMed | one shot | 30.54 / 0.8526 | 30.46 / 0.8528 | 30.01 / 0.8535 | 30.53 / 0.8543 |
| PubMed | adaptive | 31.05 / 0.8587 | 30.88 / 0.8593 | 30.50 / 0.8562 | 31.04 / 0.8593 |

Detailed corrected runs. [Cora](results/research_twinmask_v2/cora.json) [CiteSeer](results/research_twinmask_v2/citeseer.json) [PubMed](results/research_twinmask_v2/pubmed.json) [WikiCS](results/research_twinmask_v2/wikics.json) [Reddit](results/research_twinmask_v2/reddit.json) [Instagram](results/research_twinmask_v2/instagram.json)

Detailed ablations. [Cora tied](results/research_twinmask_v2/cora_tied.json) [Cora no rollout](results/research_twinmask_v2/cora_no_rollout.json) [Cora hard](results/research_twinmask_v2/cora_hard.json) [CiteSeer tied](results/research_twinmask_v2/citeseer_tied.json) [CiteSeer no rollout](results/research_twinmask_v2/citeseer_no_rollout.json) [CiteSeer hard](results/research_twinmask_v2/citeseer_hard.json) [PubMed tied](results/research_twinmask_v2/pubmed_tied.json) [PubMed no rollout](results/research_twinmask_v2/pubmed_no_rollout.json) [PubMed hard](results/research_twinmask_v2/pubmed_hard.json)

#### One-shot directional conditioning

Cells report token cross-entropy / edge binary cross-entropy. Lower is better. Each seed contributes the mean over its four cases. The 95% paired t intervals are for learned minus joint loss and treat seeds as the run unit within each graph.

| Dataset | Unimodal | Joint | Learned | Oracle | Token CE paired change [95% CI] | Edge BCE paired change [95% CI] |
|---|---|---|---|---|---|---|
| Cora | 4.8618 / 0.6845 | 4.8598 / 0.6467 | 4.8522 / 0.6506 | 4.8510 / 0.6458 | -0.0076 [-0.0124, -0.0028] | +0.0039 [+0.0008, +0.0069] |
| CiteSeer | 5.1641 / 0.7133 | 5.1780 / 0.6468 | 5.1593 / 0.6503 | 5.1585 / 0.6466 | -0.0188 [-0.0243, -0.0132] | +0.0035 [-0.0001, +0.0071] |
| PubMed | 3.9963 / 0.5672 | 4.0184 / 0.4946 | 3.9960 / 0.4976 | 3.9960 / 0.4946 | -0.0224 [-0.0288, -0.0159] | +0.0030 [-0.0000, +0.0059] |
| WikiCS | 5.0541 / 0.4945 | 5.0585 / 0.4260 | 5.0434 / 0.4260 | 5.0430 / 0.4260 | -0.0151 [-0.0367, +0.0066] | +0.0000 [+0.0000, +0.0000] |
| Reddit | 4.9682 / 0.6171 | 4.9656 / 0.6191 | 4.9627 / 0.6150 | 4.9625 / 0.6130 | -0.0030 [-0.0159, +0.0100] | -0.0041 [-0.0173, +0.0092] |
| Instagram | 5.0314 / 0.5434 | 5.0202 / 0.4770 | 5.0173 / 0.4801 | 5.0170 / 0.4766 | -0.0029 [-0.0065, +0.0007] | +0.0031 [-0.0018, +0.0080] |

Synthetic interventions change expected homophily (0.85 to 0.30), lexical signal (0.80 to 0.20), or add spurious test-context edges (25% of the number of real training edges), one factor at a time. The independently varied text and edge missingness cases are stored in the JSON. A positive evidence gain means that removing the other modality increased loss.

| Synthetic case | Structure to text CE gain | Text to edge BCE gain | Joint CE / BCE | Learned CE / BCE |
|---|---|---|---|---|
| reference | +0.1397 | -0.0827 | 2.7725 / 0.8324 | 2.7736 / 0.5314 |
| low homophily | -0.0671 | -0.0793 | 2.9402 / 0.7740 | 2.8729 / 0.6952 |
| weak text | -0.0038 | -0.0047 | 3.5266 / 0.6981 | 3.5231 / 0.6938 |
| noisy context | +0.1282 | +0.0156 | 2.7994 / 0.7262 | 2.8049 / 0.5362 |

Synthetic runs. [reference](results/research_twinmask_v2/synthetic_reference.json) [low homophily](results/research_twinmask_v2/synthetic_low_homophily.json) [weak text](results/research_twinmask_v2/synthetic_weak_text.json) [noisy context](results/research_twinmask_v2/synthetic_noisy_context.json)

#### Pretrained encoder, alternate split, and text audit


| Cora configuration | Seeds | One-shot accuracy / AUROC | Adaptive accuracy / AUROC | Learned CE / BCE |
|---|---|---|---|---|
| Frozen DistilRoBERTa | 1 | 3.91 / 0.4902 | — | 9.5699 / 0.7205 |
| Full lexical vocabulary smoke test | 1 | 4.69 / 0.5049 | 4.69 / 0.5034 | 10.3883 / 0.7336 |
| Seed-specific link re-splits, CPU | 3 | 16.63 / 0.6878 | 16.75 / 0.6880 | 4.8530 / 0.6455 |

The pretrained directional feasibility run uses 128 token positions, a 256-dimensional graph mixer, five training epochs, all 50,261 RoBERTa lexical IDs, and up to 128 masked token targets per case. The frozen encoder runs on the currently visible text. The CPU vocabulary smoke test uses up to 64 masked token targets per case. Other masked positions stay concealed. The feasibility rows use different encoders and target panels. A multi-seed pretrained comparison remains open. The alternate Cora protocol re-splits warm pairs per seed and preserves held-out positives outside training adjacency.
[Pretrained run](results/research_twinmask_v2/cora_pretrained_full_vocab_directional.json) [Full vocabulary smoke test](results/research_twinmask_v2/cora_full_vocab.json) [Link re-splits](results/research_twinmask_v2/cora_resplit.json)

The cached raw text audit counted blank strings and median whitespace-delimited words. WikiCS had 0 blank among 11,701 texts (median 219 words). Reddit had 0 among 33,434 (median 87). Instagram had 99 among 11,339 (median 16). These counts describe coverage and length. The ogbn-arxiv cache had 0 blanks among 169,343 texts (median 167). Its cached links are undirected and lack an evaluation-ready temporal split, so no ogbn-arxiv V2 result is reported.

#### Reproduction commands

```powershell
python -m scripts.research_ensemble
python -m scripts.research_twinmask --seeds 42 43 44 45 46 47 48 49 50 51
python -m scripts.research_twinmask --clock-training tied
python -m scripts.research_twinmask --no-rollout
python -m scripts.research_twinmask --hard-edges
python -m scripts.research_twinmask --datasets wikics reddit instagram
python -m scripts.research_twinmask --datasets synthetic_reference synthetic_low_homophily synthetic_weak_text synthetic_noisy_context
python -m scripts.research_twinmask --datasets cora --seeds 42 --epochs 5 --encoder hf_models/all-distilroberta-v1 --width 256 --length 128 --full-vocab --max-token-targets 128 --directional-only
python -m scripts.research_twinmask --datasets cora --resplit-links --device cpu
python -m scripts.research_twinmask --datasets cora --seeds 42 --epochs 5 --device cpu --full-vocab --max-token-targets 64
python -m unittest scripts.test_research_pilots -v
python -m scripts.report_research_v2
```

## Research continuation pilots (2026-11-27)

The following is the archived V1 report. Its sparse TwinMask clock comparison predates the correction above. Historical checkpoint replay used a separate implementation and protocol.

Both research proposals were implemented as pilots. Each pilot ran on Cora, CiteSeer, and PubMed with paired seeds 42, 43, and 44 on an NVIDIA GeForce RTX 5070 (12 GiB, PyTorch 2.7.1+cu128). The results do not yet support a diffusion-specific ensemble advantage or an adaptive reconstruction advantage over one-shot inference.

```powershell
python -m scripts.research_ensemble
python -m scripts.research_twinmask
python -m scripts.research_twinmask --clock-training tied
python -m unittest scripts.test_research_pilots -v
```

### Proposal 1. One shared denoiser for multiple views

[Proposal](01_Diffusion_Ensemble_Research.md) and [runner](scripts/research_ensemble.py). This node-classification pilot used cached frozen MiniLM embeddings, training-node standardization, the existing training-edge graph, and seeded random 60/20/20 label splits. The validation set was split in two for head checkpointing and view selection. One GraphSAGE denoiser trained for 100 epochs on Gaussian VP corruption and optional observed-edge dropout. Five deterministic DDIM trajectories used distinct noise seeds and policies, with 18 denoiser calls across the pool. The shared classifier trained on alternating raw and diffusion views. The Gaussian control used the same perturbation seeds and matched each diffusion view's mean cosine to MiniLM. The selector started with the shared raw prediction and added views using validation NLL improvement per call under a nine-call cap, with cosine at least 0.80 and identity retrieval at least 0.90.

Test accuracy in percent (mean ± sample SD across seeds).

| Arm | Cora | CiteSeer | PubMed |
|---|---:|---:|---:|
| Raw MiniLM + GraphSAGE | 85.49 ± 2.22 | 74.76 ± 1.18 | 89.36 ± 0.32 |
| Shared head on raw input | 85.73 ± 2.31 | 74.50 ± 0.86 | 89.70 ± 0.69 |
| One diffusion view | 85.24 ± 2.84 | 74.45 ± 0.72 | 89.78 ± 0.74 |
| Two diffusion views | 85.42 ± 2.60 | 74.66 ± 0.77 | 89.82 ± 0.81 |
| Five diffusion views | 85.12 ± 2.79 | 74.19 ± 1.10 | 89.49 ± 0.83 |
| Five cosine-matched Gaussian views | 85.06 ± 2.58 | 74.14 ± 1.03 | 89.45 ± 0.80 |
| Five MC-dropout predictions | 85.24 ± 2.24 | 74.40 ± 1.27 | 89.43 ± 0.33 |
| Five independently seeded GraphSAGE heads | 85.49 ± 2.44 | 76.54 ± 0.77 | 89.74 ± 0.34 |
| Validation-selected diffusion views + raw | 85.55 ± 2.46 | 74.50 ± 0.65 | 89.70 ± 0.69 |
| One diffusion view at t=30 | 85.06 ± 3.05 | 73.62 ± 1.76 | 88.70 ± 0.55 |
| Fixed-noise autoencoder at t=30 | 84.38 ± 1.66 | 73.41 ± 1.42 | 86.98 ± 0.70 |
| Mean of five diffusion embeddings | 85.30 ± 2.38 | 74.19 ± 0.89 | 89.32 ± 0.66 |
| Raw + four views from one denoiser | 85.24 ± 2.48 | 74.61 ± 1.18 | 89.52 ± 0.79 |
| Raw + four independently trained denoisers | 85.18 ± 2.46 | 74.50 ± 1.42 | 89.53 ± 0.70 |

| Dataset | Diffusion-5 minus Gaussian-5 (pp) | Diffusion-5 / Gaussian-5 / dropout-5 NLL | Selected view counts by seed | One / four denoiser fit median (s) | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|
| Cora | +0.06 | 0.4881 / 0.4887 / 0.4829 | 1, 2, 2 | 0.26 / 1.01 | 131 |
| CiteSeer | +0.05 | 0.8367 / 0.8384 / 0.8376 | 3, 1, 2 | 0.26 / 1.01 | 148 |
| PubMed | +0.03 | 0.2797 / 0.2808 / 0.2842 | 0, 0, 0 | 0.46 / 1.83 | 761 |

On CiteSeer, the independently seeded heads had higher mean accuracy and lower NLL. On PubMed, two diffusion views had the highest mean accuracy among these arms, and the selector added zero views in all three runs. Across datasets, five diffusion views differed from matched Gaussian noise by at most 0.07 percentage points in mean accuracy. The experiment used cached embeddings, so the encoder extraction cost is unavailable. Per-stage timings are exploratory and include GPU warm-up and, for some runs, concurrent activity. With raw input plus four views, the difference between the shared and independently trained denoisers ranged from -0.01 to +0.10 percentage points across datasets. The four independently trained denoisers were evaluated with the same first four sampling policies and the same shared head as the one-denoiser arm. The fixed-noise autoencoder used the same capacity, corruption timestep, noise draw at inference, and classifier as the single-view t=30 comparison. The common classifier was trained on the shared denoiser's views, which may favor that arm. A same-split DIVERGE cost comparison, formal noninferiority, and matched end-to-end compute remain open.

Mean pairwise test prediction disagreement and oracle accuracy. Oracle accuracy counts a node as correct if any member predicts its label and is a diagnostic upper bound.

| Dataset | Diffusion-5 disagreement | Gaussian-5 disagreement | Diffusion-5 oracle accuracy | Gaussian-5 oracle accuracy |
|---|---:|---:|---:|---:|
| Cora | 4.76% | 4.89% | 88.25% | 88.50% |
| CiteSeer | 5.57% | 5.51% | 78.32% | 78.27% |
| PubMed | 5.64% | 5.67% | 93.54% | 93.48% |

Per-seed macro-F1, NLL, Brier, ECE, view identity and cosine, pairwise double fault, error correlation, selected policies, and stage timings are in [Cora](results/research_ensemble/cora.json), [CiteSeer](results/research_ensemble/citeseer.json), and [PubMed](results/research_ensemble/pubmed.json). Local checkpoints under `checkpoints/research_ensemble/` contain the denoiser, splits, test labels, and per-arm and per-member probabilities.

### Proposal 2. Asymmetric conditional graph inpainting

[Proposal](02_TwinMask_Asymmetric_Research.md) and [runner](scripts/research_twinmask.py). The pilot trained a 64-wide token embedding, local token convolution, and two sparse GraphSAGE layers for 120 epochs. It used the first 16 positions after the start token and the 1,024 most frequent tokens in the fitting nodes. Out-of-vocabulary tokens map to a separate symbol and are outside the token reconstruction target set. The full DistilRoBERTa architecture and full-length vocabulary reconstruction remain future work.

Training, controller fitting, and development selection used disjoint token target nodes and disjoint edge target pools. Validation/test positives from the cached graph split were absent from training adjacency. Visible positive pairs were used as message edges, candidate pairs carried explicit unknown/absent/present states, and observed states were clamped throughout inference. The shuffled test candidate list came from the cached reference positives and sampled negatives. As a result, edge AUROC and AP measure ranking within that candidate set. Open-world retrieval remains untested.

The four test cases used random text/edge mask probabilities of (0.25, 0.75), (0.50, 0.50), (0.75, 0.25), and (0.90, 0.90). The ridge controller selected from joint, text-priority, edge-priority, and corrective actions using observable mask fractions, prediction entropies, and remaining calls. Its training targets came from free-running simulated actions scored on a fixed subset kept masked. Corrective remasking was a confidence heuristic capped at 10% of currently generated entries per modality. The last call filled all remaining targets. The fixed comparator was selected per seed using mean development token accuracy plus edge accuracy.

Mean final token accuracy (%) / test candidate edge AUROC across three seeds and four corruption cases. Edge probabilities are taken at the final reveal of each target.

| Policy | Calls | Cora | CiteSeer | PubMed |
|---|---:|---:|---:|---:|
| One-shot | 1 | 16.80 / 0.6882 | 13.40 / 0.6937 | 30.87 / 0.8611 |
| Joint | 4 | 16.42 / 0.6839 | 12.95 / 0.6702 | 30.24 / 0.8522 |
| Text priority | 4 | 16.60 / 0.6836 | 13.12 / 0.6890 | 30.92 / 0.8594 |
| Edge priority | 4 | 16.79 / 0.6865 | 13.20 / 0.6817 | 30.44 / 0.8602 |
| Joint with correction | 4 | 16.34 / 0.6813 | 12.72 / 0.6675 | 30.08 / 0.8508 |
| Entropy | 4 | 16.79 / 0.6865 | 13.20 / 0.6817 | 30.44 / 0.8602 |
| Learned controller | 4 | 16.79 / 0.6865 | 13.24 / 0.6818 | 30.67 / 0.8586 |
| Development-selected fixed | 4 | 16.54 / 0.6826 | 13.23 / 0.6851 | 30.99 / 0.8597 |

| Dataset | Token vocabulary coverage | Structure → text CE gain | Text → edge BCE gain | Dev. action-gain Spearman | One-shot / adaptive latency (ms) |
|---|---:|---:|---:|---:|---:|
| Cora | 71.0% | +0.0225 | +0.0040 | 0.813 | 5.2 / 16.4 |
| CiteSeer | 72.5% | +0.0188 | +0.0526 | 0.802 | 5.4 / 16.9 |
| PubMed | 81.8% | -0.0169 | +0.0638 | 0.904 | 27.9 / 99.5 |

Directional gains measure the increase in initial held-out loss when evidence from the other modality is removed. Positive values favor retaining the observed evidence. In this setting, structure modestly helped Cora and CiteSeer text, while removing it slightly improved PubMed token CE. PubMed and CiteSeer showed larger text-to-edge gains than Cora. The learned policy did not consistently improve over the development-selected fixed policy at equal calls. One-shot had the highest edge AUROC on all three datasets and was faster. The controller correlations summarize pooled development actions and do not establish calibration, cross-graph transfer, or causal effects.

Per-seed and per-case losses, edge AP/Brier/accuracy, token accuracy, directional gains, timing, and action traces are in [Cora](results/research_twinmask/cora.json), [CiteSeer](results/research_twinmask/citeseer.json), and [PubMed](results/research_twinmask/pubmed.json). Local checkpoints under `checkpoints/research_twinmask/` include the model, vocabulary, split indices, candidate masks, and fitted controller coefficients. Candidate retrieval, synthetic interventions, whole-node/span and bridge missingness, full baseline comparisons, and confirmatory multi-split inference remain open.

Before the mandatory final joint fill, the learned policy chose the edge action in 36 of 36 Cora decisions, 32 of 36 CiteSeer decisions, and 19 of 36 PubMed decisions. PubMed also had 15 text and two correction actions. The controller used a wider range of actions on PubMed, where it still trailed one-shot on both primary endpoints.

#### Tied-clock training ablation

The tied control used the same architecture, seeds, training length, and case generation, with both training clocks set to the text clock at every step. The evaluation allowed different text and edge mask rates in both arms. The table reports one-shot token accuracy (%) / edge AUROC.

| Dataset | Mixed-clock training | Tied-clock training |
|---|---:|---:|
| Cora | 16.80 / 0.6882 | 16.67 / 0.6859 |
| CiteSeer | 13.40 / 0.6937 | 13.29 / 0.7033 |
| PubMed | 30.87 / 0.8611 | 30.68 / 0.8644 |

One-shot token accuracy favored mixed-clock training by a small margin on all three datasets. The effect on edge AUROC varied by graph. Full tied-control results include every policy and mask case. [Cora](results/research_twinmask/cora_tied.json) [CiteSeer](results/research_twinmask/citeseer_tied.json) [PubMed](results/research_twinmask/pubmed_tied.json)

#### Historical checkpoint replay

The original seed-0 joint checkpoints were replayed with the repository's cold-node episode builder, 50% text masking, three repeat masks, and free-running generation. The table compares the historical sampler's one-call diagonal token NLL against its best eight-call nonlearned path by token NLL. These values use the legacy scoring and episode protocol separately from the sparse pilot above.

| Dataset | One call | Eight calls, edge_lead2 |
|---|---:|---:|
| Cora | 4.243 | 4.547 |
| PubMed | 2.966 | 3.117 |

Replay outputs include all fixed paths and one- and eight-call sweeps. [Cora](results/research_twinmask/cora_legacy_replay.json) [PubMed](results/research_twinmask/pubmed_legacy_replay.json)

```powershell
python -m twinmask.path_rl --dataset cora --no_policy --sweep 1 8 --output-tag research_twinmask
python -m twinmask.path_rl --dataset pubmed --no_policy --sweep 1 8 --output-tag research_twinmask
```

### Paired pilot estimates

Each contrast uses one paired difference per seed. TwinMask differences average the four masks within each seed before computing the interval. The 95% Student t intervals have two degrees of freedom and describe three resampled splits of the same graph. They provide no cross-graph inference.

| Dataset | Diffusion-5 vs Gaussian-5 accuracy (pp) | Adaptive vs selected fixed token accuracy (pp) | Adaptive vs selected fixed edge AUROC |
|---|---:|---:|---:|
| Cora | +0.06 [-0.47, +0.59] | +0.25 [-0.29, +0.79] | +0.0038 [-0.0112, +0.0188] |
| CiteSeer | +0.05 [-0.17, +0.28] | +0.02 [-0.08, +0.11] | -0.0034 [-0.0143, +0.0076] |
| PubMed | +0.03 [-0.12, +0.19] | -0.33 [-1.10, +0.45] | -0.0011 [-0.0059, +0.0037] |

---

## 1. TL;DR

A **text-attributed graph (TAG)** gives every node a piece of raw text and links nodes with edges. Almost all existing methods encode the text *once* with a language model and then run a GNN on top; text and structure are never modelled **jointly and generatively**.

**TwinMask** treats both modalities as discrete variables and corrupts them with a *single* absorbing-state (mask) diffusion process:

* every **text token** of every node can become `[MASK]`;
* every **edge variable** `e_ij ∈ {absent, present}` can become `[MASK]` ("unknown", which is **not** the same as "absent");
* the two modalities get **independent noise levels** `(t_x, t_a) ∈ [0,1]²`.

One denoiser is trained on the whole square, so one model represents all conditionals at once:

| Conditioning `(t_x, t_a)` | What the model computes | Task |
|---|---|---|
| `t_x = 0`, edge masked | `p(A ∣ X, partial A)` | link prediction |
| `t_x = 0`, all of a node's edges masked | `p(edges of v ∣ text)` | cold-start node attachment |
| `t_a = 0`, text masked | `p(X ∣ A, neighbours' text)` | text completion / imputation |
| `t_x = 1` (text fully masked) | `p(A ∣ partial A)` | link prediction with **missing text** |
| `(1, 1) → (0, 0)` | `p(X, A)` | joint TAG generation (planned) |
| `(0, 0)` | node state `h_v` | representation learning |

To my knowledge (literature search, Nov 2026) no published work diffuses **raw node text tokens and edges together with separate noise levels**. Closest work: TAG-DLM (masks only the answer token of a frozen LLaDA-8B), Mao et al. EMNLP 2026 (autoregressive over Eulerian traversals), UTGDiff (molecules; text only as condition), GraphMaker (attribute *vectors*, not text).

**Headline findings so far** (details in §7):

1. **Robust to a missing modality.** Trained with text masking, TwinMask still ranks edges from structure alone (graph-only link-prediction AUC 57–95 across 7 datasets); an edge-only model collapses to chance (42–46 on Cora/CiteSeer).
2. **"Unknown" ≠ "absent" matters.** Treating masked edges as absent (edge-dropout view) lowers graph-only AUC by up to **5 points** on PubMed (86.9 → 81.9) compared with a dedicated `[MASK]` edge state.
3. **Hard-negative link prediction (HeaRT).** On PubMed, token-level TwinMask reaches MRR 18–25 versus 13.6 for plain text similarity and 7.6–8.8 for embedding-level GCN models.
4. **When does neighbourhood text help reconstruct text?** An NELBO-difference estimator (*TSMI*) shows it helps on homophilous graphs (Cora, CiteSeer, PubMed) and **hurts** on low-homophily ones (WikiCS, Instagram, Reddit).
5. **Cheap ensemble diversity (DIVERGE bridge, 4 seeds, fairly tuned).** Four label-free diffusion views plus the raw embedding of a frozen 22M encoder match the 5×2.6B-parameter LoRA ensemble of DIVERGE within noise (Cora 88.65 vs 88.88; CiteSeer 78.25 vs 78.21), and combining both pools gives the best mean on both datasets (Cora 89.58, CiteSeer 79.08; gains of +0.7 / +0.9 points, smaller than the seed std of the baseline).
6. **A claim that did *not* survive:** separate noise levels beat a shared level for text reconstruction at the *embedding* level (Phase 0), but not consistently at the token level (Phase 1). See §9.

---

## 2. Method

### 2.1 Formulation

For a sampled subgraph `S` with `k` nodes:

* node text `s_i = (s_i^1 … s_i^L)`, `L = 64` RoBERTa tokens;
* edge variables `e_ij ∈ {0,1}` for all pairs `i<j` in `S`.

**Forward process** (absorbing state, linear schedule `α(t) = 1 − t`), independent per modality:

```
q(z_token | s)  = α(t_x)·δ_s  + (1 − α(t_x))·δ_[M]        every token, independently
q(z_edge  | e)  = α(t_a)·δ_e  + (1 − α(t_a))·δ_[M]        every pair, independently
(t_x, t_a) ~ U[0,1]²  with small probability mass on the borders t=0 (condition) and t=1 (marginal)
```

A masked edge state means *unknown*. This is what separates the method from edge dropout, where a removed edge is indistinguishable from a non-edge.

**Training objective.** The continuous-time MDLM bound for two modalities is

```
NELBO = E_{t_x,t_a}[ (1/t_x) Σ_{masked tokens} −log p_θ(s | z)  +  λ (1/t_a) Σ_{masked pairs} −log p_θ(e | z) ]
```

For optimisation I use the low-variance form **mean cross-entropy over masked items** (the 1/t weights produced gradient spikes on short sequences and few edge variables); the exact 1/t-weighted NELBO is used for *evaluation*, in particular for TSMI.

### 2.2 Architecture

```mermaid
flowchart TB
    subgraph IN["Input subgraph  (k ≤ 12 nodes, sampled from the TRAINING graph only)"]
        T["Node texts → 64 tokens each"]
        A["Edge states per pair<br/>absent / present / [M] / self"]
    end

    T --> MT["Token masking at rate t_x"]
    A --> MA["Edge masking at rate t_a"]

    subgraph DEN["Denoiser"]
        direction TB
        EMB["Token embeddings + time embedding(t_x, t_a)<br/>on the first token"]
        L1["DistilRoBERTa layers 1-2<br/>(intra-node attention)"]
        M1["GraphMixer 1"]
        L2["layers 3-4"]
        M2["GraphMixer 2"]
        L3["layers 5-6"]
        M3["GraphMixer 3"]
        POOL["mean-pool tokens → node state h_v<br/>+ degree embedding"]
        FM["final GraphMixer"]
        EMB --> L1 --> M1 --> L2 --> M2 --> L3 --> M3 --> POOL --> FM
    end

    MT --> EMB
    MA -.edge states bias the<br/>inter-node attention.-> M1
    MA -.-> M2
    MA -.-> M3
    MA -.-> FM

    L3 --> LMH["LM head on masked token positions<br/>→ text loss"]
    FM --> EH["Pair head: [h_i⊙h_j, |h_i−h_j|, structural features]<br/>→ edge logit → edge loss on masked pairs"]
    FM --> REP["h_v : node representation"]
```

**GraphMixer (the cross-modal component).** Per-node summaries (mean over the node's tokens) attend to each other with a **per-head additive bias that depends on the *current, partially masked* state of each edge**:

```
bias(i,j,head) = E[state_ij][head],   state ∈ {absent, present, masked, self}
init:  absent = −4,  masked = −2,  present = 0,  self = 0     (GAT-like attention over the visible graph)
```

so every edge that is revealed during denoising immediately changes which neighbours exchange information (**dynamic topology attention**). TAG-DLM uses a *fixed* topology mask. The mixer's update is added back to *every token* of the node, so neighbour information reaches the LM head. Because softmax attention is a weighted mean and cannot count neighbours when inputs coincide (e.g. fully masked text), **degree** and **path-count features** (`log1p(A²)`, `log1p(A³)`, `log1p(deg_i·deg_j)`, computed from *visible* edges only) are given explicitly to the node states and the pair head.

**Backbone.** DistilRoBERTa (82M), encoder weights initialised from `all-distilroberta-v1` (contrastively pre-trained, same tokenizer), MLM head from `distilroberta-base`. Mixed precision (bf16), chunked LM head to keep the 50k-vocab logits small. 6.7 GB peak on a 12 GB RTX 5070.

### 2.3 Variants used in ablations

| Variant | `t_x` | `t_a` | Purpose |
|---|---|---|---|
| `joint` | independent | independent | the proposed model |
| `tied` | = `t_a` | = `t_x` | does a *separate* noise level help? |
| `text_only` | sampled | 0 | no edge supervision (GraphMAE-style) |
| `edge_only` | 0 | sampled | no text corruption (MaskGAE-style) |
| `static` | indep. | indep. | masked edges treated as *absent* in attention |
| `notopo` | indep. | indep. | structure-blind mixer |

### 2.4 Text–Structure Mutual Information (TSMI)

Because one model gives both `p(A|X)` and `p(A)` (set `t_x = 1`), differences of Monte-Carlo NELBO estimates (stratified `t` grid, 10 points) give a diagnostic of *how much one modality tells about the other*:

```
TSMI_A  = [NELBO(A | X masked) − NELBO(A | X)] / #pairs          nats per edge variable   (text → structure)
TSMI_X  = [NELBO(X | A masked) − NELBO(X | A)] / #tokens         nats per token           (edges → text)
CTX_X   = [NELBO(X_v | v alone) − NELBO(X_v | subgraph)] / #tok  nats per token           (neighbourhood text+edges → text)
```

These are differences of *upper bounds* on `−log p`, i.e. estimates, not exact mutual information.

---

## 3. Experimental setup

### 3.1 Data
Seven TAGs with raw text, taken from the graph files of the DIVERGE / LLMNodeBed codebase (`datasets/*.pt`):

| Dataset | Nodes | Edges (undirected) | Classes | Edge homophily | Domain |
|---|---:|---:|---:|---:|---|
| Cora | 2,708 | 5,278 | 7 | 0.810 | citations |
| CiteSeer | 3,186 | 4,225 | 6 | 0.744 | citations |
| PubMed | 19,717 | 44,324 | 3 | 0.802 | citations |
| WikiCS | 11,701 | ~216k | 10 | 0.654 | web links |
| Instagram | 11,339 | ~78k | 2 | 0.586 | social |
| Reddit | 33,434 | ~151k | 2 | 0.588 | social |
| ogbn-arxiv | 169,343 | 1.17M | 40 | 0.654 | citations |

### 3.2 Leak-free link-prediction protocol
* 5 % of nodes are **cold-start nodes**: *all* their edges are removed from training.
* Of the remaining edges: 10 % test, 5 % validation, rest = training graph. Every model, including its subgraph sampler and neighbourhood features, sees **only the training graph**.
* **Random negatives:** AUC / AP. **HeaRT hard negatives** (Li et al., NeurIPS 2023 D&B): for each test positive, 100 negatives from the top-ranked candidates of a mix of Resource-Allocation, Personalized-PageRank and text-cosine, excluding every true edge of the *full* graph. Metrics: MRR, Hits@K. Deviations from the paper: K = 100 (not 500), ≤ 1000 positives per dataset.

### 3.3 Node-classification protocols
Representations are learned **without labels**; labels enter only in the final classifier.
* **Low-label:** public split (Cora 140 / CiteSeer 120 / PubMed 60 labels; WikiCS 5 %; Instagram, Reddit 10 %).
* **Supervised 60/20/20:** random splits, 10 repetitions (linear probe) or DIVERGE's seeds 42–45 (GNN).

### 3.4 Evaluation lessons (documented because they changed conclusions)
* Random-negative AUC saturates (all models 97–99 on Cora/CiteSeer/PubMed); **HeaRT changes the ranking of methods** — on citation graphs plain text cosine beats every trained model.
* A linear probe with fixed `C = 1` underestimated Phase 0 by up to 5 points (CiteSeer 68.5 → 73.4 when tuned on validation); using the link-prediction graph (77 % of edges) for classification cost another 0.5–2 points.
* A training-free baseline (MiniLM + 2-step SGC propagation) already reaches 82.2 on Cora; any NC claim must be compared against it. A supervised GCN on our features (82.46) reproduces GLBench's published GCN (82.11), confirming that the data pipeline is sound.

---

## 4. Phases

| Phase | Content | Status |
|---|---|---|
| **0** | Embedding-level version of the original idea: Gaussian diffusion on MiniLM embeddings + masked edges, GCN denoiser, 6 variants × 6 datasets × 3 seeds, HeaRT | ✅ done |
| **1** | Token-level TwinMask (DistilRoBERTa), Cora/CiteSeer/PubMed × 4 variants | ✅ done (1 seed) |
| **2** | Topology ablations, (t_x, t_a) heatmaps, TSMI on 6 datasets | ✅ done |
| **3** | Scale to ogbn-arxiv (169k nodes) | ✅ done (joint, edge_only) |
| **Bridge** | DIVERGE-style view ensembles with diffusion views, fair tuning | ✅ done for Cora and CiteSeer (4 seeds) |
| 4 | TAG generation (joint sampling), comparison with Mao et al. / GraphMaker | ⏳ planned |

---

## 5. How the work relates to DIVERGE (view-ensemble bridge)

DIVERGE (my earlier work, the codebase this project draws data and tooling from) fine-tunes an LLM five times with different LoRA initialisations *using labels*, trains one GNN per resulting embedding, and ensembles them. Its diversity source is the **fine-tuning trajectory**. The bridge asks whether **diffusion noise schedules** can supply diversity instead — without fine-tuning any LLM:

```mermaid
flowchart LR
    TXT["Node texts"] --> MINI["Frozen MiniLM (22M)<br/>no labels"]
    MINI --> RAW["raw embedding"]
    MINI --> J["joint diffusion<br/>(t_x ⟂ t_a)"]
    MINI --> TD["tied diffusion<br/>(t_x = t_a)"]
    MINI --> TO["text-only diffusion"]
    MINI --> EO["edge-only diffusion"]
    RAW --> G1[SAGE]
    J --> G2[SAGE]
    TD --> G3[SAGE]
    TO --> G4[SAGE]
    EO --> G5[SAGE]
    G1 & G2 & G3 & G4 & G5 --> ENS["Ensemble<br/>uniform or DIVERGE WeightLearner"]
    ENS --> PRED["Node classification"]

    TXT2["Node texts + labels"] --> LORA["DIVERGE: Gemma-2-2B LoRA ×5 inits<br/>(PiSSA, Gaussian, EVA, LoftQ, Orthogonal)"]
    LORA --> V5["5 label-aware embeddings"]
    V5 --> G6["5 × SAGE"] --> ENS
```

Reused directly from DIVERGE: `GNNEncoder`, its 60/20/20 splitter and seeds (42–45), its tuned GNN configs, its `WeightLearner` ensemble head and search grid, and its cached fine-tuned embeddings.

**"Label-free" means:** the *representation* (MiniLM + diffusion) never sees class labels; labels are used only to train the SAGE heads and ensemble weights, exactly as in DIVERGE. DIVERGE additionally injects labels into the text encoder via LoRA fine-tuning.

---

## 6. Repository layout

```
twinmask/
  data.py          dataset extraction from DIVERGE, leak-free link splits, cold-start nodes, MiniLM + RoBERTa-token caches
  heart.py         HeaRT hard-negative construction (RA + PPR + text-cosine) and MRR / Hits@K
  phase0.py        embedding-level joint diffusion (Gaussian text + masked edges, GCN denoiser), all variants
  model.py         TwinMask: DistilRoBERTa + GraphMixer (dynamic topology attention) + token/edge heads
  sampler.py       subgraph sampler over the TRAINING graph, forced-mask pairs, cold-start masking
  tm.py            token-level corruption, losses, training loop with validation-AUC checkpoint selection, evaluation
  phase2.py        (t_x, t_a) heatmaps, NELBO estimators, TSMI / CTX_X
  common.py        time embedding, noise-level sampler, metrics, two-setting NC probe
  report.py        aggregates results/*.json into results/REPORT.md and figures
scripts/
  calibrate.py, calibrate_tm.py   controlled checks of where NC gaps to published numbers come from
  diverge_bridge.py               untuned DIVERGE x diffusion-view ensembles
  tune_bridge.py                  fair tuning: diffusion grid, per-view SAGE search, learned ensembles
  diag_lp.py, smoke_test.py, reeval.py
run_all.py         job queue for phases 1-3
research_proposal.md   original literature review and research plan (Persian)
```

---

## 7. Results

All numbers are percentages unless stated. "±" is the standard deviation over seeds where more than one seed was run.

### 7.1 Phase 0 — embedding level (MiniLM + GCN denoiser; 3 seeds)

**Does the graph help reconstruct a heavily noised text embedding?** Cosine gain (graph visible − graph hidden) at noise level `t = 0.9`:

| Dataset | `text_only` | **`joint`** | `tied` |
|---|---:|---:|---:|
| Cora | +0.146 | **+0.089** | +0.039 |
| CiteSeer | +0.116 | **+0.081** | +0.045 |
| PubMed | +0.039 | 3.44 / 5.02 | 761 |
| WikiCS | +0.139 | **+0.118** | +0.051 |
| Instagram | +0.023 | **+0.021** | +0.010 |
| Reddit | +0.047 | **+0.042** | +0.016 |

Independent noise levels (`joint`) make the model exploit the graph **2–3× more than a shared level** (`tied`), on all six datasets. At `t = 0.5` the gain is ≈ 0 (the text alone suffices).

**Link prediction, random vs. hard negatives (MRR, HeaRT):**

| | Text cosine | GAE | Edge-only | Tied | Joint |
|---|---:|---:|---:|---:|---:|
| Cora | **20.6** | 16.7 | 15.7 | 13.7 | 14.5 |
| PubMed | **13.6** | 7.6 | 8.3 | 8.8 | 7.7 |
| WikiCS | 8.1 | **23.0** | 18.7 | 15.6 | 19.1 |
| Reddit | 3.1 | 3.3 | 3.4 | 3.7 | **3.9** |

With random negatives every trained model scores AUC 97–99 on the citation graphs, which hides these differences.

### 7.2 Phase 1 — token-level TwinMask (1 seed, DistilRoBERTa; 2000 steps, checkpoint selected on validation AUC)

| Dataset | Model | LP AUC | **Graph-only LP AUC** (text masked) | HeaRT MRR | Cold-start AUC | NC 60/20/20 |
|---|---|---:|---:|---:|---:|---:|
| Cora | edge-only | 85.0 | 42.5 | **16.0** | **83.6** | 73.5 |
| | tied | 89.5 | **67.0** | 15.8 | 81.5 | 80.4 |
| | **joint** | **89.8** | 64.5 | 14.7 | 83.0 | 79.9 |
| CiteSeer | edge-only | 89.2 | 45.6 | **24.7** | **89.6** | 64.9 |
| | tied | **90.2** | **61.3** | 15.0 | 85.0 | 67.1 |
| | **joint** | 90.0 | 57.4 | 16.3 | 89.5 | 67.5 |
| PubMed | edge-only | **97.3** | 78.4 | **24.6** | **90.6** | 88.3 |
| | tied | 96.6 | **88.7** | 20.8 | 87.6 | 87.7 |
| | **joint** | 96.5 | 86.9 | 18.4 | 89.4 | 88.1 |

Other datasets (joint only): WikiCS LP 97.5 / graph-only 95.1 / HeaRT MRR 25.3; Instagram 90.7 / 85.3 / 8.8; Reddit 77.6 / 69.9 / 3.5. **ogbn-arxiv (169k nodes, 8000 steps):** joint LP 98.1, graph-only 88.4 (edge-only: 73.1), HeaRT MRR 7.5, text-completion NLL 3.65 with context vs. 3.76 alone.

**Reading the table:**
* Training with text masking is what makes the model usable when **text is missing** (graph-only AUC 57–95 vs. 42–78 for edge-only). The price is some ranking precision when text is available (edge-only wins HeaRT/cold-start on all three datasets).
* On the largest citation graph here (PubMed), HeaRT MRR (18–25) is well above plain text similarity (13.6) and embedding-level GNNs (≤ 8.8). On the two smaller graphs TwinMask is *not* better than text similarity.
* `joint` is **not** consistently better than `tied` at the token level.

### 7.3 Phase 2 — does the dynamic topology matter? (1 seed)

| | Graph-only LP, Cora | Graph-only LP, PubMed | HeaRT MRR, PubMed |
|---|---:|---:|---:|
| **Dynamic** (`[M]` is its own state) | **64.5** | **86.9** | 18.4 |
| Static (`[M]` treated as absent) | 63.4 | 81.9 | **22.7** |
| No topology | 61.3 | 79.7 | 20.4 |

Distinguishing *unknown* from *absent* helps when edges are scarce (graph-only), but the static variant ranks hard negatives better on PubMed. Node classification is unaffected (79.6–79.9 / 88.0–88.1).

### 7.4 Phase 2 — TSMI across datasets

| Dataset | Edge homophily | TSMI_A (text → edges) | TSMI_X (edges → text) | **CTX_X** (neighbourhood → text) |
|---|---:|---:|---:|---:|
| Cora | 0.810 | 0.139 | 0.023 | **+0.066** |
| PubMed | 0.802 | 0.043 | −0.002 | **+0.022** |
| CiteSeer | 0.744 | 0.103 | 0.008 | **+0.017** |
| WikiCS | 0.654 | 0.131 | −0.003 | −0.009 |
| Reddit | 0.588 | 0.020 | −0.001 | **−0.020** |
| Instagram | 0.586 | 0.007 | −0.004 | **−0.024** |

(nats per item.) **`CTX_X` is positive on the three homophilous graphs and negative on the three low-homophily ones**: giving a node its neighbours' text *helps* reconstruct its own text when neighbours are similar and *hurts* when they are not. `TSMI_A` does not track homophily monotonically (WikiCS is high), so homophily alone does not explain text→structure information. Estimates come from one trained model per dataset and 256 subgraphs; they are not yet accompanied by confidence intervals.

The `(t_x, t_a)` heatmaps (edge AUC and text NLL over the 5×5 noise grid, one model) are written to `results/fig_heatmap_*.png`.

### 7.5 Bridge — DIVERGE-style ensembles (60/20/20, seeds 42–45)

**Untuned diffusion views** (DIVERGE's SAGE config reused as is), 4 seeds:

| | Cora | CiteSeer |
|---|---:|---:|
| MiniLM + SAGE | 86.90 ± 0.89 | 76.57 ± 2.08 |
| One DIVERGE view (Gemma-2-2B LoRA, best init) | 88.84 | 77.55 |
| **DIVERGE ensemble (5 LoRA inits)** | **88.88 ± 1.40** | **78.21 ± 1.64** |
| **MiniLM + 4 diffusion views** (no LLM fine-tuning) | 88.65 ± 0.62 | 77.43 ± 1.89 |
| DIVERGE-5 + diffusion views | 88.47 | 78.06 |

**Fairly tuned, 4 seeds** — all selection on validation only: the diffusion model's hyper-parameters are grid-searched per variant (hidden size, epochs, edge-loss weight, boundary probability; 16 configs) on the seed-42 split and then frozen; SAGE is searched per view and per seed with DIVERGE's search space (64 random candidates, top 6 re-run with 3 GNN seeds); the learned ensemble is DIVERGE's own `WeightLearner` with its grid. DIVERGE's Gemma views keep DIVERGE's own tuned configs.

| Test accuracy (%, mean ± std over seeds 42–45) | Cora | CiteSeer |
|---|---:|---:|
| MiniLM + SAGE (single view, tuned) | 87.68 ± 0.48 | 76.96 ± 1.52 |
| best single diffusion view (tuned; Cora: text-only, CiteSeer: joint) | 88.24 ± 0.46 | 77.08 ± 1.67 |
| best single DIVERGE view (Gemma-2-2B LoRA, best init) | 88.84 ± 1.20 | 77.55 ± 1.54 |
| **DIVERGE-5, uniform** | **88.88 ± 1.40** | 78.21 ± 1.64 |
| DIVERGE-5, learned head | 88.70 ± 1.29 | 78.10 ± 2.03 |
| **MiniLM + 4 diffusion views, uniform** (no LLM fine-tuning) | 88.65 ± 0.71 | **78.25 ± 1.17** |
| MiniLM + 4 diffusion views, learned head | 88.47 ± 0.66 | 78.06 ± 1.20 |
| **DIVERGE-5 + our 5 views, learned head** | **89.58 ± 0.55** | **79.08 ± 1.68** |

What the data support:
* A **single** diffusion view is only marginally better than the raw encoder (≤ 0.6 points, inside the noise). The benefit appears when views are **combined**: four diffusion views plus the raw embedding reach the level of the five-LoRA DIVERGE ensemble, with minutes of training instead of five fine-tunings of a 2.6B model.
* Per-seed, the combined pool is at least as good as DIVERGE-5 with the learned head on 3 of 4 seeds on both datasets (Cora seed 45: −0.2); against DIVERGE-5 with uniform averaging it wins on 2 of 4 seeds on Cora and 3 of 4 on CiteSeer. The mean gains (+0.7, +0.9) are **smaller than the baseline's seed-to-seed std (1.4–1.6)**, so this is evidence of complementarity, not of a significant improvement.
* **Tuning did not change the conclusion.** Re-tuning SAGE for our views moved single-view accuracy by −0.5 to +0.8 points, with no consistent direction; the untuned comparison reported earlier (Cora 88.65 vs 88.88, CiteSeer 77.43 vs 78.21) was not biased in a way that mattered.
* My reproduction of DIVERGE (Gemma-2-2B only, plain or head-weighted averaging) is about 1 point below the numbers in the DIVERGE paper (89.9 / 79.5), which uses additional design choices not replicated here.
* Not yet tested: PubMed, WikiCS, ogbn-arxiv; learned ensembles on more than 4 seeds; significance tests.

---

### 7.6 Bridge 2 — DIVERGE semi-supervised augmentation (public low-label split)

`scripts/semi_aug_bridge.py` re-uses DIVERGE's `tags_data_augmentation` pipeline (Llama-3.2-1B LoRA adapters trained on the 140 / 120 public labels; generated nodes added as pseudo-labelled training nodes with DIVERGE's own validation-selected edge rule, pseudo-label handling, lambda and tuned GNN hyper-parameters = `ref_selected`). Our label-free diffusion views are trained on the same LLM embeddings over the same (augmented) graph: 4 variants on PiSSA + `joint` on the other 4 inits = 8 views. Test accuracy %, mean ± std over 3 GNN repeats, uniform softmax ensembles.

| | Cora base | Cora + aug | CiteSeer base | CiteSeer + aug |
|---|---|---|---|---|
| DIVERGE-1 (5 LoRA inits) | 84.48 ± 0.51 | 84.75 ± 0.44 | 71.60 ± 0.68 | 73.92 ± 0.10 |
| 8 diffusion views only | 82.79 ± 0.32 | 85.09 ± 0.29 | 71.55 ± 0.28 | 73.68 ± 0.02 |
| DIVERGE-1 + diffusion views | 83.67 ± 0.19 | **85.49 ± 0.34** | 71.93 ± 0.19 | **74.08 ± 0.14** |

Reading: (i) the reproduction matches DIVERGE's published numbers (84.8 / 85.0 on Cora) within ~0.3; (ii) without augmentation diffusion views do *not* help on Cora (-0.8); (iii) with augmented nodes the diffusion views are as good as or better than the LoRA members and the combined pool is best (+0.7 Cora, +0.2 CiteSeer over DIVERGE + aug) -- the label-free views seem to be more robust to noisy pseudo-labelled nodes/edges; (iv) all gaps are about one seed-std, only 3 repeats and one LoRA seed, so this is suggestive, not significant. A first run with naive kNN edges and no lambda (not DIVERGE's selected config) collapsed DIVERGE to 73% while diffusion views stayed at 77.8% -- an unfair baseline, discarded. Validation-screened ensembles did not help.

#### 7.6b All augmentation methods of the DIVERGE semi-supervised study x diffusion views

Every method is built with DIVERGE's own code (`diva_ideas.build`: its filters, weights, soft labels and edges, unmodified). Test accuracy %, mean ± std over 3 GNN repeats (LoRA seed fixed); "both" = 5 DIVERGE-1 members + 8 diffusion views, uniform ensemble.

| method | Cora D-1 | Cora diff | Cora both | CiteSeer D-1 | CiteSeer diff | CiteSeer both |
|---|---|---|---|---|---|---|
| ref_plain | 85.15 | 84.77 | 85.15 | 73.92 | 74.05 | 74.19 |
| ref_selected | 85.15 | 84.93 | 85.19 | 74.32 | 73.75 | 73.94 |
| com_vote | 85.09 | 85.06 | 85.43 | 73.69 | 73.88 | 74.31 |
| own_attn | 85.38 | 85.12 | **85.85** | 74.08 | 74.28 | 74.25 |
| own_viewdis | 85.49 | 85.06 | 85.54 | 74.06 | 73.86 | 74.07 |
| lit_conf | 85.43 | 85.17 | 85.41 | 74.18 | 73.89 | 74.19 |
| diva2_ens | 84.95 | 84.86 | 85.17 | 74.37 | 74.20 | 74.24 |
| diva_mi | 83.82 | 84.11 | 84.53 | 73.43 | 74.02 | 74.21 |
| diva_full | 84.53 | 84.78 | 85.14 | 72.80 | 73.69 | 73.73 |

Reading (honest): adding diffusion views changes the DIVERGE-1 ensemble by +0.27 (Cora) / +0.25 (CiteSeer) on average over the 9 methods, with a range of -0.4 to +0.9 and a seed-std of 0.1-0.8, i.e. **no significant effect for any single method**. The only consistent pattern is that the weakest DIVERGE variants (`diva_mi`, `diva_full`) gain most (+0.6 to +0.9), because diffusion views do not suffer from their pseudo-label handling as much; for the strongest ones the gain is ~0. The diffusion views alone are on par with the LoRA members (within about 0.5). No combination beats the best DIVERGE-only number by more than seed noise on CiteSeer; on Cora the best is `own_attn` + diffusion 85.85 vs 85.49 (`own_viewdis`, DIVERGE-1 only). Not run: `own_bandit`, `own_calcls`, `own_verifier_w`, the remaining `lit_*` methods, PubMed/WikiCS, other LoRA seeds. Note the embeddings come from LoRA adapters fine-tuned with the public labels, so the diffusion views are label-free in their own loss only, not in the pipeline.

## 8. Reproducing

```bash
# environment: Python 3.13, PyTorch 2.7 (CUDA 12.8), PyG 2.8, transformers 4.55, sentence-transformers 5.1
# 0. point twinmask/data.py:DIVERGE_DIR at a folder with the DIVERGE / LLMNodeBed graph files (cora.pt, ...)
python -m twinmask.data --datasets cora citeseer pubmed wikics instagram reddit arxiv   # splits + MiniLM + tokens
python -m twinmask.heart                                  # HeaRT hard negatives
python -m twinmask.phase0 --epochs 1000                   # Phase 0 (all datasets/variants/seeds)
python run_all.py                                         # Phases 1-3 (several GPU-hours on one 12 GB card)
python -m twinmask.report                                 # -> results/REPORT.md + figures
python scripts/tune_bridge.py cora citeseer               # bridge with fair tuning (~2 GPU-hours for both datasets)
```
`all-distilroberta-v1` weights are expected under `hf_models/all-distilroberta-v1/` (or set `TWINMASK_ENCODER=none` to use the plain MLM encoder).

Hardware used: one RTX 5070 (12 GB). All stages were run with fixed seeds; raw per-run JSON files are in `results/`.

---

## 9. Limitations and honest assessment

* **The central hypothesis is only partly supported.** Independent noise levels beat a shared level for *embedding-level* text denoising (6/6 datasets), but at the token level `joint` ≈ `tied`. Possible causes (untested): early stopping on link-prediction AUC leaves the text side under-trained on small graphs; 82M parameters memorise 3k-node graphs.
* **TwinMask is not a state-of-the-art classifier.** On Cora and CiteSeer it is weaker than our own embedding-level model and than a training-free MiniLM + SGC baseline; neighbour information is not absorbed into the node state the way a GCN does it (adding 2-step propagation to its representations adds ~9 points on Cora). On ogbn-arxiv, NC (≈ 62–66 with a linear probe on a 20k-node subsample) is far below published supervised numbers.
* **Compared with SOTA TAG methods the setting differs:** they fine-tune LLMs (up to 8B) with labels; this project uses an 82M label-free backbone and a lightweight head. I do not claim to beat them.
* **Single seed** in Phase 1–3, no confidence intervals for TSMI, 64-token text truncation (keeps ~50 % of a Cora abstract, ~20 % of a PubMed abstract; measured effect on a frozen encoder: 0.3–2 points).
* **Model-selection hygiene:** during the architectural debugging of TwinMask on Cora I looked at test metrics as well as validation metrics when choosing among model variants. The reported Phase-1 numbers use validation-based checkpoint selection, but those design decisions should be re-verified on validation only before publication. The bridge tuning (Stage A/B) selects on validation only.
* **Not yet done:** joint TAG generation (the main justification for a *generative* joint model), comparison against LLaDA-8B-based TAG-DLM (does not fit on a 12 GB GPU), larger/pre-trained diffusion backbones.
* **Novelty claim** ("first joint masked diffusion over raw node text and edges") is based on a literature search in Nov 2026 and must be re-checked before submission; the area moves quickly.

## 10. Roadmap

1. Extend the bridge study to PubMed / WikiCS / ogbn-arxiv (where Phase 0 was strongest) with more seeds and paired significance tests.
2. Token-level model with DIVERGE's recipe — LoRA instead of full fine-tuning, 512 tokens, explicit propagation in the readout, checkpoint selection on a combined text+edge criterion — to address memorisation and the classification gap.
3. **Label as a third modality** in the same diffusion (text + edges + label): classification becomes denoising of the label variable, putting supervision inside the model.
4. TAG generation with confidence-ordered unmasking over both modalities; evaluation against Mao et al. and GraphMaker; multi-seed runs and TSMI confidence intervals.

## 11. Related work

* Masked diffusion language models: D3PM (Austin et al., 2021), SEDD (Lou et al., 2024), MDLM (Sahoo et al., 2024), LLaDA (Nie et al., 2025).
* Per-modality noise levels / multimodal diffusion: UniDiffuser (Bao et al., 2023), UniDisc (Swerdlow et al., 2025).
* Diffusion for graphs: GDSS, DiGress, EDGE, GraphMaker (Li et al.), DDM (Yang et al., 2023), Graffe (2025).
* Masked graph autoencoders: GraphMAE (Hou et al., 2022), MaskGAE (Li et al., 2023).
* LMs on TAGs: TAPE (He et al., 2024), GLEM, LLaGA, **TAG-DLM** (Chen et al., 2026), Mao et al. (EMNLP 2026), UTGDiff (2024).
* Evaluation: HeaRT (Li et al., NeurIPS 2023 D&B), GLBench (Li et al., NeurIPS 2024).
* DIVERGE: diverse LoRA initialisations + learned GNN ensembles for TAGs (my prior work).

## 12. Contact

*Author:* `<your name>` — `<affiliation / email / webpage>`
