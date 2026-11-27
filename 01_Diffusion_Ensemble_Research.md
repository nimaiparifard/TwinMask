# Proposal 1 — One Encoder, Many Views
## Budgeted Conditional Diffusion Ensembles for Text-Attributed Graphs
Research and implementation specification | 24 November 2026 | Proposed, not validated

### 1. Research question and novelty boundary
Can a single shared conditional denoiser generate useful, semantically faithful alternative node representations at a lower total cost than independently trained language encoders? This is an amortized view-generation problem, not another taxonomy of diversity sources (MADE), and not a new LoRA initialization method (DIVERGE). Do not claim that sampling, diffusion representation learning or ensembles are themselves novel. The candidate contribution is a graph-conditioned distribution of views with a measurable fidelity–complementarity trade-off, learned once and sampled under explicit compute budgets. If ordinary noise or dropout matches it, reject the diffusion-specific claim.

Existing internal evidence: the tuned MiniLM bridge (one frozen MiniLM, four independently trained denoisers, five GNNs including raw input) reports 88.65 ± 0.71 on Cora and 78.25 ± 1.17 on CiteSeer, versus DIVERGE uniform 88.88 ± 1.40 and 78.21 ± 1.64, four runs. Combined ten-member learned ensemble: 89.58 ± 0.55 and 79.08 ± 1.68. These are not results for a single denoiser; cost and statistical equivalence are unestablished. Low-label results are mixed and use a different split. Source: supplied Pasted markdown(3).md, sections 1–3 and result tables. Do not merge protocols.

### 2. Literature map and collision audit
The following are primary paper/author pages located in targeted literature searches. This is an abstract-level evidence map, not a systematic full-text review. Before claiming priority, retrieve full papers, inspect equations and search their citing papers, particularly 2025–2026 works; mark unavailable implementations explicitly.

| Work | Core idea | Collision / required comparison |
|---|---|---|
| [Graffe, Chen et al. 2025](https://arxiv.org/abs/2505.04956) | Graph encoder conditions a diffusion decoder; studies why denoising helps representations. | High: diffusion representations already exist. Compare its encoder/probe with shared sampling; our target is multiple useful views, not reconstruction theory. |
| [GNSD, Lin et al., ICML 2024](https://proceedings.mlr.press/v235/lin24x.html) | Stochastic graph dynamics model predictive uncertainty using learned drift and forcing. | High: stochastic graph predictions already exist. Compare calibration, single model repeated inference and total cost. |
| [UDPD, Yu et al., ICML 2026](https://proceedings.mlr.press/v306/yu26ab.html) | Separate semantic/structural diffusion branches, uncertainty-modulated interaction. | High for graph-conditioned denoising; use as robust representation baseline. Do not claim first semantic/structural diffusion. |
| [TAG-DLM, Chen et al. 2026 preprint](https://arxiv.org/abs/2606.31166) | Masked diffusion language model with topology attention over linearized neighborhoods. | Different prediction space; comparison on downstream tasks where feasible. |
| [GraphMAE, Hou et al. 2022](https://arxiv.org/abs/2205.10803) | Masked node-feature reconstruction. | Essential simpler baseline; sample masks at inference with matched cost. |
| [GraphMAE2, Hou et al. 2023](https://github.com/THUDM/GraphMAE2) | Enhanced decoding for masked self-supervised graph learning. | Strong non-diffusion counterpart. |
| [GiGaMAE, Shi et al. 2023](https://arxiv.org/abs/2308.09663) | Reconstructs multiple latent targets containing graph and attribute information. | Multiple complementary targets precede us; compare, avoid claiming novelty for multiple representations alone. |
| [UniDiffuser, Bao et al., ICML 2023](https://proceedings.mlr.press/v202/bao23a.html) | Independent modality timesteps support several conditional distributions in a shared model. | Shared conditional diffusion is established; novelty must be useful amortized views on TAGs. |
| [DoG, Wang et al. 2025](https://arxiv.org/abs/2503.12563) | Generates synthetic graph structures for node classification. | Alternative graph augmentation; distinguish generating representations of existing nodes from new nodes. |

Additional mandatory baseline families: deep ensembles, MC dropout, Gaussian perturbations, snapshot ensembles, shared-head test-time augmentation and label-space diffusion. Audit existing MADE experiments before duplicating any. No priority claim until full-text comparison against stochastic single-model graph ensembles is complete.

### 3. Data and notation
G=(V,A,X), N nodes, raw text X, training labels y_L. Encoder E produces H0 in R^(N×d). Start frozen MiniLM, d from actual checkpoint, then repeat with one frozen or one LoRA-adapted backbone used by DIVERGE. Normalize per feature using training nodes only; record mean/std. Use original topology A_obs, never hidden edges. For transductive classification unlabeled test features and observed edges are allowed but test labels are not. Inductive protocol: fit denoiser and normalization only on training graph; new nodes handled by neighbor sampling. Separate public low-label, random 60/20/20, and official OGB splits. Never compare across split families.

### 4. Minimal generative model (V0)
Forward VP corruption: h_t = sqrt(alpha_bar_t) h0 + sqrt(1-alpha_bar_t) eps, eps~N(0,I). Independently drop observed edges at rate r (optional conditioning perturbation, not a claim of edge diffusion); A_r subset A_obs. T=100, cosine schedule, t uniform in {1,...,T}, r sampled from {0,.1,.3,.5}. Condition graph network on sinusoidal timestep and r. Avoid supplying uncorrupted target h0 through a skip/conditioning path. 2–3 layers GraphSAGE or graph transformer, hidden 256, LayerNorm, residuals from noisy states only, dropout .1. Predict eps, shape [batch_nodes,d]; denoising loss mean squared error on central nodes. Optional x0 estimate = (h_t-sqrt(1-alpha_bar_t) eps_hat)/sqrt(alpha_bar_t); clip only based on training distribution. Train with AdamW lr 1e-3, weight decay 1e-4, 100–300 epochs, validation denoising loss early stop 20; these are starting settings, not optimized results. Keep graph conditioning fixed per sampling trajectory initially.

Generate view k via partial corruption of H0 at t_start, using a recorded random seed and optional sampled A_r. Reverse with DDIM (eta=0 or .5) on a decreasing timestep grid; use standard scheduler implementation, test alpha indexing and terminal variance. Output reconstructed x0 in original embedding space, not arbitrary hidden layers. Start t_start in {10,30,50}, steps {1,4,8}; compare K={1,2,5,10}. Partial corruption uses observed test inputs, not their labels. At full noise without clean conditions, node identity may not survive; do not use full-noise unconditional samples as aligned node views. Evaluate identity retrieval, cosine fidelity and neighborhood preservation. Store per-node predictions and embedding IDs. Cache samples once for the fixed-ensemble condition; report cache costs.

### 5. Candidate method (V1): fidelity-constrained sample selection
Core hypothesized improvement: select a small set of sampling policies rather than train a model for each policy. Policy a=(t_start,r,eta,steps); candidate policies numbered P, each uses fixed seed schedule. For candidate k obtain predictions p_k with shared classifier or a matched separate head. Validation score J(S)=CE(y_val,mean p_S)+lambda*cost(S), minimized subject to mean cosine(H_k,H0)>=tau and identity retrieval >=rho. Fidelity thresholds fixed from development data, e.g. tau=.8 and rho=.9 as initial exploratory choices; report sensitivity. Greedy add policy with largest validation CE reduction per marginal inference cost; stop at budget or nonpositive improvement. Use independent validation subsets for classifier fitting, calibration and selection (or nested CV on small graphs). Compare random selection, top individual accuracy, greedy without fidelity and raw averaging. Selection from a precomputed pool saves deployment cost only, not pool-generation cost; count both. Stronger extension: train a small policy selector from input-only statistics on training/validation graphs, assess transfer without tuning on test labels. Defer until V0 succeeds.

Avoid artificial repulsion between embeddings: maximizing distance alone creates useless noise. Use validation complementarity as selection signal; proposed optional fidelity regularizer MSE or cosine on predicted x0 should be ablated. If stochastic samples barely differ, report collapse, do not create diversity by arbitrary rotations (CKA insensitive to rotations and heads can undo them).

### 6. Inference and classifier controls
A: one shared GNN trained with a random view each epoch, average probabilities over K samples at test. B: K independent GNNs, one fixed representation each, matched against K GNN seed ensemble. C: average embeddings then one GNN. D: raw H0 single classifier. E: single denoiser deterministic reconstruction. Use same GraphSAGE layers, hidden size, training epochs and search budget across arms. Softmax each head before averaging; calibrate temperatures on held-out validation only. Compare logit averaging as secondary ablation. No supervised labels in denoiser unless a separately labeled task-guided variant is clearly reported. A frozen common classifier is needed to isolate view effects from head randomness; run both shared and independent heads.

### 7. Implementation contract
Suggested repository:
```
src/data.py                 # splits, graph provenance, text cache
src/diffusion/schedule.py    # VP forward and DDIM reverse
src/diffusion/model.py       # time-conditioned GNN
src/diffusion/sample.py      # chunked view sampler
src/classifier.py           # shared and independent heads
src/select.py               # validation-only policy selection
src/metrics.py              # quality, diversity, calibration
src/profile.py              # synchronized GPU timing and peak memory
configs/{pilot,full}.yaml
scripts/{train,sample,evaluate,aggregate}.py
tests/{test_shapes,test_schedule,test_split,test_repro}.py
```
Config fields: dataset, split_id, encoder_id, encoder_hash, seed_data, seed_encoder, seed_denoiser, seed_sample, seed_head, t_start, r, eta, steps, K, compute_budget, head_mode, selection_mode. Artifacts: config.yaml, split.npz, embeddings.pt, model.pt, pred.npz (node_id,y,p[N,C]), metrics.json, profile.json, commit_sha. Do not store all N×K×d tensors on GPU; use CPU memmap, fp16 cache if accuracy unaffected, sample sequentially. Use PyTorch/PyG, neighbor sampler fanouts [15,10], batch 512 starting point for arxiv; full graph small datasets. For minibatch sampling keep overlapping-neighbor states consistent within a trajectory, or label approximation and compare full-graph on small data. 12 GB GPU: mixed precision, gradient accumulation, no dense adjacency.

Training pseudocode:
```
H = frozen_encoder(text); fit_normalizer(train_nodes)
for batch in loader:
    t, r, eps = sample_conditions()
    Ht = q(H, t, eps); Ar = drop_observed_edges(A, r)
    eps_hat = model(Ht, Ar, t, r)
    loss = mse(eps_hat[centers], eps[centers]); update(loss)
freeze(model)
for policy in candidate_policies:
    Hk = ddim_partial(H, A, policy, seed)
    train_or_apply_head(Hk); save_validation_predictions()
S = select_on_validation_only()
evaluate_once(S, test_nodes)
```
Tests: no labels passed to denoiser; t=0 identity; fixed noise deterministic when eta=0; decreasing alpha schedule; no test labels in selector; reproducible node alignment; duplicate views produce same mean; K=1 matches single model; actual model count exactly one.

### 8. Experiments and analysis
Pilot: Cora/CiteSeer/PubMed, 3 paired seeds for screening; final at least 10 paired seeds, multiple splits as applicable. Add WikiCS and ogbn-arxiv, one noncitation TAG with verified raw text and licensing; official splits. Baselines: raw encoder+GNN, GNN seed ensemble, MC dropout, Gaussian noise (match cosine distortion), edge dropout, mix of raw and GraphMAE/GraphMAE2 views, standard denoising autoencoder (one noise level, matched capacity), independent denoisers, DIVERGE (same backbone where possible), Graffe/GNSD/UDPD if reproducible. Include equal K and equal total GPU-hour comparisons separately; give baselines same tuning budget. Factorial ablation: graph conditioning off, r=0, t fixed vs variable, DDIM steps, stochasticity off, fidelity constraint off, selector off, shared vs independent classifier. At minimum run one-step prediction against iterative sampler at matched calls and report actual runtime.

Metrics: accuracy, macro-F1, NLL, Brier, ECE with fixed binning, per-class recall; CKA, pairwise disagreement, double fault (both wrong), error correlation and oracle accuracy (diagnostic upper bound only), cosine and identity retrieval. Diversity metrics on held-out nodes, no hyperparameter selection on test. Plot accuracy–total GPUh and accuracy–latency Pareto curves, K scaling, time/noise sweep. Measure encoder extraction, LoRA fitting, diffusion fitting, policy pool generation, heads, tuning and test inference separately; synchronize CUDA, warm up, report hardware, peak allocated RAM/VRAM, cache GB. Shared encoder cost charged once, not zero. Compute paired seed differences, bootstrap over independent run units, CI, effect size, Holm correction for secondary comparisons. Don't treat nodes as independent replicates; discuss graph dependence.

### 9. Decision gates and journal strategy
Gate A: V0 beats matched simple stochastic controls or offers reproducible better calibration at similar accuracy and cost. Gate B: one-denoiser model reaches four-denoiser baseline within predeclared 0.5 pp noninferiority margin on development datasets; confirm with paired lower CI > -0.5 pp. Gate C: target at least 2× lower total training cost than comparable DIVERGE, with no material inference penalty or explicit trade-off. These are proposed engineering targets, not observed results. Stop or pivot if Gaussian/dropout/DAE matches the method over datasets or samples collapse. If only accuracy rises because more GNNs are trained, do not publish as diffusion innovation. KBS / ESWA / Neurocomputing are scope candidates; verify JCR/SJR quartile for relevant year/category before submission. Main paper figures: architecture, Pareto frontier, K scaling, error correlation, robustness and ablations. Distinct from MADE: focus on amortization and constrained sampling, not a multi-axis inventory. No Q1 acceptance guarantee.
