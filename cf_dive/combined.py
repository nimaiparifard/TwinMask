"""Run structural counterfactual acquisition with DIVERGE and TwinMask views."""

import hashlib
import json
import time
from pathlib import Path
from statistics import mean

import numpy as np
import scipy
import sklearn
import torch
from scipy.sparse import csr_matrix
from torch.nn.functional import normalize

from .__main__ import save
from .diverge_views import arguments, prepare_augmented
from .experiment import trial
from .graph import ROOT, edge_proposals
from .report import table


def prepare_graph(task, features, device):
    src, dst = torch.cat([task.und, task.und.flip(0)], dim=1).cpu().numpy()
    adj = csr_matrix((np.ones(len(src), dtype=np.float32), (src, dst)), shape=(task.n, task.n))
    degree = torch.as_tensor(np.diff(adj.indptr), device=device)[:, None]
    text = normalize(features["lora_pissa"]).numpy()
    proposals = torch.as_tensor(edge_proposals(adj, text), device=device)
    anchors = np.random.default_rng(38).choice(task.n_base, 24, replace=False)
    groups = (text @ text[anchors].T).argmax(1)
    views = []
    for x in features.values():
        x = (x - x.mean(0)) / x.std(0).clamp_min(1e-5)
        total = torch.from_numpy(adj @ x.numpy()).to(device)
        views.append((x.to(device), total, total / degree.clamp_min(1)))

    # Reserve the public validation labels for tuning. Partition the original public
    # test nodes so acquisitions are outside the LoRA adapters' public training set.
    train, val, test = [np.flatnonzero(mask[:task.n_base].cpu().numpy()) for mask in task.masks]
    if len(np.unique(np.r_[train, val, test])) != len(train) + len(val) + len(test):
        raise ValueError("Overlapping public DIVERGE masks")
    pool, test = np.split(np.random.default_rng(312).permutation(test), [int(.6 * len(test))])
    if len(pool) < 40 or not len(val) or not len(test):
        raise ValueError("The public masks do not support a 40-label acquisition study")
    return (views, degree, proposals, groups, task.y.cpu(), task.C), (pool, val, test)


def run(dataset, args):
    start = time.perf_counter()
    task, features, fingerprint = prepare_augmented(dataset, args.diverge, args.cache,
                                                    args.device, epochs=1 if args.smoke else 1000)
    graph, splits = prepare_graph(task, features, args.device)
    names = list(features)
    lora = [i for i, name in enumerate(names) if name.startswith("lora_")]
    diff = [i for i, name in enumerate(names) if name.startswith("diff_")]
    if (len(lora), len(diff)) != (5, 8):
        raise ValueError(f"Expected five LoRA and eight diffusion views, found {names}")
    del task, features

    package = Path(__file__).parent
    code = hashlib.sha256(b"".join((package / name).read_bytes() for name in
                          ("combined.py", "diverge_views.py", "experiment.py", "graph.py", "acquire.py")))
    signature = dict(inputs=fingerprint, code=code.hexdigest(), smoke=args.smoke,
                     numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__)
    path = args.out / f"{dataset}.json"
    data = json.loads(path.read_text(encoding="utf8")) if path.exists() else dict(
        dataset=dataset, signature=signature, views=names,
        splits={key: nodes.tolist() for key, nodes in zip(("pool", "validation", "test"), splits)},
        preparation_seconds=time.perf_counter()-start, runs={})
    if data.get("signature") != signature:
        raise ValueError(f"Inputs or settings changed for {dataset}. Choose a fresh --out directory.")
    runs = data["runs"]

    def series(stage, label, seeds, **settings):
        values = []
        method = settings.pop("method", "combined")
        for seed in seeds:
            key = f"{stage}/{label}/{seed}"
            if args.repeat or key not in runs:
                print(f"{dataset}: {key}", flush=True)
                runs[key] = trial(graph, seed, stage, method, splits=splits, **settings)
                save(path, data)
            values.append(runs[key])
        return values

    validation_seeds, test_seeds = ((11,), (41,)) if args.smoke else ((11, 12, 13), range(41, 46))
    weights, bonuses = ((0., .3), (.15,)) if args.smoke else ((0., .1, .3, .6), (.05, .15, .3))
    scores = {(g, b): mean(row["aulc"] for row in series(
        "validation", f"gamma-{g}_bonus-{b}", validation_seeds, gamma=g, bonus=b))
        for g in weights for b in bonuses}
    gamma, bonus = max((key for key in scores if key[0] > 0), key=scores.get)
    _, null_bonus = max((key for key in scores if key[0] == 0), key=scores.get)
    data.update(gamma=gamma, bonus=bonus, null_bonus=null_bonus,
                validation_winner=max(scores, key=scores.get))
    test = {}
    for label, indices in (("combined", None), ("lora", lora), ("diffusion", diff)):
        for suffix, weight in (("cf", gamma), ("no_cf", 0.)):
            arm = f"{label}_{suffix}"
            test[arm] = series("test", arm, test_seeds, gamma=weight, bonus=bonus, view_indices=indices)
    test["combined_tuned_no_cf"] = series("test", "combined_tuned_no_cf", test_seeds, bonus=null_bonus)
    test["combined_cf_milp"] = series("test", "combined_cf_milp", test_seeds, gamma=gamma,
                                      bonus=bonus, selector="milp")
    test["random"] = series("test", "random", test_seeds, method="random")
    save(path, data)

    for i, seed in enumerate(test_seeds):
        initial = test["combined_cf"][i]["initial"]
        for arm, trials in test.items():
            if trials[i]["initial"] != initial:
                raise ValueError(f"Unpaired initial labels for {dataset}/{arm}/{seed}")
    rows = []
    for label, trials in test.items():
        budgets = [100 * mean(row["curve"][i]["accuracy"] for row in trials) for i in range(4)]
        aulc = 100 * mean(row["aulc"] for row in trials)
        f1 = 100 * mean(row["curve"][-1]["macro_f1"] for row in trials)
        rows.append([label, *(f"{value:.2f}" for value in (*budgets, aulc, f1))])
    paired = [100 * (a["aulc"] - b["aulc"]) for a, b in
              zip(test["combined_cf"], test["combined_no_cf"], strict=True)]
    print(f"{dataset}: complete, paired CF AULC difference {mean(paired):+.3f} pp", flush=True)
    return [f"## {dataset}",
            f"Validation-selected CF weight {gamma}, matched coverage bonus {bonus}, "
            f"and independently selected zero-CF bonus {null_bonus}. "
            f"Pool, validation, and test sizes are {tuple(map(len, splits))}.",
            table(["Arm", "16 labels", "24 labels", "32 labels", "40 labels", "AULC", "F1 @ 40"], rows),
            "Combined CF minus matched no-CF AULC by seed (percentage points) "
            + ", ".join(f"{d:+.2f}" for d in paired) + f". Mean {mean(paired):+.2f}."]


def main():
    parser = arguments()
    parser.description = __doc__
    parser.add_argument("--out", type=Path, help="Directory for resumable JSON and Markdown results")
    parser.add_argument("--repeat", action="store_true", help="Recompute trials using the existing view cache")
    parser.add_argument("--smoke", action="store_true", help="Use one diffusion epoch and one seed per stage")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this Python environment")
    if args.out is None:
        args.out = Path(ROOT, "results", "cf_dive_combined_smoke" if args.smoke else "cf_dive_combined")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    sections = ["# Counterfactual acquisition with DIVERGE and diffusion views",
                f"Run mode is {'smoke check' if args.smoke else 'full experiment'}. "
                "Metrics are percentages. The budget counts newly acquired real-node labels for the SAGE heads. "
                "The frozen DIVERGE LoRA caches and augmented topology carry supervision from the existing public "
                "training set. Synthetic nodes contribute graph context. The original public test nodes are "
                "partitioned into an acquisition pool and a held-out test set. Public validation selects the "
                "acquisition weights. Counterfactuals add one undirected edge to the SAGE graph with views frozen."]
    for dataset in dict.fromkeys(args.datasets):
        sections.extend(run(dataset, args))
    path = args.out / "REPORT.md"
    path.write_text("\n\n".join(sections) + "\n", encoding="utf8")
    print(f"Report: {path}", flush=True)


if __name__ == "__main__":
    main()
