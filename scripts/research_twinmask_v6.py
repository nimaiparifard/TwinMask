"""Development-only TwinMask tuning and paired gradient-transfer experiments."""
import argparse
import json
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
import math
import subprocess

import numpy as np
import optuna
import torch
from torch.nn import functional as F

from scripts.research_ensemble import tick
from scripts.research_twinmask import MASK, UNKNOWN
from scripts.research_twinmask_v5 import (IsolatedTwin, RATES, clock_at, edge_loss, prepare_cases, reveal,
                                          text_loss)
from twinmask.audit_protocol import graph_split, induced_raw_split
from twinmask.common import save_json, set_seed


FAMILIES = ("mlm", "structure", "structure_gat", "text_link", "gae", "sage", "gat", "twinmask")
JOINT = ("sage", "gat", "twinmask")
CONVERGENCE = (32, 100, 300, 600, 1000)
LOCKED = (3107, 3137)
SEARCH_SPLIT = 3001


@dataclass(frozen=True)
class Config:
    lr: float = .003
    wd: float = .01
    width: int = 48
    depth: int = 2
    dropout: float = 0.
    edge_weight: float = 1.
    fusion: float = 1.
    modality_dropout: float = .2
    rollout: float = 0.
    lr_text: float = 1.
    lr_structure: float = 1.
    lr_fusion: float = 1.
    normalize: bool = False
    scheduler: str = "constant"
    structural_stats: bool = False
    text_encoder: str = "cnn"
    length: int = 16
    vocab_limit: int = 1024
    learned_gates: bool = False


def grouped_parameters(model):
    groups = {"text": [], "structure": [], "fusion_heads": []}
    for name, param in model.named_parameters():
        prefix = name.split(".")[0]
        tag = "text" if prefix in {"emb", "position", "local", "pretrained_projection"} else (
            "structure" if prefix in {"structure", "layers"} else "fusion_heads")
        groups[tag].append(param)
    return groups


def combine_gradients(x, a, method):
    """Two-task projections on the intersection of differentiable parameters."""
    dot = x.dot(a)
    xx, aa = x.square().sum().clamp_min(1e-12), a.square().sum().clamp_min(1e-12)
    if method == "pcgrad":
        return x+a-dot.clamp(max=0)*(a/aa+x/xx)
    if method == "cagrad":
        average = (x+a)/2
        w = torch.linspace(0, 1, 65, device=x.device, dtype=x.dtype)
        norm = (w.square()*xx+(1-w).square()*aa+2*w*(1-w)*dot).clamp_min(1e-12).sqrt()
        objective = w*x.dot(average)+(1-w)*a.dot(average)+.4*average.norm()*norm
        best = w[objective.argmin()]
        direction = best*x+(1-best)*a
        return 2*(average+.4*average.norm()*direction/direction.norm().clamp_min(1e-8))
    return x+a


def gradient_step(losses, params, weight, method, probe, groups, loss_origin):
    tx, ta = losses
    a, b = [torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True) for loss in losses]
    shared = [i for i, (x, y) in enumerate(zip(a, b)) if x is not None and y is not None]
    record = {}
    for label, members in {"shared": params, **groups}.items():
        ids = {id(p) for p in members}
        index = [i for i in shared if id(params[i]) in ids]
        if index and probe:
            x, y = (torch.cat([seq[i].flatten() for i in index]) for seq in (a, b))
            record[label] = dict(cosine=float(F.cosine_similarity(x, y, dim=0)),
                                 text_norm=float(x.norm()), edge_norm=float(y.norm()))
    wa, wb = weight
    if method == "gradnorm" and shared:
        x, y = (torch.cat([g[i].flatten() for i in shared]).norm().clamp_min(1e-9) for g in (a, b))
        rate = torch.stack((tx.detach(), ta.detach()))/loss_origin
        # Inverse gradient magnitude times relative training rate. Closed-form balancing proxy.
        weights = rate.sqrt()/torch.stack((x, y))
        wa, wb = (weights/weights.sum()*sum(weight)).tolist()
    for p, x, y in zip(params, a, b):
        p.grad = (wa*x if y is None and x is not None else wb*y if x is None and y is not None else
                  wa*x+wb*y if x is not None else None)
    if shared and method in ("pcgrad", "cagrad"):
        left = torch.cat([a[i].flatten()*wa for i in shared])
        right = torch.cat([b[i].flatten()*wb for i in shared])
        projected = combine_gradients(left, right, method)
        for i, grad in zip(shared, projected.split([params[i].numel() for i in shared])):
            params[i].grad = grad.reshape_as(params[i])
    return record


def protocol(name, seed):
    return induced_raw_split(name, seed) if name == "computer" else graph_split(name, seed)


def locked_protocol(search, split):
    """Partition the untouched search-test pool, then draw fresh fit/dev partitions."""
    data, base, positives, negatives, audit = search
    j = LOCKED.index(split)
    rng = torch.Generator().manual_seed(split)
    pool = torch.cat((base, *positives[:2]), 1)
    counts = [g.shape[1] for g in positives[:2]]
    fit, dev, new_base = torch.randperm(pool.shape[1], generator=rng).split(counts+[pool.shape[1]-sum(counts)])
    p_test = positives[2].chunk(2, 1)[j]
    n_test = negatives[2].chunk(2, 1)[j]
    pool_neg = torch.cat(negatives[:2], 1)
    neg_fit, neg_dev = torch.randperm(len(pool_neg.T), generator=rng).split(counts)
    order = torch.randperm(len(data["tokens"]), generator=torch.Generator().manual_seed(SEARCH_SPLIT+17))
    search_fit, search_dev, reserve = order.split([len(order)*3//5, len(order)//5,
                                                   len(order)-len(order)*3//5-len(order)//5])
    eligible = torch.cat((search_fit, search_dev))
    shuffled = eligible[torch.randperm(len(eligible), generator=rng)]
    held = reserve.chunk(2)[j]
    fit_nodes, dev_nodes = shuffled.split([len(search_fit), len(search_dev)])
    graph = pool[:, new_base]
    ps, qs = (pool[:, fit], pool[:, dev], p_test), (pool_neg[:, neg_fit], pool_neg[:, neg_dev], n_test)
    digest = sha256()
    for t in (graph, *ps, *qs, fit_nodes, dev_nodes, held):
        digest.update(t.contiguous().numpy().tobytes())
    return (data, graph, ps, qs, dict(audit, split_hash=digest.hexdigest(),
                                      locked_test_positive_count=p_test.shape[1],
                                      locked_test_node_count=len(held)), (fit_nodes, dev_nodes, held))


def build_cases(shared, split, config, device):
    data, base, positives, negatives, audit, *node_groups = shared
    common = data.get("capacity_vocabulary")
    make, ids, vocab, nodes = prepare_cases(data, base, positives, negatives, split, device, config.vocab_limit,
                                           length=config.length,
                                           fixed_vocab=common[:config.vocab_limit] if common is not None else None,
                                           node_groups=node_groups[0] if node_groups else None)
    return make, ids, vocab, nodes, audit


def sample_queries(case, rng, cap, state=None):
    x, a = case.x_index, case.a_index
    if state is not None:
        mask = state[0][x] == MASK
        x = tuple(i[mask] for i in x)
        a = a[state[1][a] == UNKNOWN]
    xi = torch.randperm(len(x[0]), device=a.device, generator=rng)[:cap]
    ai = torch.randperm(len(a), device=a.device, generator=rng)[:cap]
    return tuple(i[xi] for i in x), a[ai]


def metric_score(values, vocab, family):
    if family == "mlm":
        return values["text"]/math.log(vocab)
    if family not in JOINT:
        return values["edge"]/math.log(2)
    return (values["text"]/math.log(vocab)+values["edge"]/math.log(2))/2


@torch.no_grad()
def measure(model, cases, cap=384, *, common_prefix=16):
    from sklearn.metrics import roc_auc_score

    sums = {k: [] for k in ("text", "edge", "token_accuracy", "edge_auc", "conditioning_text", "conditioning_edge")}
    for j, case in enumerate(cases):
        rng = torch.Generator(device=case.ids.device).manual_seed(80+j)
        x, a = sample_queries(case, rng, cap)
        common = x[1] < common_prefix
        x = tuple(i[common] for i in x)
        tx, ta = model.predict(case, text_query=x, edge_query=a)
        lx, la = text_loss(tx[0], case.ids[x]), edge_loss(ta[0], case.labels[a])
        sums["text"].append(float(lx.mean()))
        sums["edge"].append(float(la.mean()))
        sums["token_accuracy"].append(float((tx[0][:, 3:].argmax(-1)+3 == case.ids[x]).float().mean()))
        if case.labels[a].unique().numel() > 1:
            sums["edge_auc"].append(roc_auc_score(case.labels[a].cpu(), ta[0].cpu()))
        sums["conditioning_text"].append(float((text_loss(tx[1], case.ids[x])-lx).mean()))
        sums["conditioning_edge"].append(float((edge_loss(ta[1], case.labels[a])-la).mean()))
    return {key: float(np.mean(v)) if v else None for key, v in sums.items()}


def stage_cases(make, nodes, split, device, side):
    rng = torch.Generator(device=device).manual_seed(split+731+side)
    return [make(side, rate, nodes[side], rng) for rate in RATES]


def train(shared, split, model_seed, family, config, device, updates, *, method="weighted",
          directions=(1., 1.), fixed_steps=False, trial=None, convergence=False, diagnostic=False,
          model_factory=None, mask_bank=None):
    set_seed(model_seed)
    make, ids, vocab, nodes, _ = build_cases(shared, split, config, device)
    model = (IsolatedTwin(vocab, config.width, family, depth=config.depth, length=config.length,
                          dropout=config.dropout, fusion=config.fusion, text_encoder=config.text_encoder,
                          structural_stats=config.structural_stats).to(device) if model_factory is None else
             model_factory(make, vocab, config, family))
    model.directions, model.detach_cross = directions, method == "stop_gradient"
    if config.learned_gates:
        model.gate_logits = torch.nn.Parameter(torch.zeros(2, device=device))
    groups = grouped_parameters(model)
    optimizer = torch.optim.AdamW([{"params": p, "lr": config.lr*m}
                                  for p, m in zip(groups.values(), (config.lr_text, config.lr_structure, config.lr_fusion))],
                                 weight_decay=config.wd)
    schedule = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, updates, eta_min=1e-5)
                if config.scheduler == "cosine" else None)
    dev = stage_cases(make, nodes, split, device, 1)
    checkpoints = sorted(set([step for step in (CONVERGENCE if convergence else (16, 32, 64, 100, 160, 300))
                              if step <= updates]+[updates]))
    best, best_state, best_step, stale = math.inf, None, 0, 0
    log, accum, origin = [], [], None
    params = list(model.parameters())
    signature = sha256()
    for p in params:
        signature.update(p.detach().cpu().numpy().tobytes())
    initialization_hash = signature.hexdigest()
    started = tick(device)
    for step in range(1, updates+1):
        model.train()
        # Pair masks and target samples across architectures without sharing optimizer state.
        mask_number = step if mask_bank is None else (step-1)%mask_bank+1
        mask_rng = torch.Generator(device=device).manual_seed(split*100003+model_seed*1109+mask_number)
        aug_rng = torch.Generator(device=device).manual_seed(model_seed*4049+step)
        case = make(0, RATES[(step-1) % len(RATES)], nodes[0], mask_rng)
        state = case.initial()
        x, a = sample_queries(case, mask_rng, 384)
        if family == "twinmask" and config.rollout and float(torch.rand((), device=device, generator=aug_rng)) < config.rollout:
            with torch.no_grad():
                scores = model.predict(case, state)
                state, _ = reveal(case, state, scores, case.mask, case.edge_mask, 2, "free", aug_rng)
            x = tuple(i[state[0][x] == MASK] for i in x)
            a = a[state[1][a] == UNKNOWN]
        x_logit, a_logit = model.predict(case, state, text_query=x, edge_query=a, clocks=clock_at(case, state))
        mode = int(float(torch.rand((), device=device, generator=aug_rng)) < config.modality_dropout)
        lx = text_loss(x_logit[mode], ids[x]).mean() if family == "mlm" or family in JOINT else None
        la = edge_loss(a_logit[mode], case.labels[a]).mean() if family != "mlm" else None
        accum.append([float(lx.detach()) if lx is not None else None, float(la.detach()) if la is not None else None])
        optimizer.zero_grad()
        report = {}
        probe = diagnostic and step in checkpoints and family in JOINT
        if family in JOINT:
            w = (1., config.edge_weight)
            if config.normalize:
                w = tuple(v*2/sum(w) for v in w)
            if origin is None:
                origin = torch.stack((lx.detach(), la.detach())).clamp_min(1e-8)
            if probe or method in ("pcgrad", "cagrad", "gradnorm"):
                report = gradient_step((lx, la), params, w, method, probe, groups, origin)
            else:
                (w[0]*lx+w[1]*la).backward()
        else:
            (lx if lx is not None else la).backward()
        torch.nn.utils.clip_grad_norm_(params, 5.)
        optimizer.step()
        if schedule:
            schedule.step()
        if step not in checkpoints:
            continue
        validation = measure(model.eval(), dev)
        objective = metric_score(validation, vocab, family)
        log.append(dict(step=step, dev=validation, score=objective, gradients=report,
                        train=[float(np.mean([v[i] for v in accum if v[i] is not None])) if accum[-1][i] is not None
                               else None for i in range(2)]))
        accum = []
        if objective < best:
            best, best_step, stale = objective, step, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if trial is not None:
            trial.set_user_attr("curve", log)
            trial.report(objective, step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if not fixed_steps and stale >= 3:
            break
    if not fixed_steps:
        model.load_state_dict(best_state)
    final_hash = sha256()
    for p in model.state_dict().values():
        final_hash.update(p.detach().cpu().contiguous().numpy().tobytes())
    return model.eval(), (make, nodes, vocab), dict(curve=log, best_step=best_step,
                                                   initialization_hash=initialization_hash,
                                                   checkpoint_hash=final_hash.hexdigest(),
                                                   learned_gates=(model.gate_logits.detach().sigmoid().tolist()
                                                                  if model.gate_logits is not None else None),
                                                   seconds=tick(device)-started)


def suggest(trial, family):
    params = dict(lr=trial.suggest_float("lr", 1e-4, .01, log=True),
                  wd=trial.suggest_float("wd", 1e-6, .1, log=True),
                  width=trial.suggest_categorical("width", [32, 48, 64, 128, 256]),
                  depth=trial.suggest_int("depth", 1, 3),
                  dropout=trial.suggest_categorical("dropout", [0., .1, .2, .3]),
                  scheduler=trial.suggest_categorical("scheduler", ["constant", "cosine"]),
                  lr_text=trial.suggest_categorical("lr_text", [.5, 1., 2.]),
                  lr_structure=trial.suggest_categorical("lr_structure", [.5, 1., 2.]),
                  lr_fusion=trial.suggest_categorical("lr_fusion", [.5, 1., 2.]),
                  structural_stats=trial.suggest_categorical("structural_stats", [False, True]))
    if family in JOINT:
        params.update(edge_weight=trial.suggest_float("edge_weight", .1, 10., log=True),
                      fusion=trial.suggest_categorical("fusion", [.1, .25, .5, 1., 2.]),
                      normalize=trial.suggest_categorical("normalize", [False, True]),
                      modality_dropout=trial.suggest_categorical("modality_dropout", [0., .1, .2, .4]))
    if family == "twinmask":
        params["rollout"] = trial.suggest_categorical("rollout", [0., .25, .5, .75])
    return Config(**params)


def optimize(shared, family, count, updates, device, *, replay=()):
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=497, n_startup_trials=5),
                               pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=32))
    baseline = asdict(replace(Config(), rollout=.25 if family == "twinmask" else 0.))
    if replay:
        for record in replay:
            study.enqueue_trial(record["config"])
    else:
        study.enqueue_trial(baseline)

    def objective(trial):
        config = suggest(trial, family)
        trial.set_user_attr("config", asdict(config))
        if trial.number < len(replay):
            old = replay[trial.number]
            trial.set_user_attr("curve", old["curve"])
            for row in old["curve"]:
                trial.report(row["score"], row["step"])
            if old["status"] == "PRUNED":
                raise optuna.TrialPruned()
            trial.set_user_attr("best_step", old["best_step"])
            return old["score"]
        _, _, stats = train(shared, SEARCH_SPLIT, 71, family, config, device, updates, trial=trial)
        trial.set_user_attr("best_step", stats["best_step"])
        return min(row["score"] for row in stats["curve"])

    study.optimize(objective, n_trials=count)
    trials = [dict(number=t.number, status=t.state.name, score=t.value, **t.user_attrs) for t in study.trials]
    # Recheck top configurations on a second development fit seed before freezing.
    leaders = sorted((t for t in trials if t["status"] == "COMPLETE"), key=lambda t: t["score"])[:3]
    for leader in leaders:
        _, _, trace = train(shared, SEARCH_SPLIT, 83, family, Config(**leader["config"]), device, updates)
        leader["confirmation"] = min(r["score"] for r in trace["curve"])
        leader["selection_score"] = (leader["score"]+leader["confirmation"])/2
    best = min(leaders, key=lambda t: t["selection_score"])
    return dict(config=best["config"], steps=best["best_step"], winner=best["number"], trials=trials,
                dev_objective=best["selection_score"])


def locked_eval(name, studies, device, *, tag=""):
    outcomes = []
    search = protocol(name, SEARCH_SPLIT)
    directory = Path("checkpoints/research_twinmask_v6")
    directory.mkdir(parents=True, exist_ok=True)
    for split in LOCKED:
        shared = locked_protocol(search, split)
        for family, winner in studies.items():
            config = Config(**winner["config"])
            model, (make, nodes, _), stats = train(shared, split, 97, family, config, device,
                                                    winner["steps"], fixed_steps=True)
            checkpoint = directory/f"{name}_{split}_{family}{tag}.pt"
            torch.save(model.state_dict(), checkpoint)
            outcomes.append(dict(family=family, split=split, model_seed=97, split_hash=shared[4]["split_hash"],
                                 config=winner["config"], steps=winner["steps"], history=stats,
                                 checkpoint=str(checkpoint), checkpoint_file_sha256=sha256(checkpoint.read_bytes()).hexdigest(),
                                 test=measure(model, stage_cases(make, nodes, split, device, 2))))
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer", "pubmed", "computer"])
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--trial-updates", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=FAMILIES)
    parser.add_argument("--locked-only", action="store_true")
    parser.add_argument("--default-control", action="store_true")
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    device = torch.device(args.device)
    for name in args.datasets:
        if args.default_control:
            control = {family: dict(config=asdict(replace(Config(), rollout=1/3 if family == "twinmask" else 0.)),
                                    steps=32) for family in args.families}
            save_json(locked_eval(name, control, device, tag="_default"), "research_twinmask_v6", name+"_default.json")
            continue
        if args.locked_only:
            studies = json.loads(Path(f"results/research_twinmask_v6/{name}_search.json").read_text())
            save_json(locked_eval(name, studies, device), "research_twinmask_v6", name+"_locked.json")
            continue
        shared = protocol(name, SEARCH_SPLIT)
        studies = {}
        for family in args.families:
            studies[family] = optimize(shared, family, args.trials, args.trial_updates, device)
            save_json(studies, "research_twinmask_v6", name+"_search.json")
            print(name, family, studies[family]["dev_objective"], flush=True)
        save_json(locked_eval(name, studies, device), "research_twinmask_v6", name+"_locked.json")


if __name__ == "__main__":
    main()
