import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from .acquire import select


class MeanSAGE(nn.Module):
    def __init__(self, width, classes):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * width, 48), nn.ReLU(), nn.Linear(48, classes))

    def forward(self, own, neighbor_mean):
        return self.net(torch.cat((own, neighbor_mean), 1))


def fit(view, labels, nodes, classes, seed):
    x, _, mean = view
    torch.manual_seed(seed)
    model = MeanSAGE(x.shape[1], classes).to(x.device)
    opt = torch.optim.AdamW(model.parameters(), lr=.003)
    target = labels[nodes].to(x.device)
    own, neighbors = x[nodes], mean[nodes]
    for _ in range(20):
        opt.zero_grad()
        nn.functional.cross_entropy(model(own, neighbors), target).backward()
        opt.step()
    with torch.no_grad():
        prob = model(x, mean).softmax(-1)
    return model, prob


@torch.no_grad()
def flips(nodes, proposals, degree, views, models, probs):
    """Independent undirected additions. One SAGE layer changes only the two endpoints."""
    node = torch.as_tensor(nodes, device=degree.device)
    candidate = proposals[node]
    row, slot = (candidate >= 0).nonzero(as_tuple=True)
    a, b = node[row], candidate[row, slot]
    votes = torch.zeros(candidate.shape, device=degree.device)
    spill = torch.zeros_like(votes)
    for view, model, prob in zip(views, models, probs):
        x, total, _ = view
        for src, dst, accumulator in ((a, b, votes), (b, a, spill)):
            new_mean = (total[src] + x[dst]) / (degree[src] + 1)
            changed = model(x[src], new_mean).argmax(1) != prob[src].argmax(1)
            accumulator[row, slot] += changed.float() / len(models)
    return votes.max(1).values.cpu().numpy(), {
        "probes": len(a), "target_flip": float(votes.sum()) / max(len(a), 1),
        "other_endpoint_flip": float(spill.sum()) / max(len(a), 1)}


def entropy(prob):
    return -(prob * prob.clamp_min(1e-10).log()).sum(-1)


def trial(graph, seed, stage, method, gamma=0., selector="greedy", view_mode="both", bonus=.15,
          *, splits=None, view_indices=None):
    views, degree, proposals, groups, labels, classes = graph
    # The raw ablation keeps ensemble size and initialization seeds fixed.
    if view_mode == "raw_ensemble":
        views = [views[0], views[0]]
    elif view_mode == "single":
        views = views[:1]
    stride = max(3, len(views))
    members = list(enumerate(views))
    if view_indices is not None:
        members = [members[i] for i in view_indices]
        views = [view for _, view in members]
    if splits is None:
        n = len(labels)
        split = np.random.default_rng(312).permutation(n)
        splits = np.split(split, [int(.6*n), int(.8*n)])
    train, valid, test = splits
    evaluation = valid if stage == "validation" else test
    rng = np.random.default_rng(seed)
    initial = rng.choice(train, 16, replace=False)
    labeled = initial.copy()
    pool = np.setdiff1d(train, labeled)
    start = time.perf_counter()
    curve = []
    for step in range(4):
        fitted = [fit(v, labels, labeled, classes, seed*100 + step*stride + i)
                  for i, v in members]
        models, probs = zip(*fitted)
        stack = torch.stack(probs)
        avg = stack.mean(0)
        pred = avg[evaluation].argmax(1).cpu().numpy()
        true = labels[evaluation].numpy()
        point = dict(labels=len(labeled), accuracy=float((true == pred).mean()),
                     macro_f1=f1_score(true, pred, labels=np.arange(classes), average="macro", zero_division=0),
                     seconds=time.perf_counter()-start)
        curve.append(point)
        if step == 3:
            break
        if method == "random":
            chosen = rng.choice(pool, 8, replace=False)
        else:
            u = entropy(avg)
            d = u - entropy(stack).mean(0)
            base = {"entropy": u, "disagreement": d, "combined": u+d}[method]
            scores = (base[pool] / np.log(classes)).cpu().numpy()
            order = np.lexsort((pool, -scores))[:128]
            nodes, scores = pool[order], scores[order]
            if gamma:
                cf, point["interventions"] = flips(nodes, proposals, degree, views, models, probs)
                scores = scores + gamma * cf
            chosen = select(nodes, scores, groups, 8, selector, bonus)
        assert len(np.unique(chosen)) == 8 and np.isin(chosen, pool).all()
        point["queried"] = chosen.tolist()
        labeled = np.r_[labeled, chosen]
        pool = np.setdiff1d(pool, chosen)
    return dict(seed=seed, initial=initial.tolist(), curve=curve,
                aulc=sum((a["accuracy"] + b["accuracy"]) / 6 for a, b in zip(curve, curve[1:])),
                seconds=curve[-1]["seconds"])
