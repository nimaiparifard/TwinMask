import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


def select(pool, scores, group, budget, mode, bonus):
    # All solvers share the same shortlist and group-coverage objective.
    order = np.lexsort((pool, -scores))[:max(64, budget)]
    pool, scores = pool[order], scores[order]
    _, groups = np.unique(group[pool], return_inverse=True)
    if mode == "top":
        return pool[:budget]
    if mode == "greedy":
        used = np.zeros(groups.max() + 1, dtype=bool)
        value = scores.copy()
        picks = []
        for _ in range(budget):
            i = np.argmax(value + bonus * ~used[groups])
            picks.append(i)
            used[groups[i]] = True
            value[i] = -np.inf
        return pool[picks]
    if mode != "milp":
        raise ValueError(mode)
    coverage = (np.arange(groups.max() + 1)[:, None] == groups).astype(float)
    k, n = coverage.shape
    constraints = [
        LinearConstraint(np.r_[np.ones(n), np.zeros(k)], budget, budget),
        LinearConstraint(np.c_[-coverage, np.eye(k)], -np.inf, 0),
    ]
    result = milp(-np.r_[scores, np.full(k, bonus)], integrality=np.ones(n+k),
                  bounds=Bounds(0, 1), constraints=constraints)
    if not result.success:
        raise RuntimeError(result.message)
    picks = pool[result.x[:n] > .5]
    assert len(picks) == budget
    return picks
