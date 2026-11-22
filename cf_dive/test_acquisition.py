import itertools
import unittest

import numpy as np
import torch
from torch import nn

from .acquire import select
from .experiment import flips


class AcquisitionTests(unittest.TestCase):
    def test_coverage_solvers_match_exhaustive_objective(self):
        score = np.array([.90, .89, .88, .72, .71, .70])
        group = np.array([0, 0, 0, 1, 2, 2])
        pool = np.arange(len(score))

        def objective(nodes):
            return score[nodes].sum() + .25 * len(set(group[nodes]))

        best = max(objective(list(comb)) for comb in itertools.combinations(pool, 3))
        for mode in ("milp", "greedy"):
            self.assertAlmostEqual(objective(select(pool, score, group, 3, mode, .25)), best)

    def test_independent_edges_match_full_graph(self):
        class NeighborRule(nn.Module):
            def forward(self, own, neighbor_mean):
                return torch.cat([neighbor_mean, -neighbor_mean], dim=1)

        model = NeighborRule()
        x = torch.tensor([[2.], [-1.], [4.], [-3.]])
        adj = torch.tensor([[0., 1., 0., 0.], [1., 0., 1., 0.],
                            [0., 1., 0., 1.], [0., 0., 1., 0.]])
        deg = adj.sum(1, keepdim=True)
        view = (x, adj @ x, adj @ x / deg)
        prob = model(x, view[2]).softmax(1)
        proposals = torch.tensor([[2, 3], [-1, -1], [-1, -1], [-1, -1]])
        cf, stats = flips([0, 1], proposals, deg, [view], [model], [prob])
        target, spill = [], []
        for dst in (2, 3):
            edited = adj.clone()
            edited[0, dst] = edited[dst, 0] = 1
            changed = model(x, edited @ x / edited.sum(1, keepdim=True)).argmax(1) != prob.argmax(1)
            target.append(changed[0].item())
            spill.append(changed[dst].item())
            self.assertFalse(any(changed[i] for i in range(len(x)) if i not in (0, dst)))
        np.testing.assert_equal(cf, [max(target), 0])
        self.assertEqual(stats["target_flip"], np.mean(target))
        self.assertEqual(stats["other_endpoint_flip"], np.mean(spill))
