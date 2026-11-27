import unittest

import torch

from scripts.research_twinmask import Case, SparseTwinMask
from scripts.research_twinmask_v3 import MessagePanel, budget_choices, subset_weights


class SelectiveTransferTests(unittest.TestCase):
    @torch.no_grad()
    def test_full_weights_match_frozen_model_and_budget(self):
        ids = torch.tensor([[3, 4], [4, 5], [5, 6], [6, 7]])
        mask = torch.tensor([[False, True]]*len(ids))
        pairs = torch.tensor([[0, 1], [2, 3]])
        base = torch.tensor([[0, 1, 2, 0], [1, 2, 3, 3]])
        labels = torch.tensor([0, 1])
        hidden = torch.ones_like(labels, dtype=torch.bool)
        model = SparseTwinMask(9, 2, width=8).eval()
        case = Case(ids, ids, mask, pairs, labels, hidden, base)
        logits = case.predict(model, case.initial())
        for direction, expected in zip(("text", "edge"), logits):
            rng = torch.Generator().manual_seed(7)
            panel = MessagePanel(model, case, direction, 16, rng)
            panel_order = torch.Generator().manual_seed(7)
            x = torch.randperm(len(case.x_truth), generator=panel_order)
            a = torch.randperm(len(case.a_truth), generator=panel_order)
            subsets = subset_weights(panel)
            torch.testing.assert_close(panel.predict(subsets[-1], slice(None)),
                                       expected[x if direction == "text" else a])
            utility = torch.zeros_like(panel.valid, dtype=torch.float)
            synergy = torch.zeros(len(panel.truth), 6)
            counts, choices = budget_choices(panel, subsets, utility, synergy, utility, rng)
            for budget, arms in choices.items():
                for selection in arms.values():
                    torch.testing.assert_close(counts[selection, torch.arange(len(panel.truth))],
                                               panel.valid.sum(1).clamp(max=budget).float())


if __name__ == "__main__":
    unittest.main()
