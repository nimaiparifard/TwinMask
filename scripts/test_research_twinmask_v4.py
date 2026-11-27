import unittest

import torch

from scripts.research_twinmask import Case, SparseTwinMask
from scripts.research_twinmask_v3 import MessagePanel
from scripts.research_twinmask_v4 import deployment, features_for
from twinmask.set_utility import SetUtility, choose, enumerate_subsets, shapley_values, subset_objective


class SetUtilityTests(unittest.TestCase):
    def test_unique_subsets_and_budget(self):
        valid = torch.tensor([[True, False, True, False], [False]*4])
        masks, allowed = enumerate_subsets(valid)
        torch.testing.assert_close(allowed.sum(0), torch.tensor([4, 1]))
        scores = -masks.sum(-1)
        torch.testing.assert_close(choose(scores, masks, allowed, 2), torch.zeros(2, dtype=torch.long))
        torch.testing.assert_close(choose(scores, masks, allowed, 2, exact=True), torch.tensor([5, 0]))
        shapley = shapley_values(-masks.sum(-1), 4)
        torch.testing.assert_close(shapley, valid.float())

    def test_set_permutation_invariance_and_decision_gradient(self):
        model = SetUtility(5)
        features = torch.randn(3, 4, 5)
        valid = torch.ones(3, 4, dtype=torch.bool)
        masks, allowed = enumerate_subsets(valid)
        mean, spread = model(features, masks, valid)
        perm = torch.tensor([2, 0, 3, 1])
        new_mean, _ = model(features[:, perm], masks[:, :, perm], valid[:, perm])
        torch.testing.assert_close(mean, new_mean)
        target = torch.randn_like(mean)
        loss = subset_objective(mean, spread, target, masks, allowed, .5, True)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    @torch.no_grad()
    def test_heldout_labels_cannot_change_selected_messages(self):
        ids = torch.tensor([[3, 4, 6], [4, 5, 6], [5, 6, 7], [6, 7, 8]])
        mask = torch.tensor([[False, True, True]]*4)
        pairs = torch.tensor([[0, 1], [2, 3]])
        base = torch.tensor([[0, 1, 2, 0], [1, 2, 3, 3]])
        labels = torch.tensor([0, 1])
        hidden = torch.ones_like(labels, dtype=torch.bool)
        model = SparseTwinMask(10, 3, width=8).eval()
        changed = ids.masked_fill(mask, 9)
        original = Case(ids, ids, mask, pairs, labels, hidden, base, (.5, .75))
        modified = Case(changed, changed, mask, pairs, 1-labels, hidden, base, (.5, .75))
        for direction in ("text", "edge"):
            a, b = (MessagePanel(model, case, direction, 32, torch.Generator().manual_seed(42), 4, "both")
                    for case in (original, modified))
            torch.testing.assert_close(features_for(a)["aware"], features_for(b)["aware"])
            selector = SetUtility(features_for(a)["aware"].shape[-1]).eval()
            pa, ma = deployment(a, selector, True)
            pb, mb = deployment(b, selector, True)
            torch.testing.assert_close(ma, mb)
            torch.testing.assert_close(pa, pb)
            baseline = MessagePanel(model, original, direction, 32, torch.Generator().manual_seed(42))
            torch.testing.assert_close(a.predict(a.valid.float(), slice(None)),
                                       baseline.predict(baseline.valid.float(), slice(None)))

    @torch.no_grad()
    def test_swapping_edge_endpoints_swaps_slots(self):
        ids = torch.tensor([[3, 4], [4, 5], [5, 6], [6, 7], [7, 8]])
        pairs = torch.tensor([[0, 1], [2, 3]])
        base = torch.tensor([[0, 1, 2, 0, 1], [1, 2, 3, 4, 4]])
        labels = torch.tensor([1, 0])
        model = SparseTwinMask(10, 2, width=8).eval()
        panels = []
        for candidate in (pairs, pairs.flip(0)):
            case = Case(ids, ids, torch.ones_like(ids, dtype=torch.bool), candidate, labels,
                        torch.ones_like(labels, dtype=torch.bool), base)
            panels.append(MessagePanel(model, case, "edge", 32, torch.Generator().manual_seed(7)))
        original, flipped = panels
        permutation = [2, 3, 0, 1]
        torch.testing.assert_close(features_for(original)["aware"][:, permutation], features_for(flipped)["aware"])
        weights, _ = enumerate_subsets(original.valid)
        torch.testing.assert_close(original.predict(weights, slice(None)),
                                   flipped.predict(weights[..., permutation], slice(None)))


if __name__ == "__main__":
    unittest.main()
