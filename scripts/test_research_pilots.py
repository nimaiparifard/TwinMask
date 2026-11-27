import unittest

import torch

from scripts.research_ensemble import SAGE, generate, noise_view, schedule
from scripts.research_twinmask import Case, SparseTwinMask, corruption, reconstruct, token_loss


class ResearchPilotTests(unittest.TestCase):
    def test_diffusion_endpoints_and_seed(self):
        alpha = schedule(torch.device("cpu"))
        self.assertEqual(float(alpha[0]), 1.)
        self.assertTrue((alpha.diff() < 0).all())
        x = torch.randn(4, 8)
        self.assertTrue(torch.equal(noise_view(x, alpha[0], torch.randn_like(x)), x))
        model = SAGE(8, 8, conditional=True).eval()
        pairs = torch.tensor([[0, 1, 2], [1, 2, 3]])
        first, _ = generate(model, x, pairs, alpha, 30, 4, .1, 42)
        second, _ = generate(model, x, pairs, alpha, 30, 4, .1, 42)
        self.assertTrue(torch.equal(first, second))

    def test_discrete_mask_endpoints(self):
        eligible = torch.tensor([True, False, True])
        rng = torch.Generator().manual_seed(3)
        self.assertFalse(corruption(eligible, 0., rng).any())
        self.assertTrue(torch.equal(corruption(eligible, 1., rng), eligible))
        self.assertEqual(float(token_loss(torch.empty(0, 2), torch.empty(0, dtype=torch.long))), 0.)

    def test_pair_symmetry_and_hidden_label_invariance(self):
        model = SparseTwinMask(9, 3, width=8).eval()
        ids = torch.tensor([[3, 4, 1], [4, 5, 1], [5, 6, 1], [6, 7, 1]])
        mask = torch.tensor([[False, True, False]]*len(ids))
        pairs = torch.tensor([[0, 1, 0], [1, 2, 3]])
        labels = torch.tensor([1, 0, 1])
        hidden = torch.tensor([True, True, False])
        base = torch.tensor([[2], [3]])
        first = Case(ids, ids, mask, pairs, labels, hidden, base)
        changed = ids.masked_fill(mask, 8)
        second = Case(changed, changed, mask, pairs.flip(0), torch.tensor([0, 1, 1]), hidden, base)
        for a, b in zip(first.predict(model, first.initial()), second.predict(model, second.initial())):
            torch.testing.assert_close(a, b)
        result = reconstruct(model, first, "correct", 4, None)
        self.assertEqual(result["calls"], 4)
        self.assertEqual(len(result["trace"]), 4)
        self.assertEqual(result["trace"][-1]["action"], "joint")


if __name__ == "__main__":
    unittest.main()
