import unittest

import torch

from scripts.research_twinmask_v6 import Config, LOCKED, combine_gradients, locked_protocol
from scripts.research_twinmask_v5 import IsolatedTwin, as_log_probs, subset_predictions
from scripts.test_research_twinmask_v5 import fixture
from twinmask.audit_protocol import keys


class TuningProtocolTests(unittest.TestCase):
    def test_locked_targets_are_disjoint_from_search_and_each_other(self):
        n = 40
        graph = torch.combinations(torch.arange(n), r=2).T
        base, fit, dev, hidden, nfit, ndev, ntest = graph[:, :140].split([32, 12, 12, 24, 12, 12, 36], 1)
        ntest = ntest[:, :hidden.shape[1]]
        protocol = (dict(tokens=torch.ones(n, 17)), base, (fit, dev, hidden), (nfit, ndev, ntest), {})
        folds = [locked_protocol(protocol, split) for split in LOCKED]
        for row in folds:
            _, observed, positive, negative, _, (fit_nodes, dev_nodes, test_nodes) = row
            self.assertFalse(keys(positive[-1], n) & keys(torch.cat((base, fit, dev, observed), 1), n))
            self.assertFalse(keys(negative[-1], n) & keys(torch.cat((nfit, ndev), 1), n))
            self.assertFalse(torch.isin(test_nodes, torch.cat((fit_nodes, dev_nodes))).any())
        self.assertFalse(keys(folds[0][2][-1], n) & keys(folds[1][2][-1], n))
        self.assertFalse(torch.isin(folds[0][-1][-1], folds[1][-1][-1]).any())

    def test_gradient_projection(self):
        x, y = torch.tensor([1., 0.]), torch.tensor([-1., 1.])
        torch.testing.assert_close(combine_gradients(x, y, "pcgrad"), torch.tensor([.5, 1.5]))
        torch.testing.assert_close(combine_gradients(x, x, "pcgrad"), 2*x)
        self.assertTrue(torch.isfinite(combine_gradients(x, -x, "cagrad")).all())

    @torch.no_grad()
    def test_learned_fusion_has_full_subset_equivalence(self):
        case, _ = fixture()
        model = IsolatedTwin(10, text_encoder="transformer", structural_stats=True).eval()
        model.gate_logits = torch.nn.Parameter(torch.tensor([-.3, .7]))
        joint = model.predict(case)
        for direction, scores_, n in zip(("text", "edge"), joint, (8, 2)):
            scores, *_ = subset_predictions(model, case, direction, torch.arange(n))
            torch.testing.assert_close(scores[-1], as_log_probs(scores_[0], direction))
        self.assertEqual(Config().vocab_limit, 1024)


if __name__ == "__main__":
    unittest.main()
