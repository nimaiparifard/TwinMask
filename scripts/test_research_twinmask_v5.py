import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.research_twinmask import Case, SparseTwinMask
from scripts.research_twinmask_v3 import MessagePanel
from scripts.research_twinmask_v4 import actual_clocks
from scripts.research_twinmask_v41 import audited_contexts
from scripts.research_twinmask_v5 import (IsolatedTwin, as_log_probs, expected_utility, fusion_panel,
                                          subset_predictions, teachers, temporal)
from twinmask.audit_protocol import graph_split, keys
from twinmask.data import DIVERGE_DIR
from twinmask.set_utility import SetUtility, choose, enumerate_subsets, shapley_values


def fixture():
    ids = torch.tensor([[3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]])
    observed = torch.tensor([[0, 0, 1], [1, 3, 2]])
    pairs = torch.tensor([[0, 1], [2, 3]])
    mask = torch.tensor([[False, True, True]]*4)
    hidden = torch.ones(2, dtype=torch.bool)
    labels = torch.tensor([0, 1])
    changed = ids.masked_fill(mask, 9)
    return (Case(ids, ids, mask, pairs, labels, hidden, observed, (.75, .5)),
            Case(changed, changed, mask, pairs, 1-labels, hidden, observed, (.75, .5)))


class ProtocolTests(unittest.TestCase):
    @unittest.skipUnless((Path(DIVERGE_DIR)/"cora.pt").exists(), "Raw reference is required")
    def test_all_negative_edges_are_valid_and_split_disjointness(self):
        for name in ("cora", "citeseer", "pubmed"):
            data, base, positives, negatives, audit = graph_split(name, 2101)
            n = len(data["tokens"])
            old = keys(torch.cat((data["val_pos"], data["test_pos"]), 1), n)
            self.assertFalse(old & keys(positives[-1], n))
            self.assertTrue(all(not keys(base, n) & keys(pairs, n) for pairs in positives))
            self.assertGreaterEqual(audit["reference_edges"], base.shape[1])
            self.assertEqual(audit["split_hash"], graph_split(name, 2101)[-1]["split_hash"])
            self.assertNotEqual(audit["split_hash"], graph_split(name, 2113)[-1]["split_hash"])

    @unittest.skipUnless((Path(DIVERGE_DIR)/"cora.pt").exists() and
                         Path("checkpoints/research_twinmask_v2/cora_42.pt").exists(),
                         "Raw reference and V2 checkpoint are required")
    def test_matched_context(self):
        audit = {}
        matched = audited_contexts("cora", 42, torch.device("cpu"), True, audit)
        growing = audited_contexts("cora", 42, torch.device("cpu"), False, {})
        for (_, fit, dev, test, _), (_, other_fit, other_dev, other_test, _) in zip(matched, growing):
            self.assertTrue(torch.equal(fit.base, dev.base) and torch.equal(dev.base, test.base))
            self.assertGreater(other_test.base.shape[1], test.base.shape[1])
            for original, comparing in zip((fit, dev, test), (other_fit, other_dev, other_test)):
                for field in ("pairs", "mask", "edge_mask", "labels"):
                    torch.testing.assert_close(getattr(original, field), getattr(comparing, field))

    def test_actual_diffusion_clock(self):
        x, a = torch.ones(1000, dtype=torch.bool), torch.ones(800, dtype=torch.bool)
        tx, ta = actual_clocks((.75, .9), (x, a), (x[96:], a[50:]))
        self.assertAlmostEqual(tx, .678)
        self.assertAlmostEqual(ta, .84375)
        self.assertEqual(actual_clocks((.8, .6), (x[:0], a[:0]), (x[:0], a[:0])), (0., 0.))
        case, _ = fixture()
        traces = temporal(IsolatedTwin(10).eval(), case, 97, steps=4)
        for frozen, free, teacher in zip(traces["static"], traces["free"], traces["teacher"]):
            self.assertEqual(frozen["clocks"], free["clocks"])
            self.assertEqual(free["clocks"], teacher["clocks"])
            self.assertEqual(frozen["counts"], free["counts"])

    def test_expected_utility_formula(self):
        probabilities = torch.tensor([[.3, .7], [.2, .8], [.4, .6]])
        q = torch.tensor([.3, .7])
        actual = expected_utility(probabilities.log()[:, None], q, 0).flatten()
        expected = (q*probabilities[0].log()).neg().sum() + (q*probabilities.log()).sum(1)
        torch.testing.assert_close(actual, expected)
        p = probabilities[0]
        self.assertTrue((expected_utility(probabilities.log()[:, None], p, 0) <= 1e-7).all())

    @torch.no_grad()
    def test_hidden_target_invariance_and_teacher_no_leakage(self):
        original, edited = fixture()
        models = {k: IsolatedTwin(10, architecture=k).eval() for k in ("mlm", "text_link", "structure", "gae")}
        model = IsolatedTwin(10).eval()
        for d, size in (("text", 8), ("edge", 2)):
            index = torch.arange(size)
            for a, b in zip(teachers(models, original, d, index).values(), teachers(models, edited, d, index).values()):
                torch.testing.assert_close(a, b)
            x, y = [fusion_panel(model, case, d, index) for case in (original, edited)]
            torch.testing.assert_close(x[-1], y[-1])
            teacher = torch.full((size, 10 if d == "text" else 2), .1 if d == "text" else .5)
            selections = []
            for case in (original, edited):
                probability, masks, allowed, *_ = subset_predictions(model, case, d, index)
                gain = expected_utility(probability, teacher, len(probability)-1)
                selections.append(choose(gain, masks, allowed, 2))
            torch.testing.assert_close(*selections)

    @torch.no_grad()
    def test_full_panel_equivalence_and_encoder_isolation(self):
        original, _ = fixture()
        model = IsolatedTwin(10).eval()
        joint = model.predict(original)
        for d, logits in zip(("text", "edge"), joint):
            indices = torch.arange(8 if d == "text" else 2)
            p, *_ = subset_predictions(model, original, d, indices)
            torch.testing.assert_close(p[-1], as_log_probs(logits[0], d))
            torch.testing.assert_close(p[0], as_log_probs(logits[1], d))
        context, a, _ = original.initial()
        streams = model.streams(context, original.pairs, a, original.base, original.rates)
        changed = model.streams(context.masked_fill(context != 1, 8), original.pairs, a, original.base, original.rates)
        torch.testing.assert_close(streams[2], changed[2])
        empty = model.streams(context, original.pairs, a, original.base[:, :0], original.rates)
        torch.testing.assert_close(streams[0], empty[0])

    @torch.no_grad()
    def test_local_intervention_equivalence(self):
        case, _ = fixture()
        model = SparseTwinMask(10, 3, width=8).eval()
        generator = torch.Generator().manual_seed(12)
        panel = MessagePanel(model, case, "text", 32, generator, layer="both")
        ids, states, _ = case.initial()
        _, semantic, _, _, mass, adj = model.encode_messages(ids, case.pairs, states, case.base, panel.query,
                                                              *case.rates)
        h0 = semantic+model.clock_x(semantic.new_tensor(case.rates[0]))+model.clock_a(semantic.new_tensor(case.rates[1]))
        source, dest = adj
        h = h0.clone()
        graph = torch.ones(adj.shape[1], dtype=torch.bool)
        for sender in panel.slots[0][panel.valid[0]]:
            graph &= ~((source == sender) & (dest == panel.targets[0, 0]))
        for layer in model.mix:
            total = torch.zeros_like(h).index_add_(0, dest[graph], h[source[graph]])
            h = h+F.gelu(layer.lin_l(total/mass.clamp_min(1)[:, None])+layer.lin_r(h))
        result = model.text(model.norm(panel.selected[0]+h[panel.targets[0, 0]]))
        torch.testing.assert_close(result, panel.predict(torch.zeros(1, 4), slice(0, 1))[0])

    def test_shapley_efficiency_and_budget_constraints(self):
        valid = torch.tensor([[True, False, True, True], [False]*4])
        masks, allowed = enumerate_subsets(valid)
        loss = -(masks.sum(-1)**2 + masks[..., 0])
        torch.testing.assert_close(shapley_values(loss, 4).sum(1), loss[0]-loss[-1])
        decision = choose(-loss, masks, allowed, 2)
        self.assertTrue((masks[decision, torch.arange(len(valid))].sum(1) <= 2).all())
        network = SetUtility(2, anchored=True)
        mean, spread = network(torch.randn(2, 4, 2), masks, valid)
        self.assertTrue((mean[-1] == 0).all() and (spread[-1] == 0).all())

    @torch.no_grad()
    def test_reproducibility(self):
        case, _ = fixture()
        values = []
        for _ in range(2):
            torch.manual_seed(42)
            model = IsolatedTwin(10, architecture="gat").eval()
            values.append(model.predict(case)[0][0])
        torch.testing.assert_close(*values, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
