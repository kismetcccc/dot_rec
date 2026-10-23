"""CPU only: tiny forward/math/autograd checks; no optimizer or training loop."""
from dataclasses import replace
import itertools
import math
import unittest
import torch
from utils.incidence import Incidence, safe_normalize
from model import BAST, ModelConfig
from model.modules.missing import MissingTransport
from model.modules.observed import ObservedTransport


def fixture(mask=None, config=None):
    torch.manual_seed(7)
    edges = torch.tensor([[0, 0], [1, 0], [0, 1], [0, 2], [1, 3]])
    visual = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [0., 1.], [1., 0.]])
    text = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [0., 1.], [1., 0.]])
    mask = torch.tensor([[False, True], [True, True], [True, True], [True, False], [False, False]]) if mask is None else mask
    model = BAST(3, 5, edges, visual, text, mask, config or ModelConfig(dim=2))
    with torch.no_grad():
        for projection in model.projections:
            projection.weight.copy_(torch.eye(2))
    return model


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_f1_f2_and_mask_a_b_a(self):
        model = fixture()
        _, _, a = model.encode(diagnostics=True)
        torch.testing.assert_close(a["c"][0, 0], torch.tensor([.25, 0.]))
        bmask = model.observed.clone()
        bmask[2, 1] = False
        _, _, b = model.encode(bmask, diagnostics=True)
        torch.testing.assert_close(b["c"][0, 0], torch.tensor([1/6, 0.]))
        _, _, again = model.encode(diagnostics=True)
        self.assertTrue(torch.equal(a["c"], again["c"]))
        # The image-only member is not a paired donor: changing it cannot affect M2.
        with torch.no_grad():
            model.visual[3] = torch.tensor([-20., 3.])
        _, _, changed = model.encode(diagnostics=True)
        self.assertTrue(torch.equal(a["c"], changed["c"]))

    def test_missing_opposite_orthogonal_and_zero_source_mean(self):
        model = fixture()
        for source in ([-1., 0.], [0., 1.], [0., 0.]):
            with torch.no_grad():
                model.text[0] = torch.tensor(source)
            _, _, details = model.encode(diagnostics=True)
            self.assertTrue(torch.equal(details["c"][0], torch.zeros(2, 2)))
        with torch.no_grad():
            model.text[0] = torch.tensor([1., 0.])
            model.text[2] = torch.tensor([-1., 0.])
        _, _, details = model.encode(diagnostics=True)
        self.assertTrue(torch.equal(details["c"][0], torch.zeros(2, 2)))

    def test_small_quality_is_not_renormalized(self):
        normal = fixture()
        weak = fixture(config=replace(normal.config, kappa=1e9))
        _, _, a = normal.encode(diagnostics=True)
        _, _, b = weak.encode(diagnostics=True)
        self.assertGreater(float(a["c"][0, 0].norm()), .2)
        self.assertLess(float(b["c"][0, 0].norm()), 2e-9)

    def test_quality_variance_and_ablation_hand_values(self):
        graph = Incidence(1, 3, torch.tensor([[0, 0], [0, 1], [0, 2]]))
        mask = torch.tensor([[False, True], [True, True], [True, True]])
        # Image donors e1/e2: mean=(.5,.5), variance=.5; text donors both e1.
        z = torch.tensor([[[0., 0.], [1., 0.]], [[1., 0.], [1., 0.]], [[0., 1.], [1., 0.]]])
        module = MissingTransport()
        expected_q = {"full": .5*math.exp(-.5), "no_support": math.exp(-.5),
                      "no_agreement": .5*math.exp(-.5), "unit": 1.}
        for mode, expected in expected_q.items():
            c, q = module(z, mask, graph, 2., 1., mode)
            self.assertAlmostEqual(q[0, 0].item(), expected, places=6)
            torch.testing.assert_close(c[0, 0], torch.tensor([.5, .5])*expected)
            self.assertTrue(((q >= 0) & (q <= 1)).all())
        z[0, 1] = torch.tensor([-1., 0.])
        self.assertEqual(module(z, mask, graph, 2., 1., "full")[1][0, 0].item(), 0.)
        self.assertGreater(module(z, mask, graph, 2., 1., "no_agreement")[1][0, 0].item(), 0.)

    def test_interleaved_missing_has_no_paired_donors(self):
        mask = torch.tensor([[True, False], [False, True], [True, False], [False, True], [True, False]])
        model = fixture(mask)
        _, _, d = model.encode(diagnostics=True)
        self.assertEqual(torch.count_nonzero(d["c"]).item(), 0)

    def test_full_missing_exact_same_model_cf_and_cold_rows(self):
        model = fixture(torch.zeros(5, 2, dtype=torch.bool))
        p, v, d = model.encode(diagnostics=True)
        self.assertTrue(torch.equal(v, d["behavior"]))
        self.assertEqual(torch.count_nonzero(d["h"] + d["c"]).item(), 0)
        self.assertEqual(torch.count_nonzero(v - d["behavior"]).item(), 0)
        self.assertTrue(torch.equal(p[2], torch.zeros(2)))
        self.assertTrue(torch.equal(v[4], torch.zeros(2)))
        cold = fixture(torch.ones(5, 2, dtype=torch.bool))
        _, cv, cd = cold.encode(diagnostics=True)
        self.assertTrue(torch.equal(cd["behavior"][4], torch.zeros(2)))
        torch.testing.assert_close(cv[4], cold.config.alpha * cd["z"][4].sum(0))

    def test_all_observed_compensation_switch_exactly_inactive(self):
        on = fixture(torch.ones(5, 2, dtype=torch.bool))
        off = fixture(torch.ones(5, 2, dtype=torch.bool), replace(on.config, compensation=False))
        self.assertTrue(torch.equal(on.encode()[1], off.encode()[1]))
        self.assertEqual(torch.count_nonzero(on.encode(diagnostics=True)[2]["c"]).item(), 0)

    def test_hidden_nan_and_arbitrary_features_are_isolated(self):
        model = fixture()
        before = model.encode()[1].detach().clone()
        with torch.no_grad():
            model.visual[~model.observed[:, 0]] = float("nan")
            model.text[~model.observed[:, 1]] = 1e30
        after = model.encode()[1]
        self.assertTrue(torch.equal(before, after))
        loss = model.calculate_loss((torch.tensor([0]), torch.tensor([1]), torch.tensor([3])))
        loss["total"].backward()  # finite-gradient inspection only, no updates
        for p in model.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())
        model = fixture()
        x = model.visual.clone()
        x[1] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            BAST(3, 5, torch.stack((model.graph.anchor, model.graph.item), 1), x, model.text, model.observed, model.config)

    def test_duplicate_train_is_binary_and_weighted_input_rejected(self):
        a = Incidence(2, 3, torch.tensor([[0, 0], [0, 1], [0, 0]]))
        self.assertEqual(len(a.item), 2)
        self.assertEqual(a.user_degree[0].item(), 2)
        with self.assertRaises(ValueError):
            Incidence(2, 3, torch.tensor([[0., 0., .5]]))

    def test_singleton_and_empty_support(self):
        for edges in (torch.tensor([[0, 0]]), torch.empty((0, 2), dtype=torch.long)):
            graph = Incidence(2, 2, edges)
            z = torch.ones(2, 2, 1)
            mask = torch.ones(2, 2, dtype=torch.bool)
            h, g = ObservedTransport()(z, mask, torch.zeros(2, 1), graph, .2)
            self.assertTrue(torch.equal(h, z))
            self.assertEqual(torch.count_nonzero(g).item(), 0)

    def test_zero_norm_and_full_missing_gradients_are_finite(self):
        for mask in (torch.zeros(5, 2, dtype=torch.bool), torch.ones(5, 2, dtype=torch.bool)):
            model = fixture(mask)
            with torch.no_grad():
                model.visual.zero_()
                model.text.zero_()
                model.user_ids.weight.zero_()
                model.item_ids.weight.zero_()
            loss = model.calculate_loss((torch.tensor([0]), torch.tensor([1]), torch.tensor([3])))
            loss["total"].backward()
            self.assertTrue(torch.isfinite(loss["total"]))
            for parameter in model.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_loo_hand_values_and_no_target_in_summary(self):
        graph = Incidence(1, 3, torch.tensor([[0, 0], [0, 1], [0, 2]]))
        z = torch.tensor([[[1., 0.]] * 2, [[0., 1.]] * 2, [[-1., 0.]] * 2])
        mask = torch.ones(3, 2, dtype=torch.bool)
        module = ObservedTransport()
        h, _ = module(z, mask, torch.zeros(3, 2), graph, 1, learned=False)
        expected = torch.tensor([[[-.5, .5]] * 2, [[0., 0.]] * 2, [[.5, .5]] * 2])
        torch.testing.assert_close(h - z, expected)
        changed = z.clone()
        changed[0] = torch.tensor([0., -1.])
        h2, _ = module(changed, mask, torch.zeros(3, 2), graph, 1, learned=False)
        # Only fixed-gate incoming LOO message is invariant; whole score is not.
        torch.testing.assert_close((h2 - changed)[0], (h - z)[0])

    def test_learned_sparse_matches_independent_scalar_reference(self):
        torch.manual_seed(12)
        graph = Incidence(3, 4, torch.tensor([[0, 0], [0, 1], [0, 2], [1, 0], [1, 3], [2, 2]]))
        z = safe_normalize(torch.randn(4, 2, 3))
        mask = torch.tensor([[True, True], [True, False], [False, True], [True, True]])
        z = z * mask[:, :, None]
        behavior = torch.randn(4, 3)
        module = ObservedTransport()
        with torch.no_grad():
            module.theta.copy_(torch.tensor([[.1, -.2, .3, -.4, .5], [-.2, .1, -.3, .4, -.5]]))
        result, gates = module(z, mask, behavior, graph, .2)
        membership = list(zip(graph.item.tolist(), graph.anchor.tolist()))
        unit_b = safe_normalize(behavior)
        refg = torch.zeros_like(gates)
        cosine = lambda x, y: 0. if x.norm() == 0 or y.norm() == 0 else float(torch.dot(x, y)/(x.norm()*y.norm()))
        for row, (i, e) in enumerate(membership):
            group = [j for j, anchor in membership if anchor == e and j != i]
            for m in range(2):
                donors = [j for j in group if mask[j, m]]
                if not mask[i, m] or not donors:
                    continue
                mu = z[donors, m].mean(0)
                var = max(0., float(z[donors, m].square().sum(1).mean() - mu.square().sum()))
                mub = unit_b[group].mean(0)
                th = module.theta[m].detach().tolist()
                sp = lambda x: math.log1p(math.exp(x))
                logit = th[0] + sp(th[1])*cosine(unit_b[i], mub) + sp(th[2])*cosine(z[i, m], mu) - sp(th[3])*var + sp(th[4])*math.log1p(len(donors))
                refg[row, m] = 1/(1+math.exp(-logit))
        reference = z.clone()
        for row, (i, e) in enumerate(membership):
            for m in range(2):
                donorrows = [r for r,(j,a) in enumerate(membership) if a == e and j != i and mask[j,m]]
                if mask[i,m] and donorrows:
                    message = sum(refg[r,m] * z[membership[r][0],m] for r in donorrows) / len(donorrows)
                    reference[i,m] += .2 * refg[row,m] * message / graph.item_degree[i]
        torch.testing.assert_close(gates, refg, atol=2e-7, rtol=1e-6)
        torch.testing.assert_close(result, reference, atol=2e-7, rtol=1e-6)

    def test_bounds_for_all_visibility_patterns_and_ablations(self):
        for learned, compensation in itertools.product((False, True), repeat=2):
            cfg = ModelConfig(dim=2, learned_group=learned, compensation=compensation)
            for pattern in ((True, True), (True, False), (False, True), (False, False)):
                mask = torch.ones(5, 2, dtype=torch.bool)
                mask[0] = torch.tensor(pattern)
                model = fixture(mask, cfg)
                _, v, d = model.encode(diagnostics=True)
                t = (d["h"] - d["z"]) / cfg.eta
                self.assertTrue((t.norm(dim=-1) <= 1 + 1e-6).all())
                self.assertTrue((d["c"].norm(dim=-1) <= 1 + 1e-6).all())
                bound = .48 if sum(pattern) == 2 else .34 if sum(pattern) == 1 else 0.
                self.assertLessEqual(float((v[0] - d["behavior"][0]).norm()), bound + 1e-6)
                if not any(pattern):
                    self.assertTrue(torch.equal(v[0], d["behavior"][0]))

    def test_2x2_parameter_layout_and_other_defaults_identical(self):
        expected = None
        for learned, compensation in itertools.product((False, True), repeat=2):
            cfg = ModelConfig(dim=2, learned_group=learned, compensation=compensation)
            model = fixture(config=cfg)
            signature = [(n, tuple(p.shape)) for n, p in model.named_parameters()]
            if expected is None:
                expected = signature
            self.assertEqual(signature, expected)
            self.assertEqual(replace(cfg, learned_group=True, compensation=True), ModelConfig(dim=2))

    def test_scores_and_eval_determinism(self):
        model = fixture().eval()
        users, items = torch.tensor([0, 1]), torch.tensor([2, 3])
        scores = model.score(users, items)
        all_scores = model.full_sort_predict(users)
        torch.testing.assert_close(scores, all_scores[torch.arange(2), items])
        self.assertTrue(torch.equal(all_scores, model.full_sort_predict(users)))


if __name__ == "__main__":
    unittest.main()
