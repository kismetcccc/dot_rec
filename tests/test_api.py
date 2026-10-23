"""Public API rejection tests; CPU only, no optimization or training."""
import unittest
import numpy as np
import torch
from utils.incidence import Incidence, safe_normalize
from model import ModelConfig
from utils.config import TrainConfig
from utils.masks import fixed_mask, dropout_view
from test_model import fixture


class ApiTests(unittest.TestCase):
    def test_score_rejects_invalid_ids_and_broadcast_with_or_without_cache(self):
        model = fixture()
        for encoded in (None, model.encode()):
            for bad in (torch.tensor([-1]), torch.tensor([3]), torch.tensor([0.]),
                        torch.tensor([True]), torch.tensor([[0]]), torch.tensor(0)):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    model.score(bad, torch.tensor([1]), encoded)
                with self.subTest(fullsort=bad), self.assertRaises(ValueError):
                    model.full_sort_predict(bad, encoded)
            for bad in (torch.tensor([-1]), torch.tensor([5]), torch.tensor([1.]),
                        torch.tensor([True]), torch.tensor([[1]]), torch.tensor(1)):
                with self.subTest(item=bad), self.assertRaises(ValueError):
                    model.score(torch.tensor([0]), bad, encoded)
            with self.assertRaisesRegex(ValueError, "broadcasting"):
                model.score(torch.tensor([0, 1]), torch.tensor([2]), encoded)

    def test_loss_rejects_empty_malformed_and_broadcast_batches(self):
        model = fixture()
        valid = (torch.tensor([0]), torch.tensor([1]), torch.tensor([3]))
        for slot, upper in enumerate((3, 5, 5)):
            for bad in (torch.tensor([-1]), torch.tensor([upper]), torch.tensor([0.]),
                        torch.tensor([False]), torch.tensor([[0]]), torch.tensor(0), torch.tensor([0, 1])):
                batch = list(valid)
                batch[slot] = bad
                with self.subTest(slot=slot, bad=bad), self.assertRaises(ValueError):
                    model.calculate_loss(batch)
        empty = torch.empty(0, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            model.calculate_loss((empty, empty, empty))
        for malformed in (None, (), (valid[0], valid[1]), torch.tensor([[0], [1], [3]])):
            with self.assertRaises(ValueError):
                model.calculate_loss(malformed)
        for weight in (True, False, float("nan"), float("inf"), "0.1"):
            with self.assertRaises(ValueError):
                model.calculate_loss(valid, l2_weight=weight)

    def test_valid_integer_ids_and_explicit_empty_prediction(self):
        model = fixture()
        users = np.array([0, 1], dtype=np.int32)
        items = np.array([1, 3], dtype=np.int64)
        paired = model.score(users, items)
        full = model.full_sort_predict(users)
        torch.testing.assert_close(paired, full[torch.arange(2), torch.tensor(items)])
        empty = torch.empty(0, dtype=torch.int64)
        self.assertEqual(model.score(empty, empty).shape, (0,))
        self.assertEqual(model.full_sort_predict(empty).shape, (0, 5))

    def test_full_precision_only_including_zero_half_and_amp(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.assertRaisesRegex(ValueError, "float32/float64"):
                safe_normalize(torch.zeros(2, 2, dtype=dtype))
            with self.assertRaisesRegex(ValueError, "float32/float64"):
                fixture().to(dtype=dtype).encode()
        for dtype in (torch.float32, torch.float64):
            zero = torch.zeros(2, 2, dtype=dtype, requires_grad=True)
            output = safe_normalize(zero)
            self.assertTrue(torch.equal(output, torch.zeros_like(output)))
            output.sum().backward()
            self.assertTrue(torch.isfinite(zero.grad).all())
            model = fixture().to(dtype=dtype)
            self.assertTrue(torch.isfinite(model.encode()[1]).all())
        model = fixture()
        cached = model.encode()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with self.assertRaisesRegex(ValueError, "AMP"):
                model.encode()
            with self.assertRaisesRegex(ValueError, "AMP"):
                model.score([0], [1], cached)

    def test_cpu_construction_view_conversion_and_cached_contract(self):
        old = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            model = fixture()
        finally:
            torch.set_default_dtype(old)
        for tensor in list(model.parameters()) + list(model.buffers()):
            self.assertEqual(tensor.device.type, "cpu")
            if tensor.is_floating_point():
                self.assertEqual(tensor.dtype, torch.float32)
        a = model.encode(model.observed.numpy())[1]
        self.assertTrue(torch.equal(a, model.encode()[1]))
        with self.assertRaises(ValueError):
            model.encode(np.ones((5, 2), dtype=np.int64))
        with self.assertRaises(ValueError):
            model.encode(torch.ones(5, 2, dtype=torch.bool))
        p, v = model.encode()
        for invalid in ((p[:1], v), (p.double(), v.double()), (p.to("meta"), v), (p,)):
            with self.assertRaises(ValueError):
                model.score([0], [1], invalid)

    def test_numeric_configs_reject_bool_nonfinite_and_wrong_types(self):
        for name in ("eta", "alpha", "beta", "kappa", "tau"):
            for value in (True, False, float("nan"), float("inf"), -float("inf"), "0.2", None):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    ModelConfig(**{name: value})
        for name in ("learning_rate", "l2_weight", "dropout"):
            for value in (True, False, float("nan"), float("inf"), "0.2", None):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    TrainConfig(**{name: value})
        for counts in ((True, 5), (3, False), (3., 5), (3, 5.), (0, 5), (3, -1)):
            with self.assertRaises(ValueError):
                Incidence(*counts, torch.tensor([[0, 0]]))
        natural = np.ones((2, 2), dtype=bool)
        for value in (True, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                fixed_mask(natural, value, 1)
            with self.assertRaises(ValueError):
                dropout_view(torch.from_numpy(natural), value, torch.Generator())
        with self.assertRaises(ValueError):
            fixed_mask(natural, .5, True)


if __name__ == "__main__":
    unittest.main()
