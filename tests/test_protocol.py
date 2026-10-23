import contextlib
import io
import json
from pathlib import Path
import shutil
import uuid
import unittest
from unittest.mock import patch
import numpy as np
import torch
from common.dataset import load_manifest, validate_splits, load_mmrec_inter, load_edges
from utils.masks import fixed_mask, dropout_view
from common.dataset import sha256_file
from common.evaluator import recommend, ranking_metrics, evaluate
from model import ModelConfig
from model import BAST
from utils.config import load_config
from utils.sampling import TrainNegativeSampler
from common.trainer import main
from utils.reproducibility import configure_determinism, source_fingerprint
from test_model import fixture


@contextlib.contextmanager
def fixture_directory():
    # Use a workspace-local default-permission folder, not Windows tempfile's
    # owner-only ACL, which is inaccessible in some managed execution sandboxes.
    base = (Path(__file__).resolve().parents[1] / ".cache" / "unit-tests").resolve()
    path = base / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield str(path)
    finally:
        resolved = path.resolve()
        if resolved.parent != base:
            raise RuntimeError("Refusing cleanup outside the test scratch directory")
        shutil.rmtree(resolved)


class ProtocolTests(unittest.TestCase):
    def test_ranked_duplicates_types_padding_and_direct_edges(self):
        for ranked, targets in (([[3, 3]], {3}), ([[3.9]], {3}), ([[-1]], {-1})):
            with self.assertRaises(ValueError):
                ranking_metrics(ranked, [0], {0: targets}, (2,))
        self.assertEqual(ranking_metrics([[3, -1, -1]], [0], {0: {3}}, (3,))["recall@3"], 1.)
        model = fixture()
        valid = np.array([[0, 3]])
        for edges in (np.array([[-1, 0]]), np.array([[0, -1]]), np.array([[0., 3.]]),
                      np.array([0, 3]), np.array([[3, 0]]), np.array([[0, 5]])):
            with self.assertRaises(ValueError):
                evaluate(model, edges, np.array([[0, 0]]))
            with self.assertRaises(ValueError):
                evaluate(model, valid, edges)

    def test_ranking_public_input_validation(self):
        model = fixture()
        encoded = model.encode()
        targets = np.array([[0, 3]])
        train = np.array([[0, 0]])
        for ks in ((1, 1), (), (0,), (-1,), (True,), (1.0,)):
            with self.subTest(ks=ks):
                with self.assertRaises(ValueError):
                    ranking_metrics(np.array([[3]]), [0], {0: {3}}, ks)
                with self.assertRaises(ValueError):
                    evaluate(model, targets, train, ks=ks)
        for value in (0, -1, True, 1.0):
            for field in ("user_batch", "item_chunk"):
                with self.assertRaises(ValueError):
                    evaluate(model, targets, train, **{field: value})
            with self.assertRaises(ValueError):
                recommend(encoded, [0], {}, value)
            with self.assertRaises(ValueError):
                recommend(encoded, [0], {}, 1, item_chunk=value)
        for users in ([-1], [0.5], [0.0], [True], [[0]], [len(encoded[0])]):
            with self.assertRaises(ValueError):
                recommend(encoded, users, {}, 1)
        for ranked, users in (([[3]], [0, 1]), ([[3], [2]], [0]), ([3], [0])):
            with self.assertRaises(ValueError):
                ranking_metrics(ranked, users, {0: {3}}, (1,))
        self.assertEqual(ranking_metrics([[3]], [0], {0: {3}}, (1,)),
                         {"recall@1": 1., "ndcg@1": 1.})

    def test_source_fingerprint_tracks_only_allowed_contents(self):
        with fixture_directory() as tmp:
            root = Path(tmp)
            (root / "model").mkdir()
            (root / "configs").mkdir()
            source = root / "model" / "model.py"
            source.write_text("value = 1", encoding="utf-8")
            (root / "configs" / "test.json").write_text("{}", encoding="utf-8")
            (root / "model" / ".env").write_text("synthetic excluded fixture", encoding="utf-8")
            a = source_fingerprint(root)
            self.assertEqual(set(a["files"]), {"model/model.py", "configs/test.json"})
            (root / "model" / ".env").write_text("changed synthetic fixture", encoding="utf-8")
            self.assertEqual(a, source_fingerprint(root))
            source.write_text("value = 2", encoding="utf-8")
            b = source_fingerprint(root)
            self.assertNotEqual(a["digest"], b["digest"])
            self.assertNotEqual(a["files"]["model/model.py"], b["files"]["model/model.py"])
            self.assertEqual(b, source_fingerprint(root))

    def test_cold_coverage_reporting_and_warm_only_rejection(self):
        model = fixture()
        train = np.array([[0, 0], [1, 0], [0, 1], [0, 2], [1, 3]])
        targets = np.array([[0, 4], [2, 1], [2, 4]])
        unchanged = targets.copy()
        report = evaluate(model, targets, train, ks=(1, 2), return_report=True)
        coverage = report["coverage"]
        for key, expected in {"target_users": 2, "target_items": 2, "cold_target_users": 1,
                              "cold_target_items": 1, "interactions_with_cold_user": 2,
                              "interactions_with_cold_item": 2, "interactions_with_either_cold_endpoint": 3,
                              "interactions_with_both_cold_endpoints": 1, "catalog_cold_users": 1,
                              "catalog_cold_items": 1}.items():
            self.assertEqual(coverage[key], expected)
        np.testing.assert_array_equal(recommend(model.encode(), [2], {}, 3, 2), [[0, 1, 2]])
        with self.assertRaisesRegex(ValueError, "no filtering"):
            evaluate(model, targets, train, ks=(1,), population="warm_only")
        np.testing.assert_array_equal(targets, unchanged)
        warm = evaluate(model, np.array([[1, 2]]), train, ks=(1,), population="warm_only", return_report=True)
        self.assertEqual(warm["coverage"]["candidate_catalog"], "all_declared_items")
        self.assertEqual(warm["coverage"]["catalog_cold_items"], 1)

    def test_mtmt_ftmt_visibility_finiteness_and_unchanged_support(self):
        with fixture_directory() as tmp:
            root = Path(tmp)
            arrays = {"train": np.array([[0, 0], [0, 1]]), "valid": np.array([[0, 2]]), "test": np.array([[1, 1]]),
                      "visual": np.array([[np.nan, 0.], [1., 0.], [0., 1.]], dtype=np.float32),
                      "text": np.ones((3, 2), dtype=np.float32), "natural_mask": np.ones((3, 2), dtype=bool),
                      "experiment_mask": np.array([[False, True], [True, True], [True, True]])}
            files = {}
            for name, arr in arrays.items():
                np.save(root / (name + ".npy"), arr)
                files[name] = {"path": name + ".npy", "sha256": sha256_file(root / (name + ".npy"))}
            meta = {k: "synthetic" for k in ("dataset", "source", "version", "filtering", "split_protocol", "id_mapping", "encoders", "mask_protocol")}
            manifest = {"n_users": 2, "n_items": 3, "metadata": meta, "files": files}
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            load_manifest(path, protocol="mtmt")  # evaluation-hidden NaN is also training-hidden here
            with self.assertRaisesRegex(ValueError, "training-visible"):
                load_manifest(path, protocol="ftmt")
            arrays["visual"][0] = [1., 0.]
            np.save(root / "visual.npy", arrays["visual"])
            files["visual"]["sha256"] = sha256_file(root / "visual.npy")
            path.write_text(json.dumps(manifest), encoding="utf-8")
            for protocol in ("mtmt", "ftmt"):
                data = load_manifest(path, protocol=protocol)
                self.assertEqual(data.provenance["protocol"], protocol)
                model = BAST(2, 3, data.train, data.visual, data.text, data.train_observed, ModelConfig(dim=2))
                graph_before = (model.graph.anchor.clone(), model.graph.item.clone())
                eval_mask = torch.from_numpy(data.observed)
                before = model.encode(eval_mask, diagnostics=True)
                with torch.no_grad():
                    model.visual[0] = torch.tensor([99., -12.])
                after = model.encode(eval_mask, diagnostics=True)
                self.assertTrue(torch.equal(before[1], after[1]))
                self.assertEqual(after[2]["z"][0, 0].norm().item(), 0.)
                if protocol == "ftmt":
                    self.assertGreater(model.encode(diagnostics=True)[2]["z"][0, 0].norm().item(), 0.)
                else:
                    with self.assertRaisesRegex(ValueError, "reveal"):
                        model.encode(torch.from_numpy(data.natural_mask))
                with patch.object(model, "encode", wraps=model.encode) as encode:
                    evaluate(model, data.valid, data.train, ks=(1,), view_mask=eval_mask)
                    self.assertIs(encode.call_args.kwargs["view_mask"], eval_mask)
                self.assertTrue(torch.equal(graph_before[0], model.graph.anchor))
                self.assertTrue(torch.equal(graph_before[1], model.graph.item))

    def test_cuda_determinism_preflight_with_mocked_state_only(self):
        # No actual CUDA query, initialization, environment change or optimization.
        with patch("utils.reproducibility.os.environ", {}) as environment, \
             patch("utils.reproducibility.torch.cuda.is_initialized", return_value=False) as initialized, \
             patch("utils.reproducibility.torch.use_deterministic_algorithms") as enable:
            configure_determinism("cpu")
            initialized.assert_not_called()
            self.assertEqual(environment, {})
            configure_determinism("cuda:0")
            self.assertEqual(environment["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
            environment["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
            configure_determinism("cuda")
            self.assertEqual(environment["CUBLAS_WORKSPACE_CONFIG"], ":16:8")
            environment["CUBLAS_WORKSPACE_CONFIG"] = "invalid"
            with self.assertRaises(ValueError):
                configure_determinism("cuda")
            initialized.return_value = True
            with self.assertRaises(RuntimeError):
                configure_determinism("cuda")
            enable.assert_called_with(True)

    def test_split_checks(self):
        train = np.array([[0, 0]], dtype=np.int64)
        empty = np.empty((0, 2), dtype=np.int64)
        validate_splits(train, empty, np.array([[1, 1]]), 2, 3)
        for valid in (train.copy(), np.array([[1, 1], [1, 1]]), np.array([[0, 3]]), np.array([[.5, 1.]])):
            with self.assertRaises(ValueError):
                validate_splits(train, valid, empty, 2, 3)

    def test_masks_repeat_seed_and_share_view(self):
        natural = np.ones((20, 2), dtype=bool)
        natural[0] = False
        for pattern in ("independent", "coupled", "visual", "text"):
            a = fixed_mask(natural, .5, 11, pattern)
            b = fixed_mask(natural, .5, 11, pattern)
            np.testing.assert_array_equal(a, b)
            self.assertFalse(a[0].any())
        observed = torch.from_numpy(natural)
        a = dropout_view(observed, .4, torch.Generator().manual_seed(5))
        b = dropout_view(observed, .4, torch.Generator().manual_seed(5))
        self.assertTrue(torch.equal(a, b))
        self.assertFalse((a & ~observed).any())

    def test_chunked_ranking_mask_ties_and_metrics(self):
        encoded = (torch.tensor([[1., 0.]]), torch.tensor([[100., 0.], [3., 0.], [3., 0.], [-2., 0.]]))
        a = recommend(encoded, [0], {0: {0}}, 4, item_chunk=1)
        b = recommend(encoded, [0], {0: {0}}, 4, item_chunk=3)
        np.testing.assert_array_equal(a, [[1, 2, 3, -1]])
        np.testing.assert_array_equal(a, b)
        metrics = ranking_metrics(a, [0], {0: {2, 3}}, ks=(1, 2, 3))
        self.assertEqual(metrics["recall@1"], 0.)
        self.assertEqual(metrics["recall@2"], .5)
        self.assertEqual(metrics["recall@3"], 1.)
        self.assertAlmostEqual(metrics["ndcg@2"], (1/np.log2(3))/(1+1/np.log2(3)))

    def test_eval_target_exclusion_and_determinism(self):
        model = fixture()
        train = np.array([[0, 0], [1, 0], [0, 1], [0, 2], [1, 3]])
        valid = np.array([[0, 3], [1, 2]])
        a = evaluate(model, valid, train, ks=(1, 2), item_chunk=1)
        b = evaluate(model, valid, train, ks=(1, 2), item_chunk=3)
        self.assertEqual(a, b)
        self.assertTrue(model.training)
        with self.assertRaises(ValueError):
            evaluate(model, valid, np.concatenate((train, valid)))

    def test_negative_sampling_does_not_consult_heldout(self):
        sampler = TrainNegativeSampler(np.array([[0, 0], [0, 1]]), 3)
        # The only negative is a hypothetical held-out positive; it must remain eligible.
        np.testing.assert_array_equal(sampler.sample([0, 0], np.random.default_rng(1)), [2, 2])
        with self.assertRaises(ValueError):
            TrainNegativeSampler(np.array([[0, 0], [0, 1], [0, 2]]), 3)

    def test_local_manifest_hashes_and_mmrec_import(self):
        with fixture_directory() as tmp:
            root = Path(tmp)
            arrays = {"train": np.array([[0, 0]]), "valid": np.array([[0, 1]]), "test": np.array([[1, 2]]),
                      "visual": np.ones((3, 2), dtype=np.float32), "text": np.ones((3, 4), dtype=np.float32),
                      "natural_mask": np.ones((3, 2), dtype=bool), "experiment_mask": np.ones((3, 2), dtype=bool)}
            files = {}
            for name, arr in arrays.items():
                np.save(root / (name + ".npy"), arr)
                files[name] = {"path": name + ".npy", "sha256": sha256_file(root / (name + ".npy"))}
            meta = {k: "synthetic; no real dataset claim" for k in ("dataset", "source", "version", "filtering", "split_protocol", "id_mapping", "encoders", "mask_protocol")}
            manifest = {"n_users": 2, "n_items": 3, "metadata": meta, "files": files}
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            data = load_manifest(path)
            self.assertEqual(data.n_items, 3)
            self.assertEqual(data.provenance["evaluation_coverage"]["valid"]["cold_target_items"], 1)
            self.assertEqual(data.provenance["evaluation_coverage"]["test"]["cold_target_users"], 1)
            self.assertEqual(data.provenance["evaluation_coverage"]["test"]["cold_target_items"], 1)
            with self.assertRaisesRegex(ValueError, "warm_only"):
                load_manifest(path, evaluation_population="warm_only")
            self.assertEqual(data.provenance["file_sha256"]["train"], files["train"]["sha256"])
            inter = root / "tiny.inter"
            inter.write_text("userID\titemID\tx_label\n0\t0\t0\n0\t1\t1\n1\t2\t2\n", encoding="utf-8")
            for got, expected in zip(load_mmrec_inter(inter), (data.train, data.valid, data.test)):
                np.testing.assert_array_equal(got, expected)
            alternate = {k: v for k,v in files.items() if k not in ("train", "valid", "test")}
            alternate["mmrec_inter"] = {"path": "tiny.inter", "sha256": sha256_file(inter)}
            manifest["files"] = alternate
            path.write_text(json.dumps(manifest), encoding="utf-8")
            np.testing.assert_array_equal(load_manifest(path).train, data.train)
            alternate["mmrec_inter"]["sha256"] = "wrong"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_manifest(path)
            csv = root / "edges.csv"
            csv.write_text("user,item\n0,1\n", encoding="utf-8")
            np.testing.assert_array_equal(load_edges(csv), [[0, 1]])

    def test_unknown_configs_and_cli_refuses_without_optin(self):
        with self.assertRaises(ValueError):
            ModelConfig.from_dict({"invented": 1})
        with self.assertRaises(ValueError):
            ModelConfig(beta=1.1)
        with fixture_directory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text('{"training":{"invented":1}}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)
        with patch("common.trainer.run_training") as run:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--config", "unused", "--manifest", "unused", "--output", "unused"])
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
