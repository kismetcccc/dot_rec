"""Independent tiny metric references; no upstream code or training."""
import ast
import inspect
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from common.dataset import load_manifest
from common.evaluator import ranking_metrics, recommend, evaluate
from common import trainer
from utils.config import TrainConfig, load_config, resolved_config
from model import ModelConfig
from test_model import fixture


class LGMRecTests(unittest.TestCase):
    def test_map_denominator_precision_and_macro(self):
        # user 0: two hits at ranks 1,3; three positives. user 1: one positive/hit.
        ranked = [[1, 9, 2], [4, -1, -1]]
        targets = {0: {1, 2, 3}, 1: {4}}
        got = ranking_metrics(ranked, [0, 1], targets, (2, 3, 5), ("recall", "ndcg", "precision", "map"))
        self.assertAlmostEqual(got["map@2"], (.5 + 1.) / 2)
        self.assertAlmostEqual(got["map@3"], ((1 + 2/3)/3 + 1) / 2)
        self.assertAlmostEqual(got["map@5"], got["map@3"])
        self.assertAlmostEqual(got["precision@5"], (.4 + .2) / 2)
        self.assertAlmostEqual(got["recall@3"], (2/3 + 1) / 2)
        self.assertNotAlmostEqual(got["recall@3"], 3/4)  # micro would differ
        encoded = (torch.ones(1, 1), torch.ones(2, 1))
        short = recommend(encoded, [0], {0: {1}}, 5)
        np.testing.assert_array_equal(short, [[0, -1]])
        metric = ranking_metrics(short, [0], {0: {0}}, (5,), ("precision", "map"))
        self.assertEqual(metric, {"precision@5": .2, "map@5": 1.})

    def test_train_users_only_keeps_cold_items_and_support(self):
        model = fixture()
        targets = np.array([[0, 4], [2, 1], [2, 4]])
        original = targets.copy()
        support = model.graph.item.clone()
        report = evaluate(model, targets, np.array([[0, 0]]), ks=(1, 2),
                          metrics=("recall", "precision", "map"), population="train_users_only", return_report=True)
        c = report["coverage"]
        for name, value in {"raw_target_users": 2, "effective_target_users": 1, "filtered_target_users": 1,
                            "raw_target_interactions": 3, "effective_target_interactions": 1,
                            "filtered_target_interactions": 2, "effective_cold_target_items": 1}.items():
            self.assertEqual(c[name], value)
        expected = evaluate(model, np.array([[0, 4]]), np.array([[0, 0]]), ks=(1, 2),
                            metrics=("recall", "precision", "map"))
        self.assertEqual(report["metrics"], expected)
        np.testing.assert_array_equal(targets, original)
        self.assertTrue(torch.equal(support, model.graph.item))
        self.assertEqual(c["candidate_catalog"], "all_declared_items")
        with self.assertRaisesRegex(ValueError, "Empty effective"):
            evaluate(model, np.array([[2, 4]]), np.array([[0, 0]]), population="train_users_only")
        with self.assertRaises(ValueError):
            evaluate(model, np.array([[0, 4]]), np.array([[0, 0]]), population="warm_only")

    def test_preset_config_and_validation_call_contract(self):
        root = Path(__file__).resolve().parents[1]
        model_cfg, cfg = load_config(root / "configs/model/bast_lgmrec.json")
        self.assertEqual(cfg.metrics, ("recall", "ndcg", "precision", "map"))
        self.assertEqual(cfg.cutoffs, (5, 10, 20, 50))
        self.assertEqual(cfg.selection_metric, "recall@20")
        self.assertFalse(cfg.test_mask_valid)
        self.assertEqual(cfg.evaluation_population, "train_users_only")
        self.assertEqual(resolved_config(model_cfg, cfg)["training"]["cutoffs"], cfg.cutoffs)
        got = evaluate(fixture(), np.array([[0, 3]]), np.array([[0, 0]]), ks=cfg.cutoffs, metrics=cfg.metrics)
        self.assertEqual(len(got), 16)
        for kwargs in ({"metrics": ["recall", "recall"]}, {"metrics": ["unknown"]},
                       {"cutoffs": [5, 5]}, {"cutoffs": [True]}, {"cutoffs": [5.0]},
                       {"metrics": ["map"], "selection_metric": "recall@20"}):
            with self.assertRaises(ValueError):
                TrainConfig(**kwargs)
        calls = [n for n in ast.walk(ast.parse(inspect.getsource(trainer.run_training)))
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "evaluate"]
        keywords = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
        self.assertEqual(keywords["ks"], "cfg.cutoffs")
        self.assertEqual(keywords["metrics"], "cfg.metrics")

    def test_manifest_placeholders_rejected_before_file_access(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("baby", "sports", "clothing"):
            with patch("common.dataset.sha256_file") as hash_file:
                with self.assertRaisesRegex(ValueError, "placeholder or Git blob"):
                    load_manifest(root / f"configs/dataset/{name}.manifest.example.json")
                hash_file.assert_not_called()
