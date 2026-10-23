import ast
import contextlib
import io
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from common import trainer as cli
from utils.config import TrainConfig, resolved_config
from model import ModelConfig
from common.dataset import sha256_file
from common.final_test import run_final_test, main as final_main
from test_model import fixture
from test_protocol import fixture_directory


class FinalTestTests(unittest.TestCase):
    def test_final_cli_requires_explicit_optin(self):
        with patch("common.final_test.run_final_test") as run:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                final_main(["--run", "unused", "--manifest", "unused"])
            run.assert_not_called()
            final_main(["--run", "unused", "--manifest", "unused", "--execute-final-test"])
            run.assert_called_once_with("unused", "unused")

    def test_training_control_flow_is_validation_only(self):
        # Static call inspection plus stub dispatch; never enter an optimizer loop.
        tree = ast.parse(inspect.getsource(cli.run_training))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "evaluate"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(ast.unparse(calls[0].args[1]), "data.valid")
        self.assertNotIn("final_test.json", inspect.getsource(cli.run_training))
        with patch.object(cli, "run_training") as run:
            cli.main(["--config", "unused", "--manifest", "unused", "--output", "unused", "--execute-training"])
            run.assert_called_once()

    def test_final_test_artifact_validation_single_call_and_no_overwrite(self):
        with fixture_directory() as tmp:
            root = Path(tmp)
            model = fixture()
            cfg = TrainConfig()
            def save(name, value):
                (root / name).write_text(json.dumps(value), encoding="utf-8")
            save("resolved_config.json", resolved_config(ModelConfig(), cfg))
            provenance = {"synthetic": "fixture"}
            save("provenance.json", provenance)
            save("selection.json", {"epoch": 2, "metric": cfg.selection_metric})
            fingerprint = {"digest": "synthetic", "files": {name: "hash" for name in
                ("main.py", "evaluate.py", "model/bast.py", "common/trainer.py", "utils/config.py")}}
            save("environment.json", {"code": {"source_fingerprint": fingerprint, "commit": "old"}})
            torch.save(dict(model.named_parameters()), root / "best.pt")
            def bind():
                save("selection_binding.json", {name: sha256_file(root / name) for name in
                    ("resolved_config.json", "provenance.json", "selection.json", "best.pt", "environment.json")})
            bind()
            data = SimpleNamespace(provenance=provenance, n_users=3, n_items=5,
                train=np.array([[0, 0]]), valid=np.array([[0, 1]]), test=np.array([[0, 2]]),
                visual=None, text=None, train_observed=None, observed=np.ones((5, 2), dtype=bool))
            with patch("common.final_test.source_fingerprint", return_value=fingerprint) as source, \
                 patch("common.final_test.load_manifest", return_value=data), \
                 patch("common.final_test.BAST", return_value=model), \
                 patch("common.final_test.configure_determinism"), \
                 patch("common.final_test.evaluate", return_value={"metrics": {}, "coverage": {}}) as evaluate:
                save("selection.json", {"epoch": 3, "metric": cfg.selection_metric})
                with self.assertRaisesRegex(ValueError, "artifact changed"):
                    run_final_test(root, "unused")
                evaluate.assert_not_called()
                bind()
                original = torch.load(root / "best.pt", weights_only=True)
                torch.save({"wrong": torch.zeros(1)}, root / "best.pt")
                bind()
                with self.assertRaisesRegex(ValueError, "parameter names"):
                    run_final_test(root, "unused")
                evaluate.assert_not_called()
                torch.save(original, root / "best.pt")
                bind()
                data.provenance = {"changed": True}
                with self.assertRaisesRegex(ValueError, "data hashes"):
                    run_final_test(root, "unused")
                data.provenance = provenance
                source.return_value = {**fingerprint, "digest": "changed"}
                with self.assertRaisesRegex(ValueError, "Runtime source"):
                    run_final_test(root, "unused")
                source.return_value = {"digest": None, "files": {}}
                with self.assertRaisesRegex(ValueError, "unavailable"):
                    run_final_test(root, "unused")
                evaluate.assert_not_called()
                source.return_value = fingerprint
                run_final_test(root, "unused")
                evaluate.assert_called_once()
                self.assertIs(evaluate.call_args.args[1], data.test)
                with self.assertRaises(FileExistsError):
                    run_final_test(root, "unused")
                evaluate.assert_called_once()
