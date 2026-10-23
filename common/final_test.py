"""Explicit final test of a user-selected validation run; never trains."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from common.dataset import load_manifest
from common.dataset import sha256_file
from model import BAST
from utils.config import load_config
from utils.reproducibility import configure_determinism, source_fingerprint
from .evaluator import evaluate


def run_final_test(run, manifest):
    run = Path(run).resolve()
    destination = run / "final_test.json"
    if destination.exists():
        raise FileExistsError("final_test.json already exists; refusing another final test")
    model_cfg, cfg = load_config(run / "resolved_config.json")
    saved = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    data = load_manifest(manifest, protocol=cfg.protocol, evaluation_population=cfg.evaluation_population)
    if data.provenance != saved:
        raise ValueError("Manifest, data hashes or protocol differ from selected run")
    selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    if selection["metric"] != cfg.selection_metric:
        raise ValueError("Selection metric differs from resolved configuration")
    binding = json.loads((run / "selection_binding.json").read_text(encoding="utf-8"))
    for name in ("resolved_config.json", "provenance.json", "selection.json", "best.pt", "environment.json"):
        if binding.get(name) != sha256_file(run / name):
            raise ValueError(f"Selected run artifact changed: {name}")
    environment = json.loads((run / "environment.json").read_text(encoding="utf-8"))
    saved_source = environment.get("code", {}).get("source_fingerprint")
    current_source = source_fingerprint(Path(__file__).resolve().parents[1])
    required_sources = {"main.py", "evaluate.py", "model/bast.py", "common/trainer.py", "utils/config.py"}
    for fingerprint in (saved_source, current_source):
        if not isinstance(fingerprint, dict) or not fingerprint.get("digest") or not required_sources.issubset(fingerprint.get("files", {})):
            raise ValueError("Complete source fingerprint unavailable; use the original source checkout")
    if saved_source != current_source:
        raise ValueError("Runtime source fingerprint differs from selected run")
    configure_determinism(cfg.device)
    model = BAST(data.n_users, data.n_items, data.train, data.visual, data.text,
                 data.train_observed, model_cfg).to(cfg.device)
    selected = torch.load(run / "best.pt", map_location=cfg.device, weights_only=True)
    parameters = dict(model.named_parameters())
    if set(selected) != set(parameters):
        raise ValueError("Checkpoint parameter names do not match model")
    with torch.no_grad():
        for name, parameter in parameters.items():
            tensor = selected[name]
            if not isinstance(tensor, torch.Tensor) or tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
                raise ValueError(f"Checkpoint parameter mismatch: {name}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Nonfinite checkpoint parameter: {name}")
            parameter.copy_(tensor)
    excluded = np.concatenate((data.train, data.valid)) if cfg.test_mask_valid else data.train
    # Exclusive creation also prevents concurrent evaluations of this run.
    # A failed evaluation leaves a marker; it must be inspected, never overwritten.
    with destination.open("x", encoding="utf-8") as stream:
        report = evaluate(model, data.test, excluded, user_batch=cfg.user_batch,
                          item_chunk=cfg.item_chunk, view_mask=data.observed,
                          population=cfg.evaluation_population, return_report=True, ks=cfg.cutoffs, metrics=cfg.metrics)
        json.dump({"selected_epoch": selection["epoch"], "selection_source": "validation_only",
                   "protocol": cfg.protocol, "test_mask_valid": cfg.test_mask_valid,
                   "test": report["metrics"], "coverage": report["coverage"],
                   "artifact_sha256": binding}, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Final test of one explicitly selected validation run")
    parser.add_argument("--run", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--execute-final-test", action="store_true", help="Explicit final evaluation after validation-only run selection")
    args = parser.parse_args(argv)
    if not args.execute_final_test:
        parser.error("No test executed. --execute-final-test is required after freezing validation-based selection.")
    run_final_test(args.run, args.manifest)


if __name__ == "__main__":
    main()
