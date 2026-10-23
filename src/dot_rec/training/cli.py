"""Explicit user training entrypoint. DO NOT execute training on the development PC."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import numpy as np
import torch
from dot_rec.data import load_manifest, dropout_view
from dot_rec.models import BAST
from dot_rec.evaluation import evaluate
from .config import load_config, resolved_config
from .sampling import TrainNegativeSampler


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_fingerprint(root):
    """Hash only regular Python source and JSON configuration files; no data/secrets."""
    root = Path(root).resolve()
    hashes = {}
    for directory, suffix in (("src", ".py"), ("configs", ".json")):
        base = root / directory
        if not base.is_dir() or base.is_symlink():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix != suffix or not path.is_file():
                continue
            if any(part.is_symlink() for part in (path, *path.parents) if part != root):
                continue
            relative = path.relative_to(root).as_posix()
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"algorithm": "sha256", "scope": "regular src/**/*.py and configs/**/*.json; symlinks excluded",
            "files": hashes, "digest": hashlib.sha256(payload).hexdigest() if hashes else None}


def code_version():
    root = Path(__file__).resolve().parents[3]
    fingerprint = source_fingerprint(root)
    # Installed wheels may not have repository metadata; do not invent a SHA.
    try:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True)
    except OSError:
        return {"commit": "unknown", "dirty": None, "source_fingerprint": fingerprint}
    return {"commit": result.stdout.strip() if result.returncode == 0 else "unknown",
            "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
            "source_fingerprint": fingerprint}


def configure_determinism(device):
    """Only called inside an explicit user run, before seeds or CUDA tensors."""
    if torch.device(device).type == "cuda":
        if torch.cuda.is_initialized():
            raise RuntimeError("Deterministic CUDA runs require a fresh process before CUDA initialization")
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace is None:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        elif workspace not in (":4096:8", ":16:8"):
            raise ValueError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8 for deterministic CUDA")
    torch.use_deterministic_algorithms(True)


def run_training(args):
    # This function contains the only optimization loop. Never invoked by tests.
    model_cfg, cfg = load_config(args.config)
    data = load_manifest(args.manifest, protocol=cfg.protocol, evaluation_population=cfg.evaluation_population)
    if not len(data.valid) or not len(data.test):
        raise ValueError("Nonempty validation and test splits required")
    sampler = TrainNegativeSampler(data.train, data.n_items)
    configure_determinism(cfg.device)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)  # never overwrite a prior run
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
    model = BAST(data.n_users, data.n_items, data.train, data.visual, data.text, data.train_observed, model_cfg).to(cfg.device)
    eval_mask = torch.as_tensor(data.observed, dtype=torch.bool, device=cfg.device)
    write_json(output / "resolved_config.json", resolved_config(model_cfg, cfg))
    write_json(output / "provenance.json", data.provenance)
    write_json(output / "environment.json", {"python": platform.python_version(), "torch": torch.__version__,
                                            "numpy": np.__version__, "code": code_version(), "device": cfg.device})
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    best, best_epoch, stale = -float("inf"), -1, 0
    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(len(data.train))
        total, examples = 0.0, 0
        for start in range(0, len(order), cfg.batch_size):
            edges = data.train[order[start:start + cfg.batch_size]]
            negatives = sampler.sample(edges[:, 0], rng)
            batch = tuple(torch.as_tensor(a, dtype=torch.long, device=cfg.device)
                          for a in (edges[:, 0], edges[:, 1], negatives))
            view = dropout_view(model.observed, cfg.dropout, generator)
            optimizer.zero_grad(set_to_none=True)
            losses = model.calculate_loss(batch, cfg.l2_weight, view)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError("Nonfinite training loss")
            losses["total"].backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Nonfinite gradient")
            optimizer.step()
            total += float(losses["total"].detach()) * len(edges)
            examples += len(edges)
        valid_report = evaluate(model, data.valid, data.train, user_batch=cfg.user_batch, item_chunk=cfg.item_chunk,
                                view_mask=eval_mask, population=cfg.evaluation_population, return_report=True)
        valid = valid_report["metrics"]
        score = valid[cfg.selection_metric]
        with (output / "validation.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"epoch": epoch, "loss": total / examples, "validation": valid,
                                     "coverage": valid_report["coverage"]}) + "\n")
        if score > best:
            best, best_epoch, stale = score, epoch, 0
            # Features/support stay external and hash-checked, not duplicated into checkpoints.
            torch.save({name: p.detach().cpu() for name, p in model.named_parameters()}, output / "best.pt")
            write_json(output / "selection.json", {"epoch": epoch, "metric": cfg.selection_metric, "validation": valid,
                                                   "coverage": valid_report["coverage"]})
        else:
            stale += 1
        if stale >= cfg.patience:
            break
    selected = torch.load(output / "best.pt", map_location=cfg.device, weights_only=True)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(selected[name])
    excluded = np.concatenate((data.train, data.valid)) if cfg.test_mask_valid else data.train
    test_report = evaluate(model, data.test, excluded, user_batch=cfg.user_batch, item_chunk=cfg.item_chunk,
                           view_mask=eval_mask, population=cfg.evaluation_population, return_report=True)
    write_json(output / "final_test.json", {"selected_epoch": best_epoch, "test_mask_valid": cfg.test_mask_valid,
                                           "test": test_report["metrics"], "coverage": test_report["coverage"],
                                           "selection_source": "validation_only", "protocol": cfg.protocol})
    print(f"Completed explicit user run; artifacts at {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="BAST user training CLI. Training is prohibited on the development PC.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute-training", action="store_true", help="Explicit opt-in for a separately authorized execution environment")
    args = parser.parse_args(argv)
    if not args.execute_training:
        parser.error("No training executed. --execute-training is required in an authorized environment; prohibited on the development PC.")
    run_training(args)


if __name__ == "__main__":
    main()
