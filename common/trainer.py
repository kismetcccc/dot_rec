"""Explicit user training entrypoint. DO NOT execute training on the development PC."""
import argparse
import json
from pathlib import Path
import platform
import random
import numpy as np
import torch
from common.dataset import load_manifest
from utils.masks import dropout_view
from common.dataset import sha256_file
from model import BAST
from common.evaluator import evaluate
from utils.config import load_config, resolved_config
from utils.sampling import TrainNegativeSampler
from utils.reproducibility import code_version, configure_determinism


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


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
    write_json(output / "selection_binding.json", {
        name: sha256_file(output / name)
        for name in ("resolved_config.json", "provenance.json", "selection.json", "best.pt", "environment.json")})
    print(f"Validation-only run completed; selected epoch {best_epoch}; artifacts at {output}")


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
