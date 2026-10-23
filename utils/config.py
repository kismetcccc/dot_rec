from dataclasses import dataclass, fields, asdict
import json
import math
from pathlib import Path
from model import ModelConfig


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 2026
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 0.001
    l2_weight: float = 0.00001
    dropout: float = 0.1
    patience: int = 10
    metrics: tuple = ("recall", "ndcg")
    cutoffs: tuple = (10, 20)
    selection_metric: str = "recall@20"
    user_batch: int = 128
    item_chunk: int = 4096
    device: str = "cpu"
    test_mask_valid: bool = False
    protocol: str = "mtmt"
    evaluation_population: str = "all"

    def __post_init__(self):
        for name in ("epochs", "batch_size", "patience", "user_batch", "item_chunk"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0 or self.seed >= 2**32:
            raise ValueError("seed must be an integer in [0,2**32)")
        for name in ("learning_rate", "l2_weight", "dropout"):
            if type(getattr(self, name)) not in (int, float) or not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.learning_rate <= 0 or self.l2_weight < 0 or not 0 <= self.dropout <= 1:
            raise ValueError("Invalid learning rate, L2 or dropout")
        if not isinstance(self.metrics, (list, tuple)) or not self.metrics or any(
                not isinstance(m, str) or m not in ("recall", "ndcg", "precision", "map") for m in self.metrics):
            raise ValueError("Invalid metrics")
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError("Duplicate metrics")
        if not isinstance(self.cutoffs, (list, tuple)) or not self.cutoffs or any(type(k) is not int or k < 1 for k in self.cutoffs):
            raise ValueError("Invalid cutoffs")
        if len(set(self.cutoffs)) != len(self.cutoffs):
            raise ValueError("Duplicate cutoffs")
        object.__setattr__(self, "metrics", tuple(self.metrics))
        object.__setattr__(self, "cutoffs", tuple(self.cutoffs))
        if self.selection_metric not in {f"{m}@{k}" for m in self.metrics for k in self.cutoffs}:
            raise ValueError("Selection metric must be an implemented validation metric")
        if type(self.test_mask_valid) is not bool or not isinstance(self.device, str):
            raise ValueError("Invalid device or test_mask_valid")
        if self.protocol not in ("mtmt", "ftmt"):
            raise ValueError("protocol must be mtmt or ftmt")
        if self.evaluation_population not in ("all", "warm_only", "train_users_only"):
            raise ValueError("evaluation_population must be all, warm_only or train_users_only")


def load_config(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Configuration must be an object")
    if set(data) - {"model", "training"}:
        raise ValueError("Unknown top-level config keys")
    model = ModelConfig.from_dict(data.get("model", {}))
    train_data = data.get("training", {})
    if not isinstance(train_data, dict):
        raise ValueError("Training configuration must be an object")
    if set(train_data) - {f.name for f in fields(TrainConfig)}:
        raise ValueError("Unknown training config keys")
    return model, TrainConfig(**train_data)


def resolved_config(model, training):
    return {"model": asdict(model), "training": asdict(training)}
