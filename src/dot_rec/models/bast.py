from dataclasses import dataclass, fields
import math
import torch
from torch import nn
from torch.nn import functional as F
from dot_rec.graph import Incidence, safe_normalize
from dot_rec.graph.incidence import require_full_precision
from .observed import ObservedTransport
from .missing import MissingTransport


@dataclass(frozen=True)
class ModelConfig:
    dim: int = 64
    layers: int = 1
    eta: float = 0.2
    alpha: float = 0.2
    beta: float = 0.1
    kappa: float = 2.0
    tau: float = 1.0
    learned_group: bool = True
    compensation: bool = True
    quality_mode: str = "full"

    def __post_init__(self):
        if type(self.dim) is not int or self.dim < 1 or type(self.layers) is not int or self.layers < 0:
            raise ValueError("dim must be positive and layers nonnegative integers")
        for name in ("eta", "alpha", "beta"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite in [0,1]")
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in (self.kappa, self.tau)):
            raise ValueError("kappa/tau must be positive finite")
        if type(self.learned_group) is not bool or type(self.compensation) is not bool:
            raise ValueError("Ablation switches must be booleans")
        if self.quality_mode not in ("full", "unit", "no_support", "no_agreement"):
            raise ValueError("Invalid quality_mode")

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("Model configuration must be an object")
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown model keys: {sorted(unknown)}")
        return cls(**data)


class BAST(nn.Module):
    def __init__(self, n_users, n_items, train_edges, visual, text, observed, config=None):
        super().__init__()
        self.config = config if config is not None else ModelConfig()
        if not isinstance(self.config, ModelConfig):
            raise ValueError("config must be ModelConfig")
        self.graph = Incidence(n_users, n_items, train_edges)
        # All construction is CPU/float32. Move the entire model afterwards.
        mask = torch.as_tensor(observed, device="cpu")
        if mask.dtype != torch.bool or mask.shape != (n_items, 2):
            raise ValueError("observed must be bool [n_items,2]")
        self.register_buffer("observed", mask.clone())
        self.projections = nn.ModuleList()
        for m, (name, raw) in enumerate((("visual", visual), ("text", text))):
            x = torch.as_tensor(raw, dtype=torch.float32, device="cpu").detach().clone()
            if x.ndim != 2 or x.shape[0] != n_items or x.shape[1] < 1:
                raise ValueError(f"Invalid {name} feature shape")
            if not torch.isfinite(x[mask[:, m]]).all():
                raise ValueError("Observed features must be finite")
            self.register_buffer(name, x)
            self.projections.append(nn.Linear(x.shape[1], self.config.dim, bias=False, device="cpu", dtype=torch.float32))
        self.user_ids = nn.Embedding(n_users, self.config.dim, device="cpu", dtype=torch.float32)
        self.item_ids = nn.Embedding(n_items, self.config.dim, device="cpu", dtype=torch.float32)
        nn.init.normal_(self.user_ids.weight, std=0.1)
        nn.init.normal_(self.item_ids.weight, std=0.1)
        self.observed_transport = ObservedTransport()
        self.missing_transport = MissingTransport()

    def _check_execution(self):
        reference = self.user_ids.weight
        tensors = list(self.parameters()) + list(self.buffers())
        floating = [t for t in tensors if t.is_floating_point()]
        require_full_precision(*floating)
        if any(t.device != reference.device for t in tensors):
            raise ValueError("Move the whole model with model.to(device); mixed model devices are unsupported")
        if any(t.dtype != reference.dtype for t in floating):
            raise ValueError("Model floating tensors must share float32 or float64 dtype")

    def _ids(self, value, name, upper):
        ids = torch.as_tensor(value)
        if ids.dtype not in (torch.int32, torch.int64) or ids.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional int32/int64 array")
        if ((ids < 0) | (ids >= upper)).any():
            raise ValueError(f"{name} IDs must lie in [0,{upper})")
        return ids.to(device=self.user_ids.weight.device, dtype=torch.long)

    def _encoded(self, encoded):
        self._check_execution()
        if encoded is None:
            return self.encode()
        if not isinstance(encoded, (tuple, list)) or len(encoded) != 2:
            raise ValueError("encoded must be a pair of user/item tensors")
        for tensor, count in zip(encoded, (self.graph.n_users, self.graph.n_items)):
            if not isinstance(tensor, torch.Tensor) or tensor.shape != (count, self.config.dim):
                raise ValueError("Encoded tensors must match catalog and embedding dimensions")
            if tensor.device != self.user_ids.weight.device or tensor.dtype != self.user_ids.weight.dtype:
                raise ValueError("Encoded tensors must match model device and dtype")
        return encoded

    def encode(self, view_mask=None, diagnostics=False):
        self._check_execution()
        mask = self.observed if view_mask is None else torch.as_tensor(view_mask, device=self.observed.device)
        if mask.dtype != torch.bool or mask.shape != self.observed.shape:
            raise ValueError("view_mask must match observed bool shape")
        if (mask & ~self.observed).any():
            raise ValueError("A view cannot reveal unavailable features")
        z = []
        for m, raw in enumerate((self.visual, self.text)):
            clean = torch.where(mask[:, m, None], raw, torch.zeros_like(raw))
            projected = safe_normalize(self.projections[m](clean))
            z.append(torch.where(mask[:, m, None], projected, torch.zeros_like(projected)))
        z = torch.stack(z, 1)
        p, b = self.graph.collaborative(self.user_ids.weight, self.item_ids.weight, self.config.layers)
        h, gate = self.observed_transport(z, mask, b, self.graph, self.config.eta, self.config.learned_group)
        if self.config.compensation:
            c, q = self.missing_transport(z, mask, self.graph, self.config.kappa, self.config.tau, self.config.quality_mode)
        else:
            c = torch.zeros_like(z)
            q = z.new_zeros((len(self.graph.item), 2))
        items = b + self.config.alpha * h.sum(1) + self.config.beta * c.sum(1)
        if diagnostics:
            return p, items, {"behavior": b, "z": z, "h": h, "c": c, "g": gate, "q": q}
        return p, items

    def score(self, users, items, encoded=None):
        users = self._ids(users, "users", self.graph.n_users)
        items = self._ids(items, "items", self.graph.n_items)
        if users.shape != items.shape:
            raise ValueError("Paired users/items must have identical shapes; broadcasting is forbidden")
        p, v = self._encoded(encoded)
        return (p[users] * v[items]).sum(-1)

    def full_sort_predict(self, users, encoded=None):
        users = self._ids(users, "users", self.graph.n_users)
        p, v = self._encoded(encoded)
        return p[users] @ v.T

    def calculate_loss(self, batch, l2_weight=1e-5, view_mask=None):
        if type(l2_weight) not in (int, float) or not math.isfinite(l2_weight) or l2_weight < 0:
            raise ValueError("l2_weight must be nonnegative finite")
        if not isinstance(batch, (tuple, list)) or len(batch) != 3:
            raise ValueError("batch must contain users, positive items and negative items")
        users = self._ids(batch[0], "users", self.graph.n_users)
        pos = self._ids(batch[1], "positive items", self.graph.n_items)
        neg = self._ids(batch[2], "negative items", self.graph.n_items)
        if users.numel() == 0 or users.shape != pos.shape or users.shape != neg.shape:
            raise ValueError("Loss needs nonempty identically shaped 1D IDs; broadcasting is forbidden")
        encoded = self.encode(view_mask)
        bpr = F.softplus(self.score(users, neg, encoded) - self.score(users, pos, encoded)).mean()
        l2 = sum(parameter.square().mean() for parameter in self.parameters())
        return {"total": bpr + l2_weight * l2, "bpr": bpr, "l2": l2}
