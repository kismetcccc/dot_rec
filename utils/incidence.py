"""One binary TRAIN support, shared by CF and both transport modules."""
import torch
from torch import nn


def segment_sum(values, index, size):
    return values.new_zeros((size,) + values.shape[1:]).index_add(0, index, values)


def safe_normalize(x):
    require_full_precision(x)
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return x / norm.clamp_min(1e-12)


def require_full_precision(*tensors):
    # No AMP/half claim: epsilon 1e-12 is only used in float32/float64.
    try:
        autocast = torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda")
    except TypeError:  # PyTorch versions before the device_type overload
        autocast = torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled()
    if autocast or any(x.dtype not in (torch.float32, torch.float64) for x in tensors):
        raise ValueError("BAST supports float32/float64 only; half, bfloat16 and AMP are unsupported")


def safe_cosine(x, y):
    return (safe_normalize(x) * safe_normalize(y)).sum(-1).clamp(-1, 1)


class Incidence(nn.Module):
    def __init__(self, n_users, n_items, train_edges):
        super().__init__()
        if type(n_users) is not int or type(n_items) is not int or n_users <= 0 or n_items <= 0:
            raise ValueError("Catalog dimensions must be positive integers, not booleans")
        edges = torch.as_tensor(train_edges, device="cpu")
        if edges.dtype not in (torch.int32, torch.int64) or edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("TRAIN edges must be integer [E,2]")
        edges = edges.to(dtype=torch.long)
        if edges.numel() and ((edges < 0).any() or (edges[:, 0] >= n_users).any() or (edges[:, 1] >= n_items).any()):
            raise ValueError("TRAIN ID out of range")
        edges = torch.unique(edges, dim=0, sorted=True)
        self.n_users, self.n_items = n_users, n_items
        self.register_buffer("anchor", edges[:, 0])
        self.register_buffer("item", edges[:, 1])
        self.register_buffer("user_degree", torch.bincount(self.anchor, minlength=n_users).float())
        self.register_buffer("item_degree", torch.bincount(self.item, minlength=n_items).float())
        scale = (self.user_degree[self.anchor] * self.item_degree[self.item]).rsqrt()
        self.register_buffer("cf_scale", scale)

    def collaborative(self, users, items, layers):
        require_full_precision(users, items)
        if type(layers) is not int or layers < 0:
            raise ValueError("layers must be a nonnegative integer")
        if users.device != self.anchor.device or items.device != self.anchor.device or users.dtype != items.dtype:
            raise ValueError("CF inputs and graph must share a device; embeddings must share a dtype")
        # Zero-degree IDs are zero even in layer 0, not random cold-start signal.
        u = users * (self.user_degree > 0).unsqueeze(1)
        b = items * (self.item_degree > 0).unsqueeze(1)
        us, bs = u, b
        for _ in range(layers):
            next_u = segment_sum(b[self.item] * self.cf_scale[:, None], self.anchor, self.n_users)
            next_b = segment_sum(u[self.anchor] * self.cf_scale[:, None], self.item, self.n_items)
            u, b = next_u, next_b
            us, bs = us + u, bs + b
        return us / (layers + 1), bs / (layers + 1)
