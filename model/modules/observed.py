"""Module 1: group-conditioned leave-one-out transport of observed content."""
import torch
from torch import nn
from torch.nn import functional as F
from utils.incidence import segment_sum, safe_normalize, safe_cosine


class ObservedTransport(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(2, 5, device="cpu", dtype=torch.float32))

    def forward(self, z, observed, behavior, graph, eta, learned=True):
        i, e = graph.item, graph.anchor
        degree = graph.item_degree.clamp_min(1)[:, None]
        normalized_b = safe_normalize(behavior)
        b_sum = segment_sum(normalized_b[i], e, graph.n_users)
        b_loo = (b_sum[e] - normalized_b[i]) / (graph.user_degree[e] - 1).clamp_min(1)[:, None]
        outputs, gates = [], []
        for m in range(2):
            obs = observed[:, m].to(z.dtype)
            zm = z[:, m]
            n = segment_sum(obs[i], e, graph.n_users)
            s = segment_sum(obs[i, None] * zm[i], e, graph.n_users)
            sq = zm.square().sum(-1)
            t = segment_sum(obs[i] * sq[i], e, graph.n_users)
            nlo = n[e] - obs[i]
            denom = nlo.clamp_min(1)
            mu = (s[e] - obs[i, None] * zm[i]) / denom[:, None]
            var = ((t[e] - obs[i] * sq[i]) / denom - mu.square().sum(-1)).clamp_min(0)
            eligible = observed[i, m] & (nlo > 0)
            if learned:
                th = self.theta[m]
                logits = (th[0] + F.softplus(th[1]) * safe_cosine(normalized_b[i], b_loo)
                          + F.softplus(th[2]) * safe_cosine(zm[i], mu)
                          - F.softplus(th[3]) * var + F.softplus(th[4]) * torch.log1p(nlo))
                gate = torch.where(eligible, torch.sigmoid(logits), torch.zeros_like(logits))
            else:
                gate = eligible.to(z.dtype)
            weighted = obs[i, None] * gate[:, None] * zm[i]
            group = segment_sum(weighted, e, graph.n_users)
            message = gate[:, None] * (group[e] - weighted) / denom[:, None]
            # Explicit guards guarantee exactly zero with no LOO donor.
            message = torch.where(eligible[:, None], message, torch.zeros_like(message))
            transported = obs[:, None] * segment_sum(message, i, graph.n_items) / degree
            outputs.append(obs[:, None] * zm + eta * transported)
            gates.append(gate)
        return torch.stack(outputs, 1), torch.stack(gates, 1)
