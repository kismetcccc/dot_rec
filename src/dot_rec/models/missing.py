"""Module 2: one-way transport from paired, currently observed donors."""
import torch
from torch import nn
from dot_rec.graph import segment_sum, safe_cosine


class MissingTransport(nn.Module):
    def forward(self, z, observed, graph, kappa, tau, mode="full"):
        i, e = graph.item, graph.anchor
        paired = observed.all(dim=1).to(z.dtype)
        k = segment_sum(paired[i], e, graph.n_users)
        # Same paired membership mask for both means, never completed values.
        sums = segment_sum(paired[i, None, None] * z[i], e, graph.n_users)
        mu = sums / k.clamp_min(1)[:, None, None]
        t = segment_sum(paired[i, None] * z[i].square().sum(-1), e, graph.n_users)
        var = (t / k.clamp_min(1)[:, None] - mu.square().sum(-1)).clamp_min(0)
        outputs, qualities = [], []
        for m in range(2):
            other = 1 - m
            eligible = ~observed[i, m] & observed[i, other] & (k[e] > 0)
            support = k[e] / (k[e] + kappa)
            consistency = torch.exp(-var[e, m] / tau)
            agreement = safe_cosine(z[i, other], mu[e, other]).clamp_min(0)
            if mode == "unit":
                q = torch.ones_like(support)
            elif mode == "no_support":
                q = consistency * agreement
            elif mode == "no_agreement":
                q = support * consistency
            elif mode == "full":
                q = support * consistency * agreement
            else:
                raise ValueError("Unknown quality ablation")
            q = torch.where(eligible, q, torch.zeros_like(q))
            c = segment_sum(q[:, None] * mu[e, m], i, graph.n_items)
            # Divide by TRAIN degree, NOT sum of q: retain attenuation.
            c = c / graph.item_degree.clamp_min(1)[:, None]
            outputs.append(c)
            qualities.append(q)
        return torch.stack(outputs, 1), torch.stack(qualities, 1)
