import math
import numpy as np
import torch
from dot_rec.data.dataset import split_coverage


def _positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _cutoffs(ks):
    ks = tuple(ks)
    if not ks:
        raise ValueError("At least one cutoff is required")
    for k in ks:
        _positive_integer(k, "cutoff")
    if len(set(ks)) != len(ks):
        raise ValueError("Duplicate cutoffs are not allowed")
    return ks


def histories(edges):
    result = {}
    for user, item in edges:
        result.setdefault(int(user), set()).add(int(item))
    return result


def _topk(scores, ids, k):
    # Stable exact ties: lower item ID first, independent of chunk size.
    order = torch.argsort(ids, dim=1, stable=True)
    scores, ids = scores.gather(1, order), ids.gather(1, order)
    order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :k]
    return scores.gather(1, order), ids.gather(1, order)


@torch.no_grad()
def recommend(encoded, users, excluded, k, item_chunk=4096):
    p, v = encoded
    _positive_integer(k, "k")
    _positive_integer(item_chunk, "item_chunk")
    users = torch.as_tensor(users, device=p.device)
    if users.ndim != 1 or users.dtype not in (torch.int32, torch.int64):
        raise ValueError("users must be a one-dimensional integer vector")
    if torch.any(users < 0) or torch.any(users >= len(p)):
        raise ValueError("User ID outside encoded catalog")
    users = users.to(dtype=torch.long)
    best_scores = p.new_empty((len(users), 0))
    best_ids = torch.empty((len(users), 0), dtype=torch.long, device=p.device)
    user_list = users.tolist()
    for start in range(0, len(v), item_chunk):
        stop = min(start + item_chunk, len(v))
        scores = p[users] @ v[start:stop].T
        if not torch.isfinite(scores).all():
            raise ValueError("Nonfinite scores before ranking mask")
        for row, user in enumerate(user_list):
            cols = [j - start for j in excluded.get(user, ()) if start <= j < stop]
            if cols:
                scores[row, cols] = -torch.inf
        ids = torch.arange(start, stop, device=p.device).expand(len(users), -1)
        best_scores, best_ids = _topk(torch.cat((best_scores, scores), 1), torch.cat((best_ids, ids), 1), k)
    # Never return excluded items if fewer than k candidates remain.
    return torch.where(torch.isfinite(best_scores), best_ids, -torch.ones_like(best_ids)).cpu().numpy()


def ranking_metrics(ranked, users, positives, ks=(10, 20)):
    ks = _cutoffs(ks)
    ranked = np.asarray(ranked)
    if ranked.ndim != 2 or len(ranked) != len(users):
        raise ValueError("ranked must have exactly one row per user")
    totals = {f"{metric}@{k}": 0.0 for k in ks for metric in ("recall", "ndcg")}
    count = 0
    for row, user in zip(ranked, users):
        targets = positives.get(int(user), set())
        if not targets:
            continue
        count += 1
        for k in ks:
            hits = [1.0 if int(j) in targets else 0.0 for j in row[:k]]
            totals[f"recall@{k}"] += sum(hits) / len(targets)
            dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
            ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(targets))))
            totals[f"ndcg@{k}"] += dcg / ideal
    if count == 0:
        raise ValueError("Evaluation requires at least one user with a positive")
    return {key: value / count for key, value in totals.items()}


@torch.no_grad()
def evaluate(model, target_edges, excluded_edges, ks=(10, 20), user_batch=128, item_chunk=4096,
             view_mask=None, population="all", return_report=False):
    ks = _cutoffs(ks)
    _positive_integer(user_batch, "user_batch")
    _positive_integer(item_chunk, "item_chunk")
    positives, excluded = histories(target_edges), histories(excluded_edges)
    if any(items & excluded.get(u, set()) for u, items in positives.items()):
        raise ValueError("Evaluation targets overlap masked history")
    users = sorted(positives)
    if not users:
        raise ValueError("Empty evaluation split")
    coverage = split_coverage(model.graph.user_degree.detach().cpu().numpy(),
                              model.graph.item_degree.detach().cpu().numpy(),
                              np.asarray(target_edges), population)
    was_training = model.training
    model.eval()
    try:
        encoded = model.encode(view_mask=view_mask)  # fixed mask; fresh statistics, no dropout
        totals = {f"{m}@{k}": 0.0 for k in ks for m in ("recall", "ndcg")}
        for offset in range(0, len(users), user_batch):
            batch = users[offset:offset + user_batch]
            ranked = recommend(encoded, batch, excluded, max(ks), item_chunk)
            metrics = ranking_metrics(ranked, batch, positives, ks)
            for key, value in metrics.items():
                totals[key] += value * len(batch)
        metrics = {key: value / len(users) for key, value in totals.items()}
        return {"metrics": metrics, "coverage": coverage} if return_report else metrics
    finally:
        model.train(was_training)
