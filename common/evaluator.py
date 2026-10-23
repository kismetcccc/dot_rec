import math
import numpy as np
import torch
from common.dataset import split_coverage


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


def _metrics(metrics):
    if not isinstance(metrics, (tuple, list)) or not metrics or any(
            not isinstance(m, str) or m not in ("recall", "ndcg", "precision", "map") for m in metrics):
        raise ValueError("metrics must be a nonempty list of supported lowercase names")
    if len(set(metrics)) != len(metrics):
        raise ValueError("Duplicate metrics are not allowed")
    return tuple(metrics)


def _edges(edges, n_users, n_items):
    edges = np.asarray(edges)
    if edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype.kind not in "iu":
        raise ValueError("Edges must be integer [E,2]")
    if np.any(edges < 0) or np.any(edges[:, 0] >= n_users) or np.any(edges[:, 1] >= n_items):
        raise ValueError("Edge ID outside catalog")
    return edges


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


def ranking_metrics(ranked, users, positives, ks=(10, 20), metrics=("recall", "ndcg")):
    ks = _cutoffs(ks)
    metrics = _metrics(metrics)
    ranked = np.asarray(ranked)
    if ranked.ndim != 2 or len(ranked) != len(users):
        raise ValueError("ranked must have exactly one row per user")
    if ranked.dtype.kind not in "iu" or np.any(ranked < -1):
        raise ValueError("ranked must contain integer IDs or -1 padding")
    for row in ranked:
        valid = row[row >= 0]
        if len(np.unique(valid)) != len(valid):
            raise ValueError("Duplicate ranked IDs")
    for targets in positives.values():
        if any(isinstance(j, (bool, np.bool_)) or not isinstance(j, (int, np.integer)) or j < 0 for j in targets):
            raise ValueError("Targets must be nonnegative integer IDs")
    totals = {f"{metric}@{k}": 0.0 for k in ks for metric in metrics}
    count = 0
    for row, user in zip(ranked, users):
        targets = positives.get(int(user), set())
        if not targets:
            continue
        count += 1
        for k in ks:
            hits = [1.0 if j >= 0 and int(j) in targets else 0.0 for j in row[:k]]
            dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
            ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(targets))))
            cumulative, ap_sum = 0.0, 0.0
            for rank, hit in enumerate(hits, 1):
                cumulative += hit
                ap_sum += hit * cumulative / rank
            values = {"recall": sum(hits) / len(targets), "ndcg": dcg / ideal,
                      "precision": sum(hits) / k, "map": ap_sum / min(k, len(targets))}
            for metric in metrics:
                totals[f"{metric}@{k}"] += values[metric]
    if count == 0:
        raise ValueError("Evaluation requires at least one user with a positive")
    return {key: value / count for key, value in totals.items()}


@torch.no_grad()
def evaluate(model, target_edges, excluded_edges, ks=(10, 20), user_batch=128, item_chunk=4096,
             view_mask=None, population="all", return_report=False, metrics=("recall", "ndcg")):
    ks = _cutoffs(ks)
    metrics = _metrics(metrics)
    _positive_integer(user_batch, "user_batch")
    _positive_integer(item_chunk, "item_chunk")
    target_edges = _edges(target_edges, model.graph.n_users, model.graph.n_items)
    excluded_edges = _edges(excluded_edges, model.graph.n_users, model.graph.n_items)
    coverage = split_coverage(model.graph.user_degree.detach().cpu().numpy(),
                              model.graph.item_degree.detach().cpu().numpy(), target_edges, population)
    if population == "train_users_only":
        degree = model.graph.user_degree.detach().cpu().numpy()
        target_edges = target_edges[degree[target_edges[:, 0]] > 0]
    positives, excluded = histories(target_edges), histories(excluded_edges)
    if any(items & excluded.get(u, set()) for u, items in positives.items()):
        raise ValueError("Evaluation targets overlap masked history")
    users = sorted(positives)
    if not users:
        raise ValueError("Empty effective evaluation split")
    was_training = model.training
    model.eval()
    try:
        encoded = model.encode(view_mask=view_mask)  # fixed mask; fresh statistics, no dropout
        totals = {f"{m}@{k}": 0.0 for k in ks for m in metrics}
        for offset in range(0, len(users), user_batch):
            batch = users[offset:offset + user_batch]
            ranked = recommend(encoded, batch, excluded, max(ks), item_chunk)
            batch_metrics = ranking_metrics(ranked, batch, positives, ks, metrics)
            for key, value in batch_metrics.items():
                totals[key] += value * len(batch)
        metrics = {key: value / len(users) for key, value in totals.items()}
        return {"metrics": metrics, "coverage": coverage} if return_report else metrics
    finally:
        model.train(was_training)
