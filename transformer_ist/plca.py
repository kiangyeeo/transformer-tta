"""IST PLCA with exact L2 graph semantics and relative-CG stopping."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from .memory import MemorySnapshot


@dataclass(frozen=True)
class PLCAResult:
    corrected_hard_labels: torch.Tensor
    corrected_one_hot: torch.Tensor
    graph_sample_count: int
    memory_sample_count: int


def _knn_l2(features, neighbor_count: int, chunk_size: int):
    values, indices = [], []
    for start in range(0, features.shape[0], int(chunk_size)):
        distances = torch.cdist(
            features[start : start + chunk_size], features, p=2
        ).square()
        chunk_values, chunk_indices = torch.topk(
            distances, k=neighbor_count, dim=1, largest=False, sorted=True
        )
        values.append(chunk_values)
        indices.append(chunk_indices)
    return torch.cat(values), torch.cat(indices)


def _normalized_affinity(features, k: int, gamma: float, chunk_size: int):
    count = int(features.shape[0])
    distances, neighbors = _knn_l2(
        features, min(int(k) + 2, count), chunk_size
    )
    maximum = distances.amax(dim=1, keepdim=True).clamp_min(1.0e-8)
    relation = (1.0 - distances / maximum).clamp_min(0.0)
    relation[:, 0] = 1.0
    relation = relation.pow(float(gamma))
    rows = (
        torch.arange(count, device=features.device)[:, None]
        .expand_as(neighbors)
        .reshape(-1)
    )
    directed = torch.sparse_coo_tensor(
        torch.stack((rows, neighbors.reshape(-1))),
        relation.reshape(-1),
        (count, count),
        device=features.device,
    ).coalesce()
    symmetric = (directed + directed.transpose(0, 1)).coalesce()
    edge_indices, edge_values = symmetric.indices(), symmetric.values()
    degree = torch.zeros(count, device=features.device, dtype=features.dtype)
    degree.scatter_add_(0, edge_indices[0], edge_values)
    inv = degree.clamp_min(1.0e-8).rsqrt()
    normalized = edge_values * inv[edge_indices[0]] * inv[edge_indices[1]]
    return torch.sparse_coo_tensor(
        edge_indices, normalized, symmetric.shape, device=features.device
    ).coalesce()


def _conjugate_gradient(adjacency, rhs, alpha: float, max_steps: int, rtol: float):
    def multiply(value):
        return value - float(alpha) * torch.sparse.mm(adjacency, value)

    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual.clone()
    residual_norm = torch.sum(residual * residual, dim=0)
    threshold = float(rtol) * torch.linalg.vector_norm(rhs, dim=0)
    for _ in range(int(max_steps)):
        product = multiply(direction)
        denominator = torch.sum(direction * product, dim=0).clamp_min(1.0e-12)
        step = residual_norm / denominator
        solution = solution + direction * step
        next_residual = residual - product * step
        next_norm = torch.sum(next_residual * next_residual, dim=0)
        if bool(torch.all(torch.sqrt(next_norm) <= threshold).item()):
            break
        coefficient = next_norm / residual_norm.clamp_min(1.0e-12)
        direction = next_residual + direction * coefficient
        residual = next_residual
        residual_norm = next_norm
    return solution


def robust_plca(
    current_features, soft_targets, memory: MemorySnapshot, config
) -> PLCAResult:
    if not isinstance(memory, MemorySnapshot):
        raise TypeError("PLCA requires an immutable MemorySnapshot")
    if config["mode"] != "l2" or int(config["repeat"]) != 1:
        raise ValueError("IST Transformer freezes PLCA to l2/repeat=1")
    if float(config.get("solver_atol", 0.0)) != 0.0:
        raise ValueError("IST Transformer freezes PLCA solver_atol=0")
    features = current_features.detach().to(dtype=torch.float32)
    targets = soft_targets.detach().to(
        device=features.device, dtype=torch.float32
    )
    if features.ndim != 2 or targets.ndim != 2:
        raise ValueError("PLCA features and soft targets must both be matrices")
    if features.shape[0] == 0 or features.shape[0] != targets.shape[0]:
        raise ValueError("PLCA requires aligned non-empty current view inputs")
    if not torch.isfinite(features).all() or not torch.isfinite(targets).all():
        raise ValueError("PLCA inputs contain NaN or Inf")
    memory_count = int(memory.batch_ids.numel())
    if memory_count:
        features = torch.cat((memory.features.to(features.device), features), dim=0)
        targets = torch.cat((memory.labels.to(features.device), targets), dim=0)
    adjacency = _normalized_affinity(
        features,
        int(config["k"]),
        float(config["gamma"]),
        int(config["distance_chunk_size"]),
    )
    normalized = targets / targets.sum(dim=0, keepdim=True).clamp_min(1.0e-8)
    scores = _conjugate_gradient(
        adjacency,
        normalized,
        float(config["propagation_alpha"]),
        int(config["solver_max_steps"]),
        float(config["solver_rtol"]),
    )
    hard = scores.argmax(dim=1)
    corrected = functional.one_hot(
        hard, num_classes=targets.shape[1]
    ).float()[memory_count:]
    return PLCAResult(
        corrected.argmax(dim=1),
        corrected,
        int(features.shape[0]),
        memory_count,
    )
