"""Robust PLCA label correction used by IST."""

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


def _knn_l2(features, neighbor_count, chunk_size):
    distances = []
    indices = []
    for start in range(0, features.shape[0], chunk_size):
        query = features[start : start + chunk_size]
        chunk_distances = torch.cdist(query, features, p=2).square()
        values, neighbors = torch.topk(
            chunk_distances,
            k=neighbor_count,
            dim=1,
            largest=False,
            sorted=True,
        )
        distances.append(values)
        indices.append(neighbors)
    return torch.cat(distances), torch.cat(indices)


def _normalized_affinity(features, k, gamma, chunk_size):
    sample_count = int(features.shape[0])
    # Official IST uses K+2 for L2 PLCA. Clamp only for short synthetic/early
    # streams where fewer graph nodes exist.
    neighbor_count = min(int(k) + 2, sample_count)
    distances, neighbors = _knn_l2(
        features, neighbor_count, int(chunk_size)
    )
    row_maximum = distances.amax(dim=1, keepdim=True).clamp_min(1.0e-8)
    relation = (1.0 - distances / row_maximum).clamp_min(0.0)
    relation[:, 0] = 1.0
    relation = relation.pow(float(gamma))

    rows = torch.arange(sample_count, device=features.device)
    rows = rows[:, None].expand_as(neighbors).reshape(-1)
    columns = neighbors.reshape(-1)
    directed = torch.sparse_coo_tensor(
        torch.stack((rows, columns)),
        relation.reshape(-1),
        (sample_count, sample_count),
        device=features.device,
    ).coalesce()
    symmetric = (directed + directed.transpose(0, 1)).coalesce()
    edge_indices = symmetric.indices()
    edge_values = symmetric.values()
    degree = torch.zeros(
        sample_count, device=features.device, dtype=features.dtype
    )
    degree.scatter_add_(0, edge_indices[0], edge_values)
    inverse_sqrt_degree = degree.clamp_min(1.0e-8).rsqrt()
    normalized_values = (
        edge_values
        * inverse_sqrt_degree[edge_indices[0]]
        * inverse_sqrt_degree[edge_indices[1]]
    )
    return torch.sparse_coo_tensor(
        edge_indices,
        normalized_values,
        symmetric.shape,
        device=features.device,
    ).coalesce()


def _conjugate_gradient(adjacency, right_hand_side, alpha, max_steps, tolerance):
    def matrix_multiply(value):
        return value - float(alpha) * torch.sparse.mm(adjacency, value)

    solution = torch.zeros_like(right_hand_side)
    residual = right_hand_side - matrix_multiply(solution)
    direction = residual.clone()
    residual_norm = torch.sum(residual * residual, dim=0)
    # Official IST calls scipy.sparse.linalg.cg(..., rtol=1e-6, atol=0).
    # Match that relative residual criterion instead of an absolute tolerance.
    rhs_norm = torch.linalg.vector_norm(right_hand_side, dim=0)
    stopping_threshold = float(tolerance) * rhs_norm
    for _ in range(int(max_steps)):
        product = matrix_multiply(direction)
        denominator = torch.sum(direction * product, dim=0).clamp_min(1.0e-12)
        step = residual_norm / denominator
        solution = solution + direction * step
        next_residual = residual - product * step
        next_norm = torch.sum(next_residual * next_residual, dim=0)
        if bool(
            torch.all(torch.sqrt(next_norm) <= stopping_threshold).item()
        ):
            residual = next_residual
            break
        coefficient = next_norm / residual_norm.clamp_min(1.0e-12)
        direction = next_residual + direction * coefficient
        residual = next_residual
        residual_norm = next_norm
    return solution


def robust_plca(current_features, pre_correction_soft_targets, memory, config):
    """Correct current labels using only current views and a past snapshot."""
    if not isinstance(memory, MemorySnapshot):
        raise TypeError("PLCA requires an immutable causal memory snapshot")
    features = current_features.detach().to(dtype=torch.float32)
    soft_targets = pre_correction_soft_targets.detach().to(
        device=features.device, dtype=torch.float32
    )
    memory_count = int(memory.batch_ids.numel())
    if memory_count:
        memory_features = memory.features.to(features.device)
        memory_labels = memory.labels.to(features.device)
        all_features = torch.cat((memory_features, features), dim=0)
        labels = torch.cat((memory_labels, soft_targets), dim=0)
    else:
        all_features = features
        labels = soft_targets
    if all_features.shape[0] != labels.shape[0]:
        raise ValueError("PLCA feature/label lengths do not match")
    if all_features.shape[0] == 0:
        raise ValueError("PLCA requires at least one current view")
    if config["mode"] != "l2":
        raise ValueError("P1 IST contract freezes PLCA mode=l2")

    adjacency = _normalized_affinity(
        all_features,
        k=int(config["k"]),
        gamma=float(config["gamma"]),
        chunk_size=int(config["distance_chunk_size"]),
    )
    propagated = labels
    for _ in range(int(config["repeat"])):
        normalized_labels = propagated / propagated.sum(
            dim=0, keepdim=True
        ).clamp_min(1.0e-8)
        scores = _conjugate_gradient(
            adjacency,
            normalized_labels,
            alpha=float(config["propagation_alpha"]),
            max_steps=int(config["solver_max_steps"]),
            tolerance=float(config["solver_tolerance"]),
        )
        hard = scores.argmax(dim=1)
        propagated = functional.one_hot(
            hard, num_classes=labels.shape[1]
        ).to(dtype=torch.float32)

    corrected = propagated[memory_count:]
    corrected_hard = corrected.argmax(dim=1)
    return PLCAResult(
        corrected_hard_labels=corrected_hard,
        corrected_one_hot=corrected,
        graph_sample_count=int(all_features.shape[0]),
        memory_sample_count=memory_count,
    )
