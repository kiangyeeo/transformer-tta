"""Source-W0 paired-group L2 ranking for DeiT-S structural groups."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Iterable

import torch

from transformer.group_random.groups import (
    BLOCKS,
    GROUP_KINDS,
    HIDDEN_DIM,
    STRUCTURAL_GROUPS,
    StructuralGroup,
    build_masks,
    hash_masked_values,
    mask_sha256,
)

from .config import GROUP_SIZE, TOTAL_GROUPS, budget_group_count


def compute_group_l2_scores(named_parameters) -> torch.Tensor:
    """Compute all 6912 paired-group L2 scores on CPU in float64."""
    lookup = {name: parameter.detach().to(device="cpu", dtype=torch.float64) for name, parameter in named_parameters}
    scores: list[torch.Tensor] = []
    for block in BLOCKS:
        prefix = f"blocks.{block}."
        qkv = lookup[prefix + "attn.qkv.weight"]
        proj = lookup[prefix + "attn.proj.weight"]
        fc1 = lookup[prefix + "mlp.fc1.weight"]
        fc2 = lookup[prefix + "mlp.fc2.weight"]
        q = qkv[:HIDDEN_DIM]
        k = qkv[HIDDEN_DIM : 2 * HIDDEN_DIM]
        v = qkv[2 * HIDDEN_DIM :]
        scores.append(torch.sqrt(q.square().sum(dim=1) + k.square().sum(dim=1)))
        scores.append(torch.sqrt(v.square().sum(dim=1) + proj.square().sum(dim=0)))
        scores.append(torch.sqrt(fc1.square().sum(dim=1) + fc2.square().sum(dim=0)))
    result = torch.cat(scores).contiguous()
    if result.shape != (TOTAL_GROUPS,):
        raise RuntimeError(f"Expected {TOTAL_GROUPS} group scores, got {tuple(result.shape)}")
    if not torch.isfinite(result).all():
        raise RuntimeError("Magnitude group scores contain NaN or Inf")
    return result


def score_vector_sha256(scores: torch.Tensor) -> str:
    value = scores.detach().to(device="cpu", dtype=torch.float64).contiguous()
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()


def select_magnitude_group_ids(scores: torch.Tensor, budget: float) -> tuple[int, ...]:
    """Stable descending top-K; canonical group id resolves exact ties."""
    value = scores.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if value.numel() != TOTAL_GROUPS:
        raise ValueError(f"scores must contain exactly {TOTAL_GROUPS} values")
    if not torch.isfinite(value).all():
        raise ValueError("scores contain NaN or Inf")
    ranking = torch.argsort(value, descending=True, stable=True)
    count = budget_group_count(budget)
    return tuple(int(group_id) for group_id in ranking[:count].tolist())


def magnitude_mask_record(
    group_ids: Iterable[int],
    *,
    scores: torch.Tensor,
    budget: float,
    checkpoint_sha256: str,
) -> dict:
    ids = tuple(int(value) for value in group_ids)
    if len(ids) != budget_group_count(budget) or len(set(ids)) != len(ids):
        raise ValueError("Magnitude support must contain exactly K unique groups")
    groups = [STRUCTURAL_GROUPS[value] for value in ids]
    selected_scores = [float(scores[value].item()) for value in ids]
    ranking = torch.argsort(scores, descending=True, stable=True)
    k = len(ids)
    next_score = float(scores[int(ranking[k])].item()) if k < TOTAL_GROUPS else None
    return {
        "schema_version": 1,
        "selection": "source_w0_structural_group_l2",
        "score": "paired_group_l2_norm",
        "score_device": "cpu",
        "score_dtype": "float64",
        "tie_break": "ascending_canonical_group_id",
        "ranking_source": "source_checkpoint_pre_adaptation",
        "source_checkpoint_sha256": checkpoint_sha256,
        "requested_budget": float(budget),
        "requested_group_count": budget_group_count(budget),
        "realized_group_count": len(ids),
        "active_candidate_scalars": len(ids) * GROUP_SIZE,
        "score_vector_sha256": score_vector_sha256(scores),
        "mask_sha256": mask_sha256(ids),
        "top_k_min_score": min(selected_scores),
        "first_excluded_score": next_score,
        "selected_group_ids": list(ids),
        "selected_group_scores": selected_scores,
        "selected_groups": [asdict(group) for group in groups],
        "selected_by_kind": {
            kind: sum(group.kind == kind for group in groups) for kind in GROUP_KINDS
        },
        "selected_by_block": {
            str(block): sum(group.block == block for group in groups) for block in BLOCKS
        },
    }


__all__ = [
    "STRUCTURAL_GROUPS",
    "StructuralGroup",
    "build_masks",
    "compute_group_l2_scores",
    "hash_masked_values",
    "magnitude_mask_record",
    "score_vector_sha256",
    "select_magnitude_group_ids",
]

