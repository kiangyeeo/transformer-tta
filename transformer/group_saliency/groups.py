"""Dynamic first-order saliency scoring for paired DeiT-S structural groups."""

from __future__ import annotations

import hashlib
import json
from typing import Iterable

import torch

from transformer.group_random.groups import (
    BLOCKS,
    GROUP_KINDS,
    HIDDEN_DIM,
    STRUCTURAL_GROUPS,
    StructuralGroup,
    build_masks,
    mask_sha256,
)

from .config import GROUP_SIZE, TOTAL_GROUPS, budget_group_count


def compute_group_saliency_scores(named_parameters) -> torch.Tensor:
    """Return L2(|W*grad|) for all groups in canonical order on model device."""
    lookup = dict(named_parameters)
    if not lookup:
        raise ValueError("Saliency scoring requires candidate parameters")
    scores: list[torch.Tensor] = []
    for block in BLOCKS:
        prefix = f"blocks.{block}."
        qkv = lookup[prefix + "attn.qkv.weight"]
        proj = lookup[prefix + "attn.proj.weight"]
        fc1 = lookup[prefix + "mlp.fc1.weight"]
        fc2 = lookup[prefix + "mlp.fc2.weight"]
        for name, parameter in (
            (prefix + "attn.qkv.weight", qkv),
            (prefix + "attn.proj.weight", proj),
            (prefix + "mlp.fc1.weight", fc1),
            (prefix + "mlp.fc2.weight", fc2),
        ):
            if parameter.grad is None:
                raise RuntimeError(f"Candidate gradient is missing for {name}")
            if parameter.grad.shape != parameter.shape:
                raise RuntimeError(f"Candidate gradient shape differs for {name}")
        qkv_saliency = (qkv.detach() * qkv.grad.detach()).abs()
        proj_saliency = (proj.detach() * proj.grad.detach()).abs()
        fc1_saliency = (fc1.detach() * fc1.grad.detach()).abs()
        fc2_saliency = (fc2.detach() * fc2.grad.detach()).abs()
        q = qkv_saliency[:HIDDEN_DIM]
        k = qkv_saliency[HIDDEN_DIM : 2 * HIDDEN_DIM]
        v = qkv_saliency[2 * HIDDEN_DIM :]
        scores.append(torch.sqrt(q.square().sum(dim=1) + k.square().sum(dim=1)))
        scores.append(
            torch.sqrt(v.square().sum(dim=1) + proj_saliency.square().sum(dim=0))
        )
        scores.append(
            torch.sqrt(
                fc1_saliency.square().sum(dim=1)
                + fc2_saliency.square().sum(dim=0)
            )
        )
    result = torch.cat(scores).contiguous()
    if result.shape != (TOTAL_GROUPS,):
        raise RuntimeError(
            f"Expected {TOTAL_GROUPS} group scores, got {tuple(result.shape)}"
        )
    if not torch.isfinite(result).all():
        raise RuntimeError("Saliency group scores contain NaN or Inf")
    return result


def select_saliency_group_ids(scores: torch.Tensor, budget: float) -> tuple[int, ...]:
    """Select global exact-K groups; canonical group id resolves score ties."""
    value = scores.detach().reshape(-1)
    if value.numel() != TOTAL_GROUPS:
        raise ValueError(f"scores must contain exactly {TOTAL_GROUPS} values")
    if not torch.isfinite(value).all():
        raise ValueError("scores contain NaN or Inf")
    ranking = torch.argsort(value, descending=True, stable=True)
    count = budget_group_count(budget)
    return tuple(int(group_id) for group_id in ranking[:count].cpu().tolist())


def dynamic_mask_record(
    group_ids: Iterable[int], *, scores: torch.Tensor, budget: float
) -> dict:
    ids = tuple(int(value) for value in group_ids)
    expected = budget_group_count(budget)
    if len(ids) != expected or len(set(ids)) != len(ids):
        raise ValueError("Saliency support must contain exactly K unique groups")
    groups = [STRUCTURAL_GROUPS[value] for value in ids]
    score_cpu = scores.detach().to(device="cpu")
    selected_scores = [float(score_cpu[value].item()) for value in ids]
    ranking = torch.argsort(score_cpu, descending=True, stable=True)
    next_score = (
        float(score_cpu[int(ranking[expected])].item())
        if expected < TOTAL_GROUPS
        else None
    )
    return {
        "requested_group_count": expected,
        "realized_group_count": len(ids),
        "active_candidate_scalars": len(ids) * GROUP_SIZE,
        "mask_sha256": mask_sha256(ids),
        "top_k_min_score": min(selected_scores),
        "first_excluded_score": next_score,
        "selected_group_ids": list(ids),
        "selected_group_scores": selected_scores,
        "selected_by_kind": {
            kind: sum(group.kind == kind for group in groups) for kind in GROUP_KINDS
        },
        "selected_by_block": {
            str(block): sum(group.block == block for group in groups) for block in BLOCKS
        },
    }


def mask_history_sha256(mask_hashes: Iterable[str]) -> str:
    payload = json.dumps(list(mask_hashes), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "STRUCTURAL_GROUPS",
    "StructuralGroup",
    "build_masks",
    "compute_group_saliency_scores",
    "dynamic_mask_record",
    "mask_history_sha256",
    "select_saliency_group_ids",
]
