"""Architecture-aware paired groups and deterministic Random masks for DeiT-S."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable

import torch

from transformer.candidate_dense.config import candidate_parameter_names
from transformer.candidate_dense.model import EXPECTED_SHAPES

from .config import GROUP_SIZE, TOTAL_GROUPS, budget_group_count


HIDDEN_DIM = 384
MLP_DIM = 1536
BLOCKS = (9, 10, 11)
GROUP_KINDS = ("QK", "VO", "FFN")


@dataclass(frozen=True)
class StructuralGroup:
    group_id: int
    block: int
    kind: str
    coordinate: int


def structural_groups() -> tuple[StructuralGroup, ...]:
    groups: list[StructuralGroup] = []
    for block in BLOCKS:
        for kind, count in (("QK", HIDDEN_DIM), ("VO", HIDDEN_DIM), ("FFN", MLP_DIM)):
            for coordinate in range(count):
                groups.append(
                    StructuralGroup(len(groups), block, kind, coordinate)
                )
    if len(groups) != TOTAL_GROUPS:
        raise RuntimeError(f"Expected {TOTAL_GROUPS} groups, got {len(groups)}")
    return tuple(groups)


STRUCTURAL_GROUPS = structural_groups()


def selected_group_ids(budget: float, seed: int) -> tuple[int, ...]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    count = budget_group_count(budget)
    permutation = torch.randperm(TOTAL_GROUPS, generator=generator)
    return tuple(int(value) for value in permutation[:count].tolist())


def _validate_candidates(named_parameters) -> dict[str, torch.nn.Parameter]:
    lookup = dict(named_parameters)
    expected_names = set(candidate_parameter_names())
    if set(lookup) != expected_names:
        raise ValueError(
            "Candidate tensors differ from the frozen 12-tensor set: "
            f"{sorted(set(lookup) ^ expected_names)}"
        )
    for name, parameter in lookup.items():
        suffix = name.split(".", maxsplit=2)[2]
        if tuple(parameter.shape) != EXPECTED_SHAPES[suffix]:
            raise ValueError(f"Unexpected candidate shape for {name}: {tuple(parameter.shape)}")
    return lookup


def _apply_group(masks: dict[str, torch.Tensor], group: StructuralGroup) -> None:
    prefix = f"blocks.{group.block}."
    p = group.coordinate
    if group.kind == "QK":
        masks[prefix + "attn.qkv.weight"][p, :] = True
        masks[prefix + "attn.qkv.weight"][HIDDEN_DIM + p, :] = True
    elif group.kind == "VO":
        masks[prefix + "attn.qkv.weight"][2 * HIDDEN_DIM + p, :] = True
        masks[prefix + "attn.proj.weight"][:, p] = True
    elif group.kind == "FFN":
        masks[prefix + "mlp.fc1.weight"][p, :] = True
        masks[prefix + "mlp.fc2.weight"][:, p] = True
    else:
        raise ValueError(f"Unsupported group kind: {group.kind}")


def build_masks(named_parameters, group_ids: Iterable[int]) -> dict[str, torch.Tensor]:
    lookup = _validate_candidates(named_parameters)
    ids = tuple(int(value) for value in group_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("Selected structural group ids must be unique")
    if any(value < 0 or value >= TOTAL_GROUPS for value in ids):
        raise ValueError("Selected structural group id is outside [0, 6912)")
    masks = {
        name: torch.zeros_like(parameter, dtype=torch.bool, device=parameter.device)
        for name, parameter in lookup.items()
    }
    for group_id in ids:
        _apply_group(masks, STRUCTURAL_GROUPS[group_id])
    active_scalars = sum(int(mask.count_nonzero().item()) for mask in masks.values())
    expected = len(ids) * GROUP_SIZE
    if active_scalars != expected:
        raise RuntimeError(
            f"Structural masks contain {active_scalars} scalars, expected {expected}"
        )
    return masks


def mask_sha256(group_ids: Iterable[int]) -> str:
    payload = json.dumps(list(group_ids), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def mask_record(group_ids: Iterable[int], *, budget: float, seed: int) -> dict:
    ids = tuple(int(value) for value in group_ids)
    groups = [STRUCTURAL_GROUPS[value] for value in ids]
    by_kind = {kind: sum(group.kind == kind for group in groups) for kind in GROUP_KINDS}
    by_block = {str(block): sum(group.block == block for group in groups) for block in BLOCKS}
    return {
        "schema_version": 1,
        "selection": "uniform_structural_group_random",
        "requested_budget": float(budget),
        "requested_group_count": budget_group_count(budget),
        "realized_group_count": len(ids),
        "active_candidate_scalars": len(ids) * GROUP_SIZE,
        "mask_seed": int(seed),
        "mask_sha256": mask_sha256(ids),
        "selected_group_ids": list(ids),
        "selected_groups": [asdict(group) for group in groups],
        "selected_by_kind": by_kind,
        "selected_by_block": by_block,
    }


def hash_masked_values(named_parameters, masks, *, selected: bool) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(named_parameters):
        mask = masks[name]
        if not selected:
            mask = ~mask
        values = parameter.detach()[mask].cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(values.numpy().tobytes(order="C"))
    return digest.hexdigest()

