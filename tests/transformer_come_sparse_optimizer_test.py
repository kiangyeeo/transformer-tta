#!/usr/bin/env python3
"""Group/budget and strict masked AdamW contracts for COME (C11, C12, C17)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(PROJECT_ROOT), str(PROJECT_ROOT / "tests")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from transformer_come.config import (  # noqa: E402
    FORMAL_BUDGETS,
    GROUP_SIZE,
    MASK_SEEDS,
    TOTAL_GROUPS,
    budget_group_count,
)
from transformer_come.groups import (  # noqa: E402
    STRUCTURAL_GROUPS,
    build_masks,
    selected_group_ids,
)
from transformer_come.optimizer import (  # noqa: E402
    assert_off_mask_adam_state_zero,
    strict_masked_adamw_step,
)

from transformer_come_fixtures import candidate_tensors  # noqa: E402


def check_group_partition() -> dict:
    """C11: 6912 disjoint paired groups covering the 12 candidate tensors."""
    assert len(STRUCTURAL_GROUPS) == TOTAL_GROUPS == 6912
    assert [group.group_id for group in STRUCTURAL_GROUPS] == list(range(TOTAL_GROUPS))
    assert sum(group.kind == "QK" for group in STRUCTURAL_GROUPS) == 3 * 384
    assert sum(group.kind == "VO" for group in STRUCTURAL_GROUPS) == 3 * 384
    assert sum(group.kind == "FFN" for group in STRUCTURAL_GROUPS) == 3 * 1536

    candidates = candidate_tensors()
    paired = build_masks(candidates, [0, 384, 768])
    qkv = paired["blocks.9.attn.qkv.weight"]
    proj = paired["blocks.9.attn.proj.weight"]
    fc1 = paired["blocks.9.mlp.fc1.weight"]
    fc2 = paired["blocks.9.mlp.fc2.weight"]
    assert qkv[0, :].all() and qkv[384, :].all()  # Q row and K row
    assert qkv[768, :].all() and proj[:, 0].all()  # V row and O column
    assert fc1[0, :].all() and fc2[:, 0].all()  # FFN row and column
    assert sum(int(mask.count_nonzero()) for mask in paired.values()) == 3 * GROUP_SIZE

    coverage = build_masks(candidates, range(TOTAL_GROUPS))
    assert all(mask.all().item() for mask in coverage.values())
    assert sum(int(mask.count_nonzero()) for mask in coverage.values()) == 5_308_416
    return {"group_size": GROUP_SIZE, "total_groups": TOTAL_GROUPS}


def check_exact_integer_budget() -> dict:
    """C12: floor budgets 3/6/13 with matching paired scalar counts."""
    candidates = candidate_tensors()
    counts = {}
    for budget in FORMAL_BUDGETS:
        group_count = budget_group_count(budget)
        counts[str(budget)] = group_count
        for seed in MASK_SEEDS:
            group_ids = selected_group_ids(budget, seed)
            assert len(group_ids) == len(set(group_ids)) == group_count
            masks = build_masks(candidates, group_ids)
            active = sum(int(mask.count_nonzero()) for mask in masks.values())
            assert active == group_count * GROUP_SIZE
    assert counts == {"0.0005": 3, "0.001": 6, "0.002": 13}
    return {"integer_budgets": counts}


def check_strict_masked_adamw() -> dict:
    """C17: off-mask values exact; moments kept, cleared and never restored."""
    parameter = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    named = [("weight", parameter)]
    first_mask = {"weight": torch.tensor([[True, False], [False, False]])}
    second_mask = {"weight": torch.tensor([[False, False], [False, True]])}
    optimizer = torch.optim.AdamW([parameter], lr=0.1, betas=(0.9, 0.999), weight_decay=0.2)

    off_first = parameter.detach()[~first_mask["weight"]].clone()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        parameter.grad = torch.ones_like(parameter)
        strict_masked_adamw_step(optimizer, named, first_mask)
        # Exact equality, never allclose.
        assert torch.equal(parameter.detach()[~first_mask["weight"]], off_first)
        assert_off_mask_adam_state_zero(optimizer, named, first_mask)
    state = optimizer.state[parameter]
    kept = float(state["exp_avg"][0, 0].item())
    assert kept != 0.0
    # A continuously selected coordinate accumulates: 1-0.9^3 for grad=1.
    assert abs(kept - (1.0 - 0.9**3)) < 1e-6

    # Leaving the support clears that coordinate's moments and freezes it.
    frozen_value = parameter.detach()[0, 0].clone()
    optimizer.zero_grad(set_to_none=True)
    parameter.grad = torch.ones_like(parameter)
    strict_masked_adamw_step(optimizer, named, second_mask)
    assert torch.equal(parameter.detach()[0, 0], frozen_value)
    assert float(optimizer.state[parameter]["exp_avg"][0, 0].item()) == 0.0
    assert float(optimizer.state[parameter]["exp_avg_sq"][0, 0].item()) == 0.0

    # Re-entering the support must not resurrect the stale moment.
    optimizer.zero_grad(set_to_none=True)
    parameter.grad = torch.ones_like(parameter)
    strict_masked_adamw_step(optimizer, named, first_mask)
    restarted = float(optimizer.state[parameter]["exp_avg"][0, 0].item())
    assert abs(restarted - 0.1) < 1e-6
    assert restarted != kept
    return {"kept_moment": kept, "restarted_moment": restarted}


def check_masking_requires_gradients() -> None:
    parameter = nn.Parameter(torch.zeros(2, 2))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    masks = {"weight": torch.tensor([[True, False], [False, False]])}
    try:
        strict_masked_adamw_step(optimizer, [("weight", parameter)], masks)
    except RuntimeError as error:
        assert "gradient is missing" in str(error)
    else:
        raise AssertionError("A missing gradient was accepted")


def main() -> int:
    report = {"status": "PASS"}
    report.update(check_group_partition())
    report.update(check_exact_integer_budget())
    report.update(check_strict_masked_adamw())
    check_masking_requires_gradients()
    print(json.dumps(report, sort_keys=True))
    print("Transformer COME sparse optimizer tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
