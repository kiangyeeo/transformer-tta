#!/usr/bin/env python3
"""CPU-only Conv Group-LBI dynamics contracts (P2 D-F, H-M)."""

import os.path as osp
import sys

import torch
import torch.nn as nn


PROJECT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from core.lbi import SplitLBIEngine  # noqa: E402
from core.lbi.groups import group_lasso_prox, group_support_masks, selected_group_count  # noqa: E402
import core.lbi.engine as engine_module  # noqa: E402


def _config(**overrides):
    config = {
        "alpha": 1.0, "kappa": 1.0, "nu": 1.0, "omega": 0.25,
        "stage1_max_steps": 4, "budget_tolerance": 0.0,
        "stage2_lr": 0.1, "stage2_steps": 1,
        "delta_nonzero_tolerance": 1.0e-12, "support_threshold": 1.0e-4,
        "requested_budget": 1.0, "group_mode": "out_channel",
    }
    config.update(overrides)
    return config


def _linear_closure(named, gradient):
    parameter = dict(named)["weight"]

    def closure():
        return torch.sum(parameter * gradient), {}

    return closure


def _check_group_prox():
    for value in (torch.zeros(1, 1, 1, 1), torch.tensor([[[[0.5]]]]), torch.tensor([[[[1.0]]]])):
        output = group_lasso_prox(value, 2.0, "out_channel")
        assert torch.equal(output, torch.zeros_like(value))
        assert torch.isfinite(output).all()
    value = torch.tensor([[[[3.0]], [[4.0]]]])
    output = group_lasso_prox(value, 2.0, "out_channel")
    assert torch.allclose(output / torch.linalg.vector_norm(output), value / torch.linalg.vector_norm(value))
    assert torch.allclose(torch.linalg.vector_norm(output), torch.tensor(8.0))


def _check_filter_connection_group_prox():
    # Each (out, in) kernel is an independent group.  In particular, the
    # unit-norm kernel at [0, 1] must shrink to zero even though [0, 0] is
    # large enough that an incorrect out-channel aggregation would retain it.
    z_value = torch.tensor([
        [[[3.0, 4.0], [0.0, 0.0]], [[0.6, 0.8], [0.0, 0.0]]],
        [[[1.0, 2.0], [2.0, 2.0]], [[0.0, 0.0], [0.0, 0.0]]],
    ])
    output = group_lasso_prox(z_value, kappa=2.0, group_mode="filter_connection")
    expected = torch.zeros_like(z_value)
    expected[0, 0] = 2.0 * (1.0 - 1.0 / 5.0) * z_value[0, 0]
    expected[1, 0] = 2.0 * (1.0 - 1.0 / torch.sqrt(torch.tensor(13.0))) * z_value[1, 0]
    assert torch.allclose(output, expected)
    assert torch.equal(output[0, 1], torch.zeros_like(output[0, 1]))


def _check_old_state_update():
    parameter = nn.Parameter(torch.zeros(1, 1, 1, 1))
    named = [("weight", parameter)]
    result = SplitLBIEngine(1).run_step(
        named, _linear_closure(named, torch.tensor([[[[-3.0]]]])),
        _config(nu=2.0, stage1_max_steps=2),
    )
    # theta^1=3, gamma^1=0; c^1=1.5, so the old-state rule gives z^2=1.5.
    assert torch.allclose(result.state.z["weight"], torch.tensor([[[[1.5]]]]))
    assert not torch.allclose(result.state.z["weight"], torch.tensor([[[[2.25]]]]))


def _check_tau_support():
    tau = 1.0e-4
    gamma = torch.tensor([[[[tau - 1.0e-7]]], [[[tau]]], [[[tau + 1.0e-7]]]])
    mask = group_support_masks({"weight": gamma}, "out_channel", tau)["weight"]
    assert mask[:, 0, 0, 0].tolist() == [False, True, True]


def _mask_with_first_groups(count, parameter):
    group_mask = torch.zeros(parameter.shape[0], dtype=torch.bool)
    group_mask[:count] = True
    return group_mask.reshape(-1, 1, 1, 1).expand_as(parameter)


def _run_controlled_trajectory(trajectory, budget):
    parameter = nn.Parameter(torch.zeros(7, 1, 1, 1))
    named = [("weight", parameter)]
    final_count = (
        trajectory[-2] if trajectory[-1] > budget else trajectory[-1]
    )
    calls = iter([*trajectory, final_count])
    original = engine_module.group_support_masks

    def fake_masks(gamma, group_mode, support_threshold):
        del gamma, group_mode, support_threshold
        return {"weight": _mask_with_first_groups(next(calls), parameter)}

    engine_module.group_support_masks = fake_masks
    try:
        return SplitLBIEngine(parameter.numel()).run_step(
            named, _linear_closure(named, -torch.ones_like(parameter)),
            _config(requested_budget=budget / 7, stage1_max_steps=len(trajectory)),
        )
    finally:
        engine_module.group_support_masks = original


def _check_strict_rollback():
    # The sequence is the controllable thresholded-support trajectory at each
    # candidate state.  The fifth value is the final-mask re-evaluation.
    result = _run_controlled_trajectory([0, 2, 4, 7], budget=5)
    assert result.statistics["selected_group_count"] == 4
    assert result.statistics["stage1_rollback_used"] is True
    assert result.statistics["stage1_stop_reason"] == "strict_budget_rollback"
    assert result.statistics["selected_group_count"] <= 5
    exact = _run_controlled_trajectory([0, 2, 5], budget=5)
    assert exact.statistics["selected_group_count"] == 5
    assert exact.statistics["stage1_rollback_used"] is False
    assert exact.statistics["stage1_stop_reason"] == "budget_reached"


def _check_zero_group_budget():
    parameter = nn.Parameter(torch.ones(2, 1, 1, 1))
    named = [("weight", parameter)]
    calls = {"count": 0}
    base_closure = _linear_closure(named, -torch.ones_like(parameter))

    def counted_closure():
        calls["count"] += 1
        return base_closure()

    result = SplitLBIEngine(parameter.numel()).run_step(
        named, counted_closure, _config(requested_budget=0.1, stage1_max_steps=4),
    )
    assert calls["count"] == 1  # Stage 2 only; no Stage-1 iteration occurred.
    assert result.statistics["stage1_branch_enabled"] is False
    assert result.statistics["stage1_steps_completed"] == 0
    assert result.statistics["selected_group_count"] == 0
    assert result.statistics["stage1_rollback_used"] is False


def _check_masked_stage2_and_writeback_and_restart():
    parameter = nn.Parameter(torch.tensor([[[[2.0]]], [[[3.0]]]]))
    named = [("weight", parameter)]
    gradient = torch.tensor([[[[-4.0]]], [[[-1.0]]]])
    closure_calls = {"count": 0, "stage2_initial": None}

    def closure():
        closure_calls["count"] += 1
        if closure_calls["count"] <= 2:
            return torch.sum(parameter * gradient), {}
        closure_calls["stage2_initial"] = parameter.detach().clone()
        return torch.sum(parameter ** 2), {}

    result = SplitLBIEngine(parameter.numel()).run_step(
        named, closure, _config(stage1_max_steps=2, requested_budget=0.5, omega=0.25),
    )
    mask = result.state.mask["weight"]
    assert mask[:, 0, 0, 0].tolist() == [True, False]
    expected_initial = result.base_parameters["weight"] + mask.to(torch.float32) * result.state.theta_delta["weight"]
    assert torch.equal(closure_calls["stage2_initial"], expected_initial)
    assert closure_calls["stage2_initial"][1].item() == result.base_parameters["weight"][1].item()
    # Nesterov, momentum, and weight decay are enabled by the engine; restore
    # must nevertheless make the masked-out coordinate bitwise unchanged.
    assert result.refined_parameters["weight"][0].item() != expected_initial[0].item()
    assert result.refined_parameters["weight"][1].item() == expected_initial[1].item()
    expected_applied = (1.0 - 0.25) * result.base_parameters["weight"][0] + 0.25 * result.refined_parameters["weight"][0]
    assert torch.equal(result.applied_parameters["weight"][0], expected_applied)
    assert result.applied_parameters["weight"][1].item() == result.base_parameters["weight"][1].item()

    # A new online batch preserves the model but restarts all local LBI state.
    second = SplitLBIEngine(parameter.numel()).run_step(
        named, _linear_closure(named, torch.ones_like(parameter)),
        _config(requested_budget=0.1),
    )
    for values in (second.state.theta_delta, second.state.gamma, second.state.z):
        assert torch.count_nonzero(values["weight"]).item() == 0


def main():
    _check_group_prox()
    _check_filter_connection_group_prox()
    _check_old_state_update()
    _check_tau_support()
    _check_strict_rollback()
    _check_zero_group_budget()
    _check_masked_stage2_and_writeback_and_restart()
    print("conv LBI dynamics contracts passed")


if __name__ == "__main__":
    main()
