#!/usr/bin/env python3
"""Tiny deterministic math and parity checks for module_lbi."""

import copy
import json
import math
import os
import os.path as osp
import sys
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim
import yaml


PROJECT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
WORKSPACE_ROOT = osp.dirname(PROJECT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from core.lbi import SplitLBIEngine  # noqa: E402
from experiment_identity import build_experiment_identity  # noqa: E402
from shot_otta.config import load_yaml, resolve_effective_config  # noqa: E402
from shot_otta.trainer import (  # noqa: E402
    MODULE_CANDIDATE_NAMES,
    _collect_module_candidates,
    _configure_variant,
)
import shot_otta.trainer as trainer_module  # noqa: E402
from tools.plan_experiments import build_plan  # noqa: E402
from tools.summarize_runs import (  # noqa: E402
    build_summary_outputs,
    write_summary_outputs,
)


def _config(**overrides):
    config = {
        "alpha": 0.1,
        "kappa": 1.0,
        "nu": 1.0,
        "omega": 0.1,
        "stage1_max_steps": 4,
        "budget_tolerance": 0.0,
        "stage2_lr": 0.01,
        "stage2_steps": 1,
        "delta_nonzero_tolerance": 1.0e-12,
        "requested_budget": 0.5,
    }
    config.update(overrides)
    return config


def _linear_closure(named_parameters, gradients):
    lookup = dict(named_parameters)

    def closure():
        loss = sum(
            torch.sum(lookup[name] * gradient)
            for name, gradient in gradients.items()
        )
        return loss, {
            "loss_cls": 0.0,
            "loss_ent": 0.0,
            "loss_div": 0.0,
        }

    return closure


def _quadratic_closure(named_parameters, targets):
    lookup = dict(named_parameters)

    def closure():
        loss = sum(
            torch.sum((lookup[name] - targets[name]) ** 2)
            for name in lookup
        )
        return loss, {
            "loss_cls": float(loss.detach().item()),
            "loss_ent": 0.0,
            "loss_div": 0.0,
        }

    return closure


def _check_one_step_math():
    parameter = nn.Parameter(torch.tensor([1.0, -2.0]))
    named = [("weight", parameter)]
    gradients = {"weight": torch.tensor([2.0, -3.0])}
    config = _config(
        alpha=0.2,
        kappa=1.5,
        nu=2.0,
        requested_budget=0.5,
        stage1_max_steps=1,
    )
    result = SplitLBIEngine(parameter.numel()).run_step(
        named,
        _linear_closure(named, gradients),
        config,
    )
    expected_theta = -0.2 * 1.5 * gradients["weight"]
    expected_z = 0.2 * expected_theta / 2.0
    expected_gamma = (
        1.5
        * torch.sign(expected_z)
        * torch.clamp(expected_z.abs() - 1.0, min=0.0)
    )
    assert result.statistics["stage1_steps_completed"] == 1
    assert result.statistics["stage1_branch_enabled"] is True
    assert torch.allclose(
        result.state.theta_delta["weight"], expected_theta
    )
    assert torch.allclose(result.state.z["weight"], expected_z)
    assert torch.allclose(result.state.gamma["weight"], expected_gamma)
    assert result.statistics["stage1_support_count"] == 0
    assert result.statistics["selected_over_scope_ratio"] == 0.0
    assert result.statistics["selected_over_model_ratio"] == 0.0


def _check_candidate_scope():
    class TinyF(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature = nn.Linear(3, 3)

    class TinyB(nn.Module):
        def __init__(self):
            super().__init__()
            self.bottleneck = nn.Linear(3, 2)

    class TinyC(nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = nn.Linear(2, 2)

    models = (TinyF(), TinyB(), TinyC())
    all_named = [
        (f"{prefix}.{name}", parameter)
        for prefix, model in zip(("netF", "netB", "netC"), models)
        for name, parameter in model.named_parameters()
    ]
    candidates = _collect_module_candidates(all_named)
    assert {name for name, _ in candidates} == MODULE_CANDIDATE_NAMES
    assert [name for name, _ in candidates] == [
        "netB.bottleneck.weight",
        "netB.bottleneck.bias",
    ]


def _check_budget_and_rollback():
    parameter = nn.Parameter(torch.zeros(4))
    named = [("weight", parameter)]
    gradients = {
        "weight": torch.tensor([-2.0, -0.6, -0.6, -0.6])
    }
    result = SplitLBIEngine(8).run_step(
        named,
        _linear_closure(named, gradients),
        _config(
            alpha=1.0,
            requested_budget=0.5,
            stage1_max_steps=5,
            omega=1.0,
        ),
    )
    assert result.statistics["stage1_steps_completed"] == 2
    assert result.statistics["stage1_stop_reason"] == "rollback_feasible"
    assert result.statistics["stage1_rollback_used"] is True
    assert result.statistics["stage1_last_feasible_step"] == 1
    assert result.statistics["stage1_support_count"] == 1
    assert result.statistics["stage1_support_ratio"] == 0.25
    assert result.statistics["support_over_model_ratio"] == 0.125
    assert int(result.state.mask["weight"].count_nonzero()) == 1

    slow_parameter = nn.Parameter(torch.zeros(3))
    slow_named = [("weight", slow_parameter)]
    slow_result = SplitLBIEngine(3).run_step(
        slow_named,
        _linear_closure(
            slow_named, {"weight": torch.full((3,), -0.01)}
        ),
        _config(
            alpha=0.1,
            requested_budget=1.0,
            stage1_max_steps=2,
        ),
    )
    assert slow_result.statistics["stage1_steps_completed"] == 2
    assert (
        slow_result.statistics["stage1_stop_reason"]
        == "max_steps_reached"
    )
    assert slow_result.statistics["stage1_support_count"] == 0

    zero_budget_parameter = nn.Parameter(torch.tensor([1.0, -1.0]))
    zero_named = [("weight", zero_budget_parameter)]
    zero_closure_calls = {"count": 0}
    base_zero_closure = _linear_closure(
        zero_named, {"weight": torch.tensor([0.2, -0.3])}
    )

    def counted_zero_closure():
        zero_closure_calls["count"] += 1
        return base_zero_closure()

    zero_result = SplitLBIEngine(2).run_step(
        zero_named,
        counted_zero_closure,
        _config(
            requested_budget=0.0,
            omega=1.0,
            stage1_max_steps=4,
        ),
    )
    assert zero_closure_calls["count"] == 1
    assert zero_result.statistics["stage1_support_count"] == 0
    assert zero_result.statistics["stage1_branch_enabled"] is False
    assert zero_result.statistics["stage1_steps_completed"] == 0
    assert (
        zero_result.statistics["stage1_stop_reason"]
        == "branch_disabled"
    )
    assert zero_result.statistics["stage1_rollback_used"] is False
    assert zero_result.statistics["stage1_last_feasible_step"] == 0
    assert zero_result.statistics["stage1_feasible_found"] is True
    assert zero_result.statistics["stage1_final_loss"] is None
    for values in (
        zero_result.state.theta_delta,
        zero_result.state.gamma,
        zero_result.state.z,
        zero_result.state.mask,
    ):
        assert all(
            torch.count_nonzero(value).item() == 0
            for value in values.values()
        )
    assert (
        zero_result.statistics["effective_delta_nonzero_count"] > 0
    )
    assert not torch.equal(
        zero_result.applied_parameters["weight"],
        zero_result.base_parameters["weight"],
    )


def _check_stage2_and_stage3():
    parameter = nn.Parameter(torch.tensor([0.5, -0.5]))
    named = [("weight", parameter)]
    result = SplitLBIEngine(4).run_step(
        named,
        _linear_closure(
            named, {"weight": torch.tensor([0.2, -0.3])}
        ),
        _config(
            requested_budget=0.5,
            stage1_max_steps=1,
            omega=0.25,
        ),
    )
    assert result.statistics["stage2_steps_requested"] == 1
    assert result.statistics["stage2_steps_completed"] == 1
    assert result.statistics["stage2_optimizer"] == "sgd"
    assert int(result.state.mask["weight"].count_nonzero()) == 0
    assert torch.count_nonzero(parameter.grad).item() == 0
    assert torch.count_nonzero(
        result.state.theta_delta["weight"]
    ).item() > 0
    assert torch.count_nonzero(
        (
            result.refined_parameters["weight"]
            - result.base_parameters["weight"]
        ).abs()
        > 1.0e-12
    ).item() == result.statistics["effective_delta_nonzero_count"]
    expected_applied = (
        0.75 * result.base_parameters["weight"]
        + 0.25 * result.refined_parameters["weight"]
    )
    assert torch.allclose(result.applied_parameters["weight"], expected_applied)

    for omega in (0.0, 1.0):
        test_parameter = nn.Parameter(torch.tensor([0.3, -0.7]))
        test_named = [("weight", test_parameter)]
        test_result = SplitLBIEngine(2).run_step(
            test_named,
            _linear_closure(
                test_named, {"weight": torch.tensor([0.1, -0.2])}
            ),
            _config(requested_budget=0.0, omega=omega),
        )
        expected = (
            test_result.base_parameters["weight"]
            if omega == 0.0
            else test_result.refined_parameters["weight"]
        )
        assert torch.allclose(test_parameter, expected)
        if omega == 0.0:
            assert (
                test_result.statistics["applied_update_nonzero_count"]
                == 0
            )


def _legacy_reference(named_parameters, targets, config, total_count):
    """Small CPU harness transcribed from old split_lbi_otta.py."""
    named_parameters = list(named_parameters)
    lookup = dict(named_parameters)
    base = {
        name: parameter.detach().clone()
        for name, parameter in named_parameters
    }
    delta = {
        name: torch.zeros_like(parameter)
        for name, parameter in named_parameters
    }
    gamma = {
        name: torch.zeros_like(parameter)
        for name, parameter in named_parameters
    }
    z = {
        name: torch.zeros_like(parameter)
        for name, parameter in named_parameters
    }
    candidate_count = sum(
        parameter.numel() for _, parameter in named_parameters
    )
    # This mirrors the old FC branch flags rather than the new engine:
    # fc_branch_enabled=(fc_total>0) and (rho_max_fc>0.0).
    branch_enabled = (
        candidate_count > 0 and config["requested_budget"] > 0.0
    )
    branch_closed = not branch_enabled
    close_step = 0 if branch_closed else None
    close_reason = "branch_disabled" if branch_closed else None
    rollback_used = False
    feasible = None
    feasible_step = 0 if branch_closed else None
    steps = 0
    stage1_final_loss = None

    if branch_enabled:
        for step_index in range(config["stage1_max_steps"]):
            with torch.no_grad():
                for name, parameter in named_parameters:
                    parameter.add_(delta[name])
            loss = sum(
                torch.sum((lookup[name] - targets[name]) ** 2)
                for name in lookup
            )
            gradients = torch.autograd.grad(
                loss, [parameter for _, parameter in named_parameters]
            )
            stage1_final_loss = float(loss.item())
            with torch.no_grad():
                for name, parameter in named_parameters:
                    parameter.copy_(base[name])
            for (name, _), gradient in zip(
                named_parameters, gradients
            ):
                grad_reg = (
                    delta[name] - gamma[name]
                ) / config["nu"]
                delta[name] = delta[name] - (
                    config["alpha"]
                    * config["kappa"]
                    * (gradient + grad_reg)
                )
            for name, _ in named_parameters:
                grad_gamma = -(
                    delta[name] - gamma[name]
                ) / config["nu"]
                z[name] = z[name] - config["alpha"] * grad_gamma
            for name, _ in named_parameters:
                z_value = z[name]
                gamma[name] = (
                    config["kappa"]
                    * torch.sign(z_value)
                    * torch.clamp(
                        torch.abs(z_value) - 1.0, min=0.0
                    )
                )
            mask = {
                name: value.abs().gt(1.0e-4)
                for name, value in gamma.items()
            }
            support = sum(
                int(value.count_nonzero().item())
                for value in mask.values()
            )
            density = support / candidate_count
            steps = step_index + 1
            if density <= (
                config["requested_budget"]
                + config["budget_tolerance"]
            ):
                feasible = {
                    "delta": {
                        name: value.detach().clone()
                        for name, value in delta.items()
                    },
                    "gamma": {
                        name: value.detach().clone()
                        for name, value in gamma.items()
                    },
                    "z": {
                        name: value.detach().clone()
                        for name, value in z.items()
                    },
                }
                feasible_step = steps
            if density >= config["requested_budget"]:
                branch_closed = True
                close_step = steps
                close_reason = "cross_no_feasible"
                if feasible is not None:
                    delta = feasible["delta"]
                    gamma = feasible["gamma"]
                    z = feasible["z"]
                    close_reason = "rollback_feasible"
                    rollback_used = True
                break
        if not branch_closed:
            close_reason = "max_steps_reached"
            if feasible is not None:
                delta = feasible["delta"]
                gamma = feasible["gamma"]
                z = feasible["z"]
                rollback_used = True

    mask = {
        name: value.abs().gt(1.0e-4)
        for name, value in gamma.items()
    }
    with torch.no_grad():
        for name, parameter in named_parameters:
            parameter.copy_(base[name] + delta[name])
    groups = [
        {"params": [parameter], "lr": config["stage2_lr"]}
        for _, parameter in named_parameters
    ]
    optimizer = optim.SGD(
        groups,
        momentum=0.9,
        weight_decay=1.0e-3,
        nesterov=True,
    )
    for group in optimizer.param_groups:
        group["lr0"] = group["lr"]
    for stage2_index in range(config["stage2_steps"]):
        decay = (
            1.0
            + 10.0
            * (stage2_index + 1)
            / config["stage2_steps"]
        ) ** (-0.75)
        for group in optimizer.param_groups:
            group["lr"] = group["lr0"] * decay
            group["momentum"] = 0.9
            group["nesterov"] = True
        optimizer.zero_grad()
        loss = sum(
            torch.sum((lookup[name] - targets[name]) ** 2)
            for name in lookup
        )
        loss.backward()
        for name, parameter in named_parameters:
            parameter.grad.mul_(mask[name])
        optimizer.step()
    refined = {
        name: parameter.detach().clone()
        for name, parameter in named_parameters
    }
    applied = {
        name: (1.0 - config["omega"]) * base[name]
        + config["omega"] * refined[name]
        for name, _ in named_parameters
    }
    with torch.no_grad():
        for name, parameter in named_parameters:
            parameter.copy_(applied[name])
    support_count = sum(
        int(value.count_nonzero().item()) for value in mask.values()
    )
    support_over_scope_ratio = support_count / candidate_count
    support_over_model_ratio = support_count / total_count

    def change_statistics(values, prefix):
        tolerance = config["delta_nonzero_tolerance"]
        nonzero_count = 0
        l1 = 0.0
        squared_l2 = 0.0
        for name, value in values.items():
            difference = value - base[name]
            nonzero_count += int(
                difference.abs().gt(tolerance).count_nonzero().item()
            )
            l1 += float(difference.abs().sum().item())
            squared_l2 += float(
                torch.sum(difference * difference).item()
            )
        return {
            f"{prefix}_nonzero_count": nonzero_count,
            f"{prefix}_over_scope_ratio": (
                nonzero_count / candidate_count
            ),
            f"{prefix}_over_model_ratio": nonzero_count / total_count,
            f"{prefix}_l1": l1,
            f"{prefix}_l2": math.sqrt(squared_l2),
        }

    return {
        "branch_enabled": branch_enabled,
        "branch_closed": branch_closed,
        "steps": steps,
        "close_step": close_step,
        "close_reason": close_reason,
        "rollback_used": rollback_used,
        "feasible_step": feasible_step,
        "feasible_found": feasible is not None or not branch_enabled,
        "stage1_final_loss": stage1_final_loss,
        "delta": delta,
        "gamma": gamma,
        "z": z,
        "mask": mask,
        "support_count": support_count,
        "support_over_scope_ratio": support_over_scope_ratio,
        "support_over_model_ratio": support_over_model_ratio,
        "refined": refined,
        "applied": applied,
        **change_statistics(refined, "effective_delta"),
        **change_statistics(applied, "applied_update"),
        "total_count": total_count,
    }


def _check_old_parity():
    initial = {
        "weight": torch.tensor([0.2, -0.3, 0.4, -0.1]),
        "bias": torch.tensor([0.05, -0.02]),
    }
    targets = {
        "weight": torch.tensor([1.0, -0.5, -0.2, 0.3]),
        "bias": torch.tensor([0.4, -0.6]),
    }
    config = _config(
        alpha=0.7,
        kappa=1.0,
        nu=1.0,
        omega=0.3,
        requested_budget=0.5,
        stage1_max_steps=6,
        budget_tolerance=1.0e-4,
        stage2_steps=2,
    )
    new_named = [
        (name, nn.Parameter(value.clone()))
        for name, value in initial.items()
    ]
    old_named = [
        (name, nn.Parameter(value.clone()))
        for name, value in initial.items()
    ]
    total_count = 20
    new_result = SplitLBIEngine(total_count).run_step(
        new_named,
        _quadratic_closure(new_named, targets),
        config,
    )
    old_result = _legacy_reference(
        old_named, targets, config, total_count
    )
    assert (
        new_result.statistics["stage1_steps_completed"]
        == old_result["steps"]
    )
    assert new_result.statistics["stage1_branch_enabled"] is True
    assert old_result["branch_enabled"] is True
    assert (
        new_result.statistics["stage1_stop_reason"]
        == old_result["close_reason"]
    )
    assert (
        new_result.statistics["stage1_rollback_used"]
        == old_result["rollback_used"]
    )
    assert (
        new_result.statistics["stage1_last_feasible_step"]
        == old_result["feasible_step"]
    )
    for name in initial:
        assert torch.allclose(
            new_result.state.theta_delta[name],
            old_result["delta"][name],
            atol=1.0e-7,
            rtol=1.0e-6,
        )
        assert torch.allclose(
            new_result.state.z[name],
            old_result["z"][name],
            atol=1.0e-7,
            rtol=1.0e-6,
        )
        assert torch.allclose(
            new_result.state.gamma[name],
            old_result["gamma"][name],
            atol=1.0e-7,
            rtol=1.0e-6,
        )
        assert torch.equal(
            new_result.state.mask[name], old_result["mask"][name]
        )
        assert torch.allclose(
            new_result.refined_parameters[name],
            old_result["refined"][name],
            atol=1.0e-7,
            rtol=1.0e-6,
        )
        assert torch.allclose(
            new_result.applied_parameters[name],
            old_result["applied"][name],
            atol=1.0e-7,
            rtol=1.0e-6,
        )


def _check_zero_budget_old_parity():
    initial = {
        "weight": torch.tensor([0.8, -0.4, 0.2, -0.1]),
        "bias": torch.tensor([0.3, -0.6]),
    }
    targets = {
        "weight": torch.tensor([-1.0, 0.5, 0.7, -0.9]),
        "bias": torch.tensor([0.2, 0.4]),
    }
    config = _config(
        requested_budget=0.0,
        omega=0.4,
        stage1_max_steps=6,
        stage2_steps=2,
    )
    new_named = [
        (name, nn.Parameter(value.clone()))
        for name, value in initial.items()
    ]
    old_named = [
        (name, nn.Parameter(value.clone()))
        for name, value in initial.items()
    ]
    total_count = 20
    new_result = SplitLBIEngine(total_count).run_step(
        new_named,
        _quadratic_closure(new_named, targets),
        config,
    )
    old_result = _legacy_reference(
        old_named, targets, config, total_count
    )
    statistics = new_result.statistics
    assert statistics["stage1_branch_enabled"] is False
    assert old_result["branch_enabled"] is False
    assert statistics["stage1_steps_completed"] == old_result["steps"] == 0
    assert (
        statistics["stage1_stop_reason"]
        == old_result["close_reason"]
        == "branch_disabled"
    )
    assert statistics["stage1_rollback_used"] is False
    assert old_result["rollback_used"] is False
    assert statistics["stage1_last_feasible_step"] == 0
    assert statistics["stage1_feasible_found"] is True
    assert statistics["stage1_final_loss"] is None
    assert statistics["selected_param_count"] == 0
    assert statistics["selected_over_scope_ratio"] == 0.0
    assert statistics["selected_over_model_ratio"] == 0.0
    assert statistics["effective_delta_nonzero_count"] > 0

    for name in initial:
        for tensor_map in (
            new_result.state.theta_delta,
            new_result.state.gamma,
            new_result.state.z,
            new_result.state.mask,
        ):
            assert torch.count_nonzero(tensor_map[name]).item() == 0
        assert torch.equal(
            new_result.state.theta_delta[name],
            old_result["delta"][name],
        )
        assert torch.equal(
            new_result.state.gamma[name], old_result["gamma"][name]
        )
        assert torch.equal(new_result.state.z[name], old_result["z"][name])
        assert torch.equal(
            new_result.state.mask[name], old_result["mask"][name]
        )
        assert torch.allclose(
            new_result.refined_parameters[name],
            old_result["refined"][name],
            atol=1.0e-7,
            rtol=1.0e-6,
        )
        assert torch.allclose(
            new_result.applied_parameters[name],
            old_result["applied"][name],
            atol=1.0e-7,
            rtol=1.0e-6,
        )

    assert statistics["support_param_count"] == old_result["support_count"]
    assert (
        statistics["support_over_scope_ratio"]
        == old_result["support_over_scope_ratio"]
    )
    assert (
        statistics["support_over_model_ratio"]
        == old_result["support_over_model_ratio"]
    )
    for prefix in ("effective_delta", "applied_update"):
        for suffix in (
            "nonzero_count",
            "over_scope_ratio",
            "over_model_ratio",
            "l1",
            "l2",
        ):
            field = f"{prefix}_{suffix}"
            assert math.isclose(
                statistics[field],
                old_result[field],
                rel_tol=1.0e-6,
                abs_tol=1.0e-8,
            )


def _check_lifecycle_and_bn():
    class TinyF(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 3)
            self.bn = nn.BatchNorm1d(3)

        def forward(self, values):
            return self.bn(self.linear(values))

    class TinyB(nn.Module):
        def __init__(self):
            super().__init__()
            self.bottleneck = nn.Linear(3, 2)
            self.bn = nn.BatchNorm1d(2)

        def forward(self, values):
            return self.bn(self.bottleneck(values))

    class TinyC(nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = nn.Linear(2, 2)

        def forward(self, values):
            return self.classifier(values)

    net_f, net_b, net_c = TinyF(), TinyB(), TinyC()
    config = {
        "variant": "module_lbi",
        "requested_budget": 0.0,
        "selection_seed": None,
        "optimization": {
            "optimizer": "sgd",
            "lr": 0.01,
            "lr_decay1": 0.1,
            "lr_decay2": 1.0,
            "momentum": 0.9,
            "weight_decay": 0.001,
            "nesterov": True,
            "lr_gamma": 10.0,
            "lr_power": 0.75,
        },
        "lbi": {
            key: value
            for key, value in _config(
                requested_budget=0.0,
                stage1_max_steps=2,
            ).items()
            if key != "requested_budget"
        },
    }
    groups, selection, _ = _configure_variant(
        config, net_f, net_b, net_c
    )
    assert groups == []
    assert selection["bn_stats_frozen"] is True
    assert selection["lbi_state_lifecycle"] == "reset_every_online_step"
    assert all(
        not module.training
        for model in (net_f, net_b, net_c)
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    )
    candidates = [
        (f"netB.{name}", parameter)
        for name, parameter in net_b.named_parameters()
        if f"netB.{name}" in MODULE_CANDIDATE_NAMES
    ]
    total_count = sum(
        parameter.numel()
        for model in (net_f, net_b, net_c)
        for parameter in model.parameters()
    )
    engine = SplitLBIEngine(total_count)
    inputs = torch.randn(4, 3)
    bn_before = [
        (
            module.running_mean.clone(),
            module.running_var.clone(),
            module.num_batches_tracked.clone(),
        )
        for model in (net_f, net_b, net_c)
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]

    def closure():
        output = net_c(net_b(net_f(inputs)))
        loss = output.square().mean()
        return loss, {}

    base_before_first = {
        name: parameter.detach().clone()
        for name, parameter in candidates
    }
    first = engine.run_step(
        candidates,
        closure,
        {**config["lbi"], "requested_budget": 0.0},
    )
    second_start = {
        name: parameter.detach().clone()
        for name, parameter in candidates
    }
    second = engine.run_step(
        candidates,
        closure,
        {**config["lbi"], "requested_budget": 0.0},
    )
    assert not hasattr(engine, "optimizer")
    assert all(
        torch.count_nonzero(value).item() == 0
        for value in first.state.gamma.values()
    )
    assert all(
        torch.count_nonzero(value).item() == 0
        for value in second.state.gamma.values()
    )
    for name in second_start:
        assert torch.equal(second.base_parameters[name], second_start[name])
    assert any(
        not torch.equal(second_start[name], base_before_first[name])
        for name in second_start
    )
    bn_after = [
        (
            module.running_mean,
            module.running_var,
            module.num_batches_tracked,
        )
        for model in (net_f, net_b, net_c)
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    for before, after in zip(bn_before, bn_after):
        for old_value, new_value in zip(before, after):
            assert torch.equal(old_value, new_value)


def _check_identity_and_planner():
    base_path = osp.join(PROJECT_DIR, "configs", "shot_otta.yaml")
    base = load_yaml(base_path)
    base["variant"] = "module_lbi"
    base["requested_budget"] = 0.001
    effective = resolve_effective_config(base, WORKSPACE_ROOT)
    identity = build_experiment_identity(effective)
    changes = {
        "alpha": 0.2,
        "kappa": 2.0,
        "nu": 2.0,
        "omega": 0.2,
        "stage1_max_steps": 7,
        "budget_tolerance": 0.002,
        "stage2_lr": 0.02,
        "stage2_steps": 2,
        "delta_nonzero_tolerance": 1.0e-10,
    }
    for field, value in changes.items():
        changed = copy.deepcopy(effective)
        changed["lbi"][field] = value
        changed_identity = build_experiment_identity(changed)
        assert (
            identity["experiment_config_sha256"]
            != changed_identity["experiment_config_sha256"]
        )

    matrix = {
        "method": "shot",
        "task": "otta",
        "datasets": {"office": {"transfers": [[0, 1]]}},
        "seeds": [2020],
        "variants": {
            "module_lbi": {
                "budgets": [0.001],
                "lbi": copy.deepcopy(base["lbi"]),
            }
        },
        "common_training": {
            "batch_size": 4,
            "workers": 0,
            "gpu_id": "0",
            "save_model": False,
        },
        "output_root": "iclr2027/runs",
    }
    plan = build_plan(matrix, load_yaml(base_path), base_path)
    assert plan["experiment_count"] == 1
    command = plan["experiments"][0]["command_args"]
    assert "--requested-budget" in command
    assert "--lbi-alpha" in command
    assert "--lbi-stage1-max-steps" in command
    assert plan["experiments"][0]["variant"] == "module_lbi"

    for field, invalid_value in (
        ("alpha", 0.0),
        ("kappa", -1.0),
        ("nu", 0.0),
        ("omega", 1.1),
        ("stage1_max_steps", 0),
        ("budget_tolerance", -0.1),
        ("stage2_lr", 0.0),
        ("stage2_steps", 0),
        ("delta_nonzero_tolerance", -1.0),
    ):
        invalid_matrix = copy.deepcopy(matrix)
        invalid_matrix["variants"]["module_lbi"]["lbi"][
            field
        ] = invalid_value
        try:
            build_plan(
                invalid_matrix, load_yaml(base_path), base_path
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"planner accepted invalid module_lbi {field}"
            )


def _check_artifact_integration():
    class TinyF(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 3)
            self.bn = nn.BatchNorm1d(3)

        def forward(self, values):
            return self.bn(self.linear(values))

    class TinyB(nn.Module):
        def __init__(self):
            super().__init__()
            self.bottleneck = nn.Linear(3, 2)
            self.bn = nn.BatchNorm1d(2)

        def forward(self, values):
            return self.bn(self.bottleneck(values))

    class TinyC(nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = nn.Linear(2, 2)

        def forward(self, values):
            return self.classifier(values)

    base_path = osp.join(PROJECT_DIR, "configs", "shot_otta.yaml")
    config = load_yaml(base_path)
    config["variant"] = "module_lbi"
    config["requested_budget"] = 0.0
    config["data"]["workers"] = 0
    config["data"]["batch_size"] = 4
    config["lbi"]["stage1_max_steps"] = 2
    config = resolve_effective_config(config, WORKSPACE_ROOT)
    inputs = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10
    labels = torch.tensor([0, 1, 0, 1])
    batches = [(inputs, labels, torch.arange(4))]
    original_build_loaders = trainer_module.build_loaders
    original_load_models = trainer_module.load_source_models

    def fake_build_loaders(_config):
        return (
            {"target": batches, "test": batches},
            {"target_indices": [0, 1, 2, 3], "test_shuffle": False},
        )

    def fake_load_models(_config, _device):
        models = (TinyF(), TinyB(), TinyC())
        return models, {
            "netF": "fake/source_F.pt",
            "netB": "fake/source_B.pt",
            "netC": "fake/source_C.pt",
        }

    with tempfile.TemporaryDirectory(
        prefix="iclr2027_lbi_artifacts_"
    ) as temp_dir:
        config["output"]["root"] = temp_dir
        config["output"]["save_model"] = False
        trainer_module.build_loaders = fake_build_loaders
        trainer_module.load_source_models = fake_load_models
        try:
            summary = trainer_module.run_experiment(
                config, WORKSPACE_ROOT
            )
        finally:
            trainer_module.build_loaders = original_build_loaders
            trainer_module.load_source_models = original_load_models

        output_dir = summary["output_dir"]
        with open(
            osp.join(output_dir, "manifest.json"),
            "r",
            encoding="utf-8",
        ) as file_obj:
            manifest = json.load(file_obj)
        with open(
            osp.join(output_dir, "metrics.jsonl"),
            "r",
            encoding="utf-8",
        ) as file_obj:
            metrics = [json.loads(line) for line in file_obj]
        with open(
            osp.join(output_dir, "summary.json"),
            "r",
            encoding="utf-8",
        ) as file_obj:
            saved_summary = json.load(file_obj)

        for artifact in (manifest, metrics[0], saved_summary):
            assert artifact["variant"] == "module_lbi"
            assert artifact["selection"] == "lbi"
            assert artifact["lbi_alpha"] == config["lbi"]["alpha"]
            assert (
                artifact["lbi_state_lifecycle"]
                == "reset_every_online_step"
            )
            assert artifact["lbi_initialization"] == "dense"
            assert artifact["stage3_mode"] == "accumulation"
            assert artifact["stage1_branch_enabled"] is False
        online_metric = next(
            record for record in metrics if record["event"] == "online_step"
        )
        for field in (
            "stage1_branch_enabled",
            "target_support_count",
            "support_gap_count",
            "budget_reached",
            "max_steps_hit",
            "valid_lbi_step",
            "stage1_steps_completed",
            "stage1_stop_reason",
            "stage1_rollback_used",
            "stage1_last_feasible_step",
            "stage1_support_count",
            "stage1_support_ratio",
            "stage1_final_loss",
            "stage2_steps_completed",
            "stage2_final_loss",
            "selected_param_count",
            "effective_delta_nonzero_count",
            "effective_delta_l1",
            "effective_delta_l2",
            "applied_update_nonzero_count",
            "applied_update_l1",
            "applied_update_l2",
            "acc_post",
        ):
            assert field in online_metric
        assert online_metric["requested_budget"] == 0.0
        assert online_metric["target_support_count"] == 0
        assert online_metric["support_gap_count"] == 0
        assert online_metric["budget_reached"] is True
        assert online_metric["max_steps_hit"] is False
        assert online_metric["valid_lbi_step"] is True
        assert online_metric["stage1_branch_enabled"] is False
        assert online_metric["stage1_steps_completed"] == 0
        assert online_metric["stage1_stop_reason"] == "branch_disabled"
        assert online_metric["selected_param_count"] == 0
        assert online_metric["selected_over_scope_ratio"] == 0.0
        assert online_metric["selected_over_model_ratio"] == 0.0
        for field in (
            "target_support_count",
            "budget_hit_count",
            "budget_hit_rate",
            "budget_reached_all_steps",
            "valid_lbi_step_count",
            "valid_lbi_step_rate",
            "valid_lbi_run",
            "underfilled_step_count",
            "max_steps_hit_count",
            "max_steps_hit_rate",
            "stage1_stop_reason_counts",
            "support_gap_count_min",
            "support_gap_count_max",
            "stage1_support_count_min",
            "stage1_support_count_max",
            "selected_param_count_first",
            "selected_param_count_last",
            "selected_param_count_min",
            "selected_param_count_max",
            "selected_param_count_mean",
            "effective_delta_nonzero_count_mean",
            "applied_update_nonzero_count_mean",
            "stage1_steps_completed_mean",
            "stage2_steps_completed_mean",
            "PU-Acc",
            "FO-Acc",
            "runtime",
        ):
            assert field in saved_summary
        assert saved_summary["schema_version"] == 9
        assert saved_summary["budget_diagnostics_available"] is True
        assert (
            saved_summary["budget_diagnostics_source"]
            == "summary_json"
        )
        summary_outputs = build_summary_outputs(temp_dir)
        summary_dir = osp.join(temp_dir, "tables")
        write_summary_outputs(summary_outputs, summary_dir)
        for filename in ("all_runs.csv", "all_runs.md"):
            with open(
                osp.join(summary_dir, filename),
                "r",
                encoding="utf-8",
            ) as file_obj:
                table_text = file_obj.read()
            assert "module_lbi" in table_text
            assert str(config["lbi"]["alpha"]) in table_text
            assert "effective_delta_over_scope_ratio" in table_text


def main():
    torch.manual_seed(17)
    _check_one_step_math()
    _check_candidate_scope()
    _check_budget_and_rollback()
    _check_stage2_and_stage3()
    _check_old_parity()
    _check_zero_budget_old_parity()
    _check_lifecycle_and_bn()
    _check_identity_and_planner()
    _check_artifact_integration()
    print("iclr2027 module_lbi math/parity smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
