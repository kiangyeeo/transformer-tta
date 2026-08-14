"""Split-LBI math migrated from nips2026/SHOT-OTTA-LBI.

The engine owns no dataset, model architecture, evaluation, or YAML logic.
Every call creates a fresh local trajectory and a fresh Stage-2 optimizer.
Only the Stage-3 parameter values written back to the supplied Parameters
survive into the next online batch.
"""

import math

import torch
import torch.optim as optim

from .diagnostics import compute_lbi_step_budget_diagnostics
from .state import LBIResult, LBIState


SUPPORT_THRESHOLD = 1.0e-4
STAGE2_OPTIMIZER = "sgd"
STAGE2_MOMENTUM = 0.9
STAGE2_WEIGHT_DECAY = 1.0e-3
STAGE2_NESTEROV = True
STAGE2_LR_GAMMA = 10.0
STAGE2_LR_POWER = 0.75


def _loss_and_parts(loss_output):
    if isinstance(loss_output, tuple):
        loss_value, parts = loss_output
        return loss_value, dict(parts)
    return loss_output, {}


def _clone_map(named_parameters):
    return {
        name: parameter.detach().clone()
        for name, parameter in named_parameters
    }


def _copy_map(named_parameters, values):
    with torch.no_grad():
        for name, parameter in named_parameters:
            parameter.copy_(values[name])


def _support_masks(gamma):
    return {
        name: value.detach().abs().gt(SUPPORT_THRESHOLD)
        for name, value in gamma.items()
    }


def _support_count(masks):
    return sum(int(mask.count_nonzero().item()) for mask in masks.values())


def _change_statistics(
    values,
    base_values,
    tolerance,
    candidate_count,
    total_model_param_count,
    prefix,
):
    nonzero_count = 0
    l1 = 0.0
    squared_l2 = 0.0
    for name, value in values.items():
        difference = value.detach() - base_values[name]
        nonzero_count += int(
            difference.abs().gt(tolerance).count_nonzero().item()
        )
        l1 += float(difference.abs().sum().item())
        squared_l2 += float(torch.sum(difference * difference).item())
    return {
        f"{prefix}_nonzero_count": nonzero_count,
        f"{prefix}_over_scope_ratio": (
            float(nonzero_count / candidate_count)
            if candidate_count > 0
            else 0.0
        ),
        f"{prefix}_over_model_ratio": (
            float(nonzero_count / total_model_param_count)
            if total_model_param_count > 0
            else 0.0
        ),
        f"{prefix}_l1": l1,
        f"{prefix}_l2": math.sqrt(squared_l2),
    }


class SplitLBIEngine:
    """Run one reset-per-online-step element-wise Split-LBI trajectory."""

    def __init__(self, total_model_param_count):
        self.total_model_param_count = int(total_model_param_count)

    @staticmethod
    def initialize(candidate_parameters):
        zeros = {
            name: torch.zeros_like(parameter)
            for name, parameter in candidate_parameters
        }
        return LBIState(
            theta_delta={
                name: value.detach().clone() for name, value in zeros.items()
            },
            gamma={
                name: value.detach().clone() for name, value in zeros.items()
            },
            z={
                name: value.detach().clone() for name, value in zeros.items()
            },
            mask={
                name: torch.zeros_like(value, dtype=torch.bool)
                for name, value in zeros.items()
            },
        )

    @staticmethod
    def element_prox(z_value, kappa):
        return (
            kappa
            * torch.sign(z_value)
            * torch.clamp(torch.abs(z_value) - 1.0, min=0.0)
        )

    @staticmethod
    def _validate(candidate_parameters, config):
        if not candidate_parameters:
            raise ValueError("SplitLBIEngine requires candidate parameters")
        names = [name for name, _ in candidate_parameters]
        if len(names) != len(set(names)):
            raise ValueError("SplitLBIEngine candidate names must be unique")
        required = {
            "alpha",
            "kappa",
            "nu",
            "omega",
            "stage1_max_steps",
            "budget_tolerance",
            "stage2_lr",
            "stage2_steps",
            "delta_nonzero_tolerance",
            "requested_budget",
        }
        missing = sorted(required - set(config))
        if missing:
            raise ValueError(f"Missing Split-LBI config fields: {missing}")
        for key in (
            "alpha",
            "kappa",
            "nu",
            "omega",
            "budget_tolerance",
            "stage2_lr",
            "delta_nonzero_tolerance",
            "requested_budget",
        ):
            if not math.isfinite(float(config[key])):
                raise ValueError(f"Split-LBI {key} must be finite")
        if float(config["alpha"]) <= 0:
            raise ValueError("Split-LBI alpha must be > 0")
        if float(config["kappa"]) <= 0:
            raise ValueError("Split-LBI kappa must be > 0")
        if float(config["nu"]) <= 0:
            raise ValueError("Split-LBI nu must be > 0")
        if not 0.0 <= float(config["omega"]) <= 1.0:
            raise ValueError("Split-LBI omega must be in [0, 1]")
        if not 0.0 <= float(config["requested_budget"]) <= 1.0:
            raise ValueError(
                "Split-LBI requested_budget must be in [0, 1]"
            )
        if float(config["budget_tolerance"]) < 0:
            raise ValueError(
                "Split-LBI budget_tolerance must be >= 0"
            )
        if float(config["stage2_lr"]) <= 0:
            raise ValueError("Split-LBI stage2_lr must be > 0")
        if float(config["delta_nonzero_tolerance"]) < 0:
            raise ValueError(
                "Split-LBI delta_nonzero_tolerance must be >= 0"
            )
        for key in ("stage1_max_steps", "stage2_steps"):
            value = config[key]
            if (
                isinstance(value, bool)
                or int(value) != value
                or int(value) <= 0
            ):
                raise ValueError(
                    f"Split-LBI {key} must be a positive integer"
                )

    def run_step(self, candidate_parameters, loss_closure, config):
        """Run Stage 1, Stage 2, and Stage 3 on the current online batch."""
        candidate_parameters = list(candidate_parameters)
        self._validate(candidate_parameters, config)
        parameter_lookup = dict(candidate_parameters)
        candidate_count = sum(
            parameter.numel() for _, parameter in candidate_parameters
        )
        base_parameters = _clone_map(candidate_parameters)
        state = self.initialize(candidate_parameters)

        alpha = float(config["alpha"])
        kappa = float(config["kappa"])
        nu = float(config["nu"])
        requested_budget = float(config["requested_budget"])
        budget_target = requested_budget + float(
            config["budget_tolerance"]
        )
        max_steps = int(config["stage1_max_steps"])
        stage1_branch_enabled = requested_budget > 0.0

        feasible_state = state.clone() if not stage1_branch_enabled else None
        feasible_step = 0 if not stage1_branch_enabled else None
        stage1_steps_completed = 0
        stage1_final_loss = None
        stage1_stop_reason = (
            "max_steps_reached"
            if stage1_branch_enabled
            else "branch_disabled"
        )
        stage1_rollback_used = False

        stage1_iterations = (
            range(max_steps) if stage1_branch_enabled else ()
        )
        for step_index in stage1_iterations:
            # Old Stage 1 evaluates the SHOT loss at base + theta_delta.
            with torch.no_grad():
                for name, parameter in candidate_parameters:
                    parameter.add_(state.theta_delta[name])
            try:
                stage1_loss, _ = _loss_and_parts(loss_closure())
                gradients = torch.autograd.grad(
                    stage1_loss,
                    [parameter for _, parameter in candidate_parameters],
                )
            finally:
                _copy_map(candidate_parameters, base_parameters)

            stage1_final_loss = float(stage1_loss.item())
            stage1_steps_completed = step_index + 1

            # Exact old update order:
            # theta_delta <- theta_delta - alpha*kappa*(grad + coupling)
            for (name, _), gradient in zip(
                candidate_parameters, gradients
            ):
                coupling = (
                    state.theta_delta[name] - state.gamma[name]
                ) / nu
                state.theta_delta[name] = (
                    state.theta_delta[name]
                    - alpha * kappa * (gradient + coupling)
                )

            # z <- z + alpha*(theta_delta-gamma)/nu
            for name, _ in candidate_parameters:
                state.z[name] = state.z[name] + alpha * (
                    state.theta_delta[name] - state.gamma[name]
                ) / nu

            # gamma <- kappa*sign(z)*max(abs(z)-1, 0)
            for name, _ in candidate_parameters:
                state.gamma[name] = self.element_prox(
                    state.z[name], kappa
                )

            state.mask = _support_masks(state.gamma)
            support_count = _support_count(state.mask)
            support_ratio = (
                float(support_count / candidate_count)
                if candidate_count > 0
                else 0.0
            )
            if support_ratio <= budget_target:
                feasible_state = state.clone()
                feasible_step = stage1_steps_completed

            if support_ratio >= requested_budget:
                stage1_stop_reason = "cross_no_feasible"
                if feasible_state is not None:
                    state = feasible_state.clone()
                    stage1_stop_reason = "rollback_feasible"
                    stage1_rollback_used = True
                break
        else:
            if stage1_branch_enabled and feasible_state is not None:
                state = feasible_state.clone()
                stage1_rollback_used = True

        state.mask = _support_masks(state.gamma)
        support_count = _support_count(state.mask)
        support_ratio = (
            float(support_count / candidate_count)
            if candidate_count > 0
            else 0.0
        )

        # Dense initialization: retain every Stage-1 theta_delta element.
        dense_initial_parameters = {
            name: base_parameters[name] + state.theta_delta[name]
            for name, _ in candidate_parameters
        }
        _copy_map(candidate_parameters, dense_initial_parameters)

        # The local optimizer and its state are discarded when this call ends.
        stage2_lr = float(config["stage2_lr"])
        stage2_steps_requested = int(config["stage2_steps"])
        parameter_groups = [
            {"params": [parameter], "lr": stage2_lr}
            for _, parameter in candidate_parameters
        ]
        stage2_optimizer = optim.SGD(
            parameter_groups,
            momentum=STAGE2_MOMENTUM,
            weight_decay=STAGE2_WEIGHT_DECAY,
            nesterov=STAGE2_NESTEROV,
        )
        for group in stage2_optimizer.param_groups:
            group["lr0"] = group["lr"]

        stage2_final_loss = None
        stage2_loss_parts = {}
        stage2_steps_completed = 0
        stage2_effective_lr = stage2_lr
        for stage2_index in range(stage2_steps_requested):
            decay = (
                1.0
                + STAGE2_LR_GAMMA
                * (stage2_index + 1)
                / max(stage2_steps_requested, 1)
            ) ** (-STAGE2_LR_POWER)
            for group in stage2_optimizer.param_groups:
                group["lr"] = group["lr0"] * decay
                group["momentum"] = STAGE2_MOMENTUM
                group["nesterov"] = STAGE2_NESTEROV

            stage2_optimizer.zero_grad()
            stage2_loss, stage2_loss_parts = _loss_and_parts(
                loss_closure()
            )
            stage2_loss.backward()
            for name, parameter in candidate_parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(
                        state.mask[name].to(
                            device=parameter.grad.device,
                            dtype=parameter.grad.dtype,
                        )
                    )
            stage2_optimizer.step()
            stage2_final_loss = float(stage2_loss.item())
            stage2_steps_completed = stage2_index + 1
            stage2_effective_lr = float(
                stage2_optimizer.param_groups[0]["lr"]
            )

        refined_parameters = _clone_map(candidate_parameters)
        tolerance = float(config["delta_nonzero_tolerance"])
        effective_stats = _change_statistics(
            refined_parameters,
            base_parameters,
            tolerance,
            candidate_count,
            self.total_model_param_count,
            "effective_delta",
        )

        omega = float(config["omega"])
        applied_parameters = {
            name: (1.0 - omega) * base_parameters[name]
            + omega * refined_parameters[name]
            for name, _ in candidate_parameters
        }
        _copy_map(candidate_parameters, applied_parameters)
        applied_stats = _change_statistics(
            applied_parameters,
            base_parameters,
            tolerance,
            candidate_count,
            self.total_model_param_count,
            "applied_update",
        )

        support_over_model_ratio = (
            float(support_count / self.total_model_param_count)
            if self.total_model_param_count > 0
            else 0.0
        )
        budget_diagnostics = compute_lbi_step_budget_diagnostics(
            requested_budget=requested_budget,
            candidate_scope_param_count=candidate_count,
            stage1_support_count=support_count,
            stage1_stop_reason=stage1_stop_reason,
        )
        statistics = {
            "selection": "lbi",
            "requested_budget": requested_budget,
            "lbi_alpha": alpha,
            "lbi_kappa": kappa,
            "lbi_nu": nu,
            "lbi_omega": omega,
            "stage1_branch_enabled": stage1_branch_enabled,
            "stage1_steps_completed": stage1_steps_completed,
            "stage1_stop_reason": stage1_stop_reason,
            "stage1_rollback_used": stage1_rollback_used,
            "stage1_last_feasible_step": feasible_step,
            "stage1_feasible_found": feasible_state is not None,
            "stage1_budget_target": budget_target,
            "stage1_support_count": support_count,
            "stage1_support_ratio": support_ratio,
            "stage1_final_loss": stage1_final_loss,
            "stage2_steps_requested": stage2_steps_requested,
            "stage2_steps_completed": stage2_steps_completed,
            "stage2_lr": stage2_lr,
            "stage2_effective_lr": stage2_effective_lr,
            "stage2_optimizer": STAGE2_OPTIMIZER,
            "stage2_final_loss": stage2_final_loss,
            "loss": stage2_final_loss,
            **stage2_loss_parts,
            "support_param_count": support_count,
            "support_over_scope_ratio": support_ratio,
            "support_over_model_ratio": support_over_model_ratio,
            "selected_param_count": support_count,
            "selected_over_scope_ratio": support_ratio,
            "selected_over_model_ratio": support_over_model_ratio,
            "candidate_scope_param_count": candidate_count,
            "total_model_param_count": self.total_model_param_count,
            **budget_diagnostics,
            **effective_stats,
            **applied_stats,
        }
        return LBIResult(
            state=state.clone(),
            base_parameters=base_parameters,
            refined_parameters=refined_parameters,
            applied_parameters=applied_parameters,
            statistics=statistics,
        )
