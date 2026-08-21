"""Split-LBI math migrated from nips2026/SHOT-OTTA-LBI.

The engine owns no dataset, model architecture, evaluation, or YAML logic.
Every call creates a fresh local trajectory and a fresh Stage-2 optimizer.
Only the Stage-3 parameter values written back to the supplied Parameters
survive into the next online batch.
"""

import math

import torch
import torch.optim as optim

from .diagnostics import (
    compute_lbi_step_budget_diagnostics,
    max_support_count,
)
from .state import LBIResult, LBIState


STAGE2_OPTIMIZER = "sgd"
STAGE2_MOMENTUM = 0.9
STAGE2_WEIGHT_DECAY = 1.0e-3
STAGE2_NESTEROV = True


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


def _support_masks(gamma, support_threshold):
    """Build the sole thresholded support used by all LBI stages."""
    return {
        name: value.detach().abs().ge(support_threshold)
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
            "support_threshold",
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
            "support_threshold",
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
        if float(config["support_threshold"]) < 0:
            raise ValueError(
                "Split-LBI support_threshold must be >= 0"
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

    def run_step(self, candidate_parameters, loss_closure, config, timing=None):
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
        support_threshold = float(config["support_threshold"])
        requested_budget = float(config["requested_budget"])
        max_support = max_support_count(requested_budget, candidate_count)
        # This implementation deliberately uses a strict-budget rollback
        # variant: the selected state must never exceed max_support.
        # budget_tolerance remains recorded for experiment compatibility but
        # is not a relaxation of the selected-support constraint.
        max_steps = int(config["stage1_max_steps"])
        stage1_branch_enabled = requested_budget > 0.0

        # The zero initialization is always feasible.  Retaining it ensures
        # an immediate support overshoot still rolls back below the budget.
        feasible_state = state.clone()
        feasible_step = 0
        stage1_steps_completed = 0
        stage1_final_loss = None
        stage1_stop_reason = (
            "max_steps_reached"
            if stage1_branch_enabled
            else "branch_disabled"
        )
        stage1_rollback_used = False

        stage1_started_at = timing.start("lbi_stage1") if timing else None
        try:
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

                # Eq. (5) uses the coupling from the old local state.  Keep it
                # separate so both theta_delta and z consume c^k, not c^(k+1).
                old_coupling = {
                    name: (state.theta_delta[name] - state.gamma[name]) / nu
                    for name, _ in candidate_parameters
                }

                # theta_delta^(k+1) = theta_delta^k
                #   - alpha*kappa*(grad + c^k)
                for (name, _), gradient in zip(
                    candidate_parameters, gradients
                ):
                    state.theta_delta[name] = (
                        state.theta_delta[name]
                        - alpha * kappa * (gradient + old_coupling[name])
                    )

                # z^(k+1) = z^k + alpha*c^k
                for name, _ in candidate_parameters:
                    state.z[name] = state.z[name] + alpha * old_coupling[name]

                # gamma <- kappa*sign(z)*max(abs(z)-1, 0)
                for name, _ in candidate_parameters:
                    state.gamma[name] = self.element_prox(
                        state.z[name], kappa
                    )

                state.mask = _support_masks(state.gamma, support_threshold)
                support_count = _support_count(state.mask)
                if support_count < max_support:
                    feasible_state = state.clone()
                    feasible_step = stage1_steps_completed
                    continue

                if support_count == max_support:
                    # The current state is feasible and reaches the integer
                    # budget exactly, so retain it without a rollback.
                    feasible_state = state.clone()
                    feasible_step = stage1_steps_completed
                    stage1_stop_reason = "budget_reached"
                    break

                # Only an over-budget state is rejected and restored.  This is
                # the strict-budget rollback variant, not the paper's direct
                # stopping rule.
                state = feasible_state.clone()
                stage1_stop_reason = "strict_budget_rollback"
                stage1_rollback_used = True
                break
        finally:
            if timing:
                timing.stop("lbi_stage1", stage1_started_at)

        state.mask = _support_masks(state.gamma, support_threshold)
        support_count = _support_count(state.mask)
        support_ratio = (
            float(support_count / candidate_count)
            if candidate_count > 0
            else 0.0
        )

        # Initialize refinement only on gamma's selected support.  This keeps
        # every mask-out value at the current online-step base parameter.
        masked_delta_initial_parameters = {
            name: base_parameters[name]
            + state.mask[name].to(
                device=state.theta_delta[name].device,
                dtype=state.theta_delta[name].dtype,
            )
            * state.theta_delta[name]
            for name, _ in candidate_parameters
        }
        _copy_map(candidate_parameters, masked_delta_initial_parameters)

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

        stage2_final_loss = None
        stage2_loss_parts = {}
        stage2_steps_completed = 0
        stage2_started_at = timing.start("lbi_stage2") if timing else None
        try:
            for _ in range(stage2_steps_requested):
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
                # Gradient masking alone cannot freeze parameter values when SGD
                # has weight decay, momentum, or Nesterov enabled.  Preserve the
                # full pre-step values and restore every mask-out element after
                # the normal optimizer update; optimizer state remains untouched.
                stage2_pre_step_parameters = _clone_map(candidate_parameters)
                stage2_optimizer.step()
                with torch.no_grad():
                    for name, parameter in candidate_parameters:
                        mask = state.mask[name].to(
                            device=parameter.device,
                            dtype=torch.bool,
                        )
                        parameter.copy_(
                            torch.where(
                                mask,
                                parameter,
                                stage2_pre_step_parameters[name],
                            )
                        )
                stage2_final_loss = float(stage2_loss.item())
                stage2_steps_completed += 1
        finally:
            if timing:
                timing.stop("lbi_stage2", stage2_started_at)

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
            "lbi_support_threshold": support_threshold,
            "stage1_branch_enabled": stage1_branch_enabled,
            "stage1_steps_completed": stage1_steps_completed,
            "stage1_stop_reason": stage1_stop_reason,
            "stage1_rollback_used": stage1_rollback_used,
            "stage1_last_feasible_step": feasible_step,
            "stage1_feasible_found": feasible_state is not None,
            "stage1_budget_target": requested_budget,
            "stage1_max_support_count": max_support,
            "stage1_support_count": support_count,
            "stage1_support_ratio": support_ratio,
            "stage1_final_loss": stage1_final_loss,
            "stage2_steps_requested": stage2_steps_requested,
            "stage2_steps_completed": stage2_steps_completed,
            "stage2_lr": stage2_lr,
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
