"""Batch-local structural Group Split-LBI for the controlled DeiT candidate."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import torch

from transformer.candidate_dense.config import candidate_parameter_names
from transformer.group_random.groups import (
    BLOCKS,
    GROUP_KINDS,
    HIDDEN_DIM,
    MLP_DIM,
    STRUCTURAL_GROUPS,
    build_masks,
)
from transformer.group_random.optimizer import (
    assert_off_mask_adam_state_zero,
    strict_masked_adamw_step,
)

from .config import GROUP_SIZE, TOTAL_GROUPS


TensorMap = dict[str, torch.Tensor]


@dataclass
class GroupLBIState:
    """Local tensors that are deliberately recreated for every online batch."""

    theta_delta: TensorMap
    gamma: TensorMap
    z: TensorMap
    masks: TensorMap
    active_group_ids: tuple[int, ...]

    def clone(self) -> "GroupLBIState":
        return GroupLBIState(
            theta_delta={name: value.detach().clone() for name, value in self.theta_delta.items()},
            gamma={name: value.detach().clone() for name, value in self.gamma.items()},
            z={name: value.detach().clone() for name, value in self.z.items()},
            masks={name: value.detach().clone() for name, value in self.masks.items()},
            active_group_ids=tuple(self.active_group_ids),
        )


@dataclass
class GroupLBIResult:
    state: GroupLBIState
    base_parameters: TensorMap
    refined_parameters: TensorMap
    applied_parameters: TensorMap
    statistics: dict[str, Any]


def _clone_map(named_parameters) -> TensorMap:
    return {name: parameter.detach().clone() for name, parameter in named_parameters}


def _copy_map(named_parameters, values: TensorMap) -> None:
    with torch.no_grad():
        for name, parameter in named_parameters:
            parameter.copy_(values[name])


def _loss_and_parts(loss_output):
    if isinstance(loss_output, tuple):
        loss, parts = loss_output
        return loss, dict(parts)
    return loss_output, {}


def _validate_candidate_map(values: TensorMap) -> None:
    expected = set(candidate_parameter_names())
    if set(values) != expected:
        raise ValueError(
            "Group-LBI candidate tensors differ from the frozen set: "
            f"{sorted(set(values) ^ expected)}"
        )


def pack_groups(values: TensorMap) -> torch.Tensor:
    """Pack partitioned candidate tensors into canonical [6912, 768] groups."""
    _validate_candidate_map(values)
    packed = []
    for block in BLOCKS:
        prefix = f"blocks.{block}."
        qkv = values[prefix + "attn.qkv.weight"]
        proj = values[prefix + "attn.proj.weight"]
        fc1 = values[prefix + "mlp.fc1.weight"]
        fc2 = values[prefix + "mlp.fc2.weight"]
        q, k, v = qkv.split(HIDDEN_DIM, dim=0)
        packed.extend(
            (
                torch.cat((q, k), dim=1),
                torch.cat((v, proj.transpose(0, 1)), dim=1),
                torch.cat((fc1, fc2.transpose(0, 1)), dim=1),
            )
        )
    result = torch.cat(packed, dim=0).contiguous()
    if result.shape != (TOTAL_GROUPS, GROUP_SIZE):
        raise RuntimeError(
            f"Packed groups must have shape {(TOTAL_GROUPS, GROUP_SIZE)}, "
            f"got {tuple(result.shape)}"
        )
    return result


def unpack_groups(groups: torch.Tensor) -> TensorMap:
    """Invert :func:`pack_groups` without changing fused timm qkv storage."""
    if groups.shape != (TOTAL_GROUPS, GROUP_SIZE):
        raise ValueError(
            f"groups must have shape {(TOTAL_GROUPS, GROUP_SIZE)}, "
            f"got {tuple(groups.shape)}"
        )
    result: TensorMap = {}
    offset = 0
    for block in BLOCKS:
        qk = groups[offset : offset + HIDDEN_DIM]
        offset += HIDDEN_DIM
        vo = groups[offset : offset + HIDDEN_DIM]
        offset += HIDDEN_DIM
        ffn = groups[offset : offset + MLP_DIM]
        offset += MLP_DIM
        prefix = f"blocks.{block}."
        q = qk[:, :HIDDEN_DIM]
        k = qk[:, HIDDEN_DIM:]
        v = vo[:, :HIDDEN_DIM]
        result[prefix + "attn.qkv.weight"] = torch.cat((q, k, v), dim=0).contiguous()
        result[prefix + "attn.proj.weight"] = vo[:, HIDDEN_DIM:].transpose(0, 1).contiguous()
        result[prefix + "mlp.fc1.weight"] = ffn[:, :HIDDEN_DIM].contiguous()
        result[prefix + "mlp.fc2.weight"] = ffn[:, HIDDEN_DIM:].transpose(0, 1).contiguous()
    if offset != TOTAL_GROUPS:
        raise RuntimeError("Group unpacking did not consume the canonical group pool")
    return result


def group_soft_threshold(
    packed_z: torch.Tensor, *, kappa: float, prox_lambda: float
) -> torch.Tensor:
    """Apply the group L2 prox row-wise with an exact zero-norm branch."""
    if packed_z.ndim != 2:
        raise ValueError("packed_z must be a two-dimensional group matrix")
    norms = torch.linalg.vector_norm(packed_z, ord=2, dim=1)
    factors = torch.zeros_like(norms)
    nonzero = norms > 0
    factors[nonzero] = torch.clamp(
        1.0 - float(prox_lambda) / norms[nonzero], min=0.0
    )
    return float(kappa) * factors.unsqueeze(1) * packed_z


def old_state_split_lbi_update(
    theta_delta: TensorMap,
    gamma: TensorMap,
    z: TensorMap,
    gradients: TensorMap,
    *,
    alpha: float,
    kappa: float,
    nu: float,
) -> tuple[TensorMap, TensorMap]:
    """Update delta and z from the same old-state coupling."""
    names = set(theta_delta)
    if set(gamma) != names or set(z) != names or set(gradients) != names:
        raise ValueError("Old-state Split-LBI tensor maps must have identical names")
    coupling = {name: (theta_delta[name] - gamma[name]) / float(nu) for name in names}
    new_delta = {
        name: theta_delta[name]
        - float(alpha) * float(kappa) * (gradients[name] + coupling[name])
        for name in names
    }
    new_z = {name: z[name] + float(alpha) * coupling[name] for name in names}
    return new_delta, new_z


def strict_budget_action(support_count: int, maximum: int) -> str:
    """Return the protocol action for one newly computed support count."""
    if support_count < 0 or maximum < 0:
        raise ValueError("Support counts must be non-negative")
    if support_count < maximum:
        return "continue"
    if support_count == maximum:
        return "accept_and_stop"
    return "rollback_and_stop"


def masked_delta_initialization(base: TensorMap, delta: TensorMap, masks: TensorMap) -> TensorMap:
    if set(base) != set(delta) or set(base) != set(masks):
        raise ValueError("Masked-delta maps must have identical names")
    return {
        name: base[name] + masks[name].to(dtype=delta[name].dtype) * delta[name]
        for name in base
    }


def omega_accumulation(base: TensorMap, refined: TensorMap, omega: float) -> TensorMap:
    if set(base) != set(refined):
        raise ValueError("Omega accumulation maps must have identical names")
    if not 0.0 <= float(omega) <= 1.0:
        raise ValueError("omega must be in [0, 1]")
    return {
        # The increment form is algebraically equivalent to the weighted sum,
        # but preserves off-mask coordinates bitwise when refined == base.
        name: base[name] + float(omega) * (refined[name] - base[name])
        for name in base
    }


def support_from_gamma(
    gamma: TensorMap, *, tau_g: float
) -> tuple[tuple[int, ...], torch.Tensor]:
    packed = pack_groups(gamma)
    normalized_norms = torch.linalg.vector_norm(packed, ord=2, dim=1) / math.sqrt(
        GROUP_SIZE
    )
    support = normalized_norms.ge(float(tau_g))
    ids = tuple(int(value) for value in torch.nonzero(support).flatten().cpu().tolist())
    return ids, normalized_norms


def group_kind_counts(group_ids) -> dict[str, int]:
    groups = [STRUCTURAL_GROUPS[int(group_id)] for group_id in group_ids]
    return {kind: sum(group.kind == kind for group in groups) for kind in GROUP_KINDS}


def group_block_counts(group_ids) -> dict[str, int]:
    groups = [STRUCTURAL_GROUPS[int(group_id)] for group_id in group_ids]
    return {str(block): sum(group.block == block for group in groups) for block in BLOCKS}


def changed_group_ids(values: TensorMap, base: TensorMap, tolerance: float) -> tuple[int, ...]:
    differences = {name: values[name].detach() - base[name] for name in values}
    packed = pack_groups(differences)
    changed = packed.abs().gt(float(tolerance)).any(dim=1)
    return tuple(int(value) for value in torch.nonzero(changed).flatten().cpu().tolist())


class GroupSplitLBIEngine:
    """Run one corrected, strict-budget Group Split-LBI online batch."""

    @staticmethod
    def initialize(candidate_parameters) -> GroupLBIState:
        candidates = list(candidate_parameters)
        zeros = {name: torch.zeros_like(parameter) for name, parameter in candidates}
        return GroupLBIState(
            theta_delta={name: value.detach().clone() for name, value in zeros.items()},
            gamma={name: value.detach().clone() for name, value in zeros.items()},
            z={name: value.detach().clone() for name, value in zeros.items()},
            masks={name: torch.zeros_like(value, dtype=torch.bool) for name, value in zeros.items()},
            active_group_ids=(),
        )

    @staticmethod
    def _validate(candidate_parameters, config: dict) -> None:
        names = [name for name, _ in candidate_parameters]
        if names != list(candidate_parameter_names()):
            raise ValueError("Group-LBI requires candidates in canonical frozen order")
        required = {
            "alpha",
            "kappa",
            "nu",
            "omega",
            "prox_lambda",
            "tau_g",
            "stage1_max_steps",
            "stage2_lr",
            "requested_group_count",
            "stage2_optimization",
            "delta_nonzero_tolerance",
        }
        missing = sorted(required - set(config))
        if missing:
            raise ValueError(f"Missing Group-LBI config fields: {missing}")
        if not 0 <= int(config["requested_group_count"]) <= TOTAL_GROUPS:
            raise ValueError("requested_group_count must be in [0, 6912]")
        if int(config["stage1_max_steps"]) <= 0:
            raise ValueError("stage1_max_steps must be positive")

    def run_batch(
        self,
        candidate_parameters,
        loss_closure: Callable,
        config: dict,
        *,
        timing=None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> GroupLBIResult:
        candidates = list(candidate_parameters)
        self._validate(candidates, config)
        base = _clone_map(candidates)
        state = self.initialize(candidates)
        feasible_state = state.clone()
        feasible_step = 0

        alpha = float(config["alpha"])
        kappa = float(config["kappa"])
        nu = float(config["nu"])
        prox_lambda = float(config["prox_lambda"])
        tau_g = float(config["tau_g"])
        maximum = int(config["requested_group_count"])
        maximum_steps = int(config["stage1_max_steps"])
        stop_reason = "budget_reached" if maximum == 0 else "max_steps_reached"
        rollback = False
        stage1_loss_value = None
        stage1_steps = 0

        stage1_started = timing.start("lbi_stage1") if timing else None
        try:
            for step_index in (() if maximum == 0 else range(maximum_steps)):
                with torch.no_grad():
                    for name, parameter in candidates:
                        parameter.copy_(base[name] + state.theta_delta[name])
                try:
                    stage1_loss, _ = _loss_and_parts(loss_closure())
                    if not torch.isfinite(stage1_loss):
                        raise FloatingPointError("Stage-1 host loss is NaN or Inf")
                    gradients = torch.autograd.grad(
                        stage1_loss,
                        [parameter for _, parameter in candidates],
                        allow_unused=False,
                    )
                finally:
                    _copy_map(candidates, base)
                if any(not torch.isfinite(gradient).all() for gradient in gradients):
                    raise FloatingPointError("Stage-1 gradient contains NaN or Inf")

                stage1_steps = step_index + 1
                stage1_loss_value = float(stage1_loss.detach().item())
                gradient_map = {
                    name: gradient for (name, _), gradient in zip(candidates, gradients)
                }
                state.theta_delta, state.z = old_state_split_lbi_update(
                    state.theta_delta,
                    state.gamma,
                    state.z,
                    gradient_map,
                    alpha=alpha,
                    kappa=kappa,
                    nu=nu,
                )
                for state_name, values in (
                    ("theta_delta", state.theta_delta),
                    ("z", state.z),
                ):
                    if any(not torch.isfinite(value).all() for value in values.values()):
                        raise FloatingPointError(
                            f"Stage-1 {state_name} contains NaN or Inf"
                        )
                packed_gamma = group_soft_threshold(
                    pack_groups(state.z), kappa=kappa, prox_lambda=prox_lambda
                )
                state.gamma = unpack_groups(packed_gamma)
                if any(not torch.isfinite(value).all() for value in state.gamma.values()):
                    raise FloatingPointError("Stage-1 gamma contains NaN or Inf")
                group_ids, _ = support_from_gamma(state.gamma, tau_g=tau_g)
                state.active_group_ids = group_ids
                state.masks = build_masks(candidates, group_ids)
                support_count = len(group_ids)
                if progress_callback is not None:
                    progress_callback(stage1_steps, support_count, "running")

                action = strict_budget_action(support_count, maximum)
                if action != "rollback_and_stop":
                    feasible_state = state.clone()
                    feasible_step = stage1_steps
                    if action == "accept_and_stop":
                        stop_reason = "budget_reached"
                        break
                    continue

                state = feasible_state.clone()
                stop_reason = "strict_budget_rollback"
                rollback = True
                if progress_callback is not None:
                    progress_callback(stage1_steps, len(state.active_group_ids), stop_reason)
                break
        finally:
            if timing:
                timing.stop("lbi_stage1", stage1_started)

        support_count = len(state.active_group_ids)
        if support_count > maximum:
            raise RuntimeError("Strict rollback left Group-LBI over budget")
        if maximum == 0 and stage1_steps != 0:
            raise RuntimeError("Zero-budget Group-LBI unexpectedly ran Stage 1")

        masked_delta = masked_delta_initialization(base, state.theta_delta, state.masks)
        _copy_map(candidates, masked_delta)

        stage2 = config["stage2_optimization"]
        if int(stage2["steps"]) != 1 or stage2["optimizer"] != "adamw":
            raise ValueError("Group-LBI Stage 2 is frozen to one AdamW step")
        optimizer = torch.optim.AdamW(
            [parameter for _, parameter in candidates],
            lr=float(config["stage2_lr"]),
            betas=tuple(float(value) for value in stage2["betas"]),
            eps=float(stage2["eps"]),
            weight_decay=float(stage2["weight_decay"]),
        )
        stage2_started = timing.start("lbi_stage2") if timing else None
        try:
            optimizer.zero_grad(set_to_none=True)
            stage2_loss, stage2_parts = _loss_and_parts(loss_closure())
            if not torch.isfinite(stage2_loss):
                raise FloatingPointError("Stage-2 host loss is NaN or Inf")
            stage2_loss.backward()
            if any(
                parameter.grad is None or not torch.isfinite(parameter.grad).all()
                for _, parameter in candidates
            ):
                raise FloatingPointError("Stage-2 gradient contains NaN or Inf")
            strict_masked_adamw_step(optimizer, candidates, state.masks)
            assert_off_mask_adam_state_zero(optimizer, candidates, state.masks)
        finally:
            if timing:
                timing.stop("lbi_stage2", stage2_started)

        refined = _clone_map(candidates)
        if any(not torch.isfinite(value).all() for value in refined.values()):
            raise FloatingPointError("Stage-2 refined state contains NaN or Inf")
        for name, _ in candidates:
            if not torch.equal(refined[name][~state.masks[name]], base[name][~state.masks[name]]):
                raise RuntimeError(f"Stage-2 off-mask value changed for {name}")

        omega = float(config["omega"])
        applied = omega_accumulation(base, refined, omega)
        if any(not torch.isfinite(value).all() for value in applied.values()):
            raise FloatingPointError("Persistent omega state contains NaN or Inf")
        _copy_map(candidates, applied)
        tolerance = float(config["delta_nonzero_tolerance"])
        effective_groups = changed_group_ids(refined, base, tolerance)
        applied_groups = changed_group_ids(applied, base, tolerance)
        selected_kind = group_kind_counts(state.active_group_ids)
        selected_block = group_block_counts(state.active_group_ids)
        statistics = {
            "selection": "group_split_lbi",
            "requested_group_count": maximum,
            "realized_group_count": support_count,
            "utilization": float(support_count / maximum) if maximum else 0.0,
            "active_candidate_scalars": support_count * GROUP_SIZE,
            "selected_group_ids": list(state.active_group_ids),
            "selected_by_kind": selected_kind,
            "selected_by_block": selected_block,
            "stage1_steps_completed": stage1_steps,
            "stage1_max_steps": maximum_steps,
            "stage1_stop_reason": stop_reason,
            "stage1_rollback_used": rollback,
            "stage1_last_feasible_step": feasible_step,
            "stage1_final_loss": stage1_loss_value,
            "stage2_steps_completed": 1,
            "local_restart_count": 1,
            "support_discovery_count": 1,
            "stage2_optimizer_instance_count": 1,
            "stage2_optimizer_step_count": 1,
            "omega_writeback_count": 1,
            "host_optimizer_persistent_step_count": 0,
            "host_scheduler_step_count": 0,
            "native_ema_commit_count": 0,
            "objective_call_count": stage1_steps + 1,
            "stage2_lr": float(config["stage2_lr"]),
            "stage2_optimizer": "adamw",
            "stage2_final_loss": float(stage2_loss.detach().item()),
            "loss": float(stage2_loss.detach().item()),
            "effective_delta_group_count": len(effective_groups),
            "applied_update_group_count": len(applied_groups),
            **stage2_parts,
        }
        return GroupLBIResult(
            state=state.clone(),
            base_parameters=base,
            refined_parameters=refined,
            applied_parameters=applied,
            statistics=statistics,
        )


__all__ = [
    "GroupLBIResult",
    "GroupLBIState",
    "GroupSplitLBIEngine",
    "changed_group_ids",
    "group_block_counts",
    "group_kind_counts",
    "group_soft_threshold",
    "masked_delta_initialization",
    "old_state_split_lbi_update",
    "omega_accumulation",
    "pack_groups",
    "strict_budget_action",
    "support_from_gamma",
    "unpack_groups",
]
