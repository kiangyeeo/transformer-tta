"""Strict sparse AdamW step with exact off-mask value/state freezing."""

from __future__ import annotations

import torch


ADAM_COORDINATE_STATE = ("exp_avg", "exp_avg_sq")


def _clear_off_mask_state(optimizer, parameter, mask: torch.Tensor) -> None:
    state = optimizer.state.get(parameter, {})
    for name in ADAM_COORDINATE_STATE:
        value = state.get(name)
        if value is not None:
            if tuple(value.shape) != tuple(parameter.shape):
                raise RuntimeError(f"AdamW state {name} has the wrong shape")
            value.mul_(mask.to(device=value.device, dtype=value.dtype))


def strict_masked_adamw_step(optimizer, named_parameters, masks) -> None:
    """Apply AdamW only on mask coordinates and erase off-mask moments."""
    named_parameters = list(named_parameters)
    if {name for name, _ in named_parameters} != set(masks):
        raise ValueError("Optimizer parameter names and structural mask names differ")
    snapshots = {}
    with torch.no_grad():
        for name, parameter in named_parameters:
            mask = masks[name].to(device=parameter.device, dtype=torch.bool)
            _clear_off_mask_state(optimizer, parameter, mask)
            if parameter.grad is None:
                raise RuntimeError(f"Candidate gradient is missing for {name}")
            parameter.grad.mul_(mask.to(dtype=parameter.grad.dtype))
            snapshots[name] = parameter.detach().clone()

    optimizer.step()

    with torch.no_grad():
        for name, parameter in named_parameters:
            mask = masks[name].to(device=parameter.device, dtype=torch.bool)
            parameter.copy_(torch.where(mask, parameter, snapshots[name]))
            _clear_off_mask_state(optimizer, parameter, mask)


def assert_off_mask_adam_state_zero(optimizer, named_parameters, masks) -> None:
    for name, parameter in named_parameters:
        mask = masks[name].to(device=parameter.device, dtype=torch.bool)
        state = optimizer.state.get(parameter, {})
        for state_name in ADAM_COORDINATE_STATE:
            value = state.get(state_name)
            if value is not None and torch.count_nonzero(value[~mask]).item() != 0:
                raise RuntimeError(f"Off-mask AdamW state drifted: {name}.{state_name}")

