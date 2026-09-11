"""Immutable view-level IST tasks and exact mean-loss accumulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class FixedISTTask:
    """Batch-local task; target ground truth is intentionally impossible to pass."""

    views: torch.Tensor
    hard_targets: torch.Tensor
    soft_targets: torch.Tensor

    def __post_init__(self) -> None:
        count = int(self.views.shape[0])
        if count <= 0:
            raise ValueError("FixedISTTask requires at least one view")
        if self.hard_targets.shape != (count,):
            raise ValueError("hard targets must have one entry per view")
        if self.soft_targets.ndim != 2 or self.soft_targets.shape[0] != count:
            raise ValueError("soft targets must have one row per view")
        object.__setattr__(self, "hard_targets", self.hard_targets.detach().clone())
        object.__setattr__(self, "soft_targets", self.soft_targets.detach().clone())

    @property
    def sample_count(self) -> int:
        return int(self.views.shape[0])


def ist_loss(logits, hard_targets, soft_targets, config):
    hard_ce = functional.cross_entropy(logits, hard_targets)
    soft_kl = functional.kl_div(
        torch.log_softmax(logits, dim=-1), soft_targets, reduction="batchmean"
    )
    total = (
        float(config["hard_ce_weight"]) * hard_ce
        + float(config["soft_kl_weight"]) * soft_kl
    )
    return total, {
        "loss_hard_ce": float(hard_ce.detach().item()),
        "loss_soft_kl": float(soft_kl.detach().item()),
    }


def accumulate_full_objective(
    task: FixedISTTask,
    forward_logits: Callable[[torch.Tensor], torch.Tensor],
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    loss_config: dict,
    chunk_size: int,
) -> dict:
    """Backpropagate the exact mean over all views without optimizer/state writes."""
    named_parameters = list(named_parameters)
    for _, parameter in named_parameters:
        parameter.grad = None
    totals = {"loss": 0.0, "loss_hard_ce": 0.0, "loss_soft_kl": 0.0}
    backwards = 0
    for start in range(0, task.sample_count, int(chunk_size)):
        end = min(start + int(chunk_size), task.sample_count)
        weight = float((end - start) / task.sample_count)
        logits = forward_logits(task.views[start:end])
        total, parts = ist_loss(
            logits,
            task.hard_targets[start:end],
            task.soft_targets[start:end],
            loss_config,
        )
        (total * weight).backward()
        totals["loss"] += float(total.detach().item()) * weight
        for name in ("loss_hard_ce", "loss_soft_kl"):
            totals[name] += parts[name] * weight
        backwards += 1
    for name, parameter in named_parameters:
        if parameter.grad is None:
            raise RuntimeError(f"full IST objective produced no gradient for {name}")
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"non-finite full IST gradient for {name}")
    totals["full_objective_chunk_backwards"] = backwards
    return totals
