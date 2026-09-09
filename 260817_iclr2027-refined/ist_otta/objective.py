"""Fixed full-outer-batch IST objective and exact gradient accumulation."""

from dataclasses import dataclass

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class FixedISTTask:
    """Batch-local IST task; intentionally contains no target ground truth."""

    views: torch.Tensor
    hard_targets: torch.Tensor
    soft_targets: torch.Tensor

    def __post_init__(self):
        sample_count = int(self.views.shape[0])
        if sample_count <= 0:
            raise ValueError("FixedISTTask requires at least one view")
        if int(self.hard_targets.shape[0]) != sample_count:
            raise ValueError("hard target count does not match IST views")
        if int(self.soft_targets.shape[0]) != sample_count:
            raise ValueError("soft target count does not match IST views")
        object.__setattr__(
            self, "hard_targets", self.hard_targets.detach().clone()
        )
        object.__setattr__(
            self, "soft_targets", self.soft_targets.detach().clone()
        )

    @property
    def sample_count(self):
        return int(self.views.shape[0])


def ist_loss(logits, hard_targets, soft_targets, loss_config):
    hard_ce = functional.cross_entropy(logits, hard_targets)
    soft_kl = functional.kl_div(
        torch.log_softmax(logits, dim=-1),
        soft_targets,
        reduction="batchmean",
    )
    total = (
        float(loss_config["hard_ce_weight"]) * hard_ce
        + float(loss_config["soft_kl_weight"]) * soft_kl
    )
    return total, hard_ce, soft_kl


class FullObjectiveGradientAccumulator:
    """Accumulate an exact sample mean without optimizer/state updates."""

    def __init__(
        self,
        task,
        forward_logits,
        loss_config,
        chunk_size,
        prepare_forward=None,
    ):
        if not isinstance(task, FixedISTTask):
            raise TypeError("task must be a FixedISTTask")
        if isinstance(chunk_size, bool) or int(chunk_size) <= 0:
            raise ValueError("chunk_size must be a positive integer")
        self.task = task
        self.forward_logits = forward_logits
        self.loss_config = loss_config
        self.chunk_size = int(chunk_size)
        self.prepare_forward = prepare_forward
        self.call_count = 0
        self.chunk_backward_count = 0
        self.optimizer_step_count = 0
        self.state_update_count = 0

    def __call__(self, candidate_parameters):
        candidate_parameters = list(candidate_parameters)
        for _, parameter in candidate_parameters:
            parameter.grad = None
        if self.prepare_forward is not None:
            self.prepare_forward()

        totals = {
            "loss": 0.0,
            "loss_hard_ce": 0.0,
            "loss_soft_kl": 0.0,
        }
        sample_count = self.task.sample_count
        for start in range(0, sample_count, self.chunk_size):
            end = min(start + self.chunk_size, sample_count)
            count = end - start
            weight = float(count / sample_count)
            logits = self.forward_logits(self.task.views[start:end])
            total, hard_ce, soft_kl = ist_loss(
                logits,
                self.task.hard_targets[start:end],
                self.task.soft_targets[start:end],
                self.loss_config,
            )
            (total * weight).backward()
            totals["loss"] += float(total.detach().item()) * weight
            totals["loss_hard_ce"] += (
                float(hard_ce.detach().item()) * weight
            )
            totals["loss_soft_kl"] += (
                float(soft_kl.detach().item()) * weight
            )
            self.chunk_backward_count += 1

        for name, parameter in candidate_parameters:
            if parameter.grad is None:
                raise RuntimeError(
                    "full IST objective produced no gradient for " + name
                )
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError(
                    "non-finite full IST objective gradient for " + name
                )
        self.call_count += 1
        loss_tensor = self.task.views.new_tensor(totals["loss"])
        return loss_tensor, {
            "loss_hard_ce": totals["loss_hard_ce"],
            "loss_soft_kl": totals["loss_soft_kl"],
        }
