"""Shared COME adaptation entry point, diagnostics and defensive contracts.

Every variant (dense and sparse) calls :func:`come_objective_step` so that a
single stable objective implementation is used everywhere, instead of each
runner re-deriving the formula.
"""

from __future__ import annotations

from datetime import datetime, timezone

import torch

from transformer.candidate_dense.model import hash_tensors

from .objective import come_loss


class ComeRunInvalid(RuntimeError):
    """A run-invalidating protocol violation with an explicit reason code."""

    def __init__(self, message: str, *, invalid_reason: str) -> None:
        super().__init__(message)
        self.invalid_reason = invalid_reason


def invalid_reason_of(error: BaseException) -> str:
    return str(getattr(error, "invalid_reason", "runtime_exception"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def primary_metric_name(dataset: str) -> str:
    return (
        "sample_overall_accuracy"
        if dataset == "office31"
        else "fixed_12_class_macro_accuracy"
    )


def classifier_hash(model) -> str:
    return hash_tensors(model.get_classifier().state_dict().items())


def guard_online_batch(batch_size: int) -> None:
    """Reject an actual singleton batch before any adaptation transition.

    Protocol section 2: the formal Transformer stream merges the Amazon tail
    into ``43x64 + 65``, so an actual ``BS=1`` online batch means the stream
    identity is wrong.  This check runs before the objective, the selector,
    the optimizer, any LBI state and PU, and it fails instead of silently
    dropping the sample from PU.
    """

    if int(batch_size) == 1:
        raise ComeRunInvalid(
            "Actual online batch size is 1; the formal Transformer stream "
            "must not produce singleton batches",
            invalid_reason="stream_protocol_mismatch",
        )
    if int(batch_size) < 1:
        raise ComeRunInvalid(
            "Online batch is empty", invalid_reason="stream_protocol_mismatch"
        )


def come_objective_step(model, images, class_count: int):
    """One current-state COME objective call plus one backward.

    No pseudo-label, teacher, memory bank, source anchor or target label is
    involved: the closure only sees the current model and the current batch.
    """

    logits = model(images)
    if logits.ndim != 2:
        raise ComeRunInvalid(
            f"COME logits must be [batch, classes]; got {tuple(logits.shape)}",
            invalid_reason="logits_shape_mismatch",
        )
    if int(logits.shape[1]) != int(class_count):
        raise ComeRunInvalid(
            "COME class_count must equal the adaptation classifier output "
            f"dimension; got C={int(class_count)}, logits={tuple(logits.shape)}",
            invalid_reason="class_count_mismatch",
        )
    try:
        result = come_loss(logits, int(class_count))
    except RuntimeError as error:
        raise ComeRunInvalid(str(error), invalid_reason="nonfinite_objective") from error
    result.loss.backward()
    return result.loss, logits, dict(result.diagnostics)


def assert_finite_gradients(named_parameters) -> None:
    for name, parameter in named_parameters:
        gradient = parameter.grad
        if gradient is None:
            raise ComeRunInvalid(
                f"Candidate gradient is missing for {name}",
                invalid_reason="missing_gradient",
            )
        if not torch.isfinite(gradient).all():
            raise ComeRunInvalid(
                f"Non-finite COME gradient for {name}",
                invalid_reason="nonfinite_gradient",
            )


def prediction_diagnostics(logits, class_count: int) -> dict:
    """Read-only collapse diagnostics derived from the current logits.

    The COME opinion entropy and mean uncertainty mass come from the
    diagnostics of the objective call that produced these logits, so this
    helper never re-enters the objective (protocol section 7) and never
    mutates model, optimizer or RNG state.
    """

    with torch.no_grad():
        detached = logits.detach()
        probabilities = detached.softmax(dim=1)
        predictions = probabilities.argmax(dim=1)
        histogram = torch.bincount(predictions, minlength=int(class_count)).cpu()
        dominant_count = int(histogram.max().item()) if histogram.numel() else 0
        sample_count = int(predictions.numel())
        softmax_entropy = -(
            probabilities * probabilities.clamp_min(1.0e-12).log()
        ).sum(dim=1)
    return {
        "predicted_class_histogram": histogram.tolist(),
        "predicted_class_count": int(torch.count_nonzero(histogram).item()),
        "dominant_class_count": dominant_count,
        "dominant_class_ratio": (
            float(dominant_count / sample_count) if sample_count else 0.0
        ),
        "mean_softmax_entropy": float(softmax_entropy.mean().item()),
    }


def rng_snapshot(device) -> dict:
    """Capture CPU/CUDA RNG state for read-only selector audits."""

    snapshot = {"cpu": torch.random.get_rng_state().clone()}
    if getattr(device, "type", None) == "cuda":
        snapshot["cuda"] = torch.cuda.get_rng_state(device).clone()
    return snapshot


def assert_rng_unchanged(snapshot: dict, device, *, stage: str) -> None:
    if not torch.equal(torch.random.get_rng_state(), snapshot["cpu"]):
        raise ComeRunInvalid(
            f"{stage} consumed CPU RNG state", invalid_reason="rng_state_changed"
        )
    if "cuda" in snapshot:
        if not torch.equal(torch.cuda.get_rng_state(device), snapshot["cuda"]):
            raise ComeRunInvalid(
                f"{stage} consumed CUDA RNG state",
                invalid_reason="rng_state_changed",
            )


__all__ = [
    "ComeRunInvalid",
    "assert_finite_gradients",
    "assert_rng_unchanged",
    "classifier_hash",
    "come_objective_step",
    "guard_online_batch",
    "invalid_reason_of",
    "prediction_diagnostics",
    "primary_metric_name",
    "rng_snapshot",
    "utc_now",
]
