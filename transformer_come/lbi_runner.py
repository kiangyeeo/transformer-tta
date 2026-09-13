"""COME host integration for the shared Transformer Group Split-LBI engine.

The structural state machine lives in :mod:`transformer.group_lbi`; this file
only supplies the stable current-logit COME closure and COME artifact identity.
"""

from __future__ import annotations

import statistics
from pathlib import Path

from transformer.group_lbi.runner import run_transfer as run_group_lbi_transfer

from .common import ComeRunInvalid, guard_online_batch, prediction_diagnostics
from .config import (
    GROUP_LBI,
    IMPLEMENTATION_REVISIONS,
    PROTOCOL_DOCUMENTS,
)
from .data import build_target_loaders
from .model import load_group_lbi_model
from .objective import come_loss


def come_lbi_objective(model, images, config: dict):
    """Return a differentiable COME loss without calling ``backward``.

    The shared engine owns ``autograd.grad`` in Stage 1 and ``backward`` in
    Stage 2.  Diagnostics are detached and therefore cannot alter the update.
    """

    logits = model(images)
    class_count = int(config["num_classes"])
    if logits.ndim != 2 or int(logits.shape[1]) != class_count:
        raise ComeRunInvalid(
            "COME class_count must equal the Group-LBI classifier output "
            f"dimension; got C={class_count}, logits={tuple(logits.shape)}",
            invalid_reason="class_count_mismatch",
        )
    try:
        result = come_loss(logits, class_count)
    except RuntimeError as error:
        raise ComeRunInvalid(
            str(error), invalid_reason="nonfinite_objective"
        ) from error
    diagnostics = {
        **prediction_diagnostics(logits, class_count),
        **dict(result.diagnostics),
        "host_objective": "come",
    }
    return result.loss, diagnostics


def run_transfer(
    config: dict,
    project_root: Path,
    *,
    show_progress: bool = True,
    resume: bool = False,
    model_loader=load_group_lbi_model,
    target_loaders_builder=build_target_loaders,
    engine_factory=None,
):
    """Run one COME Group-LBI transfer on the shared corrected engine."""

    if config.get("method") != "come" or config.get("variant") != GROUP_LBI:
        raise ValueError("COME Group-LBI runner requires method/variant come/group_lbi")

    def summarize_diagnostics(records: list[dict]) -> dict:
        if not records:
            return {}
        return {
            "collapse_diagnostics_last_batch": {
                key: records[-1][key]
                for key in (
                    "predicted_class_histogram",
                    "predicted_class_count",
                    "dominant_class_count",
                    "dominant_class_ratio",
                    "mean_softmax_entropy",
                    "come_opinion_entropy",
                    "come_mean_uncertainty_mass",
                )
            },
            "predicted_class_count_min": min(
                row["predicted_class_count"] for row in records
            ),
            "predicted_class_count_mean": statistics.fmean(
                row["predicted_class_count"] for row in records
            ),
            "dominant_class_ratio_mean": statistics.fmean(
                row["dominant_class_ratio"] for row in records
            ),
            "dominant_class_ratio_max": max(
                row["dominant_class_ratio"] for row in records
            ),
            "come_opinion_entropy_mean": statistics.fmean(
                row["come_opinion_entropy"] for row in records
            ),
            "come_mean_uncertainty_mass_mean": statistics.fmean(
                row["come_mean_uncertainty_mass"] for row in records
            ),
        }

    runner_kwargs = {}
    if engine_factory is not None:
        runner_kwargs["engine_factory"] = engine_factory
    return run_group_lbi_transfer(
        config,
        project_root,
        show_progress=show_progress,
        resume=resume,
        model_loader=model_loader,
        objective_fn=come_lbi_objective,
        method="come",
        protocol_document=PROTOCOL_DOCUMENTS[GROUP_LBI],
        implementation_revision=IMPLEMENTATION_REVISIONS[GROUP_LBI],
        batch_guard=guard_online_batch,
        objective_artifact_key="come",
        target_loaders_builder=target_loaders_builder,
        summary_enricher=summarize_diagnostics,
        artifact_extra={"come_objective": config["come_objective"]},
        **runner_kwargs,
    )


__all__ = ["come_lbi_objective", "run_transfer"]
