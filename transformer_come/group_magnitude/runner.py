"""Run one static source-W0 Magnitude COME condition through PU and FO."""

from __future__ import annotations

from pathlib import Path

from transformer_come.sparse_runner import run_sparse_transfer

from .data import build_target_loaders
from .groups import (
    build_masks,
    compute_group_l2_scores,
    magnitude_mask_record,
    select_magnitude_group_ids,
)
from .model import load_group_magnitude_model


def _prepare_static_support(*, model, candidates, config, checkpoint_record):
    """Score the source W0 once, on CPU in float64, before the target stream."""

    del model
    budget = config["selection"]["requested_budget"]
    scores = compute_group_l2_scores(candidates)
    group_ids = select_magnitude_group_ids(scores, budget)
    record = magnitude_mask_record(
        group_ids,
        scores=scores,
        budget=budget,
        checkpoint_sha256=checkpoint_record["sha256"],
    )
    return build_masks(candidates, group_ids), record


def run_transfer(
    config: dict,
    project_root: Path,
    *,
    show_progress: bool = True,
    model_loader=load_group_magnitude_model,
) -> dict:
    """Run exactly one transfer/budget Magnitude condition."""

    return run_sparse_transfer(
        config,
        project_root,
        variant="group_magnitude",
        model_loader=model_loader,
        build_target_loaders=build_target_loaders,
        prepare_selection=_prepare_static_support,
        show_progress=show_progress,
    )


__all__ = ["run_transfer"]
