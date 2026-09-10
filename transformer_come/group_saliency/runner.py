"""Run one dynamic Group-Saliency COME condition through PU and FO.

Each valid outer batch runs exactly one COME forward/backward; the resulting
gradient is scored, the exact-K paired support is built from it, and the very
same gradient is then consumed by the strict masked AdamW step.  The reuse is
only valid while selection leaves the model, the RNG and the module modes
untouched, so those are audited.
"""

from __future__ import annotations

import os
from pathlib import Path

from transformer_come.sparse_runner import run_sparse_transfer

from .data import build_target_loaders
from .groups import (
    build_masks,
    compute_group_saliency_scores,
    dynamic_mask_record,
    select_saliency_group_ids,
)
from .model import load_group_saliency_model


# Parameter/buffer-hash audits around the selection perturb the measured
# adaptation cost (CUDA synchronization, cache) in ways that subtracting the
# measured audit seconds cannot fully undo, so they are OFF by default: a
# formal efficiency run carries no hash instrumentation.  The correctness
# contract is proved instead by the CPU contract tests and by the real-data
# smoke, which switch the audit on through
# ``COME_SELECTION_STATE_AUDIT_BATCHES``.  The cheap CPU/CUDA RNG comparison
# stays on for every batch.  This knob changes no numerics and is deliberately
# not part of the scientific config hash.
SELECTION_AUDIT_BATCHES = int(os.environ.get("COME_SELECTION_STATE_AUDIT_BATCHES", "0"))


def _select_support(*, candidates, config):
    budget = config["selection"]["requested_budget"]
    scores = compute_group_saliency_scores(candidates)
    group_ids = select_saliency_group_ids(scores, budget)
    record = dynamic_mask_record(group_ids, scores=scores, budget=budget)
    return build_masks(candidates, group_ids), record


def run_transfer(
    config: dict,
    project_root: Path,
    *,
    show_progress: bool = True,
    model_loader=load_group_saliency_model,
) -> dict:
    """Run exactly one transfer/budget dynamic Saliency condition."""

    return run_sparse_transfer(
        config,
        project_root,
        variant="group_saliency",
        model_loader=model_loader,
        build_target_loaders=build_target_loaders,
        select_per_batch=_select_support,
        selection_audit_batches=SELECTION_AUDIT_BATCHES,
        show_progress=show_progress,
    )


__all__ = ["SELECTION_AUDIT_BATCHES", "run_transfer"]
