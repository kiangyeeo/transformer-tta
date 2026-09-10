"""Run three real Random children for one COME transfer/budget condition.

Each child restarts from the same source W0 with a fresh optimizer and the
same target stream; only the support seed differs.  The three children are
executed for real - the top-level accuracy is their mean with population std,
never a single run relabelled three times.
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import yaml

from transformer.source_only.runner import _atomic_json, _git_info

# The child-config derivation and the three-child aggregation are part of the
# SHOT substrate: identical seeds, identical mean/population-std semantics.
from transformer.group_random.runner import (
    _aggregate_children as _shot_aggregate_children,
    _child_config,
)

from transformer_come.common import invalid_reason_of, utc_now
from transformer_come.identity import PROTOCOL_DOCUMENT
from transformer_come.runner_common import ARTIFACT_SCHEMA_VERSION
from transformer_come.sparse_runner import run_sparse_transfer

from .config import (
    IMPLEMENTATION_REVISION,
    MASK_SEEDS,
    NUM_RANDOM_MASKS,
    budget_key,
)
from .data import build_target_loaders
from .groups import build_masks, mask_record, selected_group_ids
from .model import load_group_random_model


def _prepare_static_support(*, model, candidates, config, checkpoint_record):
    """Draw this child's exact-K support from the canonical permutation."""

    del model, checkpoint_record
    budget = config["selection"]["requested_budget"]
    group_ids = selected_group_ids(budget, config["mask_seed"])
    record = mask_record(group_ids, budget=budget, seed=config["mask_seed"])
    return build_masks(candidates, group_ids), record


def run_mask_child(
    config: dict,
    project_root: Path,
    *,
    show_progress: bool = True,
    model_loader=load_group_random_model,
) -> dict:
    """Run one mask child from a fresh source model through PU and FO."""

    return run_sparse_transfer(
        config,
        project_root,
        variant="group_random",
        model_loader=model_loader,
        build_target_loaders=build_target_loaders,
        prepare_selection=_prepare_static_support,
        show_progress=show_progress,
        manifest_extra={
            "parent_experiment_key": config["parent_experiment_key"],
            "random_mask_index": config["random_mask_index"],
            "mask_seed": config["mask_seed"],
        },
        summary_extra={
            "parent_experiment_key": config["parent_experiment_key"],
            "random_mask_index": config["random_mask_index"],
            "mask_seed": config["mask_seed"],
        },
        progress_suffix=f" mask={config['random_mask_index']}",
    )


def _aggregate_children(config: dict, children: list[dict], wall_runtime: float) -> dict:
    summary = _shot_aggregate_children(config, children, wall_runtime)
    summary["method"] = "come"
    summary["protocol_document"] = config.get("protocol_document", PROTOCOL_DOCUMENT)
    summary["implementation_revision"] = config.get(
        "implementation_revision", IMPLEMENTATION_REVISION
    )
    summary["result_validity"] = "valid"
    summary["come"] = copy.deepcopy(config["come"])
    summary["come_objective"] = copy.deepcopy(config["come_objective"])
    return summary


def run_transfer(
    config: dict,
    project_root: Path,
    *,
    show_progress: bool = True,
    child_runner=run_mask_child,
) -> dict:
    """Run all three deterministic Random masks for one formal condition."""

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=False)
    with open(output_dir / "effective_config.yaml", "w", encoding="utf-8") as file_obj:
        yaml.safe_dump(config, file_obj, sort_keys=False, allow_unicode=True)
    started_at = utc_now()
    wall_started = time.perf_counter()
    parent_manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "running",
        "created_at_utc": started_at,
        "command": list(sys.argv),
        "git": _git_info(project_root),
        "protocol_revision": config["protocol_revision"],
        "protocol_document": config.get("protocol_document", PROTOCOL_DOCUMENT),
        "implementation_revision": config["implementation_revision"],
        "method": "come",
        "variant": "group_random",
        "dataset": config["dataset"],
        "transfer": config["transfer"],
        "formal_seed": config["formal_seed"],
        "experiment_key": config["experiment_key"],
        "scientific_config_sha256": config["scientific_config_sha256"],
        "selection": copy.deepcopy(config["selection"]),
        "come": copy.deepcopy(config["come"]),
        "come_objective": copy.deepcopy(config["come_objective"]),
    }
    _atomic_json(output_dir / "manifest.json", parent_manifest)
    try:
        children = []
        for mask_index, mask_seed in enumerate(MASK_SEEDS):
            child = _child_config(
                config, mask_index, mask_seed, output_dir / f"mask_{mask_index:02d}"
            )
            children.append(
                child_runner(child, project_root, show_progress=show_progress)
            )
        if len(children) != NUM_RANDOM_MASKS:
            raise RuntimeError("Random parent did not complete exactly three masks")
        completed_at = utc_now()
        summary = _aggregate_children(
            config, children, float(time.perf_counter() - wall_started)
        )
        summary["started_at_utc"] = started_at
        summary["completed_at_utc"] = completed_at
        _atomic_json(output_dir / "summary.json", summary)
        _atomic_json(
            output_dir / "manifest.json",
            {
                **parent_manifest,
                "status": "completed",
                "completed_at_utc": completed_at,
                "mask_seeds": list(MASK_SEEDS),
                "mask_sha256": [child["selection"]["mask_sha256"] for child in children],
            },
        )
        print(
            f"[come group_random {config['dataset']} {config['transfer']} "
            f"rho={budget_key(config['selection']['requested_budget'])}] "
            f"Random mean PU={summary['PU-Acc']:.4f}+-{summary['PU-Acc-mask-std']:.4f} "
            f"FO={summary['FO-Acc']:.4f}+-{summary['FO-Acc-mask-std']:.4f}",
            flush=True,
        )
        return summary
    except BaseException as error:
        failed_at = utc_now()
        failure = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "status": "failed",
            "result_validity": "invalid",
            "invalid_reason": invalid_reason_of(error),
            "method": "come",
            "variant": "group_random",
            "dataset": config["dataset"],
            "transfer": config["transfer"],
            "requested_budget": config["selection"]["requested_budget"],
            "experiment_key": config["experiment_key"],
            "error_type": type(error).__name__,
            "error": str(error),
            "started_at_utc": started_at,
            "completed_at_utc": failed_at,
        }
        _atomic_json(output_dir / "summary.json", failure)
        _atomic_json(output_dir / "manifest.json", {**parent_manifest, **failure})
        raise


__all__ = ["run_mask_child", "run_transfer"]
