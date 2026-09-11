"""Rebuild COME Random parent/matrix aggregates from completed children.

Unlike the SHOT recovery tool this performs no artifact repair: the COME
sparse runner writes the authoritative ``selection`` record from the start, so
the only job here is to re-aggregate real completed children when a parent or
matrix step failed after the children finished.  No adaptation is rerun.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from transformer.source_only.runner import _atomic_json

from .aggregate import aggregate_matrix, print_aggregate, write_aggregate
from .common import utc_now
from .config import FORMAL_BUDGETS, GROUP_RANDOM, MASK_SEEDS, TRANSFERS, budget_tag
from .runner import _aggregate_children


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _backup_once(path: Path, backup_name: str) -> Path:
    backup = path.with_name(backup_name)
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def _load_child(mask_dir: Path, *, mask_index: int, mask_seed: int) -> dict:
    summary_path = mask_dir / "summary.json"
    mask_path = mask_dir / "mask.json"
    for path in (summary_path, mask_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required child artifact is missing: {path}")
    summary = _read_json(summary_path)
    mask = _read_json(mask_path)
    if summary.get("status") != "completed":
        raise ValueError(f"Child is not completed: {summary_path}")
    if summary.get("method") != "come":
        raise ValueError(f"Child is not a COME run: {summary_path}")
    if summary.get("random_mask_index") != mask_index:
        raise ValueError(f"Child mask index mismatch: {mask_dir}")
    if summary.get("mask_seed") != mask_seed or mask.get("mask_seed") != mask_seed:
        raise ValueError(f"Child deterministic mask seed mismatch: {mask_dir}")
    if summary["selection"].get("mask_sha256") != mask.get("mask_sha256"):
        raise ValueError(f"Child selection record and mask.json disagree: {mask_dir}")
    return summary


def recover_completed_run(run_root: Path, *, dry_run: bool = False) -> dict:
    """Rebuild every parent aggregate and the matrix from real children."""

    run_root = Path(run_root).resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(f"Run root does not exist: {run_root}")
    matrix_path = run_root / "matrix.json"
    if not matrix_path.is_file():
        raise FileNotFoundError(f"matrix.json is missing: {matrix_path}")

    parent_count = 0
    for budget in FORMAL_BUDGETS:
        for dataset, source, target in TRANSFERS:
            parent_dir = (
                run_root / "results" / budget_tag(budget) / dataset / f"{source}-{target}"
            )
            config_path = parent_dir / "effective_config.yaml"
            summary_path = parent_dir / "summary.json"
            manifest_path = parent_dir / "manifest.json"
            for path in (config_path, summary_path, manifest_path):
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Required parent artifact is missing: {path}"
                    )
            with open(config_path, "r", encoding="utf-8") as file_obj:
                config = yaml.safe_load(file_obj)
            children = [
                _load_child(
                    parent_dir / f"mask_{mask_index:02d}",
                    mask_index=mask_index,
                    mask_seed=mask_seed,
                )
                for mask_index, mask_seed in enumerate(MASK_SEEDS)
            ]
            old_summary = _read_json(summary_path)
            wall_runtime = sum(float(child["wall_runtime_sec"]) for child in children)
            parent_summary = _aggregate_children(config, children, wall_runtime)
            parent_summary.update(
                {
                    "started_at_utc": old_summary.get("started_at_utc"),
                    "completed_at_utc": old_summary.get("completed_at_utc"),
                    "recovered_from_completed_children": True,
                    "recovered_at_utc": utc_now(),
                    "wall_runtime_reconstructed_from_child_sum": True,
                }
            )
            parent_count += 1
            if dry_run:
                continue
            _backup_once(summary_path, "summary.failed_before_recovery.json")
            _atomic_json(summary_path, parent_summary)
            parent_manifest = _read_json(manifest_path)
            _backup_once(manifest_path, "manifest.failed_before_recovery.json")
            parent_manifest.update(
                {
                    "status": "completed",
                    "completed_at_utc": parent_summary["completed_at_utc"],
                    "mask_seeds": list(MASK_SEEDS),
                    "mask_sha256": [
                        child["selection"]["mask_sha256"] for child in children
                    ],
                    "recovered_from_completed_children": True,
                    "recovered_at_utc": parent_summary["recovered_at_utc"],
                }
            )
            for key in ("error_type", "error", "invalid_reason", "result_validity"):
                parent_manifest.pop(key, None)
            _atomic_json(manifest_path, parent_manifest)

    report = {
        "status": "validated" if dry_run else "completed",
        "method": "come",
        "variant": GROUP_RANDOM,
        "run_root": str(run_root),
        "parent_condition_count": parent_count,
        "child_run_count": parent_count * len(MASK_SEEDS),
        "gpu_experiments_rerun": 0,
        "dry_run": dry_run,
        "recovered_at_utc": utc_now(),
    }
    if dry_run:
        return report

    aggregate = aggregate_matrix(
        run_root,
        variant=GROUP_RANDOM,
        transfers=TRANSFERS,
        budgets=FORMAL_BUDGETS,
    )
    write_aggregate(run_root, aggregate, variant=GROUP_RANDOM)
    matrix = _read_json(matrix_path)
    _backup_once(matrix_path, "matrix.failed_before_recovery.json")
    matrix.update(
        {
            "status": "completed",
            "failures": [],
            "recovered_from_completed_children": True,
            "recovered_at_utc": report["recovered_at_utc"],
            "gpu_experiments_rerun": 0,
        }
    )
    _atomic_json(matrix_path, matrix)
    _atomic_json(run_root / "recovery.json", report)
    print_aggregate(aggregate, variant=GROUP_RANDOM)
    print(f"Recovered artifacts: {run_root}")
    return report


__all__ = ["recover_completed_run"]
