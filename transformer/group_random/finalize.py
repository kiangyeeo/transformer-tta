"""Recover parent and matrix aggregates from completed Group-Random children."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

from transformer.source_only.runner import _atomic_json

from .aggregate import aggregate_matrix, print_aggregate, write_aggregate
from .config import FORMAL_BUDGETS, MASK_SEEDS, TRANSFERS, budget_tag
from .runner import _aggregate_children


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _backup_once(path: Path, backup_name: str) -> Path:
    backup = path.with_name(backup_name)
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as file_obj:
            for row in rows:
                file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_repaired_child(mask_dir: Path, *, dry_run: bool) -> tuple[dict, bool]:
    summary_path = mask_dir / "summary.json"
    mask_path = mask_dir / "mask.json"
    metrics_path = mask_dir / "metrics.jsonl"
    for path in (summary_path, mask_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required child artifact is missing: {path}")
    summary = _read_json(summary_path)
    mask = _read_json(mask_path)
    if summary.get("status") != "completed":
        raise ValueError(f"Child is not completed: {summary_path}")
    if mask.get("mask_seed") != summary.get("mask_seed"):
        raise ValueError(f"Child mask seed mismatch: {mask_dir}")
    needs_repair = not isinstance(summary.get("selection"), dict)
    if not needs_repair:
        needs_repair = (
            summary["selection"].get("mask_sha256") != mask.get("mask_sha256")
        )
    summary["selection"] = mask
    summary["selection_method"] = "uniform_structural_group_random"
    if not needs_repair or dry_run:
        return summary, needs_repair

    recovered_at = _utc_now()
    _backup_once(summary_path, "summary.pre_recovery.json")
    summary["metadata_recovered_at_utc"] = recovered_at
    summary["metadata_recovery_reason"] = "selection_scope_key_collision"
    _atomic_json(summary_path, summary)

    metric_rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    final_rows = [row for row in metric_rows if row.get("event") == "final"]
    if len(final_rows) != 1:
        raise ValueError(f"Expected exactly one final metrics row: {metrics_path}")
    final_rows[0]["selection"] = mask
    final_rows[0]["selection_method"] = "uniform_structural_group_random"
    final_rows[0]["metadata_recovered_at_utc"] = recovered_at
    final_rows[0]["metadata_recovery_reason"] = "selection_scope_key_collision"
    _backup_once(metrics_path, "metrics.pre_recovery.jsonl")
    _atomic_jsonl(metrics_path, metric_rows)
    return summary, True


def recover_completed_run(run_root: Path, *, dry_run: bool = False) -> dict:
    """Rebuild the complete 21-condition matrix without rerunning adaptation."""
    run_root = run_root.resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(f"Run root does not exist: {run_root}")
    matrix_path = run_root / "matrix.json"
    if not matrix_path.is_file():
        raise FileNotFoundError(f"matrix.json is missing: {matrix_path}")

    repaired_children = 0
    parent_count = 0
    for budget in FORMAL_BUDGETS:
        for dataset, source, target in TRANSFERS:
            parent_dir = (
                run_root
                / "results"
                / budget_tag(budget)
                / dataset
                / f"{source}-{target}"
            )
            config_path = parent_dir / "effective_config.yaml"
            summary_path = parent_dir / "summary.json"
            manifest_path = parent_dir / "manifest.json"
            for path in (config_path, summary_path, manifest_path):
                if not path.is_file():
                    raise FileNotFoundError(f"Required parent artifact is missing: {path}")
            with open(config_path, "r", encoding="utf-8") as file_obj:
                config = yaml.safe_load(file_obj)
            children = []
            for mask_index, mask_seed in enumerate(MASK_SEEDS):
                child, repaired = _load_repaired_child(
                    parent_dir / f"mask_{mask_index:02d}", dry_run=dry_run
                )
                if child.get("random_mask_index") != mask_index:
                    raise ValueError(f"Child mask index mismatch: {parent_dir}")
                if child.get("mask_seed") != mask_seed:
                    raise ValueError(f"Child deterministic mask seed mismatch: {parent_dir}")
                repaired_children += int(repaired)
                children.append(child)

            old_summary = _read_json(summary_path)
            wall_runtime = sum(float(child["wall_runtime_sec"]) for child in children)
            parent_summary = _aggregate_children(config, children, wall_runtime)
            parent_summary.update(
                {
                    "started_at_utc": old_summary.get("started_at_utc"),
                    "completed_at_utc": old_summary.get("completed_at_utc"),
                    "recovered_from_completed_children": True,
                    "recovered_at_utc": _utc_now(),
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
            for key in ("error_type", "error"):
                parent_manifest.pop(key, None)
            _atomic_json(manifest_path, parent_manifest)

    report = {
        "status": "validated" if dry_run else "completed",
        "variant": "group_random",
        "run_root": str(run_root),
        "parent_condition_count": parent_count,
        "child_run_count": parent_count * len(MASK_SEEDS),
        "child_selection_records_requiring_repair": repaired_children,
        "gpu_experiments_rerun": 0,
        "dry_run": dry_run,
        "recovered_at_utc": _utc_now(),
    }
    if dry_run:
        return report

    aggregate = aggregate_matrix(
        run_root, transfers=TRANSFERS, budgets=FORMAL_BUDGETS
    )
    write_aggregate(run_root, aggregate)
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
    print_aggregate(aggregate)
    print(f"Recovered artifacts: {run_root}")
    return report
