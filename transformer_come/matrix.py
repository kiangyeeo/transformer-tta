"""One-process-per-GPU COME matrix launcher; never called implicitly."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm.auto import tqdm

from .aggregate import aggregate_matrix, print_aggregate, write_aggregate
from .config import (
    CHILDREN_PER_CONDITION,
    FORMAL_SEED,
    IMPLEMENTATION_REVISIONS,
    PROTOCOL_REVISIONS,
    RUN_PREFIXES,
    SPARSE_VARIANTS,
    budget_tag,
    require_supported_variant,
    select_transfers,
)


def validate_devices(devices: list[str]) -> None:
    if not devices:
        raise ValueError("At least one GPU id, or cpu, is required")
    if "cpu" in devices and devices != ["cpu"]:
        raise ValueError("cpu cannot be mixed with CUDA device ids")
    if len(set(devices)) != len(devices):
        raise ValueError("Device ids must be unique")


def new_run_root(output_root: Path, variant: str) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = output_root / f"{RUN_PREFIXES[variant]}_{run_id}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def condition_tasks(transfers, budgets):
    if budgets is None:
        return [(dataset, source, target, None) for dataset, source, target in transfers]
    return [
        (dataset, source, target, budget)
        for budget in budgets
        for dataset, source, target in transfers
    ]


def result_dir_for(run_root: Path, dataset: str, source: str, target: str, budget) -> Path:
    base = run_root / "results"
    if budget is not None:
        base = base / budget_tag(budget)
    return base / dataset / f"{source}-{target}"


def run_matrix(
    *,
    variant: str,
    project_root: Path,
    config_path: Path,
    output_root: Path,
    selection: str,
    devices: list[str],
    budgets=None,
    lbi_overrides=None,
    allow_provisional_lbi: bool = False,
    stream_checkpoint: bool = True,
    resume_run_root: Path | None = None,
) -> Path:
    """Schedule every selected condition, then aggregate the real artifacts."""

    require_supported_variant(variant)
    if (budgets is None) == (variant in SPARSE_VARIANTS):
        raise ValueError(
            "sparse COME variants require budgets and dense variants reject them"
        )
    transfers = list(select_transfers(selection))
    tasks = condition_tasks(transfers, budgets)
    validate_devices(devices)
    if resume_run_root is not None and variant != "group_lbi":
        raise ValueError("Matrix resume is supported only for group_lbi")
    run_root = (
        resume_run_root.resolve()
        if resume_run_root is not None
        else new_run_root(output_root, variant)
    )
    if resume_run_root is not None and not run_root.is_dir():
        raise FileNotFoundError(f"Matrix resume root does not exist: {run_root}")
    logs_root = run_root / "logs"
    logs_root.mkdir(exist_ok=resume_run_root is not None)
    pending = []
    already_completed = 0
    for task in tasks:
        dataset, source, target, budget = task
        result_dir = result_dir_for(run_root, dataset, source, target, budget)
        summary_path = result_dir / "summary.json"
        if resume_run_root is not None and summary_path.is_file():
            with open(summary_path, "r", encoding="utf-8") as file_obj:
                completed_summary = json.load(file_obj)
            if (
                completed_summary.get("status") == "completed"
                and completed_summary.get("valid_lbi_run") is not False
            ):
                already_completed += 1
                continue
        pending.append(task)
    available = list(devices)
    active: dict[str, dict] = {}
    failures: list[dict] = []
    progress = tqdm(
        total=len(tasks), initial=already_completed,
        desc=f"come {variant} matrix", unit="condition"
    )

    while pending or active:
        while pending and available:
            dataset, source, target, budget = pending.pop(0)
            assigned = available.pop(0)
            task_name = f"{dataset}_{source}-{target}"
            if budget is not None:
                task_name = f"{budget_tag(budget)}_{task_name}"
            result_dir = result_dir_for(run_root, dataset, source, target, budget)
            log_path = logs_root / f"{task_name}.log"
            checkpoint_state = result_dir / ".stream_checkpoint" / "state.pt"
            resuming_condition = resume_run_root is not None and checkpoint_state.is_file()
            log_handle = open(
                log_path, "a" if resuming_condition else "w", encoding="utf-8"
            )
            if resume_run_root is not None and result_dir.exists() and not resuming_condition:
                log_handle.close()
                failures.append({
                    "task": task_name,
                    "returncode": None,
                    "log_path": str(log_path),
                    "error": "existing incomplete condition has no resumable checkpoint",
                })
                available.append(assigned)
                progress.update(1)
                continue
            if resuming_condition:
                command = [
                    sys.executable, "-m", "transformer_come", "transfer",
                    "--variant", "group_lbi", "--resume-run-dir", str(result_dir),
                ]
            else:
                command = [
                    sys.executable,
                    "-m",
                    "transformer_come",
                    "transfer",
                    "--config",
                    str(config_path),
                    "--variant",
                    variant,
                    "--dataset",
                    dataset,
                    "--source",
                    source,
                    "--target",
                    target,
                ]
            if not resuming_condition:
                if budget is not None:
                    command += ["--budget", str(budget)]
                if variant == "group_lbi":
                    for key, value in (lbi_overrides or {}).items():
                        command += ["--lbi-" + key.replace("_", "-"), str(value)]
                    if allow_provisional_lbi:
                        command.append("--allow-provisional-lbi")
                    if not stream_checkpoint:
                        command.append("--no-stream-checkpoint")
                command += [
                    "--device",
                    "cpu" if assigned == "cpu" else "cuda",
                    "--output-dir",
                    str(result_dir),
                ]
            environment = os.environ.copy()
            environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            if assigned != "cpu":
                environment["CUDA_VISIBLE_DEVICES"] = assigned
            process = subprocess.Popen(
                command,
                cwd=project_root,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            active[task_name] = {
                "process": process,
                "device": assigned,
                "log_path": str(log_path),
                "log_handle": log_handle,
            }

        completed = []
        for task_name, record in active.items():
            returncode = record["process"].poll()
            if returncode is None:
                continue
            record["log_handle"].close()
            available.append(record["device"])
            completed.append(task_name)
            if returncode != 0:
                failures.append(
                    {
                        "task": task_name,
                        "returncode": returncode,
                        "log_path": record["log_path"],
                    }
                )
            progress.update(1)
            progress.set_postfix(failed=len(failures))
        for task_name in completed:
            del active[task_name]
        if active and not completed:
            time.sleep(0.2)
    progress.close()

    matrix_record = {
        "status": "failed" if failures else "completed",
        "method": "come",
        "variant": variant,
        "protocol_revision": PROTOCOL_REVISIONS[variant],
        "implementation_revision": IMPLEMENTATION_REVISIONS[variant],
        "selection": selection,
        "formal_seed": FORMAL_SEED,
        "devices": devices,
        "condition_count": len(tasks),
        "already_completed_before_resume": already_completed,
        "resume_run_root": str(run_root) if resume_run_root is not None else None,
        "one_process_per_device": True,
        "failures": failures,
    }
    if budgets is not None:
        matrix_record["budgets"] = list(budgets)
    if variant == "group_lbi":
        matrix_record["lbi_overrides"] = dict(lbi_overrides or {})
        matrix_record["allow_provisional_lbi"] = bool(allow_provisional_lbi)
        matrix_record["stream_checkpoint_enabled"] = bool(stream_checkpoint)
    if CHILDREN_PER_CONDITION[variant] > 1:
        matrix_record["child_run_count"] = len(tasks) * CHILDREN_PER_CONDITION[variant]
    with open(run_root / "matrix.json", "w", encoding="utf-8") as file_obj:
        json.dump(matrix_record, file_obj, indent=2, ensure_ascii=False)
        file_obj.write("\n")
    if failures:
        details = ", ".join(
            f"{row['task']} (see {row['log_path']})" for row in failures
        )
        raise RuntimeError(f"COME {variant} matrix failed: {details}")

    aggregate = aggregate_matrix(
        run_root,
        variant=variant,
        transfers=tuple(transfers),
        budgets=None if budgets is None else tuple(budgets),
    )
    write_aggregate(run_root, aggregate, variant=variant)
    print_aggregate(aggregate, variant=variant)
    print(f"Artifacts: {run_root}")
    return run_root


__all__ = [
    "condition_tasks",
    "new_run_root",
    "result_dir_for",
    "run_matrix",
    "validate_devices",
]
