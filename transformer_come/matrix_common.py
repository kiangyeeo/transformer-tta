"""One-process-per-GPU scheduler shared by the COME variants."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm.auto import tqdm

from .budget import budget_tag


def validate_devices(devices: list[str]) -> None:
    if not devices:
        raise ValueError("At least one GPU id, or cpu, is required")
    if "cpu" in devices and devices != ["cpu"]:
        raise ValueError("cpu cannot be mixed with CUDA device ids")
    if len(set(devices)) != len(devices):
        raise ValueError("Device ids must be unique")


def new_run_root(output_root: Path, prefix: str) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = output_root / f"{prefix}_{run_id}"
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
    spec,
    project_root: Path,
    config_path: Path,
    output_root: Path,
    selection: str,
    devices: list[str],
    budgets=None,
) -> Path:
    """Schedule every selected condition, then aggregate the real artifacts."""

    transfers = list(spec.select_transfers(selection))
    tasks = condition_tasks(transfers, budgets)
    validate_devices(devices)
    run_root = new_run_root(output_root, spec.run_prefix)
    logs_root = run_root / "logs"
    logs_root.mkdir()
    pending = list(tasks)
    available = list(devices)
    active: dict[str, dict] = {}
    failures: list[dict] = []
    progress = tqdm(
        total=len(tasks), desc=f"come {spec.variant} matrix", unit="condition"
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
            log_handle = open(log_path, "w", encoding="utf-8")
            command = [
                sys.executable,
                "-m",
                spec.module,
                "transfer",
                "--config",
                str(config_path),
                "--dataset",
                dataset,
                "--source",
                source,
                "--target",
                target,
            ]
            if budget is not None:
                command += ["--budget", str(budget)]
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
        "variant": spec.variant,
        "protocol_revision": spec.protocol_revision,
        "implementation_revision": spec.implementation_revision,
        "selection": selection,
        "formal_seed": 2026,
        "devices": devices,
        "condition_count": len(tasks),
        "one_process_per_device": True,
        "failures": failures,
    }
    if budgets is not None:
        matrix_record["budgets"] = list(budgets)
    if spec.children_per_condition > 1:
        matrix_record["child_run_count"] = len(tasks) * spec.children_per_condition
    with open(run_root / "matrix.json", "w", encoding="utf-8") as file_obj:
        json.dump(matrix_record, file_obj, indent=2, ensure_ascii=False)
        file_obj.write("\n")
    if failures:
        details = ", ".join(
            f"{row['task']} (see {row['log_path']})" for row in failures
        )
        raise RuntimeError(f"COME {spec.variant} matrix failed: {details}")

    aggregate_matrix, write_aggregate, print_aggregate = spec.aggregate_functions()
    if budgets is None:
        aggregate = aggregate_matrix(run_root, transfers=tuple(transfers))
    else:
        aggregate = aggregate_matrix(
            run_root, transfers=tuple(transfers), budgets=tuple(budgets)
        )
    write_aggregate(run_root, aggregate)
    print_aggregate(aggregate)
    print(f"Artifacts: {run_root}")
    return run_root


__all__ = ["condition_tasks", "new_run_root", "result_dir_for", "run_matrix", "validate_devices"]
