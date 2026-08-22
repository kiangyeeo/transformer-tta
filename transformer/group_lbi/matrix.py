"""One-process-per-GPU scheduler for Group-LBI transfer/budget conditions."""

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
from .config import budget_tag, select_transfers


def new_run_root(output_root: Path) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = output_root / f"group_lbi_seed2026_{run_id}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def validate_devices(devices: list[str]) -> None:
    if not devices:
        raise ValueError("At least one GPU id, or cpu, is required")
    if "cpu" in devices and devices != ["cpu"]:
        raise ValueError("cpu cannot be mixed with CUDA device ids")
    if len(set(devices)) != len(devices):
        raise ValueError("Device ids must be unique")


def run_matrix(
    *,
    project_root: Path,
    config_path: Path,
    output_root: Path,
    selection: str,
    budgets: tuple[float, ...],
    devices: list[str],
    lbi_overrides: dict,
    stream_checkpoint: bool,
    resume_run_root: Path | None = None,
) -> Path:
    transfers = list(select_transfers(selection))
    tasks = [(*transfer, budget) for budget in budgets for transfer in transfers]
    validate_devices(devices)
    run_root = resume_run_root.resolve() if resume_run_root else new_run_root(output_root)
    if resume_run_root and not run_root.is_dir():
        raise FileNotFoundError(f"Matrix resume root does not exist: {run_root}")
    logs_root = run_root / "logs"
    logs_root.mkdir(exist_ok=True)
    pending = []
    already_completed = 0
    failures: list[dict] = []
    for task in tasks:
        dataset, source, target, budget = task
        result_dir = run_root / "results" / budget_tag(budget) / dataset / f"{source}-{target}"
        summary_path = result_dir / "summary.json"
        if resume_run_root and summary_path.is_file():
            try:
                with open(summary_path, "r", encoding="utf-8") as file_obj:
                    if json.load(file_obj).get("status") == "completed":
                        already_completed += 1
                        continue
            except (OSError, json.JSONDecodeError):
                pass
        pending.append(task)

    available = list(devices)
    active: dict[str, dict] = {}
    progress = tqdm(total=len(tasks), initial=already_completed, desc="group-lbi matrix", unit="condition")

    while pending or active:
        while pending and available:
            dataset, source, target, budget = pending.pop(0)
            assigned = available.pop(0)
            task_name = f"{budget_tag(budget)}_{dataset}_{source}-{target}"
            result_dir = run_root / "results" / budget_tag(budget) / dataset / f"{source}-{target}"
            checkpoint_state = result_dir / ".stream_checkpoint" / "state.pt"
            log_path = logs_root / f"{task_name}.log"
            log_handle = open(log_path, "a" if checkpoint_state.is_file() else "w", encoding="utf-8")
            command = [sys.executable, "-m", "transformer.group_lbi", "transfer"]
            if checkpoint_state.is_file():
                command.extend(["--resume-run-dir", str(result_dir)])
            else:
                if result_dir.exists():
                    log_handle.close()
                    failures.append(
                        {
                            "task": task_name,
                            "returncode": None,
                            "log_path": str(log_path),
                            "error": "existing incomplete directory has no resumable checkpoint",
                        }
                    )
                    available.append(assigned)
                    progress.update(1)
                    continue
                command.extend(
                    [
                        "--config", str(config_path),
                        "--dataset", dataset,
                        "--source", source,
                        "--target", target,
                        "--budget", str(budget),
                        "--device", "cpu" if assigned == "cpu" else "cuda",
                        "--output-dir", str(result_dir),
                    ]
                )
                for key, value in lbi_overrides.items():
                    command.extend(["--lbi-" + key.replace("_", "-"), str(value)])
                if not stream_checkpoint:
                    command.append("--no-stream-checkpoint")
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
        "variant": "group_lbi",
        "selection": selection,
        "formal_seed": 2026,
        "budgets": list(budgets),
        "devices": devices,
        "condition_count": len(tasks),
        "already_completed_before_resume": already_completed,
        "resume_run_root": str(resume_run_root.resolve()) if resume_run_root else None,
        "one_process_per_device": True,
        "stream_checkpoint_enabled": stream_checkpoint,
        "lbi_overrides": lbi_overrides,
        "failures": failures,
    }
    with open(run_root / "matrix.json", "w", encoding="utf-8") as file_obj:
        json.dump(matrix_record, file_obj, indent=2, ensure_ascii=False)
        file_obj.write("\n")
    if failures:
        details = ", ".join(f"{row['task']} (see {row['log_path']})" for row in failures)
        raise RuntimeError(f"Group-LBI matrix failed: {details}")

    aggregate = aggregate_matrix(run_root, transfers=tuple(transfers), budgets=budgets)
    write_aggregate(run_root, aggregate)
    print_aggregate(aggregate)
    print(f"Artifacts: {run_root}")
    return run_root
