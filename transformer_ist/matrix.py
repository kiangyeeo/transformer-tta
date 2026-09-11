"""One-process-per-device IST matrix launcher; never called implicitly."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .aggregate import aggregate_matrix, write_aggregate
from .config import DENSE_VARIANTS, budget_tag, select_transfers


def validate_devices(devices):
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("devices must be a non-empty unique list")
    if "cpu" in devices and devices != ["cpu"]:
        raise ValueError("cpu cannot be mixed with CUDA device ids")


def run_matrix(
    *,
    project_root: Path,
    config_path: Path,
    output_root: Path,
    datasets: str,
    variants,
    budgets,
    devices,
):
    validate_devices(devices)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_root = output_root / f"ist_transformer_seed2026_{stamp}"
    (run_root / "logs").mkdir(parents=True, exist_ok=False)
    tasks = []
    for variant in variants:
        variant_budgets = (None,) if variant in DENSE_VARIANTS else budgets
        for budget in variant_budgets:
            for dataset, source, target in select_transfers(datasets):
                tasks.append((variant, budget, dataset, source, target))
    pending = list(tasks)
    available = list(devices)
    active = {}
    failures = []
    while pending or active:
        while pending and available:
            variant, budget, dataset, source, target = pending.pop(0)
            device = available.pop(0)
            budget_part = "" if budget is None else "/" + budget_tag(budget)
            result_dir = (
                run_root
                / "results"
                / variant
                / budget_part.lstrip("/")
                / dataset
                / f"{source}-{target}"
            )
            name = f"{variant}_{dataset}_{source}-{target}_{budget}"
            command = [
                sys.executable,
                "-m",
                "transformer_ist",
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
                "--device",
                "cpu" if device == "cpu" else "cuda",
                "--output-dir",
                str(result_dir),
                "--no-progress",
            ]
            if budget is not None:
                command += ["--budget", str(budget)]
            log_handle = open(
                run_root / "logs" / f"{name}.log", "w", encoding="utf-8"
            )
            environment = os.environ.copy()
            environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            if device != "cpu":
                environment["CUDA_VISIBLE_DEVICES"] = device
            process = subprocess.Popen(
                command,
                cwd=project_root,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            active[name] = (process, device, log_handle)
        completed = []
        for name, (process, device, log_handle) in active.items():
            returncode = process.poll()
            if returncode is None:
                continue
            log_handle.close()
            available.append(device)
            completed.append(name)
            if returncode:
                failures.append({"task": name, "returncode": returncode})
        for name in completed:
            del active[name]
        if active and not completed:
            time.sleep(0.2)
    record = {
        "status": "failed" if failures else "completed",
        "method": "ist",
        "variants": list(variants),
        "budgets": list(budgets),
        "condition_count": len(tasks),
        "failures": failures,
        "one_process_per_device": True,
    }
    with open(run_root / "matrix.json", "w", encoding="utf-8") as file_obj:
        json.dump(record, file_obj, indent=2)
        file_obj.write("\n")
    if failures:
        raise RuntimeError(f"IST matrix failed: {failures}")
    aggregate = aggregate_matrix(run_root)
    write_aggregate(run_root, aggregate)
    return run_root
