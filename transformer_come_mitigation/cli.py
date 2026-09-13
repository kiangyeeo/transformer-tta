"""CLI and one-process-per-GPU scheduler for COME mitigation pilots."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .experiments import (
    EXPERIMENT_SUITES,
    IMPLEMENTATION_REVISION,
    PROTOCOL_REVISION,
    experiment_by_id,
)
from .runner import PROJECT_ROOT, resolve_experiment_config, run_experiment


DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT.parent / "results" / "transformer_come_mitigation"
)


def _parser():
    parser = argparse.ArgumentParser(description="COME VisDA collapse mitigation pilots")
    subparsers = parser.add_subparsers(dest="command", required=True)

    transfer = subparsers.add_parser("transfer")
    transfer.add_argument("--experiment", required=True)
    transfer.add_argument("--output-dir", type=Path, required=True)
    transfer.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    transfer.add_argument("--dry-run", action="store_true")

    matrix = subparsers.add_parser("matrix")
    matrix.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    matrix.add_argument(
        "--suite", choices=tuple(EXPERIMENT_SUITES), default="initial-8gpu"
    )
    matrix.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    matrix.add_argument("--dry-run", action="store_true")
    return parser


def _run_root(output_root: Path, suite: str):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return output_root.resolve() / f"visda_{suite}_{stamp}"


def _plan(run_root: Path, devices: list[str], *, suite: str):
    experiments = EXPERIMENT_SUITES[suite]
    if len(devices) != len(experiments) or len(set(devices)) != len(devices):
        raise ValueError(
            f"Suite {suite} requires {len(experiments)} unique devices"
        )
    tasks = []
    for experiment, device in zip(experiments, devices):
        output_dir = run_root / "results" / experiment["id"]
        config = resolve_experiment_config(experiment, output_dir, device="cuda")
        tasks.append(
            {
                "experiment": experiment,
                "physical_device": device,
                "output_dir": str(output_dir),
                "experiment_key": config["experiment_key"],
                "scientific_config_sha256": config["scientific_config_sha256"],
            }
        )
    return {
        "status": "planned",
        "study": "come_collapse_mitigation_pilot",
        "suite": suite,
        "formal": False,
        "dataset": "visda-c",
        "transfer": "train->validation",
        "formal_seed": 2026,
        "protocol_revision": PROTOCOL_REVISION,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "one_process_per_device": True,
        "tasks": tasks,
    }


def _write_json(path: Path, payload):
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)
        file_obj.write("\n")


def _aggregate(run_root: Path, plan: dict):
    rows = []
    for task in plan["tasks"]:
        summary_path = Path(task["output_dir"]) / "summary.json"
        with open(summary_path, "r", encoding="utf-8") as file_obj:
            summary = json.load(file_obj)
        experiment = task["experiment"]
        rows.append(
            {
                "experiment": experiment["id"],
                "scope": experiment["scope"],
                "optimizer": experiment["optimizer"],
                "lr": experiment["lr"],
                "tau": experiment["tau"],
                "gradient_clip_norm": experiment.get("gradient_clip_norm"),
                "PU": summary["PU-Acc"],
                "FO": summary["FO-Acc"],
                "PU_overall": summary["PU-overall-Acc"],
                "FO_overall": summary["FO-overall-Acc"],
                "predicted_class_count_min": summary["predicted_class_count_min"],
                "dominant_class_ratio_mean": summary["dominant_class_ratio_mean"],
                "dominant_class_ratio_max": summary["dominant_class_ratio_max"],
                "trainable_scalars": summary["trainable_scalars"],
                "status": summary["status"],
                "summary": str(summary_path),
            }
        )
    rows.sort(key=lambda row: row["experiment"])
    aggregate = {**plan, "status": "completed", "results": rows}
    _write_json(run_root / "aggregate.json", aggregate)
    with open(run_root / "results.csv", "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return aggregate


def _run_matrix(
    devices: list[str], output_root: Path, *, suite: str, dry_run: bool
):
    run_root = _run_root(output_root, suite)
    plan = _plan(run_root, devices, suite=suite)
    if dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return None

    run_root.mkdir(parents=True, exist_ok=False)
    logs_root = run_root / "logs"
    logs_root.mkdir()
    _write_json(run_root / "plan.json", plan)
    active = []
    for task in plan["tasks"]:
        experiment_id = task["experiment"]["id"]
        log_path = logs_root / f"{experiment_id}.log"
        log_handle = open(log_path, "w", encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "transformer_come_mitigation",
            "transfer",
            "--experiment",
            experiment_id,
            "--output-dir",
            task["output_dir"],
            "--device",
            "cuda",
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = task["physical_device"]
        environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        active.append((experiment_id, task["physical_device"], process, log_handle, log_path))

    failures = []
    while active:
        remaining = []
        for experiment_id, device, process, handle, log_path in active:
            returncode = process.poll()
            if returncode is None:
                remaining.append((experiment_id, device, process, handle, log_path))
                continue
            handle.close()
            if returncode:
                failures.append(
                    {
                        "experiment": experiment_id,
                        "device": device,
                        "returncode": returncode,
                        "log": str(log_path),
                    }
                )
            print(f"[{experiment_id}] finished rc={returncode} gpu={device}", flush=True)
        active = remaining
        if active:
            time.sleep(1.0)

    if failures:
        failed = {**plan, "status": "failed", "failures": failures}
        _write_json(run_root / "matrix.json", failed)
        raise RuntimeError(f"COME mitigation matrix failed: {failures}")
    matrix = {**plan, "status": "completed", "failures": []}
    _write_json(run_root / "matrix.json", matrix)
    aggregate = _aggregate(run_root, plan)
    for row in aggregate["results"]:
        print(
            f"{row['experiment']}: PU={row['PU']:.4f} FO={row['FO']:.4f} "
            f"dominant_mean={row['dominant_class_ratio_mean']:.4f}",
            flush=True,
        )
    print(f"Artifacts: {run_root}", flush=True)
    return run_root


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command == "transfer":
        experiment = experiment_by_id(args.experiment)
        if args.dry_run:
            config = resolve_experiment_config(
                experiment, args.output_dir, device=args.device
            )
            print(json.dumps(config, indent=2, ensure_ascii=False))
            return 0
        run_experiment(experiment, args.output_dir, device=args.device)
        return 0

    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    _run_matrix(
        devices, args.output_root, suite=args.suite, dry_run=args.dry_run
    )
    return 0


__all__ = ["DEFAULT_OUTPUT_ROOT", "main"]
