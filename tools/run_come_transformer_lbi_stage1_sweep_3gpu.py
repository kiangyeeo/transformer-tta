#!/usr/bin/env python3
"""Queue the frozen COME-Transformer Stage-1 screen on three GPUs.

Queue order is intentionally scientific and stable:

1. VisDA-C, rho 0.002 -> 0.001 -> 0.0005;
2. Office-31, rho 0.002 -> 0.001 -> 0.0005.

Each dataset/budget has the 18 alpha x nu configurations from the protocol.
An Office configuration expands to all six transfers with one shared tuple.
Completed (including scientifically invalid cap-hit) summaries are terminal;
an interrupted condition resumes only from its complete online-batch checkpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "transformer_come" / "config.yaml"
DEFAULT_OUTPUT_ROOT = Path(
    "/home/nas3/biod/wangkangyi/results/transformer_come_lbi_stage1_sweep_20260913"
)
PYTHON_BIN = Path("/home/nas3/biod/wangkangyi/envs/lbi/bin/python")
TMP_ROOT = Path("/home/nas3/biod/wangkangyi/tmp/transformer_come_lbi_stage1_sweep")

ALPHAS = (0.025, 0.05, 0.10, 0.125, 0.15, 0.20)
NUS = (0.25, 0.50, 1.00)
BUDGETS = (0.002, 0.001, 0.0005)
DATASET_ORDER = ("visda-c", "office31")
TRANSFERS = {
    "visda-c": (("train", "validation"),),
    "office31": (
        ("amazon", "dslr"),
        ("amazon", "webcam"),
        ("dslr", "amazon"),
        ("dslr", "webcam"),
        ("webcam", "amazon"),
        ("webcam", "dslr"),
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def tag(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def number(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def task_label(task: "Task") -> str:
    return (
        f"#{task.queue_index} {task.dataset} rho={number(task.budget)} "
        f"alpha={number(task.alpha)} nu={number(task.nu)} "
        f"{task.source}->{task.target}"
    )


@dataclass(frozen=True)
class Task:
    queue_index: int
    dataset: str
    budget: float
    alpha: float
    nu: float
    source: str
    target: str
    output_dir: str
    log_path: str


def build_tasks(output_root: Path, dataset_order=DATASET_ORDER) -> list[Task]:
    tasks: list[Task] = []
    for dataset in dataset_order:
        for budget in BUDGETS:
            for alpha in ALPHAS:
                for nu in NUS:
                    profile = f"alpha-{tag(alpha)}_nu-{tag(nu)}"
                    for source, target in TRANSFERS[dataset]:
                        relative = (
                            Path(f"rho-{number(budget)}")
                            / dataset
                            / profile
                            / f"{source}-{target}"
                        )
                        tasks.append(
                            Task(
                                queue_index=len(tasks),
                                dataset=dataset,
                                budget=budget,
                                alpha=alpha,
                                nu=nu,
                                source=source,
                                target=target,
                                output_dir=str(output_root / "results" / relative),
                                log_path=str(output_root / "logs" / relative.with_suffix(".log")),
                            )
                        )
    return tasks


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)
        file_obj.write("\n")
    os.replace(temporary, path)


def summary_state(task: Task) -> str | None:
    path = Path(task.output_dir) / "summary.json"
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as file_obj:
            summary = json.load(file_obj)
    except (OSError, json.JSONDecodeError):
        return None
    if summary.get("status") != "completed":
        return None
    if summary.get("valid_lbi_run") is False:
        return "completed_invalid"
    return "completed_valid"


def command_for(
    task: Task, config_path: Path, *, omega: float, stage2_lr: float
) -> tuple[list[str], bool]:
    output_dir = Path(task.output_dir)
    checkpoint = output_dir / ".stream_checkpoint" / "state.pt"
    if checkpoint.is_file():
        return (
            [
                str(PYTHON_BIN),
                "-m",
                "transformer_come",
                "transfer",
                "--variant",
                "group_lbi",
                "--resume-run-dir",
                str(output_dir),
                "--no-progress",
            ],
            True,
        )
    if output_dir.exists():
        raise RuntimeError(
            f"Existing non-terminal output has no resumable checkpoint: {output_dir}"
        )
    return (
        [
            str(PYTHON_BIN),
            "-m",
            "transformer_come",
            "transfer",
            "--config",
            str(config_path),
            "--variant",
            "group_lbi",
            "--dataset",
            task.dataset,
            "--source",
            task.source,
            "--target",
            task.target,
            "--rho",
            str(task.budget),
            "--device",
            "cuda",
            "--output-dir",
            task.output_dir,
            "--allow-provisional-lbi",
            "--lbi-alpha",
            str(task.alpha),
            "--lbi-kappa",
            "1.0",
            "--lbi-nu",
            str(task.nu),
            "--lbi-omega",
            str(omega),
            "--lbi-stage2-lr",
            str(stage2_lr),
            "--no-progress",
        ],
        False,
    )


def status_payload(
    tasks: list[Task], active: dict[str, dict], failures: list[dict], started: str
) -> dict:
    states = [summary_state(task) for task in tasks]
    return {
        "schema_version": 1,
        "status": "running" if active or any(state is None for state in states) else "completed",
        "started_at_utc": started,
        "updated_at_utc": utc_now(),
        "task_count": len(tasks),
        "dataset_budget_configuration_count": len(
            {(task.dataset, task.budget) for task in tasks}
        )
        * 18,
        "transfer_run_count": len(tasks),
        "completed_valid": sum(state == "completed_valid" for state in states),
        "completed_invalid": sum(state == "completed_invalid" for state in states),
        "remaining": sum(state is None for state in states),
        "active": [
            {
                "gpu": gpu,
                "pid": record["process"].pid,
                **asdict(record["task"]),
            }
            for gpu, record in sorted(active.items())
        ],
        "failures": failures,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument(
        "--datasets",
        choices=("all", "visda-c", "office31"),
        default="all",
        help="dataset queue to run; default is the full VisDA-C then Office-31 sweep",
    )
    parser.add_argument("--omega", type=float, default=0.20)
    parser.add_argument("--stage2-lr", type=float, default=1.0e-5)
    parser.add_argument(
        "--no-phase-barrier",
        action="store_true",
        help="let an idle GPU start the next dataset/budget phase immediately",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("--gpus must be a non-empty unique list")
    if not args.config.is_file() or not PYTHON_BIN.is_file():
        raise FileNotFoundError("COME config or lbi Python interpreter is missing")

    output_root = args.output_root.resolve()
    selected_datasets = (
        DATASET_ORDER if args.datasets == "all" else (args.datasets,)
    )
    tasks = build_tasks(output_root, selected_datasets)
    plan = {
        "schema_version": 1,
        "method": "come",
        "variant": "group_lbi",
        "phase": "stage1_screen",
        "queue_order": [
            f"{dataset}/rho-{number(budget)}"
            for dataset in selected_datasets
            for budget in BUDGETS
        ],
        "datasets": list(selected_datasets),
        "alpha": list(ALPHAS),
        "kappa": 1.0,
        "nu": list(NUS),
        "omega": args.omega,
        "stage2_lr": args.stage2_lr,
        "stage2_steps": 1,
        "stage1_max_steps": 3000,
        "phase_barrier": not args.no_phase_barrier,
        "dataset_budget_configuration_count": len(
            {(task.dataset, task.budget) for task in tasks}
        )
        * 18,
        "transfer_run_count": len(tasks),
        "gpus": gpus,
        "config": str(args.config.resolve()),
        "output_root": str(output_root),
        "tasks": [asdict(task) for task in tasks],
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "plan.json", plan)
    started = utc_now()
    pending = [task for task in tasks if summary_state(task) is None]
    available = list(gpus)
    active: dict[str, dict] = {}
    failures: list[dict] = []
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HOME": "/home/nas3/biod/wangkangyi/hf-cache",
            "TORCH_HOME": "/home/nas3/biod/wangkangyi/hf-cache/torch",
            "PIP_CACHE_DIR": "/home/nas3/biod/wangkangyi/pip-cache",
            "CONDA_PKGS_DIRS": "/home/nas3/biod/wangkangyi/conda-pkgs",
            "TMPDIR": str(TMP_ROOT),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "OMP_NUM_THREADS": "4",
        }
    )
    print(
        f"[{utc_now()}] queued {len(pending)}/{len(tasks)} transfer runs "
        f"on CUDA_VISIBLE_DEVICES={','.join(gpus)}",
        flush=True,
    )

    while pending or active:
        # By default a dataset-budget phase is a strict scheduling barrier.
        # Screening sweeps may opt into work-conserving cross-phase scheduling.
        current_phase = (
            (next(iter(active.values()))["task"].dataset,
             next(iter(active.values()))["task"].budget)
            if active
            else (pending[0].dataset, pending[0].budget)
        )
        while (
            pending
            and available
            and (
                args.no_phase_barrier
                or (pending[0].dataset, pending[0].budget) == current_phase
            )
        ):
            task = pending.pop(0)
            gpu = available.pop(0)
            try:
                command, resumed = command_for(
                    task,
                    args.config.resolve(),
                    omega=args.omega,
                    stage2_lr=args.stage2_lr,
                )
            except RuntimeError as error:
                failures.append(
                    {"queue_index": task.queue_index, "error": str(error), **asdict(task)}
                )
                print(f"[{utc_now()}] SKIP {task_label(task)}: {error}", flush=True)
                available.append(gpu)
                continue
            log_path = Path(task.log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = open(log_path, "a" if resumed else "w", encoding="utf-8")
            child_environment = environment.copy()
            child_environment["CUDA_VISIBLE_DEVICES"] = gpu
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=child_environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            active[gpu] = {
                "process": process,
                "task": task,
                "log_handle": log_handle,
                "resumed": resumed,
            }
            action = "RESUME" if resumed else "START"
            print(
                f"[{utc_now()}] GPU {gpu} {action} {task_label(task)} "
                f"log={task.log_path}",
                flush=True,
            )
        atomic_json(output_root / "status.json", status_payload(tasks, active, failures, started))

        completed_gpus = []
        for gpu, record in active.items():
            returncode = record["process"].poll()
            if returncode is None:
                continue
            record["log_handle"].close()
            task = record["task"]
            state = summary_state(task)
            if returncode != 0 or state is None:
                failures.append(
                    {
                        "queue_index": task.queue_index,
                        "gpu": gpu,
                        "returncode": returncode,
                        "summary_state": state,
                        "log_path": task.log_path,
                    }
                )
                print(
                    f"[{utc_now()}] GPU {gpu} FAIL rc={returncode} "
                    f"state={state} {task_label(task)}",
                    flush=True,
                )
            else:
                print(
                    f"[{utc_now()}] GPU {gpu} DONE state={state} "
                    f"{task_label(task)}",
                    flush=True,
                )
            completed_gpus.append(gpu)
        for gpu in completed_gpus:
            del active[gpu]
            available.append(gpu)
        if active and not completed_gpus:
            time.sleep(2.0)

    final = status_payload(tasks, active, failures, started)
    final["status"] = "completed_with_failures" if failures else "completed"
    final["completed_at_utc"] = utc_now()
    atomic_json(output_root / "status.json", final)
    print(
        f"[{utc_now()}] {final['status']}: "
        f"valid={final['completed_valid']} invalid={final['completed_invalid']} "
        f"failures={len(failures)}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
