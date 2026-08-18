#!/usr/bin/env python3
"""Dispatch DeiT OTTA saliency structural-group runs across GPUs.

Greedy scheduler: at most one subprocess per GPU, refilling free GPUs as jobs
finish.  The job set is ``budgets x directions`` where directions are the six
Office-31 ordered transfers plus VisDA-C train->validation (seven total).
Each job rebuilds its Top-K gradient-saliency mask every online step and
writes a single deterministic ``summary.json``.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ENTRYPOINT = PROJECT_ROOT / "evaluate_deit_otta_saliency.py"
OFFICE_DIRECTIONS = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
VISDA_DIRECTION = ((0, 1),)
DEFAULT_BUDGETS = ("0.005", "0.01", "0.02", "0.001", "0.003")
POLL_SECONDS = 5.0


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def config_output_root():
    config_path = PROJECT_ROOT / "configs" / "deit_otta_group_saliency.yaml"
    with open(config_path, "r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)
    root = config.get("output", {}).get("root")
    if not root:
        raise RuntimeError(f"config has no output.root: {config_path}")
    return Path(root).resolve()


def load_failed_jobs(summary_path):
    """Load the failed jobs recorded in a previous launcher summary."""
    if not summary_path.is_file():
        raise SystemExit(f"no launcher summary to resume from: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    failed = []
    for result in summary.get("results", []):
        if result.get("return_code") != 0:
            failed.append(
                {
                    "budget": result["budget"],
                    "dataset": result["dataset"],
                    "source": int(result["source"]),
                    "target": int(result["target"]),
                }
            )
    if not failed:
        raise SystemExit("no failed jobs to resume")
    return failed


def build_jobs(budgets, datasets):
    jobs = []
    for budget in budgets:
        if "office31" in datasets:
            for source, target in OFFICE_DIRECTIONS:
                jobs.append(
                    {
                        "budget": budget,
                        "dataset": "office31",
                        "source": int(source),
                        "target": int(target),
                    }
                )
        if "visda-c" in datasets:
            for source, target in VISDA_DIRECTION:
                jobs.append(
                    {
                        "budget": budget,
                        "dataset": "visda-c",
                        "source": int(source),
                        "target": int(target),
                    }
                )
    return jobs


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run DeiT OTTA saliency group baseline across GPUs."
    )
    parser.add_argument(
        "--budgets",
        default=",".join(DEFAULT_BUDGETS),
        help="Comma-separated structural budgets (default: 0.005,0.01,0.02,0.001,0.003).",
    )
    parser.add_argument(
        "--gpus",
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated GPU ids to schedule onto.",
    )
    parser.add_argument(
        "--datasets",
        default="office31,visda-c",
        help="Comma-separated datasets (office31,visda-c).",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data-root")
    parser.add_argument("--source-checkpoint-root")
    parser.add_argument("--output-root")
    parser.add_argument(
        "--resume-failed",
        action="store_true",
        help="Re-run only the jobs that failed in a previous launcher summary.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the job->GPU assignment without launching anything.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise SystemExit("no GPUs selected")

    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else config_output_root()
    )
    log_dir = output_root / "launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    resume = bool(args.resume_failed)
    log_prefix = "retry_" if resume else ""
    summary_filename = (
        "launcher_summary_retry.json" if resume else "launcher_summary.json"
    )

    if resume:
        jobs = load_failed_jobs(log_dir / "launcher_summary.json")
        budgets = sorted({job["budget"] for job in jobs})
    else:
        budgets = [
            float(value.strip())
            for value in args.budgets.split(",")
            if value.strip()
        ]
        if not budgets:
            raise SystemExit("no budgets selected")
        datasets = [
            value.strip()
            for value in args.datasets.split(",")
            if value.strip()
        ]
        jobs = build_jobs(budgets, datasets)
    total_jobs = len(jobs)
    if total_jobs == 0:
        raise SystemExit("no jobs to run")

    print(
        f"jobs={total_jobs} gpus={gpus} budgets={budgets} "
        f"resume={resume} log_dir={log_dir}",
        flush=True,
    )

    # Deduplicate GPU ids while preserving order.  A duplicated id would make
    # the running/free bookkeeping (both keyed by GPU id) collide.
    seen = set()
    unique_gpus = []
    for gpu in gpus:
        if gpu not in seen:
            seen.add(gpu)
            unique_gpus.append(gpu)
    gpus = unique_gpus
    if args.dry_run:
        for index, job in enumerate(jobs):
            gpu = gpus[index % len(gpus)]
            print(
                f"  gpu {gpu:2d}  budget={job['budget']:<6} "
                f"{job['dataset']:<8} s{job['source']}->t{job['target']}",
                flush=True,
            )
        return 0

    base_command = [sys.executable, str(ENTRYPOINT)]
    overrides = []
    for name, value in (
        ("batch-size", args.batch_size),
        ("workers", args.workers),
        ("seed", args.seed),
        ("data-root", args.data_root),
        ("source-checkpoint-root", args.source_checkpoint_root),
        ("output-root", args.output_root),
    ):
        if value is not None:
            overrides.extend([f"--{name}", str(value)])

    pending = list(jobs)
    free_gpus = list(gpus)
    running = {}
    results = []
    started_at = utc_now()

    while pending or running:
        while pending and free_gpus:
            job = pending.pop(0)
            gpu = free_gpus.pop(0)
            task_name = f"{job['dataset']}_s{job['source']}_t{job['target']}"
            log_path = (
                log_dir
                / f"{log_prefix}budget_{job['budget']}_{task_name}_gpu{gpu}.log"
            )
            command = (
                base_command
                + ["--gpu-id", str(gpu)]
                + ["--budget", str(job["budget"])]
                + ["--dataset", job["dataset"]]
                + ["--source", str(job["source"]), "--target", str(job["target"])]
                + overrides
            )
            print(
                f"[start] gpu {gpu} budget={job['budget']} {job['dataset']} "
                f"s{job['source']}->t{job['target']} log={log_path.name}",
                flush=True,
            )
            with open(log_path, "w", encoding="utf-8") as log_file:
                proc = subprocess.Popen(
                    command,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            running[gpu] = (proc, job, log_path, time.monotonic())

        finished = False
        for gpu, (proc, job, log_path, started) in list(running.items()):
            return_code = proc.poll()
            if return_code is not None:
                elapsed = time.monotonic() - started
                status = "ok" if return_code == 0 else f"FAIL(rc={return_code})"
                print(
                    f"[done] gpu {gpu} budget={job['budget']} "
                    f"{job['dataset']} s{job['source']}->t{job['target']} "
                    f"{status} {elapsed:.0f}s log={log_path.name}",
                    flush=True,
                )
                results.append({**job, "gpu": gpu, "return_code": return_code})
                del running[gpu]
                free_gpus.append(gpu)
                finished = True
        if not finished and running:
            time.sleep(POLL_SECONDS)

    summary = {
        "schema_version": 1,
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "jobs": total_jobs,
        "ok": sum(1 for result in results if result["return_code"] == 0),
        "failed": sum(1 for result in results if result["return_code"] != 0),
        "results": results,
    }
    summary_path = log_dir / summary_filename
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"launcher done: ok={summary['ok']} failed={summary['failed']} "
        f"summary={summary_path}",
        flush=True,
    )
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
