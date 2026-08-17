#!/usr/bin/env python3
"""Dispatch DeiT OTTA adaptation runs (full/candidate-dense) across GPUs.

Simple greedy scheduler: at most one subprocess per GPU, refilling free GPUs
as jobs finish.  The default job set is the seven OTTA directions (six
Office-31 ordered transfers plus VisDA-C train->validation) for every
requested variant; with both variants that is 14 jobs and all 8 GPUs are used
across two waves.  Each job writes its merged stdout/stderr to
``<output.root>/launcher_logs/<variant>_<dataset>_s<source>_t<target>_gpu<g>.log``
and a ``launcher_summary.json`` is written at the end.
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

ENTRYPOINT = PROJECT_ROOT / "evaluate_deit_otta_full_dense.py"
SUPPORTED_VARIANTS = ("full_dense", "candidate_dense")
OFFICE_DIRECTIONS = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
VISDA_DIRECTION = ((0, 1),)
POLL_SECONDS = 5.0


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def config_output_root(variant):
    config_path = PROJECT_ROOT / "configs" / f"deit_otta_{variant}.yaml"
    with open(config_path, "r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)
    root = config.get("output", {}).get("root")
    if not root:
        raise RuntimeError(f"config has no output.root: {config_path}")
    return Path(root).resolve()


def build_jobs(variants, datasets):
    jobs = []
    for variant in variants:
        if "office31" in datasets:
            for source, target in OFFICE_DIRECTIONS:
                jobs.append(
                    {
                        "variant": variant,
                        "dataset": "office31",
                        "source": int(source),
                        "target": int(target),
                    }
                )
        if "visda-c" in datasets:
            for source, target in VISDA_DIRECTION:
                jobs.append(
                    {
                        "variant": variant,
                        "dataset": "visda-c",
                        "source": int(source),
                        "target": int(target),
                    }
                )
    return jobs


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run DeiT OTTA adaptation baselines across GPUs."
    )
    parser.add_argument(
        "--variant",
        default="full_dense,candidate_dense",
        help="Comma-separated variants (full_dense,candidate_dense).",
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
        "--dry-run",
        action="store_true",
        help="Print the job->GPU assignment without launching anything.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    variants = [value.strip() for value in args.variant.split(",") if value.strip()]
    for variant in variants:
        if variant not in SUPPORTED_VARIANTS:
            raise SystemExit(f"unsupported variant: {variant}")
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise SystemExit("no GPUs selected")
    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    jobs = build_jobs(variants, datasets)
    total_jobs = len(jobs)
    if total_jobs == 0:
        raise SystemExit("no jobs to run")

    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else config_output_root(variants[0])
    )
    log_dir = output_root / "launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"jobs={total_jobs} gpus={gpus} variants={variants} "
        f"datasets={datasets} log_dir={log_dir}",
        flush=True,
    )

    assignment = []
    for index, job in enumerate(jobs):
        gpu = gpus[index % len(gpus)]
        assignment.append((gpu, job))
    if args.dry_run:
        for gpu, job in assignment:
            print(
                f"  gpu {gpu:2d}  {job['variant']:<15} {job['dataset']:<8} "
                f"s{job['source']}->t{job['target']}",
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

    pending = list(assignment)
    free_gpus = list(gpus)
    running = {}
    results = []
    started_at = utc_now()

    while pending or running:
        while pending and free_gpus:
            gpu, job = pending.pop(0)
            free_gpus.remove(gpu)
            task_name = f"{job['dataset']}_s{job['source']}_t{job['target']}"
            log_path = log_dir / f"{job['variant']}_{task_name}_gpu{gpu}.log"
            command = (
                base_command
                + ["--variant", job["variant"], "--gpu-id", str(gpu)]
                + ["--dataset", job["dataset"]]
                + ["--source", str(job["source"]), "--target", str(job["target"])]
                + overrides
            )
            print(
                f"[start] gpu {gpu} {job['variant']} {job['dataset']} "
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
                    f"[done] gpu {gpu} {job['variant']} {job['dataset']} "
                    f"s{job['source']}->t{job['target']} {status} "
                    f"{elapsed:.0f}s log={log_path.name}",
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
    summary_path = log_dir / "launcher_summary.json"
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
