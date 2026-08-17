"""Shared CLI plumbing for the DeiT OTTA and TTDA source-only controls."""

import argparse
import os
import os.path as osp
import sys

from shot_otta.deit_source_only.config import (
    apply_overrides,
    load_yaml,
    resolve_config,
)


def build_parser(task, project_root):
    if task not in {"otta", "ttda"}:
        raise ValueError(f"Unsupported source-only task: {task}")
    parser = argparse.ArgumentParser(
        description=f"Evaluate DeiT-S under {task.upper()} with no adaptation."
    )
    parser.add_argument(
        "--config",
        default=osp.join(
            project_root, "configs", f"deit_{task}_source_only.yaml"
        ),
    )
    parser.add_argument("--dataset", choices=["office31", "visda-c"])
    parser.add_argument("--source", type=int)
    parser.add_argument("--target", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpu-id")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--data-root")
    parser.add_argument("--source-checkpoint-root")
    parser.add_argument("--output-root")
    parser.add_argument("--run-name")
    parser.add_argument("--experiment-key")
    parser.add_argument("--experiment-config-sha256")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve assets and identity without loading the model or images.",
    )
    return parser


def main_for_task(task, project_root):
    args = build_parser(task, project_root).parse_args()
    config = apply_overrides(load_yaml(args.config), args)
    effective = resolve_config(
        config,
        project_root,
        provided_key=args.experiment_key,
        provided_sha256=args.experiment_config_sha256,
        expected_task=task,
    )
    if args.dry_run:
        import yaml

        yaml.safe_dump(effective, sys.stdout, sort_keys=False, allow_unicode=True)
        return 0
    os.environ["CUDA_VISIBLE_DEVICES"] = effective["device"]["gpu_id"]
    if task == "otta":
        from shot_otta.otta.source_only import run_source_only_otta_experiment

        run_source_only_otta_experiment(effective, project_root)
    else:
        from shot_otta.ttda.source_only import run_source_only_experiment

        run_source_only_experiment(effective, project_root)
    return 0
