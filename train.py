#!/usr/bin/env python3
"""Unified entry point for the first-stage SHOT-OTTA baseline."""

import argparse
import os
import os.path as osp

from shot_otta.config import (
    apply_overrides,
    load_yaml,
    resolve_effective_config,
)


PROJECT_DIR = osp.dirname(osp.abspath(__file__))
WORKSPACE_ROOT = osp.dirname(PROJECT_DIR)
DEFAULT_CONFIG = osp.join(PROJECT_DIR, "configs", "shot_otta.yaml")


def build_parser():
    parser = argparse.ArgumentParser(description="iclr2027 SHOT-OTTA baseline")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--source", type=int, default=None)
    parser.add_argument("--target", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gpu-id", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--source-checkpoint-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--variant",
        choices=[
            "source_only",
            "full_dense",
            "module_dense",
            "module_random",
            "module_magnitude",
            "module_saliency",
            "module_lbi",
        ],
        default=None,
    )
    parser.add_argument("--requested-budget", type=float, default=None)
    parser.add_argument("--selection-seed", type=int, default=None)
    parser.add_argument("--num-random-masks", type=int, default=None)
    parser.add_argument("--lbi-alpha", type=float, default=None)
    parser.add_argument("--lbi-kappa", type=float, default=None)
    parser.add_argument("--lbi-nu", type=float, default=None)
    parser.add_argument("--lbi-omega", type=float, default=None)
    parser.add_argument(
        "--lbi-stage1-max-steps", type=int, default=None
    )
    parser.add_argument(
        "--lbi-budget-tolerance", type=float, default=None
    )
    parser.add_argument("--lbi-stage2-lr", type=float, default=None)
    parser.add_argument("--lbi-stage2-steps", type=int, default=None)
    parser.add_argument(
        "--lbi-delta-nonzero-tolerance", type=float, default=None
    )
    parser.add_argument("--experiment-key", default=None)
    parser.add_argument("--experiment-config-sha256", default=None)
    parser.add_argument(
        "--resume-run-dir",
        default=None,
        help="Resume a checkpointed VISDA-C module_lbi stream directory.",
    )
    parser.add_argument(
        "--enable-stream-checkpoint",
        action="store_true",
        help="Persist an atomic stream checkpoint after each online step.",
    )
    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument("--save-model", action="store_true")
    save_group.add_argument("--no-save-model", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and validate config without importing training code",
    )
    return parser


def main():
    args = build_parser().parse_args()
    raw_config = load_yaml(args.config)
    effective_config = apply_overrides(raw_config, args)
    effective_config = resolve_effective_config(
        effective_config, WORKSPACE_ROOT
    )

    if args.dry_run:
        import sys
        import yaml

        yaml.safe_dump(
            effective_config,
            sys.stdout,
            sort_keys=False,
            allow_unicode=True,
        )
        return 0

    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        effective_config["device"]["gpu_id"]
    )
    from shot_otta.trainer import run_experiment

    run_experiment(
        effective_config,
        WORKSPACE_ROOT,
        resume_run_dir=args.resume_run_dir,
        enable_stream_checkpoint=args.enable_stream_checkpoint,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
