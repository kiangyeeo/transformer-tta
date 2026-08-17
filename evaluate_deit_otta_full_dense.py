#!/usr/bin/env python3
"""Evaluate DeiT-S OTTA adaptation: SHOT objective + AdamW updates.

Supports ``full_dense`` (all parameters) and ``candidate_dense`` (weight
tensors of the last-3 candidate blocks only).
"""

import argparse
import os
import os.path as osp
import sys

import yaml

from shot_otta.deit_source_only.config import apply_overrides, load_yaml
from shot_otta.otta.full_dense import run_deit_otta_adaptation_experiment
from shot_otta.otta.full_dense_config import resolve_config


PROJECT_ROOT = osp.dirname(osp.abspath(__file__))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate DeiT-S under OTTA with dense adaptation."
    )
    parser.add_argument("--variant", choices=["full_dense", "candidate_dense"])
    parser.add_argument(
        "--config",
        help="Path to the run configuration (default: per-variant yaml).",
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


def main():
    args = build_parser().parse_args()
    variant = args.variant or "full_dense"
    if args.config is None:
        args.config = osp.join(
            PROJECT_ROOT, "configs", f"deit_otta_{variant}.yaml"
        )
    config = apply_overrides(load_yaml(args.config), args)
    if args.variant is not None:
        config["variant"] = args.variant
    effective = resolve_config(
        config,
        PROJECT_ROOT,
        provided_key=args.experiment_key,
        provided_sha256=args.experiment_config_sha256,
    )
    if args.dry_run:
        yaml.safe_dump(effective, sys.stdout, sort_keys=False, allow_unicode=True)
        return 0
    os.environ["CUDA_VISIBLE_DEVICES"] = effective["device"]["gpu_id"]
    run_deit_otta_adaptation_experiment(effective, PROJECT_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
