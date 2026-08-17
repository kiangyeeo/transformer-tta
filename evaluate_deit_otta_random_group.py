#!/usr/bin/env python3
"""Evaluate DeiT-S OTTA adaptation: random structural-group baseline.

Selects ``ceil(budget * 6912)`` paired Q-K / V-O / FFN groups once before the
stream, then adapts only those groups with the SHOT objective + AdamW.
Multiple random masks are run inside one process and aggregated into
mean ± std.
"""

import argparse
import os
import os.path as osp
import sys

import yaml

from shot_otta.deit_source_only.config import apply_overrides, load_yaml
from shot_otta.otta.random_group import (
    run_deit_otta_random_group_experiment,
)
from shot_otta.otta.random_group_config import resolve_config


PROJECT_ROOT = osp.dirname(osp.abspath(__file__))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate DeiT-S under OTTA with random structural groups."
    )
    parser.add_argument(
        "--config",
        help="Path to the run configuration (default: group-random yaml).",
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
    parser.add_argument("--budget", type=float)
    parser.add_argument(
        "--mask-seeds",
        help="Comma-separated random mask seeds (default: config value).",
    )
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
    if args.config is None:
        args.config = osp.join(
            PROJECT_ROOT, "configs", "deit_otta_group_random.yaml"
        )
    config = apply_overrides(load_yaml(args.config), args)
    if args.budget is not None:
        config["adaptation"]["budget"] = float(args.budget)
    if args.mask_seeds is not None:
        seeds = [
            int(value.strip())
            for value in args.mask_seeds.split(",")
            if value.strip()
        ]
        config["adaptation"]["mask_seeds"] = seeds
        config["adaptation"]["num_random_masks"] = len(seeds)
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
    run_deit_otta_random_group_experiment(effective, PROJECT_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
