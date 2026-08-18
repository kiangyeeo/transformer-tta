#!/usr/bin/env python3
"""Evaluate DeiT-S OTTA adaptation: saliency structural-group baseline.

After every TTA-loss backward, scores the paired Q-K / V-O / FFN groups by
gradient L2 norm and updates the Top-K (``ceil(budget * 6912)``) groups with
the SHOT objective + AdamW.  The mask is rebuilt every online step.
"""

import argparse
import os
import os.path as osp
import sys

import yaml

from shot_otta.deit_source_only.config import apply_overrides, load_yaml
from shot_otta.otta.saliency import run_deit_otta_saliency_experiment
from shot_otta.otta.saliency_config import resolve_config


PROJECT_ROOT = osp.dirname(osp.abspath(__file__))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate DeiT-S under OTTA with saliency structural groups."
    )
    parser.add_argument(
        "--config",
        help="Path to the run configuration (default: group-saliency yaml).",
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
            PROJECT_ROOT, "configs", "deit_otta_group_saliency.yaml"
        )
    config = apply_overrides(load_yaml(args.config), args)
    if args.budget is not None:
        config["adaptation"]["budget"] = float(args.budget)
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
    run_deit_otta_saliency_experiment(effective, PROJECT_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
