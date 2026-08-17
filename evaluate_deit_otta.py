#!/usr/bin/env python3
"""Evaluate a DeiT-S source checkpoint as a no-adaptation OTTA stream."""

import argparse
import os
import os.path as osp

from shot_otta.ttda.config import apply_overrides, load_yaml, resolve_config


PROJECT_ROOT = osp.dirname(osp.abspath(__file__))
DEFAULT_CONFIG = osp.join(
    PROJECT_ROOT, "configs", "deit_otta_source_only.yaml"
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
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
    config = apply_overrides(load_yaml(args.config), args)
    effective = resolve_config(
        config,
        PROJECT_ROOT,
        provided_key=args.experiment_key,
        provided_sha256=args.experiment_config_sha256,
        expected_task="otta",
    )
    if args.dry_run:
        import sys
        import yaml

        yaml.safe_dump(effective, sys.stdout, sort_keys=False, allow_unicode=True)
        return 0
    os.environ["CUDA_VISIBLE_DEVICES"] = effective["device"]["gpu_id"]
    from shot_otta.otta.source_only import run_source_only_otta_experiment

    run_source_only_otta_experiment(effective, PROJECT_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
