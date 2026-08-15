#!/usr/bin/env python3
"""Train a local-initialized DeiT-S source model for Office-31 or VisDA-C."""

import argparse
import os.path as osp
import sys

import yaml

from source_training.deit_config import (
    apply_source_overrides,
    load_source_config,
    resolve_source_config,
)


PROJECT_ROOT = osp.dirname(osp.abspath(__file__))
DEFAULT_CONFIG = osp.join(
    PROJECT_ROOT, "configs", "source_deit_office31.yaml"
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Full-model DeiT-S source-domain fine-tuning"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--source-domain", default=None)
    parser.add_argument("--source-list", default=None)
    parser.add_argument("--pretrained-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume an exact .last.pth state; otherwise auto-resume if present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and validate the protocol without loading data or timm.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    raw_config = load_source_config(args.config)
    overridden = apply_source_overrides(
        raw_config,
        source_domain=args.source_domain,
        source_list=args.source_list,
        pretrained_path=args.pretrained_path,
        output_path=args.output_path,
        device=args.device,
    )
    effective = resolve_source_config(overridden, PROJECT_ROOT)
    if args.dry_run:
        yaml.safe_dump(
            effective,
            sys.stdout,
            sort_keys=False,
            allow_unicode=True,
        )
        return 0

    from source_training.deit_trainer import run_source_training

    result = run_source_training(
        effective,
        PROJECT_ROOT,
        resume_path=args.resume,
    )
    print(yaml.safe_dump(result, sort_keys=False, allow_unicode=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

