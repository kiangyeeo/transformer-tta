#!/usr/bin/env python3
"""Unified entry point for refined SHOT-, IST-, NCTTA-, and COME-OTTA."""

import argparse
import os
import os.path as osp

from shot_otta.config import (
    apply_overrides,
    load_yaml,
)


PROJECT_DIR = osp.dirname(osp.abspath(__file__))
WORKSPACE_ROOT = osp.dirname(PROJECT_DIR)
DEFAULT_CONFIG = osp.join(
    PROJECT_DIR, "configs", "otta_fc_lbi_protocol_20260817_v1.yaml"
)


def build_parser():
    parser = argparse.ArgumentParser(description="iclr2027 refined OTTA baseline")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--source", type=int, default=None)
    parser.add_argument("--target", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gpu-id", default=None)
    parser.add_argument("--workers-per-gpu", type=int, default=None)
    parser.add_argument(
        "--runtime-comparable",
        action="store_true",
        help="Mark this run as the formal one-GPU efficiency protocol.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--debug-max-outer-batches",
        type=int,
        default=None,
        help=(
            "Debug-only IST/NCTTA/COME smoke limit for processed non-singleton "
            "outer batches; rejected by formal runs."
        ),
    )
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
            "conv_module_dense",
            "conv_out_random",
            "conv_out_magnitude",
            "conv_out_saliency",
            "conv_out_lbi",
            "conv_filter_random",
            "conv_filter_magnitude",
            "conv_filter_saliency",
            "conv_filter_lbi",
            "ist_full_dense",
            "ist_fc_module_dense",
            "ist_fc_random",
            "ist_fc_magnitude",
            "ist_fc_saliency",
            "ist_fc_lbi",
            "ist_conv_module_dense",
            "ist_conv_out_random",
            "ist_conv_out_magnitude",
            "ist_conv_out_saliency",
            "ist_conv_out_lbi",
            "nctta_full_dense",
            "nctta_fc_module_dense",
            "nctta_conv_module_dense",
            "nctta_native_norm",
            "nctta_fc_random",
            "nctta_fc_magnitude",
            "nctta_fc_saliency",
            "nctta_fc_lbi",
            "nctta_conv_out_random",
            "nctta_conv_out_magnitude",
            "nctta_conv_out_saliency",
            "nctta_conv_out_lbi",
            "come_full_dense",
            "come_fc_module_dense",
            "come_conv_module_dense",
            "come_fc_random",
            "come_fc_magnitude",
            "come_fc_saliency",
            "come_fc_lbi",
            "come_conv_out_random",
            "come_conv_out_magnitude",
            "come_conv_out_saliency",
            "come_conv_out_lbi",
        ],
        default=None,
    )
    parser.add_argument(
        "--group-mode",
        choices=["out_channel", "filter_connection"],
        default=None,
    )
    parser.add_argument("--requested-budget", type=float, default=None)
    parser.add_argument("--selection-seed", type=int, default=None)
    parser.add_argument("--num-random-masks", type=int, default=None)
    parser.add_argument("--lbi-alpha", type=float, default=None)
    parser.add_argument("--lbi-kappa", type=float, default=None)
    parser.add_argument("--lbi-nu", type=float, default=None)
    parser.add_argument("--lbi-omega", type=float, default=None)
    parser.add_argument("--lbi-stage1-max-steps", type=int, default=None)
    parser.add_argument("--lbi-budget-tolerance", type=float, default=None)
    parser.add_argument("--lbi-stage2-lr", type=float, default=None)
    parser.add_argument("--lbi-stage2-steps", type=int, default=None)
    parser.add_argument("--lbi-delta-nonzero-tolerance", type=float, default=None)
    parser.add_argument("--lbi-support-threshold", type=float, default=None)
    parser.add_argument("--experiment-key", default=None)
    parser.add_argument("--experiment-config-sha256", default=None)
    parser.add_argument(
        "--resume-run-dir",
        default=None,
        help="Resume a supported checkpointed online stream directory.",
    )
    parser.add_argument(
        "--enable-stream-checkpoint",
        action="store_true",
        help="Persist an atomic checkpoint after each committed online step.",
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
    if raw_config.get("method") == "NCTTA":
        from nctta_otta.config import apply_overrides as apply_method_overrides
    elif raw_config.get("method") == "COME":
        from come_otta.config import apply_overrides as apply_method_overrides
    else:
        apply_method_overrides = apply_overrides
    effective_config = apply_method_overrides(raw_config, args)
    if effective_config.get("method") == "IST":
        from ist_otta.config import resolve_effective_config
    elif effective_config.get("method") == "NCTTA":
        from nctta_otta.config import resolve_effective_config
    elif effective_config.get("method") == "COME":
        from come_otta.config import resolve_effective_config
    else:
        from shot_otta.config import resolve_effective_config
    effective_config = resolve_effective_config(effective_config, WORKSPACE_ROOT)

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

    os.environ["CUDA_VISIBLE_DEVICES"] = str(effective_config["device"]["gpu_id"])
    if effective_config["method"] == "IST":
        from ist_otta.trainer import run_experiment
    elif effective_config["method"] == "NCTTA":
        from nctta_otta.trainer import run_experiment
    elif effective_config["method"] == "COME":
        from come_otta.trainer import run_experiment
    else:
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
