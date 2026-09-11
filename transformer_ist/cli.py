"""CLI for IST transfer execution, validation, and explicit matrices."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .config import (
    DENSE_VARIANTS,
    SUPPORTED_VARIANTS,
    budget_tag,
    load_config,
    parse_budgets,
    resolve_transfer_config,
    select_transfers,
)
from .matrix import validate_devices


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def _parser():
    parser = argparse.ArgumentParser(
        description="Protocol-aligned non-LBI IST for source-trained DeiT-S"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    transfer = subparsers.add_parser("transfer")
    transfer.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    transfer.add_argument("--variant", choices=SUPPORTED_VARIANTS, required=True)
    transfer.add_argument(
        "--dataset", choices=["office31", "visda-c"], required=True
    )
    transfer.add_argument("--source", required=True)
    transfer.add_argument("--target", required=True)
    transfer.add_argument("--budget", type=float)
    transfer.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    transfer.add_argument("--output-dir", type=Path)
    transfer.add_argument("--debug-max-outer-batches", type=int)
    transfer.add_argument("--dry-run", action="store_true")
    transfer.add_argument("--no-progress", action="store_true")

    matrix = subparsers.add_parser("matrix")
    matrix.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    matrix.add_argument(
        "--datasets", choices=["all", "office31", "visda-c"], default="all"
    )
    matrix.add_argument(
        "--variants",
        default="all",
        help="all or comma-separated non-LBI variants",
    )
    matrix.add_argument("--budgets", default="all")
    matrix.add_argument("--devices", default="0")
    matrix.add_argument("--output-root", type=Path)
    matrix.add_argument("--dry-run", action="store_true")
    return parser


def _parse_variants(value: str):
    if value.strip().lower() == "all":
        return SUPPORTED_VARIANTS
    variants = tuple(item.strip() for item in value.split(",") if item.strip())
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("variants must be a non-empty unique list")
    unsupported = sorted(set(variants) - set(SUPPORTED_VARIANTS))
    if unsupported:
        if "group_lbi" in unsupported:
            raise ValueError(
                "group_lbi is intentionally not implemented in this revision"
            )
        raise ValueError(f"unsupported variants: {unsupported}")
    return variants


def _default_output(raw, variant, dataset, source, target, budget):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    budget_part = "" if budget is None else "_" + budget_tag(budget)
    return (
        Path(raw["output"]["root"])
        / f"single_{variant}{budget_part}_seed2026_{stamp}"
        / dataset
        / f"{source}-{target}"
    )


def _resolve(
    raw,
    args,
    variant,
    dataset,
    source,
    target,
    budget,
    output_dir,
    device,
    debug=None,
):
    return resolve_transfer_config(
        raw,
        project_root=PROJECT_ROOT,
        dataset=dataset,
        source=source,
        target=target,
        variant=variant,
        budget=budget,
        device=device,
        output_dir=output_dir,
        debug_max_outer_batches=debug,
    )


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    raw = load_config(args.config)
    if args.command == "transfer":
        output_dir = args.output_dir or _default_output(
            raw,
            args.variant,
            args.dataset,
            args.source,
            args.target,
            args.budget,
        )
        resolved = _resolve(
            raw,
            args,
            args.variant,
            args.dataset,
            args.source,
            args.target,
            args.budget,
            output_dir,
            args.device,
            args.debug_max_outer_batches,
        )
        if args.dry_run:
            print(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True))
            return 0
        from .runner import run_transfer

        run_transfer(
            resolved, PROJECT_ROOT, show_progress=not args.no_progress
        )
        return 0

    variants = _parse_variants(args.variants)
    budgets = parse_budgets(args.budgets)
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    validate_devices(devices)
    output_root = (
        args.output_root or Path(raw["output"]["root"])
    ).resolve()
    if args.dry_run:
        tasks = []
        for variant in variants:
            variant_budgets = (None,) if variant in DENSE_VARIANTS else budgets
            for budget in variant_budgets:
                for dataset, source, target in select_transfers(args.datasets):
                    resolved = _resolve(
                        raw,
                        args,
                        variant,
                        dataset,
                        source,
                        target,
                        budget,
                        output_root
                        / "DRY_RUN"
                        / variant
                        / dataset
                        / f"{source}-{target}",
                        "cpu" if devices == ["cpu"] else "cuda",
                    )
                    tasks.append(
                        {
                            "variant": variant,
                            "dataset": dataset,
                            "transfer": resolved["transfer"],
                            "budget": budget,
                            "requested_group_count": (
                                None
                                if resolved["selection"] is None
                                else resolved["selection"][
                                    "requested_group_count"
                                ]
                            ),
                            "experiment_key": resolved["experiment_key"],
                        }
                    )
        print(
            json.dumps(
                {
                    "status": "validated",
                    "condition_count": len(tasks),
                    "random_child_execution_count": 3
                    * sum(row["variant"] == "group_random" for row in tasks),
                    "tasks": tasks,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    from .matrix import run_matrix

    run_matrix(
        project_root=PROJECT_ROOT,
        config_path=args.config.resolve(),
        output_root=output_root,
        datasets=args.datasets,
        variants=variants,
        budgets=budgets,
        devices=devices,
    )
    return 0
