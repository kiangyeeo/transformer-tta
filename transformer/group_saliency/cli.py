"""CLI for one Saliency condition, validation, and GPU matrix execution."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .config import (
    budget_tag,
    load_config,
    parse_budgets,
    resolve_transfer_config,
    select_transfers,
)
from .matrix import validate_devices


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Protocol-aligned DeiT-S dynamic structural-group Saliency SHOT-OTTA"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    transfer = subparsers.add_parser(
        "transfer", help="run one transfer and saliency budget"
    )
    transfer.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    transfer.add_argument("--dataset", required=True, choices=["office31", "visda-c"])
    transfer.add_argument("--source", required=True)
    transfer.add_argument("--target", required=True)
    transfer.add_argument(
        "--budget", "--rho", dest="budget", required=True, type=float,
        help="structural-group ratio rho; custom values are allowed",
    )
    transfer.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    transfer.add_argument("--output-dir", type=Path)
    transfer.add_argument("--no-progress", action="store_true")
    transfer.add_argument(
        "--dry-run",
        action="store_true",
        help="validate local assets and print the resolved config only",
    )

    matrix = subparsers.add_parser(
        "matrix", help="schedule transfer/budget conditions over GPUs"
    )
    matrix.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    matrix.add_argument(
        "--datasets", choices=["all", "office31", "visda-c"], default="all"
    )
    matrix.add_argument(
        "--budgets", "--rhos", "--rho",
        dest="budgets",
        default="all",
        help="all (default formal rhos), or custom comma-separated rho values",
    )
    matrix.add_argument(
        "--devices", default="0", help="comma-separated physical GPU ids, or cpu"
    )
    matrix.add_argument("--output-root", type=Path)
    matrix.add_argument(
        "--dry-run",
        action="store_true",
        help="validate every selected condition without creating a run directory",
    )
    return parser


def _default_transfer_output(raw, dataset, source, target, budget) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return (
        Path(raw["output"]["root"])
        / f"single_group_saliency_seed2026_{stamp}"
        / "results"
        / budget_tag(budget)
        / dataset
        / f"{source}-{target}"
    )


def _dry_run_matrix(raw, config_path, selection, budgets, devices, output_root) -> None:
    validate_devices(devices)
    tasks = []
    conditions = [
        (*transfer, budget)
        for budget in budgets
        for transfer in select_transfers(selection)
    ]
    for index, (dataset, source, target, budget) in enumerate(conditions):
        assigned = devices[index % len(devices)]
        output_dir = (
            output_root
            / "DRY_RUN"
            / "results"
            / budget_tag(budget)
            / dataset
            / f"{source}-{target}"
        )
        resolved = resolve_transfer_config(
            raw,
            project_root=PROJECT_ROOT,
            dataset=dataset,
            source=source,
            target=target,
            budget=budget,
            device="cpu" if assigned == "cpu" else "cuda",
            output_dir=output_dir,
        )
        tasks.append(
            {
                "dataset": dataset,
                "transfer": resolved["transfer"],
                "requested_budget": budget,
                "requested_group_count": resolved["selection"][
                    "requested_group_count"
                ],
                "device": assigned,
                "checkpoint": resolved["checkpoint_path"],
                "checkpoint_sha256": resolved["checkpoint_sha256"],
                "target_list": resolved["target_list"],
                "experiment_key": resolved["experiment_key"],
            }
        )
    print(
        json.dumps(
            {
                "status": "validated",
                "config": str(config_path.resolve()),
                "selection": selection,
                "formal_seed": 2026,
                "condition_count": len(tasks),
                "tasks": tasks,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    raw = load_config(args.config)
    if args.command == "transfer":
        output_dir = args.output_dir or _default_transfer_output(
            raw, args.dataset, args.source, args.target, args.budget
        )
        resolved = resolve_transfer_config(
            raw,
            project_root=PROJECT_ROOT,
            dataset=args.dataset,
            source=args.source,
            target=args.target,
            budget=args.budget,
            device=args.device,
            output_dir=output_dir,
        )
        if args.dry_run:
            print(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True))
            return 0
        from .runner import run_transfer

        run_transfer(resolved, PROJECT_ROOT, show_progress=not args.no_progress)
        return 0

    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    budgets = parse_budgets(args.budgets)
    output_root = (args.output_root or Path(raw["output"]["root"])).resolve()
    if args.dry_run:
        _dry_run_matrix(raw, args.config, args.datasets, budgets, devices, output_root)
        return 0
    from .matrix import run_matrix

    run_matrix(
        project_root=PROJECT_ROOT,
        config_path=args.config.resolve(),
        output_root=output_root,
        selection=args.datasets,
        budgets=budgets,
        devices=devices,
    )
    return 0
