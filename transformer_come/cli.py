"""CLI for COME transfer execution, validation, and explicit matrices."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .config import (
    CHILDREN_PER_CONDITION,
    FORMAL_SEED,
    IMPLEMENTATION_REVISIONS,
    PROTOCOL_REVISIONS,
    RUN_PREFIXES,
    SPARSE_VARIANTS,
    SUPPORTED_VARIANTS,
    budget_tag,
    load_config,
    parse_budgets,
    parse_variants,
    resolve_transfer_config,
    select_transfers,
    variant_output_root,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Protocol-aligned non-LBI COME-OTTA on the SHOT-Transformer substrate"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    transfer = subparsers.add_parser("transfer", help="run one formal condition")
    transfer.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    transfer.add_argument("--variant", choices=SUPPORTED_VARIANTS, required=True)
    transfer.add_argument("--dataset", required=True, choices=["office31", "visda-c"])
    transfer.add_argument("--source", required=True)
    transfer.add_argument("--target", required=True)
    transfer.add_argument(
        "--budget",
        "--rho",
        dest="budget",
        type=float,
        help=(
            "structural-group ratio rho, required by the sparse variants; "
            "custom values are allowed"
        ),
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
        "matrix", help="schedule the selected conditions over GPUs"
    )
    matrix.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    matrix.add_argument(
        "--datasets", choices=["all", "office31", "visda-c"], default="all"
    )
    matrix.add_argument(
        "--variants",
        default="all",
        help="all or comma-separated non-LBI variants",
    )
    matrix.add_argument(
        "--budgets",
        "--rhos",
        "--rho",
        dest="budgets",
        default="all",
        help="all (default formal rhos), or custom comma-separated rho values",
    )
    matrix.add_argument(
        "--devices", default="0", help="comma-separated physical GPU ids, or cpu"
    )
    matrix.add_argument(
        "--output-root",
        type=Path,
        help=(
            "directory that will contain the timestamped run root; only valid "
            "with a single --variants entry"
        ),
    )
    matrix.add_argument(
        "--dry-run",
        action="store_true",
        help="validate every selected condition without creating a run directory",
    )

    finalize = subparsers.add_parser(
        "finalize", help="recover aggregates from already-completed Random children"
    )
    finalize.add_argument("--run-root", type=Path, required=True)
    finalize.add_argument(
        "--dry-run",
        action="store_true",
        help="validate recoverability without modifying existing artifacts",
    )
    return parser


def _single_output_dir(raw: dict, variant: str, dataset, source, target, budget) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = (
        Path(variant_output_root(raw, variant))
        / f"single_{RUN_PREFIXES[variant]}_{stamp}"
        / "results"
    )
    if budget is not None:
        path = path / budget_tag(budget)
    return path / dataset / f"{source}-{target}"


def _resolve(raw, *, variant, dataset, source, target, budget, device, output_dir):
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
    )


def _variant_budgets(variant: str, budgets):
    return budgets if variant in SPARSE_VARIANTS else None


def _dry_run_plan(raw, config_path, variant, datasets, budgets, devices, output_root):
    from .matrix import condition_tasks, result_dir_for, validate_devices

    validate_devices(devices)
    transfers = list(select_transfers(datasets))
    conditions = condition_tasks(transfers, _variant_budgets(variant, budgets))
    tasks = []
    for index, (dataset, source, target, budget) in enumerate(conditions):
        assigned = devices[index % len(devices)]
        resolved = _resolve(
            raw,
            variant=variant,
            dataset=dataset,
            source=source,
            target=target,
            budget=budget,
            device="cpu" if assigned == "cpu" else "cuda",
            output_dir=result_dir_for(
                output_root / "DRY_RUN", dataset, source, target, budget
            ),
        )
        task = {
            "dataset": dataset,
            "transfer": resolved["transfer"],
            "device": assigned,
            "checkpoint": resolved["checkpoint_path"],
            "checkpoint_sha256": resolved["checkpoint_sha256"],
            "target_list": resolved["target_list"],
            "experiment_key": resolved["experiment_key"],
        }
        if budget is not None:
            task["requested_budget"] = budget
            task["requested_group_count"] = resolved["selection"][
                "requested_group_count"
            ]
        if CHILDREN_PER_CONDITION[variant] > 1:
            task["mask_seeds"] = resolved["selection"]["mask_seeds"]
        tasks.append(task)
    plan = {
        "status": "validated",
        "method": "come",
        "variant": variant,
        "protocol_revision": PROTOCOL_REVISIONS[variant],
        "implementation_revision": IMPLEMENTATION_REVISIONS[variant],
        "config": str(Path(config_path).resolve()),
        "selection": datasets,
        "formal_seed": FORMAL_SEED,
        "condition_count": len(tasks),
        "output_root": str(output_root),
        "tasks": tasks,
    }
    if CHILDREN_PER_CONDITION[variant] > 1:
        plan["child_run_count"] = len(tasks) * CHILDREN_PER_CONDITION[variant]
    return plan


def main(argv=None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "finalize":
        from .finalize import recover_completed_run

        report = recover_completed_run(args.run_root, dry_run=args.dry_run)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    raw = load_config(args.config)

    if args.command == "transfer":
        output_dir = args.output_dir or _single_output_dir(
            raw, args.variant, args.dataset, args.source, args.target, args.budget
        )
        resolved = _resolve(
            raw,
            variant=args.variant,
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

    variants = parse_variants(args.variants)
    budgets = parse_budgets(args.budgets)
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if args.output_root is not None and len(variants) != 1:
        raise ValueError(
            "--output-root names one variant's artifact root; select exactly "
            "one --variants entry or omit it"
        )
    output_roots = {
        variant: (
            args.output_root
            if args.output_root is not None
            else Path(variant_output_root(raw, variant))
        ).resolve()
        for variant in variants
    }

    if args.dry_run:
        plans = [
            _dry_run_plan(
                raw,
                args.config,
                variant,
                args.datasets,
                budgets,
                devices,
                output_roots[variant],
            )
            for variant in variants
        ]
        print(
            json.dumps(
                {
                    "status": "validated",
                    "method": "come",
                    "formal_seed": FORMAL_SEED,
                    "config": str(Path(args.config).resolve()),
                    "selection": args.datasets,
                    "variant_count": len(plans),
                    "condition_count": sum(plan["condition_count"] for plan in plans),
                    "child_run_count": sum(
                        plan.get("child_run_count", plan["condition_count"])
                        for plan in plans
                    ),
                    "variants": plans,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    from .matrix import run_matrix

    for variant in variants:
        run_matrix(
            variant=variant,
            project_root=PROJECT_ROOT,
            config_path=Path(args.config).resolve(),
            output_root=output_roots[variant],
            selection=args.datasets,
            devices=devices,
            budgets=_variant_budgets(variant, budgets),
        )
    return 0


__all__ = ["PROJECT_ROOT", "main"]
