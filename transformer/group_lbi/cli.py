"""CLI for Group-LBI transfer, validation, tuning overrides, and GPU matrix."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .config import (
    FORMAL_BUDGETS,
    budget_tag,
    lbi_cli_overrides,
    load_config,
    parse_budgets,
    resolve_transfer_config,
    select_transfers,
)
from .matrix import validate_devices


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def _add_lbi_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lbi-alpha", type=float)
    parser.add_argument("--lbi-kappa", type=float)
    parser.add_argument("--lbi-nu", type=float)
    parser.add_argument("--lbi-omega", type=float)
    parser.add_argument("--lbi-prox-lambda", type=float)
    parser.add_argument("--lbi-tau-g", type=float)
    parser.add_argument("--lbi-stage1-max-steps", type=int)
    parser.add_argument("--lbi-stage2-lr", type=float)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Protocol-aligned DeiT-S structural Group Split-LBI SHOT-OTTA"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    transfer = subparsers.add_parser("transfer", help="run or resume one transfer/budget")
    transfer.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    transfer.add_argument("--dataset", choices=["office31", "visda-c"])
    transfer.add_argument("--source")
    transfer.add_argument("--target")
    transfer.add_argument("--budget", type=float, choices=FORMAL_BUDGETS)
    transfer.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    transfer.add_argument("--output-dir", type=Path)
    transfer.add_argument("--resume-run-dir", type=Path)
    transfer.add_argument("--no-stream-checkpoint", action="store_true")
    transfer.add_argument("--no-progress", action="store_true")
    transfer.add_argument("--dry-run", action="store_true")
    _add_lbi_arguments(transfer)

    matrix = subparsers.add_parser("matrix", help="schedule conditions over GPUs")
    matrix.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    matrix.add_argument("--datasets", choices=["all", "office31", "visda-c"], default="all")
    matrix.add_argument("--budgets", default="all")
    matrix.add_argument("--devices", default="0")
    matrix.add_argument("--output-root", type=Path)
    matrix.add_argument("--resume-run-root", type=Path)
    matrix.add_argument("--no-stream-checkpoint", action="store_true")
    matrix.add_argument("--dry-run", action="store_true")
    _add_lbi_arguments(matrix)
    return parser


def _default_transfer_output(raw, dataset, source, target, budget) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return (
        Path(raw["output"]["root"])
        / f"single_group_lbi_seed2026_{stamp}"
        / "results"
        / budget_tag(budget)
        / dataset
        / f"{source}-{target}"
    )


def _load_resume_config(run_dir: Path) -> dict:
    path = run_dir.resolve() / "effective_config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Resume directory is missing effective_config.yaml: {run_dir}")
    with open(path, "r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)
    if not isinstance(config, dict) or config.get("variant") != "group_lbi":
        raise ValueError("Resume effective config is not a Group-LBI condition")
    config["output_dir"] = str(run_dir.resolve())
    return config


def _dry_run_matrix(raw, selection, budgets, devices, output_root, overrides, checkpoint) -> None:
    validate_devices(devices)
    tasks = []
    conditions = [(*transfer, budget) for budget in budgets for transfer in select_transfers(selection)]
    for index, (dataset, source, target, budget) in enumerate(conditions):
        assigned = devices[index % len(devices)]
        output_dir = output_root / "DRY_RUN" / "results" / budget_tag(budget) / dataset / f"{source}-{target}"
        resolved = resolve_transfer_config(
            raw,
            project_root=PROJECT_ROOT,
            dataset=dataset,
            source=source,
            target=target,
            budget=budget,
            device="cpu" if assigned == "cpu" else "cuda",
            output_dir=output_dir,
            lbi_overrides=overrides,
            stream_checkpoint=checkpoint,
        )
        tasks.append(
            {
                "dataset": dataset,
                "transfer": resolved["transfer"],
                "requested_budget": budget,
                "requested_group_count": resolved["selection"]["requested_group_count"],
                "device": assigned,
                "lbi": resolved["lbi"],
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
                "formal_seed": 2026,
                "condition_count": len(tasks),
                "stream_checkpoint_enabled": checkpoint,
                "tasks": tasks,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "transfer":
        if args.resume_run_dir is not None:
            conflicting = [args.dataset, args.source, args.target, args.budget, args.output_dir]
            if any(value is not None for value in conflicting) or lbi_cli_overrides(args):
                raise ValueError(
                    "--resume-run-dir cannot be combined with transfer identity or LBI overrides"
                )
            config = _load_resume_config(args.resume_run_dir)
            if args.no_stream_checkpoint:
                raise ValueError("Resume requires stream checkpointing")
            if args.dry_run:
                print(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
                return 0
            from .runner import run_transfer

            run_transfer(config, PROJECT_ROOT, show_progress=not args.no_progress, resume=True)
            return 0

        missing = [
            name
            for name in ("dataset", "source", "target", "budget")
            if getattr(args, name) is None
        ]
        if missing:
            raise ValueError(f"New transfer run is missing arguments: {missing}")
        raw = load_config(args.config)
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
            lbi_overrides=lbi_cli_overrides(args),
            stream_checkpoint=not args.no_stream_checkpoint,
        )
        if args.dry_run:
            print(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True))
            return 0
        from .runner import run_transfer

        run_transfer(resolved, PROJECT_ROOT, show_progress=not args.no_progress)
        return 0

    raw = load_config(args.config)
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    budgets = parse_budgets(args.budgets)
    output_root = (args.output_root or Path(raw["output"]["root"])).resolve()
    overrides = lbi_cli_overrides(args)
    checkpoint = not args.no_stream_checkpoint
    if args.resume_run_root is not None and (overrides or args.no_stream_checkpoint):
        raise ValueError(
            "Matrix resume reuses stored condition configs and cannot accept "
            "LBI overrides or --no-stream-checkpoint"
        )
    if args.dry_run:
        if args.resume_run_root is not None:
            raise ValueError("--dry-run cannot be combined with --resume-run-root")
        _dry_run_matrix(raw, args.datasets, budgets, devices, output_root, overrides, checkpoint)
        return 0
    from .matrix import run_matrix

    run_matrix(
        project_root=PROJECT_ROOT,
        config_path=args.config.resolve(),
        output_root=output_root,
        selection=args.datasets,
        budgets=budgets,
        devices=devices,
        lbi_overrides=overrides,
        stream_checkpoint=checkpoint,
        resume_run_root=args.resume_run_root,
    )
    return 0
