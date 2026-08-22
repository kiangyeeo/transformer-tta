"""Command-line entry points for single-transfer and matrix execution."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from .config import load_config, resolve_transfer_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Protocol-aligned DeiT-S source-only OTTA baseline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    transfer = subparsers.add_parser("transfer", help="run one formal transfer")
    transfer.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    transfer.add_argument("--dataset", required=True, choices=["office31", "visda-c"])
    transfer.add_argument("--source", required=True)
    transfer.add_argument("--target", required=True)
    transfer.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    transfer.add_argument("--output-dir", type=Path)
    transfer.add_argument("--no-progress", action="store_true")
    transfer.add_argument(
        "--dry-run",
        action="store_true",
        help="validate assets/config and print the resolved config only",
    )

    matrix = subparsers.add_parser(
        "matrix", help="schedule Office/VisDA transfers over dedicated GPUs"
    )
    matrix.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    matrix.add_argument(
        "--datasets", choices=["all", "office31", "visda-c"], default="all"
    )
    matrix.add_argument(
        "--devices",
        default="0",
        help="comma-separated physical GPU ids, or cpu",
    )
    matrix.add_argument("--output-root", type=Path)
    return parser


def _default_transfer_output(raw: dict, dataset: str, source: str, target: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return (
        Path(raw["output"]["root"])
        / f"single_seed2026_{stamp}"
        / "results"
        / dataset
        / f"{source}-{target}"
    )


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    raw = load_config(args.config)
    if args.command == "transfer":
        output_dir = args.output_dir or _default_transfer_output(
            raw, args.dataset, args.source, args.target
        )
        resolved = resolve_transfer_config(
            raw,
            project_root=PROJECT_ROOT,
            dataset=args.dataset,
            source=args.source,
            target=args.target,
            device=args.device,
            output_dir=output_dir,
        )
        if args.dry_run:
            import yaml

            print(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True))
            return 0
        from .runner import run_transfer

        run_transfer(
            resolved, PROJECT_ROOT, show_progress=not args.no_progress
        )
        return 0


    output_root = args.output_root or Path(raw["output"]["root"])
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    from .matrix import run_matrix

    run_matrix(
        project_root=PROJECT_ROOT,
        config_path=args.config.resolve(),
        output_root=output_root.resolve(),
        selection=args.datasets,
        devices=devices,
    )
    return 0

