"""Shared CLI wiring for the COME variants.

Each variant keeps its own ``python -m transformer_come.<variant>`` entry
point with the same subcommands and flags as its SHOT counterpart; only the
module path and the run-directory prefix differ.
"""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .budget import budget_tag


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class VariantSpec:
    """Everything the generic CLI/matrix code needs about one variant."""

    variant: str
    module: str
    description: str
    run_prefix: str
    protocol_revision: str
    implementation_revision: str
    sparse: bool = False
    children_per_condition: int = 1
    with_finalize: bool = False

    @property
    def package_dir(self) -> Path:
        return Path(importlib.import_module(self.module).__file__).parent

    @property
    def default_config(self) -> Path:
        return self.package_dir / "config.yaml"

    def _submodule(self, name: str):
        return importlib.import_module(f"{self.module}.{name}")

    def load_config(self, path):
        return self._submodule("config").load_config(path)

    def resolve_transfer_config(self, *args, **kwargs):
        return self._submodule("config").resolve_transfer_config(*args, **kwargs)

    def select_transfers(self, selection: str):
        return self._submodule("config").select_transfers(selection)

    def parse_budgets(self, selection):
        return self._submodule("config").parse_budgets(selection)

    def aggregate_functions(self):
        module = self._submodule("aggregate")
        return module.aggregate_matrix, module.write_aggregate, module.print_aggregate

    def run_transfer(self, *args, **kwargs):
        return self._submodule("runner").run_transfer(*args, **kwargs)

    def recover_completed_run(self, *args, **kwargs):
        return self._submodule("finalize").recover_completed_run(*args, **kwargs)


def _parser(spec: VariantSpec) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=spec.description)
    subparsers = parser.add_subparsers(dest="command", required=True)
    default_config = spec.default_config

    transfer = subparsers.add_parser("transfer", help="run one formal condition")
    transfer.add_argument("--config", type=Path, default=default_config)
    transfer.add_argument("--dataset", required=True, choices=["office31", "visda-c"])
    transfer.add_argument("--source", required=True)
    transfer.add_argument("--target", required=True)
    if spec.sparse:
        transfer.add_argument(
            "--budget",
            "--rho",
            dest="budget",
            required=True,
            type=float,
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
        "matrix", help="schedule the selected conditions over GPUs"
    )
    matrix.add_argument("--config", type=Path, default=default_config)
    matrix.add_argument(
        "--datasets", choices=["all", "office31", "visda-c"], default="all"
    )
    if spec.sparse:
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
    matrix.add_argument("--output-root", type=Path)
    matrix.add_argument(
        "--dry-run",
        action="store_true",
        help="validate every selected condition without creating a run directory",
    )

    if spec.with_finalize:
        finalize = subparsers.add_parser(
            "finalize", help="recover aggregates from already-completed child masks"
        )
        finalize.add_argument("--run-root", type=Path, required=True)
        finalize.add_argument(
            "--dry-run",
            action="store_true",
            help="validate recoverability without modifying existing artifacts",
        )
    return parser


def _single_output_dir(spec: VariantSpec, raw: dict, dataset, source, target, budget) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = Path(raw["output"]["root"]) / f"single_{spec.run_prefix}_{stamp}" / "results"
    if budget is not None:
        path = path / budget_tag(budget)
    return path / dataset / f"{source}-{target}"


def _resolve(spec: VariantSpec, raw: dict, *, dataset, source, target, budget, device, output_dir):
    kwargs = dict(
        project_root=PROJECT_ROOT,
        dataset=dataset,
        source=source,
        target=target,
        device=device,
        output_dir=output_dir,
    )
    if budget is not None:
        kwargs["budget"] = budget
    return spec.resolve_transfer_config(raw, **kwargs)


def _dry_run_matrix(spec, raw, config_path, selection, budgets, devices, output_root) -> None:
    from .matrix_common import condition_tasks, result_dir_for, validate_devices

    validate_devices(devices)
    transfers = list(spec.select_transfers(selection))
    conditions = condition_tasks(transfers, budgets)
    tasks = []
    for index, (dataset, source, target, budget) in enumerate(conditions):
        assigned = devices[index % len(devices)]
        output_dir = result_dir_for(
            output_root / "DRY_RUN", dataset, source, target, budget
        )
        resolved = _resolve(
            spec,
            raw,
            dataset=dataset,
            source=source,
            target=target,
            budget=budget,
            device="cpu" if assigned == "cpu" else "cuda",
            output_dir=output_dir,
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
        if spec.children_per_condition > 1:
            task["mask_seeds"] = resolved["selection"]["mask_seeds"]
        tasks.append(task)
    payload = {
        "status": "validated",
        "method": "come",
        "variant": spec.variant,
        "protocol_revision": spec.protocol_revision,
        "implementation_revision": spec.implementation_revision,
        "config": str(Path(config_path).resolve()),
        "selection": selection,
        "formal_seed": 2026,
        "condition_count": len(tasks),
        "tasks": tasks,
    }
    if spec.children_per_condition > 1:
        payload["child_run_count"] = len(tasks) * spec.children_per_condition
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main(spec: VariantSpec, argv=None) -> int:
    args = _parser(spec).parse_args(argv)
    if args.command == "finalize":
        report = spec.recover_completed_run(args.run_root, dry_run=args.dry_run)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    raw = spec.load_config(args.config)
    if args.command == "transfer":
        budget = getattr(args, "budget", None)
        output_dir = args.output_dir or _single_output_dir(
            spec, raw, args.dataset, args.source, args.target, budget
        )
        resolved = _resolve(
            spec,
            raw,
            dataset=args.dataset,
            source=args.source,
            target=args.target,
            budget=budget,
            device=args.device,
            output_dir=output_dir,
        )
        if args.dry_run:
            print(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True))
            return 0
        spec.run_transfer(resolved, PROJECT_ROOT, show_progress=not args.no_progress)
        return 0

    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    budgets = spec.parse_budgets(args.budgets) if spec.sparse else None
    output_root = (args.output_root or Path(raw["output"]["root"])).resolve()
    if args.dry_run:
        _dry_run_matrix(
            spec, raw, args.config, args.datasets, budgets, devices, output_root
        )
        return 0
    from .matrix_common import run_matrix

    run_matrix(
        spec=spec,
        project_root=PROJECT_ROOT,
        config_path=Path(args.config).resolve(),
        output_root=output_root,
        selection=args.datasets,
        devices=devices,
        budgets=budgets,
    )
    return 0


__all__ = ["PROJECT_ROOT", "VariantSpec", "main"]
