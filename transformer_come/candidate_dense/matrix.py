"""One-process-per-GPU scheduler for the seven COME candidate-dense transfers."""

from __future__ import annotations

from pathlib import Path

from transformer_come.matrix_common import (
    new_run_root as _new_run_root,
    run_matrix as _run_matrix,
    validate_devices,
)

from .spec import SPEC


def new_run_root(output_root: Path) -> Path:
    return _new_run_root(output_root, SPEC.run_prefix)


def run_matrix(
    *,
    project_root: Path,
    config_path: Path,
    output_root: Path,
    selection: str,
    devices: list[str],
) -> Path:
    return _run_matrix(
        spec=SPEC,
        project_root=project_root,
        config_path=config_path,
        output_root=output_root,
        selection=selection,
        devices=devices,
    )


__all__ = ["new_run_root", "run_matrix", "validate_devices"]
