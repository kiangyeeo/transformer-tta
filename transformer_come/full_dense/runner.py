"""Run one full-dense online COME stream followed by independent FO."""

from __future__ import annotations

from pathlib import Path

from transformer_come.dense_runner import run_dense_transfer

from .data import build_target_loaders
from .model import load_full_dense_model


def run_transfer(
    config: dict,
    project_root: Path,
    *,
    show_progress: bool = True,
    model_loader=load_full_dense_model,
) -> dict:
    """Run exactly one protocol-aligned full-dense COME condition."""

    return run_dense_transfer(
        config,
        project_root,
        variant="full_dense",
        model_loader=model_loader,
        build_target_loaders=build_target_loaders,
        show_progress=show_progress,
    )


__all__ = ["run_transfer"]
