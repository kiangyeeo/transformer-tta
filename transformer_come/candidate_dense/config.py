"""Frozen configuration and transfer identity for COME candidate-dense OTTA."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from transformer.candidate_dense.config import (
    CANDIDATE_BLOCKS,
    CANDIDATE_SCALAR_COUNT,
    CANDIDATE_SUFFIXES,
    CANDIDATE_TENSOR_COUNT,
    candidate_parameter_names,
)
from transformer.source_only.config import FORMAL_SEED, TRANSFERS

from transformer_come.common_config import (
    CANDIDATE_ADAPTATION,
    finalize_identity,
    load_config,
    resolve_common_transfer_config,
    select_transfers,
    validate_common_fields,
)
from transformer_come.identity import IMPLEMENTATION_REVISIONS, PROTOCOL_REVISIONS


VARIANT = "candidate_dense"
PROTOCOL_REVISION = PROTOCOL_REVISIONS[VARIANT]
IMPLEMENTATION_REVISION = IMPLEMENTATION_REVISIONS[VARIANT]


def _validate_frozen_fields(config: dict[str, Any]) -> None:
    validate_common_fields(config, variant=VARIANT, adaptation=CANDIDATE_ADAPTATION)


def resolve_transfer_config(
    raw_config: dict[str, Any],
    *,
    project_root: Path,
    dataset: str,
    source: str,
    target: str,
    device: str,
    output_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Resolve one formal transfer and fail closed on every frozen field."""

    _validate_frozen_fields(raw_config)
    resolved = resolve_common_transfer_config(
        raw_config,
        variant=VARIANT,
        project_root=project_root,
        dataset=dataset,
        source=source,
        target=target,
        device=device,
        output_dir=output_dir,
    )
    return finalize_identity(resolved, variant=VARIANT)


__all__ = [
    "CANDIDATE_BLOCKS",
    "CANDIDATE_SCALAR_COUNT",
    "CANDIDATE_SUFFIXES",
    "CANDIDATE_TENSOR_COUNT",
    "FORMAL_SEED",
    "IMPLEMENTATION_REVISION",
    "PROTOCOL_REVISION",
    "TRANSFERS",
    "VARIANT",
    "_validate_frozen_fields",
    "candidate_parameter_names",
    "load_config",
    "resolve_transfer_config",
    "select_transfers",
]
