"""CLI for single-transfer, dry-run, and GPU-matrix COME full-dense runs."""

from transformer_come.cli_common import PROJECT_ROOT, main as _main

from .spec import SPEC


def main(argv=None) -> int:
    return _main(SPEC, argv)


__all__ = ["PROJECT_ROOT", "SPEC", "main"]
