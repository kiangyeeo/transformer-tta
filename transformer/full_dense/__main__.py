#!/usr/bin/env python3
"""Allow ``python -m transformer.full_dense`` execution."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
