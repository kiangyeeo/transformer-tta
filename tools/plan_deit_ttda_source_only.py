#!/usr/bin/env python3
"""Build the fixed seven-task DeiT TTDA source-only execution plan."""

import os.path as osp
import sys


PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.deit_source_only_plan import build_plan as _build_plan
from tools.deit_source_only_plan import main_for_task


def build_plan(matrix, base_config, config_path, matrix_path):
    return _build_plan(
        matrix,
        base_config,
        config_path,
        matrix_path,
        task="ttda",
    )


if __name__ == "__main__":
    raise SystemExit(main_for_task("ttda"))
