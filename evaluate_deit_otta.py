#!/usr/bin/env python3
"""Evaluate a DeiT-S source checkpoint as a no-adaptation OTTA stream."""

import os.path as osp

from shot_otta.deit_source_only.entrypoint import main_for_task


PROJECT_ROOT = osp.dirname(osp.abspath(__file__))


if __name__ == "__main__":
    raise SystemExit(main_for_task("otta", PROJECT_ROOT))
