#!/usr/bin/env python3
"""Regression checks for resolved LBI CLI config plumbing only."""

import copy
import json
import os.path as osp
import subprocess
import sys
import tempfile

import yaml


PROJECT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
WORKSPACE_ROOT = osp.dirname(PROJECT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from shot_otta.config import (  # noqa: E402
    apply_overrides,
    load_yaml,
    resolve_effective_config,
)


CONFIG_PATH = osp.join(
    PROJECT_DIR, "configs", "otta_fc_lbi_protocol_20260817_v1.yaml"
)


def _resolved_lbi():
    return {
        "alpha": 0.1,
        "kappa": 1.0,
        "nu": 0.5,
        "omega": 0.2,
        "stage1_max_steps": 3000,
        "budget_tolerance": 1.0e-4,
        "stage2_lr": 0.02,
        "stage2_steps": 1,
        "delta_nonzero_tolerance": 1.0e-12,
        "support_threshold": 1.0e-4,
    }


def _resolved_cli_args():
    from train import build_parser  # noqa: WPS433

    return build_parser().parse_args(
        [
            "--config",
            CONFIG_PATH,
            "--dataset",
            "office",
            "--source",
            "0",
            "--target",
            "1",
            "--seed",
            "2026",
            "--variant",
            "module_lbi",
            "--requested-budget",
            "0.0005",
            "--gpu-id",
            "0",
            "--workers",
            "4",
            "--no-save-model",
            "--lbi-alpha",
            "0.1",
            "--lbi-kappa",
            "1.0",
            "--lbi-nu",
            "0.5",
            "--lbi-omega",
            "0.2",
            "--lbi-stage1-max-steps",
            "3000",
            "--lbi-budget-tolerance",
            "1e-4",
            "--lbi-stage2-lr",
            "0.02",
            "--lbi-stage2-steps",
            "1",
            "--lbi-delta-nonzero-tolerance",
            "1e-12",
            "--lbi-support-threshold",
            "1e-4",
        ]
    )


def main():
    raw = load_yaml(CONFIG_PATH)
    assert raw["lbi"] is None

    args = _resolved_cli_args()
    overridden = apply_overrides(raw, args)
    assert overridden["lbi"] == _resolved_lbi()
    effective = resolve_effective_config(overridden, WORKSPACE_ROOT)
    assert effective["lbi"] == _resolved_lbi()

    matrix = {
        "method": "shot",
        "task": "otta",
        "datasets": {"office": {"transfers": [[0, 1]]}},
        "seeds": [2026],
        "variants": {
            "module_lbi": {
                "budgets": [0.0005],
                "lbi": _resolved_lbi(),
            }
        },
        "common_training": {
            "workers": 4,
            "gpu_id": "0",
            "save_model": False,
        },
        "output_root": "260817_iclr2027-refined/runs",
    }
    with tempfile.TemporaryDirectory(prefix="lbi_plumbing_plan_") as root:
        matrix_path = osp.join(root, "matrix.yaml")
        plan_dir = osp.join(root, "plan")
        with open(matrix_path, "w", encoding="utf-8") as file_obj:
            yaml.safe_dump(matrix, file_obj, sort_keys=False)
        planner = subprocess.run(
            [
                sys.executable,
                osp.join(PROJECT_DIR, "tools", "plan_experiments.py"),
                matrix_path,
                CONFIG_PATH,
                "--output-dir",
                plan_dir,
            ],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        assert planner.returncode == 0, planner.stderr
        with open(
            osp.join(plan_dir, "plan.json"),
            "r",
            encoding="utf-8",
        ) as file_obj:
            plan = json.load(file_obj)
    assert plan["experiment_count"] == 1
    entry = plan["experiments"][0]
    assert entry["lbi_resolved"] is True

    command = list(entry["command_args"])
    completed = subprocess.run(
        command + ["--dry-run"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    resolved_from_command = yaml.safe_load(completed.stdout)
    assert resolved_from_command["experiment_key"] == entry["experiment_key"]
    assert (
        resolved_from_command["experiment_config_sha256"]
        == entry["experiment_config_sha256"]
    )

    unresolved = copy.deepcopy(raw)
    unresolved["variant"] = "module_lbi"
    unresolved["requested_budget"] = 0.0005
    try:
        resolve_effective_config(unresolved, WORKSPACE_ROOT)
    except ValueError as error:
        assert "resolved tuned tuple" in str(error)
    else:
        raise AssertionError(
            "formal module_lbi without a tuple did not fail closed"
        )

    print("LBI config plumbing regression test passed")


if __name__ == "__main__":
    main()
