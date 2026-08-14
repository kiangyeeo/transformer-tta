#!/usr/bin/env python3
"""Prepare the Office random-baseline subset from the formal plan.

This is an execution-only orchestration helper.  It delegates all scientific
configuration and command construction to ``plan_experiments`` and only
selects the requested formal-plan entries.
"""

import argparse
import copy
import json
import os
import os.path as osp
import sys


PROJECT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
WORKSPACE_ROOT = osp.dirname(PROJECT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from shot_otta.config import load_yaml  # noqa: E402
from tools.plan_experiments import (  # noqa: E402
    build_plan,
    write_plan_outputs,
)


TRANSFERS = {(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)}
BUDGETS = {0.0005, 0.001, 0.002}


def build_random_office_plan(matrix_path, base_config_path):
    matrix = load_yaml(matrix_path)
    base_config = load_yaml(base_config_path)
    formal_plan = build_plan(matrix, base_config, base_config_path)
    selected = []
    for experiment in formal_plan["experiments"]:
        if (
            experiment["dataset"] == "office"
            and experiment["variant"] == "module_random"
            and experiment["seed"] == 2020
            and (experiment["source"], experiment["target"]) in TRANSFERS
            and float(experiment["requested_budget"]) in BUDGETS
        ):
            selected.append(copy.deepcopy(experiment))
    selected.sort(
        key=lambda item: (
            item["source"],
            item["target"],
            float(item["requested_budget"]),
        )
    )
    if len(selected) != 18:
        raise RuntimeError(
            f"Expected 18 formal random entries, found {len(selected)}"
        )
    if {item["num_random_masks"] for item in selected} != {3}:
        raise RuntimeError("Formal plan does not specify exactly 3 masks")
    if {item["seed"] for item in selected} != {2020}:
        raise RuntimeError("Formal plan seed filter is not exactly 2020")
    if {item["variant"] for item in selected} != {"module_random"}:
        raise RuntimeError("Formal plan contains a non-random variant")
    plan = copy.deepcopy(formal_plan)
    plan.update(
        {
            "experiment_count": len(selected),
            "experiments": selected,
            "selection_from_formal_plan": {
                "matrix_path": osp.abspath(matrix_path),
                "base_config_path": osp.abspath(base_config_path),
                "dataset": "office",
                "variant": "module_random",
                "seed": 2020,
                "transfers": sorted(TRANSFERS),
                "budgets": sorted(BUDGETS),
                "num_random_masks": 3,
            },
        }
    )
    return formal_plan, plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix")
    parser.add_argument("base_config")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    formal_plan, selected_plan = build_random_office_plan(
        args.matrix,
        args.base_config,
    )
    output_dir = osp.abspath(args.output_dir)
    write_plan_outputs(selected_plan, output_dir)
    with open(
        osp.join(output_dir, "formal_plan.json"),
        "w",
        encoding="utf-8",
    ) as file_obj:
        json.dump(formal_plan, file_obj, indent=2, ensure_ascii=False)
    with open(
        osp.join(output_dir, "selection_manifest.json"),
        "w",
        encoding="utf-8",
    ) as file_obj:
        json.dump(
            selected_plan["selection_from_formal_plan"],
            file_obj,
            indent=2,
            ensure_ascii=False,
        )
    print(
        json.dumps(
            {
                "formal_experiment_count": formal_plan["experiment_count"],
                "selected_experiment_count": selected_plan["experiment_count"],
                "output_dir": output_dir,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
