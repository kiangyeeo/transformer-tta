#!/usr/bin/env python3
"""No-training checks for the frozen OTTA-FC-LBI formal entry point."""

import copy
import math
import os.path as osp
import sys
import tempfile

PROJECT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
WORKSPACE_ROOT = osp.dirname(PROJECT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from experiment_identity import build_experiment_identity  # noqa: E402
from protocol_constants import (  # noqa: E402
    EFFICIENCY_PROTOCOL_REVISION,
    FORMAL_SEED,
    PROTOCOL_REVISION,
    SOURCE_CHECKPOINT_REVISION,
)
from shot_otta.config import load_yaml, resolve_effective_config  # noqa: E402
from shot_otta.efficiency import aggregate_random_efficiency  # noqa: E402
from tools.plan_experiments import build_plan  # noqa: E402
from tools.run_experiments_multi_gpu import (  # noqa: E402
    command_for_experiment,
    execute,
)


CONFIG_PATH = osp.join(
    PROJECT_DIR, "configs", "otta_fc_lbi_protocol_20260817_v1.yaml"
)
MATRIX_PATH = osp.join(
    PROJECT_DIR, "experiments", "shot_otta_fc_lbi_formal_20260817_v1.yaml"
)


def main():
    raw = load_yaml(CONFIG_PATH)
    matrix = load_yaml(MATRIX_PATH)
    assert raw["seed"] == FORMAL_SEED
    assert raw["loss"] == {
        "components": ["ent", "div", "pseudo"],
        "cls_par": 0.3,
        "ent_par": 1.0,
        "threshold": 0.0,
    }
    assert raw["lbi_protocol"] == {
        "support_threshold": 1.0e-4,
        "stage1_max_steps": 3000,
        "stage2_steps": 1,
        "delta_nonzero_tolerance": 1.0e-12,
        "budget_tolerance": 1.0e-4,
    }

    office = copy.deepcopy(raw)
    office["data"].update({"dataset": "office", "source": 0, "target": 1})
    office = resolve_effective_config(office, WORKSPACE_ROOT)
    assert office["data"]["batch_size"] == 64
    assert office["data"]["workers"] == 4
    assert office["optimization"]["lr"] == 0.01

    visda = copy.deepcopy(raw)
    visda["data"].update({"dataset": "VISDA-C", "source": 0, "target": 1})
    visda = resolve_effective_config(visda, WORKSPACE_ROOT)
    assert visda["data"]["batch_size"] == 256
    assert visda["data"]["workers"] == 4
    assert visda["optimization"]["lr"] == 0.001
    assert "/source/uda/" in visda["source_checkpoint"]["resolved_dir"]
    assert "/uda/uda/" not in visda["source_checkpoint"]["resolved_dir"]

    identity = build_experiment_identity(office)
    assert identity["scientific_config"]["source_checkpoint_revision"] == (
        SOURCE_CHECKPOINT_REVISION
    )
    changed_source = copy.deepcopy(office)
    changed_source["source_checkpoint_revision"] = "different_source_v2"
    assert build_experiment_identity(changed_source)[
        "experiment_config_sha256"
    ] != identity["experiment_config_sha256"]
    changed_efficiency = copy.deepcopy(office)
    changed_efficiency.setdefault("runtime", {})[
        "efficiency_protocol_revision"
    ] = "different_efficiency_v2"
    assert build_experiment_identity(changed_efficiency) == identity

    plan = build_plan(matrix, raw, CONFIG_PATH)
    assert plan["experiment_count"] == 105
    assert plan["formal_matrix_summary"]["dataset_counts"] == {
        "office": 90,
        "VISDA-C": 15,
    }
    assert plan["formal_matrix_summary"]["variant_counts"] == {
        "source_only": 7,
        "full_dense": 7,
        "module_dense": 7,
        "module_random": 21,
        "module_magnitude": 21,
        "module_saliency": 21,
        "module_lbi": 21,
    }
    assert [math.floor(r * 524544) for r in (0.0005, 0.001, 0.002)] == [
        262,
        524,
        1049,
    ]

    baseline = next(
        entry for entry in plan["experiments"] if entry["variant"] == "full_dense"
    )
    office_lbi = next(
        entry
        for entry in plan["experiments"]
        if entry["dataset"] == "office" and entry["variant"] == "module_lbi"
    )
    visda_lbi = next(
        entry
        for entry in plan["experiments"]
        if entry["dataset"] == "VISDA-C" and entry["variant"] == "module_lbi"
    )
    baseline_command = command_for_experiment(
        baseline, "0", enable_stream_checkpoint=True
    )
    assert "--enable-stream-checkpoint" not in baseline_command
    for entry in (office_lbi, visda_lbi):
        command = command_for_experiment(
            entry, "0", enable_stream_checkpoint=True, resume_run_dir="/tmp/lbi"
        )
        assert "--enable-stream-checkpoint" in command
        assert "--no-save-model" in command

    random_efficiency = aggregate_random_efficiency(
        [
            {"online_batch_runtime_mean_sec": 1.0, "online_compute_runtime_sec": 1.0, "fo_eval_runtime_sec": 1.0},
            {"online_batch_runtime_mean_sec": 2.0, "online_compute_runtime_sec": 2.0, "fo_eval_runtime_sec": 2.0},
            {"online_batch_runtime_mean_sec": 3.0, "online_compute_runtime_sec": 3.0, "fo_eval_runtime_sec": 3.0},
        ]
    )
    assert random_efficiency["fo_eval_runtime_sec"] == 2.0
    assert random_efficiency["random_total_fo_eval_runtime_sec"] == 6.0
    assert random_efficiency["random_total_online_compute_runtime_sec"] == 6.0
    assert random_efficiency["fo_eval_runtime_mask_std_sec"] == math.sqrt(2 / 3)

    try:
        with tempfile.TemporaryDirectory(prefix="formal_unresolved_lbi_") as root:
            office_lbi_for_launcher = dict(
                office_lbi,
                expected_output_root=osp.join(root, "runs"),
            )
            execute(
                {"experiments": [office_lbi_for_launcher]},
                osp.join(root, "runs"),
                osp.join(root, "logs"),
                ["0"],
                WORKSPACE_ROOT,
                process_executor=lambda *_args: 0,
            )
    except RuntimeError as error:
        assert "unresolved tuned tuple" in str(error)
    else:
        raise AssertionError("unresolved formal LBI was not blocked")

    assert plan["protocol_revision"] == PROTOCOL_REVISION
    assert plan["efficiency_protocol_revision"] == EFFICIENCY_PROTOCOL_REVISION
    print("protocol alignment smoke test passed")


if __name__ == "__main__":
    main()
