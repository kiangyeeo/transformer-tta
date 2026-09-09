#!/usr/bin/env python3
"""Build and statically validate Node 7's frozen VisDA-C filter plan."""

import argparse
import copy
import json
import os
import os.path as osp
import shlex
import sys

import yaml


PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiment_identity import build_experiment_identity
from protocol_constants import (
    CONV_IMPLEMENTATION_REVISION,
    CONV_PROTOCOL_REVISION,
    EFFICIENCY_PROTOCOL_REVISION,
    FORMAL_SEED,
    SOURCE_CHECKPOINT_REVISION,
)
from shot_otta.artifacts import experiment_output_root
from shot_otta.config import DATASETS, load_yaml, resolve_effective_config


BASELINE_PROTOCOL = "OTTA_CONV_BASELINE_FORMAL_20260827_v1"
NODE7_CONDITIONS = {
    "conv_filter_random": "filter_connection",
    "conv_filter_magnitude": "filter_connection",
    "conv_filter_saliency": "filter_connection",
}
NODE6_CONDITIONS = {
    "conv_module_dense": None,
    "conv_out_random": "out_channel",
    "conv_out_magnitude": "out_channel",
    "conv_out_saliency": "out_channel",
}
RHOS = (0.0005, 0.001, 0.002)
GROUP_COUNT = 6553600
EXPECTED_K = {0.0005: 3276, 0.001: 6553, 0.002: 13107}
RANDOM_VARIANTS = {"conv_filter_random"}


def fail(message):
    raise ValueError(message)


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        fail(f"matrix must be a mapping: {path}")
    return value


def assert_frozen_inputs(matrix, base_config):
    protocol_dir = osp.join(PROJECT_ROOT, "protocol", "shot-otta_conv")
    expected_files = (
        ("OTTA_CONV_BASELINE_FORMAL_20260827_v1.md", f"# {BASELINE_PROTOCOL}"),
        ("OTTA_CONV_LBI_PROTOCOL_20260826_v1.md", f"# {CONV_PROTOCOL_REVISION}"),
    )
    for filename, expected_heading in expected_files:
        path = osp.join(protocol_dir, filename)
        if not osp.isfile(path):
            fail(f"frozen protocol is missing: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            if not handle.read().startswith(expected_heading):
                fail(f"frozen protocol heading drifted: {path}")
    expected = {
        "formal_baseline_protocol_revision": BASELINE_PROTOCOL,
        "conv_method_protocol_revision": CONV_PROTOCOL_REVISION,
        "implementation_revision": CONV_IMPLEMENTATION_REVISION,
        "source_checkpoint_revision": SOURCE_CHECKPOINT_REVISION,
        "formal_seed": FORMAL_SEED,
    }
    for field, value in expected.items():
        if matrix.get(field) != value:
            fail(f"matrix {field} drifted: {matrix.get(field)!r}")
    if base_config.get("protocol_track") != "conv":
        fail("base config must use protocol_track=conv")
    if base_config.get("protocol_revision") != CONV_PROTOCOL_REVISION:
        fail("base config Conv protocol revision drifted")
    if base_config.get("implementation_revision") != CONV_IMPLEMENTATION_REVISION:
        fail("base config Conv implementation revision drifted")
    if base_config.get("formal_protocol") is not True:
        fail("base config must retain formal_protocol=true")


def assert_matrix(matrix):
    if DATASETS.get("VISDA-C", {}).get("domains") != ["train", "validation"]:
        fail("VisDA-C domain order is not [train, validation]")
    if matrix.get("dataset") != "VISDA-C":
        fail("Node 7 must use dataset=VISDA-C")
    if matrix.get("transfer") != {"source": 0, "target": 1, "identifier": "TV"}:
        fail("Node 7 must be exactly VisDA-C Train->Validation: source=0, target=1")
    if tuple(float(value) for value in matrix.get("formal_rhos", [])) != RHOS:
        fail("formal rho grid must be exactly 0.0005/0.001/0.002")
    if matrix.get("common_training") != {
        "batch_size": 256, "workers": 4, "gpu_id": "0", "save_model": False,
        "workers_per_gpu": 1,
    }:
        fail("VisDA-C common training configuration drifted")
    serialized = json.dumps(matrix, sort_keys=True)
    if "0.005" in serialized or "_lbi" in serialized or "conv_out_" in serialized or "conv_module_dense" in serialized:
        fail("matrix contains prohibited budget or Node 6/LBI condition")
    conditions = matrix.get("conditions")
    observed = {item.get("variant"): item.get("group_mode") for item in conditions or [] if isinstance(item, dict)}
    if observed != NODE7_CONDITIONS or len(conditions or []) != len(NODE7_CONDITIONS):
        fail("matrix condition families do not match Node 7's required filter families")
    if matrix.get("random") != {
        "selection_seed": 2026,
        "num_random_masks": 3,
        "child_mask_seed_rule": "mask_seed = formal_seed * 100 + mask_index",
        "child_mask_seeds": [202600, 202601, 202602],
    }:
        fail("Random must use seed 2026 and the three frozen child masks")
    budget_block = matrix.get("expected_group_budgets", {}).get("filter_connection", {})
    if budget_block.get("total_group_count") != GROUP_COUNT:
        fail("filter-connection total group count drifted")
    if budget_block.get("K_G") != [EXPECTED_K[rho] for rho in RHOS]:
        fail("filter-connection K_G sequence drifted")


def build_entry(base_config, matrix, variant, group_mode, rho):
    config = copy.deepcopy(base_config)
    output_root = osp.abspath(osp.join(PROJECT_ROOT, matrix["output_root"]))
    config.update({
        "seed": FORMAL_SEED,
        "variant": variant,
        "requested_budget": 1.0 if group_mode is None else rho,
        "group_mode": group_mode,
        "selection_seed": 2026 if variant in {"conv_filter_random", "conv_out_random"} else None,
        "num_random_masks": 3 if variant in {"conv_filter_random", "conv_out_random"} else None,
    })
    config["data"].update({"dataset": "VISDA-C", "source": 0, "target": 1})
    config["output"].update({"root": output_root, "save_model": False})
    effective = resolve_effective_config(config, PROJECT_ROOT)
    identity = build_experiment_identity(effective)
    command = [
        "python", osp.join(PROJECT_ROOT, "train.py"), "--config", osp.abspath(BASE_CONFIG_PATH),
        "--dataset", "VISDA-C", "--source", "0", "--target", "1", "--seed", "2026",
        "--gpu-id", "0", "--variant", variant, "--output-root", output_root,
        "--no-save-model",
    ]
    if group_mode is not None:
        command.extend(["--group-mode", group_mode, "--requested-budget", str(rho)])
    if variant in {"conv_filter_random", "conv_out_random"}:
        command.extend(["--selection-seed", "2026", "--num-random-masks", "3"])
    command.extend([
        "--experiment-key", identity["experiment_key"],
        "--experiment-config-sha256", identity["experiment_config_sha256"],
    ])
    return {
        "formal_baseline_protocol_revision": BASELINE_PROTOCOL,
        "implementation_revision": CONV_IMPLEMENTATION_REVISION,
        "protocol_revision": CONV_PROTOCOL_REVISION,
        "efficiency_protocol_revision": EFFICIENCY_PROTOCOL_REVISION,
        "source_checkpoint_revision": SOURCE_CHECKPOINT_REVISION,
        "experiment_key": identity["experiment_key"],
        "experiment_config_sha256": identity["experiment_config_sha256"],
        "scientific_config": identity["scientific_config"],
        "method": "shot", "task": "otta", "dataset": "VISDA-C", "source": 0, "target": 1,
        "seed": FORMAL_SEED, "variant": variant, "group_mode": group_mode,
        "requested_budget": 1.0 if group_mode is None else rho,
        "selection_seed": 2026 if variant in {"conv_filter_random", "conv_out_random"} else None,
        "selection_seed_mode": "same_as_run_seed" if variant in {"conv_filter_random", "conv_out_random"} else None,
        "num_random_masks": 3 if variant in {"conv_filter_random", "conv_out_random"} else None,
        "child_mask_seeds": [202600, 202601, 202602] if variant in {"conv_filter_random", "conv_out_random"} else None,
        "total_group_count": GROUP_COUNT if group_mode == "filter_connection" else 9216 if group_mode == "out_channel" else None,
        "max_group_count": EXPECTED_K[rho] if group_mode == "filter_connection" else None,
        "effective_overrides": {
            "dataset": "VISDA-C", "source": 0, "target": 1, "seed": FORMAL_SEED,
            "variant": variant, "group_mode": group_mode,
            "requested_budget": 1.0 if group_mode is None else rho,
            "selection_seed": 2026 if variant in {"conv_filter_random", "conv_out_random"} else None,
            "num_random_masks": 3 if variant in {"conv_filter_random", "conv_out_random"} else None,
            "gpu_id": "0", "save_model": False, "workers_per_gpu": 1, "output_root": output_root,
        },
        "command_args": command,
        "expected_output_root": experiment_output_root(effective),
    }


def build_plan(matrix, base_config):
    entries = [
        build_entry(base_config, matrix, variant, mode, rho)
        for variant, mode in NODE7_CONDITIONS.items() for rho in RHOS
    ]
    return {
        "plan_schema_version": 1, "node": "node7_filter",
        "formal_baseline_protocol_revision": BASELINE_PROTOCOL,
        "protocol_revision": CONV_PROTOCOL_REVISION,
        "implementation_revision": CONV_IMPLEMENTATION_REVISION,
        "efficiency_protocol_revision": EFFICIENCY_PROTOCOL_REVISION,
        "source_checkpoint_revision": SOURCE_CHECKPOINT_REVISION,
        "matrix_path": osp.abspath(MATRIX_PATH), "base_config_path": osp.abspath(BASE_CONFIG_PATH),
        "experiment_count": len(entries), "experiments": entries,
        "formal_matrix_summary": {
            "dataset_counts": {"VISDA-C": len(entries)},
            "variant_counts": {variant: 3 for variant in NODE7_CONDITIONS},
            "transfer": {"source": 0, "target": 1, "identifier": "TV"},
            "scientific_condition_count": len(entries),
        },
    }


def expected_node6_entries(base_config, matrix):
    return [
        build_entry(base_config, matrix, variant, mode, rho)
        for variant, mode in NODE6_CONDITIONS.items()
        for rho in ((None,) if mode is None else RHOS)
    ]


def validate_plan(plan, base_config, matrix, runs_root):
    entries = plan.get("experiments", [])
    if plan.get("experiment_count") != 9 or len(entries) != 9:
        fail("plan must contain exactly 9 scientific conditions")
    for field in ("experiment_key", "experiment_config_sha256", "expected_output_root"):
        if len({entry[field] for entry in entries}) != 9:
            fail(f"plan {field} values are not unique")
    expected = {(variant, rho) for variant in NODE7_CONDITIONS for rho in RHOS}
    observed = {(entry["variant"], entry["requested_budget"]) for entry in entries}
    if observed != expected:
        fail("plan conditions are not exactly the required nine filter entries")
    for entry in entries:
        if (entry["dataset"], entry["source"], entry["target"], entry["seed"]) != ("VISDA-C", 0, 1, 2026):
            fail("plan VisDA-C transfer or seed drifted")
        if entry["variant"] not in NODE7_CONDITIONS or entry["group_mode"] != "filter_connection":
            fail("plan contains a prohibited non-filter condition")
        rho = entry["requested_budget"]
        if rho not in RHOS or entry["total_group_count"] != GROUP_COUNT or entry["max_group_count"] != EXPECTED_K[rho]:
            fail("formal filter budget/K_G drifted")
        if entry["variant"] == "conv_filter_random" and (
            entry["selection_seed"] != 2026 or entry["num_random_masks"] != 3
            or entry["child_mask_seeds"] != [202600, 202601, 202602]
        ):
            fail("Random three-mask semantics drifted")
        try:
            common = osp.commonpath([osp.abspath(entry["expected_output_root"]), runs_root])
        except ValueError as error:
            raise ValueError("invalid planned output root") from error
        if common != runs_root:
            fail("planned output root escapes formal runs root")
    node6 = expected_node6_entries(base_config, matrix)
    for field in ("experiment_key", "experiment_config_sha256", "expected_output_root"):
        overlap = {entry[field] for entry in entries} & {entry[field] for entry in node6}
        if overlap:
            fail(f"Node 7 overlaps Node 6 planned {field}: {sorted(overlap)}")
    node6_plan = osp.join(osp.dirname(MATRIX_PATH), "..", "node6_out", "plan.json")
    node6_plan = osp.abspath(node6_plan)
    if osp.isfile(node6_plan):
        with open(node6_plan, "r", encoding="utf-8") as handle:
            actual_node6 = json.load(handle).get("experiments", [])
        for field in ("experiment_key", "experiment_config_sha256", "expected_output_root"):
            overlap = {entry[field] for entry in entries} & {entry.get(field) for entry in actual_node6}
            if overlap:
                fail(f"Node 7 overlaps existing Node 6 {field}: {sorted(overlap)}")


def write_outputs(plan, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(osp.join(output_dir, "plan.json"), "w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with open(osp.join(output_dir, "plan.jsonl"), "w", encoding="utf-8") as handle:
        for entry in plan["experiments"]:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with open(osp.join(output_dir, "commands.sh"), "w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        for entry in plan["experiments"]:
            command = ["conda", "run", "--no-capture-output", "-n", "SHOT_TTA", *entry["command_args"]]
            handle.write(shlex.join(command) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    global MATRIX_PATH, BASE_CONFIG_PATH
    MATRIX_PATH, BASE_CONFIG_PATH = args.matrix, args.base_config
    matrix, base_config = read_yaml(MATRIX_PATH), load_yaml(BASE_CONFIG_PATH)
    assert_frozen_inputs(matrix, base_config)
    assert_matrix(matrix)
    plan = build_plan(matrix, base_config)
    validate_plan(plan, base_config, matrix, osp.abspath(osp.join(PROJECT_ROOT, matrix["output_root"])))
    if not args.verify_only:
        write_outputs(plan, args.output_dir)
    print(json.dumps({"valid": True, "experiment_count": 9, "transfer": "TV", "node6_overlap": False}))


if __name__ == "__main__":
    main()
