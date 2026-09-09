#!/usr/bin/env python3
"""Build and statically validate Node 0's frozen Office Conv baseline plan."""

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
EXPECTED_CONDITIONS = {
    "conv_module_dense": None,
    "conv_out_random": "out_channel",
    "conv_out_magnitude": "out_channel",
    "conv_out_saliency": "out_channel",
    "conv_filter_random": "filter_connection",
    "conv_filter_magnitude": "filter_connection",
    "conv_filter_saliency": "filter_connection",
}
RHOS = (0.0005, 0.001, 0.002)
GROUP_COUNTS = {"out_channel": 9216, "filter_connection": 6553600}
EXPECTED_K = {
    "out_channel": {0.0005: 4, 0.001: 9, 0.002: 18},
    "filter_connection": {0.0005: 3276, 0.001: 6553, 0.002: 13107},
}
RANDOM_VARIANTS = {"conv_out_random", "conv_filter_random"}


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
    baseline_path = osp.join(
        protocol_dir, "OTTA_CONV_BASELINE_FORMAL_20260827_v1.md"
    )
    conv_path = osp.join(protocol_dir, "OTTA_CONV_LBI_PROTOCOL_20260826_v1.md")
    for path, expected_heading in (
        (baseline_path, f"# {BASELINE_PROTOCOL}"),
        (conv_path, f"# {CONV_PROTOCOL_REVISION}"),
    ):
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
    if DATASETS.get("office", {}).get("domains") != ["amazon", "dslr", "webcam"]:
        fail("Office domain order is not [amazon, dslr, webcam]")
    transfer = matrix.get("transfer")
    if transfer != {"source": 0, "target": 1, "identifier": "AD"}:
        fail("Node 0 must be exactly Office Amazon->DSLR: source=0, target=1")
    if tuple(float(value) for value in matrix.get("formal_rhos", [])) != RHOS:
        fail("formal rho grid must be exactly 0.0005/0.001/0.002")
    serialized = json.dumps(matrix, sort_keys=True)
    if "0.005" in serialized or "conv_" in serialized and "_lbi" in serialized:
        fail("matrix contains a prohibited .005 budget or Conv-LBI condition")
    conditions = matrix.get("conditions")
    observed = {
        item.get("variant"): item.get("group_mode")
        for item in conditions or []
        if isinstance(item, dict)
    }
    if observed != EXPECTED_CONDITIONS or len(conditions or []) != len(EXPECTED_CONDITIONS):
        fail("matrix condition families do not match Node 0's required seven families")
    random = matrix.get("random")
    if random != {
        "selection_seed": 2026,
        "num_random_masks": 3,
        "child_mask_seed_rule": "mask_seed = formal_seed * 100 + mask_index",
        "child_mask_seeds": [202600, 202601, 202602],
    }:
        fail("Random must use seed 2026 and the three frozen child masks")
    for mode, count in GROUP_COUNTS.items():
        block = matrix.get("expected_group_budgets", {}).get(mode, {})
        if block.get("total_group_count") != count:
            fail(f"{mode} total group count drifted")
        if block.get("K_G") != [EXPECTED_K[mode][rho] for rho in RHOS]:
            fail(f"{mode} K_G sequence drifted")


def build_entry(base_config, matrix, variant, group_mode, rho):
    config = copy.deepcopy(base_config)
    output_root = osp.abspath(osp.join(PROJECT_ROOT, matrix["output_root"]))
    config.update(
        {
            "seed": FORMAL_SEED,
            "variant": variant,
            "requested_budget": 1.0 if variant == "conv_module_dense" else rho,
            "group_mode": group_mode,
            "selection_seed": 2026 if variant in RANDOM_VARIANTS else None,
            "num_random_masks": 3 if variant in RANDOM_VARIANTS else None,
        }
    )
    config["data"].update({"dataset": "office", "source": 0, "target": 1})
    config["output"].update({"root": output_root, "save_model": False})
    effective = resolve_effective_config(config, PROJECT_ROOT)
    identity = build_experiment_identity(effective)
    command = [
        "python", osp.join(PROJECT_ROOT, "train.py"), "--config",
        osp.abspath(BASE_CONFIG_PATH), "--dataset", "office", "--source", "0",
        "--target", "1", "--seed", "2026", "--gpu-id", "0", "--variant",
        variant, "--output-root", output_root, "--no-save-model",
    ]
    if group_mode is not None:
        command.extend(["--group-mode", group_mode, "--requested-budget", str(rho)])
    if variant in RANDOM_VARIANTS:
        command.extend(["--selection-seed", "2026", "--num-random-masks", "3"])
    command.extend(
        ["--experiment-key", identity["experiment_key"], "--experiment-config-sha256", identity["experiment_config_sha256"]]
    )
    total_group_count = GROUP_COUNTS.get(group_mode)
    return {
        "formal_baseline_protocol_revision": BASELINE_PROTOCOL,
        "implementation_revision": CONV_IMPLEMENTATION_REVISION,
        "protocol_revision": CONV_PROTOCOL_REVISION,
        "efficiency_protocol_revision": EFFICIENCY_PROTOCOL_REVISION,
        "source_checkpoint_revision": SOURCE_CHECKPOINT_REVISION,
        "experiment_key": identity["experiment_key"],
        "experiment_config_sha256": identity["experiment_config_sha256"],
        "scientific_config": identity["scientific_config"],
        "method": "shot",
        "task": "otta",
        "dataset": "office",
        "source": 0,
        "target": 1,
        "seed": FORMAL_SEED,
        "variant": variant,
        "group_mode": group_mode,
        "requested_budget": 1.0 if group_mode is None else rho,
        "selection_seed": 2026 if variant in RANDOM_VARIANTS else None,
        "selection_seed_mode": "same_as_run_seed" if variant in RANDOM_VARIANTS else None,
        "num_random_masks": 3 if variant in RANDOM_VARIANTS else None,
        "child_mask_seeds": [202600, 202601, 202602] if variant in RANDOM_VARIANTS else None,
        "total_group_count": total_group_count,
        "max_group_count": None if group_mode is None else EXPECTED_K[group_mode][rho],
        "effective_overrides": {
            "dataset": "office", "source": 0, "target": 1, "seed": FORMAL_SEED,
            "variant": variant, "group_mode": group_mode,
            "requested_budget": 1.0 if group_mode is None else rho,
            "selection_seed": 2026 if variant in RANDOM_VARIANTS else None,
            "num_random_masks": 3 if variant in RANDOM_VARIANTS else None,
            "gpu_id": "0", "save_model": False, "workers_per_gpu": 1,
            "output_root": output_root,
        },
        "command_args": command,
        "expected_output_root": experiment_output_root(effective),
    }


def build_plan(matrix, base_config):
    entries = []
    for variant, group_mode in EXPECTED_CONDITIONS.items():
        for rho in ((None,) if group_mode is None else RHOS):
            entries.append(build_entry(base_config, matrix, variant, group_mode, rho))
    return {
        "plan_schema_version": 1,
        "node": "node0_ad",
        "formal_baseline_protocol_revision": BASELINE_PROTOCOL,
        "protocol_revision": CONV_PROTOCOL_REVISION,
        "implementation_revision": CONV_IMPLEMENTATION_REVISION,
        "efficiency_protocol_revision": EFFICIENCY_PROTOCOL_REVISION,
        "source_checkpoint_revision": SOURCE_CHECKPOINT_REVISION,
        "matrix_path": osp.abspath(MATRIX_PATH),
        "base_config_path": osp.abspath(BASE_CONFIG_PATH),
        "experiment_count": len(entries),
        "experiments": entries,
        "formal_matrix_summary": {
            "dataset_counts": {"office": len(entries)},
            "variant_counts": {
                variant: sum(entry["variant"] == variant for entry in entries)
                for variant in EXPECTED_CONDITIONS
            },
            "transfer": {"source": 0, "target": 1, "identifier": "AD"},
            "scientific_condition_count": len(entries),
        },
    }


def validate_plan(plan, runs_root):
    entries = plan.get("experiments", [])
    if plan.get("experiment_count") != 19 or len(entries) != 19:
        fail("plan must contain exactly 19 scientific conditions")
    if len({entry["experiment_key"] for entry in entries}) != 19:
        fail("plan experiment keys are not unique")
    if len({entry["experiment_config_sha256"] for entry in entries}) != 19:
        fail("plan scientific identities are not unique")
    if len({entry["expected_output_root"] for entry in entries}) != 19:
        fail("planned output roots are not unique")
    observed = {(entry["variant"], entry["requested_budget"]) for entry in entries}
    expected = {("conv_module_dense", 1.0)} | {
        (variant, rho)
        for variant, mode in EXPECTED_CONDITIONS.items() if mode is not None
        for rho in RHOS
    }
    if observed != expected:
        fail("plan conditions are not exactly the required 19 entries")
    for entry in entries:
        if (entry["dataset"], entry["source"], entry["target"], entry["seed"]) != ("office", 0, 1, 2026):
            fail("plan transfer or seed drifted")
        if entry["variant"] not in EXPECTED_CONDITIONS or "_lbi" in entry["variant"]:
            fail("plan contains a prohibited variant")
        if entry["variant"] == "conv_module_dense":
            if entry["group_mode"] is not None or entry["requested_budget"] != 1.0:
                fail("dense anchor drifted")
        else:
            mode, rho = entry["group_mode"], entry["requested_budget"]
            if rho not in RHOS or entry["max_group_count"] != EXPECTED_K[mode][rho]:
                fail("formal sparse K_G drifted")
            if entry["total_group_count"] != GROUP_COUNTS[mode]:
                fail("formal sparse total group count drifted")
            if entry["variant"] in RANDOM_VARIANTS and (
                entry["selection_seed"] != 2026
                or entry["num_random_masks"] != 3
                or entry["child_mask_seeds"] != [202600, 202601, 202602]
            ):
                fail("Random three-mask semantics drifted")
        try:
            common = osp.commonpath([osp.abspath(entry["expected_output_root"]), runs_root])
        except ValueError as error:
            raise ValueError("invalid planned output root") from error
        if common != runs_root:
            fail("planned output root escapes formal runs root")


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
    matrix = read_yaml(MATRIX_PATH)
    base_config = load_yaml(BASE_CONFIG_PATH)
    assert_frozen_inputs(matrix, base_config)
    assert_matrix(matrix)
    plan = build_plan(matrix, base_config)
    validate_plan(plan, osp.abspath(osp.join(PROJECT_ROOT, matrix["output_root"])))
    if not args.verify_only:
        write_outputs(plan, args.output_dir)
    print(json.dumps({"valid": True, "experiment_count": 19, "transfer": "AD"}))


if __name__ == "__main__":
    main()
