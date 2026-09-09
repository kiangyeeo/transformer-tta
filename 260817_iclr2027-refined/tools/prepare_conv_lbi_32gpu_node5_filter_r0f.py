#!/usr/bin/env python3
"""Build and statically validate Node 5's frozen R0F filter rescue plan."""

import argparse
import copy
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
PLAN_PATH = (
    PROJECT_ROOT
    / "experiment_logs/conv_lbi_32gpu_burst_20260827/plans/node5_filter_r0f.jsonl"
)
RUNNER_PLAN_PATH = PLAN_PATH.with_name("node5_filter_r0f.runner.json")
RUNS_ROOT = (
    PROJECT_ROOT
    / "experiment_logs/office_conv_lbi_r0f_filter_reachability_seed2026_20260827/runs"
).resolve()
BASE_CONFIG_PATH = PROJECT_ROOT / "configs/otta_conv_lbi_protocol_20260826_v1.yaml"
NORMATIVE_PROTOCOLS = (
    ("OTTA_CONV_LBI_PROTOCOL_20260826_v1.md", "# OTTA_CONV_LBI_PROTOCOL_20260826_v1"),
    ("OTTA_CONV_BASELINE_FORMAL_20260827_v1.md", "# OTTA_CONV_BASELINE_FORMAL_20260827_v1"),
    ("OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v1.md", "# OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v1"),
    ("OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2.md", "# OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2"),
)
ANCHORS = (("F4", 0.2, 0.25), ("F5", 0.3, 0.25), ("F6", 0.4, 0.25), ("F7", 0.3, 0.125))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.lbi.diagnostics import max_support_count
from experiment_identity import build_experiment_identity
from protocol_constants import (
    CONV_CANDIDATE_PARAM_COUNT,
    CONV_GROUP_COUNTS,
    CONV_IMPLEMENTATION_REVISION,
    CONV_PROTOCOL_REVISION,
    EFFICIENCY_PROTOCOL_REVISION,
    FORMAL_SEED,
    SOURCE_CHECKPOINT_REVISION,
)
from shot_otta.artifacts import experiment_output_root
from shot_otta.config import DATASETS, load_yaml, resolve_effective_config


def require(condition, message):
    if not condition:
        raise ValueError(message)


def lbi(alpha, nu):
    return {
        "alpha": alpha,
        "kappa": 1.0,
        "nu": nu,
        "omega": 0.00625,
        "stage1_max_steps": 3000,
        "budget_tolerance": 0.0001,
        "stage2_lr": 0.005,
        "stage2_steps": 1,
        "delta_nonzero_tolerance": 1.0e-12,
        "support_threshold": 1.0e-4,
    }


def audit_frozen_inputs():
    protocol_dir = PROJECT_ROOT / "protocol/shot-otta_conv"
    for filename, heading in NORMATIVE_PROTOCOLS:
        path = protocol_dir / filename
        require(path.is_file(), f"missing normative protocol: {path}")
        require(path.read_text(encoding="utf-8").startswith(heading), f"normative protocol heading drifted: {path}")
    require(CONV_IMPLEMENTATION_REVISION == "iclr2027_refined_conv_20260826_v1", "Conv implementation revision drifted")
    require(DATASETS["office"]["domains"] == ["amazon", "dslr", "webcam"], "Office domain ordering drifted")
    require(max_support_count(0.002, CONV_GROUP_COUNTS["filter_connection"]) == 13107, "filter K_G=.002 drifted")


def raw_config(base_config, anchor, alpha, nu):
    config = copy.deepcopy(base_config)
    config.update(
        {
            "protocol_track": "conv",
            "method": "shot",
            "task": "otta",
            "formal_protocol": True,
            "protocol_revision": CONV_PROTOCOL_REVISION,
            "implementation_revision": CONV_IMPLEMENTATION_REVISION,
            "search_protocol_revision": "OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2",
            "source_checkpoint_revision": SOURCE_CHECKPOINT_REVISION,
            "seed": FORMAL_SEED,
            "variant": "conv_filter_lbi",
            "requested_budget": 0.002,
            "group_mode": "filter_connection",
            "lbi": lbi(alpha, nu),
            "allow_unresolved_lbi": False,
        }
    )
    config["data"].update({"dataset": "office", "source": 1, "target": 0})
    config["output"].update({"root": str((RUNS_ROOT / f"anchor_{anchor}").resolve()), "save_model": False, "run_name": None})
    config["device"]["gpu_id"] = "0"
    return config


def command_args(raw, identity):
    values = raw["lbi"]
    return [
        "python", str((PROJECT_ROOT / "train.py").resolve()), "--config", str(BASE_CONFIG_PATH.resolve()),
        "--dataset", "office", "--source", "1", "--target", "0", "--seed", "2026", "--gpu-id", "0",
        "--variant", "conv_filter_lbi", "--group-mode", "filter_connection", "--requested-budget", "0.002",
        "--lbi-alpha", str(values["alpha"]), "--lbi-kappa", "1.0", "--lbi-nu", str(values["nu"]),
        "--lbi-omega", "0.00625", "--lbi-stage1-max-steps", "3000", "--lbi-budget-tolerance", "0.0001",
        "--lbi-stage2-lr", "0.005", "--lbi-stage2-steps", "1", "--lbi-delta-nonzero-tolerance", "1e-12",
        "--lbi-support-threshold", "0.0001", "--output-root", raw["output"]["root"], "--no-save-model",
        "--experiment-key", identity["experiment_key"], "--experiment-config-sha256", identity["experiment_config_sha256"],
    ]


def build_entries():
    base_config = load_yaml(BASE_CONFIG_PATH)
    entries = []
    for anchor, alpha, nu in ANCHORS:
        raw = raw_config(base_config, anchor, alpha, nu)
        effective = resolve_effective_config(raw, str(WORKSPACE_ROOT))
        identity = build_experiment_identity(effective)
        entries.append(
            {
                "implementation_revision": CONV_IMPLEMENTATION_REVISION,
                "protocol_revision": CONV_PROTOCOL_REVISION,
                "search_protocol_revision": "OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2",
                "efficiency_protocol_revision": EFFICIENCY_PROTOCOL_REVISION,
                "source_checkpoint_revision": SOURCE_CHECKPOINT_REVISION,
                "experiment_key": identity["experiment_key"],
                "experiment_config_sha256": identity["experiment_config_sha256"],
                "scientific_config": identity["scientific_config"],
                "method": "shot", "task": "otta", "phase": "R0F", "dataset": "office", "transfer": "DA",
                "source": 1, "target": 0, "source_name": "dslr", "target_name": "amazon", "seed": FORMAL_SEED,
                "variant": "conv_filter_lbi", "group_mode": "filter_connection", "anchor": anchor,
                "alpha": alpha, "nu": nu, "requested_budget": 0.002,
                "total_group_count": CONV_GROUP_COUNTS["filter_connection"], "max_group_count": 13107,
                "lbi": lbi(alpha, nu), "candidate_conv_scalar_count": CONV_CANDIDATE_PARAM_COUNT,
                "runtime_comparable": False,
                "effective_overrides": {
                    "dataset": "office", "source": 1, "target": 0, "seed": FORMAL_SEED,
                    "variant": "conv_filter_lbi", "group_mode": "filter_connection", "requested_budget": 0.002,
                    "anchor": anchor, "lbi": lbi(alpha, nu),
                    "search_protocol_revision": "OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2",
                    "gpu_id": "0", "save_model": False, "output_root": raw["output"]["root"],
                    "runtime_comparable": False,
                },
                "config_path": str(BASE_CONFIG_PATH.resolve()),
                "command_args": command_args(raw, identity),
                "expected_output_root": experiment_output_root(effective),
            }
        )
    return entries


def validate(entries):
    require(len(entries) == 4, "R0F Node 5 must have exactly four rows")
    require([(entry["anchor"], entry["alpha"], entry["nu"]) for entry in entries] == list(ANCHORS), "R0F anchor grid drifted")
    for field in ("experiment_key", "experiment_config_sha256", "expected_output_root"):
        require(len({entry[field] for entry in entries}) == 4, f"R0F {field} values are not unique")
    for entry in entries:
        require((entry["dataset"], entry["transfer"], entry["source"], entry["target"], entry["seed"]) == ("office", "DA", 1, 0, 2026), "R0F transfer or seed drifted")
        require((entry["source_name"], entry["target_name"]) == ("dslr", "amazon"), "R0F domain names drifted")
        require((entry["variant"], entry["group_mode"], entry["requested_budget"], entry["max_group_count"]) == ("conv_filter_lbi", "filter_connection", 0.002, 13107), "R0F method, group mode, rho_G, or K_G drifted")
        require(entry["runtime_comparable"] is False, "diagnostic run cannot be runtime comparable")
        require(entry["lbi"] == lbi(entry["alpha"], entry["nu"]), "R0F frozen LBI constants drifted")
        require("out_channel" not in json.dumps(entry), "R0F plan includes prohibited out-channel")
        require("baseline" not in entry["variant"] and "VISDA" not in json.dumps(entry), "R0F plan includes prohibited baseline or VisDA")
        require(entry["requested_budget"] != 0.005, "R0F plan includes prohibited .005 budget")
        try:
            Path(entry["expected_output_root"]).resolve().relative_to(RUNS_ROOT)
        except ValueError as error:
            raise ValueError("R0F output root escapes canonical branch root") from error


def runner_plan(entries):
    return {
        "plan_schema_version": 1,
        "phase": "R0F",
        "plan_name": "office_da_conv_lbi_filter_r0f_p002_node5_seed2026_20260827",
        "protocol_revision": CONV_PROTOCOL_REVISION,
        "search_protocol_revision": "OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2",
        "implementation_revision": CONV_IMPLEMENTATION_REVISION,
        "efficiency_protocol_revision": EFFICIENCY_PROTOCOL_REVISION,
        "source_checkpoint_revision": SOURCE_CHECKPOINT_REVISION,
        "base_config_path": str(BASE_CONFIG_PATH.resolve()),
        "experiment_count": 4,
        "runs_root": str(RUNS_ROOT),
        "experiments": entries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    audit_frozen_inputs()
    expected = build_entries()
    validate(expected)
    if args.verify_only:
        require(PLAN_PATH.is_file() and RUNNER_PLAN_PATH.is_file(), "Node 5 plan artifacts are missing")
        observed = [json.loads(line) for line in PLAN_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        require(observed == expected, "Node 5 JSONL plan drifted")
        require(json.loads(RUNNER_PLAN_PATH.read_text(encoding="utf-8")) == runner_plan(expected), "Node 5 runner plan drifted")
    else:
        PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
        PLAN_PATH.write_text("".join(json.dumps(entry, sort_keys=True) + "\n" for entry in expected), encoding="utf-8")
        RUNNER_PLAN_PATH.write_text(json.dumps(runner_plan(expected), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"valid": True, "experiment_count": 4, "unique_keys": 4, "unique_config_sha256": 4, "unique_output_roots": 4, "runtime_comparable": False}, sort_keys=True))


if __name__ == "__main__":
    main()
