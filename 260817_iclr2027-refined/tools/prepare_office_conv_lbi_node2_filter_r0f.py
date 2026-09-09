#!/usr/bin/env python3
"""Build and statically validate Node 2's frozen Office D->A R0F plan."""

import argparse
import copy
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_identity import build_experiment_identity
from protocol_constants import (CONV_CANDIDATE_PARAM_COUNT, CONV_GROUP_COUNTS,
    CONV_IMPLEMENTATION_REVISION, CONV_PROTOCOL_REVISION,
    EFFICIENCY_PROTOCOL_REVISION, FORMAL_SEED, SOURCE_CHECKPOINT_REVISION)
from shot_otta.artifacts import experiment_output_root
from shot_otta.config import DATASETS, load_yaml, resolve_effective_config

NODE = "node2_filter_r0f"
SEARCH_PROTOCOL = "OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2"
BRANCH_ROOT = PROJECT_ROOT / "experiment_logs/office_conv_lbi_r0f_filter_reachability_seed2026_20260827"
RUNS_ROOT = BRANCH_ROOT / "runs"
BASE_CONFIG = PROJECT_ROOT / "configs/otta_conv_lbi_protocol_20260826_v1.yaml"
NORMATIVE = (
    ("OTTA_CONV_LBI_PROTOCOL_20260826_v1.md", "# OTTA_CONV_LBI_PROTOCOL_20260826_v1"),
    ("OTTA_CONV_BASELINE_FORMAL_20260827_v1.md", "# OTTA_CONV_BASELINE_FORMAL_20260827_v1"),
    ("OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v1.md", "# OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v1"),
    ("OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2.md", "# OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2"),
)
ANCHORS = (("F0", 0.25, 0.5), ("F1", 0.3, 0.5), ("F2", 0.4, 0.5), ("F3", 0.6, 0.5))
LBI_FIXED = {"kappa": 1.0, "omega": 0.00625, "stage1_max_steps": 3000,
    "budget_tolerance": 1.0e-4, "stage2_lr": 0.005, "stage2_steps": 1,
    "delta_nonzero_tolerance": 1.0e-12, "support_threshold": 1.0e-4}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def verify_frozen_inputs(base):
    protocol_dir = PROJECT_ROOT / "protocol/shot-otta_conv"
    for filename, heading in NORMATIVE:
        path = protocol_dir / filename
        require(path.is_file(), f"required normative file is missing: {path}")
        require(path.read_text(encoding="utf-8").startswith(heading), f"normative heading drifted: {path}")
    require(CONV_IMPLEMENTATION_REVISION == "iclr2027_refined_conv_20260826_v1", "Conv implementation revision drifted")
    require(base["implementation_revision"] == CONV_IMPLEMENTATION_REVISION, "base config implementation revision drifted")
    require(base["protocol_track"] == "conv" and base["protocol_revision"] == CONV_PROTOCOL_REVISION, "base config Conv protocol drifted")
    require((DATASETS["office"]["domains"][1], DATASETS["office"]["domains"][0]) == ("dslr", "amazon"), "Office D->A mapping must be source=1,target=0")

def lbi(alpha, nu):
    return {"alpha": alpha, "nu": nu, **LBI_FIXED}

def entry(base, anchor, alpha, nu):
    config = copy.deepcopy(base)
    tuple_lbi = lbi(alpha, nu)
    anchor_root = (RUNS_ROOT / f"anchor_{anchor}").resolve()
    config.update({"seed": FORMAL_SEED, "variant": "conv_filter_lbi", "requested_budget": 0.001,
                   "group_mode": "filter_connection", "lbi": copy.deepcopy(tuple_lbi)})
    config["data"].update({"dataset": "office", "source": 1, "target": 0})
    config["output"].update({"root": str(anchor_root), "save_model": False})
    effective = resolve_effective_config(config, str(WORKSPACE_ROOT))
    identity = build_experiment_identity(effective)
    command = ["python", str((PROJECT_ROOT / "train.py").resolve()), "--config", str(BASE_CONFIG.resolve()),
        "--dataset", "office", "--source", "1", "--target", "0", "--seed", "2026", "--gpu-id", "0",
        "--variant", "conv_filter_lbi", "--group-mode", "filter_connection", "--requested-budget", "0.001",
        "--lbi-alpha", str(alpha), "--lbi-kappa", "1.0", "--lbi-nu", str(nu), "--lbi-omega", "0.00625",
        "--lbi-stage1-max-steps", "3000", "--lbi-budget-tolerance", "0.0001", "--lbi-stage2-lr", "0.005",
        "--lbi-stage2-steps", "1", "--lbi-delta-nonzero-tolerance", "1e-12", "--lbi-support-threshold", "0.0001",
        "--output-root", str(anchor_root), "--no-save-model", "--experiment-key", identity["experiment_key"],
        "--experiment-config-sha256", identity["experiment_config_sha256"]]
    return {"plan_schema_version": 1, "node": NODE, "phase": "R0F", "search_protocol_revision": SEARCH_PROTOCOL,
        "implementation_revision": CONV_IMPLEMENTATION_REVISION, "protocol_revision": CONV_PROTOCOL_REVISION,
        "efficiency_protocol_revision": EFFICIENCY_PROTOCOL_REVISION, "source_checkpoint_revision": SOURCE_CHECKPOINT_REVISION,
        "experiment_key": identity["experiment_key"], "experiment_config_sha256": identity["experiment_config_sha256"],
        "scientific_config": identity["scientific_config"], "method": "shot", "task": "otta", "dataset": "office",
        "transfer": "DA", "source": 1, "target": 0, "source_name": "dslr", "target_name": "amazon", "seed": FORMAL_SEED,
        "anchor": anchor, "alpha": alpha, "nu": nu, "variant": "conv_filter_lbi", "group_mode": "filter_connection",
        "requested_budget": 0.001, "rho_G": 0.001, "total_group_count": CONV_GROUP_COUNTS["filter_connection"],
        "max_group_count": 6553, "K_G": 6553, "candidate_conv_scalar_count": CONV_CANDIDATE_PARAM_COUNT,
        "lbi": tuple_lbi, "runtime_comparable": False,
        "effective_overrides": {"dataset": "office", "source": 1, "target": 0, "seed": FORMAL_SEED,
            "variant": "conv_filter_lbi", "group_mode": "filter_connection", "requested_budget": 0.001,
            "anchor": anchor, "lbi": tuple_lbi, "gpu_id": "0", "save_model": False, "workers_per_gpu": 1,
            "runtime_comparable": False, "output_root": str(anchor_root)},
        "command_args": command, "expected_output_root": experiment_output_root(effective)}

def validate(entries):
    require(len(entries) == 4, "R0F node must have exactly four rows")
    for field in ("experiment_key", "experiment_config_sha256", "expected_output_root"):
        require(len({row[field] for row in entries}) == 4, f"{field} values must be unique")
    require([(row["anchor"], row["alpha"], row["nu"]) for row in entries] == list(ANCHORS), "anchor table drifted")
    for row in entries:
        require((row["dataset"], row["source"], row["target"], row["seed"]) == ("office", 1, 0, 2026), "transfer or seed drifted")
        require(row["variant"] == "conv_filter_lbi" and row["group_mode"] == "filter_connection", "non-filter LBI row present")
        require(row["rho_G"] == 0.001 and row["K_G"] == 6553 and row["max_group_count"] == 6553, "rho_G or K_G drifted")
        require(row["total_group_count"] == 6553600, "filter group count drifted")
        require(row["lbi"] == lbi(row["alpha"], row["nu"]), "LBI constants drifted")
        require(row["runtime_comparable"] is False, "R0F tuning run must not be runtime-comparable")
        serialized = json.dumps(row).lower()
        require("out_channel" not in serialized and "visda" not in serialized and "baseline" not in serialized, "prohibited condition leaked into plan")
        require(float(row["requested_budget"]) != 0.005, "prohibited rho_G=.005 present")
        Path(row["expected_output_root"]).resolve().relative_to(RUNS_ROOT.resolve())

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    base = load_yaml(BASE_CONFIG)
    verify_frozen_inputs(base)
    generated = [entry(base, *anchor) for anchor in ANCHORS]
    validate(generated)
    if args.verify_only:
        require(args.plan.is_file(), f"plan is missing: {args.plan}")
        stored = read_jsonl(args.plan)
        validate(stored)
        require(stored == generated, "stored R0F plan differs from frozen generated plan")
    else:
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        args.plan.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in generated), encoding="utf-8")
    print(json.dumps({"valid": True, "node": NODE, "experiment_count": 4, "transfer": "DA", "runtime_comparable": False}))

if __name__ == "__main__":
    main()
