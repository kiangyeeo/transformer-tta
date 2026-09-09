#!/usr/bin/env python3
"""Build and strictly validate Node 5's frozen R2 out-channel cell plan."""

import copy
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
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

CELL_ID = "omega_0p1__lr_0p0025"
OMEGA = 0.1
STAGE2_LR = 0.0025
SEARCH_V1 = "OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v1"
SEARCH_V2 = "OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2"
BASE_CONFIG_PATH = PROJECT_ROOT / "configs/otta_conv_lbi_protocol_20260826_v1.yaml"
FROZEN_PATH = PROJECT_ROOT / (
    "experiment_logs/office_conv_lbi_r1_out_stage1_validation_seed2026_20260827/"
    "selected_configs/OFFICE_OUT_STAGE1_FROZEN.json"
)
FROZEN_FINALIZE_PATH = PROJECT_ROOT / (
    "experiment_logs/office_conv_lbi_r1_out_stage1_validation_seed2026_20260827/"
    "phase_records/R1/FINALIZE.md"
)
PLAN_PATH = PROJECT_ROOT / (
    "experiment_logs/conv_lbi_r2_out_32gpu_20260828/plans/"
    "node5_omega_0p1__lr_0p0025.jsonl"
)
BRANCH_ROOT = PROJECT_ROOT / (
    "experiment_logs/office_conv_lbi_r2_out_joint_sweep_seed2026_20260828/"
    "runs/omega_0p1__lr_0p0025"
)

RHO_TO_K = {0.0005: 4, 0.001: 9, 0.002: 18}
FROZEN_ANCHORS = {
    0.0005: ("A1", 0.05, 0.50),
    0.001: ("A1", 0.05, 0.50),
    0.002: ("A4", 0.10, 1.00),
}
TRANSFERS = (
    ("AD", 0, 1, "amazon", "dslr"),
    ("AW", 0, 2, "amazon", "webcam"),
    ("DA", 1, 0, "dslr", "amazon"),
    ("DW", 1, 2, "dslr", "webcam"),
    ("WA", 2, 0, "webcam", "amazon"),
    ("WD", 2, 1, "webcam", "dslr"),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def lbi_tuple(alpha, nu):
    return {
        "alpha": alpha,
        "kappa": 1.0,
        "nu": nu,
        "omega": OMEGA,
        "stage1_max_steps": 3000,
        "budget_tolerance": 1.0e-4,
        "stage2_lr": STAGE2_LR,
        "stage2_steps": 1,
        "delta_nonzero_tolerance": 1.0e-12,
        "support_threshold": 1.0e-4,
    }


def verify_frozen_inputs():
    required = {
        "OTTA_CONV_LBI_PROTOCOL_20260826_v1.md": "# OTTA_CONV_LBI_PROTOCOL_20260826_v1",
        "OTTA_CONV_BASELINE_FORMAL_20260827_v1.md": "# OTTA_CONV_BASELINE_FORMAL_20260827_v1",
        "OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v1.md": "# OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v1",
        "OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2.md": "# OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2",
    }
    protocol_root = PROJECT_ROOT / "protocol/shot-otta_conv"
    for name, heading in required.items():
        path = protocol_root / name
        require(path.is_file(), f"missing normative protocol: {path}")
        require(path.read_text(encoding="utf-8").startswith(heading), f"normative heading drifted: {path}")
    require(FROZEN_PATH.is_file(), f"missing frozen R1 selection: {FROZEN_PATH}")
    require(FROZEN_FINALIZE_PATH.is_file(), f"missing frozen R1 finalization: {FROZEN_FINALIZE_PATH}")
    require(CONV_IMPLEMENTATION_REVISION == "iclr2027_refined_conv_20260826_v1", "implementation revision drifted")
    require(DATASETS["office"]["domains"] == ["amazon", "dslr", "webcam"], "Office domain ordering drifted")
    expected_k = {rho: max_support_count(rho, CONV_GROUP_COUNTS["out_channel"]) for rho in RHO_TO_K}
    require(expected_k == RHO_TO_K, "out-channel rho/K_G mapping drifted")

    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    require(frozen.get("status") == "COMPLETE", "R1 frozen result is not complete")
    observed = {}
    for cell in frozen.get("cells", []):
        rho = float(cell.get("requested_budget"))
        primary = cell.get("primary", {})
        observed[rho] = (
            primary.get("anchor"),
            float(primary.get("alpha")),
            float(primary.get("nu")),
        )
        require(cell.get("group_mode") == "out_channel", "frozen selection group mode drifted")
        require(int(cell.get("K_G")) == RHO_TO_K[rho], "frozen selection K_G drifted")
    require(observed == FROZEN_ANCHORS, f"frozen primary anchors drifted: {observed}")


def raw_config(base_config, transfer, rho):
    name, source, target, _, _ = transfer
    anchor, alpha, nu = FROZEN_ANCHORS[rho]
    config = copy.deepcopy(base_config)
    config.update({
        "protocol_track": "conv",
        "method": "shot",
        "task": "otta",
        "formal_protocol": True,
        "protocol_revision": CONV_PROTOCOL_REVISION,
        "implementation_revision": CONV_IMPLEMENTATION_REVISION,
        "search_protocol_revision": SEARCH_V2,
        "search_protocol_base_revision": SEARCH_V1,
        "source_checkpoint_revision": SOURCE_CHECKPOINT_REVISION,
        "seed": FORMAL_SEED,
        "variant": "conv_out_lbi",
        "group_mode": "out_channel",
        "requested_budget": rho,
        "lbi": lbi_tuple(alpha, nu),
        "allow_unresolved_lbi": False,
    })
    config["data"].update({"dataset": "office", "source": source, "target": target})
    config["device"]["gpu_id"] = "0"
    config["output"].update({
        "root": str(BRANCH_ROOT.resolve()),
        "save_model": False,
        "run_name": None,
    })
    config.setdefault("runtime", {})["runtime_comparable"] = False
    domains = DATASETS["office"]["domains"]
    require(name == f"{domains[source][0].upper()}{domains[target][0].upper()}", "transfer identifier drifted")
    require(anchor == FROZEN_ANCHORS[rho][0], "frozen anchor label drifted")
    return config


def command_for(config, identity):
    lbi = config["lbi"]
    return [
        "python", str((PROJECT_ROOT / "train.py").resolve()), "--config", str(BASE_CONFIG_PATH.resolve()),
        "--dataset", "office", "--source", str(config["data"]["source"]), "--target", str(config["data"]["target"]),
        "--seed", "2026", "--gpu-id", "0", "--variant", "conv_out_lbi", "--group-mode", "out_channel",
        "--requested-budget", str(config["requested_budget"]), "--lbi-alpha", str(lbi["alpha"]),
        "--lbi-kappa", str(lbi["kappa"]), "--lbi-nu", str(lbi["nu"]), "--lbi-omega", str(lbi["omega"]),
        "--lbi-stage1-max-steps", str(lbi["stage1_max_steps"]), "--lbi-budget-tolerance", str(lbi["budget_tolerance"]),
        "--lbi-stage2-lr", str(lbi["stage2_lr"]), "--lbi-stage2-steps", str(lbi["stage2_steps"]),
        "--lbi-delta-nonzero-tolerance", str(lbi["delta_nonzero_tolerance"]),
        "--lbi-support-threshold", str(lbi["support_threshold"]), "--output-root", config["output"]["root"],
        "--no-save-model", "--experiment-key", identity["experiment_key"],
        "--experiment-config-sha256", identity["experiment_config_sha256"],
    ]


def build_entries():
    verify_frozen_inputs()
    base_config = load_yaml(BASE_CONFIG_PATH)
    require(base_config["protocol_revision"] == CONV_PROTOCOL_REVISION, "base config protocol drifted")
    require(base_config["implementation_revision"] == CONV_IMPLEMENTATION_REVISION, "base config implementation drifted")
    entries = []
    for rho in RHO_TO_K:
        anchor, alpha, nu = FROZEN_ANCHORS[rho]
        for transfer in TRANSFERS:
            name, source, target, source_name, target_name = transfer
            raw = raw_config(base_config, transfer, rho)
            effective = resolve_effective_config(raw, str(WORKSPACE_ROOT))
            identity = build_experiment_identity(effective)
            entries.append({
                "plan_schema_version": 1,
                "node": 5,
                "phase": "R2_out_omega_stage2_lr_joint_search",
                "cell_id": CELL_ID,
                "implementation_revision": CONV_IMPLEMENTATION_REVISION,
                "protocol_revision": CONV_PROTOCOL_REVISION,
                "search_protocol_revision": SEARCH_V2,
                "search_protocol_base_revision": SEARCH_V1,
                "efficiency_protocol_revision": EFFICIENCY_PROTOCOL_REVISION,
                "source_checkpoint_revision": SOURCE_CHECKPOINT_REVISION,
                "experiment_key": identity["experiment_key"],
                "experiment_config_sha256": identity["experiment_config_sha256"],
                "scientific_config": identity["scientific_config"],
                "method": "shot",
                "task": "otta",
                "dataset": "office",
                "transfer": name,
                "source": source,
                "target": target,
                "source_name": source_name,
                "target_name": target_name,
                "seed": FORMAL_SEED,
                "variant": "conv_out_lbi",
                "group_mode": "out_channel",
                "anchor": anchor,
                "alpha": alpha,
                "nu": nu,
                "omega": OMEGA,
                "stage2_lr": STAGE2_LR,
                "requested_budget": rho,
                "total_group_count": CONV_GROUP_COUNTS["out_channel"],
                "max_group_count": RHO_TO_K[rho],
                "lbi": lbi_tuple(alpha, nu),
                "candidate_conv_scalar_count": CONV_CANDIDATE_PARAM_COUNT,
                "runtime_comparable": False,
                "effective_overrides": {
                    "dataset": "office",
                    "source": source,
                    "target": target,
                    "seed": FORMAL_SEED,
                    "variant": "conv_out_lbi",
                    "group_mode": "out_channel",
                    "requested_budget": rho,
                    "anchor": anchor,
                    "lbi": lbi_tuple(alpha, nu),
                    "gpu_id": "0",
                    "save_model": False,
                    "workers_per_gpu": 1,
                    "runtime_comparable": False,
                    "output_root": raw["output"]["root"],
                },
                "config_path": str(BASE_CONFIG_PATH.resolve()),
                "command_args": command_for(raw, identity),
                "expected_output_root": experiment_output_root(effective),
            })
    validate_entries(entries)
    return entries


def validate_entries(entries):
    require(len(entries) == 18, f"expected 18 rows, got {len(entries)}")
    for field in ("experiment_key", "experiment_config_sha256", "expected_output_root"):
        require(len({entry[field] for entry in entries}) == 18, f"{field} values are not unique")
    expected_transfers = {name: (source, target, source_name, target_name) for name, source, target, source_name, target_name in TRANSFERS}
    require({entry["transfer"] for entry in entries} == set(expected_transfers), "transfer set drifted")
    require({entry["requested_budget"] for entry in entries} == set(RHO_TO_K), "budget grid drifted")
    for rho, expected_k in RHO_TO_K.items():
        budget_rows = [entry for entry in entries if entry["requested_budget"] == rho]
        require(len(budget_rows) == 6, f"budget {rho} must have exactly six rows")
        for entry in budget_rows:
            require(entry["max_group_count"] == expected_k, f"budget {rho} K_G drifted")
            require((entry["anchor"], entry["alpha"], entry["nu"]) == FROZEN_ANCHORS[rho], f"budget {rho} frozen anchor drifted")
    for entry in entries:
        require((entry["source"], entry["target"], entry["source_name"], entry["target_name"]) == expected_transfers[entry["transfer"]], "source/target mapping drifted")
        require((entry["dataset"], entry["seed"], entry["variant"], entry["group_mode"]) == ("office", 2026, "conv_out_lbi", "out_channel"), "scientific base fields drifted")
        require(entry["lbi"] == lbi_tuple(entry["alpha"], entry["nu"]), "LBI tuple drifted")
        require(entry["omega"] == OMEGA and entry["stage2_lr"] == STAGE2_LR, "assigned R2 cell drifted")
        scientific_lbi = entry["scientific_config"].get("lbi", {})
        require(scientific_lbi.get("omega") == OMEGA and scientific_lbi.get("stage2_lr") == STAGE2_LR, "omega/stage2_lr absent from scientific identity")
        require(entry["runtime_comparable"] is False and entry["effective_overrides"]["runtime_comparable"] is False, "runtime metadata drifted")
        require(entry["requested_budget"] != 0.005, "prohibited .005 budget present")
        require((entry["omega"], entry["stage2_lr"]) != (0.00625, 0.005), "central R2 cell must not be present")
        require("filter" not in entry["variant"] and "baseline" not in entry["variant"] and "visda" not in entry["dataset"].lower(), "prohibited row type")
        try:
            Path(entry["expected_output_root"]).resolve().relative_to(BRANCH_ROOT.resolve())
        except ValueError as error:
            raise ValueError("expected output root escapes this cell's branch root") from error


def load_jsonl(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_entries(rows)
    return rows


def main():
    entries = build_entries()
    if len(sys.argv) == 2 and sys.argv[1] == "--verify-existing":
        require(PLAN_PATH.is_file(), f"plan is missing: {PLAN_PATH}")
        require(load_jsonl(PLAN_PATH) == entries, "on-disk plan does not exactly match the frozen generated plan")
    elif len(sys.argv) != 1:
        raise SystemExit("usage: prepare_conv_lbi_r2_out_node5.py [--verify-existing]")
    else:
        PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
        PLAN_PATH.write_text("".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries), encoding="utf-8")
    digest = hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest() if PLAN_PATH.is_file() else None
    print(json.dumps({
        "valid": True,
        "experiment_count": 18,
        "unique_experiment_keys": 18,
        "unique_scientific_sha256": 18,
        "unique_output_roots": 18,
        "plan_sha256": digest,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
