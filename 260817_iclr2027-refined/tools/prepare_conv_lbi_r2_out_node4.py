#!/usr/bin/env python3
"""Build and verify Node 4's frozen Office out-channel R2 cell plan."""

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

CELL_ID = "omega_0p025__lr_0p01"
OMEGA = 0.025
STAGE2_LR = 0.01
SEARCH_V1 = "OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v1"
SEARCH_V2 = "OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2"
BASE_CONFIG_PATH = PROJECT_ROOT / "configs/otta_conv_lbi_protocol_20260826_v1.yaml"
PLAN_PATH = PROJECT_ROOT / "experiment_logs/conv_lbi_r2_out_32gpu_20260828/plans/node4_omega_0p025__lr_0p01.jsonl"
CELL_RUNS_ROOT = PROJECT_ROOT / "experiment_logs/office_conv_lbi_r2_out_joint_sweep_seed2026_20260828/runs" / CELL_ID
FROZEN_R1_PATH = PROJECT_ROOT / "experiment_logs/office_conv_lbi_r1_out_stage1_validation_seed2026_20260827/selected_configs/OFFICE_OUT_STAGE1_FROZEN.json"

RHOS = (0.0005, 0.001, 0.002)
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


def lbi(anchor):
    _, alpha, nu = anchor
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


def verify_normative_inputs():
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
    require(CONV_IMPLEMENTATION_REVISION == "iclr2027_refined_conv_20260826_v1", "implementation revision drifted")
    require(DATASETS["office"]["domains"] == ["amazon", "dslr", "webcam"], "Office domain ordering drifted")
    require({rho: max_support_count(rho, CONV_GROUP_COUNTS["out_channel"]) for rho in RHOS} == RHO_TO_K, "out-channel rho/K_G mapping drifted")

    frozen = json.loads(FROZEN_R1_PATH.read_text(encoding="utf-8"))
    require(frozen.get("status") == "COMPLETE", "R1 frozen selection is not complete")
    observed = {
        float(cell["requested_budget"]): (
            cell["primary"]["anchor"],
            float(cell["primary"]["alpha"]),
            float(cell["primary"]["nu"]),
        )
        for cell in frozen.get("cells", [])
        if cell.get("group_mode") == "out_channel"
    }
    require(observed == FROZEN_ANCHORS, f"frozen R1 anchors drifted: {observed!r}")


def raw_config(base_config, transfer, rho):
    name, source, target, _, _ = transfer
    anchor = FROZEN_ANCHORS[rho]
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
        "lbi": lbi(anchor),
        "allow_unresolved_lbi": False,
    })
    config["data"].update({"dataset": "office", "source": source, "target": target})
    config["device"]["gpu_id"] = "0"
    config["output"].update({"root": str(CELL_RUNS_ROOT.resolve()), "save_model": False, "run_name": None})
    config.setdefault("runtime", {})["runtime_comparable"] = False
    domains = DATASETS["office"]["domains"]
    require(name == f"{domains[source][0].upper()}{domains[target][0].upper()}", "transfer identifier drifted")
    return config


def command_for(raw, identity):
    values, data = raw["lbi"], raw["data"]
    return [
        "python", str((PROJECT_ROOT / "train.py").resolve()), "--config", str(BASE_CONFIG_PATH.resolve()),
        "--dataset", "office", "--source", str(data["source"]), "--target", str(data["target"]),
        "--seed", "2026", "--gpu-id", "0", "--variant", "conv_out_lbi", "--group-mode", "out_channel",
        "--requested-budget", str(raw["requested_budget"]), "--lbi-alpha", str(values["alpha"]),
        "--lbi-kappa", str(values["kappa"]), "--lbi-nu", str(values["nu"]), "--lbi-omega", str(values["omega"]),
        "--lbi-stage1-max-steps", str(values["stage1_max_steps"]), "--lbi-budget-tolerance", str(values["budget_tolerance"]),
        "--lbi-stage2-lr", str(values["stage2_lr"]), "--lbi-stage2-steps", str(values["stage2_steps"]),
        "--lbi-delta-nonzero-tolerance", str(values["delta_nonzero_tolerance"]),
        "--lbi-support-threshold", str(values["support_threshold"]), "--output-root", raw["output"]["root"],
        "--no-save-model", "--experiment-key", identity["experiment_key"],
        "--experiment-config-sha256", identity["experiment_config_sha256"],
    ]


def build_entries():
    verify_normative_inputs()
    base_config = load_yaml(BASE_CONFIG_PATH)
    require(base_config["protocol_revision"] == CONV_PROTOCOL_REVISION, "base config protocol drifted")
    require(base_config["implementation_revision"] == CONV_IMPLEMENTATION_REVISION, "base config implementation drifted")
    entries = []
    for rho in RHOS:
        anchor = FROZEN_ANCHORS[rho]
        for transfer in TRANSFERS:
            name, source, target, source_name, target_name = transfer
            raw = raw_config(base_config, transfer, rho)
            effective = resolve_effective_config(raw, str(WORKSPACE_ROOT))
            identity = build_experiment_identity(effective)
            entries.append({
                "plan_schema_version": 1,
                "node": "node4_r2_out",
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
                "anchor": anchor[0],
                "alpha": anchor[1],
                "nu": anchor[2],
                "requested_budget": rho,
                "total_group_count": CONV_GROUP_COUNTS["out_channel"],
                "max_group_count": RHO_TO_K[rho],
                "lbi": lbi(anchor),
                "candidate_conv_scalar_count": CONV_CANDIDATE_PARAM_COUNT,
                "runtime_comparable": False,
                "effective_overrides": {
                    "dataset": "office", "source": source, "target": target, "seed": FORMAL_SEED,
                    "variant": "conv_out_lbi", "group_mode": "out_channel", "requested_budget": rho,
                    "anchor": anchor[0], "lbi": lbi(anchor), "search_protocol_revision": SEARCH_V2,
                    "gpu_id": "0", "save_model": False, "workers_per_gpu": 1,
                    "runtime_comparable": False, "output_root": raw["output"]["root"],
                },
                "config_path": str(BASE_CONFIG_PATH.resolve()),
                "command_args": command_for(raw, identity),
                "expected_output_root": experiment_output_root(effective),
            })
    validate_entries(entries)
    return entries


def validate_entries(entries):
    require(len(entries) == 18, f"expected exactly 18 rows, got {len(entries)}")
    for field in ("experiment_key", "experiment_config_sha256", "expected_output_root"):
        require(len({entry[field] for entry in entries}) == 18, f"{field} values are not unique")
    expected_mapping = {name: (source, target, source_name, target_name) for name, source, target, source_name, target_name in TRANSFERS}
    require({entry["transfer"] for entry in entries} == set(expected_mapping), "transfer set drifted")
    for transfer, mapping in expected_mapping.items():
        rows = [entry for entry in entries if entry["transfer"] == transfer]
        require(len(rows) == 3, f"{transfer} must contain exactly three budget rows")
        for entry in rows:
            require((entry["source"], entry["target"], entry["source_name"], entry["target_name"]) == mapping, f"{transfer} mapping drifted")
    for rho, expected_k in RHO_TO_K.items():
        rows = [entry for entry in entries if entry["requested_budget"] == rho]
        require(len(rows) == 6, f"{rho} must contain exactly six transfer rows")
        anchor = FROZEN_ANCHORS[rho]
        for entry in rows:
            require((entry["anchor"], entry["alpha"], entry["nu"]) == anchor, f"frozen anchor drifted for {rho}")
            require(entry["max_group_count"] == expected_k, f"K_G drifted for {rho}")
    for entry in entries:
        values = entry["lbi"]
        require((entry["dataset"], entry["seed"], entry["variant"], entry["group_mode"]) == ("office", 2026, "conv_out_lbi", "out_channel"), "scientific scope drifted")
        require(values["omega"] == OMEGA and values["stage2_lr"] == STAGE2_LR, "assigned R2 cell drifted")
        require(values == lbi((entry["anchor"], entry["alpha"], entry["nu"])), "LBI tuple drifted")
        require(entry["scientific_config"]["candidate_scope"] == "netF.layer4_conv", "candidate scope drifted")
        require(entry["scientific_config"]["lbi"]["omega"] == OMEGA and entry["scientific_config"]["lbi"]["stage2_lr"] == STAGE2_LR, "identity excludes R2 axes")
        require(entry["runtime_comparable"] is False and entry["effective_overrides"]["runtime_comparable"] is False, "runtime label drifted")
        require(Path(entry["expected_output_root"]).resolve().is_relative_to(CELL_RUNS_ROOT.resolve()), "output root escapes assigned cell")
        prohibited = " ".join((entry["variant"], entry["group_mode"], entry["dataset"])).lower()
        require("filter_connection" not in prohibited and "baseline" not in prohibited and "visda" not in prohibited, "prohibited row type")
        require(entry["requested_budget"] != 0.005, "prohibited .005 budget present")
        require(not (values["omega"] == 0.00625 and values["stage2_lr"] == 0.005), "central R2 cell present")


def load_jsonl(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_entries(rows)
    return rows


def main():
    entries = build_entries()
    if len(sys.argv) == 2 and sys.argv[1] == "--verify-existing":
        require(PLAN_PATH.is_file(), f"missing plan: {PLAN_PATH}")
        require(load_jsonl(PLAN_PATH) == entries, "on-disk plan does not exactly match the frozen generated plan")
    elif len(sys.argv) != 1:
        raise SystemExit("usage: prepare_conv_lbi_r2_out_node4.py [--verify-existing]")
    else:
        PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
        PLAN_PATH.write_text("".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries), encoding="utf-8")
    digest = hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest()
    print(json.dumps({"valid": True, "cell_id": CELL_ID, "experiment_count": 18, "unique_experiment_keys": 18, "unique_scientific_sha256": 18, "unique_output_roots": 18, "plan_sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
