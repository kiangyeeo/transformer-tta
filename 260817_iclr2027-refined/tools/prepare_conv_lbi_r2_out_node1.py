#!/usr/bin/env python3
"""Build and verify Node 1's frozen R2 out-channel cell plan."""

import copy
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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

CELL_ID = "omega_0p00625__lr_0p01"
OMEGA = 0.00625
STAGE2_LR = 0.01
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

PLAN_ROOT = PROJECT_ROOT / "experiment_logs/conv_lbi_r2_out_32gpu_20260828"
PLAN_PATH = PLAN_ROOT / "plans/node1_omega_0p00625__lr_0p01.jsonl"
BASE_CONFIG_PATH = PROJECT_ROOT / "configs/otta_conv_lbi_protocol_20260826_v1.yaml"
FROZEN_R1_PATH = PROJECT_ROOT / (
    "experiment_logs/office_conv_lbi_r1_out_stage1_validation_seed2026_20260827/"
    "selected_configs/OFFICE_OUT_STAGE1_FROZEN.json"
)
OUTPUT_ROOT = PROJECT_ROOT / (
    "experiment_logs/office_conv_lbi_r2_out_joint_sweep_seed2026_20260828/"
    "runs/omega_0p00625__lr_0p01"
)
SEARCH_V1 = "OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v1"
SEARCH_V2 = "OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2"
PROTOCOL_SHA256 = {
    "OTTA_CONV_LBI_PROTOCOL_20260826_v1.md": "5c2667f29b2c9847de601542fe65a518778fb21ad349a7ea0f466df300534c5e",
    "OTTA_CONV_BASELINE_FORMAL_20260827_v1.md": "41c4fe0bb69fed8389bea136e14ff7e535da258b2cbb95e2abaf9e375c759542",
    "OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v1.md": "767705c53cbe30961a7abdda782368c26531869afd6c5888dc2f338c5ffc974b",
    "OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2.md": "ff7287d2607343c53d93c2c6497e1c1aaadd703090b6937d424d1629692b7829",
}
FROZEN_R1_SHA256 = "8fcd88d347cfd4ebfcfb1cfc99d3aeb7e00db3910121f1cac19e887f5afd072a"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lbi(alpha, nu):
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
    protocol_root = PROJECT_ROOT / "protocol/shot-otta_conv"
    for name, expected_sha in PROTOCOL_SHA256.items():
        path = protocol_root / name
        require(path.is_file(), f"missing normative protocol: {path}")
        require(sha256(path) == expected_sha, f"normative protocol SHA drifted: {path}")
    require(
        CONV_IMPLEMENTATION_REVISION == "iclr2027_refined_conv_20260826_v1",
        "implementation revision drifted",
    )
    require(DATASETS["office"]["domains"] == ["amazon", "dslr", "webcam"], "Office domain ordering drifted")
    require(FROZEN_R1_PATH.is_file(), f"missing frozen R1 selection: {FROZEN_R1_PATH}")
    require(sha256(FROZEN_R1_PATH) == FROZEN_R1_SHA256, "frozen R1 SHA drifted")
    frozen = json.loads(FROZEN_R1_PATH.read_text(encoding="utf-8"))
    observed = {
        float(cell["requested_budget"]): (
            cell["primary"]["anchor"],
            float(cell["primary"]["alpha"]),
            float(cell["primary"]["nu"]),
            int(cell["K_G"]),
        )
        for cell in frozen.get("cells", [])
    }
    expected = {
        rho: (*FROZEN_ANCHORS[rho], RHO_TO_K[rho])
        for rho in RHOS
    }
    require(observed == expected, f"frozen R1 primary anchors drifted: {observed!r}")


def raw_config(base_config, transfer, rho):
    transfer_name, source, target, _, _ = transfer
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
        "lbi": lbi(alpha, nu),
        "allow_unresolved_lbi": False,
    })
    config["data"].update({"dataset": "office", "source": source, "target": target})
    config["device"]["gpu_id"] = "0"
    config["output"].update({"root": str(OUTPUT_ROOT.resolve()), "save_model": False, "run_name": None})
    config.setdefault("runtime", {})["runtime_comparable"] = False
    domains = DATASETS["office"]["domains"]
    require(transfer_name == f"{domains[source][0].upper()}{domains[target][0].upper()}", "transfer identifier drifted")
    return config


def command_for(config, identity):
    values, data = config["lbi"], config["data"]
    return [
        "python", str((PROJECT_ROOT / "train.py").resolve()), "--config", str(BASE_CONFIG_PATH.resolve()),
        "--dataset", "office", "--source", str(data["source"]), "--target", str(data["target"]),
        "--seed", "2026", "--gpu-id", "0", "--variant", "conv_out_lbi", "--group-mode", "out_channel",
        "--requested-budget", str(config["requested_budget"]), "--lbi-alpha", str(values["alpha"]),
        "--lbi-kappa", str(values["kappa"]), "--lbi-nu", str(values["nu"]), "--lbi-omega", str(values["omega"]),
        "--lbi-stage1-max-steps", str(values["stage1_max_steps"]), "--lbi-budget-tolerance", str(values["budget_tolerance"]),
        "--lbi-stage2-lr", str(values["stage2_lr"]), "--lbi-stage2-steps", str(values["stage2_steps"]),
        "--lbi-delta-nonzero-tolerance", str(values["delta_nonzero_tolerance"]),
        "--lbi-support-threshold", str(values["support_threshold"]), "--output-root", config["output"]["root"],
        "--no-save-model", "--experiment-key", identity["experiment_key"],
        "--experiment-config-sha256", identity["experiment_config_sha256"],
    ]


def build_entries():
    verify_frozen_inputs()
    base_config = load_yaml(BASE_CONFIG_PATH)
    require(base_config["protocol_revision"] == CONV_PROTOCOL_REVISION, "base config protocol drifted")
    require(base_config["implementation_revision"] == CONV_IMPLEMENTATION_REVISION, "base config implementation drifted")
    entries = []
    for rho in RHOS:
        for transfer in TRANSFERS:
            name, source, target, source_name, target_name = transfer
            anchor, alpha, nu = FROZEN_ANCHORS[rho]
            raw = raw_config(base_config, transfer, rho)
            effective = resolve_effective_config(raw, str(PROJECT_ROOT.parent))
            identity = build_experiment_identity(effective)
            for field, changed_value in (("omega", 0.0125), ("stage2_lr", 0.005)):
                changed_raw = copy.deepcopy(raw)
                changed_raw["lbi"][field] = changed_value
                changed_identity = build_experiment_identity(
                    resolve_effective_config(changed_raw, str(PROJECT_ROOT.parent))
                )
                require(
                    changed_identity["experiment_config_sha256"] != identity["experiment_config_sha256"],
                    f"{field} does not participate in the scientific identity hash",
                )
            entries.append({
                "plan_schema_version": 1,
                "node": "node1",
                "phase": "R2_out_joint_sweep",
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
                "requested_budget": rho,
                "total_group_count": CONV_GROUP_COUNTS["out_channel"],
                "max_group_count": max_support_count(rho, CONV_GROUP_COUNTS["out_channel"]),
                "lbi": lbi(alpha, nu),
                "candidate_conv_scalar_count": CONV_CANDIDATE_PARAM_COUNT,
                "runtime_comparable": False,
                "effective_overrides": {
                    "dataset": "office", "source": source, "target": target, "seed": FORMAL_SEED,
                    "variant": "conv_out_lbi", "group_mode": "out_channel", "requested_budget": rho,
                    "anchor": anchor, "lbi": lbi(alpha, nu), "search_protocol_revision": SEARCH_V2,
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
    require(len(entries) == 18, f"expected 18 rows, got {len(entries)}")
    for field in ("experiment_key", "experiment_config_sha256", "expected_output_root"):
        require(len({entry[field] for entry in entries}) == 18, f"{field} values are not unique")
    expected_mapping = {name: (source, target, source_name, target_name) for name, source, target, source_name, target_name in TRANSFERS}
    require({entry["transfer"] for entry in entries} == set(expected_mapping), "transfer set drifted")
    for name, mapping in expected_mapping.items():
        subset = [entry for entry in entries if entry["transfer"] == name]
        require(len(subset) == 3, f"{name} must have exactly three rows")
        for entry in subset:
            require((entry["source"], entry["target"], entry["source_name"], entry["target_name"]) == mapping, f"{name} mapping drifted")
    for rho in RHOS:
        subset = [entry for entry in entries if entry["requested_budget"] == rho]
        anchor, alpha, nu = FROZEN_ANCHORS[rho]
        require(len(subset) == 6, f"budget {rho} must have six rows")
        for entry in subset:
            require(entry["variant"] == "conv_out_lbi" and entry["group_mode"] == "out_channel", "method/group drifted")
            require(entry["max_group_count"] == RHO_TO_K[rho], "budget/K_G drifted")
            require((entry["anchor"], entry["alpha"], entry["nu"]) == (anchor, alpha, nu), "frozen anchor drifted")
            require(entry["lbi"] == lbi(alpha, nu), "LBI tuple drifted")
            require(entry["runtime_comparable"] is False and entry["effective_overrides"]["runtime_comparable"] is False, "runtime label drifted")
            scientific_lbi = entry["scientific_config"].get("lbi", {})
            require(scientific_lbi.get("omega") == OMEGA and scientific_lbi.get("stage2_lr") == STAGE2_LR, "omega/stage2_lr absent from scientific config")
            require(Path(entry["expected_output_root"]).resolve().is_relative_to(OUTPUT_ROOT.resolve()), "expected output root escapes cell root")
    require(all(entry["requested_budget"] != 0.005 for entry in entries), "forbidden .005 budget present")
    require(all(entry["lbi"]["stage2_lr"] != 0.005 for entry in entries), "central .00625/.005 row present")
    require(not any("filter_connection" in json.dumps(entry) or "baseline" in json.dumps(entry).lower() or "visda" in json.dumps(entry).lower() for entry in entries), "prohibited row type present")
    sample = entries[0]
    changed = copy.deepcopy(sample["scientific_config"])
    changed["lbi"]["omega"] = 0.0125
    require(changed != sample["scientific_config"], "omega mutation did not affect scientific config")


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
        raise SystemExit("usage: prepare_conv_lbi_r2_out_node1.py [--verify-existing]")
    else:
        PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
        PLAN_PATH.write_text("".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries), encoding="utf-8")
    print(json.dumps({
        "valid": True,
        "cell_id": CELL_ID,
        "experiment_count": 18,
        "unique_experiment_keys": 18,
        "unique_scientific_sha256": 18,
        "unique_output_roots": 18,
        "plan_sha256": sha256(PLAN_PATH),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
