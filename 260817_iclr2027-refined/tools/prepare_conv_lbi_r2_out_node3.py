#!/usr/bin/env python3
"""Create or verify Node 3's frozen R2 out-channel joint-search plan.

This is a CPU-only plan builder.  It deliberately uses the repository's
resolved effective configuration and canonical experiment-identity builder so
that both R2 coordinates are part of every scientific SHA256.
"""

import argparse
import copy
import hashlib
import json
import shlex
import sys
from pathlib import Path

import yaml

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

CELL_ID = "omega_0p025__lr_0p005"
OMEGA = 0.025
STAGE2_LR = 0.005
SEARCH_PROTOCOL_BASE_REVISION = "OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v1"
SEARCH_PROTOCOL_REVISION = "OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2"
BASE_CONFIG_PATH = PROJECT_ROOT / "configs/otta_conv_lbi_protocol_20260826_v1.yaml"
PLAN_ROOT = PROJECT_ROOT / "experiment_logs/conv_lbi_r2_out_32gpu_20260828/plans"
PLAN_PATH = PLAN_ROOT / f"node3_{CELL_ID}.jsonl"
CONFIG_DIR = PLAN_ROOT / f"node3_{CELL_ID}_configs"
BRANCH_RUNS_ROOT = (
    PROJECT_ROOT
    / "experiment_logs/office_conv_lbi_r2_out_joint_sweep_seed2026_20260828/runs"
    / CELL_ID
)
TRANSFERS = (
    ("AD", 0, 1, "amazon", "dslr"),
    ("AW", 0, 2, "amazon", "webcam"),
    ("DA", 1, 0, "dslr", "amazon"),
    ("DW", 1, 2, "dslr", "webcam"),
    ("WA", 2, 0, "webcam", "amazon"),
    ("WD", 2, 1, "webcam", "dslr"),
)
FROZEN_ANCHORS = {
    0.0005: ("A1", 0.05, 0.50, 4),
    0.001: ("A1", 0.05, 0.50, 9),
    0.002: ("A4", 0.10, 1.00, 18),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def frozen_lbi(alpha, nu):
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
    required_protocols = {
        "OTTA_CONV_LBI_PROTOCOL_20260826_v1.md": "# OTTA_CONV_LBI_PROTOCOL_20260826_v1",
        "OTTA_CONV_BASELINE_FORMAL_20260827_v1.md": "# OTTA_CONV_BASELINE_FORMAL_20260827_v1",
        "OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v1.md": "# OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v1",
        "OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2.md": "# OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2",
    }
    protocol_root = PROJECT_ROOT / "protocol/shot-otta_conv"
    for name, heading in required_protocols.items():
        path = protocol_root / name
        require(path.is_file(), f"missing normative protocol: {path}")
        require(path.read_text(encoding="utf-8").startswith(heading), f"normative heading drifted: {path}")
    require(CONV_IMPLEMENTATION_REVISION == "iclr2027_refined_conv_20260826_v1", "implementation revision drifted")
    require(DATASETS["office"]["domains"] == ["amazon", "dslr", "webcam"], "Office domain mapping drifted")
    require(
        {
            rho: max_support_count(rho, CONV_GROUP_COUNTS["out_channel"])
            for rho in FROZEN_ANCHORS
        }
        == {rho: values[3] for rho, values in FROZEN_ANCHORS.items()},
        "out-channel rho/K_G mapping drifted",
    )


def raw_config(base_config, transfer, rho, anchor, alpha, nu):
    _, source, target, _, _ = transfer
    config = copy.deepcopy(base_config)
    config.update(
        {
            "protocol_track": "conv",
            "formal_protocol": True,
            "protocol_revision": CONV_PROTOCOL_REVISION,
            "implementation_revision": CONV_IMPLEMENTATION_REVISION,
            "search_protocol_base_revision": SEARCH_PROTOCOL_BASE_REVISION,
            "search_protocol_revision": SEARCH_PROTOCOL_REVISION,
            "source_checkpoint_revision": SOURCE_CHECKPOINT_REVISION,
            "seed": FORMAL_SEED,
            "variant": "conv_out_lbi",
            "group_mode": "out_channel",
            "requested_budget": rho,
            "lbi": frozen_lbi(alpha, nu),
            "allow_unresolved_lbi": False,
        }
    )
    config["data"].update({"dataset": "office", "source": source, "target": target})
    config["device"]["gpu_id"] = "0"
    config["output"].update(
        {"root": str(BRANCH_RUNS_ROOT.resolve()), "save_model": False, "run_name": None}
    )
    config.setdefault("runtime", {})["runtime_comparable"] = False
    return config


def command_for(config, identity, config_path):
    lbi = config["lbi"]
    data = config["data"]
    return [
        "python", str((PROJECT_ROOT / "train.py").resolve()), "--config", str(config_path.resolve()),
        "--dataset", "office", "--source", str(data["source"]), "--target", str(data["target"]),
        "--seed", str(FORMAL_SEED), "--gpu-id", "0", "--variant", "conv_out_lbi",
        "--group-mode", "out_channel", "--requested-budget", str(config["requested_budget"]),
        "--lbi-alpha", str(lbi["alpha"]), "--lbi-kappa", str(lbi["kappa"]),
        "--lbi-nu", str(lbi["nu"]), "--lbi-omega", str(lbi["omega"]),
        "--lbi-stage1-max-steps", str(lbi["stage1_max_steps"]),
        "--lbi-budget-tolerance", str(lbi["budget_tolerance"]),
        "--lbi-stage2-lr", str(lbi["stage2_lr"]), "--lbi-stage2-steps", str(lbi["stage2_steps"]),
        "--lbi-delta-nonzero-tolerance", str(lbi["delta_nonzero_tolerance"]),
        "--lbi-support-threshold", str(lbi["support_threshold"]),
        "--output-root", config["output"]["root"], "--no-save-model",
        "--experiment-key", identity["experiment_key"],
        "--experiment-config-sha256", identity["experiment_config_sha256"],
    ]


def build_entries():
    verify_frozen_inputs()
    base_config = load_yaml(BASE_CONFIG_PATH)
    require(base_config["protocol_revision"] == CONV_PROTOCOL_REVISION, "base config protocol drifted")
    require(base_config["implementation_revision"] == CONV_IMPLEMENTATION_REVISION, "base config implementation drifted")
    entries = []
    for rho, (anchor, alpha, nu, max_groups) in FROZEN_ANCHORS.items():
        for transfer in TRANSFERS:
            name, source, target, source_name, target_name = transfer
            config = raw_config(base_config, transfer, rho, anchor, alpha, nu)
            effective = resolve_effective_config(config, str(PROJECT_ROOT.parent))
            identity = build_experiment_identity(effective)
            config_path = CONFIG_DIR / f"{name}__budget_{rho}__{CELL_ID}.yaml"
            entries.append(
                {
                    "plan_schema_version": 1,
                    "node": "node3",
                    "phase": "R2_out_joint_search",
                    "cell_id": CELL_ID,
                    "implementation_revision": CONV_IMPLEMENTATION_REVISION,
                    "protocol_revision": CONV_PROTOCOL_REVISION,
                    "search_protocol_base_revision": SEARCH_PROTOCOL_BASE_REVISION,
                    "search_protocol_revision": SEARCH_PROTOCOL_REVISION,
                    "efficiency_protocol_revision": EFFICIENCY_PROTOCOL_REVISION,
                    "source_checkpoint_revision": SOURCE_CHECKPOINT_REVISION,
                    "experiment_key": identity["experiment_key"],
                    "experiment_config_sha256": identity["experiment_config_sha256"],
                    "scientific_config": identity["scientific_config"],
                    "method": "shot", "task": "otta", "dataset": "office", "transfer": name,
                    "source": source, "target": target, "source_name": source_name, "target_name": target_name,
                    "seed": FORMAL_SEED, "variant": "conv_out_lbi", "group_mode": "out_channel",
                    "anchor": anchor, "alpha": alpha, "nu": nu, "omega": OMEGA, "stage2_lr": STAGE2_LR,
                    "requested_budget": rho, "total_group_count": CONV_GROUP_COUNTS["out_channel"],
                    "max_group_count": max_groups, "lbi": frozen_lbi(alpha, nu),
                    "candidate_conv_scalar_count": CONV_CANDIDATE_PARAM_COUNT,
                    "runtime_comparable": False,
                    "effective_overrides": {
                        "dataset": "office", "source": source, "target": target, "seed": FORMAL_SEED,
                        "variant": "conv_out_lbi", "group_mode": "out_channel", "requested_budget": rho,
                        "anchor": anchor, "lbi": frozen_lbi(alpha, nu), "gpu_id": "0", "save_model": False,
                        "workers_per_gpu": 1, "runtime_comparable": False,
                        "output_root": config["output"]["root"],
                    },
                    "config_path": str(config_path.resolve()), "command_args": None,
                    "expected_output_root": experiment_output_root(effective), "raw_config": config,
                }
            )
    validate_entries(entries)
    return entries


def validate_entries(entries):
    require(len(entries) == 18, f"expected exactly 18 rows, got {len(entries)}")
    for field in ("experiment_key", "experiment_config_sha256", "expected_output_root"):
        require(len({entry[field] for entry in entries}) == 18, f"{field} values are not unique")
    expected_transfers = {name: (source, target, source_name, target_name) for name, source, target, source_name, target_name in TRANSFERS}
    require({entry["transfer"] for entry in entries} == set(expected_transfers), "six-transfer set drifted")
    require({(entry["source"], entry["target"]) for entry in entries} == {(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)}, "source/target mapping drifted")
    require(not any(entry["omega"] == 0.00625 and entry["stage2_lr"] == 0.005 for entry in entries), "central cell must not be rerun")
    require(not any(entry["requested_budget"] == 0.005 for entry in entries), "forbidden .005 budget found")
    for rho, (anchor, alpha, nu, max_groups) in FROZEN_ANCHORS.items():
        subset = [entry for entry in entries if entry["requested_budget"] == rho]
        require(len(subset) == 6, f"{rho} must contain six transfers")
        for entry in subset:
            require((entry["anchor"], entry["alpha"], entry["nu"], entry["max_group_count"]) == (anchor, alpha, nu, max_groups), f"frozen anchor/K_G drifted for {rho}")
    for entry in entries:
        require((entry["source"], entry["target"], entry["source_name"], entry["target_name"]) == expected_transfers[entry["transfer"]], "transfer-name mapping drifted")
        require((entry["variant"], entry["group_mode"], entry["seed"]) == ("conv_out_lbi", "out_channel", 2026), "method/group/seed drifted")
        require((entry["omega"], entry["stage2_lr"]) == (OMEGA, STAGE2_LR), "assigned R2 cell drifted")
        require(entry["lbi"] == frozen_lbi(entry["alpha"], entry["nu"]), "LBI tuple drifted")
        require(entry["scientific_config"]["lbi"]["omega"] == OMEGA and entry["scientific_config"]["lbi"]["stage2_lr"] == STAGE2_LR, "R2 coordinates absent from scientific identity")
        require("filter" not in entry["variant"] and "baseline" not in entry["variant"] and entry["dataset"] != "VISDA-C", "prohibited row found")
        try:
            Path(entry["expected_output_root"]).resolve().relative_to(BRANCH_RUNS_ROOT.resolve())
        except ValueError as error:
            raise ValueError("expected output root escapes assigned cell root") from error


def verify_saved_plan():
    require(PLAN_PATH.is_file(), f"missing plan: {PLAN_PATH}")
    entries = [json.loads(line) for line in PLAN_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = build_entries()
    for entry in expected:
        config_path = Path(entry["config_path"])
        require(config_path.is_file(), f"missing row config: {entry['experiment_key']}")
        entry.pop("raw_config")
        entry["command_args"] = command_for(load_yaml(config_path), entry, config_path)
    require(entries == expected, "on-disk plan does not exactly match canonical generated plan")
    for entry in entries:
        require(Path(entry["config_path"]).is_file(), f"missing row config: {entry['experiment_key']}")
    return entries


def write_plan():
    entries = build_entries()
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        config_path = Path(entry["config_path"])
        config_path.write_text(yaml.safe_dump(entry.pop("raw_config"), sort_keys=False), encoding="utf-8")
        entry["command_args"] = command_for(load_yaml(config_path), entry, config_path)
    # Revalidate after replacing the transient raw config with executable arguments.
    validate_entries(entries)
    PLAN_ROOT.mkdir(parents=True, exist_ok=True)
    PLAN_PATH.write_text("".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries), encoding="utf-8")
    commands_path = PLAN_ROOT / f"node3_{CELL_ID}_commands.sh"
    commands_path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n" + "".join(
            shlex.join(["conda", "run", "--no-capture-output", "-n", "SHOT_TTA"] + entry["command_args"]) + "\n"
            for entry in entries
        ),
        encoding="utf-8",
    )
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    entries = verify_saved_plan() if args.verify_existing else write_plan()
    print(json.dumps({
        "valid": True,
        "experiment_count": len(entries),
        "unique_experiment_keys": len({entry["experiment_key"] for entry in entries}),
        "unique_scientific_sha256": len({entry["experiment_config_sha256"] for entry in entries}),
        "unique_output_roots": len({entry["expected_output_root"] for entry in entries}),
        "plan_sha256": hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest(),
        "cell_id": CELL_ID,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
