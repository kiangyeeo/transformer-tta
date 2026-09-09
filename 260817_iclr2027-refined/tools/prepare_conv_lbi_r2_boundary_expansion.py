#!/usr/bin/env python3
"""Build and CPU-verify the one-time Office out-channel R2 expansion plans."""

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


SEARCH_PROTOCOL_V1 = "OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v1"
SEARCH_PROTOCOL_V2 = "OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2"
BASE_CONFIG_PATH = PROJECT_ROOT / "configs/otta_conv_lbi_protocol_20260826_v1.yaml"
EXPANSION_ROOT = (
    PROJECT_ROOT
    / "experiment_logs/office_conv_lbi_r2_out_boundary_expansion_seed2026_20260828"
)
PLAN_ROOT = EXPANSION_ROOT / "plans"
R2_ROOT = PROJECT_ROOT / "experiment_logs/office_conv_lbi_r2_out_joint_sweep_seed2026_20260828"
R2_FINALIZE = R2_ROOT / "phase_records/R2/FINALIZE.md"
R2_SELECTION = R2_ROOT / "selected_configs/OFFICE_OUT_R2_SELECTION.json"
BOUNDARY_REQUEST = R2_ROOT / "selected_configs/R2_BOUNDARY_EXPANSION_REQUEST.json"
R1_FROZEN = (
    PROJECT_ROOT
    / "experiment_logs/office_conv_lbi_r1_out_stage1_validation_seed2026_20260827"
    / "selected_configs/OFFICE_OUT_STAGE1_FROZEN.json"
)
BASELINE_ROOT = PROJECT_ROOT / "experiment_logs/conv_baseline_formal_global_20260827"

TRANSFERS = (
    ("AD", 0, 1, "amazon", "dslr"),
    ("AW", 0, 2, "amazon", "webcam"),
    ("DA", 1, 0, "dslr", "amazon"),
    ("DW", 1, 2, "dslr", "webcam"),
    ("WA", 2, 0, "webcam", "amazon"),
    ("WD", 2, 1, "webcam", "dslr"),
)

CELLS = (
    {
        "id": "E0",
        "plan_name": "E0_p0005_omega_0p30_lr_0p010.jsonl",
        "runs_name": "E0_p0005_omega_0p30_lr_0p010",
        "gpu": "0",
        "rho": 0.0005,
        "K_G": 4,
        "anchor": "A1",
        "alpha": 0.05,
        "nu": 0.50,
        "omega": 0.30,
        "stage2_lr": 0.010,
    },
    {
        "id": "E1",
        "plan_name": "E1_p0005_omega_0p10_lr_0p020.jsonl",
        "runs_name": "E1_p0005_omega_0p10_lr_0p020",
        "gpu": "1",
        "rho": 0.0005,
        "K_G": 4,
        "anchor": "A1",
        "alpha": 0.05,
        "nu": 0.50,
        "omega": 0.10,
        "stage2_lr": 0.020,
    },
    {
        "id": "E2",
        "plan_name": "E2_p0005_omega_0p30_lr_0p020.jsonl",
        "runs_name": "E2_p0005_omega_0p30_lr_0p020",
        "gpu": "2",
        "rho": 0.0005,
        "K_G": 4,
        "anchor": "A1",
        "alpha": 0.05,
        "nu": 0.50,
        "omega": 0.30,
        "stage2_lr": 0.020,
    },
    {
        "id": "E3",
        "plan_name": "E3_p001_omega_0p30_lr_0p0025.jsonl",
        "runs_name": "E3_p001_omega_0p30_lr_0p0025",
        "gpu": "3",
        "rho": 0.001,
        "K_G": 9,
        "anchor": "A1",
        "alpha": 0.05,
        "nu": 0.50,
        "omega": 0.30,
        "stage2_lr": 0.0025,
    },
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    require(path.is_file(), f"missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def lbi(cell):
    return {
        "alpha": cell["alpha"],
        "kappa": 1.0,
        "nu": cell["nu"],
        "omega": cell["omega"],
        "stage1_max_steps": 3000,
        "budget_tolerance": 1.0e-4,
        "stage2_lr": cell["stage2_lr"],
        "stage2_steps": 1,
        "delta_nonzero_tolerance": 1.0e-12,
        "support_threshold": 1.0e-4,
    }


def verify_frozen_inputs():
    protocols = {
        "OTTA_CONV_LBI_PROTOCOL_20260826_v1.md": "# OTTA_CONV_LBI_PROTOCOL_20260826_v1",
        "OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v1.md": "# OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v1",
        "OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2.md": "# OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2",
    }
    for name, heading in protocols.items():
        path = PROJECT_ROOT / "protocol/shot-otta_conv" / name
        require(path.is_file() and path.read_text(encoding="utf-8").startswith(heading), f"protocol drifted: {path}")
    finalize = R2_FINALIZE.read_text(encoding="utf-8")
    require("R2_INITIAL_GRID_COMPLETE: YES" in finalize, "initial R2 grid is not complete")
    require("NEW_ROWS: 144/144" in finalize, "initial R2 new-row count drifted")
    require("REUSED_ROWS: 18/18" in finalize, "initial R2 reused-row count drifted")
    selection = read_json(R2_SELECTION)
    expected_stage1 = {
        "0.0005": (4, "A1", 0.05, 1.0, 0.50),
        "0.001": (9, "A1", 0.05, 1.0, 0.50),
        "0.002": (18, "A4", 0.10, 1.0, 1.00),
    }
    for rho, expected in expected_stage1.items():
        item = selection["frozen_stage1"].get(rho, {})
        observed = (item.get("K_G"), item.get("anchor"), item.get("alpha"), item.get("kappa"), item.get("nu"))
        require(observed == expected, f"R2 frozen Stage-1 drifted for rho={rho}")
    expected_winners = {
        "0.0005": (0.1, 0.01, ["omega", "stage2_lr"], "EXPANSION_REQUIRED"),
        "0.001": (0.1, 0.0025, ["omega"], "EXPANSION_REQUIRED"),
        "0.002": (0.025, 0.0025, [], "FROZEN_NO_EXPANSION"),
    }
    for rho, expected in expected_winners.items():
        item = selection["budgets"].get(rho, {})
        winner = item.get("initial_grid_winner", item.get("selected_tuple", {}))
        observed = (winner.get("omega"), winner.get("stage2_lr"), item.get("upper_boundary_dimensions"), item.get("status"))
        require(observed == expected, f"R2 winner/boundary state drifted for rho={rho}")
    request = read_json(BOUNDARY_REQUEST)
    require(request.get("status") == "EXPANSION_REQUIRED", "boundary request is not active")
    require(request.get("permitted_upper_extensions") == {"omega": 0.3, "stage2_lr": 0.02}, "permitted upper extensions drifted")
    requested = {str(item["requested_budget"]): item for item in request.get("budgets", [])}
    require(set(requested) == {"0.0005", "0.001"}, "boundary request must contain only .0005 and .001")
    require(requested["0.0005"]["triggered_upper_boundary_dimensions"] == ["omega", "stage2_lr"], "unexpected .0005 trigger")
    require(requested["0.001"]["triggered_upper_boundary_dimensions"] == ["omega"], "unexpected .001 trigger")
    frozen = read_json(R1_FROZEN)
    observed = {
        str(cell["requested_budget"]): (
            cell["K_G"], cell["primary"]["anchor"], cell["primary"]["alpha"], cell["primary"]["nu"]
        )
        for cell in frozen.get("cells", [])
    }
    require(observed == {"0.0005": (4, "A1", 0.05, 0.5), "0.001": (9, "A1", 0.05, 0.5), "0.002": (18, "A4", 0.1, 1.0)}, "R1 frozen anchor drifted")
    require(DATASETS["office"]["domains"] == ["amazon", "dslr", "webcam"], "Office domain mapping drifted")
    for relative in ("FINALIZE.md", "reports/OFFICE_ALL_CONDITIONS.csv", "reports/OFFICE_MACRO_PU_FO.csv"):
        path = BASELINE_ROOT / relative
        require(path.is_file(), f"missing frozen baseline reference: {path}")
    require(CONV_IMPLEMENTATION_REVISION == "iclr2027_refined_conv_20260826_v1", "implementation revision drifted")
    require(CONV_GROUP_COUNTS["out_channel"] == 9216, "out-channel group count drifted")


def raw_config(base_config, cell, transfer):
    transfer_name, source, target, _, _ = transfer
    config = copy.deepcopy(base_config)
    config.update({
        "protocol_track": "conv",
        "formal_protocol": True,
        "protocol_revision": CONV_PROTOCOL_REVISION,
        "implementation_revision": CONV_IMPLEMENTATION_REVISION,
        "search_protocol_revision": SEARCH_PROTOCOL_V2,
        "search_protocol_base_revision": SEARCH_PROTOCOL_V1,
        "source_checkpoint_revision": SOURCE_CHECKPOINT_REVISION,
        "seed": FORMAL_SEED,
        "variant": "conv_out_lbi",
        "group_mode": "out_channel",
        "requested_budget": cell["rho"],
        "lbi": lbi(cell),
        "allow_unresolved_lbi": False,
    })
    config["data"].update({"dataset": "office", "source": source, "target": target})
    config["device"]["gpu_id"] = "0"
    config["output"].update({"root": str((EXPANSION_ROOT / "runs" / cell["runs_name"]).resolve()), "save_model": False, "run_name": None})
    config.setdefault("runtime", {})["runtime_comparable"] = False
    domains = DATASETS["office"]["domains"]
    require(transfer_name == f"{domains[source][0].upper()}{domains[target][0].upper()}", "transfer mapping drifted")
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


def build_cell_entries(cell):
    base_config = load_yaml(BASE_CONFIG_PATH)
    require(base_config["protocol_revision"] == CONV_PROTOCOL_REVISION, "base config protocol drifted")
    require(base_config["implementation_revision"] == CONV_IMPLEMENTATION_REVISION, "base config implementation drifted")
    require(max_support_count(cell["rho"], CONV_GROUP_COUNTS["out_channel"]) == cell["K_G"], f"rho/K_G drifted for {cell['id']}")
    rows = []
    for transfer in TRANSFERS:
        name, source, target, source_name, target_name = transfer
        config = raw_config(base_config, cell, transfer)
        effective = resolve_effective_config(config, str(WORKSPACE_ROOT))
        identity = build_experiment_identity(effective)
        rows.append({
            "plan_schema_version": 1,
            "phase": "R2_boundary_expansion",
            "cell_id": cell["id"],
            "assigned_gpu": cell["gpu"],
            "implementation_revision": CONV_IMPLEMENTATION_REVISION,
            "protocol_revision": CONV_PROTOCOL_REVISION,
            "search_protocol_revision": SEARCH_PROTOCOL_V2,
            "search_protocol_base_revision": SEARCH_PROTOCOL_V1,
            "efficiency_protocol_revision": EFFICIENCY_PROTOCOL_REVISION,
            "source_checkpoint_revision": SOURCE_CHECKPOINT_REVISION,
            "experiment_key": identity["experiment_key"],
            "experiment_config_sha256": identity["experiment_config_sha256"],
            "scientific_config": identity["scientific_config"],
            "method": "shot", "task": "otta", "dataset": "office", "transfer": name,
            "source": source, "target": target, "source_name": source_name, "target_name": target_name,
            "seed": FORMAL_SEED, "variant": "conv_out_lbi", "group_mode": "out_channel",
            "anchor": cell["anchor"], "alpha": cell["alpha"], "kappa": 1.0, "nu": cell["nu"],
            "omega": cell["omega"], "stage2_lr": cell["stage2_lr"], "requested_budget": cell["rho"],
            "total_group_count": CONV_GROUP_COUNTS["out_channel"], "max_group_count": cell["K_G"],
            "lbi": lbi(cell), "candidate_conv_scalar_count": CONV_CANDIDATE_PARAM_COUNT,
            "runtime_comparable": False,
            "effective_overrides": {
                "dataset": "office", "source": source, "target": target, "seed": FORMAL_SEED,
                "variant": "conv_out_lbi", "group_mode": "out_channel", "requested_budget": cell["rho"],
                "anchor": cell["anchor"], "lbi": lbi(cell), "gpu_id": "0", "save_model": False,
                "workers_per_gpu": 1, "runtime_comparable": False, "output_root": config["output"]["root"],
            },
            "config_path": str(BASE_CONFIG_PATH.resolve()), "command_args": command_for(config, identity),
            "expected_output_root": experiment_output_root(effective),
        })
    return rows


def validate_all(cell_entries):
    require(set(cell_entries) == {cell["id"] for cell in CELLS}, "unexpected cell set")
    rows = [row for cell in CELLS for row in cell_entries[cell["id"]]]
    require(len(rows) == 24, f"expected 24 rows, got {len(rows)}")
    for field in ("experiment_key", "experiment_config_sha256", "expected_output_root"):
        require(len({row[field] for row in rows}) == 24, f"{field} values are not globally unique")
    expected = {(cell["rho"], cell["omega"], cell["stage2_lr"], cell["K_G"], cell["anchor"], cell["alpha"], cell["nu"]) for cell in CELLS}
    observed = {(row["requested_budget"], row["omega"], row["stage2_lr"], row["max_group_count"], row["anchor"], row["alpha"], row["nu"]) for row in rows}
    require(observed == expected, "row set is outside the four authorized cells")
    for cell in CELLS:
        rows_for_cell = cell_entries[cell["id"]]
        require(len(rows_for_cell) == 6 and [row["transfer"] for row in rows_for_cell] == [item[0] for item in TRANSFERS], f"{cell['id']} transfer set drifted")
        root = (EXPANSION_ROOT / "runs" / cell["runs_name"]).resolve()
        for row in rows_for_cell:
            require(row["assigned_gpu"] == cell["gpu"], f"GPU assignment drifted for {cell['id']}")
            require((row["variant"], row["group_mode"], row["seed"], row["kappa"]) == ("conv_out_lbi", "out_channel", 2026, 1.0), "frozen scientific invariant drifted")
            require(row["lbi"] == lbi(cell), "LBI tuple drifted")
            scientific_lbi = row["scientific_config"].get("lbi", {})
            require((scientific_lbi.get("omega"), scientific_lbi.get("stage2_lr")) == (cell["omega"], cell["stage2_lr"]), "omega/lr absent from scientific identity")
            require(row["runtime_comparable"] is False and row["effective_overrides"]["runtime_comparable"] is False, "runtime mode drifted")
            try:
                Path(row["expected_output_root"]).resolve().relative_to(root)
            except ValueError as error:
                raise ValueError(f"output root escapes {cell['id']}") from error
    require(sum(row["requested_budget"] == 0.0005 for row in rows) == 18, "wrong .0005 row count")
    require(sum(row["requested_budget"] == 0.001 for row in rows) == 6, "wrong .001 row count")
    require(sum(row["requested_budget"] == 0.002 for row in rows) == 0, "forbidden .002 row present")


def canonical_entries():
    verify_frozen_inputs()
    entries = {cell["id"]: build_cell_entries(cell) for cell in CELLS}
    validate_all(entries)
    return entries


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def plan_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    entries = canonical_entries()
    if len(sys.argv) == 1:
        PLAN_ROOT.mkdir(parents=True, exist_ok=True)
        for cell in CELLS:
            path = PLAN_ROOT / cell["plan_name"]
            path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in entries[cell["id"]]), encoding="utf-8")
    elif len(sys.argv) == 2 and sys.argv[1] == "--verify-existing":
        for cell in CELLS:
            path = PLAN_ROOT / cell["plan_name"]
            require(path.is_file(), f"missing plan: {path}")
            require(load_jsonl(path) == entries[cell["id"]], f"on-disk plan differs from canonical plan: {path}")
        validate_all({cell["id"]: load_jsonl(PLAN_ROOT / cell["plan_name"]) for cell in CELLS})
    else:
        raise SystemExit("usage: prepare_conv_lbi_r2_boundary_expansion.py [--verify-existing]")
    print(json.dumps({
        "valid": True,
        "row_count": 24,
        "unique_experiment_keys": 24,
        "unique_scientific_sha256": 24,
        "unique_expected_output_roots": 24,
        "plan_sha256": {cell["id"]: plan_digest(PLAN_ROOT / cell["plan_name"]) for cell in CELLS},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
