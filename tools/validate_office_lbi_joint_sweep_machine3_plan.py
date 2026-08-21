#!/usr/bin/env python3
"""Pure JSON/path validation for the machine3 Office joint-sweep plan.

This helper intentionally imports only Python standard-library modules.  It
does not import CUDA, torch, datasets, model code, or the launcher.
"""

import argparse
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(
    "/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/"
    "260817_iclr2027-refined"
).resolve()
RUNS_ROOT = (
    PROJECT_ROOT
    / "experiment_logs/shot_otta_office_lbi_joint_sweep_20260818/runs"
).resolve()
LEGACY_RUNS_PREFIX = "/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/experiment_logs/"
TRANSFERS = {(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)}
ALPHAS = {0.15, 0.20}
OMEGAS = {0.05, 0.10, 0.20, 0.30}
STAGE2_LRS = {0.005, 0.010, 0.020}
REFERENCE = (0.20, 0.020)


def _same_float(left, right):
    return abs(float(left) - float(right)) < 1.0e-12


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(plan_path, expected_sha256=None):
    plan_path = Path(plan_path).resolve()
    raw = plan_path.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None:
        _require(actual_sha256 == expected_sha256, "plan SHA256 mismatch")
    plan = json.loads(raw.decode("utf-8"))
    entries = plan.get("experiments")
    _require(isinstance(entries, list), "plan.experiments is not a list")
    _require(plan.get("experiment_count") == 132, "plan count is not 132")
    _require(len(entries) == 132, "plan has fewer/more than 132 entries")

    keys = [entry.get("experiment_key") for entry in entries]
    _require(all(keys), "plan contains an entry without experiment_key")
    _require(len(set(keys)) == 132, "plan experiment_key values are not unique")

    pair_counts = {alpha: 0 for alpha in ALPHAS}
    combinations = {alpha: set() for alpha in ALPHAS}
    for index, entry in enumerate(entries):
        prefix = f"entry {index}"
        _require(entry.get("dataset") == "office", f"{prefix}: dataset is not office")
        _require(entry.get("seed") == 2026, f"{prefix}: seed is not 2026")
        _require(entry.get("variant") == "module_lbi", f"{prefix}: variant is not module_lbi")
        _require(_same_float(entry.get("requested_budget"), 0.002), f"{prefix}: budget mismatch")
        source, target = entry.get("source"), entry.get("target")
        _require((source, target) in TRANSFERS, f"{prefix}: invalid Office transfer")
        alpha, kappa, nu = entry.get("alpha"), entry.get("kappa"), entry.get("nu")
        _require(any(_same_float(alpha, value) for value in ALPHAS), f"{prefix}: invalid alpha")
        _require(_same_float(kappa, 1.0), f"{prefix}: kappa mismatch")
        _require(_same_float(nu, 0.50), f"{prefix}: nu mismatch")
        omega, stage2_lr = entry.get("omega"), entry.get("stage2_lr")
        _require(any(_same_float(omega, value) for value in OMEGAS), f"{prefix}: invalid omega")
        _require(any(_same_float(stage2_lr, value) for value in STAGE2_LRS), f"{prefix}: invalid stage2_lr")
        _require(not (_same_float(omega, REFERENCE[0]) and _same_float(stage2_lr, REFERENCE[1])), f"{prefix}: reused reference was scheduled")
        _require(entry.get("stage1_max_steps") == 3000, f"{prefix}: stage1_max_steps mismatch")
        _require(_same_float(entry.get("budget_tolerance"), 1.0e-4), f"{prefix}: budget_tolerance mismatch")
        _require(entry.get("stage2_steps_requested") == 1, f"{prefix}: stage2_steps mismatch")
        _require(_same_float(entry.get("delta_nonzero_tolerance"), 1.0e-12), f"{prefix}: delta tolerance mismatch")
        scientific = entry.get("scientific_config", {})
        _require(scientific.get("model", {}).get("backbone") == "resnet50", f"{prefix}: backbone mismatch")
        _require(scientific.get("data", {}).get("batch_size") == 64, f"{prefix}: batch size mismatch")
        _require(scientific.get("data", {}).get("workers") == 4, f"{prefix}: worker count mismatch")
        _require(scientific.get("lbi", {}).get("support_threshold") == 1.0e-4, f"{prefix}: support threshold mismatch")

        expected_root = Path(entry.get("expected_output_root", "")).resolve()
        _require(expected_root == RUNS_ROOT or RUNS_ROOT in expected_root.parents, f"{prefix}: expected_output_root outside canonical RUNS_ROOT")
        _require(PROJECT_ROOT == expected_root or PROJECT_ROOT in expected_root.parents, f"{prefix}: expected_output_root outside refined project")
        _require(not str(expected_root).startswith(LEGACY_RUNS_PREFIX), f"{prefix}: legacy relative-root bug path detected")

        command = entry.get("command_args", [])
        _require("--output-root" in command, f"{prefix}: command lacks --output-root")
        output_positions = [i for i, value in enumerate(command) if value == "--output-root"]
        _require(len(output_positions) == 1 and output_positions[0] + 1 < len(command), f"{prefix}: malformed --output-root")
        _require(command[output_positions[0] + 1] == str(RUNS_ROOT), f"{prefix}: command output root mismatch")

        alpha_key = next(value for value in ALPHAS if _same_float(alpha, value))
        pair_counts[alpha_key] += 1
        combinations[alpha_key].add((round(float(omega), 12), round(float(stage2_lr), 12)))

    for alpha in sorted(ALPHAS):
        _require(pair_counts[alpha] == 66, f"alpha={alpha}: expected 66 entries")
        _require(len(combinations[alpha]) == 11, f"alpha={alpha}: expected 11 new omega/LR combinations")
    _require(sum(pair_counts.values()) == 132, "pair counts do not sum to 132")
    return {
        "plan_path": str(plan_path),
        "plan_sha256": actual_sha256,
        "experiment_count": len(entries),
        "unique_experiment_keys": len(set(keys)),
        "pair_counts": {str(alpha): pair_counts[alpha] for alpha in sorted(ALPHAS)},
        "combinations_per_pair": {str(alpha): len(combinations[alpha]) for alpha in sorted(ALPHAS)},
        "canonical_runs_root": str(RUNS_ROOT),
        "static_path_check_passed": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan")
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    print(json.dumps(validate(args.plan, args.expected_sha256), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
