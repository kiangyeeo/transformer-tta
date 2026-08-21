#!/usr/bin/env python3
"""Pure JSON/path validator for the machine4 Office LBI joint-sweep plan."""

import argparse
import hashlib
import json
from pathlib import Path


OFFICE_TRANSFERS = {(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)}
ALLOWED_OMEGAS = {0.05, 0.10, 0.20, 0.30}
ALLOWED_LRS = {0.005, 0.010, 0.020}
EXCLUDED_REFERENCE = (0.20, 0.020)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message):
    raise SystemExit(f"PLAN VALIDATION FAILED: {message}")


def validate(plan_path, runs_root, expected_sha=None):
    plan_path = Path(plan_path).resolve()
    runs_root = Path(runs_root).resolve()
    if expected_sha and sha256(plan_path) != expected_sha:
        fail("plan SHA256 does not match PREPARE_READY")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    entries = plan.get("experiments")
    if plan.get("experiment_count") != 132 or not isinstance(entries, list) or len(entries) != 132:
        fail("experiment count is not exactly 132")
    keys = [entry.get("experiment_key") for entry in entries]
    hashes = [entry.get("experiment_config_sha256") for entry in entries]
    if len(set(keys)) != 132:
        fail("experiment keys are not unique")
    if len(set(hashes)) != 132:
        fail("experiment config hashes are not unique")
    pair_counts = {}
    for entry in entries:
        if (entry.get("dataset"), entry.get("seed"), entry.get("variant")) != ("office", 2026, "module_lbi"):
            fail("non-Office, non-seed-2026, or non-module_lbi entry")
        pair = (float(entry.get("requested_budget")), float(entry.get("alpha")), float(entry.get("kappa")), float(entry.get("nu")))
        if pair not in {(0.0005, 0.10, 1.5, 0.50), (0.001, 0.15, 1.0, 1.00)}:
            fail(f"unexpected owned pair: {pair}")
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        if (entry.get("source"), entry.get("target")) not in OFFICE_TRANSFERS:
            fail("unexpected Office transfer")
        omega, lr = float(entry.get("omega")), float(entry.get("stage2_lr"))
        if omega not in ALLOWED_OMEGAS or lr not in ALLOWED_LRS:
            fail("omega or stage2_lr outside frozen sweep")
        if (omega, lr) == EXCLUDED_REFERENCE:
            fail("reused reference point appears in new-run plan")
        if entry.get("lbi_resolved") is not True:
            fail("LBI tuple is not resolved")
        if entry.get("expected_output_root") is None:
            fail("missing expected_output_root")
        expected = Path(entry["expected_output_root"]).resolve()
        try:
            expected.relative_to(runs_root)
        except ValueError:
            fail(f"expected_output_root escapes canonical RUNS_ROOT: {expected}")
        if "/PJ/Split-LBI/experiment_logs/" in str(expected):
            fail("expected_output_root is under the old relative experiment_logs path")
        command = entry.get("command_args") or []
        if "--output-root" not in command or command[command.index("--output-root") + 1] != str(runs_root):
            fail("command --output-root disagrees with canonical RUNS_ROOT")
        if "--experiment-key" not in command or command[command.index("--experiment-key") + 1] != entry["experiment_key"]:
            fail("command experiment key disagrees with plan metadata")
        if "--experiment-config-sha256" not in command or command[command.index("--experiment-config-sha256") + 1] != entry["experiment_config_sha256"]:
            fail("command experiment hash disagrees with plan metadata")
    if pair_counts != {(0.0005, 0.10, 1.5, 0.50): 66, (0.001, 0.15, 1.0, 1.00): 66}:
        fail(f"owned-pair counts are wrong: {pair_counts}")
    combos = {}
    for entry in entries:
        pair = (float(entry["requested_budget"]), float(entry["alpha"]), float(entry["kappa"]), float(entry["nu"]))
        combos.setdefault(pair, set()).add((float(entry["omega"]), float(entry["stage2_lr"])))
    expected_combos = {(omega, lr) for omega in ALLOWED_OMEGAS for lr in ALLOWED_LRS if (omega, lr) != EXCLUDED_REFERENCE}
    if any(values != expected_combos for values in combos.values()):
        fail("omega x stage2_lr combinations are not exactly the 11-point grid")
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--expected-sha")
    args = parser.parse_args()
    plan = validate(args.plan, args.runs_root, args.expected_sha)
    print(json.dumps({"valid": True, "experiment_count": len(plan["experiments"]), "unique_experiment_keys": len({e["experiment_key"] for e in plan["experiments"]})}, sort_keys=True))


if __name__ == "__main__":
    main()
