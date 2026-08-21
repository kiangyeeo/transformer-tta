#!/usr/bin/env python3
"""Pure JSON/path guard for the prepared Office LBI joint plan."""

import argparse
import hashlib
import json
import os
from pathlib import Path


OMEGAS = {0.05, 0.10, 0.20, 0.30}
STAGE2_LRS = {0.005, 0.010, 0.020}
REFERENCE = (0.20, 0.020)
PAIR_KEYS = {
    (0.0005, 0.10, 1.0, 0.25): "A6",
    (0.0005, 0.10, 1.0, 0.50): "A1",
}
TRANSFERS = {(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)}
CANONICAL_RUNS_ROOT = Path(
    "/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/"
    "260817_iclr2027-refined/experiment_logs/"
    "shot_otta_office_lbi_joint_sweep_20260818/runs"
).resolve()


def _same(left, right):
    return abs(float(left) - float(right)) < 1.0e-12


def _under(path, root):
    try:
        Path(path).resolve().relative_to(root)
        return True
    except ValueError:
        return False


def validate(plan_path, phase_record=None, expected_sha256=None):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    entries = plan.get("experiments", [])
    assert plan.get("experiment_count") == 132
    assert len(entries) == 132
    keys = [entry.get("experiment_key") for entry in entries]
    hashes = [entry.get("experiment_config_sha256") for entry in entries]
    assert len(set(keys)) == 132
    assert len(set(hashes)) == 132
    assert plan.get("method") == "shot"
    assert plan.get("task") == "otta"
    assert plan.get("protocol_revision") == "OTTA_FC_LBI_PROTOCOL_20260817_v1"
    assert plan.get("implementation_revision") == "iclr2027_refined_20260817_v1"
    assert plan.get("source_checkpoint_revision") == "nips2026_shot_otta_uda_source_v1"

    pair_counts = {name: 0 for name in PAIR_KEYS.values()}
    seen_points = {name: set() for name in PAIR_KEYS.values()}
    seen_transfers = {name: set() for name in PAIR_KEYS.values()}
    for entry in entries:
        assert entry.get("dataset") == "office"
        assert entry.get("variant") == "module_lbi"
        assert entry.get("seed") == 2026
        assert _same(entry.get("requested_budget"), 0.0005)
        assert (int(entry.get("source")), int(entry.get("target"))) in TRANSFERS
        assert entry.get("stage1_max_steps") == 3000
        assert _same(entry.get("budget_tolerance"), 1.0e-4)
        assert entry.get("stage2_steps_requested") == 1
        assert _same(entry.get("delta_nonzero_tolerance"), 1.0e-12)
        assert _same(entry.get("effective_overrides", {}).get("lbi", {}).get("support_threshold"), 1.0e-4)
        point = (
            float(entry.get("omega")),
            float(entry.get("stage2_lr")),
        )
        assert point[0] in OMEGAS
        assert point[1] in STAGE2_LRS
        assert point != REFERENCE
        pair = (
            float(entry.get("requested_budget")),
            float(entry.get("alpha")),
            float(entry.get("kappa")),
            float(entry.get("nu")),
        )
        assert pair in PAIR_KEYS
        name = PAIR_KEYS[pair]
        pair_counts[name] += 1
        seen_points[name].add(point)
        seen_transfers[name].add((int(entry["source"]), int(entry["target"])))
        expected = Path(entry.get("expected_output_root", "")).resolve()
        assert _under(expected, CANONICAL_RUNS_ROOT)
        assert str(expected).startswith(
            "/inspire/hdd/global_user/gaoyachen-253308310317/PJ/"
            "Split-LBI/260817_iclr2027-refined/"
        )
        assert "/PJ/Split-LBI/experiment_logs/" not in str(expected)
        command = entry.get("command_args", [])
        assert command
        assert "--output-root" in command
        assert "--batch-size" in command and command[command.index("--batch-size") + 1] == "64"
        assert "--workers" in command and command[command.index("--workers") + 1] == "4"
        assert "--no-save-model" in command
        output_root = Path(command[command.index("--output-root") + 1]).resolve()
        assert output_root == CANONICAL_RUNS_ROOT
        assert entry.get("effective_overrides", {}).get("output_root") == str(CANONICAL_RUNS_ROOT)

    assert pair_counts == {"A6": 66, "A1": 66}
    assert all(len(points) == 11 for points in seen_points.values())
    assert all(transfers == TRANSFERS for transfers in seen_transfers.values())
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    if expected_sha256 is not None:
        assert digest == expected_sha256
    if phase_record is not None:
        record = json.loads(Path(phase_record).read_text(encoding="utf-8"))
        assert record.get("plan_sha256") == digest
    return {
        "experiment_count": len(entries),
        "unique_experiment_keys": len(set(keys)),
        "pair_counts": pair_counts,
        "canonical_runs_root": str(CANONICAL_RUNS_ROOT),
        "plan_sha256": digest,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan")
    parser.add_argument("--phase-record")
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    print(json.dumps(validate(args.plan, args.phase_record, args.expected_sha256), sort_keys=True))


if __name__ == "__main__":
    main()
