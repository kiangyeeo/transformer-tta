#!/usr/bin/env python3
"""CPU-only Node 6 formal-plan preflight validation."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from protocol_constants import CONV_IMPLEMENTATION_REVISION, CONV_PROTOCOL_REVISION, SOURCE_CHECKPOINT_REVISION


BASELINE_PROTOCOL = "OTTA_CONV_BASELINE_FORMAL_20260827_v1"
RHOS = {0.0005, 0.001, 0.002}
EXPECTED_K = {0.0005: 4, 0.001: 9, 0.002: 18}
SPARSE_VARIANTS = {"conv_out_random", "conv_out_magnitude", "conv_out_saliency"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan")
    args = parser.parse_args()
    with open(args.plan, "r", encoding="utf-8") as handle:
        plan = json.load(handle)
    entries = plan.get("experiments", [])
    require(plan.get("experiment_count") == 10 and len(entries) == 10, "Node 6 must contain exactly 10 scientific conditions")
    require(plan.get("formal_baseline_protocol_revision") == BASELINE_PROTOCOL, "Node 6 formal baseline protocol revision drifted")
    require(plan.get("protocol_revision") == CONV_PROTOCOL_REVISION, "Node 6 Conv protocol revision drifted")
    require(plan.get("implementation_revision") == CONV_IMPLEMENTATION_REVISION, "Node 6 Conv implementation revision drifted")
    for field in ("experiment_key", "experiment_config_sha256", "expected_output_root"):
        require(len({entry.get(field) for entry in entries}) == 10, f"Node 6 {field} values are not unique")
    observed = {(entry.get("variant"), entry.get("requested_budget")) for entry in entries}
    expected = {("conv_module_dense", 1.0)} | {(variant, rho) for variant in SPARSE_VARIANTS for rho in RHOS}
    require(observed == expected, "Node 6 condition grid drifted")
    for entry in entries:
        variant = entry.get("variant")
        require("filter" not in str(variant) and "lbi" not in str(variant), "Node 6 contains a prohibited filter or Conv-LBI variant")
        require(entry.get("dataset") == "VISDA-C" and (entry.get("source"), entry.get("target"), entry.get("seed")) == (0, 1, 2026), "Node 6 dataset/transfer/seed drifted")
        require(entry.get("formal_baseline_protocol_revision") == BASELINE_PROTOCOL, "Node 6 experiment formal baseline provenance is missing")
        require(entry.get("protocol_revision") == CONV_PROTOCOL_REVISION and entry.get("implementation_revision") == CONV_IMPLEMENTATION_REVISION and entry.get("protocol_track") == "conv", "Node 6 experiment Conv provenance drifted")
        require(entry.get("source_checkpoint_revision") == SOURCE_CHECKPOINT_REVISION, "Node 6 source checkpoint provenance drifted")
        scientific = entry.get("scientific_config")
        require(isinstance(scientific, dict) and scientific.get("protocol_track") == "conv" and scientific.get("protocol_revision") == CONV_PROTOCOL_REVISION and scientific.get("implementation_revision") == CONV_IMPLEMENTATION_REVISION, "Node 6 canonical scientific_config provenance drifted")
        if variant == "conv_module_dense":
            require(entry.get("requested_budget") == 1.0 and entry.get("group_mode") is None, "Node 6 dense anchor drifted")
            continue
        rho = entry.get("requested_budget")
        require(rho in RHOS and entry.get("group_mode") == "out_channel" and entry.get("total_group_count") == 9216 and entry.get("max_group_count") == EXPECTED_K[rho], "Node 6 sparse budget/K_G drifted")
        if variant == "conv_out_random":
            require(entry.get("selection_seed") == 2026 and entry.get("num_random_masks") == 3 and entry.get("child_mask_seeds") == [202600, 202601, 202602], "Node 6 Random three-mask semantics drifted")
    print(json.dumps({"valid": True, "experiment_count": 10, "random_parent_count": 3}))


if __name__ == "__main__":
    main()
