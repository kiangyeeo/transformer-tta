#!/usr/bin/env python3
"""CPU-only checks for the frozen implementation identity."""

import copy
import hashlib
import json
import os
import os.path as osp
import sys
import tempfile


PROJECT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
WORKSPACE_ROOT = osp.dirname(PROJECT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from experiment_identity import (  # noqa: E402
    IMPLEMENTATION_REVISION,
    build_experiment_identity,
    canonical_scientific_json,
)
from shot_otta.config import load_yaml, resolve_effective_config  # noqa: E402
from tools.check_experiment_status import check_status  # noqa: E402


BASE_CONFIG_PATH = osp.join(PROJECT_DIR, "configs", "otta_fc_lbi_protocol_20260817_v1.yaml")
VARIANTS = (
    "source_only",
    "full_dense",
    "module_dense",
    "module_random",
    "module_magnitude",
    "module_saliency",
    "module_lbi",
)


def _effective(variant):
    config = load_yaml(BASE_CONFIG_PATH)
    config["variant"] = variant
    config["seed"] = 2026
    config["selection_seed"] = 2026
    config["requested_budget"] = 0.001
    if variant == "module_lbi":
        config["lbi"] = {
            "alpha": 0.1, "kappa": 1.0, "nu": 1.0, "omega": 0.1,
            "stage1_max_steps": 3000, "budget_tolerance": 0.0001,
            "stage2_lr": 0.01, "stage2_steps": 1,
            "delta_nonzero_tolerance": 1.0e-12,
            "support_threshold": 1.0e-4,
        }
    return resolve_effective_config(config, WORKSPACE_ROOT)


def _sha256(scientific_config):
    return hashlib.sha256(
        canonical_scientific_json(scientific_config).encode("utf-8")
    ).hexdigest()


def _completed_summary(path, key, sha):
    os.makedirs(path, exist_ok=True)
    with open(osp.join(path, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "completed",
                "implementation_revision": IMPLEMENTATION_REVISION,
                "experiment_key": key,
                "experiment_config_sha256": sha,
            },
            handle,
        )


def main():
    effective_by_variant = {
        variant: _effective(variant) for variant in VARIANTS
    }

    # A/B: every variant uses the one revision in its scientific config.
    identities = {}
    for variant, config in effective_by_variant.items():
        identity = build_experiment_identity(config)
        identities[variant] = identity
        assert (
            identity["scientific_config"]["implementation_revision"]
            == IMPLEMENTATION_REVISION
        )
        assert config["implementation_revision"] == IMPLEMENTATION_REVISION
    assert identities["source_only"]["scientific_config"][
        "implementation_revision"
    ] == IMPLEMENTATION_REVISION
    assert identities["full_dense"]["scientific_config"][
        "implementation_revision"
    ] == IMPLEMENTATION_REVISION

    # C: only changing the implementation revision changes the scientific SHA.
    current = identities["source_only"]
    old_scientific = copy.deepcopy(current["scientific_config"])
    old_scientific["implementation_revision"] = "iclr2027_refined_20260816_v0"
    assert _sha256(old_scientific) != current["experiment_config_sha256"]

    # D/E: status matching is still hash-based; old completed results are not
    # current completed results, while an exact current identity is skipped.
    old_sha = _sha256(old_scientific)
    old_key = current["experiment_key"].rsplit("__", 1)[0] + "__" + old_sha[:12]
    plan = {
        "experiments": [
            {
                "implementation_revision": IMPLEMENTATION_REVISION,
                "experiment_key": current["experiment_key"],
                "experiment_config_sha256": current[
                    "experiment_config_sha256"
                ],
            }
        ]
    }
    with tempfile.TemporaryDirectory(prefix="iclr2027_revision_smoke_") as root:
        old_runs = osp.join(root, "old_runs")
        _completed_summary(old_runs, old_key, old_sha)
        old_status = check_status(plan, old_runs)
        assert old_status["experiments"][0]["plan_status"] == "missing"

        current_runs = osp.join(root, "current_runs")
        _completed_summary(
            current_runs,
            current["experiment_key"],
            current["experiment_config_sha256"],
        )
        current_status = check_status(plan, current_runs)
        assert (
            current_status["experiments"][0]["plan_status"]
            == "completed"
        )

    print("implementation revision identity smoke test passed")


if __name__ == "__main__":
    main()
