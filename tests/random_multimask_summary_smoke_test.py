#!/usr/bin/env python3
"""Smoke-check multi-mask random parent summaries without training."""

import copy
import json
import os
import os.path as osp
import sys
import tempfile


PROJECT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
WORKSPACE_ROOT = osp.dirname(PROJECT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from shot_otta import trainer  # noqa: E402
from shot_otta.config import load_yaml, resolve_effective_config  # noqa: E402
from tools.summarize_runs import build_summary_outputs  # noqa: E402


def _fake_child_run(config, workspace_root, **kwargs):
    del workspace_root, kwargs
    mask_seed = config["selection_seed"]
    return {
        "PU-Acc": float(mask_seed % 100),
        "FO-Acc": float(mask_seed % 100 + 10),
        "runtime": 1.25,
        "started_at_utc": "2026-08-03T00:00:00+00:00",
        "completed_at_utc": "2026-08-03T00:00:01+00:00",
        "selection": "random",
        "mask_static": True,
        "candidate_scope": "netB.bottleneck",
        "candidate_scope_type": "fc_parameters",
        "ranking_source": "independent_random_generator",
        "mask_refresh_policy": "once_before_adaptation",
        "selected_param_count": 7,
        "candidate_scope_param_count": 10,
        "total_model_param_count": 100,
        "selected_over_scope_ratio": 0.7,
        "selected_over_model_ratio": 0.07,
        "bn_stats_policy": "frozen",
        "bn_stats_frozen": True,
        "bn_module_count": 3,
    }


def main():
    with tempfile.TemporaryDirectory(prefix="iclr2027_random_summary_") as temp_dir:
        config = load_yaml(osp.join(PROJECT_DIR, "configs", "shot_otta.yaml"))
        config.update(
            {
                "variant": "module_random",
                "seed": 2020,
                "selection_seed": 2020,
                "num_random_masks": 3,
                "requested_budget": 0.001,
            }
        )
        config["output"] = {**config["output"], "root": temp_dir}
        config = resolve_effective_config(config, WORKSPACE_ROOT)

        original = trainer._run_single_experiment
        trainer._run_single_experiment = _fake_child_run
        try:
            summary = trainer._run_random_experiment(copy.deepcopy(config), WORKSPACE_ROOT)
        finally:
            trainer._run_single_experiment = original

        assert summary["mask_seeds"] == [202000, 202001, 202002]
        assert summary["requested_budget"] == 0.001
        assert summary["selection_seed"] == 2020
        assert summary["source-target"] == "amazon-dslr"
        assert summary["runtime"] == summary["total_runtime"]
        assert all("runtime" in result for result in summary["masks"])
        for field in (
            "selection",
            "mask_static",
            "candidate_scope",
            "candidate_scope_type",
            "ranking_source",
            "mask_refresh_policy",
            "selected_param_count",
            "candidate_scope_param_count",
            "total_model_param_count",
            "selected_over_scope_ratio",
            "selected_over_model_ratio",
            "bn_stats_policy",
            "bn_stats_frozen",
            "bn_module_count",
        ):
            assert field in summary

        with open(osp.join(summary["output_dir"], "summary.json"), encoding="utf-8") as file_obj:
            persisted = json.load(file_obj)
        assert persisted["mask_seeds"] == [202000, 202001, 202002]
        outputs = build_summary_outputs(temp_dir)
        assert len(outputs["all_runs"]) == 1
        row = outputs["all_runs"][0]
        assert row["variant"] == "module_random"
        assert row["requested_budget"] == 0.001
        assert outputs["per_transfer_seed_aggregated"][0]["requested_budget"] == 0.001

    print("random multi-mask summary smoke test passed")


if __name__ == "__main__":
    main()
