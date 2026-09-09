#!/usr/bin/env python3
"""Synthetic, no-training VisDA planner/summary/analyzer integration checks."""

import json
import os
import os.path as osp
import sys
import tempfile

import numpy as np
import torch.nn as nn
from torchvision import models

PROJECT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from shot_otta.trainer import _configure_variant
from tools.analyze_visda_lbi_search import analyze
from tools.summarize_runs import build_summary_outputs, write_summary_outputs
from visda_otta.evaluator import compute_metrics


def _config(variant):
    return {"variant": variant, "requested_budget": 0.001, "selection_seed": 2020, "optimization": {"lr": 0.001, "lr_decay1": 0.1, "lr_decay2": 1.0}, "lbi": {"alpha": 0.1, "kappa": 1.0, "nu": 1.0, "omega": 0.1, "stage1_max_steps": 1, "budget_tolerance": 0.0, "stage2_lr": 0.01, "stage2_steps": 1, "delta_nonzero_tolerance": 0.0, "support_threshold": 1.0e-4}}


def _summary(key, fo, runtime):
    labels = np.arange(12)
    pu = compute_metrics(labels, labels, "PU")
    fo_metrics = compute_metrics(labels, np.asarray(fo), "FO")
    return {"status": "completed", "experiment_key": key, "experiment_config_sha256": key + "hash", "dataset": "VISDA-C", "source": 0, "target": 1, "seed": 2020, "variant": "module_lbi", "requested_budget": 0.001, "valid_lbi_run": True, "budget_diagnostics_available": True, "stage1_steps_completed_max": 3, "stage1_steps_completed_mean": 2.0, "runtime": runtime, **pu, **fo_metrics}


def main():
    labels = np.asarray([0, 0, 1])
    metrics = compute_metrics(labels, np.asarray([0, 1, 1]), "FO")
    assert metrics["FO-overall-Acc"] == 200.0 / 3.0
    assert metrics["FO-Acc"] == 150.0 / 12.0
    empty = compute_metrics([], [], "PU")
    assert len(empty["PU-Acc-per-class"]) == 12 and empty["PU-worst-class-id"] == 0

    net_f = models.resnet101(weights=None)
    net_b = nn.Module()
    net_b.bottleneck = nn.Linear(net_f.fc.in_features, 256)
    net_c = nn.Linear(256, 12)
    _, stats, _ = _configure_variant(_config("module_dense"), net_f, net_b, net_c)
    assert stats["candidate_scope_param_count"] == net_b.bottleneck.weight.numel() + net_b.bottleneck.bias.numel()
    assert 0 < stats["candidate_scope_param_count"] < stats["total_model_param_count"]

    with tempfile.TemporaryDirectory(prefix="iclr2027_visda_summary_") as root:
        plan = {"experiments": [{"experiment_key": "a", "dataset": "VISDA-C", "variant": "module_lbi", "requested_budget": 0.001, "alpha": 0.1, "kappa": 1.0, "nu": 1.0, "omega": 0.1, "stage1_max_steps": 3, "budget_tolerance": 0.0, "stage2_lr": 0.01, "stage2_steps_requested": 1, "delta_nonzero_tolerance": 0.0}, {"experiment_key": "b", "dataset": "VISDA-C", "variant": "module_lbi", "requested_budget": 0.001, "alpha": 0.2, "kappa": 1.0, "nu": 1.0, "omega": 0.1, "stage1_max_steps": 3, "budget_tolerance": 0.0, "stage2_lr": 0.01, "stage2_steps_requested": 1, "delta_nonzero_tolerance": 0.0}]}
        for key, predictions, runtime in (("a", np.arange(12), 2.0), ("b", np.zeros(12, dtype=int), 1.0)):
            run = osp.join(root, key)
            os.makedirs(run)
            with open(osp.join(run, "summary.json"), "w", encoding="utf-8") as handle:
                json.dump(_summary(key, predictions, runtime), handle)
        outputs = build_summary_outputs(root)
        out_dir = osp.join(root, "summary")
        write_summary_outputs(outputs, out_dir)
        for name in ("classwise_long.csv", "PU_classwise_wide.csv", "FO_classwise_wide.csv"):
            assert osp.isfile(osp.join(out_dir, name))
        plan_path = osp.join(root, "plan.json")
        with open(plan_path, "w", encoding="utf-8") as handle:
            json.dump(plan, handle)
        ranked = analyze(plan_path, root, osp.join(root, "ranking"))
        assert [row["experiment_key"] for row in ranked] == ["a", "b"]
        assert "PU-Acc" in ranked[0]
    print("VisDA planner/summary smoke test passed")


if __name__ == "__main__":
    main()
