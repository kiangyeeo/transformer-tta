#!/usr/bin/env python3
"""No-training checks for VisDA configuration, metrics, and BN protocol."""

import os.path as osp
import sys

import numpy as np
import torch.nn as nn

PROJECT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
WORKSPACE_ROOT = osp.dirname(PROJECT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from shot_otta.config import load_yaml, resolve_effective_config
from shot_otta.trainer import _compute_dataset_metrics, _configure_variant
from visda_otta.adapter import build_target_order
from visda_otta.evaluator import aggregate_random_mask_metrics, compute_metrics


class Feature(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.bn = nn.BatchNorm1d(2)


class Bottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.bottleneck = nn.Linear(2, 2)
        self.bn = nn.BatchNorm1d(2)


def variant_config(variant):
    return {"variant": variant, "requested_budget": 0.1, "selection_seed": 2020, "optimization": {"lr": 0.001, "lr_decay1": 0.1, "lr_decay2": 1.0}, "lbi": {"alpha": 0.1, "kappa": 1.0, "nu": 1.0, "omega": 0.1, "stage1_max_steps": 1, "budget_tolerance": 0.0, "stage2_lr": 0.001, "stage2_steps": 1, "delta_nonzero_tolerance": 0.0, "support_threshold": 1.0e-4}}


def main():
    raw = load_yaml(osp.join(PROJECT_DIR, "configs", "otta_fc_lbi_protocol_20260817_v1.yaml"))
    raw["data"]["dataset"] = "VISDA-C"
    config = resolve_effective_config(raw, WORKSPACE_ROOT)
    assert (config["data"]["dataset"], config["data"]["source"], config["data"]["target"]) == ("VISDA-C", 0, 1)
    assert config["model"]["backbone"] == "resnet101"
    assert config["optimization"]["lr"] == 0.001 and not config["output"]["save_model"]
    order1, meta1 = build_target_order(2026, 20)
    order2, meta2 = build_target_order(2026, 20)
    assert order1 == order2 and meta1 == meta2 and meta1["adapter"] == "visda_fixed_seed_order"
    labels = np.arange(12)
    pu = compute_metrics(labels, labels, "PU")
    fo = compute_metrics(labels, np.zeros(12, dtype=int), "FO")
    assert pu["PU-Acc"] == 100.0 and len(pu["PU-Acc-per-class"]) == 12
    assert fo["FO-overall-Acc"] == 100.0 / 12.0 and fo["FO-worst-class-name"] == "bicycle"
    imbalanced = _compute_dataset_metrics(
        np.asarray([0, 0, 1]),
        np.asarray([0, 1, 1]),
        "VISDA-C",
        "FO",
    )
    assert imbalanced["FO-Acc"] == 150.0 / 12.0
    assert imbalanced["FO-overall-Acc"] == 200.0 / 3.0
    aggregate = aggregate_random_mask_metrics([pu | compute_metrics(labels, labels, "FO"), pu | compute_metrics(labels, np.zeros(12, dtype=int), "FO")])
    assert len(aggregate["FO-Acc-per-class-mean"]) == 12
    for variant, policy in (("source_only", "frozen"), ("full_dense", "adaptive"), ("module_dense", "frozen"), ("module_random", "frozen"), ("module_magnitude", "frozen"), ("module_saliency", "frozen"), ("module_lbi", "frozen")):
        _, stats, _ = _configure_variant(variant_config(variant), Feature(), Bottleneck(), nn.Linear(2, 12))
        assert stats["bn_stats_policy"] == policy
    print("VisDA adapter and metrics smoke test passed")


if __name__ == "__main__":
    main()
