#!/usr/bin/env python3
"""CPU-only Conv global-budget selector and diagnostic contracts (P2 G, N-P, Q-R)."""

import copy
import math
import os.path as osp
import sys
import tempfile

import torch
import torch.nn as nn


PROJECT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
WORKSPACE_ROOT = osp.dirname(PROJECT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from core.lbi.diagnostics import compute_lbi_run_budget_diagnostics, max_support_count  # noqa: E402
from core.lbi.groups import (  # noqa: E402
    build_random_group_masks, global_group_score_mask, selected_group_count,
    selected_scalar_count,
)
from shot_otta import trainer  # noqa: E402
from shot_otta.config import load_yaml, resolve_effective_config  # noqa: E402


def _named_tensors():
    # Out-channel group sizes differ by five, which makes scalar diagnostics
    # intentionally distinct from group diagnostics.
    return [
        ("small", nn.Parameter(torch.zeros(3, 1, 1, 1))),
        ("large", nn.Parameter(torch.zeros(3, 5, 1, 1))),
    ]


def _check_global_integer_budget():
    layers = [
        ("first", nn.Parameter(torch.zeros(1, 1, 1, 1))),
        ("second", nn.Parameter(torch.zeros(1, 1, 1, 1))),
    ]
    rho = 0.6
    assert max_support_count(rho, 2) == 1
    assert sum(max_support_count(rho, 1) for _ in layers) == 0
    masks = build_random_group_masks(layers, "out_channel", rho, seed=2026)
    assert selected_group_count(masks, layers, "out_channel") == 1


def _check_random_global_exact_k_and_parent_stats():
    named = _named_tensors()
    rho = 0.5  # K=floor(.5 * 6)=3 global groups.
    child_masks = [
        build_random_group_masks(named, "out_channel", rho, seed=202600 + index)
        for index in range(3)
    ]
    rerun = build_random_group_masks(named, "out_channel", rho, seed=202600)
    assert all(torch.equal(child_masks[0][name], rerun[name]) for name, _ in named)
    group_counts = [selected_group_count(mask, named, "out_channel") for mask in child_masks]
    scalar_counts = [selected_scalar_count(mask) for mask in child_masks]
    assert group_counts == [3, 3, 3]
    assert len(set(scalar_counts)) > 1
    filter_named = [
        ("small_kernel", nn.Parameter(torch.zeros(1, 3, 1, 1))),
        ("large_kernel", nn.Parameter(torch.zeros(1, 3, 3, 3))),
    ]
    filter_masks = [
        build_random_group_masks(filter_named, "filter_connection", rho, seed=202600 + index)
        for index in range(3)
    ]
    assert [selected_group_count(mask, filter_named, "filter_connection") for mask in filter_masks] == [3, 3, 3]

    def fake_child(config, workspace_root, **kwargs):
        del workspace_root, kwargs
        index = config["selection_seed"] % 100
        scalar_count = scalar_counts[index]
        return {
            "PU-Acc": float(index), "FO-Acc": float(index + 1), "runtime": 0.0,
            "online_batch_runtime_mean_sec": 1.0, "online_batch_runtime_std_sec": 0.0,
            "online_batch_runtime_median_sec": 1.0, "online_batch_runtime_p95_sec": 1.0,
            "online_compute_runtime_sec": 1.0, "adapt_batch_runtime_mean_sec": 0.5,
            "adapt_batch_runtime_std_sec": 0.0, "adapt_runtime_total_sec": 0.5,
            "pu_batch_runtime_mean_sec": 0.5, "pu_batch_runtime_std_sec": 0.0,
            "pu_runtime_total_sec": 0.5, "gpu_peak_allocated_mean_mb": 1.0,
            "gpu_peak_allocated_max_mb": 1.0, "gpu_peak_reserved_mean_mb": 1.0,
            "gpu_peak_reserved_max_mb": 1.0, "started_at_utc": "x", "completed_at_utc": "y",
            "selection": "random", "mask_static": True, "candidate_scope": "netF.layer4_conv",
            "candidate_scope_type": "layer4_conv_weights", "ranking_source": "independent_random_generator",
            "mask_refresh_policy": "once_before_adaptation", "selected_param_count": scalar_count,
            "candidate_scope_param_count": 18, "total_model_param_count": 100,
            "selected_over_scope_ratio": scalar_count / 18, "selected_over_model_ratio": scalar_count / 100,
            "candidate_layer_names": ["small", "large"], "group_mode": "out_channel",
            "total_group_count": 6, "max_group_count": 3, "selected_group_count": 3,
            "selected_scalar_count": scalar_count, "realized_group_ratio": 0.5,
            "realized_scalar_ratio": scalar_count / 18, "bn_stats_policy": "frozen",
            "bn_stats_frozen": True, "bn_module_count": 0,
        }

    with tempfile.TemporaryDirectory(prefix="conv_random_summary_") as root:
        config = load_yaml(osp.join(PROJECT_DIR, "configs", "otta_conv_lbi_protocol_20260826_v1.yaml"))
        config.update({"variant": "conv_out_random", "group_mode": "out_channel", "requested_budget": rho})
        config["output"] = {**config["output"], "root": root}
        config = resolve_effective_config(config, WORKSPACE_ROOT)
        original = trainer._run_single_experiment
        trainer._run_single_experiment = fake_child
        try:
            summary = trainer._run_random_experiment(copy.deepcopy(config), WORKSPACE_ROOT)
        finally:
            trainer._run_single_experiment = original
    assert summary["selected_group_count"] == 3
    assert summary["selected_scalar_count_mean"] == sum(scalar_counts) / 3
    assert summary["selected_scalar_count_std"] == math.sqrt(sum((value - sum(scalar_counts) / 3) ** 2 for value in scalar_counts) / 3)
    assert summary["selected_scalar_count_min"] == min(scalar_counts)
    assert summary["selected_scalar_count_max"] == max(scalar_counts)


def _check_magnitude_and_saliency_global_top_k():
    named = [
        ("dominant", nn.Parameter(torch.tensor([[[[10.0]]], [[[9.0]]]]))),
        ("weak", nn.Parameter(torch.tensor([[[[1.0]]], [[[1.0]]]]))),
    ]
    magnitude = global_group_score_mask(named, {name: value for name, value in named}, "out_channel", 0.5)
    assert magnitude["dominant"][:, 0, 0, 0].tolist() == [True, True]
    assert magnitude["weak"][:, 0, 0, 0].tolist() == [False, False]
    assert selected_group_count(magnitude, named, "out_channel") == 2

    weights = [
        ("first", nn.Parameter(torch.ones(2, 1, 1, 1))),
        ("second", nn.Parameter(torch.ones(2, 1, 1, 1))),
    ]
    gradients = {"first": torch.tensor([[[[5.0]]], [[[5.0]]]]), "second": torch.tensor([[[[4.0]]], [[[1.0]]]])}
    scores = {name: value.detach() * gradients[name] for name, value in weights}
    saliency = global_group_score_mask(weights, scores, "out_channel", 0.5)
    # The equal 5/5 tie is resolved by deterministic global order (first layer).
    assert saliency["first"][:, 0, 0, 0].tolist() == [True, True]
    assert saliency["second"][:, 0, 0, 0].tolist() == [False, False]
    assert selected_group_count(saliency, weights, "out_channel") == 2

    filter_weights = [
        ("first_filter", nn.Parameter(torch.tensor([[[[9.0]], [[8.0]]]]))),
        ("second_filter", nn.Parameter(torch.tensor([[[[1.0]], [[1.0]]]]))),
    ]
    filter_magnitude = global_group_score_mask(
        filter_weights, {name: value for name, value in filter_weights},
        "filter_connection", 0.5,
    )
    assert filter_magnitude["first_filter"][0, :, 0, 0].tolist() == [True, True]
    assert filter_magnitude["second_filter"][0, :, 0, 0].tolist() == [False, False]
    filter_gradients = {
        "first_filter": torch.tensor([[[[5.0]], [[5.0]]]]),
        "second_filter": torch.tensor([[[[4.0]], [[1.0]]]]),
    }
    filter_saliency = global_group_score_mask(
        filter_weights,
        {name: value.detach() * filter_gradients[name] for name, value in filter_weights},
        "filter_connection", 0.5,
    )
    assert filter_saliency["first_filter"][0, :, 0, 0].tolist() == [True, True]
    assert selected_group_count(filter_saliency, filter_weights, "filter_connection") == 2


def _check_group_scalar_and_utilization_diagnostics():
    named = _named_tensors()
    selected_small = {"small": torch.ones_like(named[0][1], dtype=torch.bool), "large": torch.zeros_like(named[1][1], dtype=torch.bool)}
    selected_large = {"small": torch.zeros_like(named[0][1], dtype=torch.bool), "large": torch.ones_like(named[1][1], dtype=torch.bool)}
    first = trainer._compute_group_mask_selection_stats(selected_small, named, "out_channel", 18, 100)
    second = trainer._compute_group_mask_selection_stats(selected_large, named, "out_channel", 18, 100)
    assert first["selected_group_count"] == second["selected_group_count"] == 3
    assert first["selected_scalar_count"] == 3
    assert second["selected_scalar_count"] == 15
    assert first["realized_group_ratio"] == second["realized_group_ratio"] == 0.5
    assert first["realized_scalar_ratio"] != second["realized_scalar_ratio"]
    assert first["selected_param_count"] == first["selected_scalar_count"]

    records = [
        {"requested_budget": 1.0, "candidate_scope_param_count": 18, "total_group_count": 10,
         "selected_group_count": count, "stage1_support_count": count,
         "stage1_steps_completed": 1, "stage1_stop_reason": "budget_reached" if count == 10 else "strict_budget_rollback"}
        for count in (10, 8, 9)
    ]
    diagnostics = compute_lbi_run_budget_diagnostics(records)
    assert diagnostics["group_utilization_min"] == 0.8
    assert diagnostics["group_utilization_mean"] == 0.9
    assert diagnostics["group_utilization_p05"] == 0.81
    assert diagnostics["group_utilization_below_95_count"] == 2
    assert diagnostics["group_utilization_below_95_fraction"] == 2 / 3


def main():
    _check_global_integer_budget()
    _check_random_global_exact_k_and_parent_stats()
    _check_magnitude_and_saliency_global_top_k()
    _check_group_scalar_and_utilization_diagnostics()
    print("conv budget contracts passed")


if __name__ == "__main__":
    main()
