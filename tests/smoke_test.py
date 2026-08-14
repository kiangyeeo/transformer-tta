#!/usr/bin/env python3
"""Artifact aggregation smoke test; it does not load data or run a model."""

import csv
import json
import numpy as np
import os.path as osp
import random
import sys
import tempfile
import torch
import torch.nn as nn


PROJECT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from tools.summarize_runs import (  # noqa: E402
    collect_summaries,
    write_csv,
    write_markdown,
)
from shot_otta.trainer import (  # noqa: E402
    _build_dynamic_saliency_masks,
    _build_static_magnitude_masks,
    _compute_mask_selection_stats,
    _compute_metrics,
    _compute_saliency_score,
    _configure_variant,
    _masked_optimizer_step,
    _summarize_dynamic_selection_stats,
)


class TinyFeature(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 4)
        self.bn = nn.BatchNorm1d(4)

    def forward(self, inputs):
        return self.bn(self.linear(inputs))


class TinyBottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.bottleneck = nn.Linear(4, 2)
        self.bn = nn.BatchNorm1d(2)

    def forward(self, inputs):
        return self.bn(self.bottleneck(inputs))


def _variant_config(variant, requested_budget=0.1, selection_seed=2020):
    return {
        "variant": variant,
        "requested_budget": requested_budget,
        "selection_seed": selection_seed,
        "optimization": {
            "lr": 0.01,
            "lr_decay1": 0.1,
            "lr_decay2": 1.0,
        },
        "lbi": {
            "alpha": 0.1,
            "kappa": 1.0,
            "nu": 1.0,
            "omega": 0.1,
            "stage1_max_steps": 3,
            "budget_tolerance": 0.0001,
            "stage2_lr": 0.01,
            "stage2_steps": 1,
            "delta_nonzero_tolerance": 1.0e-12,
        },
    }


def _check_variant_counts():
    net_f = TinyFeature()
    net_b = TinyBottleneck()
    net_c = nn.Linear(2, 2)
    count_f = sum(parameter.numel() for parameter in net_f.parameters())
    count_b = sum(parameter.numel() for parameter in net_b.parameters())
    count_c = sum(parameter.numel() for parameter in net_c.parameters())
    total = count_f + count_b + count_c
    module_count = (
        net_b.bottleneck.weight.numel()
        + net_b.bottleneck.bias.numel()
    )

    groups, stats, masks = _configure_variant(
        _variant_config("source_only"), net_f, net_b, net_c
    )
    assert groups == []
    assert masks == {}
    assert stats["selected_param_count"] == 0
    assert stats["candidate_scope_param_count"] == 0
    assert stats["total_model_param_count"] == total
    assert stats["selected_over_scope_ratio"] == 0.0
    assert stats["selected_over_model_ratio"] == 0.0
    assert stats["bn_stats_policy"] == "frozen"
    assert stats["bn_stats_frozen"] is True
    assert stats["bn_module_count"] == 2
    assert not any(
        parameter.requires_grad
        for model in (net_f, net_b, net_c)
        for parameter in model.parameters()
    )
    assert not net_f.training
    assert not net_b.training
    assert not net_c.training

    inputs = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    source_before = _snapshot_bn_buffers(net_f, net_b, net_c)
    with torch.no_grad():
        net_c(net_b(net_f(inputs)))
    source_after = _snapshot_bn_buffers(net_f, net_b, net_c)
    assert _bn_buffers_equal(source_before, source_after)

    groups, stats, masks = _configure_variant(
        _variant_config("full_dense"), net_f, net_b, net_c
    )
    assert masks == {}
    assert len(groups) == len(list(net_f.parameters())) + len(
        list(net_b.parameters())
    )
    assert stats["selected_param_count"] == count_f + count_b
    assert stats["candidate_scope_param_count"] == count_f + count_b
    assert stats["selected_over_scope_ratio"] == 1.0
    assert stats["bn_stats_policy"] == "adaptive"
    assert stats["bn_stats_frozen"] is False
    assert net_f.training
    assert net_b.training
    assert not net_c.training
    assert all(
        module.training
        for model in (net_f, net_b)
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    )
    assert all(parameter.requires_grad for parameter in net_f.parameters())
    assert all(parameter.requires_grad for parameter in net_b.parameters())
    assert not any(
        parameter.requires_grad for parameter in net_c.parameters()
    )

    full_before = _snapshot_bn_buffers(net_f, net_b, net_c)
    net_c(net_b(net_f(inputs)))
    full_after = _snapshot_bn_buffers(net_f, net_b, net_c)
    assert not _bn_buffers_equal(full_before, full_after)

    groups, stats, masks = _configure_variant(
        _variant_config("module_dense"), net_f, net_b, net_c
    )
    assert masks == {}
    assert len(groups) == 2
    assert stats["selected_param_count"] == module_count
    assert stats["candidate_scope_param_count"] == module_count
    assert stats["selected_over_scope_ratio"] == 1.0
    assert stats["bn_stats_policy"] == "frozen"
    assert stats["bn_stats_frozen"] is True
    assert net_f.training
    assert net_b.training
    assert not net_c.training
    assert not any(
        module.training
        for model in (net_f, net_b, net_c)
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    )
    assert net_b.bottleneck.weight.requires_grad
    assert net_b.bottleneck.bias.requires_grad
    assert not net_b.bn.weight.requires_grad
    selected_names = {
        f"{model_name}.{name}"
        for model_name, model in (
            ("netF", net_f),
            ("netB", net_b),
            ("netC", net_c),
        )
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert selected_names == {
        "netB.bottleneck.weight",
        "netB.bottleneck.bias",
    }

    module_before = _snapshot_bn_buffers(net_f, net_b, net_c)
    net_c(net_b(net_f(inputs)))
    module_after = _snapshot_bn_buffers(net_f, net_b, net_c)
    assert _bn_buffers_equal(module_before, module_after)

    _check_module_random(
        net_f=net_f,
        net_b=net_b,
        net_c=net_c,
        inputs=inputs,
        candidate_count=module_count,
    )
    _check_module_magnitude(
        net_f=net_f,
        net_b=net_b,
        net_c=net_c,
        inputs=inputs,
        candidate_count=module_count,
        total_count=total,
    )
    _check_module_saliency(
        net_f=net_f,
        net_b=net_b,
        net_c=net_c,
        inputs=inputs,
        candidate_count=module_count,
        total_count=total,
    )
    groups, stats, masks = _configure_variant(
        _variant_config("module_lbi", requested_budget=0.001),
        net_f,
        net_b,
        net_c,
    )
    assert groups == []
    assert masks == {}
    assert stats["selection"] == "lbi"
    assert stats["candidate_scope_param_count"] == module_count
    assert stats["selected_param_count"] is None
    assert stats["mask_refresh_policy"] == (
        "every_online_step_via_split_lbi"
    )
    assert stats["lbi_state_lifecycle"] == "reset_every_online_step"
    assert net_b.bottleneck.weight.requires_grad
    assert net_b.bottleneck.bias.requires_grad
    assert not any(
        module.training
        for model in (net_f, net_b, net_c)
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    )


def _mask_dict_equal(left, right):
    return left.keys() == right.keys() and all(
        torch.equal(left[name], right[name]) for name in left
    )


def _mask_dict_is_subset(smaller, larger):
    return smaller.keys() == larger.keys() and all(
        torch.all(torch.logical_or(~smaller[name], larger[name])).item()
        for name in smaller
    )


def _exercise_masked_optimizer_step(
    groups,
    masks,
    net_f,
    net_b,
    net_c,
    inputs,
):
    parameter_lookup = {
        f"netB.{name}": parameter
        for name, parameter in net_b.named_parameters()
        if f"netB.{name}" in masks
    }
    optimizer = torch.optim.SGD(
        groups,
        lr=0.05,
        momentum=0.9,
        weight_decay=0.1,
    )
    bn_before = _snapshot_bn_buffers(net_f, net_b, net_c)
    selected_changed = False
    for _ in range(2):
        optimizer.zero_grad()
        loss = net_c(net_b(net_f(inputs))).pow(2).mean()
        loss.backward()
        step_before = {
            name: parameter.detach().clone()
            for name, parameter in parameter_lookup.items()
        }
        _masked_optimizer_step(optimizer, parameter_lookup, masks)
        for name, parameter in parameter_lookup.items():
            mask = masks[name]
            assert torch.equal(
                parameter.detach()[~mask],
                step_before[name][~mask],
            )
            if torch.any(
                parameter.detach()[mask] != step_before[name][mask]
            ):
                selected_changed = True
    assert selected_changed
    bn_after = _snapshot_bn_buffers(net_f, net_b, net_c)
    assert _bn_buffers_equal(bn_before, bn_after)


def _check_module_random(
    net_f,
    net_b,
    net_c,
    inputs,
    candidate_count,
):
    torch.manual_seed(314159)
    rng_before = torch.random.get_rng_state().clone()
    groups, stats, masks_a = _configure_variant(
        _variant_config(
            "module_random",
            requested_budget=0.5,
            selection_seed=17,
        ),
        net_f,
        net_b,
        net_c,
    )
    rng_after = torch.random.get_rng_state().clone()
    assert torch.equal(rng_before, rng_after)
    assert stats["selection"] == "random"
    assert stats["selection_seed"] == 17
    assert stats["mask_static"] is True
    assert stats["candidate_scope_type"] == "fc_parameters"
    assert stats["bn_stats_policy"] == "frozen"
    assert stats["bn_stats_frozen"] is True
    assert stats["candidate_scope_param_count"] == candidate_count
    assert stats["selected_param_count"] == sum(
        int(mask.count_nonzero().item()) for mask in masks_a.values()
    )
    assert stats["selected_over_scope_ratio"] == (
        stats["selected_param_count"] / candidate_count
    )
    assert not any(
        module.training
        for model in (net_f, net_b, net_c)
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    )

    _, _, masks_same = _configure_variant(
        _variant_config(
            "module_random",
            requested_budget=0.5,
            selection_seed=17,
        ),
        net_f,
        net_b,
        net_c,
    )
    _, _, masks_different = _configure_variant(
        _variant_config(
            "module_random",
            requested_budget=0.5,
            selection_seed=18,
        ),
        net_f,
        net_b,
        net_c,
    )
    assert _mask_dict_equal(masks_a, masks_same)
    assert not _mask_dict_equal(masks_a, masks_different)

    _, _, masks_small = _configure_variant(
        _variant_config(
            "module_random",
            requested_budget=0.25,
            selection_seed=17,
        ),
        net_f,
        net_b,
        net_c,
    )
    _, _, masks_large = _configure_variant(
        _variant_config(
            "module_random",
            requested_budget=0.75,
            selection_seed=17,
        ),
        net_f,
        net_b,
        net_c,
    )
    assert _mask_dict_is_subset(masks_small, masks_large)

    selected_names = {
        f"{model_name}.{name}"
        for model_name, model in (
            ("netF", net_f),
            ("netB", net_b),
            ("netC", net_c),
        )
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert selected_names == {
        "netB.bottleneck.weight",
        "netB.bottleneck.bias",
    }

    _exercise_masked_optimizer_step(
        groups,
        masks_a,
        net_f,
        net_b,
        net_c,
        inputs,
    )

    for budget, expected_selected in (
        (0.0, 0),
        (1.0e-12, len(masks_a)),
        (1.0, candidate_count),
    ):
        _, edge_stats, edge_masks = _configure_variant(
            _variant_config(
                "module_random",
                requested_budget=budget,
                selection_seed=17,
            ),
            net_f,
            net_b,
            net_c,
        )
        assert edge_stats["selected_param_count"] == expected_selected
        assert sum(
            int(mask.count_nonzero().item())
            for mask in edge_masks.values()
        ) == expected_selected
        assert edge_stats["selected_over_scope_ratio"] == (
            expected_selected / candidate_count
        )


def _numpy_rng_states_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _check_module_magnitude(
    net_f,
    net_b,
    net_c,
    inputs,
    candidate_count,
    total_count,
):
    distinct_parameter = nn.Parameter(
        torch.tensor([1.0, -4.0, 3.0, -2.0])
    )
    distinct_mask = _build_static_magnitude_masks(
        [("distinct", distinct_parameter)],
        requested_budget=0.5,
    )["distinct"]
    assert torch.equal(
        distinct_mask,
        torch.tensor([False, True, True, False]),
    )

    tied_parameter = nn.Parameter(
        torch.tensor([5.0, -5.0, 5.0, -5.0])
    )
    tied_mask = _build_static_magnitude_masks(
        [("tied", tied_parameter)],
        requested_budget=0.5,
    )["tied"]
    tied_mask_repeat = _build_static_magnitude_masks(
        [("tied", tied_parameter)],
        requested_budget=0.5,
    )["tied"]
    assert torch.equal(
        tied_mask,
        torch.tensor([True, True, False, False]),
    )
    assert torch.equal(tied_mask, tied_mask_repeat)

    with torch.no_grad():
        net_b.bottleneck.weight.copy_(
            torch.tensor(
                [
                    [1.0, -8.0, 3.0, -6.0],
                    [7.0, -2.0, 5.0, -4.0],
                ]
            )
        )
        net_b.bottleneck.bias.copy_(torch.tensor([2.0, -9.0]))

    random.seed(2718)
    np.random.seed(2718)
    torch.manual_seed(2718)
    python_rng_before = random.getstate()
    numpy_rng_before = np.random.get_state()
    torch_rng_before = torch.random.get_rng_state().clone()
    groups, stats, masks = _configure_variant(
        _variant_config(
            "module_magnitude",
            requested_budget=0.5,
            selection_seed=999,
        ),
        net_f,
        net_b,
        net_c,
    )
    python_rng_after = random.getstate()
    numpy_rng_after = np.random.get_state()
    torch_rng_after = torch.random.get_rng_state().clone()
    assert python_rng_before == python_rng_after
    assert _numpy_rng_states_equal(numpy_rng_before, numpy_rng_after)
    assert torch.equal(torch_rng_before, torch_rng_after)

    assert stats["selection"] == "magnitude"
    assert stats["selection_seed"] is None
    assert stats["mask_static"] is True
    assert stats["candidate_scope"] == "netB.bottleneck"
    assert stats["candidate_scope_type"] == "fc_parameters"
    assert stats["ranking_source"] == (
        "source_checkpoint_pre_adaptation"
    )
    assert stats["bn_stats_policy"] == "frozen"
    assert stats["bn_stats_frozen"] is True
    assert stats["candidate_scope_param_count"] == candidate_count
    assert stats["total_model_param_count"] == total_count
    expected_selected = sum(
        int(mask.count_nonzero().item()) for mask in masks.values()
    )
    assert stats["selected_param_count"] == expected_selected
    assert stats["selected_over_scope_ratio"] == (
        expected_selected / candidate_count
    )
    assert stats["selected_over_model_ratio"] == (
        expected_selected / total_count
    )
    assert torch.equal(
        masks["netB.bottleneck.weight"].reshape(-1),
        torch.tensor(
            [False, True, False, True, True, False, True, False]
        ),
    )
    assert torch.equal(
        masks["netB.bottleneck.bias"],
        torch.tensor([False, True]),
    )
    assert not any(
        module.training
        for model in (net_f, net_b, net_c)
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    )

    source_masks = {
        name: mask.clone() for name, mask in masks.items()
    }
    with torch.no_grad():
        net_b.bottleneck.weight.fill_(100.0)
        net_b.bottleneck.bias.zero_()
    assert _mask_dict_equal(source_masks, masks)

    _exercise_masked_optimizer_step(
        groups,
        masks,
        net_f,
        net_b,
        net_c,
        inputs,
    )

    for budget, edge_expected in (
        (0.0, 0),
        (1.0e-12, len(masks)),
        (1.0, candidate_count),
    ):
        _, edge_stats, edge_masks = _configure_variant(
            _variant_config(
                "module_magnitude",
                requested_budget=budget,
                selection_seed=None,
            ),
            net_f,
            net_b,
            net_c,
        )
        mask_count = sum(
            int(mask.count_nonzero().item())
            for mask in edge_masks.values()
        )
        assert mask_count == edge_expected
        assert edge_stats["selected_param_count"] == edge_expected
        assert edge_stats["selected_over_scope_ratio"] == (
            edge_expected / candidate_count
        )
        assert edge_stats["selected_over_model_ratio"] == (
            edge_expected / total_count
        )


def _cuda_rng_states_equal(left, right):
    return len(left) == len(right) and all(
        torch.equal(before, after)
        for before, after in zip(left, right)
    )


def _assert_step_preserves_mask_outside(
    optimizer,
    parameter_lookup,
    masks,
):
    step_before = {
        name: parameter.detach().clone()
        for name, parameter in parameter_lookup.items()
    }
    _masked_optimizer_step(optimizer, parameter_lookup, masks)
    for name, parameter in parameter_lookup.items():
        mask = masks[name]
        assert torch.equal(
            parameter.detach()[~mask],
            step_before[name][~mask],
        )


def _check_module_saliency(
    net_f,
    net_b,
    net_c,
    inputs,
    candidate_count,
    total_count,
):
    weight = nn.Parameter(torch.tensor([1.0, -2.0, 3.0, -4.0]))
    bias = nn.Parameter(torch.tensor([1.0, -2.0]))
    weight.grad = torch.tensor([8.0, 1.0, -1.0, 0.5])
    bias.grad = torch.tensor([1.0, 3.0])
    named_parameters = [("weight", weight), ("bias", bias)]

    expected_weight_scores = torch.tensor([8.0, 2.0, 3.0, 2.0])
    assert torch.equal(
        _compute_saliency_score("weight", weight),
        expected_weight_scores,
    )
    gradients_before = {
        name: parameter.grad.clone()
        for name, parameter in named_parameters
    }

    random.seed(1618)
    np.random.seed(1618)
    torch.manual_seed(1618)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1618)
    python_rng_before = random.getstate()
    numpy_rng_before = np.random.get_state()
    torch_rng_before = torch.random.get_rng_state().clone()
    cuda_rng_before = [
        state.clone() for state in torch.cuda.get_rng_state_all()
    ]
    masks = _build_dynamic_saliency_masks(
        named_parameters,
        requested_budget=0.5,
    )
    python_rng_after = random.getstate()
    numpy_rng_after = np.random.get_state()
    torch_rng_after = torch.random.get_rng_state().clone()
    cuda_rng_after = [
        state.clone() for state in torch.cuda.get_rng_state_all()
    ]

    assert python_rng_before == python_rng_after
    assert _numpy_rng_states_equal(numpy_rng_before, numpy_rng_after)
    assert torch.equal(torch_rng_before, torch_rng_after)
    assert _cuda_rng_states_equal(cuda_rng_before, cuda_rng_after)
    assert torch.equal(
        masks["weight"],
        torch.tensor([True, False, True, False]),
    )
    assert torch.equal(
        masks["bias"],
        torch.tensor([False, True]),
    )
    for name, parameter in named_parameters:
        assert torch.equal(parameter.grad, gradients_before[name])

    tied = nn.Parameter(torch.ones(4))
    tied.grad = torch.tensor([5.0, -5.0, 5.0, -5.0])
    tied_mask = _build_dynamic_saliency_masks(
        [("tied", tied)],
        requested_budget=0.5,
    )["tied"]
    assert torch.equal(
        tied_mask,
        torch.tensor([True, True, False, False]),
    )

    dynamic = nn.Parameter(torch.ones(4))
    dynamic.grad = torch.tensor([4.0, 3.0, 2.0, 1.0])
    first_mask = _build_dynamic_saliency_masks(
        [("dynamic", dynamic)],
        requested_budget=0.5,
    )["dynamic"]
    dynamic.grad = torch.tensor([1.0, 2.0, 3.0, 4.0])
    gradient_changed_mask = _build_dynamic_saliency_masks(
        [("dynamic", dynamic)],
        requested_budget=0.5,
    )["dynamic"]
    assert not torch.equal(first_mask, gradient_changed_mask)
    with torch.no_grad():
        dynamic.copy_(torch.tensor([9.0, 8.0, 1.0, 1.0]))
    parameter_changed_mask = _build_dynamic_saliency_masks(
        [("dynamic", dynamic)],
        requested_budget=0.5,
    )["dynamic"]
    assert not torch.equal(
        gradient_changed_mask,
        parameter_changed_mask,
    )

    missing_grad = nn.Parameter(torch.ones(2))
    try:
        _build_dynamic_saliency_masks(
            [("netB.bottleneck.bias", missing_grad)],
            requested_budget=0.5,
        )
    except RuntimeError as error:
        assert "netB.bottleneck.bias" in str(error)
        assert "gradient is None" in str(error)
    else:
        raise AssertionError("Expected missing-gradient RuntimeError")

    for budget, expected_selected in (
        (0.0, 0),
        (1.0e-12, len(named_parameters)),
        (1.0, sum(parameter.numel() for _, parameter in named_parameters)),
    ):
        edge_masks = _build_dynamic_saliency_masks(
            named_parameters,
            requested_budget=budget,
        )
        edge_count = sum(
            int(mask.count_nonzero().item())
            for mask in edge_masks.values()
        )
        assert edge_count == expected_selected

    direct_stats = _compute_mask_selection_stats(
        masks,
        candidate_scope_param_count=6,
        total_model_param_count=10,
    )
    assert direct_stats["selected_param_count"] == 3
    assert direct_stats["selected_over_scope_ratio"] == 3 / 6
    assert direct_stats["selected_over_model_ratio"] == 3 / 10

    update_weight = nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    update_bias = nn.Parameter(torch.tensor([1.0, 2.0]))
    update_named = [
        ("weight", update_weight),
        ("bias", update_bias),
    ]
    update_lookup = dict(update_named)
    optimizer = torch.optim.SGD(
        [update_weight, update_bias],
        lr=0.1,
        momentum=0.9,
        weight_decay=0.1,
    )
    step_masks = []
    for weight_gradient, bias_gradient in (
        (
            torch.tensor([4.0, 3.0, 2.0, 1.0]),
            torch.tensor([2.0, 1.0]),
        ),
        (
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.tensor([1.0, 2.0]),
        ),
    ):
        optimizer.zero_grad()
        update_weight.grad = weight_gradient
        update_bias.grad = bias_gradient
        current_masks = _build_dynamic_saliency_masks(
            update_named,
            requested_budget=0.5,
        )
        step_masks.append(
            {name: mask.clone() for name, mask in current_masks.items()}
        )
        _assert_step_preserves_mask_outside(
            optimizer,
            update_lookup,
            current_masks,
        )
    assert not _mask_dict_equal(step_masks[0], step_masks[1])

    groups, stats, static_masks = _configure_variant(
        _variant_config(
            "module_saliency",
            requested_budget=0.5,
            selection_seed=999,
        ),
        net_f,
        net_b,
        net_c,
    )
    assert static_masks == {}
    assert len(groups) == 2
    assert stats["selection"] == "saliency"
    assert stats["selection_seed"] is None
    assert stats["mask_static"] is False
    assert stats["mask_refresh_policy"] == (
        "every_online_step_after_backward"
    )
    assert stats["ranking_source"] == (
        "current_parameter_times_current_gradient"
    )
    assert stats["saliency_score"] == (
        "abs_parameter_times_gradient"
    )
    assert stats["candidate_scope"] == "netB.bottleneck"
    assert stats["candidate_scope_type"] == "fc_parameters"
    assert stats["selected_param_count"] is None
    assert stats["selected_over_scope_ratio"] is None
    assert stats["selected_over_model_ratio"] is None
    assert stats["expected_selected_param_count_per_step"] == 5
    assert stats["candidate_scope_param_count"] == candidate_count
    assert stats["total_model_param_count"] == total_count
    assert stats["bn_stats_policy"] == "frozen"
    assert stats["bn_stats_frozen"] is True
    assert stats["bn_module_count"] == 2
    assert net_f.training
    assert net_b.training
    assert not net_c.training
    saliency_selected_names = {
        f"{model_name}.{name}"
        for model_name, model in (
            ("netF", net_f),
            ("netB", net_b),
            ("netC", net_c),
        )
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert saliency_selected_names == {
        "netB.bottleneck.weight",
        "netB.bottleneck.bias",
    }
    assert not any(
        module.training
        for model in (net_f, net_b, net_c)
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    )

    candidate_lookup = {
        f"netB.{name}": parameter
        for name, parameter in net_b.named_parameters()
        if f"netB.{name}" in {
            "netB.bottleneck.weight",
            "netB.bottleneck.bias",
        }
    }
    model_optimizer = torch.optim.SGD(
        groups,
        lr=0.05,
        momentum=0.9,
        weight_decay=0.1,
    )
    bn_before = _snapshot_bn_buffers(net_f, net_b, net_c)
    for _ in range(2):
        model_optimizer.zero_grad()
        loss = net_c(net_b(net_f(inputs))).pow(2).mean()
        loss.backward()
        model_masks = _build_dynamic_saliency_masks(
            list(candidate_lookup.items()),
            requested_budget=0.5,
        )
        _assert_step_preserves_mask_outside(
            model_optimizer,
            candidate_lookup,
            model_masks,
        )
    bn_after = _snapshot_bn_buffers(net_f, net_b, net_c)
    assert _bn_buffers_equal(bn_before, bn_after)

    aggregate = _summarize_dynamic_selection_stats(
        [
            {
                "selected_param_count": 3,
                "selected_over_scope_ratio": 3 / 6,
                "selected_over_model_ratio": 3 / 10,
            },
            {
                "selected_param_count": 5,
                "selected_over_scope_ratio": 5 / 6,
                "selected_over_model_ratio": 5 / 10,
            },
        ]
    )
    assert aggregate["selected_param_count"] == 4.0
    assert aggregate["selected_param_count_first"] == 3
    assert aggregate["selected_param_count_last"] == 5
    assert aggregate["selected_param_count_min"] == 3
    assert aggregate["selected_param_count_max"] == 5
    assert aggregate["selected_param_count_mean"] == 4.0
    assert aggregate["selected_over_scope_ratio_mean"] == (
        (3 / 6 + 5 / 6) / 2
    )
    assert aggregate["selected_over_model_ratio_mean"] == (
        (3 / 10 + 5 / 10) / 2
    )


def _snapshot_bn_buffers(*models):
    snapshots = []
    for model in models:
        for module in model.modules():
            if not isinstance(module, nn.modules.batchnorm._BatchNorm):
                continue
            snapshots.append(
                {
                    "running_mean": module.running_mean.detach().clone(),
                    "running_var": module.running_var.detach().clone(),
                    "num_batches_tracked": (
                        module.num_batches_tracked.detach().clone()
                    ),
                }
            )
    return snapshots


def _bn_buffers_equal(left, right):
    if len(left) != len(right):
        return False
    return all(
        torch.equal(before[key], after[key])
        for before, after in zip(left, right)
        for key in (
            "running_mean",
            "running_var",
            "num_batches_tracked",
        )
    )


def main():
    _check_variant_counts()
    precise_value = 0.12345678901234568
    metric, per_class = _compute_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 1]),
    )
    assert metric == 75.0
    assert per_class == [50.0, 100.0]

    with tempfile.TemporaryDirectory(prefix="iclr2027_smoke_") as temp_dir:
        run_dir = osp.join(temp_dir, "runs", "run-a")
        import os

        os.makedirs(run_dir)
        summary = {
            "status": "completed",
            "experiment_key": "dense-experiment",
            "experiment_config_sha256": "a" * 64,
            "method": "shot",
            "variant": "full_dense",
            "task": "otta",
            "dataset": "office",
            "source": 0,
            "target": 1,
            "source-target": "amazon-dslr",
            "seed": 2020,
            "selection": "dense",
            "selection_seed": None,
            "mask_static": False,
            "candidate_scope": "netF+netB",
            "candidate_scope_type": "all_parameters",
            "requested_budget": 1.0,
            "ranking_source": None,
            "mask_refresh_policy": None,
            "saliency_score": None,
            "expected_selected_param_count_per_step": None,
            "selected_param_count": 100,
            "candidate_scope_param_count": 100,
            "total_model_param_count": 120,
            "selected_over_scope_ratio": 1.0,
            "selected_over_model_ratio": 100 / 120,
            "bn_stats_policy": "adaptive",
            "bn_stats_frozen": False,
            "bn_module_count": 2,
            "PU-Acc": precise_value,
            "FO-Acc": 88.76543210987654,
            "runtime": 12.3456789012345,
            "run_id": "run-a",
            "started_at_utc": "2026-01-01T00:00:00+00:00",
            "completed_at_utc": "2026-01-01T00:01:00+00:00",
        }
        with open(
            osp.join(run_dir, "summary.json"), "w", encoding="utf-8"
        ) as file_obj:
            json.dump(summary, file_obj)

        saliency_run_dir = osp.join(temp_dir, "runs", "run-b")
        os.makedirs(saliency_run_dir)
        saliency_summary = {
            **summary,
            "variant": "module_saliency",
            "experiment_key": "saliency-experiment",
            "experiment_config_sha256": "b" * 64,
            "selection": "saliency",
            "mask_static": False,
            "candidate_scope": "netB.bottleneck",
            "candidate_scope_type": "fc_parameters",
            "requested_budget": 0.1,
            "ranking_source": (
                "current_parameter_times_current_gradient"
            ),
            "mask_refresh_policy": (
                "every_online_step_after_backward"
            ),
            "saliency_score": "abs_parameter_times_gradient",
            "expected_selected_param_count_per_step": 13,
            "selected_param_count": 13.0,
            "candidate_scope_param_count": 128,
            "total_model_param_count": 1024,
            "selected_over_scope_ratio": 13 / 128,
            "selected_over_model_ratio": precise_value,
            "selected_param_count_first": 13,
            "selected_param_count_last": 13,
            "selected_param_count_min": 13,
            "selected_param_count_max": 13,
            "selected_param_count_mean": 13.0,
            "selected_over_scope_ratio_first": 13 / 128,
            "selected_over_scope_ratio_last": 13 / 128,
            "selected_over_scope_ratio_min": 13 / 128,
            "selected_over_scope_ratio_max": 13 / 128,
            "selected_over_scope_ratio_mean": 13 / 128,
            "selected_over_model_ratio_first": precise_value,
            "selected_over_model_ratio_last": precise_value,
            "selected_over_model_ratio_min": precise_value,
            "selected_over_model_ratio_max": precise_value,
            "selected_over_model_ratio_mean": precise_value,
            "run_id": "run-b",
        }
        with open(
            osp.join(saliency_run_dir, "summary.json"),
            "w",
            encoding="utf-8",
        ) as file_obj:
            json.dump(saliency_summary, file_obj)

        rows = collect_summaries(osp.join(temp_dir, "runs"))
        assert len(rows) == 2
        dense_row = next(
            row for row in rows if row["variant"] == "full_dense"
        )
        saliency_row = next(
            row
            for row in rows
            if row["variant"] == "module_saliency"
        )
        assert dense_row["PU-Acc"] == precise_value
        assert (
            saliency_row["selected_over_model_ratio_mean"]
            == precise_value
        )

        csv_path = osp.join(temp_dir, "summary.csv")
        markdown_path = osp.join(temp_dir, "summary.md")
        write_csv(csv_path, rows)
        write_markdown(markdown_path, rows)

        with open(csv_path, "r", encoding="utf-8") as file_obj:
            csv_rows = list(csv.DictReader(file_obj))
        saliency_csv_row = next(
            row
            for row in csv_rows
            if row["variant"] == "module_saliency"
        )
        assert (
            float(
                saliency_csv_row[
                    "selected_over_model_ratio_mean"
                ]
            )
            == precise_value
        )
        with open(markdown_path, "r", encoding="utf-8") as file_obj:
            markdown = file_obj.read()
        assert "module_saliency" in markdown
        assert str(precise_value) in markdown

    print("iclr2027 smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
