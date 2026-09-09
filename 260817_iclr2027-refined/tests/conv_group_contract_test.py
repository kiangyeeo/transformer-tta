#!/usr/bin/env python3
"""CPU-only contracts for Conv candidates and group partitions (P2 A-C, V)."""

import os.path as osp
import sys

import torch
import torch.nn as nn


PROJECT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from core.lbi.groups import (  # noqa: E402
    group_count,
    group_support_masks,
    group_view,
    local_group_count,
    selected_group_count,
)
from protocol_constants import (  # noqa: E402
    CONV_CANDIDATE_PARAM_COUNT,
    CONV_GROUP_COUNTS,
)
from shot_otta.trainer import (  # noqa: E402
    CONV_CANDIDATE_ORDER,
    CONV_CANDIDATE_SHAPES,
    _collect_conv_candidates,
    _configure_variant,
)


CONV_PROTOCOL_CANDIDATE_SHAPES = {
    "netF.layer4.0.conv1.weight": (512, 1024, 1, 1),
    "netF.layer4.0.conv2.weight": (512, 512, 3, 3),
    "netF.layer4.0.conv3.weight": (2048, 512, 1, 1),
    "netF.layer4.1.conv1.weight": (512, 2048, 1, 1),
    "netF.layer4.1.conv2.weight": (512, 512, 3, 3),
    "netF.layer4.1.conv3.weight": (2048, 512, 1, 1),
    "netF.layer4.2.conv1.weight": (512, 2048, 1, 1),
    "netF.layer4.2.conv2.weight": (512, 512, 3, 3),
    "netF.layer4.2.conv3.weight": (2048, 512, 1, 1),
}


def _meta_candidates():
    return [
        (name, nn.Parameter(torch.empty(shape, device="meta")))
        for name, shape in CONV_PROTOCOL_CANDIDATE_SHAPES.items()
    ]


def _check_candidate_scope():
    candidates = _meta_candidates()
    assert CONV_CANDIDATE_SHAPES == CONV_PROTOCOL_CANDIDATE_SHAPES
    assert tuple(CONV_PROTOCOL_CANDIDATE_SHAPES) == CONV_CANDIDATE_ORDER
    assert [name for name, _ in candidates] == list(CONV_CANDIDATE_ORDER)
    assert len(candidates) == 9
    assert (
        sum(parameter.numel() for _, parameter in candidates)
        == CONV_CANDIDATE_PARAM_COUNT
        == 12845056
    )
    assert [tuple(parameter.shape) for _, parameter in candidates] == [
        CONV_PROTOCOL_CANDIDATE_SHAPES[name]
        for name in CONV_CANDIDATE_ORDER
    ]
    names = {name for name, _ in candidates}
    assert all(".layer4." in name and name.endswith(".weight") for name in names)
    assert not any(token in name for name in names for token in ("bn", "bias", "netB", "netC"))


def _check_synthetic_partition(mode):
    value = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2)
    expected_groups = 2 if mode == "out_channel" else 6
    assert local_group_count(value, mode) == expected_groups
    view = group_view(value, mode)
    assert view.shape == (expected_groups, value.numel() // expected_groups)
    # reshape is a partition: every scalar appears once and no group aliases another.
    assert torch.equal(torch.sort(view.reshape(-1)).values, torch.arange(value.numel(), dtype=torch.float32))
    gamma = {"weight": torch.ones_like(value)}
    mask = group_support_masks(gamma, mode, support_threshold=1.0e-4)["weight"]
    assert mask.shape == value.shape
    assert selected_group_count({"weight": mask}, [("weight", value)], mode) == expected_groups
    if mode == "out_channel":
        assert torch.equal(mask, mask[:, :1, :1, :1].expand_as(mask))
    else:
        assert torch.equal(mask, mask[:, :, :1, :1].expand_as(mask))


def _check_global_group_counts():
    candidates = _meta_candidates()
    assert group_count(candidates, "out_channel") == CONV_GROUP_COUNTS["out_channel"] == 9216
    assert group_count(candidates, "filter_connection") == CONV_GROUP_COUNTS["filter_connection"] == 6553600


class _Block(nn.Module):
    def __init__(self, conv_shapes):
        super().__init__()
        self.conv1 = nn.Conv2d(conv_shapes[0][1], conv_shapes[0][0], conv_shapes[0][2:], bias=False)
        self.conv2 = nn.Conv2d(conv_shapes[1][1], conv_shapes[1][0], conv_shapes[1][2:], bias=False)
        self.conv3 = nn.Conv2d(conv_shapes[2][1], conv_shapes[2][0], conv_shapes[2][2:], bias=False)
        self.bn = nn.BatchNorm2d(1)


class _ConvScopeF(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(1, 1, 1, bias=False)
        self.layer2 = nn.Conv2d(1, 1, 1, bias=False)
        self.layer3 = nn.Conv2d(1, 1, 1, bias=False)
        self.stem_bn = nn.BatchNorm2d(1)
        self.layer4 = nn.ModuleList([
            _Block([CONV_PROTOCOL_CANDIDATE_SHAPES[f"netF.layer4.{index}.conv{conv}.weight"] for conv in (1, 2, 3)])
            for index in range(3)
        ])


def _check_controlled_freeze():
    net_f = _ConvScopeF()
    net_b = nn.Sequential(nn.Linear(1, 1), nn.BatchNorm1d(1))
    net_c = nn.Linear(1, 1)
    config = {
        "variant": "conv_out_random",
        "group_mode": "out_channel",
        "requested_budget": 1.0 / 9216,
        "selection_seed": 2026,
        "optimization": {"lr": 0.01, "lr_decay1": 0.1, "lr_decay2": 1.0},
    }
    all_named = [
        (f"netF.{name}", value) for name, value in net_f.named_parameters()
    ]
    assert [name for name, _ in _collect_conv_candidates(all_named)] == list(CONV_CANDIDATE_ORDER)
    _, stats, _ = _configure_variant(config, net_f, net_b, net_c)
    allowed = set(CONV_CANDIDATE_ORDER)
    all_after = [
        (f"{prefix}.{name}", value)
        for prefix, module in (("netF", net_f), ("netB", net_b), ("netC", net_c))
        for name, value in module.named_parameters()
    ]
    assert {name for name, value in all_after if value.requires_grad} == allowed
    assert stats["candidate_scope_param_count"] == CONV_CANDIDATE_PARAM_COUNT
    assert stats["total_group_count"] == 9216
    for module in (net_f, net_b, net_c):
        for child in module.modules():
            if isinstance(child, nn.modules.batchnorm._BatchNorm):
                assert child.training is False


def main():
    _check_candidate_scope()
    _check_synthetic_partition("out_channel")
    _check_synthetic_partition("filter_connection")
    _check_global_group_counts()
    _check_controlled_freeze()
    print("conv group contracts passed")


if __name__ == "__main__":
    main()
