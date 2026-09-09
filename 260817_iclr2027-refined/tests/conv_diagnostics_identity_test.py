#!/usr/bin/env python3
"""Conv config, scientific identity, and read-only evaluation contracts (P2 S-W)."""

import copy
import os.path as osp
import sys

import torch
import torch.nn as nn
import torch.optim as optim


PROJECT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
WORKSPACE_ROOT = osp.dirname(PROJECT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from experiment_identity import build_experiment_identity  # noqa: E402
from shot_otta.config import load_yaml, resolve_effective_config  # noqa: E402
from shot_otta.trainer import _post_update_forward  # noqa: E402


CONV_CONFIG = osp.join(PROJECT_DIR, "configs", "otta_conv_lbi_protocol_20260826_v1.yaml")
FC_CONFIG = osp.join(PROJECT_DIR, "configs", "otta_fc_lbi_protocol_20260817_v1.yaml")

FC_FROZEN_EXPERIMENT_SHA256 = {
    "source_only": "a8f849b849c040846dbd0167f6473aeda98e223065322ea8a74dbfd47f1bf0fa",
    "full_dense": "2db686b0ae127305847047b3fae70576566b06092cc553109f33935768eb5f50",
    "module_dense": "3ff8b401f4fc21bf37be7548aa0e482c4161e87ef27e3111750042fd5db2f078",
    "module_random": "322e812cbef1d483e37837048718a0537624fdca63b25c98671ab5c0a999494d",
    "module_magnitude": "cbaf72bbae7e7fd210b5901e567ba93cbcd3e19016a90872b8c110c49f2f4d73",
    "module_saliency": "335b535f7f6b9fd6af224390340ff0cadef2ef6052ac670e87269f7ce095047f",
    "module_lbi": "d2edc53b6b178a360b884602fc7866420efbef927bd99cf91b5a42a9bf17db87",
}


def _conv(variant="conv_out_lbi", budget=0.1, group_mode="out_channel", lbi=None):
    config = load_yaml(CONV_CONFIG)
    config.update({"variant": variant, "requested_budget": budget, "group_mode": group_mode})
    config["lbi"] = lbi
    return config


def _tuned_lbi(**changes):
    lbi = {"alpha": 0.1, "kappa": 1.0, "nu": 1.0, "omega": 0.2, "stage2_lr": 0.01}
    lbi.update(changes)
    return lbi


def _check_conv_identity():
    out = resolve_effective_config(_conv(lbi=_tuned_lbi()), WORKSPACE_ROOT)
    out_identity = build_experiment_identity(out)
    scientific = out_identity["scientific_config"]
    assert scientific["protocol_track"] == "conv"
    assert scientific["protocol_revision"] == out["conv_protocol_revision"]
    assert scientific["candidate_scope"] == "netF.layer4_conv"
    assert len(scientific["candidate_layer_names"]) == 9
    assert scientific["group_mode"] == "out_channel"
    assert scientific["group_semantics"] == "out_channel"
    assert scientific["requested_budget"] == 0.1
    for field in ("alpha", "kappa", "nu", "omega", "stage2_lr"):
        assert field in scientific["lbi"]
    filter_config = resolve_effective_config(
        _conv("conv_filter_lbi", 0.1, "filter_connection", _tuned_lbi()), WORKSPACE_ROOT
    )
    assert build_experiment_identity(filter_config)["experiment_config_sha256"] != out_identity["experiment_config_sha256"]
    changed_budget = copy.deepcopy(out)
    changed_budget["requested_budget"] = 0.2
    assert build_experiment_identity(changed_budget)["experiment_config_sha256"] != out_identity["experiment_config_sha256"]
    changed_lbi = copy.deepcopy(out)
    changed_lbi["lbi"]["alpha"] = 0.2
    assert build_experiment_identity(changed_lbi)["experiment_config_sha256"] != out_identity["experiment_config_sha256"]


def _check_fc_identity_regression():
    # Conv fields must not leak into the frozen FC scientific configuration.
    for variant, expected_sha256 in FC_FROZEN_EXPERIMENT_SHA256.items():
        config = load_yaml(FC_CONFIG)
        config.update({"variant": variant, "requested_budget": 0.001, "selection_seed": 2026})
        if variant == "module_lbi":
            config["lbi"] = {
                **_tuned_lbi(omega=0.1), "stage1_max_steps": 3000, "budget_tolerance": 1.0e-4,
                "stage2_steps": 1, "delta_nonzero_tolerance": 1.0e-12,
                "support_threshold": 1.0e-4,
            }
        effective = resolve_effective_config(config, WORKSPACE_ROOT)
        identity_before = build_experiment_identity(effective)
        assert identity_before["experiment_config_sha256"] == expected_sha256
        # Building a Conv identity has no stateful route into FC construction.
        resolve_effective_config(_conv(lbi=_tuned_lbi()), WORKSPACE_ROOT)
        identity_after = build_experiment_identity(effective)
        assert identity_before == identity_after
        assert identity_after["experiment_config_sha256"] == expected_sha256
        assert "protocol_track" not in identity_before["scientific_config"]
        assert identity_before["scientific_config"]["variant"] == variant


def _raises_config(config, text):
    try:
        resolve_effective_config(config, WORKSPACE_ROOT)
    except ValueError as error:
        assert text in str(error)
    else:
        raise AssertionError("expected fail-closed Conv config error")


def _check_fail_closed_config():
    missing_budget = _conv("conv_out_random", None, "out_channel")
    _raises_config(missing_budget, "explicit requested_budget")
    wrong_group = _conv("conv_out_random", 0.1, "filter_connection")
    _raises_config(wrong_group, "requires group_mode=out_channel")
    resolved = resolve_effective_config(_conv(lbi=_tuned_lbi()), WORKSPACE_ROOT)
    for key, value in {"support_threshold": 1.0e-4, "stage1_max_steps": 3000, "stage2_steps": 1, "delta_nonzero_tolerance": 1.0e-12}.items():
        assert resolved["lbi"][key] == value
    _raises_config(_conv(lbi=_tuned_lbi(stage2_steps=2)), "frozen stage2_steps=1")


def _equal_state_dict(left, right):
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key]), key


def _check_pu_read_only():
    net_f = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    net_b = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    net_c = nn.Linear(2, 2)
    for model in (net_f, net_b, net_c):
        model.eval()
    optimizer = optim.SGD(net_f.parameters(), lr=0.1, momentum=0.9)
    # Populate persistent optimizer state before the read-only PU forward.
    (net_f(torch.ones(2, 2)).sum()).backward()
    optimizer.step()
    optimizer.zero_grad()
    before_models = [copy.deepcopy(model.state_dict()) for model in (net_f, net_b, net_c)]
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    _post_update_forward(torch.randn(3, 2), net_f, net_b, net_c, preserve_bn_state=False)
    for before, model in zip(before_models, (net_f, net_b, net_c)):
        _equal_state_dict(before, model.state_dict())
    after_optimizer = optimizer.state_dict()
    assert before_optimizer["param_groups"] == after_optimizer["param_groups"]
    for key, value in before_optimizer["state"].items():
        for state_name, tensor in value.items():
            assert torch.equal(tensor, after_optimizer["state"][key][state_name])


def main():
    _check_conv_identity()
    _check_fc_identity_regression()
    _check_fail_closed_config()
    _check_pu_read_only()
    print("conv diagnostics and identity contracts passed")


if __name__ == "__main__":
    main()
