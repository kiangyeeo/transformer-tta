#!/usr/bin/env python3
"""Dedicated NCTTA P1 dense-baseline correctness contracts."""

import copy
import inspect
import json
import math
import os.path as osp
import sys
import tempfile
from unittest import mock

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
WORKSPACE_ROOT = osp.dirname(PROJECT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from experiment_identity import build_experiment_identity  # noqa: E402
from nctta_otta.config import (  # noqa: E402
    OFFICIAL_NCTTA_DEFAULTS,
    OFFICIAL_NATIVE_OPTIMIZER,
    resolve_effective_config,
)
from nctta_otta.objective import (  # noqa: E402
    NCTTAObjectiveError,
    effective_classifier_weight,
    nctta_loss_from_outputs,
)
import nctta_otta.trainer as nctta_trainer  # noqa: E402
from shot_otta.artifacts import write_initial_artifacts  # noqa: E402
from shot_otta.config import load_yaml  # noqa: E402
from shot_otta.data import (  # noqa: E402
    build_loaders as shot_build_loaders,
    build_order_record,
    resolve_target_order,
)
from shot_otta.models import (  # noqa: E402
    FeatureClassifier,
    load_source_models as shot_load_source_models,
)
from shot_otta.trainer import _evaluate  # noqa: E402


CONFIG_PATH = osp.join(
    PROJECT_DIR, "configs", "nctta_otta_p1_dense_baseline_20260906_v1.yaml"
)


class TinyF(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 4)
        self.bn = nn.BatchNorm1d(4)

    def forward(self, inputs):
        return self.bn(self.linear(inputs))


class TinyB(nn.Module):
    def __init__(self):
        super().__init__()
        self.bottleneck = nn.Linear(4, 2)
        self.bn = nn.BatchNorm1d(2)

    def forward(self, inputs):
        return self.bn(self.bottleneck(inputs))


class TinyC(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 3)

    def forward(self, inputs):
        return self.fc(inputs)


class TinyConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(1, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(1, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(1)


class TinyConvF(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(1, 1, 1)
        self.layer4 = nn.Sequential(
            TinyConvBlock(), TinyConvBlock(), TinyConvBlock()
        )

    def forward(self, inputs):
        outputs = self.stem(inputs)
        for block in self.layer4:
            outputs = block.conv3(block.conv2(block.conv1(outputs)))
            outputs = block.bn(outputs)
        return outputs.mean(dim=(2, 3)).repeat(1, 4)


def _effective(variant="nctta_full_dense"):
    config = load_yaml(CONFIG_PATH)
    config["variant"] = variant
    return resolve_effective_config(config, WORKSPACE_ROOT)


def _state(net_f, net_b, net_c):
    return {
        f"{prefix}.{name}": value.detach().clone()
        for prefix, model in (
            ("netF", net_f), ("netB", net_b), ("netC", net_c)
        )
        for name, value in model.state_dict().items()
    }


def _changed(before, after):
    return {name for name in before if not torch.equal(before[name], after[name])}


def _assert_nested_equal(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        assert torch.is_tensor(left) and torch.is_tensor(right)
        assert torch.equal(left, right)
    elif isinstance(left, dict) or isinstance(right, dict):
        assert isinstance(left, dict) and isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def _source_and_initial_logits_contract():
    assert nctta_trainer.load_source_models is shot_load_source_models
    with tempfile.TemporaryDirectory(prefix="nctta_source_contract_") as root:
        original = TinyF(), TinyB(), TinyC()
        paths = {
            "netF": osp.join(root, "source_F.pt"),
            "netB": osp.join(root, "source_B.pt"),
            "netC": osp.join(root, "source_C.pt"),
        }
        for key, model in zip(("netF", "netB", "netC"), original):
            torch.save(model.state_dict(), paths[key])
        config = {"source_checkpoint": {"resolved_dir": root}}

        def build_models(_config, device):
            return TinyF().to(device), TinyB().to(device), TinyC().to(device)

        with mock.patch("shot_otta.models.build_models", side_effect=build_models):
            shot_models, shot_paths = shot_load_source_models(config, "cpu")
            nctta_models, nctta_paths = nctta_trainer.load_source_models(
                config, "cpu"
            )
        assert shot_paths == nctta_paths == paths
        inputs = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        for model in (*shot_models, *nctta_models):
            model.eval()
        shot_logits = shot_models[2](shot_models[1](shot_models[0](inputs)))
        nctta_logits = nctta_models[2](
            nctta_models[1](nctta_models[0](inputs))
        )
        assert torch.equal(shot_logits, nctta_logits)


def _stream_singleton_and_leakage_contract():
    assert nctta_trainer.build_loaders is shot_build_loaders
    config = _effective()
    order_a, record_a = resolve_target_order(config, 129)
    order_b, record_b = resolve_target_order(config, 129)
    assert order_a == order_b and record_a == record_b
    boundaries = [order_a[start : start + 64] for start in range(0, 129, 64)]
    assert [len(batch) for batch in boundaries] == [64, 64, 1]
    assert nctta_trainer._should_skip_singleton_outer_batch(1)
    assert not nctta_trainer._should_skip_singleton_outer_batch(2)
    assert not nctta_trainer._should_skip_singleton_outer_batch(64)

    signature = inspect.signature(nctta_trainer.adapt_one_batch)
    assert "labels" not in signature.parameters
    assert "sample_indices" not in signature.parameters
    assert "future" not in signature.parameters
    source = inspect.getsource(nctta_trainer._run)
    adapt_position = source.index("diagnostics = adapt_one_batch(")
    labels_eval_position = source.index("labels_cpu = labels.cpu()")
    assert adapt_position < labels_eval_position
    assert "target_list" not in inspect.getsource(nctta_trainer.adapt_one_batch)


def _direct_reference(features, logits, weight, params):
    eps = 1.0e-8
    h_hat = features / features.norm(dim=1, keepdim=True)
    w_hat = weight / weight.norm(dim=1, keepdim=True)
    probs = torch.exp(logits) / torch.exp(logits).sum(dim=1, keepdim=True)
    sims = h_hat @ w_hat.t()
    dists = 1.0 - torch.clamp(sims, -1.0, 1.0)
    top_prob, top_idx = torch.topk(
        probs, min(params["top_k"], weight.shape[0]), dim=1
    )
    top_dist = torch.gather(dists, 1, top_idx)
    normalized_distance = (
        top_dist - top_dist.mean(dim=1, keepdim=True)
    ) / (top_dist.std(dim=1, keepdim=True, unbiased=False) + eps)
    q_dist = torch.softmax(-normalized_distance, dim=1)
    q_prob = torch.softmax(torch.log(top_prob + eps), dim=1)
    q = (1 - params["mix_prob_weight"]) * q_dist
    q = q + params["mix_prob_weight"] * q_prob + eps
    q = q / q.sum(dim=1, keepdim=True)
    exp_logits = torch.exp(sims / 1.0)
    nc = -torch.log(
        (q * torch.gather(exp_logits, 1, top_idx)).sum(dim=1)
        / exp_logits.sum(dim=1)
    )
    entropy = -(torch.softmax(logits, 1) * torch.log_softmax(logits, 1)).sum(1)
    predicted = logits.argmax(dim=1)
    distance = torch.norm(h_hat - w_hat[predicted], p=2, dim=1)
    coeff_ent = params["reweight_ent"] / torch.exp(
        entropy.detach() - params["margin_ent"]
    )
    coeff_dist = params["nu"] / (1 + params["eta"] * distance.detach())
    selected = entropy < params["thre_ent"]
    return ((entropy + params["scale"] * nc) * (coeff_ent + coeff_dist))[
        selected
    ].mean()


def _objective_numerical_contract():
    assert nctta_trainer.nctta_loss.__module__ == "nctta_otta.objective"
    torch.manual_seed(37)
    features = torch.randn(6, 5, dtype=torch.float64, requires_grad=True)
    logits = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(4, 5, dtype=torch.float64).detach()
    params = dict(OFFICIAL_NCTTA_DEFAULTS)
    params["top_k"] = 3
    result = nctta_loss_from_outputs(features, logits, weight, params)
    reference = _direct_reference(features, logits, weight, params)
    assert torch.allclose(result.loss, reference, atol=1.0e-12, rtol=1.0e-12)
    reference_grad = torch.autograd.grad(reference, (features, logits))
    result_grad = torch.autograd.grad(result.loss, (features, logits))
    for actual, expected in zip(result_grad, reference_grad):
        assert torch.allclose(actual, expected, atol=1.0e-11, rtol=1.0e-11)
    assert result.diagnostics["selected_count"] == 6
    assert result.diagnostics["classifier_weight_detached"]

    rejected = dict(params)
    rejected["thre_ent"] = -1.0
    try:
        nctta_loss_from_outputs(features, logits, weight, rejected)
    except NCTTAObjectiveError as error:
        assert error.diagnostics["selected_count"] == 0
        assert error.diagnostics["batch_size"] == 6
    else:
        raise AssertionError("empty entropy filter did not fail loudly")


def _feature_classifier_geometry_contract():
    torch.manual_seed(5)
    net_b = nn.Linear(7, 256)
    net_c = FeatureClassifier(31, bottleneck_dim=256, classifier_type="wn")
    features = net_b(torch.randn(4, 7))
    logits = net_c(features)
    weight = effective_classifier_weight(net_c, features.shape[1])
    assert features.shape == (4, 256)
    assert weight.shape == (31, 256)
    assert logits.shape == (4, 31)
    assert not weight.requires_grad
    assert weight.grad_fn is None
    try:
        effective_classifier_weight(net_c, 2048)
    except RuntimeError:
        pass
    else:
        raise AssertionError("2048-d avgpool feature was accepted")


def _run_tiny_step(variant, net_f, net_b, net_c, inputs):
    config = _effective(variant)
    expected_count = (
        sum(parameter.numel() for parameter in net_b.bottleneck.parameters())
        if variant == "nctta_fc_module_dense"
        else sum(
            parameter.numel()
            for name, parameter in net_f.named_parameters()
            if name.startswith("layer4.") and ".conv" in name
        )
        if variant == "nctta_conv_module_dense"
        else None
    )
    patches = []
    if variant == "nctta_fc_module_dense":
        patches.append(
            mock.patch.object(nctta_trainer, "FC_CANDIDATE_PARAM_COUNT", expected_count)
        )
    if variant == "nctta_conv_module_dense":
        patches.append(
            mock.patch.object(
                nctta_trainer, "CONV_CANDIDATE_PARAM_COUNT", expected_count
            )
        )
    for patcher in patches:
        patcher.start()
    try:
        groups, stats = nctta_trainer.configure_nctta_variant(
            config, net_f, net_b, net_c
        )
    finally:
        for patcher in reversed(patches):
            patcher.stop()
    probe_bn = nn.BatchNorm2d(1)
    probe_bn.track_running_stats = False
    probe_bn.running_mean = None
    probe_bn.running_var = None
    probe_bn.eval()
    probe_a = torch.tensor([0.0, 1.0, 1.0]).reshape(3, 1, 1, 1)
    probe_b = torch.tensor([0.0, 1.0, 3.0]).reshape(3, 1, 1, 1)
    assert not torch.equal(probe_bn(probe_a)[0], probe_bn(probe_b)[0])
    optimizer = nctta_trainer._build_variant_optimizer(config, groups)
    before = _state(net_f, net_b, net_c)
    with mock.patch.object(optimizer, "step", wraps=optimizer.step) as step:
        diagnostics = nctta_trainer.adapt_one_batch(
            config, inputs, net_f, net_b, net_c, optimizer, 1, 5
        )
        assert step.call_count == 1
    after = _state(net_f, net_b, net_c)
    assert diagnostics["objective_call_count"] == 1
    assert diagnostics["optimizer_step_count"] == 1
    assert diagnostics["scheduler_step_count"] == 1
    assert diagnostics["finite_gradients"]
    assert diagnostics["nonzero_gradient_count"] > 0
    assert math.isfinite(diagnostics["loss"])
    return config, optimizer, stats, before, after


def _scope_state_transition_contract():
    inputs = torch.arange(24, dtype=torch.float32).reshape(8, 3) / 10
    full_f, full_b, full_c = TinyF(), TinyB(), TinyC()
    _, _, full_stats, full_before, full_after = _run_tiny_step(
        "nctta_full_dense", full_f, full_b, full_c, inputs
    )
    full_changed = _changed(full_before, full_after)
    assert full_changed
    assert all(name.startswith(("netF.", "netB.")) for name in full_changed)
    assert not any(name.startswith("netC.") for name in full_changed)
    assert full_stats["bn_stats_policy"] == "adaptive"

    fc_f, fc_b, fc_c = TinyF(), TinyB(), TinyC()
    _, fc_optimizer, fc_stats, fc_before, fc_after = _run_tiny_step(
        "nctta_fc_module_dense", fc_f, fc_b, fc_c, inputs
    )
    fc_expected = {"netB.bottleneck.weight", "netB.bottleneck.bias"}
    assert set(fc_stats["candidate_layer_names"]) == fc_expected
    assert _changed(fc_before, fc_after) == fc_expected
    assert not any(
        module.training
        for model in (fc_f, fc_b, fc_c)
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    )
    assert fc_optimizer.state_dict()["state"]

    conv_f, conv_b, conv_c = TinyConvF(), TinyB(), TinyC()
    conv_inputs = torch.arange(128, dtype=torch.float32).reshape(8, 1, 4, 4) / 100
    _, _, conv_stats, conv_before, conv_after = _run_tiny_step(
        "nctta_conv_module_dense", conv_f, conv_b, conv_c, conv_inputs
    )
    conv_expected = set(conv_stats["candidate_layer_names"])
    assert len(conv_expected) == 9
    assert _changed(conv_before, conv_after) == conv_expected
    assert not any(
        module.training
        for model in (conv_f, conv_b, conv_c)
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    )



def _native_scope_snapshot_contract():
    torch.manual_seed(101)
    net_f, net_b, net_c = TinyConvF(), TinyB(), TinyC()
    inputs = torch.arange(128, dtype=torch.float32).reshape(8, 1, 4, 4) / 100
    config = _effective("nctta_native_norm")
    groups, stats = nctta_trainer.configure_nctta_variant(
        config, net_f, net_b, net_c
    )
    expected = {
        f"netF.layer4.{index}.bn.{parameter_name}"
        for index in range(3)
        for parameter_name in ("weight", "bias")
    }
    assert set(stats["candidate_layer_names"]) == expected
    assert stats["native_nctta_norm_only_scope"]
    assert stats["native_norm_fo_uses_batch_statistics"]
    manifest = stats["trainable_parameter_manifest"]
    assert manifest["tensor_count"] == len(expected) == 6
    assert manifest["scalar_count"] == 6
    assert {entry["name"] for entry in manifest["parameters"]} == expected
    assert {entry["module_type"] for entry in manifest["parameters"]} == {
        "BatchNorm2d"
    }
    trainable = {
        name
        for name, parameter in nctta_trainer._all_named_parameters(
            net_f, net_b, net_c
        )
        if parameter.requires_grad
    }
    assert trainable == expected
    assert not any(
        parameter.requires_grad for parameter in net_b.bn.parameters()
    )
    assert not net_b.bn.training
    assert net_b.bn.track_running_stats
    assert net_b.bn.running_mean is not None
    assert net_b.bn.running_var is not None
    assert not any(
        parameter.requires_grad
        for model in (net_f, net_b, net_c)
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
        for parameter in module.parameters(recurse=False)
    )
    for module in net_f.modules():
        if isinstance(module, nn.BatchNorm2d):
            assert module.training
            assert not module.track_running_stats
            assert module.running_mean is None
            assert module.running_var is None

    optimizer = nctta_trainer._build_variant_optimizer(config, groups)
    defaults = optimizer.defaults
    assert isinstance(optimizer, torch.optim.SGD)
    assert defaults["lr"] == OFFICIAL_NATIVE_OPTIMIZER["lr"]
    assert defaults["momentum"] == OFFICIAL_NATIVE_OPTIMIZER["momentum"]
    assert defaults["dampening"] == OFFICIAL_NATIVE_OPTIMIZER["dampening"]
    assert defaults["weight_decay"] == OFFICIAL_NATIVE_OPTIMIZER["weight_decay"]
    assert defaults["nesterov"] == OFFICIAL_NATIVE_OPTIMIZER["nesterov"]

    before = _state(net_f, net_b, net_c)
    diagnostics = nctta_trainer.adapt_one_batch(
        config, inputs, net_f, net_b, net_c, optimizer, 1, 5
    )
    after = _state(net_f, net_b, net_c)
    changed = _changed(before, after)
    assert changed
    assert changed <= expected
    assert set(diagnostics["actual_changed_tensor_names"]) == changed
    assert diagnostics["actual_changed_tensor_count"] == len(changed)
    assert 0 < diagnostics["actual_changed_scalar_count"] <= 6
    assert diagnostics["optimizer_step_count"] == 1
    assert diagnostics["scheduler_step_count"] == 0
    assert diagnostics["gradient_norm"] > 0
    assert not any(name.startswith("netB.") for name in changed)
    assert not any(name.startswith("netC.") for name in changed)
    assert not any("conv" in name or "bottleneck" in name for name in changed)

    optimizer_before = copy.deepcopy(optimizer.state_dict())
    before_pu = _state(net_f, net_b, net_c)
    nctta_trainer.read_only_post_update_forward(
        inputs, net_f, net_b, net_c, config["variant"]
    )
    assert not _changed(before_pu, _state(net_f, net_b, net_c))
    _assert_nested_equal(optimizer.state_dict(), optimizer_before)

    for model in (net_f, net_b, net_c):
        model.eval()
    assert all(
        not module.training
        for module in net_f.modules()
        if isinstance(module, nn.BatchNorm2d)
    )
    assert all(
        module.running_mean is None and module.running_var is None
        for module in net_f.modules()
        if isinstance(module, nn.BatchNorm2d)
    )
    dataset = TensorDataset(
        inputs, torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]), torch.arange(8)
    )
    _evaluate(
        DataLoader(dataset, batch_size=4),
        net_f, net_b, net_c, "cpu", "office"
    )
    assert not _changed(before_pu, _state(net_f, net_b, net_c))
    _assert_nested_equal(optimizer.state_dict(), optimizer_before)
def _read_only_pu_fo_contract():
    inputs = torch.arange(24, dtype=torch.float32).reshape(8, 3) / 10
    net_f, net_b, net_c = TinyF(), TinyB(), TinyC()
    config = _effective("nctta_full_dense")
    groups, _ = nctta_trainer.configure_nctta_variant(
        config, net_f, net_b, net_c
    )
    optimizer = nctta_trainer._build_variant_optimizer(config, groups)
    nctta_trainer.adapt_one_batch(
        config, inputs, net_f, net_b, net_c, optimizer, 1, 5
    )
    before = _state(net_f, net_b, net_c)
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    nctta_trainer.read_only_post_update_forward(
        inputs, net_f, net_b, net_c, config["variant"]
    )
    assert not _changed(before, _state(net_f, net_b, net_c))
    _assert_nested_equal(optimizer.state_dict(), optimizer_before)

    dataset = TensorDataset(
        inputs, torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]), torch.arange(8)
    )
    loader = DataLoader(dataset, batch_size=4)
    for model in (net_f, net_b, net_c):
        model.eval()
    _evaluate(loader, net_f, net_b, net_c, "cpu", "office")
    assert not _changed(before, _state(net_f, net_b, net_c))
    _assert_nested_equal(optimizer.state_dict(), optimizer_before)


def _protocol_file_existence_contract():
    protocol_path = osp.join(
        PROJECT_DIR,
        "protocol",
        "nctta-otta",
        "OTTA_NCTTA_BASELINE_PROTOCOL_20260906_v2.md",
    )
    assert osp.isfile(protocol_path), protocol_path


def _provenance_and_artifact_contract():
    config = _effective("nctta_conv_module_dense")
    identity = build_experiment_identity(config)
    scientific = identity["scientific_config"]
    assert scientific["method"] == "NCTTA"
    assert scientific["protocol_track"] == "nctta"
    assert scientific["variant"] == "nctta_conv_module_dense"
    assert scientific["protocol_revision"] == (
        "OTTA_NCTTA_BASELINE_PROTOCOL_20260906_v2"
    )
    assert scientific["implementation_revision"] == (
        "nctta_otta_baseline_20260906_v3"
    )
    assert scientific["source_checkpoint_revision"] == (
        "nips2026_shot_otta_uda_source_v1"
    )
    assert scientific["feature_source"] == "post_netB_classifier_input"
    assert scientific["classifier_reference"].endswith("weight.detach")
    assert scientific["ttab_runtime_imported"] is False
    assert scientific["nctta"] == OFFICIAL_NCTTA_DEFAULTS
    assert scientific["data"]["target_passes"] == 1

    native = _effective("nctta_native_norm")
    native_scientific = build_experiment_identity(native)["scientific_config"]
    assert native_scientific["variant"] == "nctta_native_norm"
    assert native_scientific["candidate_scope"] == (
        "official_nctta_normalization_selector"
    )
    assert native_scientific["optimization"] == OFFICIAL_NATIVE_OPTIMIZER
    assert native_scientific["optimizer_provenance"] == (
        "official_nctta_ttab_defaults"
    )
    assert native_scientific["variant_policy"][
        "netB_batchnorm1d_frozen"
    ]
    assert native_scientific["variant_policy"][
        "native_norm_fo_uses_batch_statistics"
    ]

    order, order_info = resolve_target_order(config, 8)
    record = build_order_record(order, order_info, list(range(8)), 2026)
    with tempfile.TemporaryDirectory(prefix="nctta_artifact_contract_") as root:
        artifacts = write_initial_artifacts(
            root,
            "contract",
            config,
            record,
            {"netF": "F", "netB": "B", "netC": "C"},
            {"candidate_scope": "netF.layer4_conv"},
            WORKSPACE_ROOT,
        )
        with open(artifacts["manifest"], "r", encoding="utf-8") as file_obj:
            manifest = json.load(file_obj)
        assert manifest["method"] == "NCTTA"
        assert manifest["variant"] == "nctta_conv_module_dense"
        assert manifest["experiment_config_sha256"] == identity[
            "experiment_config_sha256"
        ]
        assert manifest["reproducibility"]["dataloader_order"]["target"][
            "sha256"
        ]

    poisoned = copy.deepcopy(config)
    poisoned["method"] = "IST"
    try:
        build_experiment_identity(poisoned)
    except (KeyError, ValueError):
        pass
    else:
        raise AssertionError("identity fallback accepted mismatched method/variant")


def main():
    _source_and_initial_logits_contract()
    _stream_singleton_and_leakage_contract()
    _objective_numerical_contract()
    _feature_classifier_geometry_contract()
    _scope_state_transition_contract()
    _native_scope_snapshot_contract()
    _read_only_pu_fo_contract()
    _protocol_file_existence_contract()
    _provenance_and_artifact_contract()
    print("NCTTA-OTTA P1 contracts passed")


if __name__ == "__main__":
    main()
