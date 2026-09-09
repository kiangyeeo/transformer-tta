#!/usr/bin/env python3
"""Formal correctness contracts for the controlled dense COME baseline."""

import copy
import inspect
import math
import os.path as osp
import sys
import tempfile
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
WORKSPACE_ROOT = osp.dirname(PROJECT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from come_otta.config import (  # noqa: E402
    COME_IMPLEMENTATION_REVISION,
    COME_PROTOCOL_REVISION,
    COME_VARIANTS,
    FORMAL_DATASET_SETTINGS,
    resolve_effective_config,
)
from come_otta.objective import (  # noqa: E402
    OFFICIAL_COME_COMMIT,
    come_loss,
    constrain_logits,
)
import come_otta.trainer as come_trainer  # noqa: E402
from shot_otta.config import load_yaml  # noqa: E402
from shot_otta.data import (  # noqa: E402
    build_loaders as shot_build_loaders,
    resolve_target_order,
)
from shot_otta.models import load_source_models as shot_load_source_models  # noqa: E402


CONFIG_PATH = osp.join(
    PROJECT_DIR, "configs", "come_fast_feasibility_20260907_v1.yaml"
)
FORMAL_CONFIG_PATH = osp.join(
    PROJECT_DIR, "configs", "come_baseline_protocol_20260907_v1.yaml"
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


def _effective(variant="come_full_dense"):
    config = load_yaml(CONFIG_PATH)
    config["variant"] = variant
    return resolve_effective_config(config, WORKSPACE_ROOT)


def _formal_effective(dataset, source, target, variant="come_full_dense"):
    config = load_yaml(FORMAL_CONFIG_PATH)
    config["data"].update(
        {"dataset": dataset, "source": source, "target": target}
    )
    config["variant"] = variant
    return resolve_effective_config(config, WORKSPACE_ROOT)


def _expect_error(error_type, function):
    try:
        function()
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def _state(net_f, net_b, net_c):
    return {
        f"{prefix}.{name}": value.detach().clone()
        for prefix, model in (("netF", net_f), ("netB", net_b), ("netC", net_c))
        for name, value in model.state_dict().items()
    }


def _changed(before, after):
    return {name for name in before if not torch.equal(before[name], after[name])}


def _nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_nested_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)):
        return (
            isinstance(right, type(left))
            and len(left) == len(right)
            and all(_nested_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def _direct_reference(logits, class_count):
    norm = torch.norm(logits, p=2, dim=-1, keepdim=True)
    constrained = logits / norm * norm.detach()
    evidence = torch.exp(constrained)
    strength = evidence.sum(dim=1, keepdim=True) + class_count
    belief = evidence / strength
    uncertainty = class_count / strength
    opinion = torch.cat((belief, uncertainty), dim=1) + 1.0e-7
    return -(opinion * torch.log(opinion)).sum(dim=1).mean()


def _official_objective_contract():
    assert OFFICIAL_COME_COMMIT == "409a19b71f62c765b1a5be62347a9455524ec176"
    torch.manual_seed(79)
    actual_logits = torch.randn(7, 31, dtype=torch.float64, requires_grad=True)
    reference_logits = actual_logits.detach().clone().requires_grad_(True)
    result = come_loss(actual_logits, 31)
    reference = _direct_reference(reference_logits, 31)
    assert torch.allclose(result.loss, reference, atol=1.0e-12, rtol=1.0e-12)
    actual_gradient = torch.autograd.grad(result.loss, actual_logits)[0]
    reference_gradient = torch.autograd.grad(reference, reference_logits)[0]
    assert torch.allclose(
        actual_gradient, reference_gradient, atol=1.0e-11, rtol=1.0e-11
    )
    assert result.diagnostics["class_count_k"] == 31
    assert result.diagnostics["p"] == 2.0
    assert result.diagnostics["tau"] == 1.0
    assert result.diagnostics["finite_loss"]
    opinion = come_trainer.subjective_opinion(actual_logits, 31)
    assert torch.allclose(
        opinion["entropy_input"],
        opinion["opinion"] + 1.0e-7,
        atol=0.0,
        rtol=0.0,
    )
    assert torch.allclose(
        opinion["belief"].sum(dim=1, keepdim=True) + opinion["uncertainty"],
        torch.ones_like(opinion["uncertainty"]),
        atol=1.0e-12,
        rtol=1.0e-12,
    )

    probe = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    constrained = constrain_logits(probe)
    assert torch.allclose(constrained, probe, atol=1.0e-14, rtol=1.0e-14)
    detached_gradient = torch.autograd.grad(constrained.square().sum(), probe)[0]
    no_detach_probe = probe.detach().clone().requires_grad_(True)
    norm = no_detach_probe.norm(p=2, dim=-1, keepdim=True)
    no_detach = no_detach_probe / norm * norm
    no_detach_gradient = torch.autograd.grad(
        no_detach.square().sum(), no_detach_probe
    )[0]
    assert not torch.allclose(detached_gradient, no_detach_gradient)


def _dataset_and_config_contract():
    for variant in COME_VARIANTS:
        config = _effective(variant)
        assert config["model"]["class_num"] == 31
        assert config["data"]["dataset"] == "office"
        assert (config["data"]["source"], config["data"]["target"]) == (1, 0)
        assert config["data"]["batch_size"] == 64
        assert config["data"]["workers"] == 4
        assert config["runtime"]["debug_max_outer_batches"] == 5
        assert config["formal_protocol"] is False
        assert config["protocol_track"] == "come_baseline"
        assert "feasibility" not in str(config["scientific_config"]).lower()
        assert "gate_5b" not in str(config["scientific_config"]).lower()
    assert COME_PROTOCOL_REVISION == "OTTA_COME_BASELINE_PROTOCOL_20260907_v1"
    assert COME_IMPLEMENTATION_REVISION == "come_otta_baseline_20260908_v2"
    assert FORMAL_DATASET_SETTINGS["office"]["class_count"] == 31
    assert FORMAL_DATASET_SETTINGS["VISDA-C"]["class_count"] == 12

    office_transfers = {
        _formal_effective("office", source, target)["task_name"]
        for source in range(3)
        for target in range(3)
        if source != target
    }
    assert office_transfers == {"AD", "AW", "DA", "DW", "WA", "WD"}
    visda = _formal_effective("VISDA-C", 0, 1)
    assert visda["model"]["class_num"] == 12
    assert visda["model"]["backbone"] == "resnet101"
    assert visda["data"]["batch_size"] == 256
    assert visda["data"]["workers"] == 4
    assert visda["optimization"]["lr"] == 0.001
    assert visda["scientific_config"]["primary_metric_name"] == (
        "fixed_12_class_mAcc"
    )
    assert visda["scientific_config"]["fixed_class_count"] == 12

    bad = load_yaml(FORMAL_CONFIG_PATH)
    bad["runtime"]["debug_max_outer_batches"] = 5
    _expect_error(
        ValueError, lambda: resolve_effective_config(bad, WORKSPACE_ROOT)
    )
    bad = load_yaml(FORMAL_CONFIG_PATH)
    bad["output"]["save_model"] = True
    _expect_error(
        ValueError, lambda: resolve_effective_config(bad, WORKSPACE_ROOT)
    )
    bad = load_yaml(FORMAL_CONFIG_PATH)
    bad["seed"] = 1
    _expect_error(
        ValueError, lambda: resolve_effective_config(bad, WORKSPACE_ROOT)
    )

    _expect_error(ValueError, lambda: come_loss(torch.randn(2, 30), 31))
    source = "\n".join(
        open(osp.join(PROJECT_DIR, "come_otta", name), encoding="utf-8").read()
        for name in ("__init__.py", "config.py", "objective.py", "trainer.py")
    )
    assert "1000" not in source


def _source_and_order_contract():
    assert come_trainer.load_source_models is shot_load_source_models
    assert come_trainer.build_loaders is shot_build_loaders
    config = _effective()
    order_a, record_a = resolve_target_order(config, 129)
    order_b, record_b = resolve_target_order(config, 129)
    assert order_a == order_b and record_a == record_b
    assert [len(order_a[start : start + 64]) for start in range(0, 129, 64)] == [
        64,
        64,
        1,
    ]
    identity = config["scientific_config"]
    required = {
        "protocol_revision",
        "implementation_revision",
        "official_come_commit",
        "source_checkpoint_revision",
        "source_checkpoints",
        "method",
        "variant",
        "dataset",
        "transfer",
        "backbone",
        "seed",
        "outer_batch_size",
        "target_stream",
        "singleton_outer_batch_policy",
        "candidate_scope",
        "candidate_scalar_count",
        "bn_semantics",
        "come",
        "optimizer",
        "primary_metric_name",
    }
    assert required <= set(identity)
    for record in identity["source_checkpoints"].values():
        assert osp.isabs(record["resolved_path"])
        assert len(record["sha256"]) == 64
    assert len(identity["target_stream"]["target_order_sha256"]) == 64
    assert identity["come"] == {
        "objective": "mean_entropy_of_subjective_opinion",
        "p": 2.0,
        "tau": 1.0,
        "class_count": 31,
        "opinion_eps": 1.0e-7,
        "norm_dim": -1,
        "norm_epsilon": None,
        "norm_detach_semantics": "multiplicative_logit_norm_only",
        "evidence": "exp_constrained_logits",
        "strength": "sum_evidence_plus_dataset_class_count",
        "belief": "evidence_div_strength",
        "uncertainty": "dataset_class_count_div_strength",
    }

    with tempfile.TemporaryDirectory(prefix="come_source_contract_") as root:
        original = TinyF(), TinyB(), TinyC()
        paths = {
            "netF": osp.join(root, "source_F.pt"),
            "netB": osp.join(root, "source_B.pt"),
            "netC": osp.join(root, "source_C.pt"),
        }
        for key, model in zip(("netF", "netB", "netC"), original):
            torch.save(model.state_dict(), paths[key])
        tiny_config = {"source_checkpoint": {"resolved_dir": root}}

        def build_models(_config, device):
            return TinyF().to(device), TinyB().to(device), TinyC().to(device)

        with mock.patch("shot_otta.models.build_models", side_effect=build_models):
            shot_models, shot_paths = shot_load_source_models(tiny_config, "cpu")
            come_models, come_paths = come_trainer.load_source_models(
                tiny_config, "cpu"
            )
        assert shot_paths == come_paths == paths
        inputs = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        for model in (*shot_models, *come_models):
            model.eval()
        shot_logits = shot_models[2](shot_models[1](shot_models[0](inputs)))
        come_logits = come_models[2](come_models[1](come_models[0](inputs)))
        assert torch.equal(shot_logits, come_logits)
        assert all(
            torch.equal(left, right)
            for left, right in zip(
                shot_models[2].state_dict().values(),
                come_models[2].state_dict().values(),
            )
        )


def _tiny_step(variant, net_f, net_b, net_c, inputs):
    config = _effective(variant)
    config["model"]["class_num"] = 3
    expected_count = (
        sum(parameter.numel() for parameter in net_b.bottleneck.parameters())
        if variant == "come_fc_module_dense"
        else sum(
            parameter.numel()
            for name, parameter in net_f.named_parameters()
            if name.startswith("layer4.") and ".conv" in name
        )
        if variant == "come_conv_module_dense"
        else None
    )
    patches = []
    if variant == "come_fc_module_dense":
        patches.append(
            mock.patch.object(come_trainer, "FC_CANDIDATE_PARAM_COUNT", expected_count)
        )
    if variant == "come_conv_module_dense":
        patches.append(
            mock.patch.object(
                come_trainer, "CONV_CANDIDATE_PARAM_COUNT", expected_count
            )
        )
    for patcher in patches:
        patcher.start()
    try:
        groups, stats = come_trainer.configure_come_variant(
            config, net_f, net_b, net_c
        )
    finally:
        for patcher in reversed(patches):
            patcher.stop()
    optimizer = come_trainer._build_optimizer(config, groups)
    before = _state(net_f, net_b, net_c)
    with (
        mock.patch.object(optimizer, "step", wraps=optimizer.step) as step,
        mock.patch.object(
            come_trainer, "come_loss", wraps=come_trainer.come_loss
        ) as objective,
        mock.patch.object(
            come_trainer,
            "_schedule_learning_rate",
            wraps=come_trainer._schedule_learning_rate,
        ) as scheduler,
    ):
        diagnostics = come_trainer.adapt_one_batch(
            config, inputs, net_f, net_b, net_c, optimizer, 1, 100
        )
    after = _state(net_f, net_b, net_c)
    assert step.call_count == objective.call_count == scheduler.call_count == 1
    assert diagnostics["objective_call_count"] == 1
    assert diagnostics["optimizer_step_count"] == 1
    assert diagnostics["scheduler_step_count"] == 1
    assert diagnostics["finite_gradients"]
    assert diagnostics["finite_parameter_update"]
    assert math.isfinite(diagnostics["loss"])
    assert math.isfinite(diagnostics["relative_parameter_update_norm"])
    expected_decay = (1.0 + 10.0 / 100.0) ** (-0.75)
    expected_multiplier = (
        0.1 if variant == "come_conv_module_dense" else 1.0
    )
    if variant == "come_full_dense":
        expected_multiplier = 0.1
    assert math.isclose(
        diagnostics["lr"],
        config["optimization"]["lr"] * expected_multiplier * expected_decay,
        rel_tol=1.0e-12,
    )
    return config, optimizer, stats, before, after


def _scope_and_bn_transition_contract():
    inputs = torch.arange(24, dtype=torch.float32).reshape(8, 3) / 10
    full_f, full_b, full_c = TinyF(), TinyB(), TinyC()
    _, _, full_stats, full_before, full_after = _tiny_step(
        "come_full_dense", full_f, full_b, full_c, inputs
    )
    full_changed = _changed(full_before, full_after)
    assert full_changed
    assert all(name.startswith(("netF.", "netB.")) for name in full_changed)
    assert not any(name.startswith("netC.") for name in full_changed)
    assert full_stats["bn_stats_policy"] == "adaptive"
    assert full_stats["candidate_tensor_count"] == sum(
        1 for _ in full_f.parameters()
    ) + sum(1 for _ in full_b.parameters())

    fc_f, fc_b, fc_c = TinyF(), TinyB(), TinyC()
    _, fc_optimizer, fc_stats, fc_before, fc_after = _tiny_step(
        "come_fc_module_dense", fc_f, fc_b, fc_c, inputs
    )
    fc_expected = {"netB.bottleneck.weight", "netB.bottleneck.bias"}
    assert set(fc_stats["candidate_layer_names"]) == fc_expected
    assert _changed(fc_before, fc_after) == fc_expected
    assert all(
        not module.training
        for model in (fc_f, fc_b, fc_c)
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    )
    assert all(
        torch.equal(fc_before[name], fc_after[name])
        for name in come_trainer._bn_state_names(fc_f, fc_b, fc_c)
    )

    guard = come_trainer._read_only_guard(fc_f, fc_b, fc_c)
    optimizer_before = copy.deepcopy(fc_optimizer.state_dict())
    come_trainer.read_only_post_update_forward(
        inputs, fc_f, fc_b, fc_c, "come_fc_module_dense"
    )
    come_trainer._assert_read_only(guard, fc_f, fc_b, fc_c)
    assert _nested_equal(fc_optimizer.state_dict(), optimizer_before)

    conv_f, conv_b, conv_c = TinyConvF(), TinyB(), TinyC()
    conv_inputs = torch.arange(128, dtype=torch.float32).reshape(8, 1, 4, 4) / 100
    _, _, conv_stats, conv_before, conv_after = _tiny_step(
        "come_conv_module_dense", conv_f, conv_b, conv_c, conv_inputs
    )
    conv_expected = set(conv_stats["candidate_layer_names"])
    assert len(conv_expected) == 9
    assert _changed(conv_before, conv_after) == conv_expected
    assert all(
        torch.equal(conv_before[name], conv_after[name])
        for name in come_trainer._bn_state_names(conv_f, conv_b, conv_c)
    )


def _singleton_labels_and_diagnostics_contract():
    assert come_trainer._should_skip_singleton_outer_batch(1)
    assert not come_trainer._should_skip_singleton_outer_batch(2)
    signature = inspect.signature(come_trainer.adapt_one_batch)
    assert "labels" not in signature.parameters
    assert "sample_indices" not in signature.parameters
    source = inspect.getsource(come_trainer._run)
    assert source.index("if _should_skip_singleton_outer_batch") < source.index(
        "adapt_one_batch("
    )
    assert source.index("adapt_one_batch(") < source.index("labels_cpu = labels.cpu()")
    assert "labels" not in inspect.getsource(come_loss)

    logits = torch.randn(9, 31, requires_grad=True)
    state_before = logits.detach().clone()
    diagnostics = come_trainer._prediction_diagnostics(logits, 31)
    assert torch.equal(logits.detach(), state_before)
    assert logits.grad is None
    assert set(diagnostics) == {
        "predicted_class_histogram",
        "predicted_class_count",
        "dominant_class_count",
        "dominant_class_ratio",
        "mean_softmax_entropy",
        "come_opinion_entropy",
        "come_mean_uncertainty_mass",
    }

    singleton_inputs = torch.randn(1, 3)
    singleton_models = TinyF(), TinyB(), TinyC()
    singleton_before = _state(*singleton_models)
    calls = {"objective": 0, "scheduler": 0, "optimizer": 0, "pu": 0}
    if not come_trainer._should_skip_singleton_outer_batch(
        singleton_inputs.size(0)
    ):
        calls = {key: value + 1 for key, value in calls.items()}
    assert calls == {"objective": 0, "scheduler": 0, "optimizer": 0, "pu": 0}
    assert _state(*singleton_models).keys() == singleton_before.keys()
    assert all(
        torch.equal(singleton_before[name], _state(*singleton_models)[name])
        for name in singleton_before
    )


def _metrics_and_read_only_contract():
    office = come_trainer._compute_dataset_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 0, 1]),
        "office",
        "PU",
    )
    assert office["PU-Acc"] == 50.0
    visda_labels = np.arange(12)
    visda_predictions = np.arange(12)
    visda_predictions[-1] = 0
    visda = come_trainer._compute_dataset_metrics(
        visda_labels, visda_predictions, "VISDA-C", "FO"
    )
    assert math.isclose(visda["FO-Acc"], 100.0 * 11.0 / 12.0)
    assert visda["FO-overall-Acc"] == 100.0 * 11.0 / 12.0
    assert len(visda["FO-Acc-per-class"]) == 12
    assert "FO-worst-class-name" in visda
    assert "FO-class-std" in visda

    net_f, net_b, net_c = TinyF(), TinyB(), TinyC()
    net_f.eval()
    net_b.eval()
    net_c.eval()
    dataset = TensorDataset(
        torch.randn(5, 3), torch.tensor([0, 1, 2, 0, 1]), torch.arange(5)
    )
    loader = DataLoader(dataset, batch_size=2)
    guard = come_trainer._read_only_guard(net_f, net_b, net_c)
    come_trainer._evaluate_with_diagnostics(
        loader, net_f, net_b, net_c, "cpu", "office", 3
    )
    come_trainer._assert_read_only(guard, net_f, net_b, net_c)


def _no_resume_contract():
    with mock.patch.object(come_trainer, "_run") as run:
        _expect_error(
            ValueError,
            lambda: come_trainer.run_experiment(
                {}, WORKSPACE_ROOT, resume_run_dir="/tmp/not-used"
            ),
        )
        _expect_error(
            ValueError,
            lambda: come_trainer.run_experiment(
                {}, WORKSPACE_ROOT, enable_stream_checkpoint=True
            ),
        )
        run.assert_not_called()


def _formal_candidate_count_contract():
    config = _effective("come_fc_module_dense")
    models, _ = come_trainer.load_source_models(config, "cpu")
    config["variant"] = "come_full_dense"
    _, full_stats = come_trainer.configure_come_variant(config, *models)
    assert full_stats["candidate_tensor_count"] == 163
    assert full_stats["candidate_scope_param_count"] == 24033088
    assert set(full_stats["candidate_layer_names"]) == {
        name
        for name, _ in come_trainer._all_named_parameters(*models)
        if name.startswith(("netF.", "netB."))
    }
    config["variant"] = "come_fc_module_dense"
    _, fc_stats = come_trainer.configure_come_variant(config, *models)
    assert fc_stats["candidate_tensor_count"] == 2
    assert fc_stats["candidate_scope_param_count"] == 524544
    assert set(fc_stats["candidate_layer_names"]) == {
        "netB.bottleneck.weight",
        "netB.bottleneck.bias",
    }
    config["variant"] = "come_conv_module_dense"
    _, conv_stats = come_trainer.configure_come_variant(config, *models)
    assert conv_stats["candidate_tensor_count"] == 9
    assert conv_stats["candidate_scope_param_count"] == 12845056
    assert len(conv_stats["candidate_layer_names"]) == 9


def main():
    _official_objective_contract()
    _dataset_and_config_contract()
    _source_and_order_contract()
    _scope_and_bn_transition_contract()
    _singleton_labels_and_diagnostics_contract()
    _metrics_and_read_only_contract()
    _no_resume_contract()
    _formal_candidate_count_contract()
    print("COME formal baseline contracts passed")


if __name__ == "__main__":
    main()
