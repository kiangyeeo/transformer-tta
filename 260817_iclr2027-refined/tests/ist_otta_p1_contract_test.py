#!/usr/bin/env python3
"""Targeted P1 IST contracts; intentionally not executed during P1."""

import copy
import inspect
import json
import os.path as osp
import sys
import tempfile
from unittest import mock

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset


PROJECT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
WORKSPACE_ROOT = osp.dirname(PROJECT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from experiment_identity import build_experiment_identity  # noqa: E402
from ist_otta.data import ISTViewMaterializer  # noqa: E402
from ist_otta.ema import OuterBatchEMA  # noqa: E402
from ist_otta.memory import CausalMemoryBank, MemorySnapshot  # noqa: E402
from ist_otta.plca import robust_plca  # noqa: E402
import ist_otta.data as ist_data_module  # noqa: E402
import ist_otta.trainer as ist_trainer  # noqa: E402
from shot_otta.artifacts import write_initial_artifacts  # noqa: E402
from ist_otta.config import resolve_effective_config  # noqa: E402
from shot_otta.config import load_yaml  # noqa: E402
from shot_otta.data import build_order_record, resolve_target_order  # noqa: E402
from shot_otta.models import load_source_models as shot_load_source_models  # noqa: E402
from shot_otta.trainer import _build_optimizer, _evaluate  # noqa: E402


CONFIG_PATH = osp.join(
    PROJECT_DIR, "configs", "ist_otta_p1_baseline_20260905_v1.yaml"
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
        pooled = outputs.mean(dim=(2, 3))
        return pooled.repeat(1, 4)


def _effective(variant="ist_full_dense"):
    config = load_yaml(CONFIG_PATH)
    config["variant"] = variant
    return resolve_effective_config(config, WORKSPACE_ROOT)


def _state(models):
    return {
        model_name: {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        for model_name, model in models
    }


def _assert_state_equal(left, right):
    assert left.keys() == right.keys()
    for model_name in left:
        assert left[model_name].keys() == right[model_name].keys()
        for name in left[model_name]:
            assert torch.equal(
                left[model_name][name], right[model_name][name]
            ), f"state changed: {model_name}.{name}"




def _assert_nested_equal(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        assert torch.is_tensor(left) and torch.is_tensor(right)
        assert torch.equal(left, right)
        return
    if isinstance(left, dict) or isinstance(right, dict):
        assert isinstance(left, dict) and isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
        return
    assert left == right


def _source_contract():
    # IST imports the exact common loader; there is no fallback source path.
    assert ist_trainer.load_source_models is shot_load_source_models
    with tempfile.TemporaryDirectory(prefix="ist_source_contract_") as root:
        original = (TinyF(), TinyB(), nn.Linear(2, 2))
        paths = {
            "netF": osp.join(root, "source_F.pt"),
            "netB": osp.join(root, "source_B.pt"),
            "netC": osp.join(root, "source_C.pt"),
        }
        for key, model in zip(("netF", "netB", "netC"), original):
            torch.save(model.state_dict(), paths[key])
        config = {"source_checkpoint": {"resolved_dir": root}}

        def build_models(_config, device):
            return TinyF().to(device), TinyB().to(device), nn.Linear(2, 2).to(device)

        with mock.patch("shot_otta.models.build_models", side_effect=build_models):
            shot_models, shot_paths = shot_load_source_models(config, "cpu")
            ist_models, ist_paths = ist_trainer.load_source_models(config, "cpu")
        assert shot_paths == ist_paths == paths
        inputs = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        for model in (*shot_models, *ist_models):
            model.eval()
        shot_logits = shot_models[2](shot_models[1](shot_models[0](inputs)))
        ist_logits = ist_models[2](ist_models[1](ist_models[0](inputs)))
        assert torch.equal(shot_logits, ist_logits)


def _stream_and_raw_augmentation_contract():
    config = _effective()
    order_a, record_a = resolve_target_order(config, 131)
    order_b, record_b = resolve_target_order(config, 131)
    assert order_a == order_b and record_a == record_b
    batch_size = config["data"]["batch_size"]
    boundaries = [order_a[i : i + batch_size] for i in range(0, 131, batch_size)]
    assert [len(batch) for batch in boundaries] == [64, 64, 3]
    order_record = build_order_record(
        order_a, record_a, list(range(131)), config["seed"]
    )
    assert len(order_record["target"]["sha256"]) == 64

    source = inspect.getsource(ist_data_module)
    assert "ToPILImage" not in source
    materializer = ISTViewMaterializer(config)
    raw = Image.new("RGB", (300, 280), color=(10, 20, 30))
    batch = materializer.materialize([raw], torch.tensor([1]), torch.tensor([7]))
    assert batch.reference_views.shape == (1, 3, 224, 224)
    assert batch.adaptation_views.shape == (8, 3, 224, 224)
    assert not hasattr(batch, "adaptation_labels")

    # One raw file read per sample; the 1+extend stochastic views are then
    # generated from the same decoded image without changing RNG semantics.
    with tempfile.TemporaryDirectory(prefix="ist_raw_read_contract_") as root:
        image_path = osp.join(root, "sample.png")
        raw.save(image_path)
        with mock.patch.object(
            ist_data_module, "rgb_loader", wraps=ist_data_module.rgb_loader
        ) as loader:
            materializer.materialize(
                [image_path], torch.tensor([1]), torch.tensor([8])
            )
            assert loader.call_count == 1
    try:
        materializer.materialize(
            [torch.zeros(3, 224, 224)], torch.tensor([1]), torch.tensor([7])
        )
    except TypeError:
        pass
    else:
        raise AssertionError("normalized tensor -> PIL path was accepted")


def _dense_reference_plca(current_features, soft_targets, memory, config):
    """Small dense reference for the official L2 PLCA equations."""
    features = current_features.detach().to(dtype=torch.float32)
    labels = soft_targets.detach().to(dtype=torch.float32)
    memory_count = int(memory.batch_ids.numel())
    if memory_count:
        features = torch.cat((memory.features.float(), features), dim=0)
        labels = torch.cat((memory.labels.float(), labels), dim=0)

    sample_count = int(features.shape[0])
    neighbor_count = min(int(config["k"]) + 2, sample_count)
    distances = torch.cdist(features, features, p=2).square()
    values, neighbors = torch.topk(
        distances, k=neighbor_count, dim=1, largest=False, sorted=True
    )
    row_maximum = values.amax(dim=1, keepdim=True).clamp_min(1.0e-8)
    relation = (1.0 - values / row_maximum).clamp_min(0.0)
    relation[:, 0] = 1.0
    relation = relation.pow(float(config["gamma"]))

    affinity = torch.zeros(sample_count, sample_count, dtype=features.dtype)
    rows = torch.arange(sample_count)[:, None].expand_as(neighbors)
    affinity.index_put_((rows.reshape(-1), neighbors.reshape(-1)), relation.reshape(-1), accumulate=True)
    affinity = affinity + affinity.T
    degree = affinity.sum(dim=1)
    inv_sqrt = degree.clamp_min(1.0e-8).rsqrt()
    normalized = inv_sqrt[:, None] * affinity * inv_sqrt[None, :]
    system = torch.eye(sample_count) - float(config["propagation_alpha"]) * normalized

    propagated = labels
    for _ in range(int(config["repeat"])):
        scores = []
        for class_index in range(propagated.shape[1]):
            rhs = propagated[:, class_index]
            rhs = rhs / (rhs.sum() + 1.0e-8)
            scores.append(torch.linalg.solve(system, rhs))
        scores = torch.stack(scores, dim=1)
        hard = scores.argmax(dim=1)
        propagated = torch.nn.functional.one_hot(
            hard, num_classes=labels.shape[1]
        ).to(dtype=features.dtype)
    return propagated[memory_count:].argmax(dim=1)


def _causal_memory_and_plca_contract():
    bank = CausalMemoryBank(max_len=10000)
    current_features = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    current_labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    bank.commit(0, current_features, current_labels)
    assert bank.commit_count == 1
    snapshot = bank.snapshot_for(1)
    assert snapshot.batch_ids.tolist() == [0, 0]
    try:
        bank.commit(0, current_features, current_labels)
    except RuntimeError:
        pass
    else:
        raise AssertionError("memory accepted a second commit for one batch")
    try:
        bank.commit(2, current_features, current_labels)
    except RuntimeError:
        pass
    else:
        raise AssertionError("memory accepted a future/out-of-order commit")

    plca_config = copy.deepcopy(_effective()["ist"]["plca"])
    result = robust_plca(
        torch.tensor([[0.1, 0.1], [0.9, 0.9]]),
        torch.tensor([[0.8, 0.2], [0.1, 0.9]]),
        snapshot,
        plca_config,
    )
    assert result.memory_sample_count == 2
    assert result.graph_sample_count == 4
    assert result.corrected_hard_labels.shape == (2,)

    # Numerical contract against a direct dense implementation of the
    # official L2 PLCA equations (K+2, gamma affinity, symmetric normalized
    # graph, alpha=.99 propagation). Small graphs converge within maxiter=20.
    reference_features = torch.tensor(
        [[0.05, 0.10], [0.15, 0.05], [0.90, 0.85], [0.80, 0.95]],
        dtype=torch.float32,
    )
    reference_soft = torch.tensor(
        [[0.90, 0.10], [0.75, 0.25], [0.15, 0.85], [0.20, 0.80]],
        dtype=torch.float32,
    )
    reference_memory = MemorySnapshot(
        features=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        labels=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        batch_ids=torch.tensor([0, 0]),
    )
    sparse_result = robust_plca(
        reference_features, reference_soft, reference_memory, plca_config
    )
    dense_labels = _dense_reference_plca(
        reference_features, reference_soft, reference_memory, plca_config
    )
    assert torch.equal(sparse_result.corrected_hard_labels.cpu(), dense_labels)


def _ema_contract():
    model = nn.Sequential(nn.Linear(2, 1), nn.BatchNorm1d(1))
    ema = OuterBatchEMA(0.9)
    old_weight = model[0].weight.detach().clone()
    ema.begin_batch(0, (("model", model),))
    assert ema.anchor["model"]["0.weight"].data_ptr() != model[0].weight.data_ptr()
    with torch.no_grad():
        model[0].weight.add_(1.0)
        model[1].num_batches_tracked.add_(1)
    ema.commit(0, (("model", model),))
    assert torch.allclose(model[0].weight, old_weight + 0.1)
    assert model[1].num_batches_tracked.item() == 1
    assert ema.commit_count == 1

    frozen = nn.Linear(2, 1)
    frozen.weight.requires_grad = False
    frozen_ema = OuterBatchEMA(0.9)
    frozen_ema.begin_batch(0, (("frozen", frozen),))
    frozen_weight = frozen.weight.detach().clone()
    with torch.no_grad():
        frozen.weight.add_(1.0)
    frozen_ema.commit(0, (("frozen", frozen),))
    # EMA must never write a frozen parameter outside the selected scope.
    assert torch.equal(frozen.weight, frozen_weight + 1.0)

    try:
        ema.commit(0, (("model", model),))
    except RuntimeError:
        pass
    else:
        raise AssertionError("EMA accepted a second commit for one batch")


def _variant_scope_contract():
    full_f, full_b, full_c = TinyF(), TinyB(), nn.Linear(2, 2)
    _, full_stats = ist_trainer.configure_ist_variant(
        _effective("ist_full_dense"), full_f, full_b, full_c
    )
    ist_trainer.set_inner_train_behavior(
        "ist_full_dense", full_f, full_b, full_c
    )
    assert full_stats["bn_stats_policy"] == "native_full_train"
    assert all(parameter.requires_grad for parameter in full_f.parameters())
    assert all(parameter.requires_grad for parameter in full_b.parameters())
    assert not any(parameter.requires_grad for parameter in full_c.parameters())
    assert all(
        module.training
        for model in (full_f, full_b)
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    )

    net_f, net_b, net_c = TinyF(), TinyB(), nn.Linear(2, 2)
    config = _effective("ist_fc_module_dense")
    groups, stats = ist_trainer.configure_ist_variant(
        config, net_f, net_b, net_c
    )
    assert stats["candidate_layer_names"] == [
        "netB.bottleneck.weight", "netB.bottleneck.bias"
    ]
    selected = {
        f"{prefix}.{name}"
        for prefix, model in (("netF", net_f), ("netB", net_b), ("netC", net_c))
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert selected == set(stats["candidate_layer_names"])
    ist_trainer.set_inner_train_behavior(config["variant"], net_f, net_b, net_c)
    assert not any(
        module.training
        for model in (net_f, net_b, net_c)
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    )

    before = _state((("netF", net_f), ("netB", net_b), ("netC", net_c)))
    optimizer = _build_optimizer(config, groups)
    inputs = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    optimizer.zero_grad()
    net_c(net_b(net_f(inputs))).sum().backward()
    optimizer.step()
    after = _state((("netF", net_f), ("netB", net_b), ("netC", net_c)))
    changed = {
        f"{model}.{name}"
        for model in before
        for name in before[model]
        if not torch.equal(before[model][name], after[model][name])
    }
    assert changed <= set(stats["candidate_layer_names"])
    assert before["netC"].keys() == after["netC"].keys()
    for name in before["netC"]:
        assert torch.equal(before["netC"][name], after["netC"][name])

    conv_f, conv_b, conv_c = TinyConvF(), TinyB(), nn.Linear(2, 2)
    expected_count = sum(
        parameter.numel()
        for name, parameter in conv_f.named_parameters()
        if name.startswith("layer4.") and ".conv" in name
    )
    with mock.patch.object(
        ist_trainer, "CONV_CANDIDATE_PARAM_COUNT", expected_count
    ):
        conv_config = _effective("ist_conv_module_dense")
        conv_groups, conv_stats = ist_trainer.configure_ist_variant(
            conv_config, conv_f, conv_b, conv_c
        )
    assert len(conv_stats["candidate_layer_names"]) == 9
    assert all(name.startswith("netF.layer4.") for name in conv_stats["candidate_layer_names"])
    ist_trainer.set_inner_train_behavior(
        "ist_conv_module_dense", conv_f, conv_b, conv_c
    )
    conv_before = _state(
        (("netF", conv_f), ("netB", conv_b), ("netC", conv_c))
    )
    conv_optimizer = _build_optimizer(conv_config, conv_groups)
    conv_inputs = torch.arange(64, dtype=torch.float32).reshape(4, 1, 4, 4)
    conv_optimizer.zero_grad()
    conv_c(conv_b(conv_f(conv_inputs))).sum().backward()
    conv_optimizer.step()
    conv_after = _state(
        (("netF", conv_f), ("netB", conv_b), ("netC", conv_c))
    )
    conv_changed = {
        f"{model}.{name}"
        for model in conv_before
        for name in conv_before[model]
        if not torch.equal(conv_before[model][name], conv_after[model][name])
    }
    assert conv_changed <= set(conv_stats["candidate_layer_names"])
    assert not any(
        module.training
        for model in (conv_f, conv_b, conv_c)
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    )


def _read_only_pu_fo_contract():
    net_f, net_b, net_c = TinyF(), TinyB(), nn.Linear(2, 2)
    for model in (net_f, net_b, net_c):
        model.train()
    config = _effective("ist_fc_module_dense")
    groups, _ = ist_trainer.configure_ist_variant(config, net_f, net_b, net_c)
    optimizer = _build_optimizer(config, groups)
    # Populate SGD momentum state before the read-only check; otherwise an
    # empty optimizer state is too weak a contract.
    ist_trainer.set_inner_train_behavior(config["variant"], net_f, net_b, net_c)
    inputs = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    optimizer.zero_grad()
    net_c(net_b(net_f(inputs))).sum().backward()
    optimizer.step()
    assert optimizer.state_dict()["state"]

    bank = CausalMemoryBank(10000)
    bank.commit(0, torch.ones(2, 2), torch.eye(2))
    ema = OuterBatchEMA(0.9)
    ema.begin_batch(0, (("netF", net_f), ("netB", net_b)))
    before = _state((("netF", net_f), ("netB", net_b), ("netC", net_c)))
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    memory_before = bank.snapshot_for(1)
    anchor_before = copy.deepcopy(ema.anchor)
    ist_trainer.read_only_forward(inputs, net_f, net_b, net_c)
    _assert_state_equal(
        before, _state((("netF", net_f), ("netB", net_b), ("netC", net_c)))
    )
    _assert_nested_equal(optimizer.state_dict(), optimizer_before)
    assert bank.commit_count == 1
    assert torch.equal(bank.snapshot_for(1).features, memory_before.features)
    for model_name in anchor_before:
        for name in anchor_before[model_name]:
            assert torch.equal(anchor_before[model_name][name], ema.anchor[model_name][name])

    dataset = TensorDataset(inputs, torch.tensor([0, 1, 0, 1]), torch.arange(4))
    loader = DataLoader(dataset, batch_size=2)
    for model in (net_f, net_b, net_c):
        model.eval()
    _evaluate(loader, net_f, net_b, net_c, "cpu", "office")
    _assert_state_equal(
        before, _state((("netF", net_f), ("netB", net_b), ("netC", net_c)))
    )
    _assert_nested_equal(optimizer.state_dict(), optimizer_before)
    assert bank.commit_count == 1
    assert torch.equal(bank.snapshot_for(1).features, memory_before.features)
    for model_name in anchor_before:
        for name in anchor_before[model_name]:
            assert torch.equal(
                anchor_before[model_name][name], ema.anchor[model_name][name]
            )



def _singleton_outer_batch_policy_contract():
    # Historical SHOT semantics: drop_last=False at the loader, but an outer
    # batch whose actual size is exactly 1 is skipped before adaptation/PU.
    assert ist_trainer._should_skip_singleton_outer_batch(1)
    assert not ist_trainer._should_skip_singleton_outer_batch(2)
    assert not ist_trainer._should_skip_singleton_outer_batch(64)
    assert not ist_trainer._should_skip_singleton_outer_batch(256)



def _skipped_singleton_state_index_contract():
    """Skipped raw batches must not create gaps in state-bearing IST commits."""
    processed = 0
    raw_batch_sizes = [64, 1, 64]
    state_indices = []
    for size in raw_batch_sizes:
        if ist_trainer._should_skip_singleton_outer_batch(size):
            continue
        state_indices.append(processed)
        processed += 1
    assert state_indices == [0, 1]

    source = inspect.getsource(ist_trainer._run)
    assert "state_batch_index = processed_outer_batches" in source
    assert "memory.snapshot_for(state_batch_index" in source
    assert "memory.commit(\n            state_batch_index" in source
    assert "ema.begin_batch(state_batch_index" in source
    assert "ema.commit(state_batch_index" in source

def _provenance_contract():
    config = _effective("ist_conv_module_dense")
    with tempfile.TemporaryDirectory(prefix="ist_hash_contract_") as root:
        paths = {}
        for name, payload in (("netF", b"F"), ("netB", b"B"), ("netC", b"C")):
            path = osp.join(root, name + ".pt")
            with open(path, "wb") as file_obj:
                file_obj.write(payload)
            paths[name] = path
        hashes = ist_trainer._source_checkpoint_hashes(paths)
        assert all(len(item["sha256"]) == 64 for item in hashes.values())

    identity = build_experiment_identity(config)
    scientific = identity["scientific_config"]
    assert scientific["method"] == "IST"
    assert scientific["variant"] == "ist_conv_module_dense"
    assert scientific["source_checkpoint_revision"] == "nips2026_shot_otta_uda_source_v1"
    assert scientific["ist"]["extend"] == 8
    assert scientific["data"]["target_passes"] == 1
    order, order_info = resolve_target_order(config, 8)
    order_record = build_order_record(order, order_info, list(range(8)), 2026)
    with tempfile.TemporaryDirectory(prefix="ist_artifact_contract_") as root:
        artifacts = write_initial_artifacts(
            root,
            "contract",
            config,
            order_record,
            {"netF": "F", "netB": "B", "netC": "C"},
            {"candidate_scope": "netF.layer4_conv"},
            WORKSPACE_ROOT,
        )
        with open(artifacts["manifest"], "r", encoding="utf-8") as file_obj:
            manifest = json.load(file_obj)
        assert manifest["method"] == "IST"
        assert manifest["variant"] == "ist_conv_module_dense"
        assert manifest["reproducibility"]["dataloader_order"]["target"]["sha256"]
        assert manifest["experiment_config_sha256"] == identity["experiment_config_sha256"]


def main():
    _source_contract()
    _stream_and_raw_augmentation_contract()
    _causal_memory_and_plca_contract()
    _ema_contract()
    _variant_scope_contract()
    _read_only_pu_fo_contract()
    _singleton_outer_batch_policy_contract()
    _skipped_singleton_state_index_contract()
    _provenance_contract()
    print("IST-OTTA P1 contracts passed")


if __name__ == "__main__":
    main()
