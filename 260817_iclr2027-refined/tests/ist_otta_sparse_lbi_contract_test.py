#!/usr/bin/env python3
"""IST FC/Conv sparse-LBI implementation contracts."""

# ruff: noqa: E402

import copy
import inspect
import math
import os
import os.path as osp
import tempfile
import sys
import torch
import torch.nn as nn

PROJECT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
WORKSPACE_ROOT = osp.dirname(PROJECT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from core.lbi import SplitLBIEngine
from core.lbi.groups import group_count, selected_group_count
from experiment_identity import build_experiment_identity
from ist_otta.config import resolve_effective_config
from ist_otta.ema import OuterBatchEMA
from ist_otta.lbi import ISTLBIExecutor
from ist_otta.objective import FixedISTTask, FullObjectiveGradientAccumulator
from ist_otta.sparse import (
    build_saliency_masks,
    build_static_masks,
    masked_optimizer_step,
    restore_offmask_values,
)
import ist_otta.trainer as trainer
from protocol_constants import (
    CONV_CANDIDATE_PARAM_COUNT,
    CONV_GROUP_COUNTS,
    FC_CANDIDATE_PARAM_COUNT,
    FORMAL_BUDGETS,
    FORMAL_INTEGER_BUDGETS,
)
from shot_otta.candidates import CONV_CANDIDATE_SHAPES
from shot_otta.config import load_yaml


CONFIG_PATH = osp.join(
    PROJECT_DIR, "configs", "ist_otta_p1_baseline_20260905_v1.yaml"
)
LOSS = {
    "components": ["hard_ce", "soft_kl"],
    "hard_ce_weight": 1.0,
    "soft_kl_weight": 1.0,
}


def _config(variant, budget=None, formal=None):
    config = load_yaml(CONFIG_PATH)
    config["variant"] = variant
    if formal is not None:
        config["formal_protocol"] = formal
    if budget is not None:
        config["requested_budget"] = budget
    if variant.endswith("_random"):
        config["selection_seed"] = (
            111 if formal is False else 2026
        )
        config["num_random_masks"] = 3
    if variant.endswith("_lbi"):
        config["formal_protocol"] = False
        config["lbi"] = {
            "alpha": 0.2,
            "kappa": 1.0,
            "nu": 1.0,
            "omega": 0.5,
            "stage2_lr": 0.01,
        }
    return resolve_effective_config(config, WORKSPACE_ROOT)


def _engine_config(**updates):
    config = {
        "alpha": 1.0,
        "kappa": 1.0,
        "nu": 1.0,
        "omega": 0.5,
        "stage1_max_steps": 3,
        "budget_tolerance": 0.0,
        "stage2_lr": 0.01,
        "stage2_steps": 1,
        "delta_nonzero_tolerance": 0.0,
        "support_threshold": 1.0e-4,
        "requested_budget": 0.5,
    }
    config.update(updates)
    return config


def _state(module):
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
    }


def _candidate_and_budget_contract():
    assert FC_CANDIDATE_PARAM_COUNT == 256 * 2048 + 256 == 524544
    assert FORMAL_BUDGETS == (0.0005, 0.001, 0.002)
    assert tuple(
        math.floor(rho * FC_CANDIDATE_PARAM_COUNT)
        for rho in FORMAL_BUDGETS
    ) == FORMAL_INTEGER_BUDGETS == (262, 524, 1049)
    assert sum(math.prod(shape) for shape in CONV_CANDIDATE_SHAPES.values()) == (
        CONV_CANDIDATE_PARAM_COUNT
    ) == 12845056
    assert sum(shape[0] for shape in CONV_CANDIDATE_SHAPES.values()) == (
        CONV_GROUP_COUNTS["out_channel"]
    ) == 9216
    assert tuple(
        math.floor(rho * CONV_GROUP_COUNTS["out_channel"])
        for rho in FORMAL_BUDGETS
    ) == (4, 9, 18)


def _config_and_identity_contract():
    fc_family = (
        "ist_fc_module_dense", "ist_fc_random", "ist_fc_magnitude",
        "ist_fc_saliency", "ist_fc_lbi",
    )
    conv_family = (
        "ist_conv_module_dense", "ist_conv_out_random",
        "ist_conv_out_magnitude", "ist_conv_out_saliency",
        "ist_conv_out_lbi",
    )
    for variant in fc_family + conv_family:
        budget = None if variant.endswith("module_dense") else 0.001
        config = _config(variant, budget)
        scientific = build_experiment_identity(config)["scientific_config"]
        assert scientific["variant"] == variant
        if budget is not None:
            assert scientific["requested_budget"] == budget
            assert scientific["integer_budget"] == (
                9 if variant.startswith("ist_conv_") else 524
            )
        if variant.endswith("_lbi"):
            assert config["lbi_runtime"]["support_threshold"] == 1.0e-4
            assert config["lbi_runtime"]["stage1_max_steps"] == 3000
            assert config["lbi_runtime"]["stage2_steps"] == 1
            assert config["lbi_runtime"]["budget_tolerance"] == 0.0
            if variant.startswith("ist_conv_"):
                assert config["lbi_runtime"]["group_mode"] == "out_channel"
            else:
                assert "group_mode" not in config["lbi_runtime"]
            assert scientific["native_ist_ema"] is False
            assert scientific["persistent_writeback"] == "lbi_omega_only"
            assert set(scientific["lbi"]) == {
                "alpha", "kappa", "nu", "omega", "stage2_lr"
            }
        else:
            assert scientific["native_ist_ema"] is True
            assert scientific["persistent_writeback"] == "ist_native_ema"

    # Even a direct identity call must not silently fall back to the dense
    # baseline implementation revision for an IST sparse variant.
    raw_sparse = _config("ist_fc_random", 0.001, formal=False)
    raw_sparse.pop("implementation_revision", None)
    fallback = build_experiment_identity(raw_sparse)["scientific_config"]
    assert fallback["implementation_revision"].endswith("_v2")

    invalid = load_yaml(CONFIG_PATH)
    invalid.update({
        "variant": "ist_conv_out_random",
        "requested_budget": 0.001,
        "group_mode": "filter_connection",
    })
    try:
        resolve_effective_config(invalid, WORKSPACE_ROOT)
    except ValueError as error:
        assert "out_channel only" in str(error)
    else:
        raise AssertionError("IST accepted filter_connection")

    bad_random_seed = load_yaml(CONFIG_PATH)
    bad_random_seed.update({
        "variant": "ist_fc_random",
        "requested_budget": 0.001,
        "selection_seed": 999,
        "num_random_masks": 3,
    })
    try:
        resolve_effective_config(bad_random_seed, WORKSPACE_ROOT)
    except ValueError as error:
        assert "selection_seed == formal seed" in str(error)
    else:
        raise AssertionError("formal IST Random accepted a non-formal mask seed")

    formal_lbi = load_yaml(CONFIG_PATH)
    formal_lbi.update({
        "variant": "ist_fc_lbi",
        "requested_budget": 0.001,
        "lbi": {
            "alpha": 1.0, "kappa": 1.0, "nu": 1.0,
            "omega": 0.5, "stage2_lr": 0.01,
        },
    })
    try:
        resolve_effective_config(formal_lbi, WORKSPACE_ROOT)
    except ValueError as error:
        assert "formal IST-LBI launch is blocked" in str(error)
    else:
        raise AssertionError("unfrozen formal IST-LBI launch was accepted")


def _static_selector_contract():
    weight = nn.Parameter(torch.arange(524288, dtype=torch.float32).reshape(
        256, 2048
    ))
    bias = nn.Parameter(torch.arange(256, dtype=torch.float32))
    fc = [("netB.bottleneck.weight", weight), ("netB.bottleneck.bias", bias)]
    random_masks = [
        build_static_masks("ist_fc_random", fc, 0.001, seed)
        for seed in (701, 702, 703)
    ]
    flattened = [
        torch.cat([masks[name].reshape(-1) for name, _ in fc])
        for masks in random_masks
    ]
    assert all(int(mask.count_nonzero()) == 524 for mask in flattened)
    assert len({mask.nonzero().flatten().numpy().tobytes() for mask in flattened}) == 3
    repeated = build_static_masks("ist_fc_random", fc, 0.001, 701)
    assert all(torch.equal(random_masks[0][name], repeated[name]) for name, _ in fc)

    magnitude = build_static_masks("ist_fc_magnitude", fc, 0.001)
    selected_values = torch.cat([
        parameter.detach()[magnitude[name]]
        for name, parameter in fc
    ])
    assert int(selected_values.numel()) == 524
    assert float(selected_values.min()) == 523764.0

    conv = [("conv", nn.Parameter(torch.arange(9216.0).reshape(9216, 1, 1, 1)))]
    assert group_count(conv, "out_channel") == 9216
    conv_random = [
        build_static_masks("ist_conv_out_random", conv, 0.001, seed)
        for seed in (801, 802, 803)
    ]
    assert all(
        selected_group_count(mask, conv, "out_channel") == 9
        for mask in conv_random
    )
    conv_magnitude = build_static_masks(
        "ist_conv_out_magnitude", conv, 0.002
    )
    assert selected_group_count(conv_magnitude, conv, "out_channel") == 18
    assert int(conv_magnitude["conv"].count_nonzero()) == 18


def _full_objective_accumulation_contract():
    torch.manual_seed(2)
    model = nn.Linear(3, 2, bias=True)
    reference = copy.deepcopy(model)
    views = torch.randn(7, 3)
    hard = torch.tensor([0, 1, 1, 0, 1, 0, 1])
    soft = torch.softmax(torch.randn(7, 2), dim=1)
    task = FixedISTTask(views, hard, soft)
    hard_before = hard.clone()
    soft_before = soft.clone()
    named = list(model.named_parameters())
    accumulator = FullObjectiveGradientAccumulator(
        task, model, LOSS, chunk_size=3
    )
    accumulator(named)
    accumulated = {
        name: parameter.grad.detach().clone() for name, parameter in named
    }

    reference.zero_grad()
    logits = reference(views)
    hard_loss = torch.nn.functional.cross_entropy(logits, hard)
    soft_loss = torch.nn.functional.kl_div(
        torch.log_softmax(logits, dim=-1), soft, reduction="batchmean"
    )
    (hard_loss + soft_loss).backward()
    for name, parameter in reference.named_parameters():
        assert torch.allclose(
            accumulated[name], parameter.grad, rtol=1.0e-5, atol=1.0e-6
        )
    assert accumulator.call_count == 1
    assert accumulator.chunk_backward_count == 3
    assert accumulator.optimizer_step_count == 0
    assert accumulator.state_update_count == 0
    assert torch.equal(task.hard_targets, hard_before)
    assert torch.equal(task.soft_targets, soft_before)
    assert not hasattr(task, "labels")


def _old_state_z_and_strict_rollback_contract():
    parameter = nn.Parameter(torch.zeros(2))

    def loss():
        return (parameter * torch.tensor([10.0, 10.0])).sum()

    result = SplitLBIEngine(2).run_step(
        [("p", parameter)],
        loss,
        _engine_config(stage1_max_steps=1, requested_budget=1.0),
    )
    assert torch.equal(result.state.z["p"], torch.zeros(2))

    parameter = nn.Parameter(torch.zeros(2))

    def overshoot_loss():
        return (parameter * torch.tensor([10.0, 10.0])).sum()

    result = SplitLBIEngine(2).run_step(
        [("p", parameter)], overshoot_loss, _engine_config()
    )
    assert result.statistics["stage1_stop_reason"] == "strict_budget_rollback"
    assert result.statistics["stage1_rollback_used"] is True
    assert result.statistics["stage1_support_count"] == 0
    assert result.statistics["stage1_support_count"] <= 1


class _LinearGradientAccumulator:
    def __init__(self, gradient):
        self.gradient = gradient
        self.call_count = 0
        self.optimizer_step_count = 0
        self.state_update_count = 0

    def __call__(self, named_parameters):
        for _, parameter in named_parameters:
            parameter.grad = self.gradient.to(parameter).clone()
        self.call_count += 1
        return self.gradient.new_tensor(1.0), {}


def _lbi_stage2_writeback_contract():
    parameter = nn.Parameter(torch.tensor([2.0, -3.0]))
    base = parameter.detach().clone()
    accumulator = _LinearGradientAccumulator(torch.tensor([10.0, 0.0]))
    executor = ISTLBIExecutor(total_model_param_count=2)
    result = executor.run_outer_batch(
        [("p", parameter)], accumulator, _engine_config()
    )
    mask = result.state.mask["p"]
    assert mask.tolist() == [True, False]
    assert result.statistics["stage2_steps_completed"] == 1
    assert result.statistics["full_objective_gradient_accumulation"] is True
    assert torch.equal(parameter[~mask], base[~mask])
    assert torch.equal(
        result.refined_parameters["p"][~mask], base[~mask]
    )
    assert executor.support_discovery_count == 1
    assert executor.omega_writeback_count == 1
    assert executor.native_ema_commit_count == 0
    assert result.statistics["native_ist_ema_commit_count"] == 0
    assert result.statistics["persistent_writeback"] == "lbi_omega_only"
    assert accumulator.optimizer_step_count == 0
    assert accumulator.state_update_count == 0

    conv_parameter = nn.Parameter(torch.tensor([[[[2.0]]], [[[-3.0]]]]))
    conv_base = conv_parameter.detach().clone()
    conv_accumulator = _LinearGradientAccumulator(
        torch.tensor([[[[10.0]]], [[[0.0]]]])
    )
    conv_executor = ISTLBIExecutor(total_model_param_count=2)
    conv_result = conv_executor.run_outer_batch(
        [("conv", conv_parameter)],
        conv_accumulator,
        _engine_config(group_mode="out_channel"),
    )
    assert conv_result.statistics["total_group_count"] == 2
    assert conv_result.statistics["selected_group_count"] == 1
    assert conv_result.statistics["selected_scalar_count"] == 1
    conv_mask = conv_result.state.mask["conv"]
    assert torch.equal(conv_parameter[~conv_mask], conv_base[~conv_mask])


def _shared_engine_default_regression_contract():
    first = nn.Parameter(torch.tensor([0.3, -0.2]))
    second = nn.Parameter(first.detach().clone())

    def first_loss():
        return (first.square()).sum()

    def second_loss():
        return (second.square()).sum()

    config = _engine_config(
        alpha=0.2, stage1_max_steps=2, requested_budget=1.0
    )
    left = SplitLBIEngine(2).run_step([("p", first)], first_loss, config)
    right = SplitLBIEngine(2).run_step(
        [("p", second)], second_loss, config, gradient_accumulator=None
    )
    assert torch.equal(first, second)
    assert left.statistics == right.statistics
    for field in ("theta_delta", "gamma", "z", "mask"):
        left_map = getattr(left.state, field)
        right_map = getattr(right.state, field)
        assert torch.equal(left_map["p"], right_map["p"])


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


def _non_lbi_mask_ema_and_state_protection_contract():
    torch.manual_seed(4)
    net_f, net_b, net_c = TinyF(), TinyB(), nn.Linear(2, 2)
    for parameter in net_f.parameters():
        parameter.requires_grad = False
    for parameter in net_c.parameters():
        parameter.requires_grad = False
    candidates = [
        ("netB.bottleneck.weight", net_b.bottleneck.weight),
        ("netB.bottleneck.bias", net_b.bottleneck.bias),
    ]
    masks = {
        candidates[0][0]: torch.zeros_like(candidates[0][1], dtype=torch.bool),
        candidates[1][0]: torch.zeros_like(candidates[1][1], dtype=torch.bool),
    }
    masks[candidates[0][0]][0, 0] = True
    before_candidate = {
        name: parameter.detach().clone() for name, parameter in candidates
    }
    before_f = _state(net_f)
    before_c = _state(net_c)
    before_bn = {
        "f_mean": net_f.bn.running_mean.clone(),
        "f_var": net_f.bn.running_var.clone(),
        "b_mean": net_b.bn.running_mean.clone(),
        "b_var": net_b.bn.running_var.clone(),
    }
    config = {
        "variant": "ist_fc_random",
        "data": {"batch_size": 2},
        "ist": {"iters": 1},
        "loss": LOSS,
    }
    optimizer = torch.optim.SGD(
        [parameter for _, parameter in candidates],
        lr=0.1, momentum=0.9, weight_decay=0.001, nesterov=True,
    )
    views = torch.randn(4, 3)
    hard = torch.tensor([0, 1, 0, 1])
    soft = torch.softmax(torch.randn(4, 2), dim=1)
    ema = OuterBatchEMA(0.9)
    ema.begin_batch(0, (("netF", net_f), ("netB", net_b)))
    trainer._inner_self_training(
        config, views, hard, soft, net_f, net_b, net_c, optimizer,
        torch.Generator().manual_seed(9),
        candidate_parameters=candidates, fixed_masks=masks,
    )
    ema.commit(0, (("netF", net_f), ("netB", net_b)))
    restore_offmask_values(candidates, masks, before_candidate)
    assert ema.commit_count == 1
    for name, parameter in candidates:
        assert torch.equal(
            parameter.detach()[~masks[name]],
            before_candidate[name][~masks[name]],
        )
    assert _state(net_f).keys() == before_f.keys()
    for name in before_f:
        assert torch.equal(_state(net_f)[name], before_f[name])
    for name in before_c:
        assert torch.equal(_state(net_c)[name], before_c[name])
    assert torch.equal(net_f.bn.running_mean, before_bn["f_mean"])
    assert torch.equal(net_f.bn.running_var, before_bn["f_var"])
    assert torch.equal(net_b.bn.running_mean, before_bn["b_mean"])
    assert torch.equal(net_b.bn.running_var, before_bn["b_var"])


def _saliency_once_and_masked_momentum_contract():
    parameter = nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    parameter.grad = torch.tensor([4.0, 3.0, 2.0, 1.0])
    masks = build_saliency_masks(
        "ist_fc_saliency", [("p", parameter)], 0.5
    )
    assert masks["p"].tolist() == [False, True, True, False]
    fixed = masks["p"].clone()
    optimizer = torch.optim.SGD(
        [parameter], lr=0.1, momentum=0.9, weight_decay=0.1, nesterov=True
    )
    before_off = parameter.detach()[~fixed].clone()
    for _ in range(3):
        optimizer.zero_grad()
        parameter.grad = torch.ones_like(parameter)
        masked_optimizer_step(optimizer, [("p", parameter)], masks)
        assert torch.equal(masks["p"], fixed)
        assert torch.equal(parameter.detach()[~fixed], before_off)
        buffer = optimizer.state[parameter]["momentum_buffer"]
        assert torch.equal(buffer[~fixed], torch.zeros_like(buffer[~fixed]))



def _random_three_child_execution_contract():
    config = _config("ist_fc_random", 0.001, formal=False)
    config["output"]["root"] = tempfile.mkdtemp(prefix="ist_random_contract_")
    original_run = trainer._run
    original_root = trainer.experiment_output_root
    calls = []

    def fake_root(_config):
        return config["output"]["root"]

    def fake_run(
        child_config,
        workspace_root,
        run_id=None,
        output_dir=None,
        summary_filename="summary.json",
        random_mask_index=None,
        random_child_mask_seed=None,
    ):
        del child_config, workspace_root, summary_filename
        calls.append((random_mask_index, random_child_mask_seed, run_id))
        os.makedirs(output_dir, exist_ok=False)
        return {
            "PU-Acc": float(60 + random_mask_index),
            "FO-Acc": float(50 + random_mask_index),
            "online_compute_runtime_sec": 1.0 + random_mask_index,
            "fo_eval_runtime_sec": 0.5,
            "online_batch_runtime_mean_sec": 0.1,
            "runtime_comparable": True,
            "selected_param_count": 524,
            "selected_scalar_count": 524,
            "realized_scalar_ratio": 524 / 524544,
            "selection": "random",
            "candidate_track": "fc_scalar",
            "candidate_scope": "netB.bottleneck",
            "candidate_scope_param_count": 524544,
            "total_model_param_count": 1000000,
            "candidate_layer_names": [
                "netB.bottleneck.weight", "netB.bottleneck.bias"
            ],
            "group_mode": None,
            "total_group_count": None,
            "bn_stats_policy": "frozen",
            "bn_stats_frozen": True,
            "native_ist_ema": True,
            "persistent_writeback": "ist_native_ema",
            "mask_refresh_policy": "once_before_target_stream",
            "source_checkpoints": {},
            "source_checkpoint_sha256": {},
            "integer_budget": 524,
        }

    trainer.experiment_output_root = fake_root
    trainer._run = fake_run
    try:
        summary = trainer._run_random_experiment(config, WORKSPACE_ROOT)
    finally:
        trainer._run = original_run
        trainer.experiment_output_root = original_root

    expected_seeds = [
        config["selection_seed"] * 100 + index for index in range(3)
    ]
    assert [item[0] for item in calls] == [0, 1, 2]
    assert [item[1] for item in calls] == expected_seeds
    assert len({item[1] for item in calls}) == 3
    assert summary["mask_seeds"] == expected_seeds
    assert summary["num_random_masks"] == 3
    assert summary["mean_PU-Acc"] == 61.0
    assert summary["mean_FO-Acc"] == 51.0
    assert summary["PU-Acc"] == 61.0
    assert summary["FO-Acc"] == 51.0
    assert summary["online_compute_runtime_sec"] == 2.0
    assert summary["random_total_online_compute_runtime_sec"] == 6.0


def _ordering_singleton_and_label_boundary_contract():
    source = inspect.getsource(trainer._run)
    assert source.count("robust_plca(") == 1
    assert source.count("memory.commit(") == 1
    assert source.count("lbi_executor.run_outer_batch(") == 1
    assert source.index("_should_skip_singleton_outer_batch") < source.index(
        "materializer.materialize"
    )
    fixed_call = source[source.index("fixed_task = FixedISTTask"):source.index(
        "step_selection_stats = selection_stats"
    )]
    assert "batch.labels" not in fixed_call
    assert "labels_device" not in fixed_call
    assert "hard_targets" in fixed_call and "soft_targets" in fixed_call
    assert "labels_device = batch.labels.to(device)" in source
    assert source.index("labels_device = batch.labels.to(device)") > source.index(
        "lbi_executor.run_outer_batch"
    )
    assert "ema.commit(state_batch_index" in source
    assert "if config[\"variant\"] in LBI_VARIANTS" in source


def main():
    _candidate_and_budget_contract()
    _config_and_identity_contract()
    _static_selector_contract()
    _full_objective_accumulation_contract()
    _old_state_z_and_strict_rollback_contract()
    _lbi_stage2_writeback_contract()
    _shared_engine_default_regression_contract()
    _non_lbi_mask_ema_and_state_protection_contract()
    _saliency_once_and_masked_momentum_contract()
    _random_three_child_execution_contract()
    _ordering_singleton_and_label_boundary_contract()
    print("IST FC/Conv sparse-LBI contracts passed")


if __name__ == "__main__":
    main()
