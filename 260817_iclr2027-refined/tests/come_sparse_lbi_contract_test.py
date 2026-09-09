#!/usr/bin/env python3
"""Canonical COME FC/Conv sparse-LBI implementation contracts."""

# ruff: noqa: E402

import inspect
import math
import os
import os.path as osp
import sys
import tempfile

import torch
import torch.nn as nn

PROJECT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
WORKSPACE_ROOT = osp.dirname(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

from come_otta.config import resolve_effective_config
from come_otta.lbi import COMELBIExecutor
from come_otta.objective import OFFICIAL_COME_COMMIT, come_loss
from come_otta.sparse import (
    LBI_VARIANTS,
    MAGNITUDE_VARIANTS,
    RANDOM_VARIANTS,
    SALIENCY_VARIANTS,
    SPARSE_VARIANTS,
    build_saliency_masks,
    build_static_masks,
    masked_optimizer_step,
)
import come_otta.sparse_trainer as trainer
from core.lbi import SplitLBIEngine
from core.lbi.groups import selected_group_count
from experiment_identity import build_experiment_identity
from protocol_constants import (
    COME_LBI_FROZEN_CONSTANTS,
    COME_LBI_IMPLEMENTATION_REVISION,
    COME_LBI_PROTOCOL_REVISION,
    CONV_CANDIDATE_PARAM_COUNT,
    CONV_GROUP_COUNTS,
    FC_CANDIDATE_PARAM_COUNT,
    FORMAL_BUDGETS,
    FORMAL_INTEGER_BUDGETS,
)
from shot_otta.candidates import CONV_CANDIDATE_ORDER, MODULE_CANDIDATE_ORDER
from shot_otta.config import load_yaml

SPARSE_CONFIG = osp.join(
    PROJECT_DIR, "configs", "come_otta_sparse_debug_20260907_v1.yaml"
)
FORMAL_CONFIG = osp.join(
    PROJECT_DIR, "configs", "come_otta_sparse_formal_20260907_v2.yaml"
)
LBI_CONFIG = osp.join(
    PROJECT_DIR, "configs", "come_otta_sparse_lbi_debug_20260907_v1.yaml"
)
VARIANTS = (
    "come_fc_random",
    "come_fc_magnitude",
    "come_fc_saliency",
    "come_fc_lbi",
    "come_conv_out_random",
    "come_conv_out_magnitude",
    "come_conv_out_saliency",
    "come_conv_out_lbi",
)


def config(variant, budget=0.001):
    value = load_yaml(LBI_CONFIG if variant in LBI_VARIANTS else SPARSE_CONFIG)
    value["variant"] = variant
    value["requested_budget"] = budget
    value["group_mode"] = None
    return resolve_effective_config(value, WORKSPACE_ROOT)


def engine_config(**updates):
    value = {
        "alpha": 1.0,
        "kappa": 1.0,
        "nu": 1.0,
        "omega": 1.0,
        "stage1_max_steps": 3,
        "budget_tolerance": 0.0,
        "stage2_lr": 0.01,
        "stage2_steps": 1,
        "delta_nonzero_tolerance": 0.0,
        "support_threshold": 1.0e-4,
        "requested_budget": 0.5,
    }
    value.update(updates)
    return value


def candidate_budget_identity_contract():
    assert SPARSE_VARIANTS == (
        RANDOM_VARIANTS | MAGNITUDE_VARIANTS | SALIENCY_VARIANTS | LBI_VARIANTS
    )
    assert FC_CANDIDATE_PARAM_COUNT == 524544
    assert CONV_CANDIDATE_PARAM_COUNT == 12845056
    assert len(MODULE_CANDIDATE_ORDER) == 2
    assert len(CONV_CANDIDATE_ORDER) == 9
    assert CONV_GROUP_COUNTS["out_channel"] == 9216
    assert FORMAL_BUDGETS == (0.0005, 0.001, 0.002)
    assert (
        tuple(math.floor(ratio * FC_CANDIDATE_PARAM_COUNT) for ratio in FORMAL_BUDGETS)
        == FORMAL_INTEGER_BUDGETS
        == (262, 524, 1049)
    )
    assert tuple(math.floor(ratio * 9216) for ratio in FORMAL_BUDGETS) == (4, 9, 18)
    assert COME_LBI_FROZEN_CONSTANTS == {
        "support_threshold": 1.0e-4,
        "stage1_max_steps": 3000,
        "stage2_steps": 1,
        "delta_nonzero_tolerance": 1.0e-12,
        "budget_tolerance": 0.0,
    }
    for variant in VARIANTS:
        for ratio, fc_budget, conv_budget in zip(
            FORMAL_BUDGETS, FORMAL_INTEGER_BUDGETS, (4, 9, 18)
        ):
            value = config(variant, ratio)
            scientific = value["scientific_config"]
            is_conv = "conv_out" in variant
            assert value["protocol_revision"] == COME_LBI_PROTOCOL_REVISION
            assert value["implementation_revision"] == COME_LBI_IMPLEMENTATION_REVISION
            assert scientific["integer_budget"] == (
                conv_budget if is_conv else fc_budget
            )
            assert scientific["candidate_tensor_count"] == (9 if is_conv else 2)
            assert scientific["candidate_scalar_count"] == (
                CONV_CANDIDATE_PARAM_COUNT if is_conv else FC_CANDIDATE_PARAM_COUNT
            )
            assert scientific["group_mode"] == ("out_channel" if is_conv else None)
            assert scientific["group_count"] == (9216 if is_conv else None)
            assert scientific["come"] == {
                "p": 2.0,
                "tau": 1.0,
                "class_count": 31,
                "opinion_eps": 1.0e-7,
                "norm_detach_semantics": "multiplicative_logit_norm_only",
            }
            assert (
                scientific["official_come_commit"]
                == OFFICIAL_COME_COMMIT
                == "409a19b71f62c765b1a5be62347a9455524ec176"
            )
            assert scientific["parent_come_baseline_protocol_revision"] == (
                "OTTA_COME_BASELINE_PROTOCOL_20260907_v1"
            )
            assert scientific["source_checkpoints"]
            assert scientific["target_stream"]
            if variant in LBI_VARIANTS:
                assert scientific["persistent_writeback"] == "lbi_omega_only"
                assert scientific["host_optimizer_persistent_step"] is False
                assert scientific["lbi"]["omega"] == 1.0
                assert scientific["lbi_frozen"]["support_threshold"] == 1.0e-4
            else:
                assert scientific["persistent_writeback"] == "masked_host_optimizer"
                assert scientific["host_optimizer_persistent_step"] is True
                assert scientific["lbi_omega"] is None
    fallback = config("come_fc_lbi")
    fallback.pop("implementation_revision")
    assert (
        build_experiment_identity(fallback)["scientific_config"][
            "implementation_revision"
        ]
        == COME_LBI_IMPLEMENTATION_REVISION
    )


def formal_non_lbi_resolution_contract():
    formal_variants = (
        "come_fc_random",
        "come_fc_magnitude",
        "come_fc_saliency",
        "come_conv_out_random",
        "come_conv_out_magnitude",
        "come_conv_out_saliency",
    )
    for variant in formal_variants:
        value = load_yaml(FORMAL_CONFIG)
        value["variant"] = variant
        value["group_mode"] = None
        resolved = resolve_effective_config(value, WORKSPACE_ROOT)
        assert resolved["formal_protocol"] is True
        assert resolved["runtime"]["debug_max_outer_batches"] is None
        assert resolved["output"]["save_model"] is False
        assert "runs_smoke" not in resolved["output"]["root"]
        assert resolved["implementation_revision"] == (
            "come_otta_sparse_lbi_20260908_v3"
        )
        assert resolved["scientific_config"]["implementation_revision"] == (
            "come_otta_sparse_lbi_20260908_v3"
        )

    invalid_boolean = load_yaml(FORMAL_CONFIG)
    invalid_boolean["formal_protocol"] = "true"
    try:
        resolve_effective_config(invalid_boolean, WORKSPACE_ROOT)
    except ValueError as error:
        assert "must be boolean" in str(error)
    else:
        raise AssertionError("COME sparse formal_protocol must remain boolean")

    debug_formal = load_yaml(FORMAL_CONFIG)
    debug_formal["runtime"]["debug_max_outer_batches"] = 1
    try:
        resolve_effective_config(debug_formal, WORKSPACE_ROOT)
    except ValueError as error:
        assert "rejects debug_max_outer_batches" in str(error)
    else:
        raise AssertionError("formal COME sparse must reject debug limits")

    save_model_formal = load_yaml(FORMAL_CONFIG)
    save_model_formal["output"]["save_model"] = True
    try:
        resolve_effective_config(save_model_formal, WORKSPACE_ROOT)
    except ValueError as error:
        assert "save_model=false" in str(error)
    else:
        raise AssertionError("formal COME sparse must require save_model=false")


def fail_closed_config_contract():
    value = load_yaml(SPARSE_CONFIG)
    value["variant"] = "come_conv_out_random"
    value["group_mode"] = "filter_connection"
    try:
        resolve_effective_config(value, WORKSPACE_ROOT)
    except ValueError as error:
        assert "out_channel only" in str(error)
    else:
        raise AssertionError("COME filter_connection must be inaccessible")

    value = load_yaml(LBI_CONFIG)
    value["formal_protocol"] = True
    try:
        resolve_effective_config(value, WORKSPACE_ROOT)
    except ValueError as error:
        assert "formal COME-LBI launch remains blocked until search protocol and tuned tuples are frozen" in str(error)
    else:
        raise AssertionError("COME-LBI formal launch must fail closed")

    for key in ("alpha", "kappa", "nu", "omega", "stage2_lr"):
        assert key in config("come_fc_lbi")["lbi"]
    assert "tau" not in config("come_fc_lbi")["lbi"]
    assert "support_threshold" not in config("come_fc_lbi")["come"]


def selector_and_masked_optimizer_contract():
    weight = nn.Parameter(torch.arange(524288.0).reshape(256, 2048))
    bias = nn.Parameter(torch.arange(256.0))
    named = [
        ("netB.bottleneck.weight", weight),
        ("netB.bottleneck.bias", bias),
    ]
    random_masks = [
        build_static_masks("come_fc_random", named, 0.001, seed)
        for seed in (202600, 202601, 202602)
    ]
    flat = [
        torch.cat([mask[name].reshape(-1) for name, _ in named])
        for mask in random_masks
    ]
    assert all(int(mask.count_nonzero()) == 524 for mask in flat)
    assert len({mask.nonzero().numpy().tobytes() for mask in flat}) == 3

    magnitude = build_static_masks("come_fc_magnitude", named, 0.001)
    assert sum(int(mask.count_nonzero()) for mask in magnitude.values()) == 524
    assert magnitude["netB.bottleneck.weight"].reshape(-1)[-1]
    assert not magnitude["netB.bottleneck.weight"].reshape(-1)[0]

    conv = [("conv", nn.Parameter(torch.arange(9216.0).reshape(9216, 1, 1, 1)))]
    conv_random = [
        build_static_masks("come_conv_out_random", conv, 0.001, seed)
        for seed in (202600, 202601, 202602)
    ]
    assert all(
        selected_group_count(mask, conv, "out_channel") == 9 for mask in conv_random
    )
    assert (
        len(
            {
                mask["conv"][:, 0, 0, 0].nonzero().numpy().tobytes()
                for mask in conv_random
            }
        )
        == 3
    )
    conv_magnitude = build_static_masks("come_conv_out_magnitude", conv, 0.002)
    assert selected_group_count(conv_magnitude, conv, "out_channel") == 18

    parameter = nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    parameter.grad = torch.tensor([4.0, 3.0, 2.0, 1.0])
    saliency = build_saliency_masks("come_fc_saliency", [("p", parameter)], 0.5)
    assert saliency["p"].tolist() == [False, True, True, False]
    optimizer = torch.optim.SGD(
        [parameter], lr=0.1, momentum=0.9, weight_decay=0.1, nesterov=True
    )
    off_before = parameter.detach()[~saliency["p"]].clone()
    for _ in range(2):
        parameter.grad = torch.ones_like(parameter)
        masked_optimizer_step(optimizer, [("p", parameter)], saliency)
        assert torch.equal(parameter.detach()[~saliency["p"]], off_before)
        assert torch.equal(
            optimizer.state[parameter]["momentum_buffer"][~saliency["p"]],
            torch.zeros(2),
        )


class Scale(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0, 2.0]))

    def forward(self, inputs):
        return inputs * self.weight


def current_state_saliency_contract():
    net_f, net_b, net_c = Scale(), nn.Identity(), nn.Identity()
    parameter = net_f.weight
    candidates = [("p", parameter)]
    optimizer = torch.optim.SGD(
        [{"params": parameter, "lr": 0.1}],
        lr=0.1,
        momentum=0.9,
        weight_decay=0.001,
        nesterov=True,
    )
    optimizer.param_groups[0]["lr0"] = 0.1
    value = {
        "variant": "come_fc_saliency",
        "requested_budget": 0.5,
        "model": {"class_num": 2},
        "come": {"p": 2.0, "tau": 1.0},
        "scientific_config": {"integer_budget": 1},
        "optimization": {
            "optimizer": "sgd",
            "momentum": 0.9,
            "nesterov": True,
            "lr_gamma": 10.0,
            "lr_power": 0.75,
        },
    }
    before = parameter.detach().clone()
    diagnostics = trainer.adapt_sparse_one_batch(
        value,
        torch.tensor([[1.0, 1.0], [2.0, -1.0]]),
        net_f,
        net_b,
        net_c,
        optimizer,
        candidates,
        {},
        None,
        1,
        10,
    )
    assert diagnostics["objective_call_count"] == 1
    assert diagnostics["saliency_support_selections"] == 1
    assert diagnostics["scheduler_step_count"] == 1
    assert diagnostics["optimizer_step_count"] == 1
    assert diagnostics["saliency_gradient_reused_without_state_change"] is True
    assert diagnostics["selected_scalar_count"] == 1
    assert int(torch.ne(parameter.detach(), before).count_nonzero()) <= 1


def corrected_lbi_contract():
    parameter = nn.Parameter(torch.tensor([0.2, -0.1]))
    seen = []

    def closure():
        seen.append(parameter.detach().clone())
        logits = torch.stack(
            (
                parameter + torch.tensor([1.0, 2.0]),
                parameter + torch.tensor([2.0, 1.0]),
            )
        )
        result = come_loss(logits, 2)
        return result.loss, result.diagnostics

    executor = COMELBIExecutor(2)
    result = executor.run_outer_batch(
        [("p", parameter)],
        closure,
        engine_config(alpha=1.0, requested_budget=0.5),
    )
    stats = result.statistics
    assert len(seen) == stats["stage1_steps_completed"] + 1
    assert stats["objective_call_count"] == len(seen)
    assert stats["stage2_steps_completed"] == 1
    assert stats["stage2_optimizer_instance_count"] == 1
    assert stats["stage2_masked_delta_init"] is True
    assert stats["stage2_objective_recomputed"] is True
    assert stats["persistent_writeback"] == "lbi_omega_only"
    assert stats["host_optimizer_persistent_step_count"] == 0
    assert stats["off_mask_exact_preservation"] is True
    assert stats["stage1_support_count"] <= stats["stage1_max_support_count"]
    assert executor.local_restart_count == executor.support_discovery_count == 1
    assert executor.omega_writeback_count == 1

    old_state_parameter = nn.Parameter(torch.zeros(2))
    old_state = SplitLBIEngine(2).run_step(
        [("p", old_state_parameter)],
        lambda: (old_state_parameter * torch.tensor([10.0, 10.0])).sum(),
        engine_config(stage1_max_steps=1, requested_budget=1.0),
    )
    assert torch.equal(old_state.state.z["p"], torch.zeros(2))

    overshoot_parameter = nn.Parameter(torch.zeros(2))
    overshoot = SplitLBIEngine(2).run_step(
        [("p", overshoot_parameter)],
        lambda: (overshoot_parameter * 10.0).sum(),
        engine_config(),
    )
    assert overshoot.statistics["stage1_stop_reason"] == "strict_budget_rollback"
    assert overshoot.statistics["stage1_support_count"] <= 1

    grouped = nn.Parameter(torch.zeros(2, 1, 1, 1))
    grouped_executor = COMELBIExecutor(2)
    grouped_result = grouped_executor.run_outer_batch(
        [("g", grouped)],
        lambda: (grouped * torch.tensor([10.0, 0.0]).reshape(2, 1, 1, 1)).sum(),
        engine_config(group_mode="out_channel"),
    )
    assert (
        grouped_result.statistics["selected_group_count"]
        == grouped_result.statistics["stage1_support_count"]
    )
    assert (
        grouped_result.statistics["selected_scalar_count"]
        == grouped_result.statistics["support_param_count"]
    )


def trainer_state_machine_source_contract():
    run_source = inspect.getsource(trainer._run)
    adapt_source = inspect.getsource(trainer.adapt_sparse_one_batch)
    closure_source = inspect.getsource(trainer._objective_closure)
    random_source = inspect.getsource(trainer._run_random_experiment)
    assert run_source.index("_should_skip_singleton_outer_batch") < run_source.index(
        "inputs = inputs.to(device)"
    )
    assert "labels" not in closure_source
    assert closure_source.count("come_loss(") == 1
    assert "build_static_masks(" in run_source
    assert run_source.index("build_static_masks(") < run_source.index("for outer_index")
    assert adapt_source.count("build_saliency_masks(") == 1
    assert "masked_optimizer_step(" in adapt_source
    assert "objective_calls != expected_objectives" in run_source
    assert "lbi_executor.omega_writeback_count != processed" in run_source
    assert "lbi_executor.local_restart_count != processed" in run_source
    assert "pu_guard = _read_only_guard" in run_source
    assert "fo_guard = _read_only_guard" in run_source
    assert "bn_state_byte_identical" in run_source
    assert "source_hashes = _source_checkpoint_hashes" in run_source
    assert "seeds = [202600, 202601, 202602]" in random_source
    assert "same_target_stream_verified" in random_source
    assert "child_supports" in random_source


def random_three_child_contract():
    value = config("come_fc_random")
    original_run = trainer._run
    original_output_root = trainer._output_root
    calls = []
    with tempfile.TemporaryDirectory(prefix="come_random_contract_") as temp_dir:
        trainer._output_root = lambda _: temp_dir

        def fake_run(child_config, workspace_root, **kwargs):
            del workspace_root
            index = kwargs["random_mask_index"]
            seed = kwargs["random_child_mask_seed"]
            calls.append((index, seed))
            os.makedirs(kwargs["output_dir"], exist_ok=False)
            return {
                "PU-Acc": 60.0 + index,
                "FO-Acc": 50.0 + index,
                "method": "COME",
                "variant": "come_fc_random",
                "task": "otta",
                "dataset": "office",
                "source": 1,
                "target": 0,
                "source-target": "DA",
                "implementation_revision": COME_LBI_IMPLEMENTATION_REVISION,
                "protocol_revision": COME_LBI_PROTOCOL_REVISION,
                "source_checkpoint_revision": "nips2026_shot_otta_uda_source_v1",
                "experiment_key": "key",
                "experiment_config_sha256": "sha",
                "requested_budget": 0.001,
                "integer_budget": 524,
                "candidate_track": "fc_scalar",
                "candidate_scope": "netB.bottleneck",
                "candidate_tensor_count": 2,
                "candidate_scope_param_count": 524544,
                "candidate_layer_names": list(MODULE_CANDIDATE_ORDER),
                "group_mode": None,
                "total_group_count": None,
                "bn_stats_policy": "frozen",
                "bn_stats_frozen": True,
                "persistent_writeback": "masked_host_optimizer",
                "come_namespace": {"tau": 1.0},
                "lbi_namespace": None,
                "parent_come_baseline_protocol_revision": (
                    "OTTA_COME_BASELINE_PROTOCOL_20260907_v1"
                ),
                "come_lbi_protocol_revision": COME_LBI_PROTOCOL_REVISION,
                "official_come_commit": OFFICIAL_COME_COMMIT,
                "selector_definition": "uniform_global_exact_budget_child_mask",
                "lbi_omega": None,
                "primary_metric_name": "sample_level_overall_accuracy",
                "visda_fixed_class_count": None,
                "target_stream_identity": {"target_order_sha256": "same"},
                "processed_sample_stream_sha256": "same-processed",
                "run_id": kwargs["run_id"],
                "output_dir": kwargs["output_dir"],
                "processed_outer_batches": 5,
                "objective_call_count": 5,
                "optimizer_step_count": 5,
                "scheduler_step_count": 5,
                "host_optimizer_persistent_step_count": 5,
                "scope_preserved": True,
                "bn_state_byte_identical": True,
                "netC_byte_identical": True,
                "source_checkpoint_sha256": {
                    "netF": {"sha256": "f"},
                    "netB": {"sha256": "b"},
                    "netC": {"sha256": "c"},
                },
                "support_trajectory": [{"selected_scalar_count": 524}] * 5,
                "selected_param_count": 524,
                "selected_scalar_count": 524,
                "realized_scalar_ratio": 524 / 524544,
                "debug_only": not child_config["formal_protocol"],
                "formal": child_config["formal_protocol"],
                "formal_protocol": child_config["formal_protocol"],
                "runtime_comparable": False,
            }

        trainer._run = fake_run
        try:
            summary = trainer._run_random_experiment(value, WORKSPACE_ROOT)
            formal_value = load_yaml(FORMAL_CONFIG)
            formal_value["variant"] = "come_fc_random"
            formal_value["group_mode"] = None
            formal_value = resolve_effective_config(formal_value, WORKSPACE_ROOT)
            formal_summary = trainer._run_random_experiment(
                formal_value, WORKSPACE_ROOT
            )
        finally:
            trainer._run = original_run
            trainer._output_root = original_output_root
    assert calls == [(0, 202600), (1, 202601), (2, 202602)] * 2
    assert summary["random_child_execution_count"] == 3
    assert summary["fresh_source_model_execution_count"] == 3
    assert summary["fresh_host_optimizer_execution_count"] == 3
    assert summary["same_target_stream_verified"] is True
    assert summary["processed_outer_batches_per_child"] == [5, 5, 5]
    assert len(summary["child_paths"]) == len(summary["child_supports"]) == 3
    assert summary["optimizer_step_count"] == 15
    assert summary["scheduler_step_count"] == 15
    assert summary["scope_preserved"]
    assert summary["bn_state_byte_identical"]
    assert summary["netC_byte_identical"]
    assert summary["debug_only"] is True and summary["formal"] is False
    assert summary["formal_protocol"] is False
    assert formal_summary["debug_only"] is False
    assert formal_summary["formal"] is True
    assert formal_summary["formal_protocol"] is True
    assert all(child["formal"] is True for child in formal_summary["masks"])
    assert all(
        child["formal_protocol"] is True for child in formal_summary["masks"]
    )
    assert summary["mean_PU-Acc"] == 61.0
    assert summary["std_PU-Acc"] == math.sqrt(2.0 / 3.0)
    assert summary["mean_FO-Acc"] == 51.0
    assert summary["std_FO-Acc"] == math.sqrt(2.0 / 3.0)


def main():
    candidate_budget_identity_contract()
    formal_non_lbi_resolution_contract()
    fail_closed_config_contract()
    selector_and_masked_optimizer_contract()
    current_state_saliency_contract()
    corrected_lbi_contract()
    trainer_state_machine_source_contract()
    random_three_child_contract()
    print("COME FC/Conv sparse-LBI contracts passed")


if __name__ == "__main__":
    main()
