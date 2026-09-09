#!/usr/bin/env python3
"""NCTTA FC/Conv sparse-LBI implementation contracts."""

# ruff: noqa: E402

import copy
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

from core.lbi import SplitLBIEngine
from core.lbi.groups import selected_group_count
from experiment_identity import build_experiment_identity
from nctta_otta.config import resolve_effective_config
from nctta_otta.lbi import NCTTALBIExecutor
from nctta_otta.objective import NCTTAObjectiveError, nctta_loss_from_outputs
from nctta_otta.sparse import (
    build_saliency_masks,
    build_static_masks,
    masked_optimizer_step,
)
import nctta_otta.sparse_trainer as trainer
from protocol_constants import (
    CONV_GROUP_COUNTS,
    FC_CANDIDATE_PARAM_COUNT,
    FORMAL_BUDGETS,
    FORMAL_INTEGER_BUDGETS,
    NCTTA_LBI_IMPLEMENTATION_REVISION,
    NCTTA_LBI_PROTOCOL_REVISION,
)
from shot_otta.config import load_yaml

CONFIG = osp.join(
    PROJECT_DIR, "configs", "nctta_otta_sparse_lbi_debug_20260906_v1.yaml"
)


def config(variant, budget=0.001):
    value = load_yaml(CONFIG)
    value["variant"] = variant
    value["requested_budget"] = budget
    value["group_mode"] = None
    if not variant.endswith("_lbi"):
        value["lbi"] = None
    return resolve_effective_config(value, WORKSPACE_ROOT)


def engine_config(**updates):
    value = {
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
    value.update(updates)
    return value


def candidate_config_identity_contract():
    assert FC_CANDIDATE_PARAM_COUNT == 524544
    assert FORMAL_BUDGETS == (0.0005, 0.001, 0.002)
    assert (
        tuple(math.floor(r * 524544) for r in FORMAL_BUDGETS)
        == FORMAL_INTEGER_BUDGETS
        == (262, 524, 1049)
    )
    assert CONV_GROUP_COUNTS["out_channel"] == 9216
    assert tuple(math.floor(r * 9216) for r in FORMAL_BUDGETS) == (4, 9, 18)
    variants = (
        "nctta_fc_random",
        "nctta_fc_magnitude",
        "nctta_fc_saliency",
        "nctta_fc_lbi",
        "nctta_conv_out_random",
        "nctta_conv_out_magnitude",
        "nctta_conv_out_saliency",
        "nctta_conv_out_lbi",
    )
    for variant in variants:
        value = config(variant)
        scientific = value["scientific_config"]
        assert value["implementation_revision"] == NCTTA_LBI_IMPLEMENTATION_REVISION
        assert value["protocol_revision"] == NCTTA_LBI_PROTOCOL_REVISION
        assert scientific["integer_budget"] == (9 if "conv" in variant else 524)
        assert scientific["group_mode"] == (
            "out_channel" if "conv" in variant else None
        )
        assert scientific["nctta"]["nu"] == 5.0
        assert scientific["parent_nctta_baseline_protocol_revision"] == (
            "OTTA_NCTTA_BASELINE_PROTOCOL_20260906_v2"
        )
        assert scientific["official_nctta_upstream_commit"] == (
            "b4d442472a36af6b3f4d6e75f5138ca97d7d8eec"
        )
        assert scientific["variant_policy"]["selector_definition"]
        assert scientific["primary_metric_name"] == "sample_level_overall_accuracy"
        if variant.endswith("_lbi"):
            assert scientific["lbi"]["nu"] == 1.0
            assert scientific["persistent_writeback"] == "lbi_omega_only"
            assert scientific["host_optimizer_persistent_step"] is False
            assert value["lbi_runtime"]["stage1_max_steps"] == 3000
            assert value["lbi_runtime"]["stage2_steps"] == 1
            assert value["lbi_runtime"]["support_threshold"] == 1.0e-4
    # Namespace independence is checked in both directions after common
    # validation, without conflating either field with an unqualified nu.
    value = config("nctta_fc_lbi")
    first = copy.deepcopy(value)
    first["nctta"]["nu"] = 7.0
    assert build_experiment_identity(first)["scientific_config"]["lbi"]["nu"] == 1.0
    second = copy.deepcopy(value)
    second["lbi"]["nu"] = 3.0
    assert build_experiment_identity(second)["scientific_config"]["nctta"]["nu"] == 5.0
    assert (
        build_experiment_identity(first)["experiment_key"]
        != build_experiment_identity(second)["experiment_key"]
    )


def selector_and_masked_host_step_contract():
    weight = nn.Parameter(torch.arange(524288.0).reshape(256, 2048))
    bias = nn.Parameter(torch.arange(256.0))
    named = [("netB.bottleneck.weight", weight), ("netB.bottleneck.bias", bias)]
    random_masks = [
        build_static_masks("nctta_fc_random", named, 0.001, seed)
        for seed in (202600, 202601, 202602)
    ]
    flat = [torch.cat([m[name].reshape(-1) for name, _ in named]) for m in random_masks]
    assert all(int(x.count_nonzero()) == 524 for x in flat)
    assert len({x.nonzero().numpy().tobytes() for x in flat}) == 3
    magnitude = build_static_masks("nctta_fc_magnitude", named, 0.001)
    assert sum(int(x.count_nonzero()) for x in magnitude.values()) == 524
    conv = [("conv", nn.Parameter(torch.arange(9216.0).reshape(9216, 1, 1, 1)))]
    cmask = build_static_masks("nctta_conv_out_magnitude", conv, 0.002)
    assert selected_group_count(cmask, conv, "out_channel") == 18

    parameter = nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    parameter.grad = torch.tensor([4.0, 3.0, 2.0, 1.0])
    saliency = build_saliency_masks("nctta_fc_saliency", [("p", parameter)], 0.5)
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


def corrected_lbi_current_state_contract():
    parameter = nn.Parameter(torch.zeros(2))
    seen = []

    def closure():
        seen.append(parameter.detach().clone())
        return (parameter * torch.tensor([10.0, 0.0])).sum(), {"selected_count": 2}

    executor = NCTTALBIExecutor(2)
    result = executor.run_outer_batch([("p", parameter)], closure, engine_config())
    assert len(seen) == result.statistics["stage1_steps_completed"] + 1
    assert result.statistics["stage2_steps_completed"] == 1
    assert executor.support_discovery_count == executor.omega_writeback_count == 1
    assert executor.host_optimizer_persistent_step_count == 0
    assert result.statistics["host_optimizer_persistent_step_count"] == 0
    assert result.statistics["persistent_writeback"] == "lbi_omega_only"
    assert result.statistics["off_mask_exact_preservation"] is True
    assert (
        result.statistics["selected_param_count"]
        == result.statistics["stage1_support_count"]
    )
    assert len(seen) > 1 and any(not torch.equal(seen[0], item) for item in seen[1:])

    grouped = nn.Parameter(torch.zeros(2, 1, 1, 1))
    grouped_result = NCTTALBIExecutor(2).run_outer_batch(
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

    invalid = nn.Parameter(torch.zeros(1))
    try:
        NCTTALBIExecutor(1).run_outer_batch(
            [("invalid", invalid)],
            lambda: (invalid.sqrt().sum(), {"selected_count": 1}),
            engine_config(),
        )
    except NCTTAObjectiveError as error:
        assert error.diagnostics["objective_call_index"] == 1
        assert error.diagnostics["objective_diagnostics"]["selected_count"] == 1
    else:
        raise AssertionError("non-finite NCTTA-LBI gradients must fail loudly")

    old_state_parameter = nn.Parameter(torch.zeros(2))
    result = SplitLBIEngine(2).run_step(
        [("p", old_state_parameter)],
        lambda: (old_state_parameter * torch.tensor([10.0, 10.0])).sum(),
        engine_config(stage1_max_steps=1, requested_budget=1.0),
    )
    assert torch.equal(result.state.z["p"], torch.zeros(2))
    overshoot = nn.Parameter(torch.zeros(2))
    result = SplitLBIEngine(2).run_step(
        [("p", overshoot)], lambda: (overshoot * 10).sum(), engine_config()
    )
    assert result.statistics["stage1_stop_reason"] == "strict_budget_rollback"
    assert result.statistics["stage1_support_count"] <= 1


def objective_trace_diagnostics_contract():
    torch.manual_seed(7)
    features = torch.randn(4, 3, requires_grad=True)
    logits = torch.randn(4, 5, requires_grad=True)
    weight = torch.randn(5, 3)
    params = {
        "thre_ent": 100.0,
        "margin_ent": 2.0,
        "reweight_ent": 1.0,
        "nu": 5.0,
        "eta": 1.0,
        "scale": 5.0,
        "top_k": 3,
        "mix_prob_weight": 0.3,
    }
    result = nctta_loss_from_outputs(features, logits, weight, params)
    for key in (
        "topk_index_checksum",
        "q_dist_l2",
        "q_prob_l2",
        "hybrid_q_min",
        "hybrid_q_max",
        "hybrid_q_l2",
        "entropy_selected_index_checksum",
    ):
        assert key in result.diagnostics
    result.loss.backward()
    assert features.grad is not None and logits.grad is not None


def random_execution_and_ordering_contract():
    value = config("nctta_fc_random")
    value["output"]["root"] = tempfile.mkdtemp(prefix="nctta_random_contract_")
    original_run, original_root = trainer._run, trainer._output_root
    calls = []
    trainer._output_root = lambda _: value["output"]["root"]

    def fake_run(child_config, workspace_root, **kwargs):
        del child_config, workspace_root
        index = kwargs["random_mask_index"]
        seed = kwargs["random_child_mask_seed"]
        calls.append((index, seed))
        os.makedirs(kwargs["output_dir"], exist_ok=False)
        return {
            "PU-Acc": 60.0 + index,
            "FO-Acc": 50.0 + index,
            "method": "NCTTA",
            "variant": "nctta_fc_random",
            "integer_budget": 524,
            "requested_budget": 0.001,
            "candidate_track": "fc_scalar",
            "candidate_scope": "netB.bottleneck",
            "candidate_scope_param_count": 524544,
            "candidate_layer_names": [],
            "group_mode": None,
            "total_group_count": None,
            "bn_stats_policy": "frozen",
            "bn_stats_frozen": True,
            "persistent_writeback": "masked_host_optimizer",
            "nctta_namespace": {"nu": 5.0},
            "lbi_namespace": None,
            "runtime_comparable": False,
            "run_id": kwargs["run_id"],
            "output_dir": kwargs["output_dir"],
            "processed_outer_batches": 5,
            "objective_call_count": 5,
            "optimizer_step_count": 5,
            "host_optimizer_persistent_step_count": 5,
            "scope_preserved": True,
            "netC_byte_identical": True,
            "source_checkpoint_sha256": {"netF": "f", "netB": "b", "netC": "c"},
            "parent_nctta_baseline_protocol_revision": (
                "OTTA_NCTTA_BASELINE_PROTOCOL_20260906_v2"
            ),
            "official_nctta_upstream_commit": (
                "b4d442472a36af6b3f4d6e75f5138ca97d7d8eec"
            ),
            "selector_definition": "uniform_global_exact_budget_child_mask",
            "lbi_omega": None,
            "primary_metric_name": "sample_level_overall_accuracy",
            "visda_fixed_class_count": None,
            "debug_only": True,
        }

    trainer._run = fake_run
    try:
        summary = trainer._run_random_experiment(value, WORKSPACE_ROOT)
    finally:
        trainer._run, trainer._output_root = original_run, original_root
    assert calls == [(0, 202600), (1, 202601), (2, 202602)]
    assert summary["random_child_execution_count"] == 3
    assert summary["fresh_source_model_execution_count"] == 3
    assert summary["fresh_host_optimizer_execution_count"] == 3
    assert summary["processed_outer_batches_per_child"] == [5, 5, 5]
    assert summary["optimizer_step_count"] == 15
    assert summary["scope_preserved"] and summary["netC_byte_identical"]
    assert summary["debug_only"] is True and summary["formal"] is False
    sparse_source = inspect.getsource(trainer._run)
    assert '"debug_only": debug_limit is not None' in sparse_source
    random_source = inspect.getsource(trainer._run_random_experiment)
    assert 'all(bool(child.get("debug_only", False)) for child in children)' in random_source
    assert summary["mean_PU-Acc"] == 61.0 and summary["mean_FO-Acc"] == 51.0
    source = inspect.getsource(trainer._run)
    assert source.index("_should_skip_singleton_outer_batch") < source.index(
        "inputs = inputs.to(device)"
    )
    assert "labels" not in inspect.getsource(trainer._objective_closure)
    assert inspect.getsource(trainer._objective_closure).count("nctta_loss(") == 1


def main():
    candidate_config_identity_contract()
    selector_and_masked_host_step_contract()
    corrected_lbi_current_state_contract()
    objective_trace_diagnostics_contract()
    random_execution_and_ordering_contract()
    print("NCTTA FC/Conv sparse-LBI contracts passed")


if __name__ == "__main__":
    main()
