#!/usr/bin/env python3
"""CPU/synthetic protocol checks for Transformer Group Split-LBI."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformer.candidate_dense.config import candidate_parameter_names  # noqa: E402
from transformer.candidate_dense.model import EXPECTED_SHAPES  # noqa: E402
from transformer.group_lbi.config import (  # noqa: E402
    BUDGET_TO_K,
    FORMAL_BUDGETS,
    FORMAL_SEED,
    GROUP_SIZE,
    PROTOCOL_REVISION,
    TOTAL_GROUPS,
    _validate_frozen_fields,
    budget_group_count,
    budget_tag,
    load_config,
    parse_budgets,
    resolve_lbi_profile,
    resolve_transfer_config,
)
from transformer.group_lbi.aggregate import (  # noqa: E402
    _load_summary,
    _pooled_tuning_diagnostics,
    write_aggregate,
)
from transformer.group_lbi.engine import (  # noqa: E402
    GroupSplitLBIEngine,
    group_soft_threshold,
    masked_delta_initialization,
    old_state_split_lbi_update,
    omega_accumulation,
    pack_groups,
    strict_budget_action,
    support_from_gamma,
    unpack_groups,
)
from transformer.group_lbi.runner import (  # noqa: E402
    _checkpoint_paths,
    _empty_histories,
    _load_checkpoint,
    _remove_checkpoint,
    _save_checkpoint,
    _selection_summary,
    _truncate_online_metrics,
    _validate_histories,
)
from transformer.group_random.groups import STRUCTURAL_GROUPS, build_masks  # noqa: E402
from transformer.group_random.optimizer import (  # noqa: E402
    assert_off_mask_adam_state_zero,
    strict_masked_adamw_step,
)
from transformer.source_only.metrics import FixedClassMeter  # noqa: E402
from transformer.source_only.data import (  # noqa: E402
    FixedOrderSampler,
    MergeSingletonTailBatchSampler,
)


CONFIG_PATH = PROJECT_ROOT / "transformer" / "group_lbi" / "config.yaml"


def check_config_profiles_budgets_and_overrides() -> None:
    config = load_config(CONFIG_PATH)
    _validate_frozen_fields(config)
    assert config["formal_seed"] == FORMAL_SEED == 2026
    assert config["protocol_revision"] == PROTOCOL_REVISION
    assert FORMAL_BUDGETS == (0.0005, 0.001, 0.002)
    assert BUDGET_TO_K == {0.0005: 3, 0.001: 6, 0.002: 13}
    assert [budget_group_count(value) for value in FORMAL_BUDGETS] == [3, 6, 13]
    assert parse_budgets("all") == FORMAL_BUDGETS
    assert parse_budgets("0.0005") == (0.0005,)
    assert budget_group_count(0.0005) == 3
    assert budget_tag(0.0005) == "rho-0.0005"
    assert budget_tag(0.01) == "rho-0.010"
    office = resolve_lbi_profile(config, "office31", 0.0005)
    visda = resolve_lbi_profile(config, "visda-c", 0.002)
    assert office["status"] == visda["status"] == "provisional_default"
    assert office["alpha"] == 0.1 and office["stage2_lr"] == 1.0e-5
    overridden = resolve_lbi_profile(config, "office31", 0.0005, {"alpha": 0.2})
    assert overridden["alpha"] == 0.2
    assert resolve_lbi_profile(config, "office31", 0.001)["alpha"] == 0.1
    custom = resolve_lbi_profile(config, "office31", 0.005)
    assert custom["profile_key"] == "office31/default"
    assert custom["requested_rho_key"] == "0.005"
    assert custom["alpha"] == 0.1 and custom["stage2_lr"] == 1.0e-5


def _candidate_tensors():
    values = []
    for name in candidate_parameter_names():
        suffix = name.split(".", maxsplit=2)[2]
        values.append((name, nn.Parameter(torch.zeros(EXPECTED_SHAPES[suffix]))))
    return values


def check_group_layout_prox_and_support() -> None:
    assert len(STRUCTURAL_GROUPS) == TOTAL_GROUPS == 6912
    candidates = _candidate_tensors()
    lookup = dict(candidates)
    with torch.no_grad():
        lookup["blocks.9.attn.qkv.weight"][0, 0] = 1.0
        lookup["blocks.9.attn.qkv.weight"][384, 1] = 2.0
        lookup["blocks.9.attn.qkv.weight"][768, 2] = 3.0
        lookup["blocks.9.attn.proj.weight"][3, 0] = 4.0
        lookup["blocks.9.mlp.fc1.weight"][0, 4] = 5.0
        lookup["blocks.9.mlp.fc2.weight"][5, 0] = 6.0
    packed = pack_groups({name: value for name, value in candidates})
    assert packed.shape == (TOTAL_GROUPS, GROUP_SIZE)
    assert packed[0, 0].item() == 1.0
    assert packed[0, 384 + 1].item() == 2.0
    assert packed[384, 2].item() == 3.0
    assert packed[384, 384 + 3].item() == 4.0
    assert packed[768, 4].item() == 5.0
    assert packed[768, 384 + 5].item() == 6.0
    unpacked = unpack_groups(packed)
    for name, value in candidates:
        assert torch.equal(unpacked[name], value.detach())

    z = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    gamma = group_soft_threshold(z, kappa=2.0, prox_lambda=1.0)
    assert torch.allclose(gamma[0], torch.tensor([4.8, 6.4]))
    assert torch.equal(gamma[1], torch.zeros(2))

    gamma_map = {name: torch.zeros_like(value) for name, value in candidates}
    tau = 1.0e-4
    gamma_map["blocks.9.attn.qkv.weight"][0, :].fill_(tau)
    gamma_map["blocks.9.attn.qkv.weight"][384, :].fill_(tau)
    ids, normalized = support_from_gamma(gamma_map, tau_g=tau)
    assert ids == (0,)
    assert torch.isclose(normalized[0], torch.tensor(tau), rtol=1.0e-5)

    full_masks = build_masks(candidates, range(TOTAL_GROUPS))
    assert all(mask.all().item() for mask in full_masks.values())
    assert sum(int(mask.count_nonzero()) for mask in full_masks.values()) == 5_308_416


def check_old_state_budget_masked_delta_and_omega() -> None:
    theta = {"x": torch.tensor([2.0, -1.0])}
    gamma = {"x": torch.tensor([0.5, 0.5])}
    z = {"x": torch.tensor([0.2, -0.3])}
    gradients = {"x": torch.tensor([3.0, 4.0])}
    new_theta, new_z = old_state_split_lbi_update(
        theta, gamma, z, gradients, alpha=0.2, kappa=1.5, nu=2.0
    )
    old_coupling = (theta["x"] - gamma["x"]) / 2.0
    assert torch.allclose(
        new_theta["x"], theta["x"] - 0.2 * 1.5 * (gradients["x"] + old_coupling)
    )
    assert torch.allclose(new_z["x"], z["x"] + 0.2 * old_coupling)

    assert strict_budget_action(33, 34) == "continue"
    assert strict_budget_action(34, 34) == "accept_and_stop"
    assert strict_budget_action(35, 34) == "rollback_and_stop"

    base = {"x": torch.tensor([1.0, 2.0, 3.0])}
    delta = {"x": torch.tensor([0.5, 0.5, 0.5])}
    masks = {"x": torch.tensor([True, False, True])}
    initialized = masked_delta_initialization(base, delta, masks)
    assert torch.equal(initialized["x"], torch.tensor([1.5, 2.0, 3.5]))
    applied = omega_accumulation(base, initialized, 0.2)
    assert torch.allclose(applied["x"], torch.tensor([1.1, 2.0, 3.1]))

    # Omega accumulation must be an exact no-op wherever Stage 2 left the
    # parameter unchanged.  A weighted-sum implementation can perturb equal
    # float32 operands by one or more ULPs.
    unchanged = {"x": torch.linspace(-3.14159, 2.71828, 4096)}
    unchanged_applied = omega_accumulation(unchanged, unchanged, 0.1)
    assert torch.equal(unchanged_applied["x"], unchanged["x"])

    first = GroupSplitLBIEngine.initialize([("x", nn.Parameter(torch.ones(2)))])
    first.theta_delta["x"].fill_(3.0)
    second = GroupSplitLBIEngine.initialize([("x", nn.Parameter(torch.ones(2)))])
    assert torch.equal(second.theta_delta["x"], torch.zeros(2))
    assert torch.equal(second.gamma["x"], torch.zeros(2))
    assert torch.equal(second.z["x"], torch.zeros(2))


def check_strict_sparse_adamw() -> None:
    parameter = nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    named = [("parameter", parameter)]
    optimizer = torch.optim.AdamW([parameter], lr=0.1, weight_decay=0.2)
    mask = torch.tensor([True, False, True, False])
    before = parameter.detach().clone()
    parameter.grad = torch.ones_like(parameter)
    strict_masked_adamw_step(optimizer, named, {"parameter": mask})
    assert torch.equal(parameter.detach()[~mask], before[~mask])
    assert not torch.equal(parameter.detach()[mask], before[mask])
    assert_off_mask_adam_state_zero(optimizer, named, {"parameter": mask})


def check_end_to_end_engine_allows_below_k() -> None:
    candidates = _candidate_tensors()
    coefficient_masks = build_masks(candidates, [0, 384])
    coefficients = {
        name: -2.0 * coefficient_masks[name].to(dtype=parameter.dtype)
        for name, parameter in candidates
    }

    def closure():
        loss = sum(
            torch.sum(parameter * coefficients[name])
            for name, parameter in candidates
        )
        return loss, {
            "loss_cls": float(loss.detach().item()),
            "loss_ent": 0.0,
            "loss_div": 0.0,
            "pseudo_ratio": 1.0,
            "max_prob_mean": 1.0,
        }

    result = GroupSplitLBIEngine().run_batch(
        candidates,
        closure,
        {
            "alpha": 1.0,
            "kappa": 1.0,
            "nu": 1.0,
            "omega": 0.5,
            "prox_lambda": 1.0,
            "tau_g": 1.0e-4,
            "stage1_max_steps": 2,
            "stage2_lr": 1.0e-5,
            "requested_group_count": 34,
            "stage2_optimization": {
                "optimizer": "adamw",
                "betas": [0.9, 0.999],
                "eps": 1.0e-8,
                "weight_decay": 0.01,
                "steps": 1,
            },
            "delta_nonzero_tolerance": 1.0e-12,
        },
    )
    assert result.statistics["stage1_stop_reason"] == "max_steps_reached"
    assert result.statistics["realized_group_count"] == 2
    assert result.statistics["realized_group_count"] < result.statistics[
        "requested_group_count"
    ]
    assert result.statistics["stage1_rollback_used"] is False
    assert result.state.active_group_ids == (0, 384)
    for name, parameter in candidates:
        mask = result.state.masks[name]
        assert torch.equal(parameter.detach()[~mask], torch.zeros_like(parameter.detach()[~mask]))


def _write_resolver_assets(root: Path) -> dict:
    config = load_config(CONFIG_PATH)
    list_root = root / "lists"
    checkpoint_root = root / "checkpoints"
    dataset_root = list_root / "office31"
    dataset_root.mkdir(parents=True)
    target_list = dataset_root / "dslr_list.txt"
    target_list.write_text(
        "".join(f"/synthetic/office31/dslr/{index}.jpg {index}\n" for index in range(31)),
        encoding="utf-8",
    )
    (dataset_root / "class_to_idx.json").write_text(
        json.dumps(
            {
                "dataset": "office31",
                "class_to_idx": {f"class-{index}": index for index in range(31)},
                "domains": {"dslr": {"path": str(target_list.resolve())}},
            }
        ),
        encoding="utf-8",
    )
    checkpoint = checkpoint_root / "office31" / "amazon.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic-checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    checkpoint.with_suffix(".manifest.json").write_text(
        json.dumps({"checkpoint": {"path": str(checkpoint.resolve()), "sha256": digest}}),
        encoding="utf-8",
    )
    config["data"]["list_root"] = str(list_root)
    config["model"]["checkpoint_root"] = str(checkpoint_root)
    return config


def check_identity_changes_with_lbi_profile() -> None:
    with tempfile.TemporaryDirectory(prefix="group_lbi_identity_") as value:
        root = Path(value)
        config = _write_resolver_assets(root)
        first = resolve_transfer_config(
            config,
            project_root=PROJECT_ROOT,
            dataset="office31",
            source="amazon",
            target="dslr",
            budget=0.005,
            device="cpu",
            output_dir=root / "first",
        )
        second = resolve_transfer_config(
            config,
            project_root=PROJECT_ROOT,
            dataset="office31",
            source="amazon",
            target="dslr",
            budget=0.005,
            device="cpu",
            output_dir=root / "second",
            lbi_overrides={"alpha": 0.2},
        )
        engineering_only = resolve_transfer_config(
            config,
            project_root=PROJECT_ROOT,
            dataset="office31",
            source="amazon",
            target="dslr",
            budget=0.005,
            device="cpu",
            output_dir=root / "third",
            stream_checkpoint=False,
        )
        assert first["experiment_key"] != second["experiment_key"]
        assert first["scientific_config_sha256"] != second["scientific_config_sha256"]
        assert first["experiment_key"] == engineering_only["experiment_key"]


def check_checkpoint_roundtrip_and_metric_truncation() -> None:
    with tempfile.TemporaryDirectory(prefix="group_lbi_checkpoint_") as value:
        root = Path(value)
        parameter = nn.Parameter(torch.tensor([1.0, 2.0]))
        candidates = [("x", parameter)]
        meter = FixedClassMeter(2, ["zero", "one"])
        meter.update([0, 1], [0, 1])
        histories = _empty_histories()
        for key in histories:
            histories[key].append({} if key == "step_records" else 0.1)
        config = {"experiment_key": "key", "scientific_config_sha256": "hash"}
        _save_checkpoint(
            output_dir=root,
            config=config,
            candidates=candidates,
            completed_batches=1,
            pu_meter=meter,
            pu_indices=[0, 1],
            histories=histories,
            started_at_utc="start",
            prior_wall_runtime_sec=2.0,
        )
        parameter.data.zero_()
        payload = _load_checkpoint(root, config, candidates)
        assert torch.equal(parameter.detach(), torch.tensor([1.0, 2.0]))
        assert torch.equal(payload["pu_confusion"], torch.eye(2, dtype=torch.int64))
        _validate_histories(payload["histories"], 1)

        metrics = root / "metrics.jsonl"
        metrics.write_text(
            "\n".join(
                json.dumps({"event": "online_batch", "batch_index": index})
                for index in (1, 2)
            )
            + "\n"
            + json.dumps({"event": "fo_batch", "batch_index": 1})
            + "\n",
            encoding="utf-8",
        )
        _truncate_online_metrics(metrics, 1)
        records = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines()]
        assert records == [{"event": "online_batch", "batch_index": 1}]
        _remove_checkpoint(root)
        assert not _checkpoint_paths(root)["root"].exists()


def check_metrics_keep_tail_and_fixed_classes() -> None:
    office = FixedClassMeter(2, ["zero", "one"])
    office.update([0, 1], [0, 0])
    office.update([1], [1])  # size-one tail contributes one correct sample
    metrics = office.compute("office31")
    assert metrics["sample-count"] == 3
    assert metrics["correct"] == 2
    assert abs(metrics["Acc"] - 200.0 / 3.0) < 1.0e-12

    names = [f"class-{index}" for index in range(12)]
    visda = FixedClassMeter(12, names)
    labels = list(range(12)) + [0]
    predictions = list(range(12)) + [1]
    visda.update(labels, predictions)
    result = visda.compute("visda-c")
    assert len(result["Acc-per-class"]) == 12
    assert result["Acc"] == sum(result["Acc-per-class"]) / 12
    assert result["overall-Acc"] == 1200.0 / 13.0


def check_office_amazon_singleton_tail_batching() -> None:
    sampler = MergeSingletonTailBatchSampler(
        FixedOrderSampler(range(2817)), batch_size=64, drop_last=False
    )
    sizes = [len(batch) for batch in sampler]
    assert len(sampler) == len(sizes) == 44
    assert sizes[:-1] == [64] * 43
    assert sizes[-1] == 65
    assert sorted(index for batch in sampler for index in batch) == list(range(2817))

    ordinary = MergeSingletonTailBatchSampler(
        FixedOrderSampler(range(2816)), batch_size=64, drop_last=False
    )
    assert [len(batch) for batch in ordinary] == [64] * 44


def check_tuning_diagnostics() -> None:
    records = []
    for index, (selected, steps, rollback) in enumerate(
        ((8, 100, False), (9, 3000, False), (10, 200, True), (10, 3000, True))
    ):
        records.append(
            {
                "realized_group_count": selected,
                "requested_group_count": 10,
                "utilization": selected / 10.0,
                "stage1_steps_completed": steps,
                "stage1_rollback_used": rollback,
                "stage1_stop_reason": (
                    "strict_budget_rollback"
                    if rollback
                    else ("max_steps_reached" if steps == 3000 else "budget_reached")
                ),
                "selected_group_ids": list(range(selected)),
                "selected_by_kind": {"QK": selected, "VO": 0, "FFN": 0},
                "selected_by_block": {"9": selected, "10": 0, "11": 0},
                "batch": index,
            }
        )
    summary = _selection_summary(records, stage1_step_cap=3000)
    assert summary["online_batch_count"] == 4
    assert summary["average_selected_groups"] == 9.25
    assert summary["utilization_mean"] == 0.925
    assert summary["utilization_min"] == 0.8
    assert summary["utilization_ge_90_count"] == 3
    assert summary["utilization_ge_90_rate"] == 0.75
    assert summary["utilization_ge_95_count"] == 2
    assert summary["utilization_ge_95_rate"] == 0.5
    assert summary["stage1_3000_step_hit_rate"] == 0.5
    assert summary["stage1_step_cap_hit_rate"] == 0.5
    assert summary["stage1_steps_mean"] == 1575
    assert summary["stage1_steps_max"] == 3000
    assert summary["rollback_rate"] == 0.5
    assert summary["budget_violation_rate"] == 0.0
    assert summary["failure_rate"] == 0.0

    pooled = _pooled_tuning_diagnostics(
        [{"selection": summary}, {"selection": copy.deepcopy(summary)}]
    )
    assert pooled["condition_count"] == 2
    assert pooled["batch_count"] == 8
    assert pooled["average_utilization"] == 0.925
    assert pooled["utilization_ge_90_rate"] == 0.75
    assert pooled["stage1_step_cap_hit_rate"] == 0.5
    assert pooled["rollback_rate"] == 0.5

    with tempfile.TemporaryDirectory(prefix="group_lbi_legacy_summary_") as value:
        condition = Path(value)
        (condition / "summary.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "variant": "group_lbi",
                    "selection": {"requested_group_count": 10},
                    "lbi": {"stage1_max_steps": 3000},
                }
            ),
            encoding="utf-8",
        )
        with open(condition / "metrics.jsonl", "w", encoding="utf-8") as file_obj:
            for index, record in enumerate(records, start=1):
                file_obj.write(
                    json.dumps(
                        {
                            "event": "online_batch",
                            "batch_index": index,
                            **record,
                        }
                    )
                    + "\n"
                )
        legacy = _load_summary(condition / "summary.json")
        assert legacy["selection"]["utilization_ge_90_rate"] == 0.75
        assert legacy["selection"]["stage1_3000_step_hit_rate"] == 0.5
        assert legacy["selection"]["average_selected_groups"] == 9.25

    with tempfile.TemporaryDirectory(prefix="group_lbi_tuning_csv_") as value:
        root = Path(value)
        transfer = {
            "requested_budget": 0.005,
            "requested_group_count": 10,
            "dataset": "office31",
            "transfer": "amazon->dslr",
            "formal_seed": 2026,
            "primary_metric": "sample_overall_accuracy",
            "PU-Acc": 75.0,
            "FO-Acc": 74.0,
            "PU-overall-Acc": 75.0,
            "FO-overall-Acc": 74.0,
            "selection": summary,
            "output_dir": str(root / "condition"),
        }
        aggregate = {
            "status": "completed",
            "transfers": [transfer],
            "budgets": {
                "0.005": {
                    "requested_group_count": 10,
                    "office31": {"tuning-diagnostics-pooled": pooled},
                }
            },
        }
        write_aggregate(root, aggregate)
        rows = (root / "tuning_diagnostics.csv").read_text(
            encoding="utf-8"
        ).splitlines()
        assert len(rows) == 3
        assert "OFFICE_POOLED" in rows[2]


def main() -> None:
    check_config_profiles_budgets_and_overrides()
    check_group_layout_prox_and_support()
    check_old_state_budget_masked_delta_and_omega()
    check_strict_sparse_adamw()
    check_end_to_end_engine_allows_below_k()
    check_identity_changes_with_lbi_profile()
    check_checkpoint_roundtrip_and_metric_truncation()
    check_metrics_keep_tail_and_fixed_classes()
    check_office_amazon_singleton_tail_batching()
    check_tuning_diagnostics()
    print("Transformer Group-LBI tests passed")


if __name__ == "__main__":
    main()
