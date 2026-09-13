#!/usr/bin/env python3
"""CPU contracts for COME-Transformer Group Split-LBI integration."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(PROJECT_ROOT), str(PROJECT_ROOT / "tests")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from transformer.group_lbi.engine import GroupSplitLBIEngine  # noqa: E402
from transformer.group_lbi.runner import run_transfer as run_shared_lbi_transfer  # noqa: E402
from transformer_come.config import (  # noqa: E402
    GROUP_LBI,
    GROUP_LBI_PROTOCOL_DOCUMENT,
    IMPLEMENTATION_REVISIONS,
    PROTOCOL_REVISIONS,
    load_config,
    finalize_identity,
    resolve_lbi_profile,
    validate_variant_config,
    variant_config_view,
)
from transformer_come.lbi_runner import (  # noqa: E402
    come_lbi_objective,
    run_transfer as run_lbi_transfer,
)

from transformer_come_fixtures import (  # noqa: E402
    CandidateShapedModel,
    NUM_CLASSES,
    load_candidate_scope_model,
    build_synthetic_loaders,
    synthetic_config,
)


CONFIG_PATH = PROJECT_ROOT / "transformer_come" / "config.yaml"


def check_config_identity_and_formal_gate() -> dict:
    raw = load_config(CONFIG_PATH)
    view = variant_config_view(raw, GROUP_LBI)
    validate_variant_config(view, variant=GROUP_LBI)
    assert PROTOCOL_REVISIONS[GROUP_LBI] == (
        "come_transformer_group_lbi_otta_20260913_v1"
    )
    assert IMPLEMENTATION_REVISIONS[GROUP_LBI] == (
        "come_transformer_sparse_lbi_20260913_v1"
    )
    assert (PROJECT_ROOT / GROUP_LBI_PROTOCOL_DOCUMENT).is_file()
    assert view["selection"]["integer_budgets"] == [3, 6, 13]
    assert view["selection"]["support_measure"] == (
        "gamma_group_l2_div_sqrt_group_size"
    )
    assert view["selection"]["stage1_step_cap_hit_policy"] == (
        "continue_to_end_of_stream"
    )
    try:
        resolve_lbi_profile(raw, "office31", 0.0005)
    except ValueError as error:
        assert "formal run is blocked" in str(error)
    else:
        raise AssertionError("Provisional COME-LBI tuple passed the formal gate")
    profile = resolve_lbi_profile(
        raw,
        "office31",
        0.0005,
        allow_provisional=True,
        overrides={"alpha": 0.2},
    )
    assert profile["status"] == "override_nonformal"
    assert profile["source_profile_status"] == "provisional_default"
    assert profile["formal_eligible"] is False
    assert profile["alpha"] == 0.2
    identity_base = {
        "dataset": "office31",
        "source": "amazon",
        "target": "dslr",
        "lbi": profile,
        "checkpointing": {"enabled": True},
    }
    first = finalize_identity(copy.deepcopy(identity_base), variant=GROUP_LBI)
    no_checkpoint = copy.deepcopy(identity_base)
    no_checkpoint["checkpointing"]["enabled"] = False
    second = finalize_identity(no_checkpoint, variant=GROUP_LBI)
    changed = copy.deepcopy(identity_base)
    changed["lbi"]["alpha"] = 0.25
    third = finalize_identity(changed, variant=GROUP_LBI)
    assert first["scientific_config_sha256"] == second["scientific_config_sha256"]
    assert first["scientific_config_sha256"] != third["scientific_config_sha256"]
    return {"formal_gate": "PASS", "requested_group_counts": [3, 6, 13]}


def check_closure_is_current_state_and_backward_free() -> None:
    model = CandidateShapedModel()
    model.eval()
    images = torch.randn(2, 384)
    config = {"num_classes": NUM_CLASSES}
    loss1, diagnostics1 = come_lbi_objective(model, images, config)
    assert loss1.requires_grad
    assert all(parameter.grad is None for parameter in model.parameters())
    with torch.no_grad():
        model.blocks[11].mlp.fc2.weight.add_(0.01)
    loss2, diagnostics2 = come_lbi_objective(model, images, config)
    assert not torch.equal(loss1.detach(), loss2.detach())
    assert diagnostics1["host_objective"] == diagnostics2["host_objective"] == "come"
    assert diagnostics1["class_count_c"] == NUM_CLASSES
    assert all(parameter.grad is None for parameter in model.parameters())


def check_shared_engine_with_come_closure() -> dict:
    model, _, candidates, frozen, _ = load_candidate_scope_model(
        {}, torch.device("cpu")
    )
    images = torch.randn(2, 384)
    frozen_before = {name: value.detach().clone() for name, value in frozen}
    candidate_before = {name: value.detach().clone() for name, value in candidates}
    calls = {"count": 0}

    def closure():
        calls["count"] += 1
        return come_lbi_objective(model, images, {"num_classes": NUM_CLASSES})

    result = GroupSplitLBIEngine().run_batch(
        candidates,
        closure,
        {
            "alpha": 0.1,
            "kappa": 1.0,
            "nu": 1.0,
            "omega": 0.1,
            "prox_lambda": 1.0,
            "tau_g": 1.0e-4,
            "stage1_max_steps": 1,
            "stage2_lr": 1.0e-5,
            "requested_group_count": 3,
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
    stats = result.statistics
    assert calls["count"] == stats["objective_call_count"] == 2
    assert stats["local_restart_count"] == 1
    assert stats["stage2_optimizer_instance_count"] == 1
    assert stats["stage2_optimizer_step_count"] == 1
    assert stats["omega_writeback_count"] == 1
    assert stats["host_optimizer_persistent_step_count"] == 0
    assert stats["host_scheduler_step_count"] == 0
    assert stats["native_ema_commit_count"] == 0
    assert stats["host_objective"] == "come"
    assert stats["realized_group_count"] <= 3
    for name, parameter in candidates:
        mask = result.state.masks[name]
        assert torch.equal(
            parameter.detach()[~mask], candidate_before[name][~mask]
        )
    for name, value in frozen:
        assert torch.equal(value.detach(), frozen_before[name])
    return {
        "objective_calls": calls["count"],
        "realized_group_count": stats["realized_group_count"],
    }


def check_one_batch_runner_contract() -> dict:
    with tempfile.TemporaryDirectory(prefix="come_lbi_runner_") as value:
        output_dir = Path(value) / "run"
        config = synthetic_config(
            GROUP_LBI,
            output_dir=output_dir,
            selection={
                "requested_budget": 0.0005,
                "requested_group_count": 3,
                "maximum_active_candidate_scalars": 2304,
                "stage1_step_cap_hit_policy": "continue_to_end_of_stream",
            },
            lbi={
                "status": "provisional_default",
                "formal_eligible": False,
                "profile_key": "office31/0.0005",
                "alpha": 0.1,
                "kappa": 1.0,
                "nu": 1.0,
                "omega": 0.1,
                "prox_lambda": 1.0,
                "tau_g": 1.0e-4,
                "stage1_max_steps": 1,
                "stage2_lr": 1.0e-5,
            },
            stage2_optimization={
                "optimizer": "adamw",
                "betas": [0.9, 0.999],
                "eps": 1.0e-8,
                "weight_decay": 0.01,
                "steps": 1,
                "lr_schedule": "none",
                "strict_off_mask_value_freezing": True,
                "clear_off_mask_coordinate_state": ["exp_avg", "exp_avg_sq"],
            },
            checkpointing={
                "enabled_by_default": True,
                "granularity": "completed_online_batch",
                "save_adapted_model": False,
                "enabled": False,
            },
            diagnostics={
                "delta_nonzero_tolerance": 1.0e-12,
                "progress_refresh_steps": 25,
            },
        )
        summary = run_lbi_transfer(
            config,
            PROJECT_ROOT,
            show_progress=False,
            model_loader=load_candidate_scope_model,
            target_loaders_builder=lambda current: build_synthetic_loaders(
                current, sample_count=31, batch_size=31
            ),
        )
        rows = [
            json.loads(line)
            for line in (output_dir / "metrics.jsonl").read_text().splitlines()
        ]
    online = [row for row in rows if row["event"] == "online_batch"]
    assert len(online) == 1
    assert summary["method"] == "come" and summary["variant"] == GROUP_LBI
    assert summary["valid_lbi_run"] is True
    assert summary["lbi"]["formal_eligible"] is False
    assert summary["formal_eligible"] is False
    assert summary["result_scope"] == "search_or_smoke_nonformal"
    assert summary["protocol_revision"] == PROTOCOL_REVISIONS[GROUP_LBI]
    assert summary["implementation_revision"] == IMPLEMENTATION_REVISIONS[GROUP_LBI]
    assert online[0]["objective_call_count"] == 2
    assert online[0]["host_objective"] == "come"
    assert online[0]["pu_is_separate_read_only_forward"] is True
    assert summary["fo_is_independent_full_target_pass"] is True
    assert "come_opinion_entropy_mean" in summary
    return {"runner_online_batches": len(online)}


class _BoundaryReportingEngine:
    """Cheap runner stub whose first batch reports the frozen 3,000-step cap."""

    def __init__(self) -> None:
        self.batch_count = 0

    def run_batch(self, candidates, loss_closure, config, **kwargs):
        del candidates, loss_closure, kwargs
        self.batch_count += 1
        at_cap = self.batch_count == 1
        return SimpleNamespace(
            statistics={
                "requested_group_count": config["requested_group_count"],
                "realized_group_count": 0,
                "utilization": 0.0,
                "stage1_steps_completed": 3000 if at_cap else 4,
                "stage1_rollback_used": False,
                "stage1_stop_reason": (
                    "max_steps_reached" if at_cap else "budget_reached"
                ),
                "selected_group_ids": [],
                "selected_by_kind": {"QK": 0, "VO": 0, "FFN": 0},
                "selected_by_block": {"9": 0, "10": 0, "11": 0},
                "host_objective": "come",
                "predicted_class_histogram": [0] * NUM_CLASSES,
                "predicted_class_count": 0,
                "dominant_class_count": 0,
                "dominant_class_ratio": 0.0,
                "mean_softmax_entropy": 0.0,
                "come_opinion_entropy": 0.0,
                "come_mean_uncertainty_mass": 0.0,
            }
        )


def _boundary_config(output_dir: Path) -> dict:
    return synthetic_config(
        GROUP_LBI,
        output_dir=output_dir,
        selection={
            "requested_budget": 0.0005,
            "requested_group_count": 3,
            "maximum_active_candidate_scalars": 2304,
            "stage1_step_cap_hit_policy": "continue_to_end_of_stream",
        },
        lbi={
            "status": "provisional_default",
            "formal_eligible": False,
            "profile_key": "office31/0.0005",
            "alpha": 0.1,
            "kappa": 1.0,
            "nu": 1.0,
            "omega": 0.1,
            "prox_lambda": 1.0,
            "tau_g": 1.0e-4,
            "stage1_max_steps": 3000,
            "stage2_lr": 1.0e-5,
        },
        stage2_optimization={
            "optimizer": "adamw",
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "weight_decay": 0.01,
            "steps": 1,
            "lr_schedule": "none",
            "strict_off_mask_value_freezing": True,
            "clear_off_mask_coordinate_state": ["exp_avg", "exp_avg_sq"],
        },
        checkpointing={
            "enabled_by_default": True,
            "granularity": "completed_online_batch",
            "save_adapted_model": False,
            "enabled": False,
        },
        diagnostics={
            "delta_nonzero_tolerance": 1.0e-12,
            "progress_refresh_steps": 25,
        },
    )


def _assert_boundary_run_completed(summary: dict, rows: list[dict]) -> None:
    online = [row for row in rows if row["event"] == "online_batch"]
    assert [row["batch_index"] for row in online] == [1, 2]
    assert online[0]["stage1_steps_completed"] == 3000
    assert online[1]["stage1_steps_completed"] == 4
    assert any(row["event"] == "fo_batch" for row in rows)
    assert not any(row["event"] == "experiment_invalidated" for row in rows)
    assert summary["status"] == "completed"
    assert summary["valid_lbi_run"] is True
    assert summary["result_validity"] == "valid"
    assert summary["PU-sample-count"] == 31
    assert summary["FO-sample-count"] == 31


def check_3000_boundary_continues_for_shot_and_come() -> dict:
    with tempfile.TemporaryDirectory(prefix="lbi_boundary_continue_") as value:
        root = Path(value)

        def loader_builder(current):
            return build_synthetic_loaders(current, sample_count=31, batch_size=16)

        come_dir = root / "come"
        come_summary = run_lbi_transfer(
            _boundary_config(come_dir),
            PROJECT_ROOT,
            show_progress=False,
            model_loader=load_candidate_scope_model,
            target_loaders_builder=loader_builder,
            engine_factory=_BoundaryReportingEngine,
        )
        come_rows = [
            json.loads(line)
            for line in (come_dir / "metrics.jsonl").read_text().splitlines()
        ]
        _assert_boundary_run_completed(come_summary, come_rows)

        shot_dir = root / "shot"
        shot_config = _boundary_config(shot_dir)
        shot_config["method"] = "shot"
        shot_config["loss"] = {"components": ["ent", "div", "pseudo"]}
        shot_summary = run_shared_lbi_transfer(
            shot_config,
            PROJECT_ROOT,
            show_progress=False,
            model_loader=load_candidate_scope_model,
            target_loaders_builder=loader_builder,
            engine_factory=_BoundaryReportingEngine,
        )
        shot_rows = [
            json.loads(line)
            for line in (shot_dir / "metrics.jsonl").read_text().splitlines()
        ]
        _assert_boundary_run_completed(shot_summary, shot_rows)
    return {"boundary_continuation_methods": ["shot", "come"]}


def main() -> int:
    report = {"status": "PASS"}
    report.update(check_config_identity_and_formal_gate())
    check_closure_is_current_state_and_backward_free()
    report.update(check_shared_engine_with_come_closure())
    report.update(check_one_batch_runner_contract())
    report.update(check_3000_boundary_continues_for_shot_and_come())
    print(json.dumps(report, sort_keys=True))
    print("Transformer COME Group-LBI tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
