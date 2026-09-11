#!/usr/bin/env python3
"""CPU contracts for the COME Group-Saliency baseline (C15, C16, C17).

The dangerous part of Saliency is gradient reuse: one COME forward/backward
per batch, a support chosen from that gradient, and the same gradient consumed
by the masked step.  These tests check the reuse literally (the gradient at
step time is the gradient at selection time) and check that the audits which
make the reuse valid actually fire when violated.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(PROJECT_ROOT), str(PROJECT_ROOT / "tests")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import transformer_come.common as come_common  # noqa: E402
import transformer_come.runner as runner_module  # noqa: E402
from transformer_come.config import budget_group_count  # noqa: E402
from transformer_come.groups import build_masks  # noqa: E402
from transformer_come.config import (  # noqa: E402
    SELECTION_BLOCKS,
    IMPLEMENTATION_REVISIONS,
    PROTOCOL_REVISIONS,
    validate_variant_config,
    variant_config_view,
    load_config,
)
from transformer_come.groups import (  # noqa: E402
    compute_group_saliency_scores,
    select_saliency_group_ids,
)

from transformer_come_fixtures import (  # noqa: E402
    BATCH_SIZE,
    CandidateShapedModel,
    SAMPLE_COUNT,
    build_synthetic_loaders,
    load_candidate_scope_model,
    read_metrics,
    sparse_selection,
    synthetic_config,
)


VARIANT = "group_saliency"
CONFIG_PATH = PROJECT_ROOT / "transformer_come" / "config.yaml"
BUDGET = 0.002
CANDIDATE_PREFIXES = ("blocks.9.", "blocks.10.", "blocks.11.")


def check_config() -> None:
    config = variant_config_view(load_config(CONFIG_PATH), VARIANT)
    validate_variant_config(config, variant=VARIANT)
    assert config["protocol_revision"] == PROTOCOL_REVISIONS[VARIANT]
    assert config["method"] == "come"
    assert config["selection"] == SELECTION_BLOCKS[VARIANT]
    assert (
        config["selection"]["ranking_source"]
        == "current_pre_update_weight_and_current_come_gradient"
    )
    assert (
        config["selection"]["mask_refresh_policy"]
        == "every_online_batch_after_backward_before_step"
    )


def _named_candidates(model):
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith(CANDIDATE_PREFIXES)
    ]


def _run(*, select_override=None, audit_batches=None, keep_gradient_batches=2):
    CandidateShapedModel.forward_calls = 0
    objective_calls = {"count": 0}
    selection_gradients: list[dict] = []
    step_gradients: list[dict] = []
    selection_records: list[dict] = []
    captured = {}

    original_come_loss = come_common.come_loss
    original_loaders = runner_module.build_target_loaders
    original_select = runner_module._select_support
    original_step = runner_module.strict_masked_adamw_step
    original_audit = runner_module.SELECTION_AUDIT_BATCHES

    def counting_come_loss(logits, class_count, **kwargs):
        objective_calls["count"] += 1
        return original_come_loss(logits, class_count, **kwargs)

    def spy_select(*, candidates, config):
        if len(selection_gradients) < keep_gradient_batches:
            selection_gradients.append(
                {name: parameter.grad.detach().clone() for name, parameter in candidates}
            )
        masks, record = (select_override or original_select)(
            candidates=candidates, config=config
        )
        selection_records.append(record)
        return masks, record

    def spy_step(optimizer, named_parameters, masks):
        named_parameters = list(named_parameters)
        if len(step_gradients) < keep_gradient_batches:
            step_gradients.append(
                {
                    name: parameter.grad.detach().clone()
                    for name, parameter in named_parameters
                }
            )
        return original_step(optimizer, named_parameters, masks)

    def capturing_loader(config, device):
        loaded = load_candidate_scope_model(config, device)
        captured["model"] = loaded[0]
        return loaded

    come_common.come_loss = counting_come_loss
    runner_module.build_target_loaders = build_synthetic_loaders
    runner_module._select_support = spy_select
    runner_module.strict_masked_adamw_step = spy_step
    if audit_batches is not None:
        runner_module.SELECTION_AUDIT_BATCHES = audit_batches
    try:
        with tempfile.TemporaryDirectory(prefix="come_group_saliency_") as value:
            output_dir = Path(value) / "run"
            config = synthetic_config(
                "group_saliency",
                output_dir=output_dir,
                selection=sparse_selection(
                    BUDGET, budget_group_count(BUDGET), per_step=True
                ),
            )
            summary = runner_module.run_transfer(
                config,
                PROJECT_ROOT,
                show_progress=False,
                model_loader=capturing_loader,
            )
            return {
                "summary": summary,
                "rows": read_metrics(output_dir),
                "objective_calls": objective_calls["count"],
                "selection_gradients": selection_gradients,
                "step_gradients": step_gradients,
                "selection_records": selection_records,
                "model": captured.get("model"),
                "output_dir": output_dir,
            }
    finally:
        come_common.come_loss = original_come_loss
        runner_module.build_target_loaders = original_loaders
        runner_module._select_support = original_select
        runner_module.strict_masked_adamw_step = original_step
        runner_module.SELECTION_AUDIT_BATCHES = original_audit


def check_single_objective_and_gradient_reuse() -> dict:
    result = _run(audit_batches=10**6)
    summary = result["summary"]
    rows = [row for row in result["rows"] if row["event"] == "online_batch"]
    batch_count = SAMPLE_COUNT // BATCH_SIZE

    # C15: still exactly one objective call and one forward per batch; the
    # selection neither re-runs the objective nor re-forwards the model.
    assert result["objective_calls"] == batch_count
    assert summary["objective_call_count"] == batch_count
    assert summary["backward_calls"] == batch_count
    assert summary["optimizer_step_count"] == batch_count
    assert summary["scheduler_step_count"] == 0
    assert CandidateShapedModel.forward_calls == 2 * batch_count + batch_count
    assert all(row["support_selection_count"] == 1 for row in rows)
    assert all(row["mask_is_dynamic"] is True for row in rows)
    assert summary["selection"]["support_selection_count"] == batch_count
    assert summary["selection_state_audit_enabled"] is True
    assert summary["selection_state_audit_batches"] == list(range(1, batch_count + 1))

    # The gradient consumed by the masked step is the gradient that was
    # scored, coordinate for coordinate.
    assert result["selection_gradients"] and len(result["selection_gradients"]) == len(
        result["step_gradients"]
    )
    for selected, stepped in zip(result["selection_gradients"], result["step_gradients"]):
        assert set(selected) == set(stepped)
        for name, gradient in selected.items():
            assert torch.equal(gradient, stepped[name]), name
    return {"batch_count": batch_count}


def check_scores_and_exact_k() -> dict:
    """The support is the exact-K top of |W * grad| paired group L2."""
    result = _run()
    group_count = budget_group_count(BUDGET)
    records = result["selection_records"]
    assert len(records) == SAMPLE_COUNT // BATCH_SIZE
    for record in records:
        assert record["realized_group_count"] == group_count == 13
        assert len(record["selected_group_ids"]) == group_count
        assert len(set(record["selected_group_ids"])) == group_count
        assert record["active_candidate_scalars"] == group_count * 768
        assert record["top_k_min_score"] >= record["first_excluded_score"]

    summary = result["summary"]
    assert summary["realized_group_count_per_step"] == group_count
    assert summary["active_candidate_scalars_per_step"] == group_count * 768
    assert summary["selection"]["unique_mask_count"] >= 1
    assert summary["selection"]["historical_active_group_union_count"] >= group_count
    assert summary["strict_dynamic_off_mask_value_freezing"] is True
    assert summary["strict_dynamic_off_mask_adam_state_clearing"] is True
    assert summary["selection_rng_audit_every_batch"] is True
    # Only correctness instrumentation may be excluded from the adaptation
    # cost; the selector itself (scoring, ranking, top-K, mask construction)
    # is adaptation cost.  The raw runtime is reported alongside it so the
    # subtraction can be inspected rather than trusted.
    assert summary["selection_audit_excluded_from_adapt_runtime"] is True
    assert summary["selection_audit_excludes_only_instrumentation"] is True
    assert summary["selector_scoring_included_in_adapt_runtime"] is True
    assert summary["adapt_runtime_with_audit_total_sec"] >= summary[
        "adapt_runtime_total_sec"
    ]
    # Off by default: a formal efficiency run carries no state-hash audit.
    assert runner_module.SELECTION_AUDIT_BATCHES == 0
    assert summary["selection_state_audit_enabled"] is False
    assert summary["selection_state_audit_batches"] == []
    return {"unique_mask_count": summary["selection"]["unique_mask_count"]}


def check_saliency_score_definition() -> None:
    """The score is the paired-group L2 of |W * grad|, on the right axes."""

    from transformer_come_fixtures import candidate_tensors

    torch.manual_seed(11)
    candidates = candidate_tensors()
    for _, parameter in candidates:
        with torch.no_grad():
            parameter.copy_(torch.randn_like(parameter) * 0.05)
        parameter.grad = torch.randn_like(parameter) * 0.01
    scores = compute_group_saliency_scores(candidates)
    lookup = dict(candidates)
    qkv = lookup["blocks.9.attn.qkv.weight"]
    proj = lookup["blocks.9.attn.proj.weight"]
    fc1 = lookup["blocks.9.mlp.fc1.weight"]
    fc2 = lookup["blocks.9.mlp.fc2.weight"]
    saliency_qkv = (qkv.detach() * qkv.grad).abs()
    saliency_proj = (proj.detach() * proj.grad).abs()
    saliency_fc1 = (fc1.detach() * fc1.grad).abs()
    saliency_fc2 = (fc2.detach() * fc2.grad).abs()
    manual_qk = torch.sqrt(
        saliency_qkv[0].square().sum() + saliency_qkv[384].square().sum()
    )
    manual_vo = torch.sqrt(
        saliency_qkv[768].square().sum() + saliency_proj[:, 0].square().sum()
    )
    manual_ffn = torch.sqrt(
        saliency_fc1[0].square().sum() + saliency_fc2[:, 0].square().sum()
    )
    assert torch.allclose(scores[0], manual_qk)
    assert torch.allclose(scores[384], manual_vo)
    assert torch.allclose(scores[768], manual_ffn)
    # Exact-K global top, canonical id breaking ties.
    tied = torch.zeros(6912)
    tied[4] = tied[1] = tied[7] = 1.0
    assert select_saliency_group_ids(tied, 0.0005) == (1, 4, 7)


def check_only_historically_selected_coordinates_move() -> dict:
    """Dynamic off-mask freezing: nothing outside the support union changed."""
    result = _run()
    rows = [row for row in result["rows"] if row["event"] == "online_batch"]
    union_ids = sorted({
        group_id for row in rows for group_id in row["selected_group_ids"]
    })
    adapted = dict(_named_candidates(result["model"]))
    reference = dict(_named_candidates(CandidateShapedModel()))
    union_masks = build_masks(list(adapted.items()), union_ids)
    changed_outside = 0
    changed_inside = 0
    for name, parameter in adapted.items():
        difference = parameter.detach() != reference[name].detach()
        changed_outside += int(
            torch.count_nonzero(difference & ~union_masks[name]).item()
        )
        changed_inside += int(
            torch.count_nonzero(difference & union_masks[name]).item()
        )
    assert changed_outside == 0
    assert changed_inside > 0
    summary = result["summary"]
    assert changed_inside == summary["final_delta_nonzero_scalar_count"]
    assert summary["delta_nonzero_tolerance"] == 1.0e-12
    assert summary["final_delta_support_basis"] == "historical_selected_support_union"
    assert summary["historical_active_group_union_count"] == len(union_ids)
    delta_ids = set(summary["final_delta_support_group_ids"])
    assert delta_ids <= set(union_ids)
    assert summary["final_delta_support_group_count"] <= len(union_ids)
    # The basis must be the historical union, never the current batch mask:
    # with a persistent host optimizer, groups selected in earlier batches keep
    # their delta relative to W0 after they leave the support.  Asserting the
    # delta support against the last batch mask would be wrong, so pin that it
    # genuinely exceeds it.
    last_batch_ids = set(rows[-1]["selected_group_ids"])
    assert len(delta_ids) > len(last_batch_ids)
    assert not delta_ids <= last_batch_ids
    return {
        "historical_union_groups": len(union_ids),
        "changed_scalars_inside_union": changed_inside,
    }


def check_rng_audit_fires() -> None:
    """A selector that consumes RNG must invalidate the run."""
    original_select = runner_module._select_support

    def rng_consuming_select(*, candidates, config):
        torch.rand(1)
        return original_select(candidates=candidates, config=config)

    try:
        _run(select_override=rng_consuming_select, audit_batches=10**6)
    except come_common.ComeRunInvalid as error:
        assert error.invalid_reason == "rng_state_changed"
    else:
        raise AssertionError("An RNG-consuming selection was accepted")


def check_state_audit_fires() -> None:
    """A selector that mutates the model must invalidate the run."""
    original_select = runner_module._select_support

    def mutating_select(*, candidates, config):
        masks, record = original_select(candidates=candidates, config=config)
        with torch.no_grad():
            candidates[0][1].add_(1.0e-3)
        return masks, record

    try:
        _run(select_override=mutating_select, audit_batches=10**6)
    except come_common.ComeRunInvalid as error:
        assert error.invalid_reason == "selection_mutated_state"
    else:
        raise AssertionError("A model-mutating selection was accepted")


def check_scorer_is_shared_with_shot() -> None:
    import transformer.group_saliency.groups as shot_groups

    assert compute_group_saliency_scores is shot_groups.compute_group_saliency_scores
    assert select_saliency_group_ids is shot_groups.select_saliency_group_ids


def main() -> int:
    check_config()
    report = {"status": "PASS"}
    report.update(check_single_objective_and_gradient_reuse())
    report.update(check_scores_and_exact_k())
    report.update(check_only_historically_selected_coordinates_move())
    check_rng_audit_fires()
    check_state_audit_fires()
    check_saliency_score_definition()
    check_scorer_is_shared_with_shot()
    print(json.dumps(report, sort_keys=True))
    print("Transformer COME Group-Saliency tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
