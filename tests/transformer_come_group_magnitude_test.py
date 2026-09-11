#!/usr/bin/env python3
"""CPU contracts for the COME Group-Magnitude baseline (C12, C14, C17)."""

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

import transformer.group_magnitude.groups as shot_magnitude_groups  # noqa: E402
import transformer_come.common as come_common  # noqa: E402
import transformer_come.group_magnitude.runner as runner_module  # noqa: E402
from transformer_come.budget import budget_group_count  # noqa: E402
from transformer_come.group_magnitude.config import (  # noqa: E402
    EXPECTED_SELECTION,
    IMPLEMENTATION_REVISION,
    PROTOCOL_REVISION,
    _validate_frozen_fields,
    load_config,
)
from transformer_come.group_magnitude.groups import (  # noqa: E402
    compute_group_l2_scores,
    select_magnitude_group_ids,
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


CONFIG_PATH = PROJECT_ROOT / "transformer_come" / "group_magnitude" / "config.yaml"
BUDGET = 0.002


def check_config() -> None:
    config = load_config(CONFIG_PATH)
    _validate_frozen_fields(config)
    assert config["protocol_revision"] == PROTOCOL_REVISION
    assert config["method"] == "come"
    assert config["selection"] == EXPECTED_SELECTION
    assert config["selection"]["score_dtype"] == "float64"
    assert config["selection"]["score_device"] == "cpu"
    assert config["selection"]["ranking_source"] == "source_checkpoint_pre_adaptation"
    assert config["selection"]["mask_refresh_policy"] == "once_before_adaptation"


def check_scoring_is_paired_l2_on_w0() -> dict:
    """C14: paired L2 in CPU float64 with canonical-id tie-break."""
    assert compute_group_l2_scores is shot_magnitude_groups.compute_group_l2_scores
    model = CandidateShapedModel()
    candidates = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith(("blocks.9.", "blocks.10.", "blocks.11."))
    ]
    scores = compute_group_l2_scores(candidates)
    assert scores.dtype is torch.float64
    assert scores.device.type == "cpu"
    assert scores.shape == (6912,)

    lookup = dict(model.named_parameters())
    qkv = lookup["blocks.9.attn.qkv.weight"].detach().to(torch.float64)
    proj = lookup["blocks.9.attn.proj.weight"].detach().to(torch.float64)
    fc1 = lookup["blocks.9.mlp.fc1.weight"].detach().to(torch.float64)
    fc2 = lookup["blocks.9.mlp.fc2.weight"].detach().to(torch.float64)
    manual_qk = torch.sqrt(qkv[0].square().sum() + qkv[384].square().sum())
    manual_vo = torch.sqrt(qkv[768].square().sum() + proj[:, 0].square().sum())
    manual_ffn = torch.sqrt(fc1[0].square().sum() + fc2[:, 0].square().sum())
    assert torch.allclose(scores[0], manual_qk)
    assert torch.allclose(scores[384], manual_vo)
    assert torch.allclose(scores[768], manual_ffn)
    # Not an L1 score: the paired L2 of a real weight row differs from its
    # absolute-value sum.
    manual_l1 = qkv[0].abs().sum() + qkv[384].abs().sum()
    assert not torch.allclose(scores[0], manual_l1)

    tied = torch.zeros(6912, dtype=torch.float64)
    tied[5] = tied[9] = tied[2] = 1.0
    assert select_magnitude_group_ids(tied, 0.0005) == (2, 5, 9)
    return {"selected_ids_for_tied_scores": [2, 5, 9]}


def _run():
    CandidateShapedModel.forward_calls = 0
    scoring_calls = {"count": 0}
    objective_calls = {"count": 0}
    captured = {}
    original_scores = runner_module.compute_group_l2_scores
    original_come_loss = come_common.come_loss
    original_loaders = runner_module.build_target_loaders

    def counting_scores(named_parameters):
        scoring_calls["count"] += 1
        return original_scores(named_parameters)

    def counting_come_loss(logits, class_count, **kwargs):
        objective_calls["count"] += 1
        return original_come_loss(logits, class_count, **kwargs)

    def capturing_loader(config, device):
        loaded = load_candidate_scope_model(config, device)
        captured["model"] = loaded[0]
        return loaded

    runner_module.compute_group_l2_scores = counting_scores
    come_common.come_loss = counting_come_loss
    runner_module.build_target_loaders = build_synthetic_loaders
    try:
        with tempfile.TemporaryDirectory(prefix="come_group_magnitude_") as value:
            output_dir = Path(value) / "run"
            config = synthetic_config(
                "group_magnitude",
                output_dir=output_dir,
                selection=sparse_selection(BUDGET, budget_group_count(BUDGET)),
            )
            summary = runner_module.run_transfer(
                config,
                PROJECT_ROOT,
                show_progress=False,
                model_loader=capturing_loader,
            )
            mask = json.loads((output_dir / "mask.json").read_text(encoding="utf-8"))
            return (
                summary,
                read_metrics(output_dir),
                mask,
                scoring_calls["count"],
                objective_calls["count"],
                captured["model"],
            )
    finally:
        runner_module.compute_group_l2_scores = original_scores
        come_common.come_loss = original_come_loss
        runner_module.build_target_loaders = original_loaders


def check_static_support_and_counts() -> dict:
    summary, rows, mask, scoring_calls, objective_calls, model = _run()
    batch_count = SAMPLE_COUNT // BATCH_SIZE
    online = [row for row in rows if row["event"] == "online_batch"]
    group_count = budget_group_count(BUDGET)

    assert summary["status"] == "completed"
    assert summary["implementation_revision"] == IMPLEMENTATION_REVISION
    # Scored exactly once, before the stream: never per batch.
    assert scoring_calls == 1
    assert objective_calls == batch_count
    assert summary["objective_call_count"] == batch_count
    assert summary["optimizer_step_count"] == batch_count
    assert summary["scheduler_step_count"] == 0
    assert all(row["support_selection_count"] == 0 for row in online)
    assert all(row["mask_is_dynamic"] is False for row in online)
    assert len({row["mask_sha256"] for row in online}) == 1
    assert CandidateShapedModel.forward_calls == 2 * batch_count + batch_count

    # Exact-K static support and its scalar count.
    assert summary["requested_group_count"] == group_count == 13
    assert summary["realized_group_count"] == group_count
    assert summary["active_candidate_scalars"] == group_count * 768
    assert isinstance(summary["selection"], dict)
    assert summary["selection"]["mask_sha256"] == mask["mask_sha256"]
    assert summary["selection"]["selection"] == "source_w0_structural_group_l2"
    assert summary["selection"]["score_dtype"] == "float64"
    assert len(mask["selected_group_ids"]) == group_count
    assert summary["off_mask_state_unchanged"] is True
    # Relative-to-source delta support stays inside the historical support
    # union, which for a static mask is that one mask.
    assert summary["delta_nonzero_tolerance"] == 1.0e-12
    assert summary["final_delta_support_basis"] == "historical_selected_support_union"
    assert summary["historical_active_group_union_count"] == group_count
    assert set(summary["final_delta_support_group_ids"]) <= set(
        summary["selection"]["selected_group_ids"]
    )
    assert 0 < summary["final_delta_support_group_count"] <= group_count
    assert 0 < summary["final_delta_nonzero_scalar_count"] <= group_count * 768
    assert summary["selected_state_changed"] is True
    assert summary["off_mask_adam_state_zero"] is True
    assert summary["frozen_state_unchanged"] is True

    # The support is the W0 ranking, not a ranking of the adapted weights.
    reference = CandidateShapedModel()
    reference_candidates = [
        (name, parameter)
        for name, parameter in reference.named_parameters()
        if name.startswith(("blocks.9.", "blocks.10.", "blocks.11."))
    ]
    expected = select_magnitude_group_ids(
        compute_group_l2_scores(reference_candidates), BUDGET
    )
    assert tuple(mask["selected_group_ids"]) == expected
    adapted_candidates = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith(("blocks.9.", "blocks.10.", "blocks.11."))
    ]
    assert not torch.equal(
        compute_group_l2_scores(adapted_candidates),
        compute_group_l2_scores(reference_candidates),
    )
    return {
        "requested_group_count": group_count,
        "active_candidate_scalars": group_count * 768,
        "scoring_calls": scoring_calls,
    }


def main() -> int:
    check_config()
    report = {"status": "PASS"}
    report.update(check_scoring_is_paired_l2_on_w0())
    report.update(check_static_support_and_counts())
    print(json.dumps(report, sort_keys=True))
    print("Transformer COME Group-Magnitude tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
