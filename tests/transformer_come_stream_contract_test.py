#!/usr/bin/env python3
"""Stream, singleton and target-label contracts for COME (C08, C25, C27)."""

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

import transformer_come.candidate_dense.runner as dense_runner_module  # noqa: E402
import transformer_come.common as come_common  # noqa: E402
import transformer_come.group_saliency.runner as saliency_runner_module  # noqa: E402
from transformer.candidate_dense.model import hash_tensors  # noqa: E402
from transformer.source_only.data import (  # noqa: E402
    FixedOrderSampler,
    MergeSingletonTailBatchSampler,
)
from transformer_come.budget import budget_group_count  # noqa: E402
from transformer_come.common import ComeRunInvalid, guard_online_batch  # noqa: E402

from transformer_come_fixtures import (  # noqa: E402
    BATCH_SIZE,
    CandidateShapedModel,
    NUM_CLASSES,
    SAMPLE_COUNT,
    build_synthetic_loaders,
    load_candidate_scope_model,
    read_metrics,
    sparse_selection,
    synthetic_config,
)


CANDIDATE_PREFIXES = ("blocks.9.", "blocks.10.", "blocks.11.")


def check_amazon_tail_batching() -> dict:
    """C27: the formal Amazon stream is 43x64 + 65, never 44x64 + 1."""
    sampler = FixedOrderSampler(range(2817))
    batch_sampler = MergeSingletonTailBatchSampler(
        sampler, batch_size=64, drop_last=False
    )
    sizes = [len(batch) for batch in batch_sampler]
    assert len(sizes) == len(batch_sampler) == 44
    assert sizes == [64] * 43 + [65]
    assert sum(sizes) == 2817
    assert 1 not in sizes
    plain = [
        len(batch)
        for batch in MergeSingletonTailBatchSampler(
            FixedOrderSampler(range(795)), batch_size=64, drop_last=False
        )
    ]
    assert sum(plain) == 795 and 1 not in plain
    return {"amazon_batch_sizes": [sizes[0], sizes[-1]], "amazon_batch_count": len(sizes)}


def check_singleton_guard_unit() -> None:
    guard_online_batch(2)
    for size in (1, 0):
        try:
            guard_online_batch(size)
        except ComeRunInvalid as error:
            assert error.invalid_reason == "stream_protocol_mismatch"
        else:
            raise AssertionError(f"Batch size {size} was accepted")


def _singleton_loaders(config):
    # 33 samples at batch size 8 -> the final batch holds a single sample.
    return build_synthetic_loaders(config, sample_count=33, batch_size=BATCH_SIZE)


def check_singleton_fails_before_any_adaptation() -> dict:
    """The guard must run before the objective, the optimizer and PU."""
    CandidateShapedModel.forward_calls = 0
    objective_calls = {"count": 0}
    original_come_loss = come_common.come_loss
    original_loaders = dense_runner_module.build_target_loaders

    def counting_come_loss(logits, class_count, **kwargs):
        objective_calls["count"] += 1
        return original_come_loss(logits, class_count, **kwargs)

    come_common.come_loss = counting_come_loss
    dense_runner_module.build_target_loaders = _singleton_loaders
    try:
        with tempfile.TemporaryDirectory(prefix="come_singleton_") as value:
            output_dir = Path(value) / "run"
            config = synthetic_config("candidate_dense", output_dir=output_dir)
            try:
                dense_runner_module.run_transfer(
                    config,
                    PROJECT_ROOT,
                    show_progress=False,
                    model_loader=load_candidate_scope_model,
                )
            except ComeRunInvalid as error:
                assert error.invalid_reason == "stream_protocol_mismatch"
            else:
                raise AssertionError("A singleton online batch was adapted on")
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            rows = read_metrics(output_dir)
    finally:
        come_common.come_loss = original_come_loss
        dense_runner_module.build_target_loaders = original_loaders

    assert summary["status"] == "failed"
    assert summary["result_validity"] == "invalid"
    assert summary["invalid_reason"] == "stream_protocol_mismatch"
    assert manifest["invalid_reason"] == "stream_protocol_mismatch"
    online = [row for row in rows if row["event"] == "online_batch"]
    # Four full batches ran; the singleton batch produced no objective call,
    # no optimizer step and no PU row.
    assert len(online) == 4
    assert objective_calls["count"] == 4
    assert CandidateShapedModel.forward_calls == 8
    assert [row["batch_size"] for row in online] == [BATCH_SIZE] * 4
    assert not [row for row in rows if row["event"] == "fo_batch"]
    return {"batches_before_singleton": len(online)}


def _loaders_with_label_shift(shift: int):
    def builder(config, **kwargs):
        online, final, record = build_synthetic_loaders(config, **kwargs)
        labels = torch.tensor(
            [(index + shift) % NUM_CLASSES for index in range(len(online.dataset))],
            dtype=torch.int64,
        )
        online.dataset.labels = labels
        final.dataset.labels = labels
        return online, final, record

    return builder


def _run_dense_with_labels(shift: int):
    captured = {}
    original_loaders = dense_runner_module.build_target_loaders

    def capturing_loader(config, device):
        loaded = load_candidate_scope_model(config, device)
        captured["model"] = loaded[0]
        return loaded

    dense_runner_module.build_target_loaders = _loaders_with_label_shift(shift)
    try:
        with tempfile.TemporaryDirectory(prefix="come_labels_dense_") as value:
            output_dir = Path(value) / "run"
            summary = dense_runner_module.run_transfer(
                synthetic_config("candidate_dense", output_dir=output_dir),
                PROJECT_ROOT,
                show_progress=False,
                model_loader=capturing_loader,
            )
            return summary, captured["model"]
    finally:
        dense_runner_module.build_target_loaders = original_loaders


def _run_saliency_with_labels(shift: int):
    original_loaders = saliency_runner_module.build_target_loaders
    saliency_runner_module.build_target_loaders = _loaders_with_label_shift(shift)
    try:
        with tempfile.TemporaryDirectory(prefix="come_labels_saliency_") as value:
            output_dir = Path(value) / "run"
            budget = 0.002
            summary = saliency_runner_module.run_transfer(
                synthetic_config(
                    "group_saliency",
                    output_dir=output_dir,
                    selection=sparse_selection(
                        budget, budget_group_count(budget), per_step=True
                    ),
                ),
                PROJECT_ROOT,
                show_progress=False,
                model_loader=load_candidate_scope_model,
            )
            rows = [
                row
                for row in read_metrics(output_dir)
                if row["event"] == "online_batch"
            ]
            return summary, rows
    finally:
        saliency_runner_module.build_target_loaders = original_loaders


def check_target_labels_do_not_reach_adaptation() -> dict:
    """C08: permuting target labels changes metrics only."""
    first_summary, first_model = _run_dense_with_labels(0)
    second_summary, second_model = _run_dense_with_labels(5)

    def candidate_hash(model):
        return hash_tensors(
            [
                (name, parameter)
                for name, parameter in model.named_parameters()
                if name.startswith(CANDIDATE_PREFIXES)
            ]
        )

    assert candidate_hash(first_model) == candidate_hash(second_model)
    assert first_summary["model_state_sha256_after_stream"] == second_summary[
        "model_state_sha256_after_stream"
    ]
    assert first_summary["come_opinion_entropy_mean"] == second_summary[
        "come_opinion_entropy_mean"
    ]
    # The metrics themselves must react to the relabelling.
    assert first_summary["PU-Acc"] != second_summary["PU-Acc"] or first_summary[
        "PU-correct"
    ] != second_summary["PU-correct"]

    first_saliency, first_rows = _run_saliency_with_labels(0)
    second_saliency, second_rows = _run_saliency_with_labels(5)
    assert first_saliency["selection"]["mask_history_sha256"] == second_saliency[
        "selection"
    ]["mask_history_sha256"]
    assert [row["selected_group_ids"] for row in first_rows] == [
        row["selected_group_ids"] for row in second_rows
    ]
    assert first_saliency["model_state_sha256_after_stream"] == second_saliency[
        "model_state_sha256_after_stream"
    ]
    return {"label_shifts_compared": [0, 5]}


def check_pu_and_fo_are_read_only() -> dict:
    """C25: PU is a separate post-update forward; FO changes nothing."""
    original_loaders = dense_runner_module.build_target_loaders
    captured = {}

    def capturing_loader(config, device):
        loaded = load_candidate_scope_model(config, device)
        captured["model"] = loaded[0]
        return loaded

    dense_runner_module.build_target_loaders = build_synthetic_loaders
    CandidateShapedModel.forward_calls = 0
    try:
        with tempfile.TemporaryDirectory(prefix="come_pu_fo_") as value:
            output_dir = Path(value) / "run"
            summary = dense_runner_module.run_transfer(
                synthetic_config("candidate_dense", output_dir=output_dir),
                PROJECT_ROOT,
                show_progress=False,
                model_loader=capturing_loader,
            )
    finally:
        dense_runner_module.build_target_loaders = original_loaders

    batch_count = SAMPLE_COUNT // BATCH_SIZE
    assert summary["pu_is_post_update_same_batch"] is True
    assert summary["pu_is_separate_read_only_forward"] is True
    assert summary["model_state_sha256_after_stream"] == summary[
        "model_state_sha256_after_fo"
    ]
    assert summary["PU-sample-count"] == summary["FO-sample-count"] == SAMPLE_COUNT
    assert CandidateShapedModel.forward_calls == 3 * batch_count
    model = captured["model"]
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not model.training
    assert not any(parameter.requires_grad for parameter in model.parameters())
    return {"forward_calls": CandidateShapedModel.forward_calls}


def main() -> int:
    report = {"status": "PASS"}
    report.update(check_amazon_tail_batching())
    check_singleton_guard_unit()
    report.update(check_singleton_fails_before_any_adaptation())
    report.update(check_target_labels_do_not_reach_adaptation())
    report.update(check_pu_and_fo_are_read_only())
    print(json.dumps(report, sort_keys=True))
    print("Transformer COME stream contract tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
