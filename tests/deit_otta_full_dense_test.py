#!/usr/bin/env python3
"""CPU contracts for DeiT full-dense OTTA; no network or real W0 is used."""

import copy
import json
import math
import os.path as osp
import sys
import tempfile
from pathlib import Path

import torch


PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
TEST_ROOT = osp.dirname(osp.abspath(__file__))
for path in (PROJECT_ROOT, TEST_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from deit_source_only_fixtures import (  # noqa: E402
    TinyDeiT,
    install_real_fake_checkpoint,
    prepare_assets,
)
from shot_otta.otta.full_dense import (  # noqa: E402
    run_full_dense_otta_experiment,
)
from shot_otta.otta.full_dense_config import resolve_config  # noqa: E402


CONFIG_PATH = osp.join(PROJECT_ROOT, "configs", "deit_otta_full_dense.yaml")


class CountingTinyDeiT(TinyDeiT):
    total_forward_calls = 0

    def forward(self, images):
        type(self).total_forward_calls += 1
        return super().forward(images)


def counting_tiny_factory(model_name, **kwargs):
    assert model_name == "deit_small_patch16_224.fb_in1k"
    return CountingTinyDeiT(**kwargs)


def _base(root):
    return prepare_assets(root, CONFIG_PATH, runs_directory="full_dense_runs")


def _check_config_contract(base):
    effective = resolve_config(base, PROJECT_ROOT)
    assert effective["method"] == "shot"
    assert effective["variant"] == "full_dense"
    assert effective["task"] == "otta"
    assert effective["adaptation"]["steps_per_batch"] == 1
    assert effective["optimization"] == {
        "optimizer": "adamw",
        "lr": 1.0e-5,
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
        "weight_decay": 0.01,
    }
    scientific = effective["scientific_config"]
    assert scientific["protocol"] == (
        "sequential_target_stream_full_dense_adaptation"
    )
    assert scientific["adaptation"]["update_scope"] == "all_parameters"
    assert scientific["adaptation"]["model_mode"] == "eval"
    assert scientific["adaptation"]["delta_semantics"] == (
        "unrestricted_accumulation"
    )
    assert "full_dense" in effective["experiment_key"]
    return effective


def _check_identity_binds_optimizer(base):
    first = resolve_config(base, PROJECT_ROOT)
    changed = copy.deepcopy(base)
    changed["optimization"]["lr"] = 2.0e-5
    second = resolve_config(changed, PROJECT_ROOT)
    assert first["experiment_key"] != second["experiment_key"]
    assert first["experiment_config_sha256"] != second[
        "experiment_config_sha256"
    ]


def _expect_value_error(config, message):
    try:
        resolve_config(config, PROJECT_ROOT)
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"Expected ValueError containing: {message}")


def _check_validation(base):
    bad = copy.deepcopy(base)
    bad["method"] = "no_tta"
    _expect_value_error(bad, "method must be shot")
    bad = copy.deepcopy(base)
    bad["variant"] = "source_only"
    _expect_value_error(bad, "variant must be full_dense")
    bad = copy.deepcopy(base)
    bad["optimization"]["optimizer"] = "sgd"
    _expect_value_error(bad, "adamw")
    bad = copy.deepcopy(base)
    bad["loss"]["components"] = ["ent", "unknown"]
    _expect_value_error(bad, "components")
    bad = copy.deepcopy(base)
    bad["adaptation"]["update_scope"] = "last_3_blocks"
    _expect_value_error(bad, "update_scope")
    bad = copy.deepcopy(base)
    bad["adaptation"]["steps_per_batch"] = 0
    _expect_value_error(bad, "steps_per_batch")


def _check_end_to_end(root, base):
    install_real_fake_checkpoint(root)
    effective = resolve_config(base, PROJECT_ROOT)
    CountingTinyDeiT.total_forward_calls = 0
    summary = run_full_dense_otta_experiment(
        effective, PROJECT_ROOT, model_factory=counting_tiny_factory
    )
    assert summary["status"] == "completed"
    assert summary["adaptation_steps"] == 2
    assert summary["adaptation_steps_per_batch"] == 1
    assert summary["backward_calls"] == 2
    assert summary["optimizer_created"] is True
    assert summary["loss_computed"] is True
    assert summary["model_state_unchanged"] is False
    assert summary["model_state_sha256_before"] != summary[
        "model_state_sha256_after_stream"
    ]
    assert summary["model_state_sha256_after_stream"] == summary[
        "model_state_sha256_after_fo"
    ]
    assert summary["processed_sample_count"] == 31
    assert summary["target_batch_count"] == 2
    assert summary["tail_batch_size"] == 1
    assert summary["tail_batch_size_one_policy"] == "kept"
    assert summary["stream_prediction_passes"] == 1
    assert summary["fo_prediction_passes"] == 1
    assert summary["fo_predictions_reused"] is False
    assert summary["fo_evaluation_scope"] == "full_target_dataset"
    # Only head column 0 (gradients) and the bias (gradients + decay) can
    # move in this fixture: other head columns, cls_token, and pos_embed are
    # initialized to zero and receive no gradient, so decay leaves them at 0.
    assert 0 < summary["delta_nonzero_scalars"] <= 62
    assert summary["delta_nonzero_scalars"] < summary[
        "total_parameter_scalars"
    ]
    assert summary["delta_l2_norm"] > 0.0
    assert math.isfinite(summary["delta_l2_norm"])
    assert summary["trainable_scalars"] == summary["total_parameter_scalars"]
    assert 0.0 <= summary["PU-Acc"] <= 100.0
    assert 0.0 <= summary["FO-Acc"] <= 100.0
    # Two adaptation forwards (loss + PU prediction) per batch plus one FO
    # inference forward per batch: 3 * 2 = 6 forwards in total.
    assert CountingTinyDeiT.total_forward_calls == 6
    output = Path(summary["output_dir"])
    assert (output / "manifest.json").is_file()
    assert (output / "summary.json").is_file()
    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["protocol"] == (
        "sequential_target_stream_full_dense_adaptation"
    )
    assert manifest["adaptation"]["delta_semantics"] == (
        "unrestricted_accumulation"
    )
    assert manifest["optimization"]["optimizer"] == "adamw"
    metrics = [
        json.loads(line)
        for line in (output / "metrics.jsonl").read_text(encoding="utf-8")
        .splitlines()
    ]
    batch_rows = [row for row in metrics if row["event"] == "online_batch"]
    assert [row["sample_count"] for row in batch_rows] == [30, 1]
    assert [row["stream_batch"] for row in batch_rows] == [1, 2]
    assert all(row["adaptation_steps"] == 1 for row in batch_rows)
    assert all(row["model_update_applied"] is True for row in batch_rows)
    assert all(row["backward_calls"] >= 1 for row in batch_rows)
    for row in batch_rows:
        for field in (
            "loss_total",
            "loss_cls",
            "loss_ent",
            "loss_div",
            "pseudo_ratio",
            "max_prob_mean",
            "PU-batch-overall-Acc",
        ):
            assert field in row, field
        assert math.isfinite(float(row["loss_total"]))
    assert [row["state_carried_to_next_batch"] for row in batch_rows] == [
        True,
        False,
    ]
    final_rows = [row for row in metrics if row["event"] == "final"]
    assert len(final_rows) == 1
    assert final_rows[0]["status"] == "completed"
    assert final_rows[0]["adaptation_steps"] == 2
    return summary


def main():
    with tempfile.TemporaryDirectory(prefix="deit_otta_full_dense_") as root:
        base = _base(root)
        _check_config_contract(base)
        _check_identity_binds_optimizer(base)
        _check_validation(base)
        _check_end_to_end(root, base)
    print("DeiT OTTA full-dense tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
