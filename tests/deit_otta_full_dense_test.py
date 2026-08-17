#!/usr/bin/env python3
"""CPU contracts for DeiT OTTA adaptation baselines; no network or real W0.

Covers the full-dense (all parameters) and candidate-dense (last-3 block
weight tensors only) variants.
"""

import copy
import json
import math
import os.path as osp
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn


PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
TEST_ROOT = osp.dirname(osp.abspath(__file__))
for path in (PROJECT_ROOT, TEST_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from deit_source_only_fixtures import (  # noqa: E402
    SOURCE_CHECKPOINT_KIND,
    SOURCE_CHECKPOINT_SCHEMA_VERSION,
    checkpoint_metadata,
    install_real_fake_checkpoint,
    prepare_assets,
    sha256_file,
    write_json,
)
from shot_otta.backbones.deit import (  # noqa: E402
    apply_update_scope,
    candidate_parameter_names,
)
from shot_otta.otta.full_dense import (  # noqa: E402
    run_deit_otta_adaptation_experiment,
)
from shot_otta.otta.full_dense_config import resolve_config  # noqa: E402


CONFIG_PATH = osp.join(PROJECT_ROOT, "configs", "deit_otta_full_dense.yaml")
CANDIDATE_CONFIG_PATH = osp.join(
    PROJECT_ROOT, "configs", "deit_otta_candidate_dense.yaml"
)
CANDIDATE_SCALARS = 5_308_416  # 3 blocks x 4 weight tensors of the real DeiT-S


class DenseTinyDeiT(nn.Module):
    """Tiny DeiT whose non-head params (cls_token, pos_embed) affect logits.

    Every parameter produces gradients, so full-dense (all_except_head)
    updates cls_token + pos_embed deterministically while the head stays
    frozen.
    """

    total_forward_calls = 0

    def __init__(self, num_classes, **kwargs):
        super().__init__()
        del kwargs
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 384))
        self.pos_embed = nn.Parameter(torch.zeros(1, 2, 384))
        self.head = nn.Linear(384, num_classes)

    def get_classifier(self):
        return self.head

    def forward(self, images):
        type(self).total_forward_calls += 1
        batch = images.size(0)
        device = images.device
        dtype = images.dtype
        base = images.mean(dim=(1, 2, 3))  # per-sample feature scalar
        # Fold cls_token/pos_embed into the features fed to the (frozen) head
        # so every one of their scalars shares the same nonzero gradient and
        # the SHOT loss is not invariant to the perturbation.
        extra = base * (self.cls_token.sum() + self.pos_embed.sum())
        features = torch.zeros(batch, 384, device=device, dtype=dtype)
        features[:, 0] = base + 0.01 * extra
        return self.head(features)


def dense_tiny_factory(model_name, **kwargs):
    assert model_name == "deit_small_patch16_224.fb_in1k"
    return DenseTinyDeiT(**kwargs)


def install_dense_fake_checkpoint(root):
    """Overwrite amazon.pth with a dense-shaped fake W0 (nonzero head)."""
    install_real_fake_checkpoint(root)
    path = Path(root) / "checkpoints" / "office31" / "amazon.pth"
    model = DenseTinyDeiT(num_classes=31)
    metadata = checkpoint_metadata("office31", "amazon", 31)
    torch.save(
        {
            "schema_version": SOURCE_CHECKPOINT_SCHEMA_VERSION,
            "kind": SOURCE_CHECKPOINT_KIND,
            "state_dict": model.state_dict(),
            "metadata": metadata,
        },
        path,
    )
    write_json(
        path.with_suffix(".manifest.json"),
        {
            **metadata,
            "checkpoint": {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "schema_version": SOURCE_CHECKPOINT_SCHEMA_VERSION,
                "kind": SOURCE_CHECKPOINT_KIND,
            },
        },
    )


class CandidateTinyDeiT(nn.Module):
    """Tiny DeiT whose 12 blocks carry real-size weight tensors.

    ``forward`` reads every weight element of the candidate blocks (9, 10,
    11) so that the SHOT loss produces nonzero gradients across the whole
    candidate scope.  Blocks 0-8 exist only so ``apply_update_scope`` can
    freeze them.
    """

    total_forward_calls = 0

    def __init__(self, num_classes, **kwargs):
        super().__init__()
        del kwargs
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 384))
        self.pos_embed = nn.Parameter(torch.zeros(1, 2, 384))
        self.head = nn.Linear(384, num_classes)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": nn.ModuleDict(
                            {
                                "qkv": nn.Linear(384, 1152),
                                "proj": nn.Linear(384, 384),
                            }
                        ),
                        "mlp": nn.ModuleDict(
                            {
                                "fc1": nn.Linear(384, 1536),
                                "fc2": nn.Linear(1536, 384),
                            }
                        ),
                    }
                )
                for _ in range(12)
            ]
        )

    def get_classifier(self):
        return self.head

    def forward(self, images):
        type(self).total_forward_calls += 1
        batch = images.size(0)
        device = images.device
        dtype = images.dtype
        base = images.mean(dim=(1, 2, 3))  # per-sample feature scalar
        extra = torch.zeros(batch, device=device, dtype=dtype)
        for index in (9, 10, 11):
            block = self.blocks[index]
            # Fold each candidate weight into the features fed to the (frozen)
            # head so the SHOT loss is not invariant to these perturbations.
            extra = extra + base * block["attn"]["qkv"].weight.sum() / 1152.0
            extra = extra + base * block["attn"]["proj"].weight.sum() / 384.0
            extra = extra + base * block["mlp"]["fc1"].weight.sum() / 1536.0
            extra = extra + base * block["mlp"]["fc2"].weight.sum() / 384.0
        features = torch.zeros(batch, 384, device=device, dtype=dtype)
        features[:, 0] = base + 0.01 * extra
        return self.head(features)


def candidate_tiny_factory(model_name, **kwargs):
    assert model_name == "deit_small_patch16_224.fb_in1k"
    return CandidateTinyDeiT(**kwargs)


def install_candidate_fake_checkpoint(root):
    """Overwrite amazon.pth with a candidate-shaped fake W0."""
    install_real_fake_checkpoint(root)
    path = Path(root) / "checkpoints" / "office31" / "amazon.pth"
    model = CandidateTinyDeiT(num_classes=31)
    metadata = checkpoint_metadata("office31", "amazon", 31)
    torch.save(
        {
            "schema_version": SOURCE_CHECKPOINT_SCHEMA_VERSION,
            "kind": SOURCE_CHECKPOINT_KIND,
            "state_dict": model.state_dict(),
            "metadata": metadata,
        },
        path,
    )
    write_json(
        path.with_suffix(".manifest.json"),
        {
            **metadata,
            "checkpoint": {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "schema_version": SOURCE_CHECKPOINT_SCHEMA_VERSION,
                "kind": SOURCE_CHECKPOINT_KIND,
            },
        },
    )


def _base(root):
    return prepare_assets(root, CONFIG_PATH, runs_directory="full_dense_runs")


def _candidate_base(root):
    return prepare_assets(
        root, CANDIDATE_CONFIG_PATH, runs_directory="candidate_dense_runs"
    )


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
    assert scientific["adaptation"]["update_scope"] == "all_except_head"
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
    _expect_value_error(bad, "variant must be one of")
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
    install_dense_fake_checkpoint(root)
    effective = resolve_config(base, PROJECT_ROOT)
    DenseTinyDeiT.total_forward_calls = 0
    summary = run_deit_otta_adaptation_experiment(
        effective, PROJECT_ROOT, model_factory=dense_tiny_factory
    )
    assert summary["status"] == "completed"
    assert summary["variant"] == "full_dense"
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
    # full-dense (all_except_head) freezes the classifier: only cls_token
    # (384) and pos_embed (768) are trainable and both get gradients, while
    # the head (11,935) stays exactly at W0.
    assert summary["total_parameter_scalars"] == 13_087
    assert summary["trainable_scalars"] == 1_152
    assert summary["candidate_scalars"] == 1_152
    assert summary["delta_nonzero_scalars"] == 1_152
    assert summary["delta_nonzero_scalars"] < summary[
        "total_parameter_scalars"
    ]
    assert summary["delta_l2_norm"] > 0.0
    assert math.isfinite(summary["delta_l2_norm"])
    assert 0.0 <= summary["PU-Acc"] <= 100.0
    assert 0.0 <= summary["FO-Acc"] <= 100.0
    # Two adaptation forwards (loss + PU prediction) per batch plus one FO
    # inference forward per batch: 3 * 2 = 6 forwards in total.
    assert DenseTinyDeiT.total_forward_calls == 6
    output = Path(summary["output_dir"])
    assert (output / "manifest.json").is_file()
    assert (output / "summary.json").is_file()
    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["protocol"] == (
        "sequential_target_stream_full_dense_adaptation"
    )
    assert manifest["adaptation"]["update_scope"] == "all_except_head"
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


def _check_candidate_config_contract(base):
    effective = resolve_config(base, PROJECT_ROOT)
    assert effective["variant"] == "candidate_dense"
    assert effective["adaptation"]["update_scope"] == "last_3_block_weights"
    assert effective["adaptation"]["candidate_blocks"] == [9, 10, 11]
    scientific = effective["scientific_config"]
    assert scientific["protocol"] == (
        "sequential_target_stream_candidate_dense_adaptation"
    )
    assert scientific["adaptation"]["candidate_blocks"] == [9, 10, 11]
    assert scientific["adaptation"]["candidate_parameter_names"] == sorted(
        candidate_parameter_names([9, 10, 11])
    )
    assert len(scientific["adaptation"]["candidate_parameter_names"]) == 12
    assert "candidate_dense" in effective["experiment_key"]
    return effective


def _check_candidate_identity_separation(full_base, cand_base):
    full_effective = resolve_config(full_base, PROJECT_ROOT)
    cand_effective = resolve_config(cand_base, PROJECT_ROOT)
    assert full_effective["experiment_key"] != cand_effective["experiment_key"]
    assert full_effective["experiment_config_sha256"] != cand_effective[
        "experiment_config_sha256"
    ]


def _check_candidate_validation(base):
    bad = copy.deepcopy(base)
    bad["variant"] = "something"
    _expect_value_error(bad, "variant must be one of")
    bad = copy.deepcopy(base)
    bad["adaptation"]["update_scope"] = "all_parameters"
    _expect_value_error(bad, "last_3_block_weights")
    bad = copy.deepcopy(base)
    del bad["adaptation"]["candidate_blocks"]
    _expect_value_error(bad, "candidate_blocks")
    bad = copy.deepcopy(base)
    bad["adaptation"]["candidate_blocks"] = [9, 9, 11]
    _expect_value_error(bad, "unique")
    bad = copy.deepcopy(base)
    bad["adaptation"]["candidate_blocks"] = [9, 10, 13]
    _expect_value_error(bad, "in [0, 12)")


def _check_freeze_scope():
    model = CandidateTinyDeiT(num_classes=31)
    expected = candidate_parameter_names([9, 10, 11])
    trainable = apply_update_scope(
        model, "last_3_block_weights", [9, 10, 11]
    )
    actual = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable == expected
    assert actual == set(expected)
    assert len(actual) == 12
    assert any(name.startswith("blocks.9.") for name in actual)
    assert any(name.startswith("blocks.10.") for name in actual)
    assert any(name.startswith("blocks.11.") for name in actual)
    for name, parameter in model.named_parameters():
        if name not in actual:
            assert not parameter.requires_grad, name
    model = CandidateTinyDeiT(num_classes=31)
    all_trainable = apply_update_scope(model, "all_parameters")
    assert all(
        parameter.requires_grad for parameter in model.parameters()
    )
    assert len(all_trainable) == len(list(model.named_parameters()))
    # all_except_head (full-dense semantic): classifier frozen, everything
    # else trainable.
    model = DenseTinyDeiT(num_classes=31)
    dense_trainable = apply_update_scope(model, "all_except_head")
    assert "head.weight" not in dense_trainable
    assert "head.bias" not in dense_trainable
    assert set(dense_trainable) == {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert len(dense_trainable) == 2
    assert not model.head.weight.requires_grad
    assert not model.head.bias.requires_grad


def _check_candidate_end_to_end(root, base):
    install_candidate_fake_checkpoint(root)
    effective = resolve_config(base, PROJECT_ROOT)
    CandidateTinyDeiT.total_forward_calls = 0
    summary = run_deit_otta_adaptation_experiment(
        effective, PROJECT_ROOT, model_factory=candidate_tiny_factory
    )
    assert summary["status"] == "completed"
    assert summary["variant"] == "candidate_dense"
    assert summary["adaptation_steps"] == 2
    assert summary["backward_calls"] == 2
    assert summary["model_state_unchanged"] is False
    assert summary["model_state_sha256_before"] != summary[
        "model_state_sha256_after_stream"
    ]
    assert summary["model_state_sha256_after_stream"] == summary[
        "model_state_sha256_after_fo"
    ]
    assert summary["processed_sample_count"] == 31
    assert summary["target_batch_count"] == 2
    assert summary["trainable_scalars"] == CANDIDATE_SCALARS
    assert summary["candidate_scalars"] == CANDIDATE_SCALARS
    # Every candidate scalar moves (gradients plus AdamW decay); the frozen
    # cls/pos/head/block0-8 parameters stay exactly at their W0 values.
    assert summary["delta_nonzero_scalars"] == CANDIDATE_SCALARS
    assert summary["total_parameter_scalars"] == 21_288_223
    assert summary["delta_nonzero_scalars"] < summary[
        "total_parameter_scalars"
    ]
    assert math.isfinite(summary["delta_l2_norm"])
    assert summary["delta_l2_norm"] > 0.0
    assert 0.0 <= summary["PU-Acc"] <= 100.0
    assert 0.0 <= summary["FO-Acc"] <= 100.0
    assert CandidateTinyDeiT.total_forward_calls == 6
    output = Path(summary["output_dir"])
    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["variant"] == "candidate_dense"
    assert manifest["protocol"] == (
        "sequential_target_stream_candidate_dense_adaptation"
    )
    assert manifest["adaptation"]["candidate_blocks"] == [9, 10, 11]
    assert len(manifest["adaptation"]["candidate_parameter_names"]) == 12
    metrics = [
        json.loads(line)
        for line in (output / "metrics.jsonl").read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len([row for row in metrics if row["event"] == "online_batch"]) == 2
    assert len([row for row in metrics if row["event"] == "final"]) == 1
    assert [row for row in metrics if row["event"] == "final"][0][
        "status"
    ] == "completed"
    return summary


def main():
    with tempfile.TemporaryDirectory(prefix="deit_otta_full_dense_") as root:
        base = _base(root)
        cand_base = _candidate_base(root)
        _check_config_contract(base)
        _check_identity_binds_optimizer(base)
        _check_validation(base)
        _check_freeze_scope()
        _check_candidate_config_contract(cand_base)
        _check_candidate_identity_separation(base, cand_base)
        _check_candidate_validation(cand_base)
        _check_end_to_end(root, base)
        _check_candidate_end_to_end(root, cand_base)
    print("DeiT OTTA adaptation (full-dense + candidate-dense) tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
