#!/usr/bin/env python3
"""CPU contracts for the COME full-dense baseline (C09, C10, C16, C25, C28)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(PROJECT_ROOT), str(PROJECT_ROOT / "tests")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import transformer_come.common as come_common  # noqa: E402
import transformer_come.dense_runner as dense_runner  # noqa: E402
import transformer_come.full_dense.runner as runner_module  # noqa: E402
from transformer_come.full_dense.config import (  # noqa: E402
    FORMAL_SEED,
    IMPLEMENTATION_REVISION,
    PROTOCOL_REVISION,
    TRANSFERS,
    _validate_frozen_fields,
    load_config,
    select_transfers,
)
from transformer_come.full_dense.data import build_transforms  # noqa: E402

from transformer_come_fixtures import (  # noqa: E402
    BATCH_SIZE,
    CandidateShapedModel,
    SAMPLE_COUNT,
    build_synthetic_loaders,
    load_full_dense_scope_model,
    read_metrics,
    synthetic_config,
)


CONFIG_PATH = PROJECT_ROOT / "transformer_come" / "full_dense" / "config.yaml"


def check_config_and_transforms() -> None:
    config = load_config(CONFIG_PATH)
    _validate_frozen_fields(config)
    assert config["formal_seed"] == FORMAL_SEED == 2026
    assert config["protocol_revision"] == PROTOCOL_REVISION
    assert config["method"] == "come"
    assert config["data"]["office31"]["batch_size"] == 64
    assert config["data"]["office31"]["fo_batch_size"] == 64
    assert config["data"]["visda-c"]["batch_size"] == 256
    assert config["data"]["visda-c"]["fo_batch_size"] == 256
    assert config["data"]["stream"]["drop_last"] is False
    assert len(TRANSFERS) == 7
    assert len(select_transfers("office31")) == 6
    assert select_transfers("visda-c") == (("visda-c", "train", "validation"),)

    online, final = build_transforms(config["data"]["preprocessing"])
    assert [type(item) for item in online.transforms] == [
        transforms.Resize,
        transforms.RandomCrop,
        transforms.RandomHorizontalFlip,
        transforms.ToTensor,
        transforms.Normalize,
    ]
    assert [type(item) for item in final.transforms] == [
        transforms.Resize,
        transforms.CenterCrop,
        transforms.ToTensor,
        transforms.Normalize,
    ]
    for pipeline in (online, final):
        assert pipeline.transforms[0].size == (256, 256)
        assert pipeline.transforms[0].interpolation == InterpolationMode.BILINEAR


class CountingAdamW(torch.optim.AdamW):
    instances: list = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_calls = 0
        type(self).instances.append(self)

    def step(self, *args, **kwargs):
        self.step_calls += 1
        return super().step(*args, **kwargs)


def _run(config_overrides=None, *, model_loader=load_full_dense_scope_model, loaders=None):
    """Run one synthetic full-dense COME stream with instrumented counters."""

    CandidateShapedModel.forward_calls = 0
    CountingAdamW.instances = []
    objective_calls = {"count": 0}
    original_come_loss = come_common.come_loss
    original_adamw = torch.optim.AdamW
    original_loaders = runner_module.build_target_loaders

    def counting_come_loss(logits, class_count, **kwargs):
        objective_calls["count"] += 1
        return original_come_loss(logits, class_count, **kwargs)

    come_common.come_loss = counting_come_loss
    torch.optim.AdamW = CountingAdamW
    runner_module.build_target_loaders = loaders or build_synthetic_loaders
    try:
        with tempfile.TemporaryDirectory(prefix="come_full_dense_") as value:
            output_dir = Path(value) / "run"
            config = synthetic_config("full_dense", output_dir=output_dir)
            if config_overrides:
                config.update(config_overrides)
            summary = runner_module.run_transfer(
                config,
                PROJECT_ROOT,
                show_progress=False,
                model_loader=model_loader,
            )
            rows = read_metrics(output_dir)
            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            return summary, rows, manifest, objective_calls["count"]
    finally:
        come_common.come_loss = original_come_loss
        torch.optim.AdamW = original_adamw
        runner_module.build_target_loaders = original_loaders


def check_counts_and_state() -> dict:
    summary, rows, manifest, objective_calls = _run()
    batch_count = SAMPLE_COUNT // BATCH_SIZE
    online = [row for row in rows if row["event"] == "online_batch"]
    fo_rows = [row for row in rows if row["event"] == "fo_batch"]

    assert summary["status"] == "completed"
    assert summary["result_validity"] == "valid"
    assert summary["method"] == manifest["method"] == "come"
    assert summary["implementation_revision"] == IMPLEMENTATION_REVISION
    # C09: exactly one objective, one backward and one host step per batch,
    # and no scheduler at all.
    assert objective_calls == batch_count
    assert summary["objective_call_count"] == batch_count
    assert summary["backward_calls"] == batch_count
    assert summary["optimizer_step_count"] == batch_count
    assert summary["scheduler_step_count"] == 0
    assert summary["scheduler"] == "none"
    assert summary["support_selection_count"] == 0
    assert summary["omega_writeback_count"] == 0
    assert summary["native_ema_commit_count"] == 0
    assert all(row["objective_call_count"] == 1 for row in online)
    assert all(row["optimizer_step_count"] == 1 for row in online)
    assert all(row["scheduler_step_count"] == 0 for row in online)
    assert len(online) == batch_count
    assert [row["batch_size"] for row in online] == [BATCH_SIZE] * batch_count

    # A single persistent AdamW instance, stepped once per batch, with live
    # moments after the stream (not re-created per batch).
    assert len(CountingAdamW.instances) == 1
    optimizer = CountingAdamW.instances[0]
    assert optimizer.step_calls == batch_count
    moments = [
        state["exp_avg"]
        for state in optimizer.state.values()
        if "exp_avg" in state
    ]
    assert moments and any(torch.count_nonzero(value).item() for value in moments)
    assert optimizer.param_groups[0]["lr"] == 1.0e-3
    assert all(row["lr"] == 1.0e-3 for row in online)

    # Adaptation forward + PU forward per batch, then the independent FO pass.
    assert CandidateShapedModel.forward_calls == 2 * batch_count + len(fo_rows)
    assert summary["PU-sample-count"] == SAMPLE_COUNT
    assert summary["FO-sample-count"] == SAMPLE_COUNT
    assert summary["pu_is_separate_read_only_forward"] is True
    assert summary["fo_is_independent_full_target_pass"] is True
    assert summary["model_state_sha256_after_stream"] == summary["model_state_sha256_after_fo"]
    assert summary["model_state_sha256_before"] != summary["model_state_sha256_after_stream"]
    assert summary["frozen_head_unchanged"] is True
    assert summary["frozen_state_unchanged"] is True
    assert summary["adapted_model_saved"] is False
    assert summary["singleton_batch_policy"] == "fail_closed_before_all_adaptation"
    assert summary["come"]["tau"] == 1.0
    assert summary["come_objective"]["come_norm_epsilon"] == "none"
    assert summary["come_objective"]["come_renormalize_after_epsilon"] is False
    return {
        "batch_count": batch_count,
        "forward_calls": CandidateShapedModel.forward_calls,
    }


def check_scope_and_collapse_diagnostics() -> dict:
    """C10/C28: the head stays frozen, the body updates, diagnostics exist."""
    summary, rows, _, _ = _run()
    online = [row for row in rows if row["event"] == "online_batch"]
    trainable = summary["trainable_parameter_names"]
    assert "head.weight" not in trainable and "head.bias" not in trainable
    for name in ("norm.weight", "norm.bias", "cls_token", "pos_embed"):
        assert name in trainable, name
    assert summary["frozen_head_parameter_names"] == ["head.weight", "head.bias"]
    assert summary["candidate_state_sha256_before"] != summary[
        "candidate_state_sha256_after_stream"
    ]

    required = {
        "predicted_class_histogram",
        "predicted_class_count",
        "dominant_class_count",
        "dominant_class_ratio",
        "mean_softmax_entropy",
        "come_opinion_entropy",
        "come_mean_uncertainty_mass",
    }
    for row in online:
        assert required <= set(row), sorted(required - set(row))
        assert len(row["predicted_class_histogram"]) == 31
        assert sum(row["predicted_class_histogram"]) == row["batch_size"]
        assert 0.0 < row["dominant_class_ratio"] <= 1.0
        assert 0.0 <= row["come_mean_uncertainty_mass"] <= 1.0
    for key in (
        "collapse_diagnostics_last_batch",
        "predicted_class_count_min",
        "dominant_class_ratio_mean",
        "come_opinion_entropy_mean",
        "come_mean_uncertainty_mass_mean",
    ):
        assert key in summary, key
    return {"predicted_class_count_min": summary["predicted_class_count_min"]}


def check_determinism_and_eval_mode() -> None:
    """C16: same state and inputs reproduce the same stream."""
    first, _, _, _ = _run()
    second, _, _, _ = _run()
    assert first["model_state_sha256_after_stream"] == second[
        "model_state_sha256_after_stream"
    ]
    assert first["PU-Acc"] == second["PU-Acc"]
    assert first["FO-Acc"] == second["FO-Acc"]
    assert first["come_opinion_entropy_mean"] == second["come_opinion_entropy_mean"]

    # The model never leaves eval mode, and no dropout/stochastic module is
    # active in the fixture scope.
    model = CandidateShapedModel()
    model.eval()
    with torch.no_grad():
        images = torch.randn(3, 384)
        assert torch.equal(model(images), model(images))


def check_failure_is_recorded_as_invalid() -> None:
    """A protocol violation must land in the artifact with a reason code."""

    def broken_loader(config, device):
        model, checkpoint, trainable, frozen, scope = load_full_dense_scope_model(
            config, device
        )
        model.train()
        return model, checkpoint, trainable, frozen, scope

    try:
        _run(model_loader=broken_loader)
    except RuntimeError as error:
        assert "eval mode" in str(error)
    else:
        raise AssertionError("A train-mode model was accepted")


def check_real_deit_scope_and_objective() -> dict:
    """The COME objective on the real DeiT-S graph, without any checkpoint.

    Locally constructed (``pretrained=False``, no download) only to prove the
    architecture-level contracts: C = 31 is the classifier output dimension,
    every non-head parameter is reachable by the COME gradient, the head is
    not, and no Dropout/DropPath module is active.
    """

    import timm
    import torch.nn as nn

    from transformer_come.common import come_objective_step

    model = timm.create_model(
        "deit_small_patch16_224.fb_in1k",
        pretrained=False,
        num_classes=31,
        drop_rate=0.0,
        drop_path_rate=0.0,
    )
    model.eval()
    head_ids = {id(parameter) for parameter in model.get_classifier().parameters()}
    model.requires_grad_(True)
    model.get_classifier().requires_grad_(False)
    assert isinstance(model.get_classifier(), nn.Linear)
    assert model.get_classifier().in_features == 384
    assert model.get_classifier().out_features == 31
    assert not [
        module
        for module in model.modules()
        if isinstance(module, nn.Dropout) and module.p > 0
    ]
    assert not [
        module for module in model.modules() if "DropPath" in type(module).__name__
    ]
    assert len(model.blocks) == 12

    torch.manual_seed(2026)
    loss, logits, diagnostics = come_objective_step(
        model, torch.randn(2, 3, 224, 224), 31
    )
    assert logits.shape == (2, 31)
    assert diagnostics["class_count_c"] == 31
    assert torch.isfinite(loss).item()
    without_gradient = [
        name
        for name, parameter in model.named_parameters()
        if id(parameter) not in head_ids and parameter.grad is None
    ]
    assert without_gradient == [], without_gradient
    assert all(
        torch.isfinite(parameter.grad).all().item()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    )
    assert all(
        parameter.grad is None
        for parameter in model.get_classifier().parameters()
    )
    candidate_scalars = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name
        in {
            f"blocks.{block}.{suffix}"
            for block in (9, 10, 11)
            for suffix in (
                "attn.qkv.weight",
                "attn.proj.weight",
                "mlp.fc1.weight",
                "mlp.fc2.weight",
            )
        }
    )
    assert candidate_scalars == 5_308_416
    return {"real_deit_candidate_scalars": candidate_scalars}


def main() -> int:
    check_config_and_transforms()
    counts = check_counts_and_state()
    diagnostics = check_scope_and_collapse_diagnostics()
    check_determinism_and_eval_mode()
    check_failure_is_recorded_as_invalid()
    real_model = check_real_deit_scope_and_objective()
    print(json.dumps({"status": "PASS", **counts, **diagnostics, **real_model}, sort_keys=True))
    print("Transformer COME full-dense tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
