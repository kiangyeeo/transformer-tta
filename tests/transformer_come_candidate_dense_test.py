#!/usr/bin/env python3
"""CPU contracts for the COME candidate-dense baseline (C09-C11, C25)."""

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

import transformer_come.runner as runner_module  # noqa: E402
import transformer_come.common as come_common  # noqa: E402
from transformer.candidate_dense.config import candidate_parameter_names  # noqa: E402
from transformer_come.config import (  # noqa: E402
    CANDIDATE_SCALAR_COUNT,
    CANDIDATE_TENSOR_COUNT,
    IMPLEMENTATION_REVISIONS,
    PROTOCOL_REVISIONS,
    validate_variant_config,
    variant_config_view,
    load_config,
)

from transformer_come_fixtures import (  # noqa: E402
    BATCH_SIZE,
    CandidateShapedModel,
    SAMPLE_COUNT,
    build_synthetic_loaders,
    load_candidate_scope_model,
    read_metrics,
    synthetic_config,
)


VARIANT = "candidate_dense"
CONFIG_PATH = PROJECT_ROOT / "transformer_come" / "config.yaml"


def check_config() -> None:
    config = variant_config_view(load_config(CONFIG_PATH), VARIANT)
    validate_variant_config(config, variant=VARIANT)
    assert config["protocol_revision"] == PROTOCOL_REVISIONS[VARIANT]
    assert config["method"] == "come"
    assert config["adaptation"]["candidate_blocks"] == [9, 10, 11]
    assert config["adaptation"]["candidate_tensor_count"] == CANDIDATE_TENSOR_COUNT == 12
    assert config["adaptation"]["candidate_scalar_count"] == CANDIDATE_SCALAR_COUNT
    assert CANDIDATE_SCALAR_COUNT == 5_308_416


class CountingAdamW(torch.optim.AdamW):
    instances: list = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_calls = 0
        type(self).instances.append(self)

    def step(self, *args, **kwargs):
        self.step_calls += 1
        return super().step(*args, **kwargs)


def _run():
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
    runner_module.build_target_loaders = build_synthetic_loaders
    try:
        with tempfile.TemporaryDirectory(prefix="come_candidate_dense_") as value:
            output_dir = Path(value) / "run"
            config = synthetic_config("candidate_dense", output_dir=output_dir)
            summary = runner_module.run_transfer(
                config,
                PROJECT_ROOT,
                show_progress=False,
                model_loader=load_candidate_scope_model,
            )
            return summary, read_metrics(output_dir), objective_calls["count"]
    finally:
        come_common.come_loss = original_come_loss
        torch.optim.AdamW = original_adamw
        runner_module.build_target_loaders = original_loaders


def check_candidate_scope_and_counts() -> dict:
    summary, rows, objective_calls = _run()
    batch_count = SAMPLE_COUNT // BATCH_SIZE
    online = [row for row in rows if row["event"] == "online_batch"]

    assert summary["status"] == "completed"
    assert summary["implementation_revision"] == IMPLEMENTATION_REVISIONS[VARIANT]
    assert objective_calls == batch_count
    assert summary["objective_call_count"] == batch_count
    assert summary["optimizer_step_count"] == batch_count
    assert summary["scheduler_step_count"] == 0
    assert len(CountingAdamW.instances) == 1
    assert CountingAdamW.instances[0].step_calls == batch_count
    assert len(online) == batch_count

    # C10/C11: exactly the frozen 12 candidate tensors are trainable.
    assert summary["trainable_parameter_names"] == list(candidate_parameter_names())
    assert summary["trainable_tensor_count"] == 12
    assert summary["trainable_scalars"] == CANDIDATE_SCALAR_COUNT
    for name in ("head.weight", "head.bias", "norm.weight", "norm.bias", "cls_token", "pos_embed"):
        assert name in summary["frozen_parameter_names"], name
    assert summary["frozen_state_unchanged"] is True
    assert summary["frozen_head_unchanged"] is True
    assert summary["candidate_state_sha256_before"] != summary[
        "candidate_state_sha256_after_stream"
    ]
    return {"batch_count": batch_count, "trainable_scalars": summary["trainable_scalars"]}


def check_only_candidate_tensors_move() -> dict:
    """Off-scope parameters must be exactly equal to W0 after the stream."""

    captured = {}

    def capturing_loader(config, device):
        loaded = load_candidate_scope_model(config, device)
        captured["model"] = loaded[0]
        return loaded

    original_loaders = runner_module.build_target_loaders
    runner_module.build_target_loaders = build_synthetic_loaders
    try:
        with tempfile.TemporaryDirectory(prefix="come_candidate_dense_scope_") as value:
            output_dir = Path(value) / "run"
            runner_module.run_transfer(
                synthetic_config("candidate_dense", output_dir=output_dir),
                PROJECT_ROOT,
                show_progress=False,
                model_loader=capturing_loader,
            )
    finally:
        runner_module.build_target_loaders = original_loaders

    reference = CandidateShapedModel()
    reference_state = dict(reference.named_parameters())
    candidate_names = set(candidate_parameter_names())
    adapted = dict(captured["model"].named_parameters())
    assert set(adapted) == set(reference_state)
    changed = 0
    for name, parameter in adapted.items():
        if name in candidate_names:
            if not torch.equal(parameter.detach(), reference_state[name].detach()):
                changed += 1
        else:
            # Exact equality, not allclose: an off-scope tensor may not drift
            # by a single ULP.
            assert torch.equal(
                parameter.detach(), reference_state[name].detach()
            ), name
    assert changed == len(candidate_names)
    for name, buffer in captured["model"].named_buffers():
        assert torch.equal(buffer, dict(reference.named_buffers())[name]), name
    return {"changed_candidate_tensors": changed}


def main() -> int:
    check_config()
    report = check_candidate_scope_and_counts()
    scope = check_only_candidate_tensors_move()
    print(json.dumps({"status": "PASS", **report, **scope}, sort_keys=True))
    print("Transformer COME candidate-dense tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
