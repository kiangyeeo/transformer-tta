#!/usr/bin/env python3
"""CPU contracts for the isolated COME mitigation pilot."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformer_come_mitigation.experiments import (  # noqa: E402
    EXPERIMENTS,
    EXPERIMENT_SUITES,
    LR_REFINEMENT_EXPERIMENTS,
    MECHANISM_CONTROL_EXPERIMENTS,
    TAU_REFINEMENT_EXPERIMENTS,
    experiment_by_id,
)
from transformer_come_mitigation.runner import (  # noqa: E402
    configure_layernorm_affine_scope,
    configurable_objective_step,
    experimental_hooks,
)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.head = nn.Linear(4, 3)

    def forward(self, inputs):
        return self.head(self.norm(self.proj(inputs)))

    def get_classifier(self):
        return self.head


def main():
    assert len(EXPERIMENTS) == 8
    assert len({item["id"] for item in EXPERIMENTS}) == 8
    assert {item["scope"] for item in EXPERIMENTS} == {
        "full_dense",
        "layernorm_affine",
    }
    assert any(item.get("gradient_clip_norm") == 1.0 for item in EXPERIMENTS)
    assert any(item["tau"] == 0.25 for item in EXPERIMENTS)
    assert len(TAU_REFINEMENT_EXPERIMENTS) == 4
    assert {(item["tau"], item["lr"]) for item in TAU_REFINEMENT_EXPERIMENTS} == {
        (0.9, 1.0e-6),
        (0.9, 3.0e-6),
        (0.8, 1.0e-6),
        (0.8, 3.0e-6),
    }
    assert all(item["scope"] == "full_dense" for item in TAU_REFINEMENT_EXPERIMENTS)
    assert len(LR_REFINEMENT_EXPERIMENTS) == 3
    assert {item["lr"] for item in LR_REFINEMENT_EXPERIMENTS} == {
        1.0e-7,
        3.0e-7,
        5.0e-7,
    }
    assert all(item["tau"] == 1.0 for item in LR_REFINEMENT_EXPERIMENTS)
    assert all(item["scope"] == "full_dense" for item in LR_REFINEMENT_EXPERIMENTS)
    assert set(EXPERIMENT_SUITES) == {
        "initial-8gpu",
        "tau-refinement",
        "lr-refinement",
        "mechanism-controls",
    }
    assert experiment_by_id("T09_full_lr1e-6_tau0.9")["tau"] == 0.9
    assert experiment_by_id("LR5e-7_full_tau1")["lr"] == 5.0e-7
    assert len(MECHANISM_CONTROL_EXPERIMENTS) == 2
    assert {item["scope"] for item in MECHANISM_CONTROL_EXPERIMENTS} == {
        "candidate_dense",
        "full_dense",
    }
    assert sum(
        bool(item["reset_adam_moments_each_batch"])
        for item in MECHANISM_CONTROL_EXPERIMENTS
    ) == 1

    model = ToyModel().eval()
    trainable, frozen, record = configure_layernorm_affine_scope(model)
    assert [name for name, _ in trainable] == ["norm.weight", "norm.bias"]
    assert set(name for name, _ in frozen) == {
        "proj.weight",
        "proj.bias",
        "head.weight",
        "head.bias",
    }
    assert record["trainable_tensor_count"] == 2
    assert not any(parameter.requires_grad for parameter in model.head.parameters())

    loss, _, diagnostics = configurable_objective_step(tau=0.5)(
        model, torch.randn(5, 4), 3
    )
    assert torch.isfinite(loss)
    assert diagnostics["come_tau"] == 0.5
    assert all(parameter.grad is not None for _, parameter in trainable)
    assert all(parameter.grad is None for _, parameter in frozen)

    adam_experiment = next(item for item in EXPERIMENTS if item["id"].startswith("G_"))
    parameter = nn.Parameter(torch.tensor([1.0]))
    with experimental_hooks(adam_experiment):
        optimizer = torch.optim.AdamW(
            [parameter], lr=1.0e-3, betas=(0.9, 0.999), eps=1.0e-8,
            weight_decay=0.01,
        )
        parameter.grad = torch.tensor([100.0])
        optimizer.step()
    assert torch.isfinite(parameter)

    reset_experiment = next(
        item
        for item in MECHANISM_CONTROL_EXPERIMENTS
        if item["reset_adam_moments_each_batch"]
    )
    parameter = nn.Parameter(torch.tensor([1.0]))
    with experimental_hooks(reset_experiment):
        optimizer = torch.optim.AdamW(
            [parameter], lr=1.0e-3, betas=(0.9, 0.999), eps=1.0e-8,
            weight_decay=0.0,
        )
        parameter.grad = torch.tensor([1.0])
        optimizer.step()
        parameter.grad = torch.tensor([2.0])
        optimizer.step()
        state = optimizer.state[parameter]
        assert int(state["step"].item()) == 2
        assert torch.allclose(state["exp_avg"], torch.tensor([0.2]))
        assert torch.allclose(state["exp_avg_sq"], torch.tensor([0.004]))
        assert optimizer._come_moment_reset_calls == 2

    sgd_experiment = next(item for item in EXPERIMENTS if item["optimizer"] == "sgd")
    parameter = nn.Parameter(torch.tensor([1.0]))
    with experimental_hooks(sgd_experiment):
        optimizer = torch.optim.AdamW(
            [parameter], lr=1.0e-3, betas=(0.9, 0.999), eps=1.0e-8,
            weight_decay=0.0,
        )
        assert isinstance(optimizer, torch.optim.SGD)
        parameter.grad = torch.tensor([1.0])
        optimizer.step()
    print("Transformer COME mitigation pilot tests passed")


if __name__ == "__main__":
    main()
