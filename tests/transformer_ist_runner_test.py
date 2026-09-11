#!/usr/bin/env python3
"""CPU state-machine contracts for native IST and Random orchestration."""

from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformer_ist.data import MaterializedBatch  # noqa: E402
from transformer_ist.objective import FixedISTTask  # noqa: E402
from transformer_ist.runner import (  # noqa: E402
    _native_self_training,
    run_single,
    run_transfer,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Linear(4, 384)
        self.head = nn.Linear(384, 2)
        self.register_buffer("frozen_buffer", torch.tensor([3.0]))

    def forward(self, inputs):
        return self.head(torch.tanh(self.body(inputs)))


class TinyAdapter:
    def __init__(self, model):
        self.model = model

    def features_and_logits(self, inputs):
        features = torch.tanh(self.model.body(inputs))
        return features, self.model.head(features)

    def logits(self, inputs):
        return self.features_and_logits(inputs)[1]


def tiny_model_loader(config, device):
    del config
    model = TinyModel().to(device)
    model.requires_grad_(False)
    model.body.requires_grad_(True)
    model.eval()
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    frozen = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    ]
    return (
        model,
        TinyAdapter(model),
        {"path": "synthetic", "sha256": "0" * 64},
        trainable,
        frozen,
        {"feature_shape": [None, 384], "model_mode": "eval"},
    )


class FakeMaterializer:
    def __init__(self, config):
        del config
        self.reference_generator = torch.Generator().manual_seed(12026)
        self.adaptation_generator = torch.Generator().manual_seed(22026)

    def materialize(self, raw_images, sample_indices):
        del raw_images
        references = []
        adaptations = []
        for index in torch.as_tensor(sample_indices).tolist():
            base = torch.tensor(
                [float(index + 1), float(index == 0), 0.5, -0.25]
            )
            references.append(base)
            for view in range(8):
                adaptations.append(base + 0.01 * view)
        return MaterializedBatch(
            torch.stack(references),
            torch.stack(adaptations),
            torch.as_tensor(sample_indices),
            "a" * 64,
            "b" * 64,
            "c" * 64,
        )


def tiny_loaders(config):
    labels = torch.tensor(config.get("synthetic_metric_labels", [0, 1]))
    online = [
        (
            ["zero", "one"],
            labels,
            torch.tensor([0, 1]),
        )
    ]
    images = torch.tensor(
        [[1.0, 1.0, 0.5, -0.25], [2.0, 0.0, 0.5, -0.25]]
    )
    final = DataLoader(
        TensorDataset(images, torch.tensor([0, 1]), torch.tensor([0, 1])),
        batch_size=2,
    )
    return online, final, {
        "sampler": "fixed_random_permutation",
        "seed": 2026,
        "sample_count": 2,
        "batch_count": 1,
        "batch_size_sequence": [2],
        "online_order": [0, 1],
        "online_order_sha256": "online",
        "fo_order_sha256": "fo",
        "drop_last": False,
        "singleton_tail_merged_into_previous_batch": False,
    }


def base_config(output_dir):
    return {
        "schema_version": 1,
        "protocol_revision": "OTTA_IST_TRANSFORMER_LBI_PROTOCOL_20260909_v1",
        "implementation_revision": "ist_transformer_baseline_20260909_v1",
        "source_checkpoint_revision": "deit_source_checkpoint_manifest_v1",
        "method": "ist",
        "variant": "full_dense",
        "formal_seed": 2026,
        "formal_protocol": True,
        "debug_smoke": False,
        "debug_max_outer_batches": None,
        "dataset": "office31",
        "source": "amazon",
        "target": "dslr",
        "transfer": "amazon->dslr",
        "num_classes": 2,
        "class_names": ["zero", "one"],
        "batch_size": 2,
        "fo_batch_size": 2,
        "workers": 0,
        "target_list": "synthetic",
        "checkpoint_sha256": "0" * 64,
        "checkpoint_manifest_path": "synthetic",
        "preprocessing": {"interpolation": "bilinear"},
        "stream": {"one_pass": True, "drop_last": False},
        "ist": {
            "extend": 8,
            "iters": 1,
            "ema_momentum": 0.9,
            "feature_dim": 384,
            "memory": {"max_len": 10000},
            "plca": {
                "repeat": 1,
                "k": 50,
                "gamma": 3,
                "mode": "l2",
                "propagation_alpha": 0.99,
                "solver_max_steps": 20,
                "solver_rtol": 1.0e-6,
                "distance_chunk_size": 1024,
            },
            "rng": {"inner_order_seed_offset": 30000},
            "pre_inference_chunk_size": 4,
            "native_micro_batch_size": 1,
            "objective_chunk_size": 4,
        },
        "optimization": {
            "optimizer": "adamw",
            "lr": 1.0e-3,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "weight_decay": 0.01,
            "scheduler": "none",
        },
        "loss": {
            "hard_ce_weight": 1.0,
            "soft_kl_weight": 1.0,
        },
        "runtime": {
            "device": "cpu",
            "deterministic": True,
            "amp": False,
            "pin_memory": False,
            "save_model": False,
            "stream_checkpoint": False,
            "partial_resume": False,
        },
        "selection": None,
        "experiment_key": "synthetic-ist",
        "scientific_config_sha256": "1" * 64,
        "output_dir": str(output_dir),
    }


def check_runner_state_machine() -> None:
    with tempfile.TemporaryDirectory(prefix="transformer_ist_runner_") as value:
        config = base_config(Path(value) / "run")
        summary = run_single(
            config,
            PROJECT_ROOT,
            show_progress=False,
            model_loader=tiny_model_loader,
            loader_builder=tiny_loaders,
            materializer_factory=FakeMaterializer,
        )
    assert summary["processed_outer_batches"] == 1
    assert summary["adaptation_view_count"] == 16
    assert summary["pre_inference_task_count"] == 1
    assert summary["plca_call_count"] == 1
    assert summary["memory_commit_count"] == 1
    assert summary["native_ema_commit_count"] == 1
    assert summary["pu_forward_task_count"] == 1
    assert summary["host_optimizer_step_count"] == 8
    assert summary["scheduler_step_count"] == 0
    assert summary["pu_state_unchanged"] and summary["fo_state_unchanged"]
    assert summary["off_scope_exact"]
    assert summary["PU-sample-count"] == summary["FO-sample-count"] == 2
    assert summary["random_mask_index"] is None
    assert summary["mask_seed"] is None


def check_metric_labels_do_not_change_adaptation() -> None:
    with tempfile.TemporaryDirectory(prefix="transformer_ist_labels_") as value:
        first_config = base_config(Path(value) / "first")
        second_config = base_config(Path(value) / "second")
        second_config["synthetic_metric_labels"] = [1, 0]
        first = run_single(
            first_config,
            PROJECT_ROOT,
            show_progress=False,
            model_loader=tiny_model_loader,
            loader_builder=tiny_loaders,
            materializer_factory=FakeMaterializer,
        )
        second = run_single(
            second_config,
            PROJECT_ROOT,
            show_progress=False,
            model_loader=tiny_model_loader,
            loader_builder=tiny_loaders,
            materializer_factory=FakeMaterializer,
        )
    for field in (
        "model_state_sha256_after_stream",
        "augmentation_trace_history_sha256",
        "inner_order_history_sha256",
        "memory_commit_count",
    ):
        assert first[field] == second[field]


def check_scientific_inner_steps() -> None:
    model = nn.Linear(3, 2)
    adapter = model

    class Adapter:
        def logits(self, inputs):
            return adapter(inputs)

    for count, scientific_batch, expected_steps, expected_backwards in (
        (512, 64, 8, 8),
        (520, 64, 9, 9),
        (2048, 256, 8, 32),
    ):
        views = torch.randn(count, 3)
        task = FixedISTTask(
            views,
            torch.arange(count) % 2,
            torch.full((count, 2), 0.5),
        )
        config = {
            "batch_size": scientific_batch,
            "ist": {"iters": 1, "native_micro_batch_size": 64},
            "loss": {"hard_ce_weight": 1.0, "soft_kl_weight": 1.0},
        }
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
        result = _native_self_training(
            config,
            task,
            Adapter(),
            optimizer,
            list(model.named_parameters()),
            torch.Generator().manual_seed(32026),
        )
        assert result["inner_optimizer_steps"] == expected_steps
        assert result["native_physical_backward_count"] == expected_backwards


def check_random_three_real_children() -> None:
    with tempfile.TemporaryDirectory(prefix="transformer_ist_random_") as value:
        config = base_config(Path(value) / "random")
        config.update(
            {
                "variant": "group_random",
                "implementation_revision": "ist_transformer_sparse_lbi_20260909_v1",
                "selection": {
                    "requested_budget": 0.001,
                    "requested_group_count": 6,
                },
            }
        )
        calls = []

        def fake_child(child, project_root, show_progress=True):
            del project_root, show_progress
            calls.append(
                (
                    child["random_mask_index"],
                    child["mask_seed"],
                    child["output_dir"],
                )
            )
            return {
                "result_validity": "valid",
                "stream": {"online_order_sha256": "same"},
                "augmentation_trace_history_sha256": "views",
                "inner_order_history_sha256": "inner",
                "processed_outer_batches": 1,
                "PU-Acc": 50.0 + child["random_mask_index"],
                "FO-Acc": 60.0 + child["random_mask_index"],
                "PU-overall-Acc": 50.0,
                "FO-overall-Acc": 60.0,
                "online_batch_runtime_mean_sec": 1.0,
                "online_compute_runtime_sec": 1.0,
                "fo_eval_runtime_sec": 1.0,
                "gpu_peak_allocated_max_mb": 2.0,
                "gpu_peak_allocated_max_bytes": 2 * 1048576,
                "gpu_peak_reserved_max_mb": 3.0,
                "gpu_peak_reserved_max_bytes": 3 * 1048576,
                "class-names": ["zero", "one"],
                "checkpoint": {"path": "synthetic", "sha256": "0" * 64},
                "scope": {"trainable_scalars": 8},
                "optimization": child["optimization"],
                "loss": child["loss"],
                "ist": child["ist"],
                "host_optimizer_step_count": 8,
                "pre_inference_task_count": 1,
                "plca_call_count": 1,
                "memory_commit_count": 1,
                "native_ema_commit_count": 1,
                "pu_forward_task_count": 1,
                "off_scope_exact": True,
                "pu_state_unchanged": True,
                "fo_state_unchanged": True,
                "random_mask_index": child["random_mask_index"],
                "mask_seed": child["mask_seed"],
                "static_selection": {"mask_sha256": str(child["mask_seed"])},
                "output_dir": child["output_dir"],
            }

        with mock.patch("transformer_ist.runner.run_single", side_effect=fake_child):
            summary = run_transfer(config, PROJECT_ROOT, show_progress=False)
        assert [(index, seed) for index, seed, _ in calls] == [
            (0, 202600),
            (1, 202601),
            (2, 202602),
        ]
        assert len({path for _, _, path in calls}) == 3
        assert summary["num_random_masks"] == 3
        assert summary["PU-Acc"] == 51.0


def main() -> int:
    check_runner_state_machine()
    check_metric_labels_do_not_change_adaptation()
    check_scientific_inner_steps()
    check_random_three_real_children()
    print("Transformer IST runner contracts passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
