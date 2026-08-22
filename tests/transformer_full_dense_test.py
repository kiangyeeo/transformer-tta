#!/usr/bin/env python3
"""CPU-only protocol checks for the Transformer full-dense baseline."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformer.full_dense.aggregate import aggregate_matrix, write_aggregate  # noqa: E402
from transformer.full_dense.config import (  # noqa: E402
    FORMAL_SEED,
    PROTOCOL_REVISION,
    TRANSFERS,
    _validate_frozen_fields,
    load_config,
    select_transfers,
)
from transformer.full_dense.data import build_transforms  # noqa: E402
from transformer.full_dense.loss import shot_loss  # noqa: E402
import transformer.full_dense.runner as runner_module  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "transformer" / "full_dense" / "config.yaml"


def check_config_and_transforms() -> None:
    config = load_config(CONFIG_PATH)
    _validate_frozen_fields(config)
    assert config["formal_seed"] == FORMAL_SEED == 2026
    assert config["protocol_revision"] == PROTOCOL_REVISION
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
        assert list(pipeline.transforms[-1].mean) == [0.485, 0.456, 0.406]
        assert list(pipeline.transforms[-1].std) == [0.229, 0.224, 0.225]

    changed = copy.deepcopy(config)
    changed["data"]["preprocessing"]["interpolation"] = "bicubic"
    try:
        _validate_frozen_fields(changed)
    except ValueError as error:
        assert "preprocessing" in str(error)
    else:
        raise AssertionError("A non-FC-aligned interpolation was accepted")


def check_loss() -> None:
    logits = torch.tensor([[2.0, -1.0], [-0.5, 1.0]], requires_grad=True)
    config = load_config(CONFIG_PATH)["loss"]
    loss, stats = shot_loss(logits, config)
    loss.backward()
    assert logits.grad is not None
    assert stats["pseudo_ratio"] == 1.0
    assert set(stats) == {
        "loss_cls",
        "loss_ent",
        "loss_div",
        "pseudo_ratio",
        "max_prob_mean",
    }


class OrderedSampler(Sampler[int]):
    def __init__(self, order):
        self.order = list(order)

    def __iter__(self):
        return iter(self.order)

    def __len__(self):
        return len(self.order)


class TinyDataset(Dataset):
    def __init__(self):
        self.images = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.5],
                [0.0, 1.0, 0.5, 0.0],
                [0.5, 0.0, 1.0, 0.0],
            ]
        )
        self.labels = torch.tensor([0, 1, 0])

    def __len__(self):
        return 3

    def __getitem__(self, index):
        return self.images[index], self.labels[index], index


class TinyDeiT(nn.Module):
    forward_calls = 0

    def __init__(self):
        super().__init__()
        self.body = nn.Linear(4, 3)
        self.head = nn.Linear(3, 2)
        with torch.no_grad():
            self.body.weight.copy_(
                torch.tensor(
                    [
                        [0.4, -0.1, 0.2, 0.3],
                        [-0.2, 0.5, 0.1, -0.4],
                        [0.3, 0.2, -0.5, 0.1],
                    ]
                )
            )
            self.head.weight.copy_(torch.tensor([[0.7, -0.4, 0.2], [-0.3, 0.6, -0.5]]))

    def get_classifier(self):
        return self.head

    def forward(self, images):
        type(self).forward_calls += 1
        return self.head(torch.tanh(self.body(images)))


def _tiny_model_loader(config, device):
    del config
    model = TinyDeiT().to(device)
    model.requires_grad_(True)
    model.head.requires_grad_(False)
    model.eval()
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    return model, {"path": "synthetic", "sha256": "0" * 64}, trainable, {
        "trainable_parameter_names": [name for name, _ in trainable],
        "frozen_head_parameter_names": ["head.weight", "head.bias"],
        "trainable_scalars": sum(parameter.numel() for _, parameter in trainable),
        "frozen_head_scalars": sum(parameter.numel() for parameter in model.head.parameters()),
        "total_parameter_scalars": sum(parameter.numel() for parameter in model.parameters()),
    }


def _tiny_loaders(config):
    dataset = TinyDataset()
    online = DataLoader(dataset, batch_size=2, sampler=OrderedSampler([2, 0, 1]), drop_last=False)
    final = DataLoader(dataset, batch_size=2, sampler=OrderedSampler([0, 1, 2]), drop_last=False)
    return online, final, {
        "sampler": "fixed_random_permutation",
        "seed": 2026,
        "sample_count": 3,
        "batch_count": 2,
        "online_batch_size": 2,
        "fo_batch_count": 2,
        "fo_batch_size": 2,
        "drop_last": False,
        "online_order_sha256": "synthetic-online",
        "fo_sampler": "sequential",
        "fo_order_sha256": "synthetic-fo",
    }


def check_end_to_end() -> None:
    TinyDeiT.forward_calls = 0
    config = {
        "output_dir": "",
        "runtime": {"device": "cpu", "deterministic": True, "amp": False, "pin_memory": False},
        "formal_seed": 2026,
        "protocol_revision": PROTOCOL_REVISION,
        "experiment_key": "synthetic-full-dense",
        "scientific_config_sha256": "1" * 64,
        "dataset": "office31",
        "source": "amazon",
        "target": "dslr",
        "transfer": "amazon->dslr",
        "num_classes": 2,
        "class_names": ["zero", "one"],
        "batch_size": 2,
        "fo_batch_size": 2,
        "adaptation": {"update_scope": "all_except_head", "model_mode": "eval", "steps_per_online_batch": 1},
        "optimization": {"optimizer": "adamw", "lr": 1.0e-3, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.01},
        "loss": {"components": ["ent", "div", "pseudo"], "cls_par": 0.3, "ent_par": 1.0, "threshold": 0.0},
        "preprocessing": {"interpolation": "bilinear"},
    }
    original_builder = runner_module.build_target_loaders
    runner_module.build_target_loaders = _tiny_loaders
    try:
        with tempfile.TemporaryDirectory(prefix="transformer_full_dense_") as value:
            config["output_dir"] = str(Path(value) / "run")
            summary = runner_module.run_transfer(
                config,
                PROJECT_ROOT,
                show_progress=False,
                model_loader=_tiny_model_loader,
            )
            assert summary["status"] == "completed"
            assert summary["adaptation_steps"] == 2
            assert summary["backward_calls"] == 2
            assert summary["frozen_head_unchanged"] is True
            assert summary["model_state_sha256_before"] != summary["model_state_sha256_after_stream"]
            assert summary["model_state_sha256_after_stream"] == summary["model_state_sha256_after_fo"]
            assert summary["PU-sample-count"] == 3
            assert summary["FO-sample-count"] == 3
            assert summary["online_batch_size"] == summary["fo_batch_size"] == 2
            assert summary["fo_batch_size_policy"] == "same_as_online_not_fc_times_three"
            # Two batches x (adapt + PU), plus two independent FO batches.
            assert TinyDeiT.forward_calls == 6
            rows = [
                json.loads(line)
                for line in (Path(config["output_dir"]) / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            online_rows = [row for row in rows if row["event"] == "online_batch"]
            assert [row["batch_size"] for row in online_rows] == [2, 1]
            assert all(row["adaptation_steps"] == 1 for row in online_rows)
            assert all(row["pu_is_separate_read_only_forward"] for row in online_rows)
    finally:
        runner_module.build_target_loaders = original_builder


def check_aggregate_outputs() -> None:
    with tempfile.TemporaryDirectory(prefix="transformer_full_dense_aggregate_") as value:
        root = Path(value)
        for index, (dataset, source, target) in enumerate(TRANSFERS):
            output = root / "results" / dataset / f"{source}-{target}"
            output.mkdir(parents=True)
            names = [f"class-{item}" for item in range(12)]
            summary = {
                "status": "completed",
                "variant": "full_dense",
                "dataset": dataset,
                "source": source,
                "target": target,
                "transfer": f"{source}->{target}",
                "formal_seed": 2026,
                "primary_metric": "sample_overall_accuracy" if dataset == "office31" else "fixed_12_class_macro_accuracy",
                "PU-Acc": float(index),
                "FO-Acc": float(index + 1),
                "PU-overall-Acc": float(index + 2),
                "FO-overall-Acc": float(index + 3),
                "output_dir": str(output),
            }
            if dataset == "visda-c":
                summary.update(
                    {
                        "PU-worst-class-Acc": 0.0,
                        "FO-worst-class-Acc": 1.0,
                        "PU-class-std": 2.0,
                        "FO-class-std": 3.0,
                        "PU-Acc-per-class-by-name": {name: float(i) for i, name in enumerate(names)},
                        "FO-Acc-per-class-by-name": {name: float(i + 1) for i, name in enumerate(names)},
                    }
                )
            (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        aggregate = aggregate_matrix(root)
        assert aggregate["office31"]["PU-Acc"] == 2.5
        assert aggregate["office31"]["FO-Acc"] == 3.5
        assert len(aggregate["visda-c"]["PU-Acc-per-class"]) == 12
        write_aggregate(root, aggregate)
        assert (root / "aggregate.json").is_file()
        assert (root / "results.csv").is_file()
        assert len((root / "visda_per_class.csv").read_text(encoding="utf-8").splitlines()) == 13


def main() -> int:
    check_config_and_transforms()
    check_loss()
    check_end_to_end()
    check_aggregate_outputs()
    print("Transformer full-dense tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
