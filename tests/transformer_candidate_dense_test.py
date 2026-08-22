#!/usr/bin/env python3
"""CPU-only protocol checks for Transformer candidate-dense SHOT-OTTA."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformer.candidate_dense.aggregate import (  # noqa: E402
    aggregate_matrix,
    write_aggregate,
)
from transformer.candidate_dense.config import (  # noqa: E402
    CANDIDATE_SCALAR_COUNT,
    CANDIDATE_TENSOR_COUNT,
    FORMAL_SEED,
    PROTOCOL_REVISION,
    TRANSFERS,
    _validate_frozen_fields,
    candidate_parameter_names,
    load_config,
    resolve_transfer_config,
    select_transfers,
)
from transformer.candidate_dense.matrix import validate_devices  # noqa: E402
from transformer.candidate_dense.model import (  # noqa: E402
    configure_candidate_dense_scope,
    frozen_named_state,
    hash_tensors,
)
import transformer.candidate_dense.runner as runner_module  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "transformer" / "candidate_dense" / "config.yaml"


class Attention(nn.Module):
    def __init__(self, *, valid: bool = True):
        super().__init__()
        qkv_out = 1152 if valid else 1
        self.qkv = nn.Linear(384, qkv_out, bias=True)
        self.proj = nn.Linear(384, 384, bias=True)


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(384, 1536, bias=True)
        self.fc2 = nn.Linear(1536, 384, bias=True)


class CandidateBlock(nn.Module):
    def __init__(self, *, valid: bool = True):
        super().__init__()
        self.attn = Attention(valid=valid)
        self.mlp = Mlp()
        self.norm = nn.LayerNorm(384)


class ScopeOnlyDeiT(nn.Module):
    def __init__(self, *, valid: bool = True):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 384))
        self.blocks = nn.ModuleList(
            [nn.Identity() for _ in range(9)]
            + [CandidateBlock(valid=valid) for _ in range(3)]
        )
        self.norm = nn.LayerNorm(384)
        self.head = nn.Linear(384, 31)
        self.register_buffer("position_ids", torch.arange(4))

    def get_classifier(self):
        return self.head


def check_config_and_identity() -> None:
    config = load_config(CONFIG_PATH)
    _validate_frozen_fields(config)
    assert config["formal_seed"] == FORMAL_SEED == 2026
    assert config["protocol_revision"] == PROTOCOL_REVISION
    assert config["adaptation"]["candidate_tensor_count"] == 12
    assert config["adaptation"]["candidate_scalar_count"] == 5_308_416
    assert len(TRANSFERS) == 7
    assert len(select_transfers("office31")) == 6
    assert select_transfers("visda-c") == (("visda-c", "train", "validation"),)
    validate_devices(["0", "1"])

    changed = copy.deepcopy(config)
    changed["formal_seed"] = 2027
    try:
        _validate_frozen_fields(changed)
    except ValueError as error:
        assert "2026" in str(error)
    else:
        raise AssertionError("A non-formal stream seed was accepted")

    changed = copy.deepcopy(config)
    changed["adaptation"]["candidate_blocks"] = [8, 9, 10]
    try:
        _validate_frozen_fields(changed)
    except ValueError as error:
        assert "adaptation" in str(error)
    else:
        raise AssertionError("An invalid candidate block range was accepted")

    for invalid in ([], ["cpu", "0"], ["0", "0"]):
        try:
            validate_devices(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid devices were accepted: {invalid}")


def _write_fake_resolver_assets(root: Path, *, corrupt_manifest: bool = False) -> dict:
    config = load_config(CONFIG_PATH)
    list_root = root / "lists"
    checkpoint_root = root / "checkpoints"
    target_list = list_root / "office31" / "dslr_list.txt"
    target_list.parent.mkdir(parents=True)
    target_list.write_text("/synthetic/image.jpg 0\n", encoding="utf-8")

    mapping = {f"class-{index}": index for index in range(31)}
    class_payload = {
        "dataset": "office31",
        "class_to_idx": mapping,
        "domains": {"dslr": {"path": str(target_list.resolve())}},
    }
    (target_list.parent / "class_to_idx.json").write_text(
        json.dumps(class_payload), encoding="utf-8"
    )

    checkpoint = checkpoint_root / "office31" / "amazon.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic-source-checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest_digest = "0" * 64 if corrupt_manifest else digest
    checkpoint.with_suffix(".manifest.json").write_text(
        json.dumps(
            {"checkpoint": {"path": str(checkpoint.resolve()), "sha256": manifest_digest}}
        ),
        encoding="utf-8",
    )
    config["data"]["list_root"] = str(list_root)
    config["model"]["checkpoint_root"] = str(checkpoint_root)
    return config


def check_resolver_hashes_and_scientific_identity() -> None:
    with tempfile.TemporaryDirectory(prefix="candidate_dense_resolver_") as value:
        root = Path(value)
        config = _write_fake_resolver_assets(root)
        first = resolve_transfer_config(
            config,
            project_root=PROJECT_ROOT,
            dataset="office31",
            source="amazon",
            target="dslr",
            device="cpu",
            output_dir=root / "first",
        )
        target_list = Path(first["target_list"])
        target_list.write_text("/synthetic/changed.jpg 0\n", encoding="utf-8")
        second = resolve_transfer_config(
            config,
            project_root=PROJECT_ROOT,
            dataset="office31",
            source="amazon",
            target="dslr",
            device="cpu",
            output_dir=root / "second",
        )
        assert first["target_list_sha256"] != second["target_list_sha256"]
        assert first["scientific_config_sha256"] != second["scientific_config_sha256"]

    with tempfile.TemporaryDirectory(prefix="candidate_dense_bad_hash_") as value:
        root = Path(value)
        config = _write_fake_resolver_assets(root, corrupt_manifest=True)
        try:
            resolve_transfer_config(
                config,
                project_root=PROJECT_ROOT,
                dataset="office31",
                source="amazon",
                target="dslr",
                device="cpu",
                output_dir=root / "run",
            )
        except ValueError as error:
            assert "SHA-256" in str(error)
        else:
            raise AssertionError("A checkpoint/manifest hash mismatch was accepted")


def check_exact_candidate_scope_and_adamw_freezing() -> None:
    model = ScopeOnlyDeiT()
    trainable, frozen, record = configure_candidate_dense_scope(model)
    assert tuple(name for name, _ in trainable) == candidate_parameter_names()
    assert len(trainable) == CANDIDATE_TENSOR_COUNT == 12
    assert record["trainable_scalars"] == CANDIDATE_SCALAR_COUNT == 5_308_416
    assert all(parameter.requires_grad for _, parameter in trainable)
    assert all(not parameter.requires_grad for _, parameter in frozen)
    assert all("bias" not in name for name, _ in trainable)
    assert all(not name.startswith("blocks.8.") for name, _ in trainable)

    candidate_names = {name for name, _ in trainable}
    frozen_before = hash_tensors(frozen_named_state(model, candidate_names))
    candidate_before = hash_tensors(trainable)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=1.0e-5,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.01,
    )
    for _, parameter in trainable:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert hash_tensors(trainable) != candidate_before
    assert hash_tensors(frozen_named_state(model, candidate_names)) == frozen_before

    bad_model = ScopeOnlyDeiT(valid=False)
    try:
        configure_candidate_dense_scope(bad_model)
    except ValueError as error:
        assert "shape" in str(error)
    else:
        raise AssertionError("An invalid candidate tensor shape was accepted")


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


class TinyCandidateModel(nn.Module):
    forward_calls = 0

    def __init__(self):
        super().__init__()
        self.candidate = nn.Linear(4, 3)
        self.head = nn.Linear(3, 2)
        self.register_buffer("read_only_buffer", torch.tensor([1.0]))
        with torch.no_grad():
            self.candidate.weight.copy_(
                torch.tensor(
                    [
                        [0.4, -0.1, 0.2, 0.3],
                        [-0.2, 0.5, 0.1, -0.4],
                        [0.3, 0.2, -0.5, 0.1],
                    ]
                )
            )
            self.head.weight.copy_(
                torch.tensor([[0.7, -0.4, 0.2], [-0.3, 0.6, -0.5]])
            )

    def get_classifier(self):
        return self.head

    def forward(self, images):
        type(self).forward_calls += 1
        return self.head(torch.tanh(self.candidate(images)))


def _tiny_model_loader(config, device):
    del config
    model = TinyCandidateModel().to(device)
    model.requires_grad_(False)
    model.candidate.requires_grad_(True)
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
    return model, {"path": "synthetic", "sha256": "0" * 64}, trainable, frozen, {
        "candidate_blocks": [9, 10, 11],
        "candidate_parameter_suffixes": ["synthetic"],
        "trainable_parameter_names": [name for name, _ in trainable],
        "trainable_tensor_count": len(trainable),
        "trainable_scalars": sum(parameter.numel() for _, parameter in trainable),
        "frozen_parameter_names": [name for name, _ in frozen],
        "frozen_parameter_scalars": sum(parameter.numel() for _, parameter in frozen),
        "total_parameter_scalars": sum(parameter.numel() for parameter in model.parameters()),
    }


def _tiny_loaders(config):
    del config
    dataset = TinyDataset()
    online = DataLoader(
        dataset,
        batch_size=2,
        sampler=OrderedSampler([2, 0, 1]),
        drop_last=False,
    )
    final = DataLoader(
        dataset,
        batch_size=2,
        sampler=OrderedSampler([0, 1, 2]),
        drop_last=False,
    )
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


def check_end_to_end_pu_fo() -> None:
    TinyCandidateModel.forward_calls = 0
    config = {
        "output_dir": "",
        "runtime": {
            "device": "cpu",
            "deterministic": True,
            "amp": False,
            "pin_memory": False,
        },
        "formal_seed": 2026,
        "protocol_revision": PROTOCOL_REVISION,
        "experiment_key": "synthetic-candidate-dense",
        "scientific_config_sha256": "1" * 64,
        "dataset": "office31",
        "source": "amazon",
        "target": "dslr",
        "transfer": "amazon->dslr",
        "num_classes": 2,
        "class_names": ["zero", "one"],
        "batch_size": 2,
        "fo_batch_size": 2,
        "adaptation": {
            "update_scope": "synthetic",
            "model_mode": "eval",
            "steps_per_online_batch": 1,
        },
        "optimization": {
            "optimizer": "adamw",
            "lr": 1.0e-3,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "weight_decay": 0.01,
        },
        "loss": {
            "components": ["ent", "div", "pseudo"],
            "cls_par": 0.3,
            "ent_par": 1.0,
            "threshold": 0.0,
        },
        "preprocessing": {"interpolation": "bilinear"},
    }
    original_builder = runner_module.build_target_loaders
    runner_module.build_target_loaders = _tiny_loaders
    try:
        with tempfile.TemporaryDirectory(prefix="candidate_dense_runner_") as value:
            config["output_dir"] = str(Path(value) / "run")
            summary = runner_module.run_transfer(
                config,
                PROJECT_ROOT,
                show_progress=False,
                model_loader=_tiny_model_loader,
            )
            assert summary["status"] == "completed"
            assert summary["variant"] == "candidate_dense"
            assert summary["adaptation_steps"] == 2
            assert summary["backward_calls"] == 2
            assert summary["candidate_state_changed"] is True
            assert summary["frozen_state_unchanged"] is True
            assert summary["frozen_head_unchanged"] is True
            assert summary["PU-sample-count"] == 3
            assert summary["FO-sample-count"] == 3
            assert summary["pu_is_post_update_same_batch"] is True
            assert summary["fo_is_independent_full_target_pass"] is True
            assert TinyCandidateModel.forward_calls == 6
            rows = [
                json.loads(line)
                for line in (Path(config["output_dir"]) / "metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            online_rows = [row for row in rows if row["event"] == "online_batch"]
            assert [row["batch_size"] for row in online_rows] == [2, 1]
            assert all(row["adaptation_steps"] == 1 for row in online_rows)
            assert all(row["pu_is_post_update_same_batch"] for row in online_rows)
    finally:
        runner_module.build_target_loaders = original_builder


def check_aggregate_outputs() -> None:
    with tempfile.TemporaryDirectory(prefix="candidate_dense_aggregate_") as value:
        root = Path(value)
        for index, (dataset, source, target) in enumerate(TRANSFERS):
            output = root / "results" / dataset / f"{source}-{target}"
            output.mkdir(parents=True)
            names = [f"class-{item}" for item in range(12)]
            summary = {
                "status": "completed",
                "variant": "candidate_dense",
                "dataset": dataset,
                "source": source,
                "target": target,
                "transfer": f"{source}->{target}",
                "formal_seed": 2026,
                "primary_metric": (
                    "sample_overall_accuracy"
                    if dataset == "office31"
                    else "fixed_12_class_macro_accuracy"
                ),
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
                        "PU-Acc-per-class-by-name": {
                            name: float(i) for i, name in enumerate(names)
                        },
                        "FO-Acc-per-class-by-name": {
                            name: float(i + 1) for i, name in enumerate(names)
                        },
                    }
                )
            (output / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
        aggregate = aggregate_matrix(root)
        assert aggregate["office31"]["PU-Acc"] == 2.5
        assert aggregate["office31"]["FO-Acc"] == 3.5
        assert len(aggregate["visda-c"]["PU-Acc-per-class"]) == 12
        write_aggregate(root, aggregate)
        assert (root / "aggregate.json").is_file()
        assert (root / "results.csv").is_file()
        with open(root / "visda_per_class.csv", newline="", encoding="utf-8") as file_obj:
            class_rows = list(csv.DictReader(file_obj))
        assert len(class_rows) == 12
        assert set(class_rows[0]) == {"class", "PU-Acc", "FO-Acc"}


def main() -> int:
    check_config_and_identity()
    check_resolver_hashes_and_scientific_identity()
    check_exact_candidate_scope_and_adamw_freezing()
    check_end_to_end_pu_fo()
    check_aggregate_outputs()
    print("Transformer candidate-dense tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

