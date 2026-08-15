#!/usr/bin/env python3
"""CPU tests for the DeiT source-training contract; no network is used."""

import copy
import os
import os.path as osp
import sys
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from source_training.deit_config import (  # noqa: E402
    apply_source_overrides,
    load_source_config,
    resolve_source_config,
)
from source_training.deit_data import stratified_fixed_split  # noqa: E402
from source_training.deit_model import (  # noqa: E402
    SOURCE_CHECKPOINT_KIND,
    SOURCE_CHECKPOINT_SCHEMA_VERSION,
    build_deit_source_model,
    load_deit_source_checkpoint,
    sha256_file,
)
from source_training.deit_trainer import (  # noqa: E402
    WarmupCosineSchedule,
    _grad_scaler,
    build_adamw,
    evaluate,
    train_one_epoch,
)
from tools.build_image_lists import (  # noqa: E402
    VISDA_CLASS_NAMES,
    build_lists,
    canonical_class_order,
)


OFFICE_CONFIG = osp.join(
    PROJECT_ROOT, "configs", "source_deit_office31.yaml"
)
VISDA_CONFIG = osp.join(
    PROJECT_ROOT, "configs", "source_deit_visda.yaml"
)


class TinyDeiT(nn.Module):
    def __init__(self, num_classes, input_dim=4, **kwargs):
        super().__init__()
        del kwargs
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 384))
        self.pos_embed = nn.Parameter(torch.zeros(1, 2, 384))
        self.backbone = nn.Linear(input_dim, 384)
        self.norm = nn.LayerNorm(384)
        self.head = nn.Linear(384, num_classes)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(384, num_classes)

    def get_classifier(self):
        return self.head

    def forward(self, inputs):
        return self.head(self.norm(self.backbone(inputs)))


def tiny_factory(model_name, **kwargs):
    assert model_name == "deit_small_patch16_224.fb_in1k"
    return TinyDeiT(**kwargs)


def _check_configs():
    office = load_source_config(OFFICE_CONFIG)
    for domain in ("amazon", "dslr", "webcam"):
        overridden = apply_source_overrides(office, source_domain=domain)
        effective = resolve_source_config(overridden, PROJECT_ROOT)
        assert effective["data"]["source_domain"] == domain
        assert effective["data"]["num_classes"] == 31
        assert effective["model"]["head"] == "linear"
        assert effective["training"]["finetune_scope"] == "full_model"
        assert effective["checkpoint"]["output_path"].endswith(
            f"office31{os.sep}{domain}.pth"
        )
    visda = resolve_source_config(
        load_source_config(VISDA_CONFIG), PROJECT_ROOT
    )
    assert visda["data"]["source_domain"] == "train"
    assert visda["data"]["num_classes"] == 12
    assert visda["checkpoint"]["selection_metric"] == "macro_class_accuracy"
    assert "target" not in visda["data"]
    assert "target_list" not in visda["data"]

    invalid = copy.deepcopy(office)
    invalid["training"]["finetune_scope"] = "head_only"
    try:
        resolve_source_config(invalid, PROJECT_ROOT)
    except ValueError as error:
        assert "linear-probe" in str(error)
    else:
        raise AssertionError("head-only source training was accepted")


def _check_split():
    records = []
    for label in range(3):
        records.extend(
            (f"/fake/class-{label}/image-{index}.jpg", label)
            for index in range(10)
        )
    train, validation, manifest = stratified_fixed_split(records, 0.1, 2020)
    train_again, validation_again, manifest_again = stratified_fixed_split(
        records, 0.1, 2020
    )
    assert train == train_again
    assert validation == validation_again
    assert manifest == manifest_again
    assert len(train) == 27
    assert len(validation) == 3
    assert not (set(train) & set(validation))
    assert set(train) | set(validation) == set(range(30))
    assert all(
        counts["validation"] == 1
        for counts in manifest["per_class"].values()
    )


def _check_local_pretrained_and_checkpoint_round_trip():
    with tempfile.TemporaryDirectory(prefix="deit_source_model_") as temp_dir:
        placeholder = osp.join(temp_dir, "model.safetensors")
        with open(placeholder, "wb") as file_obj:
            file_obj.write(b"local-only-test-weight-placeholder")
        pretrained = TinyDeiT(num_classes=1000).state_dict()
        pretrained_backbone = pretrained["backbone.weight"].clone()
        model, head = build_deit_source_model(
            model_name="deit_small_patch16_224.fb_in1k",
            num_classes=31,
            pretrained_path=placeholder,
            model_factory=tiny_factory,
            state_loader=lambda path: pretrained,
        )
        assert head.in_features == 384
        assert head.out_features == 31
        assert torch.equal(model.backbone.weight, pretrained_backbone)
        assert model.head.weight.shape == (31, 384)

        checkpoint_path = osp.join(temp_dir, "amazon.pth")
        payload = {
            "schema_version": SOURCE_CHECKPOINT_SCHEMA_VERSION,
            "kind": SOURCE_CHECKPOINT_KIND,
            "state_dict": model.state_dict(),
            "metadata": {
                "model": {
                    "name": "deit_small_patch16_224.fb_in1k",
                    "num_classes": 31,
                    "drop_rate": 0.0,
                    "drop_path_rate": 0.0,
                }
            },
        }
        torch.save(payload, checkpoint_path)
        checkpoint_hash = sha256_file(checkpoint_path)
        loaded, metadata = load_deit_source_checkpoint(
            checkpoint_path,
            expected_sha256=checkpoint_hash,
            model_factory=tiny_factory,
        )
        assert metadata["model"]["num_classes"] == 31
        assert loaded.head.weight.shape == (31, 384)
        for name, expected in model.state_dict().items():
            assert torch.equal(loaded.state_dict()[name], expected)


def _check_optimizer_and_tiny_training():
    torch.manual_seed(7)
    model = TinyDeiT(num_classes=2)
    config = {
        "backbone_lr": 6.25e-5,
        "head_lr": 6.25e-4,
        "weight_decay": 0.05,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
    }
    optimizer, manifest = build_adamw(model, config)
    assert {entry["name"] for entry in manifest} == {
        "backbone_decay",
        "backbone_no_decay",
        "head_decay",
        "head_no_decay",
    }
    by_name = {entry["name"]: entry for entry in manifest}
    assert by_name["head_decay"]["learning_rate"] == 6.25e-4
    assert by_name["backbone_decay"]["learning_rate"] == 6.25e-5
    assert by_name["head_no_decay"]["weight_decay"] == 0.0
    assert by_name["backbone_no_decay"]["weight_decay"] == 0.0
    assert "cls_token" in by_name["backbone_no_decay"]["parameter_names"]
    assert "pos_embed" in by_name["backbone_no_decay"]["parameter_names"]

    inputs = torch.randn(8, 4)
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=4, shuffle=False)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    schedule = WarmupCosineSchedule(
        optimizer, total_steps=2, warmup_steps=0, min_lr=1e-6
    )
    head_before = model.head.weight.detach().clone()
    train_metrics = train_one_epoch(
        model,
        loader,
        optimizer,
        criterion,
        schedule,
        device=torch.device("cpu"),
        amp=False,
        scaler=_grad_scaler(False),
        global_step_start=0,
    )
    validation_metrics = evaluate(
        model,
        loader,
        criterion,
        device=torch.device("cpu"),
        amp=False,
        num_classes=2,
    )
    assert train_metrics["example_count"] == 8
    assert validation_metrics["example_count"] == 8
    assert len(validation_metrics["per_class_accuracy"]) == 2
    assert not torch.equal(model.head.weight, head_before)


def _check_image_list_builder():
    assert canonical_class_order(
        "visda-c", reversed(VISDA_CLASS_NAMES)
    ) == list(VISDA_CLASS_NAMES)
    assert canonical_class_order(
        "visda-c", [str(index) for index in reversed(range(12))]
    ) == [str(index) for index in range(12)]
    with tempfile.TemporaryDirectory(prefix="deit_source_lists_") as temp_dir:
        dataset_root = osp.join(temp_dir, "office31")
        for domain in ("amazon", "dslr", "webcam"):
            for class_index in range(31):
                class_dir = osp.join(
                    dataset_root, domain, "images", f"class_{class_index:02d}"
                )
                os.makedirs(class_dir)
                with open(
                    osp.join(class_dir, f"{domain}_{class_index}.jpg"), "wb"
                ) as file_obj:
                    file_obj.write(b"test")
        output_root = osp.join(temp_dir, "image_lists", "office31")
        result = build_lists("office31", dataset_root, output_root)
        assert osp.isfile(result["mapping"])
        for domain in ("amazon", "dslr", "webcam"):
            list_path = result["domains"][domain]["path"]
            assert osp.isfile(list_path)
            with open(list_path, "r", encoding="utf-8") as file_obj:
                lines = file_obj.readlines()
            assert len(lines) == 31
            assert all(osp.isabs(line.rsplit(maxsplit=1)[0]) for line in lines)


def main():
    _check_configs()
    _check_split()
    _check_local_pretrained_and_checkpoint_round_trip()
    _check_optimizer_and_tiny_training()
    _check_image_list_builder()
    print("DeiT source-training tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
