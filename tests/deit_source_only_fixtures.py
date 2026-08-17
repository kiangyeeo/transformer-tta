"""Shared synthetic assets for DeiT OTTA/TTDA source-only tests."""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from shot_otta.deit_source_only.config import (
    DATASET_SPECS,
    VISDA_CLASS_NAMES,
    load_yaml,
    sha256_file,
)
from source_training.deit_model import (
    SOURCE_CHECKPOINT_KIND,
    SOURCE_CHECKPOINT_SCHEMA_VERSION,
)


class TinyDeiT(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__()
        del kwargs
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 384))
        self.pos_embed = nn.Parameter(torch.zeros(1, 2, 384))
        self.head = nn.Linear(384, num_classes)

    def get_classifier(self):
        return self.head

    def forward(self, images):
        features = torch.zeros(
            images.size(0), 384, device=images.device, dtype=images.dtype
        )
        features[:, 0] = images.mean(dim=(1, 2, 3))
        return self.head(features)


def tiny_factory(model_name, **kwargs):
    assert model_name == "deit_small_patch16_224.fb_in1k"
    return TinyDeiT(**kwargs)


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def checkpoint_metadata(dataset, source_name, num_classes):
    return {
        "dataset": {
            "name": dataset,
            "source_domain": source_name,
            "num_classes": num_classes,
        },
        "model": {
            "name": "deit_small_patch16_224.fb_in1k",
            "num_classes": num_classes,
            "head_schema": f"Linear(384,{num_classes})",
            "drop_rate": 0.0,
            "drop_path_rate": 0.0,
        },
        "training": {"seed": 2020},
        "best": {
            "epoch": 0,
            "epoch_one_based": 1,
            "selection_metric": "overall_accuracy",
            "selection_value": 1.0,
            "source_validation": {},
        },
        "git": {"commit": "test", "dirty": False},
        "environment": {
            "python": sys.version.split()[0],
            "cuda": None,
            "torch": torch.__version__,
            "torchvision": "test",
            "timm": "test",
            "safetensors": "test",
        },
        "effective_config": {"test": True},
        "scientific_config_sha256": "1" * 64,
    }


def _write_list_assets(root, dataset):
    spec = DATASET_SPECS[dataset]
    dataset_root = Path(root) / "lists" / dataset
    dataset_root.mkdir(parents=True, exist_ok=True)
    names = (
        list(VISDA_CLASS_NAMES)
        if dataset == "visda-c"
        else [f"class_{index:02d}" for index in range(spec["num_classes"])]
    )
    mapping = {name: index for index, name in enumerate(names)}
    domains = {}
    for domain in spec["domains"]:
        list_path = dataset_root / f"{domain}_list.txt"
        lines = []
        for label in range(spec["num_classes"]):
            image_path = (
                Path(root) / "images" / dataset / domain / f"{label}.png"
            ).resolve()
            lines.append(f"{image_path} {label}\n")
        list_path.write_text("".join(lines), encoding="utf-8")
        domains[domain] = {
            "path": str(list_path.resolve()),
            "image_count": len(lines),
        }
    write_json(
        dataset_root / "class_to_idx.json",
        {"dataset": dataset, "class_to_idx": mapping, "domains": domains},
    )


def _write_checkpoint_assets(root):
    checkpoint_root = Path(root) / "checkpoints"
    for dataset, spec in DATASET_SPECS.items():
        source_domains = (
            spec["domains"][:-1] if dataset == "visda-c" else spec["domains"]
        )
        for source_name in source_domains:
            path = checkpoint_root / dataset / f"{source_name}.pth"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"placeholder-{dataset}-{source_name}".encode())
            metadata = checkpoint_metadata(
                dataset, source_name, spec["num_classes"]
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
    return checkpoint_root


def prepare_assets(root, config_path, *, runs_directory="runs"):
    for dataset in DATASET_SPECS:
        _write_list_assets(root, dataset)
    _write_checkpoint_assets(root)
    config = load_yaml(config_path)
    config["data"]["list_root"] = str((Path(root) / "lists").resolve())
    config["model"]["source_checkpoint_root"] = str(
        (Path(root) / "checkpoints").resolve()
    )
    config["device"] = {"type": "cpu", "gpu_id": "0"}
    config["evaluation"].update(
        {"batch_size": 30, "workers": 0, "amp": False, "pin_memory": False}
    )
    config["output"]["root"] = str(
        (Path(root) / runs_directory).resolve()
    )
    return config


def install_real_fake_checkpoint(root):
    path = Path(root) / "checkpoints" / "office31" / "amazon.pth"
    model = TinyDeiT(num_classes=31)
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
        model.head.bias[0] = 1.0
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
    image_root = Path(root) / "images" / "office31" / "dslr"
    image_root.mkdir(parents=True, exist_ok=True)
    for label in range(31):
        Image.new("RGB", (8, 8), color=(label, label, label)).save(
            image_root / f"{label}.png"
        )
