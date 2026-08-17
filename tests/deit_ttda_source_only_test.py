#!/usr/bin/env python3
"""CPU contracts for DeiT TTDA source-only; no network or real W0 is used."""

import copy
import json
import math
import os.path as osp
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shot_otta.backbones.deit import load_frozen_deit_source  # noqa: E402
from shot_otta.ttda.config import (  # noqa: E402
    DATASET_SPECS,
    VISDA_CLASS_NAMES,
    load_source_manifest,
    load_yaml,
    resolve_config,
    sha256_file,
)
from shot_otta.ttda.source_only import (  # noqa: E402
    compute_fixed_class_metrics,
    run_source_only_experiment,
)
from source_training.deit_model import (  # noqa: E402
    SOURCE_CHECKPOINT_KIND,
    SOURCE_CHECKPOINT_SCHEMA_VERSION,
)
from tools.plan_deit_ttda_source_only import build_plan  # noqa: E402
from tools.summarize_deit_ttda_source_only import (  # noqa: E402
    build_report,
    write_report,
)


CONFIG_PATH = osp.join(PROJECT_ROOT, "configs", "deit_ttda_source_only.yaml")
MATRIX_PATH = osp.join(
    PROJECT_ROOT, "experiments", "deit_ttda_source_only.yaml"
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


def _write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


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
        domains[domain] = {"path": str(list_path.resolve()), "image_count": len(lines)}
    _write_json(
        dataset_root / "class_to_idx.json",
        {"dataset": dataset, "class_to_idx": mapping, "domains": domains},
    )


def _checkpoint_metadata(dataset, source_name, num_classes):
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


def _write_checkpoint_assets(root):
    checkpoint_root = Path(root) / "checkpoints"
    for dataset, spec in DATASET_SPECS.items():
        source_domains = (
            spec["domains"][:-1]
            if dataset == "visda-c"
            else spec["domains"]
        )
        for source_name in source_domains:
            path = checkpoint_root / dataset / f"{source_name}.pth"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"placeholder-{dataset}-{source_name}".encode())
            metadata = _checkpoint_metadata(
                dataset, source_name, spec["num_classes"]
            )
            _write_json(
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


def _base_config(root):
    config = load_yaml(CONFIG_PATH)
    config["data"]["list_root"] = str((Path(root) / "lists").resolve())
    config["model"]["source_checkpoint_root"] = str(
        (Path(root) / "checkpoints").resolve()
    )
    config["device"] = {"type": "cpu", "gpu_id": "0"}
    config["evaluation"].update(
        {"batch_size": 30, "workers": 0, "amp": False, "pin_memory": False}
    )
    config["output"]["root"] = str((Path(root) / "runs").resolve())
    return config


def _prepare_assets(root):
    for dataset in DATASET_SPECS:
        _write_list_assets(root, dataset)
    _write_checkpoint_assets(root)
    return _base_config(root)


def _install_real_fake_checkpoint(root):
    path = Path(root) / "checkpoints" / "office31" / "amazon.pth"
    model = TinyDeiT(num_classes=31)
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
        model.head.bias[0] = 1.0
    metadata = _checkpoint_metadata("office31", "amazon", 31)
    torch.save(
        {
            "schema_version": SOURCE_CHECKPOINT_SCHEMA_VERSION,
            "kind": SOURCE_CHECKPOINT_KIND,
            "state_dict": model.state_dict(),
            "metadata": metadata,
        },
        path,
    )
    _write_json(
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


def _expect_error(callable_obj, text):
    try:
        callable_obj()
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as error:
        assert text.lower() in str(error).lower(), str(error)
    else:
        raise AssertionError(f"Expected an error containing {text!r}")


def _check_metrics():
    labels = np.asarray([0, 0, 1, 2])
    predictions = np.asarray([0, 1, 1, 0])
    metrics = compute_fixed_class_metrics(
        labels,
        predictions,
        num_classes=3,
        class_names=["zero", "one", "two"],
        prefix="PU",
    )
    assert metrics["PU-overall-Acc"] == 50.0
    assert math.isclose(metrics["PU-Acc"], 50.0)
    assert metrics["PU-Acc-per-class"] == [50.0, 100.0, 0.0]
    _expect_error(
        lambda: compute_fixed_class_metrics(
            [0, 1],
            [0, 1],
            num_classes=3,
            class_names=["zero", "one", "two"],
            prefix="FO",
        ),
        "missing classes",
    )


def _check_manifest_failures(root, base):
    valid = resolve_config(base, PROJECT_ROOT)
    bad_hash = copy.deepcopy(valid)
    bad_hash["source_checkpoint"]["sha256"] = "0" * 64
    _expect_error(
        lambda: load_frozen_deit_source(
            bad_hash, torch.device("cpu"), model_factory=tiny_factory
        ),
        "sha-256 mismatch",
    )
    checkpoint = Path(root) / "checkpoints" / "office31" / "amazon.pth"
    manifest = checkpoint.with_suffix(".manifest.json")
    last = checkpoint.with_name("amazon.last.pth")
    last.write_bytes(b"resume")
    _expect_error(
        lambda: load_source_manifest(
            str(last),
            str(last),
            dataset="office31",
            source_name="amazon",
            num_classes=31,
            model_name="deit_small_patch16_224.fb_in1k",
        ),
        "last.pth",
    )
    _expect_error(
        lambda: load_source_manifest(
            str(manifest.with_name("missing.manifest.json")),
            str(checkpoint),
            dataset="office31",
            source_name="amazon",
            num_classes=31,
            model_name="deit_small_patch16_224.fb_in1k",
        ),
        "manifest not found",
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["checkpoint"]["path"] = str(checkpoint.resolve())
    payload["dataset"]["source_domain"] = "dslr"
    wrong_domain = manifest.with_name("wrong-domain.manifest.json")
    _write_json(wrong_domain, payload)
    _expect_error(
        lambda: load_source_manifest(
            str(wrong_domain),
            str(checkpoint),
            dataset="office31",
            source_name="amazon",
            num_classes=31,
            model_name="deit_small_patch16_224.fb_in1k",
        ),
        "domain",
    )
    payload["dataset"]["source_domain"] = "amazon"
    payload["model"]["head_schema"] = "Linear(384,12)"
    wrong_head = manifest.with_name("wrong-head.manifest.json")
    _write_json(wrong_head, payload)
    _expect_error(
        lambda: load_source_manifest(
            str(wrong_head),
            str(checkpoint),
            dataset="office31",
            source_name="amazon",
            num_classes=31,
            model_name="deit_small_patch16_224.fb_in1k",
        ),
        "head",
    )


def _check_end_to_end(root, base):
    _install_real_fake_checkpoint(root)
    effective = resolve_config(base, PROJECT_ROOT)
    summary = run_source_only_experiment(
        effective, PROJECT_ROOT, model_factory=tiny_factory
    )
    expected = 100.0 / 31.0
    assert math.isclose(summary["PU-Acc"], expected)
    assert summary["PU-Acc"] == summary["FO-Acc"]
    assert summary["PU-overall-Acc"] == summary["FO-overall-Acc"]
    assert summary["processed_sample_count"] == 31
    assert summary["target_batch_count"] == 2
    assert summary["adaptation_steps"] == 0
    assert summary["optimizer_created"] is False
    assert summary["backward_calls"] == 0
    assert summary["pu_fo_predictions_equal"] is True
    assert summary["model_state_unchanged"] is True
    output = Path(summary["output_dir"])
    for name in ("config.yaml", "manifest.json", "metrics.jsonl", "summary.json"):
        assert (output / name).is_file()
    return effective


def _check_plan_and_summary(root, base):
    plan = build_plan(load_yaml(MATRIX_PATH), base, CONFIG_PATH, MATRIX_PATH)
    assert plan["experiment_count"] == 7
    assert [
        (entry["dataset"], entry["source"], entry["target"])
        for entry in plan["experiments"]
    ] == [
        ("office31", 0, 1),
        ("office31", 0, 2),
        ("office31", 1, 0),
        ("office31", 1, 2),
        ("office31", 2, 0),
        ("office31", 2, 1),
        ("visda-c", 0, 1),
    ]
    assert len({entry["experiment_key"] for entry in plan["experiments"]}) == 7
    assert len(
        {entry["source_checkpoint_sha256"] for entry in plan["experiments"]}
    ) == 4
    fake_runs = Path(root) / "fake_runs"
    for index, experiment in enumerate(plan["experiments"]):
        directory = fake_runs / f"run-{index}"
        directory.mkdir(parents=True)
        macro = float(index + 10)
        overall = float(index + 20)
        _write_json(
            directory / "summary.json",
            {
                "status": "completed",
                "experiment_key": experiment["experiment_key"],
                "experiment_config_sha256": experiment[
                    "experiment_config_sha256"
                ],
                "method": "no_tta",
                "variant": "source_only",
                "task": "ttda",
                "dataset": experiment["dataset"],
                "source": experiment["source"],
                "target": experiment["target"],
                "source_name": experiment["source_name"],
                "target_name": experiment["target_name"],
                "seed": experiment["seed"],
                "PU-Acc": macro,
                "FO-Acc": macro,
                "PU-overall-Acc": overall,
                "FO-overall-Acc": overall,
                "adaptation_steps": 0,
                "optimizer_created": False,
                "loss_computed": False,
                "backward_calls": 0,
                "target_labels_usage": "evaluation_only",
                "pu_fo_predictions_equal": True,
                "model_state_unchanged": True,
                "model_state_sha256_before": "a" * 64,
                "model_state_sha256_after": "a" * 64,
                "processed_sample_count": experiment["scientific_config"][
                    "target_data"
                ]["sample_count"],
                "runtime": 1.0 + index,
                "peak_gpu_memory_bytes": index,
                "source_checkpoint_sha256": experiment[
                    "source_checkpoint_sha256"
                ],
            },
        )
    report = build_report(plan, fake_runs)
    assert report["experiment_count"] == 7
    assert len(report["report_rows"]) == 8
    assert report["office31_six_direction_average"]["PU-Acc"] == 12.5
    report_output = Path(root) / "report"
    write_report(report, report_output)
    for name in ("report.json", "per_transfer.csv", "report.csv", "report.md"):
        assert (report_output / name).is_file()
    duplicate = fake_runs / "duplicate"
    duplicate.mkdir()
    first = fake_runs / "run-0" / "summary.json"
    (duplicate / "summary.json").write_text(
        first.read_text(encoding="utf-8"), encoding="utf-8"
    )
    _expect_error(lambda: build_report(plan, fake_runs), "not complete")


def main():
    _check_metrics()
    with tempfile.TemporaryDirectory(prefix="deit_ttda_source_only_") as root:
        base = _prepare_assets(root)
        _install_real_fake_checkpoint(root)
        _check_manifest_failures(root, base)
        _check_end_to_end(root, base)
        _check_plan_and_summary(root, base)
    print("DeiT TTDA source-only tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
