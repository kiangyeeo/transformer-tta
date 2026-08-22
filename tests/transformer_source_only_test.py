#!/usr/bin/env python3
"""CPU-only checks for the Transformer source-only protocol helpers."""

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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformer.source_only.config import (  # noqa: E402
    FORMAL_SEED,
    TRANSFERS,
    _validate_frozen_fields,
    load_config,
    select_transfers,
)
from transformer.source_only.aggregate import aggregate_matrix  # noqa: E402
from transformer.source_only.data import fixed_random_order  # noqa: E402
from transformer.source_only.data import build_transforms  # noqa: E402
from transformer.source_only.metrics import FixedClassMeter  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "transformer" / "source_only" / "config.yaml"


def check_config_and_matrix() -> None:
    config = load_config(CONFIG_PATH)
    _validate_frozen_fields(config)
    assert config["formal_seed"] == FORMAL_SEED == 2026
    assert config["data"]["office31"]["batch_size"] == 64
    assert config["data"]["office31"]["fo_batch_size"] == 64
    assert config["data"]["visda-c"]["batch_size"] == 256
    assert config["data"]["visda-c"]["fo_batch_size"] == 256
    assert config["data"]["stream"]["drop_last"] is False
    assert len(TRANSFERS) == 7
    assert len(select_transfers("office31")) == 6
    assert select_transfers("visda-c") == (
        ("visda-c", "train", "validation"),
    )

    changed = copy.deepcopy(config)
    changed["formal_seed"] = 2020
    try:
        _validate_frozen_fields(changed)
    except ValueError as error:
        assert "2026" in str(error)
    else:
        raise AssertionError("A non-formal stream seed was accepted")

    changed = copy.deepcopy(config)
    changed["data"]["visda-c"]["fo_batch_size"] = 768
    try:
        _validate_frozen_fields(changed)
    except ValueError as error:
        assert "visda-c" in str(error)
    else:
        raise AssertionError("The FC runner's 3x FO batch size was accepted")


def check_stream_order() -> None:
    first = fixed_random_order(37, 2026)
    second = fixed_random_order(37, 2026)
    different = fixed_random_order(37, 2027)
    assert first == second
    assert first != different
    assert sorted(first) == list(range(37))


def check_fc_aligned_transforms() -> None:
    config = load_config(CONFIG_PATH)["data"]["preprocessing"]
    assert config["interpolation"] == "bilinear"
    online, final = build_transforms(config)
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
        resize = pipeline.transforms[0]
        normalize = pipeline.transforms[-1]
        assert resize.size == (256, 256)
        assert resize.interpolation == InterpolationMode.BILINEAR
        assert list(normalize.mean) == [0.485, 0.456, 0.406]
        assert list(normalize.std) == [0.229, 0.224, 0.225]


def check_office_sample_aggregation() -> None:
    meter = FixedClassMeter(2, ["many", "few"])
    labels = torch.tensor([0] * 10 + [1])
    predictions = torch.tensor([0] * 9 + [1, 0])
    meter.update(labels[:4], predictions[:4])
    meter.update(labels[4:], predictions[4:])
    metrics = meter.compute("office31")
    assert metrics["correct"] == 9
    assert metrics["sample-count"] == 11
    assert abs(metrics["Acc"] - 100.0 * 9 / 11) < 1e-12
    assert metrics["Acc"] == metrics["overall-Acc"]
    assert metrics["Acc"] != metrics["mean-class-Acc"]


def check_visda_fixed_12_class_macro() -> None:
    names = [f"class-{index}" for index in range(12)]
    meter = FixedClassMeter(12, names)
    labels = torch.tensor([0] * 10 + list(range(1, 12)))
    predictions = labels.clone()
    predictions[-1] = 0
    meter.update(labels, predictions)
    metrics = meter.compute("visda-c")
    assert len(metrics["Acc-per-class"]) == 12
    assert metrics["class-count"] == [10] + [1] * 11
    assert abs(metrics["Acc"] - (11.0 / 12.0 * 100.0)) < 1e-12
    assert metrics["Acc"] == metrics["mean-class-Acc"]
    assert metrics["Acc"] != metrics["overall-Acc"]
    assert metrics["worst-class-name"] == "class-11"


def check_office_equal_transfer_aggregation() -> None:
    transfers = select_transfers("office31")
    with tempfile.TemporaryDirectory(prefix="transformer_source_only_") as value:
        root = Path(value)
        for index, (dataset, source, target) in enumerate(transfers):
            output_dir = root / "results" / dataset / f"{source}-{target}"
            output_dir.mkdir(parents=True)
            summary = {
                "status": "completed",
                "dataset": dataset,
                "transfer": f"{source}->{target}",
                "PU-Acc": float(index * 10),
                "FO-Acc": float(index * 10 + 1),
            }
            (output_dir / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
        aggregate = aggregate_matrix(root, transfers=transfers)
        assert aggregate["office31"]["PU-Acc"] == 25.0
        assert aggregate["office31"]["FO-Acc"] == 26.0
        assert aggregate["office31"]["aggregation"] == (
            "equal-weight mean of transfer accuracies"
        )


def main() -> int:
    check_config_and_matrix()
    check_stream_order()
    check_fc_aligned_transforms()
    check_office_sample_aggregation()
    check_visda_fixed_12_class_macro()
    check_office_equal_transfer_aggregation()
    print("Transformer source-only tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
