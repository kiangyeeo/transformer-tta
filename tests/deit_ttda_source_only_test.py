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


PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shot_otta.backbones.deit import load_frozen_deit_source  # noqa: E402
from shot_otta.deit_source_only.config import (  # noqa: E402
    VISDA_CLASS_NAMES,
    load_source_manifest,
    load_yaml,
    resolve_config,
)
from shot_otta.deit_source_only.runtime import (  # noqa: E402
    compute_fixed_class_metrics,
)
from shot_otta.ttda.source_only import run_source_only_experiment  # noqa: E402
from deit_source_only_fixtures import (  # noqa: E402
    install_real_fake_checkpoint,
    prepare_assets,
    tiny_factory,
    write_json,
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
    )
    assert metrics["overall-Acc"] == 50.0
    assert math.isclose(metrics["Acc"], 50.0)
    assert metrics["Acc-per-class"] == [50.0, 100.0, 0.0]
    _expect_error(
        lambda: compute_fixed_class_metrics(
            [0, 1],
            [0, 1],
            num_classes=3,
            class_names=["zero", "one", "two"],
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
    write_json(wrong_domain, payload)
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
    write_json(wrong_head, payload)
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
    corrupt_checkpoint = checkpoint.with_name("corrupt.pth")
    corrupt_checkpoint.write_bytes(b"corrupt-checkpoint")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["checkpoint"]["path"] = str(corrupt_checkpoint.resolve())
    payload["checkpoint"]["sha256"] = "0" * 64
    corrupt_manifest = corrupt_checkpoint.with_suffix(".manifest.json")
    write_json(corrupt_manifest, payload)
    _expect_error(
        lambda: load_source_manifest(
            str(corrupt_manifest),
            str(corrupt_checkpoint),
            dataset="office31",
            source_name="amazon",
            num_classes=31,
            model_name="deit_small_patch16_224.fb_in1k",
        ),
        "sha-256 mismatch",
    )


def _check_end_to_end(root, base):
    install_real_fake_checkpoint(root)
    effective = resolve_config(base, PROJECT_ROOT)
    summary = run_source_only_experiment(
        effective, PROJECT_ROOT, model_factory=tiny_factory
    )
    expected = 100.0 / 31.0
    assert math.isclose(summary["Acc"], expected)
    assert math.isclose(summary["overall-Acc"], expected)
    assert summary["processed_sample_count"] == 31
    assert summary["target_batch_count"] == 2
    assert summary["adaptation_steps"] == 0
    assert summary["optimizer_created"] is False
    assert summary["backward_calls"] == 0
    assert summary["prediction_passes"] == 1
    assert summary["single_evaluation_pass"] is True
    assert summary["model_state_unchanged"] is True
    output = Path(summary["output_dir"])
    for name in ("config.yaml", "manifest.json", "metrics.jsonl", "summary.json"):
        assert (output / name).is_file()
    return effective


def _check_plan_and_summary(root, base):
    plan = build_plan(load_yaml(MATRIX_PATH), base, CONFIG_PATH, MATRIX_PATH)
    assert plan["experiment_count"] == 7
    assert plan["supports_stream_resume"] is False
    assert plan["supports_generic_summary"] is False
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
        write_json(
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
                "Acc": macro,
                "overall-Acc": overall,
                "class-names": (
                    list(VISDA_CLASS_NAMES)
                    if experiment["dataset"] == "visda-c"
                    else None
                ),
                "Acc-per-class": (
                    [macro] * 12
                    if experiment["dataset"] == "visda-c"
                    else None
                ),
                "class-count": (
                    [1] * 12
                    if experiment["dataset"] == "visda-c"
                    else None
                ),
                "adaptation_steps": 0,
                "optimizer_created": False,
                "loss_computed": False,
                "backward_calls": 0,
                "target_labels_usage": "evaluation_only",
                "prediction_passes": 1,
                "single_evaluation_pass": True,
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
    assert report["office31_six_direction_average"]["Acc"] == 12.5
    assert len(report["visda_classwise"]) == 12
    assert report["visda_classwise"][0] == {
        "class_id": 0,
        "class_name": "aeroplane",
        "Acc": 16.0,
        "sample_count": 1,
    }
    report_output = Path(root) / "report"
    write_report(report, report_output)
    for name in (
        "report.json",
        "per_transfer.csv",
        "report.csv",
        "report.md",
        "visda_classwise.csv",
        "visda_classwise_wide.csv",
    ):
        assert (report_output / name).is_file()
    markdown = (report_output / "report.md").read_text(encoding="utf-8")
    assert "VisDA-C per-class accuracy" in markdown
    assert "aeroplane" in markdown and "truck" in markdown
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
        base = prepare_assets(root, CONFIG_PATH)
        install_real_fake_checkpoint(root)
        _check_manifest_failures(root, base)
        _check_end_to_end(root, base)
        _check_plan_and_summary(root, base)
    print("DeiT TTDA source-only tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
