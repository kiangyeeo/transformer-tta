#!/usr/bin/env python3
"""CPU contracts for DeiT OTTA source-only; no network or real W0 is used."""

import copy
import json
import math
import os.path as osp
import sys
import tempfile
from pathlib import Path

import torch


PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
TEST_ROOT = osp.dirname(osp.abspath(__file__))
for path in (PROJECT_ROOT, TEST_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from deit_source_only_fixtures import (  # noqa: E402
    TinyDeiT,
    install_real_fake_checkpoint,
    prepare_assets,
)
from shot_otta.otta.source_only import run_source_only_otta_experiment  # noqa: E402
from shot_otta.deit_source_only.config import (  # noqa: E402
    VISDA_CLASS_NAMES,
    load_yaml,
    resolve_config,
)
from tools.plan_deit_otta_source_only import build_plan  # noqa: E402
from tools.summarize_deit_otta_source_only import (  # noqa: E402
    build_report,
    write_report,
)


CONFIG_PATH = osp.join(PROJECT_ROOT, "configs", "deit_otta_source_only.yaml")
MATRIX_PATH = osp.join(
    PROJECT_ROOT, "experiments", "deit_otta_source_only.yaml"
)


class CountingTinyDeiT(TinyDeiT):
    total_forward_calls = 0

    def forward(self, images):
        type(self).total_forward_calls += 1
        return super().forward(images)


def counting_tiny_factory(model_name, **kwargs):
    assert model_name == "deit_small_patch16_224.fb_in1k"
    return CountingTinyDeiT(**kwargs)


class DriftingTinyDeiT(TinyDeiT):
    total_forward_calls = 0

    def forward(self, images):
        type(self).total_forward_calls += 1
        logits = super().forward(images)
        if type(self).total_forward_calls > 2:
            logits = torch.roll(logits, shifts=1, dims=1)
        return logits


def drifting_tiny_factory(model_name, **kwargs):
    assert model_name == "deit_small_patch16_224.fb_in1k"
    return DriftingTinyDeiT(**kwargs)


def _otta_base(root):
    return prepare_assets(root, CONFIG_PATH, runs_directory="otta_runs")


def _check_end_to_end(root, base):
    install_real_fake_checkpoint(root)
    effective = resolve_config(base, PROJECT_ROOT, expected_task="otta")
    CountingTinyDeiT.total_forward_calls = 0
    summary = run_source_only_otta_experiment(
        effective, PROJECT_ROOT, model_factory=counting_tiny_factory
    )
    expected = 100.0 / 31.0
    assert math.isclose(summary["PU-Acc"], expected)
    assert summary["PU-Acc"] == summary["FO-Acc"]
    assert summary["PU-overall-Acc"] == summary["FO-overall-Acc"]
    assert summary["PU-Acc-per-class"] == summary["FO-Acc-per-class"]
    assert summary["processed_sample_count"] == 31
    assert summary["target_batch_count"] == 2
    assert summary["tail_batch_size"] == 1
    assert summary["tail_batch_size_one_policy"] == "kept"
    assert summary["adaptation_steps"] == 0
    assert summary["adaptation_steps_per_batch"] == 0
    assert summary["optimizer_created"] is False
    assert summary["backward_calls"] == 0
    assert summary["stream_prediction_passes"] == 1
    assert summary["fo_prediction_passes"] == 1
    assert summary["prediction_passes"] == 2
    assert summary["fo_predictions_reused"] is False
    assert summary["fo_evaluation_scope"] == "full_target_dataset"
    assert summary["PU-equals-FO"] is True
    assert summary["PU-predictions-equal-FO"] is True
    assert summary["model_state_unchanged"] is True
    assert summary["model_state_sha256_before"] == summary[
        "model_state_sha256_after_stream"
    ]
    assert summary["model_state_sha256_after_stream"] == summary[
        "model_state_sha256_after_fo"
    ]
    assert summary["stream_processed_sample_count"] == 31
    assert summary["fo_processed_sample_count"] == 31
    assert CountingTinyDeiT.total_forward_calls == 4
    output = Path(summary["output_dir"])
    metrics = [
        json.loads(line)
        for line in (output / "metrics.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    batch_rows = [row for row in metrics if row["event"] == "online_batch"]
    assert [row["sample_count"] for row in batch_rows] == [30, 1]
    assert [row["stream_batch"] for row in batch_rows] == [1, 2]
    assert all(row["model_update_applied"] is False for row in batch_rows)
    assert [row["state_carried_to_next_batch"] for row in batch_rows] == [
        True,
        False,
    ]
    return effective, summary


def _check_independent_fo_mismatch_is_rejected(base):
    mismatch_config = copy.deepcopy(base)
    mismatch_config["output"]["root"] += "_fo_mismatch"
    effective = resolve_config(
        mismatch_config, PROJECT_ROOT, expected_task="otta"
    )
    DriftingTinyDeiT.total_forward_calls = 0
    try:
        run_source_only_otta_experiment(
            effective,
            PROJECT_ROOT,
            model_factory=drifting_tiny_factory,
        )
    except RuntimeError as error:
        assert "PU and FO predictions differ" in str(error), str(error)
    else:
        raise AssertionError("Independent FO mismatch was not rejected")
    assert DriftingTinyDeiT.total_forward_calls == 4


def _fake_summary(experiment, template, index):
    num_classes = experiment["scientific_config"]["target_stream"][
        "num_classes"
    ]
    class_names = (
        list(VISDA_CLASS_NAMES)
        if experiment["dataset"] == "visda-c"
        else [f"class_{class_id:02d}" for class_id in range(num_classes)]
    )
    value = float(index + 10)
    summary = copy.deepcopy(template)
    summary.update(
        {
            "experiment_key": experiment["experiment_key"],
            "experiment_config_sha256": experiment[
                "experiment_config_sha256"
            ],
            "dataset": experiment["dataset"],
            "source": experiment["source"],
            "target": experiment["target"],
            "source_name": experiment["source_name"],
            "target_name": experiment["target_name"],
            "seed": experiment["seed"],
            "source_checkpoint_sha256": experiment[
                "source_checkpoint_sha256"
            ],
            "processed_sample_count": experiment["scientific_config"][
                "target_stream"
            ]["sample_count"],
            "target_batch_count": 2 if num_classes == 31 else 1,
            "class-names": class_names,
            "runtime": float(index + 1),
            "peak_gpu_memory_bytes": index,
            "model_state_sha256_before": "a" * 64,
            "model_state_sha256_after_stream": "a" * 64,
            "model_state_sha256_after_fo": "a" * 64,
            "model_state_sha256_after": "a" * 64,
            "PU-prediction-sha256": "b" * 64,
            "FO-prediction-sha256": "b" * 64,
        }
    )
    paired_values = {
        "Acc": value,
        "mean-class-Acc": value,
        "overall-Acc": value + 1.0,
        "Acc-per-class": [value] * num_classes,
        "class-count": [1] * num_classes,
        "worst-class-Acc": value,
        "worst-class-id": 0,
        "worst-class-name": class_names[0],
        "class-std": 0.0,
    }
    for prefix in ("PU", "FO"):
        for suffix, metric_value in paired_values.items():
            summary[f"{prefix}-{suffix}"] = copy.deepcopy(metric_value)
    return summary


def _check_plan_and_summary(root, base, template):
    plan = build_plan(load_yaml(MATRIX_PATH), base, CONFIG_PATH, MATRIX_PATH)
    assert plan["experiment_count"] == 7
    assert plan["task"] == "otta"
    assert plan["supports_stream_resume"] is False
    assert plan["supports_generic_summary"] is False
    assert all(entry["task"] == "otta" for entry in plan["experiments"])
    assert all(
        entry["scientific_config"]["evaluation"] == {
            "batch_size": 30,
            "amp": False,
            "deterministic": True,
            "stream_prediction_passes": 1,
            "fo_prediction_passes": 1,
            "final_prediction_policy": "independent_full_target_pass",
        }
        for entry in plan["experiments"]
    )
    assert all(
        "evaluate_deit_otta.py" in " ".join(entry["command_args"])
        for entry in plan["experiments"]
    )
    assert len({entry["experiment_key"] for entry in plan["experiments"]}) == 7
    assert len(
        {entry["source_checkpoint_sha256"] for entry in plan["experiments"]}
    ) == 4
    fake_runs = Path(root) / "fake_otta_runs"
    for index, experiment in enumerate(plan["experiments"]):
        directory = fake_runs / f"run-{index}"
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text(
            json.dumps(
                _fake_summary(experiment, template, index),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    report = build_report(plan, fake_runs)
    assert report["experiment_count"] == 7
    assert len(report["report_rows"]) == 8
    assert report["office31_six_direction_average"]["PU-Acc"] == 12.5
    assert report["office31_six_direction_average"]["FO-Acc"] == 12.5
    assert len(report["visda_classwise"]) == 12
    assert report["visda_classwise"][0] == {
        "class_id": 0,
        "class_name": "aeroplane",
        "PU-Acc": 16.0,
        "FO-Acc": 16.0,
        "sample_count": 1,
    }
    report_output = Path(root) / "otta_report"
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
    assert "VisDA-C per-class PU/FO accuracy" in markdown
    assert "aeroplane" in markdown and "truck" in markdown


def _check_protocol_identity_separation(base):
    otta = resolve_config(base, PROJECT_ROOT, expected_task="otta")
    ttda = copy.deepcopy(base)
    ttda["task"] = "ttda"
    ttda["metrics"] = {
        "primary": "macro_class_accuracy",
        "primary_output": "Acc",
        "class_denominator": "fixed_dataset_classes",
        "report_overall": True,
        "report_per_class": True,
    }
    ttda = resolve_config(ttda, PROJECT_ROOT, expected_task="ttda")
    assert otta["experiment_key"] != ttda["experiment_key"]
    assert otta["experiment_config_sha256"] != ttda[
        "experiment_config_sha256"
    ]


def main():
    with tempfile.TemporaryDirectory(prefix="deit_otta_source_only_") as root:
        base = _otta_base(root)
        install_real_fake_checkpoint(root)
        _check_protocol_identity_separation(base)
        _, summary = _check_end_to_end(root, base)
        _check_independent_fo_mismatch_is_rejected(base)
        _check_plan_and_summary(root, base, summary)
    print("DeiT OTTA source-only tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
