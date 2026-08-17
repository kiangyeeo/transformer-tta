#!/usr/bin/env python3
"""CPU contracts for DeiT OTTA source-only; no network or real W0 is used."""

import copy
import json
import math
import os.path as osp
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
TEST_ROOT = osp.dirname(osp.abspath(__file__))
for path in (PROJECT_ROOT, TEST_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from deit_ttda_source_only_test import (  # noqa: E402
    _install_real_fake_checkpoint,
    _prepare_assets,
    tiny_factory,
)
from shot_otta.otta.source_only import run_source_only_otta_experiment  # noqa: E402
from shot_otta.ttda.config import (  # noqa: E402
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


def _otta_base(root):
    prepared = _prepare_assets(root)
    config = load_yaml(CONFIG_PATH)
    config["data"]["list_root"] = prepared["data"]["list_root"]
    config["model"]["source_checkpoint_root"] = prepared["model"][
        "source_checkpoint_root"
    ]
    config["device"] = {"type": "cpu", "gpu_id": "0"}
    config["evaluation"].update(
        {"batch_size": 30, "workers": 0, "amp": False, "pin_memory": False}
    )
    config["output"]["root"] = str((Path(root) / "otta_runs").resolve())
    return config


def _check_end_to_end(root, base):
    _install_real_fake_checkpoint(root)
    effective = resolve_config(base, PROJECT_ROOT, expected_task="otta")
    summary = run_source_only_otta_experiment(
        effective, PROJECT_ROOT, model_factory=tiny_factory
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
    assert summary["fo_prediction_passes"] == 0
    assert summary["fo_predictions_reused"] is True
    assert summary["PU-equals-FO"] is True
    assert summary["model_state_unchanged"] is True
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
    assert all(entry["task"] == "otta" for entry in plan["experiments"])
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
        _install_real_fake_checkpoint(root)
        _check_protocol_identity_separation(base)
        _, summary = _check_end_to_end(root, base)
        _check_plan_and_summary(root, base, summary)
    print("DeiT OTTA source-only tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
