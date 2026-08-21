"""Deterministic CPU-only checks for the per-batch efficiency protocol."""

import copy
import math
import os.path as osp
import sys
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim


PROJECT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from experiment_identity import IMPLEMENTATION_REVISION  # noqa: E402
from shot_otta.efficiency import (  # noqa: E402
    aggregate_batch_efficiency,
    aggregate_random_efficiency,
)
from shot_otta.trainer import (  # noqa: E402
    RuntimeInstrumentation,
    _load_stream_checkpoint,
    _save_stream_checkpoint,
)
from core.lbi import SplitLBIEngine  # noqa: E402


def _record(batch_index, adapt, pu, allocated=None, reserved=None):
    record = {
        "batch_index": batch_index,
        "batch_size": 4,
        "adapt_runtime_sec": adapt,
        "pu_runtime_sec": pu,
        "online_runtime_sec": adapt + pu,
        "peak_gpu_memory_allocated_bytes": (
            allocated * 1024**2 if allocated is not None else None
        ),
        "peak_gpu_memory_reserved_bytes": (
            reserved * 1024**2 if reserved is not None else None
        ),
        "peak_gpu_memory_allocated_mb": allocated,
        "peak_gpu_memory_reserved_mb": reserved,
    }
    return record


def main():
    tracker = RuntimeInstrumentation(torch.device("cpu"))
    for index in range(3):
        tracker.begin_batch(index, 4)
        adapt = tracker.start_adaptation()
        tracker.finish_adaptation(adapt)
        pu = tracker.start_pu()
        tracker.finish_pu(pu)
        record = tracker.finish_batch()
        assert record["batch_index"] == index
        assert record["batch_size"] == 4
        assert record["online_runtime_sec"] == (
            record["adapt_runtime_sec"] + record["pu_runtime_sec"]
        )
    assert len(tracker.batch_efficiency_records) == 3

    aggregate = aggregate_batch_efficiency(
        [_record(0, 0.5, 0.5), _record(1, 1.0, 1.0), _record(2, 1.5, 1.5), _record(3, 2.0, 2.0)]
    )
    assert aggregate["online_batch_runtime_mean_sec"] == 2.5
    assert aggregate["online_batch_runtime_median_sec"] == 2.5
    assert aggregate["online_compute_runtime_sec"] == 10.0
    assert math.isclose(aggregate["online_batch_runtime_p95_sec"], 3.85)
    assert math.isclose(
        aggregate["online_batch_runtime_std_sec"], 1.118033988749895
    )

    memory = aggregate_batch_efficiency(
        [_record(0, 0.1, 0.1, 100, 150), _record(1, 0.1, 0.1, 120, 170), _record(2, 0.1, 0.1, 110, 160)]
    )
    assert memory["gpu_peak_allocated_mean_mb"] == 110.0
    assert memory["gpu_peak_allocated_max_mb"] == 120.0
    assert memory["gpu_peak_reserved_mean_mb"] == 160.0
    assert memory["gpu_peak_reserved_max_mb"] == 170.0

    random_summary = aggregate_random_efficiency(
        [
            {"online_batch_runtime_mean_sec": 1.0, "online_compute_runtime_sec": 10.0, "gpu_peak_allocated_mean_mb": 100.0, "gpu_peak_allocated_max_mb": 110.0, "gpu_peak_reserved_mean_mb": 150.0, "gpu_peak_reserved_max_mb": 160.0},
            {"online_batch_runtime_mean_sec": 1.2, "online_compute_runtime_sec": 12.0, "gpu_peak_allocated_mean_mb": 120.0, "gpu_peak_allocated_max_mb": 130.0, "gpu_peak_reserved_mean_mb": 170.0, "gpu_peak_reserved_max_mb": 180.0},
            {"online_batch_runtime_mean_sec": 0.8, "online_compute_runtime_sec": 8.0, "gpu_peak_allocated_mean_mb": 80.0, "gpu_peak_allocated_max_mb": 90.0, "gpu_peak_reserved_mean_mb": 130.0, "gpu_peak_reserved_max_mb": 140.0},
        ]
    )
    assert random_summary["online_batch_runtime_mean_sec"] == 1.0
    assert random_summary["online_compute_runtime_sec"] == 10.0
    assert random_summary["random_total_online_compute_runtime_sec"] == 30.0
    assert math.isclose(
        random_summary["online_batch_runtime_mask_std_sec"],
        0.16329931618554522,
    )
    assert random_summary["gpu_peak_allocated_max_mb"] == 130.0
    assert random_summary["gpu_peak_reserved_max_mb"] == 180.0

    parameter = nn.Parameter(torch.tensor([0.2, -0.3]))
    lbi_tracker = RuntimeInstrumentation(torch.device("cpu"))
    lbi_tracker.begin_batch(0, 2)
    adaptation_started = lbi_tracker.start_adaptation()
    lbi_result = SplitLBIEngine(2).run_step(
        [("candidate", parameter)],
        lambda: (parameter**2).sum(),
        {
            "alpha": 0.1,
            "kappa": 1.0,
            "nu": 1.0,
            "omega": 0.1,
            "stage1_max_steps": 1,
            "budget_tolerance": 0.0001,
            "stage2_lr": 0.01,
            "stage2_steps": 1,
            "delta_nonzero_tolerance": 1e-12,
            "support_threshold": 1e-4,
            "requested_budget": 0.5,
        },
        timing=lbi_tracker,
    )
    del lbi_result
    lbi_tracker.finish_adaptation(adaptation_started)
    pu_started = lbi_tracker.start_pu()
    lbi_tracker.finish_pu(pu_started)
    lbi_record = lbi_tracker.finish_batch()
    assert lbi_record["lbi_stage1_runtime_sec"] >= 0.0
    assert lbi_record["lbi_stage2_runtime_sec"] >= 0.0

    config = {
        "implementation_revision": IMPLEMENTATION_REVISION,
        "experiment_key": "efficiency-checkpoint-smoke",
        "experiment_config_sha256": "b" * 64,
    }
    with tempfile.TemporaryDirectory(prefix="per_batch_efficiency_") as directory:
        net_f = nn.Linear(2, 2)
        net_b = nn.Linear(2, 2)
        net_c = nn.Linear(2, 2)
        optimizer = optim.SGD(net_b.parameters(), lr=0.1)
        records = [_record(0, 1.0, 0.1), _record(1, 2.0, 0.2)]
        _save_stream_checkpoint(
            output_dir=directory,
            config=config,
            run_id="efficiency",
            net_f=net_f,
            net_b=net_b,
            net_c=net_c,
            optimizer=optimizer,
            iteration=2,
            loader_batches_processed=2,
            all_post_predictions=[torch.tensor([1])],
            all_online_labels=[torch.tensor([1])],
            dynamic_selection_history=[],
            started_at_utc="2026-01-01T00:00:00+00:00",
            runtime_seconds=2.0,
            batch_efficiency_records=records,
            runtime_segment_count=1,
        )
        payload, _ = _load_stream_checkpoint(
            directory, config, net_f, net_b, net_c, optimizer
        )
        assert [row["batch_index"] for row in payload["batch_efficiency_records"]] == [0, 1]

    print("per-batch efficiency smoke test passed")


if __name__ == "__main__":
    main()
