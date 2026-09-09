#!/usr/bin/env python3
"""Tiny CPU/GPU-safe checks for runtime and memory instrumentation."""

import copy
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from shot_otta.trainer import (  # noqa: E402
    RuntimeInstrumentation,
    run_experiment,
)


def main():
    torch.manual_seed(2020)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)
    ).to(device)
    inputs = torch.randn(5, 4, device=device)
    model_state_before = copy.deepcopy(model.state_dict())
    rng_before = torch.get_rng_state()

    tracker = RuntimeInstrumentation(device, runtime_comparable=False)
    tracker.reset_peak_memory()
    outputs = []
    wall_with_checkpoint_sleep = 0.0
    for _ in range(3):
        wall_started = time.perf_counter()
        started = tracker.start_online_step()
        with torch.no_grad():
            outputs.append(model(inputs).argmax(dim=1))
        tracker.finish_online_step(started)
        # Simulates checkpoint callback / disk I/O after compute timing stops.
        time.sleep(0.01)
        wall_with_checkpoint_sleep += time.perf_counter() - wall_started

    fo_started = time.perf_counter()
    fo_output = tracker.measure_fo(
        lambda: model(inputs).argmax(dim=1)
    )
    fo_wall = time.perf_counter() - fo_started
    metadata = tracker.metadata()

    assert len(outputs) == 3
    assert tracker.online_compute_runtime_sec > 0.0
    assert tracker.fo_eval_runtime_sec > 0.0
    assert fo_wall >= tracker.fo_eval_runtime_sec
    assert wall_with_checkpoint_sleep > tracker.online_compute_runtime_sec
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(outputs[0], fo_output)
    for name, value in model.state_dict().items():
        assert torch.equal(value, model_state_before[name]), name
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert metadata["runtime_comparable"] is False
    assert metadata["torch_version"] == torch.__version__

    if torch.cuda.is_available():
        assert metadata["peak_gpu_memory_allocated_bytes"] >= 0
        assert metadata["peak_gpu_memory_reserved_bytes"] >= 0
        assert metadata["peak_gpu_memory_allocated_mb"] >= 0
        assert metadata["peak_gpu_memory_reserved_mb"] >= 0
        assert metadata["gpu_name"]
        assert metadata["gpu_total_memory_bytes"] > 0
    else:
        assert metadata["peak_gpu_memory_allocated_bytes"] is None
        assert metadata["peak_gpu_memory_reserved_bytes"] is None

    for kwargs in (
        {"enable_stream_checkpoint": True},
        {"resume_run_dir": "/unused"},
    ):
        try:
            run_experiment(
                {"variant": "module_random"}, str(PROJECT), **kwargs
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                "module_random unexpectedly enabled stream checkpointing"
            )

    print("runtime efficiency smoke test passed")


if __name__ == "__main__":
    main()
