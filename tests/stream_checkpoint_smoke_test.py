#!/usr/bin/env python3
"""CPU-only round-trip coverage for resumable VISDA stream state."""

import os
import os.path as osp
import sys
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim


PROJECT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from shot_otta.trainer import (  # noqa: E402
    _load_stream_checkpoint,
    _save_stream_checkpoint,
)


def main():
    config = {
        "experiment_key": "checkpoint-smoke-key",
        "experiment_config_sha256": "a" * 64,
    }
    with tempfile.TemporaryDirectory() as directory:
        net_f = nn.Linear(2, 2)
        net_b = nn.Linear(2, 2)
        net_c = nn.Linear(2, 2)
        optimizer = optim.SGD(net_b.parameters(), lr=0.1, momentum=0.9)
        loss = net_b(torch.ones(1, 2)).sum()
        loss.backward()
        optimizer.step()
        expected_weight = net_b.weight.detach().clone()

        _save_stream_checkpoint(
            output_dir=directory,
            config=config,
            run_id="smoke",
            net_f=net_f,
            net_b=net_b,
            net_c=net_c,
            optimizer=optimizer,
            iteration=3,
            loader_batches_processed=3,
            all_post_predictions=[torch.tensor([1, 0])],
            all_online_labels=[torch.tensor([1, 1])],
            dynamic_selection_history=[{"budget_reached": True}],
            started_at_utc="2026-01-01T00:00:00+00:00",
            runtime_seconds=12.5,
        )

        for parameter in net_b.parameters():
            parameter.data.zero_()
        payload, _ = _load_stream_checkpoint(
            directory, config, net_f, net_b, net_c, optimizer
        )
        assert payload["iteration"] == 3
        assert payload["loader_batches_processed"] == 3
        assert payload["runtime_seconds"] == 12.5
        assert torch.equal(net_b.weight, expected_weight)
        assert osp.isfile(
            osp.join(directory, "checkpoints", "stream_state.json")
        )
    print("stream checkpoint smoke test passed")


if __name__ == "__main__":
    main()
