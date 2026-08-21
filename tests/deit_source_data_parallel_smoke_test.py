#!/usr/bin/env python3
"""Two-GPU local-weight DeiT forward/backward smoke test."""

import os
import os.path as osp
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn


PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from source_training.deit_model import build_deit_source_model  # noqa: E402
from source_training.deit_trainer import (  # noqa: E402
    _cpu_state_dict,
    build_adamw,
    set_reproducibility,
)


def main():
    if torch.cuda.device_count() < 2:
        raise RuntimeError("This smoke test requires two visible CUDA devices")
    set_reproducibility(2026, deterministic=True)
    model, _ = build_deit_source_model(
        model_name="deit_small_patch16_224.fb_in1k",
        num_classes=31,
        pretrained_path=(
            "/home/nas3/biod/wangkangyi/checkpoints/"
            "deit_small_patch16_224.fb_in1k/model.safetensors"
        ),
    )
    model.cuda(0)
    optimizer, _ = build_adamw(
        model,
        {
            "backbone_lr": 6.25e-5,
            "head_lr": 6.25e-4,
            "weight_decay": 0.05,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
        },
    )
    model = nn.DataParallel(model, device_ids=[0, 1])
    inputs = torch.randn(4, 3, 224, 224, device="cuda:0")
    labels = torch.tensor([0, 1, 2, 3], device="cuda:0")
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=True):
        logits = model(inputs)
        loss = nn.functional.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    state_dict = _cpu_state_dict(model)
    assert "head.weight" in state_dict
    assert not any(name.startswith("module.") for name in state_dict)
    print(
        "two-GPU DeiT source smoke test passed: "
        f"loss={loss.item():.6f}, logits={tuple(logits.shape)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
