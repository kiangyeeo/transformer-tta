#!/usr/bin/env python3
"""C01 (CPU half): COME and SHOT load the same source W0 and agree on logits.

This uses the *actual* source checkpoints, on CPU, with a fixed synthetic
input tensor.  It proves the source-identity and initial-logit-equality half
of C01 without running any adaptation.  The remaining half - equality on real
target batches, on GPU, through the real loader - belongs to the P4 real-data
smoke.

Skips with an explicit message (exit code 0) when the source checkpoints are
not present, so it can run anywhere; it never downloads anything.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import transformer.candidate_dense.config as shot_candidate_config  # noqa: E402
import transformer.full_dense.config as shot_full_config  # noqa: E402
import transformer.full_dense.model as shot_full_model  # noqa: E402
import transformer_come.config as come_config  # noqa: E402
import transformer_come.model as come_model_module  # noqa: E402
from transformer_come.common import come_objective_step  # noqa: E402


TRANSFER = ("office31", "dslr", "amazon")
CHECKPOINT = Path(
    "/home/nas3/biod/wangkangyi/checkpoints/source_models/office31/dslr.pth"
)


COME_CONFIG = PROJECT_ROOT / "transformer_come" / "config.yaml"


def _resolve_shot(module, config_path: Path, output_dir: Path) -> dict:
    dataset, source, target = TRANSFER
    return module.resolve_transfer_config(
        module.load_config(config_path),
        project_root=PROJECT_ROOT,
        dataset=dataset,
        source=source,
        target=target,
        device="cpu",
        output_dir=output_dir,
    )


def _resolve_come(variant: str, output_dir: Path) -> dict:
    dataset, source, target = TRANSFER
    return come_config.resolve_transfer_config(
        come_config.load_config(COME_CONFIG),
        project_root=PROJECT_ROOT,
        dataset=dataset,
        source=source,
        target=target,
        variant=variant,
        device="cpu",
        output_dir=output_dir,
    )


def main() -> int:
    if not CHECKPOINT.is_file():
        print(
            json.dumps(
                {
                    "status": "SKIPPED",
                    "reason": f"source checkpoint not present: {CHECKPOINT}",
                }
            )
        )
        return 0

    shot = _resolve_shot(
        shot_full_config,
        PROJECT_ROOT / "transformer" / "full_dense" / "config.yaml",
        Path("/tmp/come-c01-shot"),
    )
    come = _resolve_come("full_dense", Path("/tmp/come-c01-come"))
    come_candidate = _resolve_come(
        "candidate_dense", Path("/tmp/come-c01-candidate")
    )

    # Same source W0/provenance/substrate, with the documented v2 host-LR
    # difference contributing to a distinct scientific identity.
    for field in (
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_manifest_path",
        "target_list",
        "target_list_sha256",
        "class_mapping_sha256",
        "model_name",
        "num_classes",
        "batch_size",
        "fo_batch_size",
        "workers",
        "preprocessing",
        "stream",
    ):
        assert shot[field] == come[field], field
    assert shot["optimization"] == {
        **come["optimization"],
        "lr": 1.0e-5,
    }
    assert come["optimization"]["lr"] == 1.0e-7
    assert come["checkpoint_sha256"] == come_candidate["checkpoint_sha256"]
    assert shot["scientific_config_sha256"] != come["scientific_config_sha256"]
    assert shot["method"] == "shot" and come["method"] == "come"

    device = torch.device("cpu")
    shot_model, shot_record, _, _ = shot_full_model.load_full_dense_model(shot, device)
    come_model, come_record, come_trainable, come_frozen, _ = (
        come_model_module.load_full_dense_model(come, device)
    )
    assert shot_record["sha256"] == come_record["sha256"] == come["checkpoint_sha256"]
    assert shot_record["path"] == come_record["path"]

    torch.manual_seed(2026)
    images = torch.randn(2, 3, 224, 224)
    shot_model.eval()
    come_model.eval()
    with torch.no_grad():
        shot_logits = shot_model(images)
        come_logits = come_model(images)
    # Bit-exact: COME changes the objective, not the source forward path.
    assert torch.equal(shot_logits, come_logits)
    assert come_logits.shape == (2, 31)

    # The same W0 also drives the candidate-dense scope identically.
    candidate_model, _, candidates, _, _ = (
        come_model_module.load_candidate_dense_model(come_candidate, device)
    )
    candidate_model.eval()
    with torch.no_grad():
        candidate_logits = candidate_model(images)
    assert torch.equal(shot_logits, candidate_logits)
    assert len(candidates) == 12
    assert sum(parameter.numel() for _, parameter in candidates) == 5_308_416

    # The COME objective on the actual W0 logits is finite, and C is 31.
    loss, logits, diagnostics = come_objective_step(come_model, images, 31)
    assert torch.equal(logits.detach(), come_logits)
    assert torch.isfinite(loss).item()
    assert diagnostics["class_count_c"] == 31
    head_names = {name for name, _ in come_frozen}
    assert head_names == {"head.weight", "head.bias"}
    assert all(parameter.grad is None for _, parameter in come_frozen)
    assert all(parameter.grad is not None for _, parameter in come_trainable)

    # A *.last.pth W0 must be refused.
    rejected = 0
    last_config = {**come, "checkpoint_path": come["checkpoint_path"].replace(
        ".pth", ".last.pth"
    )}
    try:
        come_model_module.load_full_dense_model(last_config, device)
    except ValueError as error:
        assert "last.pth" in str(error)
        rejected += 1
    else:
        raise AssertionError("A *.last.pth checkpoint was accepted as W0")
    # A mismatched hash must be refused.
    wrong_hash = {**come, "checkpoint_sha256": "0" * 64}
    try:
        come_model_module.load_full_dense_model(wrong_hash, device)
    except ValueError:
        rejected += 1
    else:
        raise AssertionError("A mismatched checkpoint SHA-256 was accepted")

    print(
        json.dumps(
            {
                "status": "PASS",
                "transfer": f"{TRANSFER[1]}->{TRANSFER[2]}",
                "checkpoint_sha256": come["checkpoint_sha256"],
                "shot_scientific_config_sha256": shot["scientific_config_sha256"],
                "come_scientific_config_sha256": come["scientific_config_sha256"],
                "initial_logits_bit_exact": True,
                "rejected_bad_source_cases": rejected,
                "come_loss_on_w0": float(loss.detach().item()),
            },
            sort_keys=True,
        )
    )
    print("Transformer COME source identity (C01, CPU half) tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
