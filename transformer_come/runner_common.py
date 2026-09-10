"""Artifact, manifest and diagnostics helpers shared by the COME runners."""

from __future__ import annotations

import copy
import statistics
import sys
import time
from pathlib import Path

import torch

from transformer.source_only.runner import ARTIFACT_SCHEMA_VERSION, _git_info

from .common import invalid_reason_of, utc_now
from .identity import PROTOCOL_DOCUMENT


def device_from_config(config: dict) -> torch.device:
    if config["runtime"]["device"] == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda:0")
    return torch.device("cpu")


def manifest_base(
    config: dict, *, variant: str, project_root: Path, started_at: str, extra: dict
) -> dict:
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "running",
        "created_at_utc": started_at,
        "command": list(sys.argv),
        "git": _git_info(project_root),
        "protocol_revision": config["protocol_revision"],
        "protocol_document": config.get("protocol_document", PROTOCOL_DOCUMENT),
        "implementation_revision": config["implementation_revision"],
        "experiment_key": config["experiment_key"],
        "scientific_config_sha256": config["scientific_config_sha256"],
        "dataset": config["dataset"],
        "transfer": config["transfer"],
        "formal_seed": config["formal_seed"],
        "method": "come",
        "variant": variant,
        "target_labels_usage": "PU/FO metrics only",
        "adaptation": copy.deepcopy(config["adaptation"]),
        "optimization": copy.deepcopy(config["optimization"]),
        "come": copy.deepcopy(config["come"]),
        "come_objective": copy.deepcopy(config["come_objective"]),
    }
    manifest.update(extra)
    return manifest


def failure_payload(config: dict, *, variant: str, error: BaseException, started_at: str) -> dict:
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "failed",
        "result_validity": "invalid",
        "invalid_reason": invalid_reason_of(error),
        "method": "come",
        "variant": variant,
        "dataset": config["dataset"],
        "transfer": config["transfer"],
        "formal_seed": config["formal_seed"],
        "experiment_key": config["experiment_key"],
        "error_type": type(error).__name__,
        "error": str(error),
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
    }


def collapse_summary(records: list[dict]) -> dict:
    """Aggregate the per-batch collapse diagnostics of one stream."""

    if not records:
        return {}
    return {
        "collapse_diagnostics_last_batch": records[-1],
        "predicted_class_count_min": min(
            row["predicted_class_count"] for row in records
        ),
        "predicted_class_count_mean": statistics.fmean(
            row["predicted_class_count"] for row in records
        ),
        "dominant_class_ratio_mean": statistics.fmean(
            row["dominant_class_ratio"] for row in records
        ),
        "dominant_class_ratio_max": max(row["dominant_class_ratio"] for row in records),
        "mean_softmax_entropy_mean": statistics.fmean(
            row["mean_softmax_entropy"] for row in records
        ),
        "come_opinion_entropy_mean": statistics.fmean(
            row["come_opinion_entropy"] for row in records
        ),
        "come_mean_uncertainty_mass_mean": statistics.fmean(
            row["come_mean_uncertainty_mass"] for row in records
        ),
    }


def gpu_memory_summary(peak_allocated: list[float], peak_reserved: list[float]) -> dict:
    return {
        "gpu_peak_allocated_mean_mb": float(statistics.fmean(peak_allocated)),
        "gpu_peak_allocated_max_mb": float(max(peak_allocated)),
        "gpu_peak_reserved_mean_mb": float(statistics.fmean(peak_reserved)),
        "gpu_peak_reserved_max_mb": float(max(peak_reserved)),
    }




def run_fo_pass(
    model,
    fo_loader,
    device: torch.device,
    config: dict,
    metrics_path: Path,
    *,
    show_progress: bool,
) -> tuple:
    """Independent read-only full-target CenterCrop evaluation.

    FO never calls the COME objective, never updates parameters and never
    consumes the online augmentation generator.
    """

    from tqdm.auto import tqdm

    from transformer.source_only.metrics import FixedClassMeter
    from transformer.source_only.runner import _append_jsonl, _sync

    fo_meter = FixedClassMeter(config["num_classes"], config["class_names"])
    fo_indices: list[int] = []
    fo_runtimes: list[float] = []
    fo_progress = tqdm(
        fo_loader,
        desc=f"FO {config['dataset']} {config['transfer']}",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
        disable=not show_progress,
    )
    with torch.inference_mode():
        for batch_index, (images, labels, indices) in enumerate(fo_progress, start=1):
            images = images.to(device, non_blocking=True)
            _sync(device)
            fo_started = time.perf_counter()
            logits = model(images)
            _sync(device)
            fo_runtime = float(time.perf_counter() - fo_started)
            predictions = logits.argmax(dim=1).cpu()
            labels = torch.as_tensor(labels, dtype=torch.int64).cpu()
            indices = torch.as_tensor(indices, dtype=torch.int64).cpu()
            fo_meter.update(labels, predictions)
            fo_indices.extend(int(index) for index in indices.tolist())
            fo_runtimes.append(fo_runtime)
            batch_correct = int((predictions == labels).sum().item())
            _append_jsonl(
                metrics_path,
                {
                    "schema_version": ARTIFACT_SCHEMA_VERSION,
                    "event": "fo_batch",
                    "batch_index": batch_index,
                    "batch_size": int(labels.numel()),
                    "correct": batch_correct,
                    "FO-batch-overall-Acc": 100.0 * batch_correct / labels.numel(),
                    "fo_batch_runtime_sec": fo_runtime,
                    "adaptation_steps": 0,
                    "objective_call_count": 0,
                    "fo_is_read_only": True,
                },
            )
            cumulative = (
                100.0
                * fo_meter.confusion.diagonal().sum().item()
                / fo_meter.sample_count
            )
            fo_progress.set_postfix(
                samples=fo_meter.sample_count, overall=f"{cumulative:.2f}%"
            )
    fo_progress.close()
    return fo_meter, fo_indices, fo_runtimes


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "collapse_summary",
    "device_from_config",
    "failure_payload",
    "gpu_memory_summary",
    "manifest_base",
    "run_fo_pass",
]
