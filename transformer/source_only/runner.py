"""Run one no-adaptation online stream followed by independent FO evaluation."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Must be present before the first deterministic CUDA GEMM is created.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from .data import build_target_loaders
from .metrics import FixedClassMeter, prefixed
from .model import hash_model_state, load_frozen_source_model


ARTIFACT_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, indent=2, ensure_ascii=False)
            file_obj.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _append_jsonl(path: Path, payload: dict) -> None:
    with open(path, "a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _git_info(project_root: Path) -> dict:
    def call(*args: str):
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    dirty_text = call("status", "--porcelain")
    return {
        "commit": call("rev-parse", "HEAD"),
        "branch": call("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if dirty_text is None else bool(dirty_text),
    }


def _package_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment(device: torch.device) -> dict:
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": _package_version("torchvision"),
        "timm": _package_version("timm"),
        "cuda": torch.version.cuda,
        "device": str(device),
        "amp_effective": False,
    }
    if device.type == "cuda":
        result.update(
            {
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_total_memory_bytes": torch.cuda.get_device_properties(
                    device
                ).total_memory,
            }
        )
    return result


def set_reproducibility(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=False)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _runtime_stats(values: list[float], prefix: str) -> dict:
    if not values:
        raise ValueError("Runtime aggregation requires at least one batch")
    array = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean_sec": float(array.mean()),
        f"{prefix}_std_sec": float(array.std(ddof=0)),
        f"{prefix}_median_sec": float(np.median(array)),
        f"{prefix}_p95_sec": float(np.percentile(array, 95)),
        f"{prefix}_total_sec": float(array.sum()),
    }


def _forward_pass(
    *,
    model,
    loader,
    device: torch.device,
    meter: FixedClassMeter,
    phase: str,
    metrics_path: Path,
    show_progress: bool,
) -> tuple[list[dict], list[int]]:
    batch_records: list[dict] = []
    observed_indices: list[int] = []
    progress = tqdm(
        loader,
        desc=f"{phase} {meter.sample_count}/{len(loader.dataset)}",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
        disable=not show_progress,
    )
    with torch.inference_mode():
        for batch_index, (images, labels, indices) in enumerate(progress, start=1):
            images = images.to(device, non_blocking=True)
            _sync(device)  # H2D is outside the formal compute timer.
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            logits = model(images)
            _sync(device)
            compute_runtime = float(time.perf_counter() - started)

            predictions = logits.argmax(dim=1).cpu()
            labels = torch.as_tensor(labels, dtype=torch.int64).cpu()
            indices = torch.as_tensor(indices, dtype=torch.int64).cpu()
            meter.update(labels, predictions)
            observed_indices.extend(int(index) for index in indices.tolist())
            batch_correct = int((labels == predictions).sum().item())
            record = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "event": f"{phase.lower()}_batch",
                "phase": phase,
                "batch_index": batch_index,
                "batch_size": int(labels.numel()),
                "correct": batch_correct,
                "overall_accuracy": 100.0 * batch_correct / labels.numel(),
                "adapt_runtime_sec": 0.0,
                "pu_runtime_sec": compute_runtime if phase == "PU" else None,
                "online_runtime_sec": compute_runtime if phase == "PU" else None,
                "fo_batch_runtime_sec": compute_runtime if phase == "FO" else None,
                "adaptation_steps": 0,
                "optimizer_created": False,
                "loss_computed": False,
                "backward_calls": 0,
                "model_update_applied": False,
                "peak_gpu_memory_allocated_mb": (
                    torch.cuda.max_memory_allocated(device) / 1048576.0
                    if device.type == "cuda"
                    else 0.0
                ),
                "peak_gpu_memory_reserved_mb": (
                    torch.cuda.max_memory_reserved(device) / 1048576.0
                    if device.type == "cuda"
                    else 0.0
                ),
            }
            batch_records.append(record)
            _append_jsonl(metrics_path, record)
            progress.set_postfix(
                samples=meter.sample_count,
                acc=f"{100.0 * meter.confusion.diagonal().sum().item() / meter.sample_count:.2f}%",
            )
    if not batch_records:
        raise RuntimeError(f"{phase} loader produced no batches")
    return batch_records, observed_indices


def _validate_indices(
    indices: list[int], sample_count: int, *, phase: str, online: bool
) -> None:
    if len(indices) != sample_count or sorted(indices) != list(range(sample_count)):
        raise RuntimeError(f"{phase} skipped or duplicated target samples")
    if not online and indices != list(range(sample_count)):
        raise RuntimeError("FO pass is not a sequential complete-target pass")


def run_transfer(
    config: dict,
    project_root: Path,
    *,
    show_progress: bool = True,
) -> dict:
    """Run exactly one formal source->target source-only condition."""
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "config": output_dir / "effective_config.yaml",
        "manifest": output_dir / "manifest.json",
        "metrics": output_dir / "metrics.jsonl",
        "summary": output_dir / "summary.json",
    }
    with open(paths["config"], "w", encoding="utf-8") as file_obj:
        yaml.safe_dump(config, file_obj, sort_keys=False, allow_unicode=True)
    started_at = _utc_now()
    wall_started = time.perf_counter()
    base_manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "running",
        "created_at_utc": started_at,
        "command": list(sys.argv),
        "git": _git_info(project_root),
        "protocol_revision": config["protocol_revision"],
        "experiment_key": config["experiment_key"],
        "scientific_config_sha256": config["scientific_config_sha256"],
        "dataset": config["dataset"],
        "transfer": config["transfer"],
        "formal_seed": config["formal_seed"],
        "target_labels_usage": "PU/FO metrics only",
        "adaptation": "none",
    }
    _atomic_json(paths["manifest"], base_manifest)

    try:
        if config["runtime"]["device"] == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable")
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        set_reproducibility(
            config["formal_seed"], config["runtime"]["deterministic"]
        )
        model, checkpoint_record = load_frozen_source_model(config, device)
        state_before = hash_model_state(model)
        online_loader, fo_loader, stream_record = build_target_loaders(config)

        # Reset again after model construction. With workers=0 this also fixes
        # the online torchvision augmentation sequence to the formal seed.
        set_reproducibility(
            config["formal_seed"], config["runtime"]["deterministic"]
        )
        pu_meter = FixedClassMeter(config["num_classes"], config["class_names"])
        pu_records, pu_indices = _forward_pass(
            model=model,
            loader=online_loader,
            device=device,
            meter=pu_meter,
            phase="PU",
            metrics_path=paths["metrics"],
            show_progress=show_progress,
        )
        _validate_indices(
            pu_indices, stream_record["sample_count"], phase="PU", online=True
        )
        state_after_stream = hash_model_state(model)
        if state_after_stream != state_before:
            raise RuntimeError("Source-only online stream changed model state")

        fo_meter = FixedClassMeter(config["num_classes"], config["class_names"])
        fo_records, fo_indices = _forward_pass(
            model=model,
            loader=fo_loader,
            device=device,
            meter=fo_meter,
            phase="FO",
            metrics_path=paths["metrics"],
            show_progress=show_progress,
        )
        _validate_indices(
            fo_indices, stream_record["sample_count"], phase="FO", online=False
        )
        state_after_fo = hash_model_state(model)
        if state_after_fo != state_before:
            raise RuntimeError("Read-only FO pass changed model state")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("Source-only evaluation unexpectedly created gradients")

        pu_metrics = prefixed(pu_meter.compute(config["dataset"]), "PU")
        fo_metrics = prefixed(fo_meter.compute(config["dataset"]), "FO")
        pu_runtimes = [float(row["pu_runtime_sec"]) for row in pu_records]
        fo_runtimes = [float(row["fo_batch_runtime_sec"]) for row in fo_records]
        online_allocated = [
            float(row["peak_gpu_memory_allocated_mb"]) for row in pu_records
        ]
        online_reserved = [
            float(row["peak_gpu_memory_reserved_mb"]) for row in pu_records
        ]
        completed_at = _utc_now()
        summary = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "status": "completed",
            "protocol_revision": config["protocol_revision"],
            "variant": "source_only",
            "dataset": config["dataset"],
            "source": config["source"],
            "target": config["target"],
            "transfer": config["transfer"],
            "formal_seed": config["formal_seed"],
            "experiment_key": config["experiment_key"],
            "scientific_config_sha256": config["scientific_config_sha256"],
            "primary_metric": (
                "sample_overall_accuracy"
                if config["dataset"] == "office31"
                else "fixed_12_class_macro_accuracy"
            ),
            "class-names": config["class_names"],
            **pu_metrics,
            **fo_metrics,
            "stream": stream_record,
            "checkpoint": checkpoint_record,
            "model_state_sha256_before": state_before,
            "model_state_sha256_after_stream": state_after_stream,
            "model_state_sha256_after_fo": state_after_fo,
            "model_state_unchanged": True,
            "adaptation_steps": 0,
            "optimizer_created": False,
            "loss_computed": False,
            "backward_calls": 0,
            "pu_is_separate_read_only_forward": True,
            "fo_is_independent_full_target_pass": True,
            "preprocessing": config["preprocessing"],
            "online_transform": "Resize-square/RandomCrop/RandomHorizontalFlip",
            "fo_transform": "Resize-square/CenterCrop",
            "online_batch_size": config["batch_size"],
            "fo_batch_size": config["fo_batch_size"],
            "fo_batch_size_policy": "same_as_online_not_fc_times_three",
            "PU-FO-equality-required": False,
            "PU-FO-equality-note": (
                "W0 is identical, but PU uses seeded online augmentation and FO "
                "uses deterministic center crop."
            ),
            **_runtime_stats(pu_runtimes, "online_batch_runtime"),
            **_runtime_stats(pu_runtimes, "pu_batch_runtime"),
            **_runtime_stats(fo_runtimes, "fo_batch_runtime"),
            "adapt_runtime_total_sec": 0.0,
            "fo_eval_runtime_sec": float(sum(fo_runtimes)),
            "gpu_peak_allocated_mean_mb": float(statistics.fmean(online_allocated)),
            "gpu_peak_allocated_max_mb": float(max(online_allocated)),
            "gpu_peak_reserved_mean_mb": float(statistics.fmean(online_reserved)),
            "gpu_peak_reserved_max_mb": float(max(online_reserved)),
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "wall_runtime_sec": float(time.perf_counter() - wall_started),
            "environment": _environment(device),
            "output_dir": str(output_dir),
        }
        _append_jsonl(
            paths["metrics"],
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "event": "final",
                **summary,
            },
        )
        _atomic_json(paths["summary"], summary)
        _atomic_json(
            paths["manifest"],
            {
                **base_manifest,
                "status": "completed",
                "completed_at_utc": completed_at,
                "checkpoint": checkpoint_record,
                "stream": stream_record,
                "environment": summary["environment"],
                "model_state_unchanged": True,
            },
        )
        print(
            f"[{config['dataset']} {config['transfer']}] "
            f"PU-Acc={summary['PU-Acc']:.4f} FO-Acc={summary['FO-Acc']:.4f}",
            flush=True,
        )
        return summary
    except BaseException as error:
        _atomic_json(
            paths["summary"],
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "status": "failed",
                "dataset": config["dataset"],
                "transfer": config["transfer"],
                "formal_seed": config["formal_seed"],
                "experiment_key": config["experiment_key"],
                "error_type": type(error).__name__,
                "error": str(error),
                "started_at_utc": started_at,
                "completed_at_utc": _utc_now(),
            },
        )
        raise
