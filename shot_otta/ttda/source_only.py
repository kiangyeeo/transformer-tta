"""Strict no-adaptation DeiT TTDA evaluation and structured artifacts."""

import hashlib
import importlib.metadata
import os.path as osp
import platform
import random
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone

import numpy as np
import torch

from shot_otta.artifacts import (
    append_jsonl,
    config_sha256,
    create_run_dir,
    dump_json,
    git_info,
)
from shot_otta.backbones.deit import hash_model_state, load_frozen_deit_source
from shot_otta.config import dump_yaml
from shot_otta.ttda.config import experiment_output_root
from source_training.deit_data import (
    SourceImageList,
    build_loader,
    build_transforms,
    read_image_records,
)


ARTIFACT_SCHEMA_VERSION = 2


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _set_reproducibility(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = False


def _autocast_context(device, enabled):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def compute_fixed_class_metrics(
    labels,
    predictions,
    *,
    num_classes,
    class_names,
):
    """Compute fixed-denominator macro, overall, and per-class accuracy."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)
    if labels.shape != predictions.shape:
        raise ValueError("Labels and predictions must have identical shapes")
    if len(class_names) != num_classes:
        raise ValueError("class_names length does not match num_classes")
    if labels.size == 0:
        raise ValueError("TTDA evaluation requires at least one target sample")
    if (
        np.any(labels < 0)
        or np.any(labels >= num_classes)
        or np.any(predictions < 0)
        or np.any(predictions >= num_classes)
    ):
        raise ValueError("Labels or predictions are outside the fixed classes")
    matrix = np.bincount(
        num_classes * labels + predictions,
        minlength=num_classes**2,
    ).reshape(num_classes, num_classes)
    counts = matrix.sum(axis=1)
    missing = np.flatnonzero(counts == 0).tolist()
    if missing:
        raise ValueError(f"Target evaluation is missing classes: {missing}")
    per_class = matrix.diagonal() * 100.0 / counts
    macro = float(per_class.mean())
    overall = float(matrix.diagonal().sum() * 100.0 / matrix.sum())
    worst_id = int(np.argmin(per_class))
    return {
        "Acc": macro,
        "mean-class-Acc": macro,
        "overall-Acc": overall,
        "Acc-per-class": per_class.astype(float).tolist(),
        "class-count": counts.astype(int).tolist(),
        "worst-class-Acc": float(per_class[worst_id]),
        "worst-class-id": worst_id,
        "worst-class-name": class_names[worst_id],
        "class-std": float(np.std(per_class, ddof=0)),
    }


def _prediction_sha256(indices, labels, predictions):
    digest = hashlib.sha256()
    for name, values in (
        ("indices", indices),
        ("labels", labels),
        ("predictions", predictions),
    ):
        array = np.asarray(values, dtype=np.int64).reshape(-1)
        digest.update(name.encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def evaluate_frozen_model(model, loader, device, *, amp):
    labels, predictions, indices = [], [], []
    model.eval()
    with torch.inference_mode():
        for images, batch_labels, batch_indices in loader:
            images = images.to(device, non_blocking=True)
            with _autocast_context(device, amp):
                logits = model(images)
            predictions.append(logits.argmax(dim=1).cpu())
            labels.append(torch.as_tensor(batch_labels).cpu())
            indices.append(torch.as_tensor(batch_indices).cpu())
    if not predictions:
        raise RuntimeError("Target loader produced no batches")
    return (
        torch.cat(labels).numpy(),
        torch.cat(predictions).numpy(),
        torch.cat(indices).numpy(),
    )


def _environment_record(device, amp_effective):
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": _package_version("torchvision"),
        "timm": _package_version("timm"),
        "safetensors": _package_version("safetensors"),
        "cuda": torch.version.cuda,
        "device": str(device),
        "amp_effective": bool(amp_effective),
    }


def run_source_only_experiment(config, project_root, *, model_factory=None):
    """Evaluate one source checkpoint without any adaptation side effects."""
    output_root = experiment_output_root(config)
    run_id, output_dir = create_run_dir(
        output_root,
        config["task_name"],
        run_name=config["output"].get("run_name"),
    )
    paths = {
        "config": osp.join(output_dir, "config.yaml"),
        "manifest": osp.join(output_dir, "manifest.json"),
        "metrics": osp.join(output_dir, "metrics.jsonl"),
        "summary": osp.join(output_dir, "summary.json"),
    }
    dump_yaml(paths["config"], config)
    started_at = _utc_now()
    timer = time.perf_counter()
    base_manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at_utc": started_at,
        "method": "no_tta",
        "task": "ttda",
        "variant": "source_only",
        "dataset": config["data"]["dataset"],
        "source": config["data"]["source"],
        "target": config["data"]["target"],
        "source_name": config["data"]["source_name"],
        "target_name": config["data"]["target_name"],
        "seed": config["seed"],
        "experiment_key": config["experiment_key"],
        "experiment_config_sha256": config["experiment_config_sha256"],
        "effective_config_sha256": config_sha256(config),
        "command": list(sys.argv),
        "git": git_info(project_root),
        "source_checkpoint": config["source_checkpoint"],
        "target_data": {
            key: config["data"][key]
            for key in (
                "target_list",
                "target_list_sha256",
                "target_sample_count",
                "class_mapping_path",
                "class_mapping_sha256",
                "class_names",
            )
        },
        "adaptation": {
            "steps": 0,
            "optimizer_created": False,
            "loss_computed": False,
            "backward_calls": 0,
            "target_labels_usage": "evaluation_only",
        },
    }
    dump_json(paths["manifest"], base_manifest)

    try:
        device_type = config["device"]["type"]
        if device_type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")
            device = torch.device("cuda:0")
            torch.cuda.reset_peak_memory_stats(device)
        else:
            device = torch.device("cpu")
        amp_effective = bool(
            config["evaluation"]["amp"] and device.type == "cuda"
        )
        _set_reproducibility(
            config["seed"], config["evaluation"]["deterministic"]
        )
        model, checkpoint_metadata = load_frozen_deit_source(
            config,
            device,
            model_factory=model_factory,
        )
        state_before = hash_model_state(model)

        records = read_image_records(
            config["data"]["target_list"],
            config["data"]["num_classes"],
        )
        _, transform = build_transforms(
            {
                **config["preprocessing"],
                "horizontal_flip_probability": 0.0,
            }
        )
        dataset = SourceImageList(records, range(len(records)), transform)
        loader = build_loader(
            dataset,
            batch_size=config["evaluation"]["batch_size"],
            workers=config["evaluation"]["workers"],
            shuffle=False,
            seed=config["seed"],
            pin_memory=(
                config["evaluation"]["pin_memory"] and device.type == "cuda"
            ),
        )
        labels, predictions, indices = evaluate_frozen_model(
            model,
            loader,
            device,
            amp=amp_effective,
        )
        expected_indices = np.arange(len(records), dtype=np.int64)
        if not np.array_equal(indices, expected_indices):
            raise RuntimeError(
                "TTDA target samples were skipped, duplicated, or reordered"
            )
        state_after = hash_model_state(model)
        if state_before != state_after:
            raise RuntimeError("Source-only evaluation modified model state")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("Source-only evaluation unexpectedly created gradients")

        metric_kwargs = {
            "num_classes": config["data"]["num_classes"],
            "class_names": config["data"]["class_names"],
        }
        metrics = compute_fixed_class_metrics(
            labels, predictions, **metric_kwargs
        )
        runtime = float(time.perf_counter() - timer)
        peak_memory = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        )
        completed_at = _utc_now()
        invariant_record = {
            "adaptation_steps": 0,
            "optimizer_created": False,
            "loss_computed": False,
            "backward_calls": 0,
            "target_labels_usage": "evaluation_only",
            "prediction_passes": 1,
            "single_evaluation_pass": True,
            "model_state_sha256_before": state_before,
            "model_state_sha256_after": state_after,
            "model_state_unchanged": True,
            "prediction_sha256": _prediction_sha256(
                indices, labels, predictions
            ),
            "processed_sample_count": int(labels.size),
            "target_batch_count": int(len(loader)),
            "drop_last": False,
        }
        final_record = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "event": "final",
            "status": "completed",
            "run_id": run_id,
            "experiment_key": config["experiment_key"],
            "experiment_config_sha256": config[
                "experiment_config_sha256"
            ],
            **metrics,
            **invariant_record,
            "runtime": runtime,
            "peak_gpu_memory_bytes": peak_memory,
        }
        append_jsonl(paths["metrics"], final_record)
        summary = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "status": "completed",
            "run_id": run_id,
            "output_dir": output_dir,
            "method": "no_tta",
            "variant": "source_only",
            "task": "ttda",
            "dataset": config["data"]["dataset"],
            "source": config["data"]["source"],
            "target": config["data"]["target"],
            "source_name": config["data"]["source_name"],
            "target_name": config["data"]["target_name"],
            "source-target": (
                f"{config['data']['source_name']}-"
                f"{config['data']['target_name']}"
            ),
            "seed": config["seed"],
            "experiment_key": config["experiment_key"],
            "experiment_config_sha256": config[
                "experiment_config_sha256"
            ],
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "primary_metric": "macro_class_accuracy",
            "class-names": config["data"]["class_names"],
            "source_checkpoint_path": config["source_checkpoint"]["path"],
            "source_checkpoint_sha256": config["source_checkpoint"]["sha256"],
            "source_training_seed": config["source_checkpoint"][
                "source_training_seed"
            ],
            "source_best": config["source_checkpoint"]["best"],
            **metrics,
            **invariant_record,
            "runtime": runtime,
            "peak_gpu_memory_bytes": peak_memory,
            "environment": _environment_record(device, amp_effective),
        }
        dump_json(paths["summary"], summary)
        dump_json(
            paths["manifest"],
            {
                **base_manifest,
                "completed_at_utc": completed_at,
                "environment": summary["environment"],
                "checkpoint_metadata_verified": True,
                "loaded_checkpoint_scientific_config_sha256": (
                    checkpoint_metadata["scientific_config_sha256"]
                ),
                **invariant_record,
            },
        )
        print(
            f"Acc={summary['Acc']} "
            f"overall-Acc={summary['overall-Acc']} "
            f"runtime={runtime}",
            flush=True,
        )
        return summary
    except BaseException as error:
        dump_json(
            paths["summary"],
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "status": "failed",
                "run_id": run_id,
                "experiment_key": config["experiment_key"],
                "experiment_config_sha256": config[
                    "experiment_config_sha256"
                ],
                "method": "no_tta",
                "variant": "source_only",
                "task": "ttda",
                "dataset": config["data"]["dataset"],
                "source": config["data"]["source"],
                "target": config["data"]["target"],
                "seed": config["seed"],
                "error_type": type(error).__name__,
                "error": str(error),
                "started_at_utc": started_at,
                "completed_at_utc": _utc_now(),
            },
        )
        raise
