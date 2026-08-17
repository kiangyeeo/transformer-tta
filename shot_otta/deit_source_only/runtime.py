"""Shared deterministic runtime for DeiT source-only TTA controls."""

import hashlib
import importlib.metadata
import platform
import random
import time
from contextlib import nullcontext
from datetime import datetime, timezone

import numpy as np
import torch

from source_training.deit_data import (
    SourceImageList,
    build_loader,
    build_transforms,
    read_image_records,
)


ARTIFACT_SCHEMA_VERSION = 3


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def set_reproducibility(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = False


def autocast_context(device, enabled):
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
        raise ValueError("DeiT source-only evaluation requires target samples")
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


def prediction_sha256(indices, labels, predictions):
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


def build_target_loader(config):
    """Build the frozen deterministic target loader and return its records."""
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
    device_type = config["device"]["type"]
    loader = build_loader(
        dataset,
        batch_size=config["evaluation"]["batch_size"],
        workers=config["evaluation"]["workers"],
        shuffle=False,
        seed=config["seed"],
        pin_memory=(
            config["evaluation"]["pin_memory"] and device_type == "cuda"
        ),
    )
    return records, loader


def evaluate_model(model, loader, device, *, amp, on_batch=None, progress_label=None):
    """Evaluate one complete loader pass and retain exact sample indices.

    When ``progress_label`` is provided, a short flush-buffered progress line
    is printed roughly every 10% of the loader so long runs are observable
    through the launcher's redirected ``stdout.log``.
    """
    all_labels, all_predictions, all_indices = [], [], []
    model.eval()
    total_batches = len(loader)
    progress_every = max(1, total_batches // 10)
    started = time.perf_counter()
    samples_seen = 0
    with torch.inference_mode():
        for batch_index, (images, labels, indices) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            with autocast_context(device, amp):
                logits = model(images)
            predictions = logits.argmax(dim=1).cpu()
            labels = torch.as_tensor(labels).cpu()
            indices = torch.as_tensor(indices).cpu()
            all_predictions.append(predictions)
            all_labels.append(labels)
            all_indices.append(indices)
            samples_seen += int(labels.numel())
            if on_batch is not None:
                on_batch(batch_index, labels, predictions, indices)
            if (
                progress_label is not None
                and (
                    batch_index % progress_every == 0
                    or batch_index == total_batches
                )
            ):
                elapsed = time.perf_counter() - started
                peak_mib = (
                    int(torch.cuda.max_memory_allocated(device)) / 1048576.0
                    if device.type == "cuda"
                    else 0.0
                )
                print(
                    f"[{progress_label}] batch {batch_index}/{total_batches} "
                    f"samples {samples_seen} elapsed {elapsed:.1f}s "
                    f"peak_gpu {peak_mib:.0f} MiB",
                    flush=True,
                )
    if not all_predictions:
        raise RuntimeError("Target loader produced no batches")
    return (
        torch.cat(all_labels).numpy(),
        torch.cat(all_predictions).numpy(),
        torch.cat(all_indices).numpy(),
    )


def require_sequential_complete(indices, record_count, *, protocol):
    expected = np.arange(record_count, dtype=np.int64)
    if not np.array_equal(indices, expected):
        raise RuntimeError(
            f"{protocol} target samples were skipped, duplicated, or reordered"
        )


def environment_record(device, amp_effective):
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": package_version("torchvision"),
        "timm": package_version("timm"),
        "safetensors": package_version("safetensors"),
        "cuda": torch.version.cuda,
        "device": str(device),
        "amp_effective": bool(amp_effective),
    }
