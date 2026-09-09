"""Per-online-batch efficiency protocol and deterministic aggregations."""

import math

import numpy as np

from protocol_constants import EFFICIENCY_PROTOCOL_REVISION

BATCH_EFFICIENCY_FIELDS = (
    "batch_index",
    "batch_size",
    "adapt_runtime_sec",
    "pu_runtime_sec",
    "online_runtime_sec",
    "peak_gpu_memory_allocated_bytes",
    "peak_gpu_memory_reserved_bytes",
    "peak_gpu_memory_allocated_mb",
    "peak_gpu_memory_reserved_mb",
)


def _finite_values(records, field):
    return [
        float(record[field])
        for record in records
        if record.get(field) is not None
        and math.isfinite(float(record[field]))
    ]


def _distribution(values):
    if not values:
        return {
            "mean": None,
            "std": None,
            "median": None,
            "p95": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
    }


def aggregate_batch_efficiency(records):
    """Aggregate raw batch records without redefining the batch protocol."""
    records = list(records or [])
    online = _finite_values(records, "online_runtime_sec")
    adapt = _finite_values(records, "adapt_runtime_sec")
    pu = _finite_values(records, "pu_runtime_sec")
    allocated = _finite_values(
        records, "peak_gpu_memory_allocated_mb"
    )
    reserved = _finite_values(
        records, "peak_gpu_memory_reserved_mb"
    )

    def memory_stats(values, prefix):
        return {
            f"{prefix}_mean_mb": (
                float(np.mean(values)) if values else None
            ),
            f"{prefix}_max_mb": (
                float(np.max(values)) if values else None
            ),
        }

    online_stats = _distribution(online)
    adapt_stats = _distribution(adapt)
    pu_stats = _distribution(pu)
    aggregate = {
        "online_batch_runtime_mean_sec": online_stats["mean"],
        "online_batch_runtime_std_sec": online_stats["std"],
        "online_batch_runtime_median_sec": online_stats["median"],
        "online_batch_runtime_p95_sec": online_stats["p95"],
        "online_compute_runtime_sec": float(sum(online)),
        "adapt_batch_runtime_mean_sec": adapt_stats["mean"],
        "adapt_batch_runtime_std_sec": adapt_stats["std"],
        "adapt_batch_runtime_median_sec": adapt_stats["median"],
        "adapt_batch_runtime_p95_sec": adapt_stats["p95"],
        "adapt_runtime_total_sec": float(sum(adapt)),
        "pu_batch_runtime_mean_sec": pu_stats["mean"],
        "pu_batch_runtime_std_sec": pu_stats["std"],
        "pu_batch_runtime_median_sec": pu_stats["median"],
        "pu_batch_runtime_p95_sec": pu_stats["p95"],
        "pu_runtime_total_sec": float(sum(pu)),
        "efficiency_batch_count": len(records),
    }
    aggregate.update(memory_stats(allocated, "gpu_peak_allocated"))
    aggregate.update(memory_stats(reserved, "gpu_peak_reserved"))
    return aggregate


def aggregate_random_efficiency(child_results):
    """Aggregate Random's three children as a formal single-mask cost."""
    children = list(child_results or [])

    def child_values(field):
        return [
            float(child[field])
            for child in children
            if child.get(field) is not None
        ]

    online_means = child_values("online_batch_runtime_mean_sec")
    online_totals = child_values("online_compute_runtime_sec")
    aggregate = {
        "online_batch_runtime_mean_sec": (
            float(np.mean(online_means)) if online_means else None
        ),
        "online_batch_runtime_mask_std_sec": (
            float(np.std(online_means, ddof=0))
            if online_means
            else None
        ),
        "online_compute_runtime_sec": (
            float(np.mean(online_totals)) if online_totals else 0.0
        ),
        "online_compute_runtime_mask_std_sec": (
            float(np.std(online_totals, ddof=0))
            if online_totals
            else None
        ),
        "random_total_online_compute_runtime_sec": float(
            sum(online_totals)
        ),
    }
    fo_values = child_values("fo_eval_runtime_sec")
    aggregate.update(
        {
            "fo_eval_runtime_sec": (
                float(np.mean(fo_values)) if fo_values else None
            ),
            "fo_eval_runtime_mask_std_sec": (
                float(np.std(fo_values, ddof=0)) if fo_values else None
            ),
            "random_total_fo_eval_runtime_sec": float(sum(fo_values)),
        }
    )
    for field in (
        "adapt_batch_runtime_mean_sec",
        "adapt_batch_runtime_std_sec",
        "adapt_runtime_total_sec",
        "pu_batch_runtime_mean_sec",
        "pu_batch_runtime_std_sec",
        "pu_runtime_total_sec",
    ):
        values = child_values(field)
        aggregate[field] = float(np.mean(values)) if values else None
    for field in (
        "online_batch_runtime_median_sec",
        "online_batch_runtime_p95_sec",
        "adapt_batch_runtime_median_sec",
        "adapt_batch_runtime_p95_sec",
        "pu_batch_runtime_median_sec",
        "pu_batch_runtime_p95_sec",
    ):
        values = child_values(field)
        aggregate[field] = float(np.mean(values)) if values else None
    for field in (
        "gpu_peak_allocated_mean_mb",
        "gpu_peak_reserved_mean_mb",
    ):
        values = child_values(field)
        aggregate[field] = float(np.mean(values)) if values else None
    for field in (
        "gpu_peak_allocated_max_mb",
        "gpu_peak_reserved_max_mb",
    ):
        values = child_values(field)
        aggregate[field] = float(np.max(values)) if values else None
    return aggregate
