"""Fixed 12-class VisDA-C evaluation and random-mask aggregation."""

import numpy as np

from .class_names import VISDA_CLASS_NAMES


CLASS_COUNT = len(VISDA_CLASS_NAMES)


def compute_metrics(labels, predictions, prefix):
    """Compute percentage metrics from a fixed 12 by 12 confusion matrix."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)
    if labels.shape != predictions.shape:
        raise ValueError("VisDA labels and predictions must have the same shape")
    if (
        np.any(labels < 0)
        or np.any(labels >= CLASS_COUNT)
        or np.any(predictions < 0)
        or np.any(predictions >= CLASS_COUNT)
    ):
        raise ValueError("VisDA class IDs must be in [0, 12)")
    matrix = np.bincount(
        CLASS_COUNT * labels + predictions,
        minlength=CLASS_COUNT**2,
    ).reshape(CLASS_COUNT, CLASS_COUNT)
    totals = matrix.sum(axis=1)
    per_class = np.divide(
        matrix.diagonal() * 100.0,
        totals,
        out=np.zeros(CLASS_COUNT, dtype=float),
        where=totals > 0,
    )
    overall = (
        float(matrix.diagonal().sum() * 100.0 / matrix.sum())
        if matrix.sum()
        else 0.0
    )
    worst_id = int(np.argmin(per_class))
    mean = float(per_class.mean())
    return {
        f"{prefix}-Acc": mean,
        f"{prefix}-Acc-per-class": per_class.tolist(),
        f"{prefix}-overall-Acc": overall,
        f"{prefix}-mean-class-Acc": mean,
        f"{prefix}-worst-class-Acc": float(per_class[worst_id]),
        f"{prefix}-worst-class-id": worst_id,
        f"{prefix}-worst-class-name": VISDA_CLASS_NAMES[worst_id],
        f"{prefix}-class-std": float(np.std(per_class, ddof=0)),
        "class-names": list(VISDA_CLASS_NAMES),
    }


def aggregate_random_mask_metrics(mask_results):
    """Aggregate fixed VisDA child metrics without changing mask seeds."""
    if not mask_results:
        raise ValueError("VisDA random aggregation requires at least one mask")
    output = {"class-names": list(VISDA_CLASS_NAMES)}
    for prefix in ("PU", "FO"):
        overall = np.asarray(
            [item[f"{prefix}-overall-Acc"] for item in mask_results], dtype=float
        )
        macro = np.asarray(
            [item[f"{prefix}-mean-class-Acc"] for item in mask_results], dtype=float
        )
        classes = np.asarray(
            [item[f"{prefix}-Acc-per-class"] for item in mask_results], dtype=float
        )
        if classes.shape[1:] != (CLASS_COUNT,):
            raise ValueError("VisDA random child per-class arrays must have length 12")
        means, stds = classes.mean(axis=0), classes.std(axis=0, ddof=0)
        worst_id = int(np.argmin(means))
        output.update(
            {
                f"mean_{prefix}-overall-Acc": float(overall.mean()),
                f"std_{prefix}-overall-Acc": float(overall.std(ddof=0)),
                f"mean_{prefix}-mean-class-Acc": float(macro.mean()),
                f"std_{prefix}-mean-class-Acc": float(macro.std(ddof=0)),
                f"{prefix}-overall-Acc": float(overall.mean()),
                f"{prefix}-mean-class-Acc": float(macro.mean()),
                f"{prefix}-Acc-per-class": means.tolist(),
                f"{prefix}-Acc-per-class-mean": means.tolist(),
                f"{prefix}-Acc-per-class-std": stds.tolist(),
                f"{prefix}-worst-class-Acc": float(means[worst_id]),
                f"{prefix}-worst-class-id": worst_id,
                f"{prefix}-worst-class-name": VISDA_CLASS_NAMES[worst_id],
                f"{prefix}-class-std": float(np.std(means, ddof=0)),
            }
        )
    output["PU-Acc"] = output["PU-mean-class-Acc"]
    output["FO-Acc"] = output["FO-mean-class-Acc"]
    return output
