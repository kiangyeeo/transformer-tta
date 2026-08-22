"""Sample-accurate Office and fixed-class VisDA metric accumulation."""

from __future__ import annotations

import statistics

import torch


class FixedClassMeter:
    """Accumulate an integer confusion matrix without averaging batches."""

    def __init__(self, num_classes: int, class_names: list[str]) -> None:
        if len(class_names) != num_classes:
            raise ValueError("class_names length must equal num_classes")
        self.num_classes = int(num_classes)
        self.class_names = tuple(class_names)
        self.confusion = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.int64
        )

    def update(self, labels, predictions) -> None:
        labels = torch.as_tensor(labels, dtype=torch.int64).cpu().reshape(-1)
        predictions = (
            torch.as_tensor(predictions, dtype=torch.int64).cpu().reshape(-1)
        )
        if labels.shape != predictions.shape:
            raise ValueError("labels and predictions must have identical shapes")
        if labels.numel() == 0:
            return
        if (
            torch.any(labels < 0)
            or torch.any(labels >= self.num_classes)
            or torch.any(predictions < 0)
            or torch.any(predictions >= self.num_classes)
        ):
            raise ValueError("labels or predictions are outside fixed classes")
        flat = labels * self.num_classes + predictions
        self.confusion += torch.bincount(
            flat, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

    @property
    def sample_count(self) -> int:
        return int(self.confusion.sum().item())

    def compute(self, dataset: str) -> dict:
        if self.sample_count == 0:
            raise ValueError("Cannot compute metrics without samples")
        class_count = self.confusion.sum(dim=1)
        missing = torch.nonzero(class_count == 0).flatten().tolist()
        if missing:
            raise ValueError(f"Evaluation is missing fixed classes: {missing}")
        class_correct = self.confusion.diagonal()
        per_class = (
            100.0 * class_correct.to(torch.float64) / class_count.to(torch.float64)
        )
        overall = float(
            100.0 * class_correct.sum().item() / self.sample_count
        )
        macro = float(per_class.mean().item())
        values = [float(value) for value in per_class.tolist()]
        worst_id = min(range(self.num_classes), key=values.__getitem__)
        primary = overall if dataset == "office31" else macro
        if dataset not in {"office31", "visda-c"}:
            raise ValueError(f"Unsupported dataset: {dataset}")
        return {
            "Acc": primary,
            "overall-Acc": overall,
            "mean-class-Acc": macro,
            "Acc-per-class": values,
            "Acc-per-class-by-name": {
                name: values[index]
                for index, name in enumerate(self.class_names)
            },
            "class-correct": [int(value) for value in class_correct.tolist()],
            "class-count": [int(value) for value in class_count.tolist()],
            "correct": int(class_correct.sum().item()),
            "sample-count": self.sample_count,
            "worst-class-Acc": values[worst_id],
            "worst-class-id": worst_id,
            "worst-class-name": self.class_names[worst_id],
            "class-std": statistics.pstdev(values),
        }


def prefixed(metrics: dict, prefix: str) -> dict:
    return {f"{prefix}-{key}": value for key, value in metrics.items()}
