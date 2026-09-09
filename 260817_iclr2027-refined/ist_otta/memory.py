"""Causal IST memory bank with one commit per incoming outer batch."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MemorySnapshot:
    features: torch.Tensor
    labels: torch.Tensor
    batch_ids: torch.Tensor


class CausalMemoryBank:
    def __init__(self, max_len):
        self.max_len = int(max_len)
        if self.max_len <= 0:
            raise ValueError("memory max_len must be positive")
        self._features = None
        self._labels = None
        self._batch_ids = torch.empty(0, dtype=torch.long)
        self.commit_count = 0

    def __len__(self):
        return int(self._batch_ids.numel())

    def state_dict(self):
        """Serialize committed history at an outer-batch boundary."""
        return {
            "max_len": self.max_len,
            "features": (
                None if self._features is None else self._features.clone()
            ),
            "labels": None if self._labels is None else self._labels.clone(),
            "batch_ids": self._batch_ids.clone(),
            "commit_count": self.commit_count,
        }

    def load_state_dict(self, state):
        if int(state["max_len"]) != self.max_len:
            raise RuntimeError("IST memory checkpoint max_len mismatch")
        features = state["features"]
        labels = state["labels"]
        self._features = None if features is None else features.clone()
        self._labels = None if labels is None else labels.clone()
        self._batch_ids = state["batch_ids"].clone().to(dtype=torch.long)
        self.commit_count = int(state["commit_count"])
        if len(self) > self.max_len:
            raise RuntimeError("IST memory checkpoint exceeds max_len")
        if self.commit_count < 0:
            raise RuntimeError("IST memory checkpoint has invalid commit_count")
        if (self._features is None) != (self._labels is None):
            raise RuntimeError("IST memory checkpoint is incomplete")
        if self._features is not None and (
            self._features.shape[0] != len(self)
            or self._labels.shape[0] != len(self)
        ):
            raise RuntimeError("IST memory checkpoint lengths disagree")

    def snapshot_for(self, incoming_batch_index, device=None):
        incoming_batch_index = int(incoming_batch_index)
        if len(self) and bool(
            torch.any(self._batch_ids >= incoming_batch_index).item()
        ):
            raise RuntimeError("IST memory contains current/future batch state")
        if self._features is None:
            features = torch.empty((0, 0), dtype=torch.float32)
            labels = torch.empty((0, 0), dtype=torch.float32)
        else:
            features = self._features.detach().clone()
            labels = self._labels.detach().clone()
        snapshot = MemorySnapshot(
            features=features,
            labels=labels,
            batch_ids=self._batch_ids.detach().clone(),
        )
        if device is None:
            return snapshot
        return MemorySnapshot(
            features=snapshot.features.to(device),
            labels=snapshot.labels.to(device),
            batch_ids=snapshot.batch_ids.to(device),
        )

    def commit(self, batch_index, features, corrected_labels):
        batch_index = int(batch_index)
        if batch_index != self.commit_count:
            raise RuntimeError(
                "memory commits must occur exactly once in outer-batch order"
            )
        features = features.detach().to(device="cpu", dtype=torch.float32)
        corrected_labels = corrected_labels.detach().to(
            device="cpu", dtype=torch.float32
        )
        if features.ndim != 2 or corrected_labels.ndim != 2:
            raise ValueError("memory features and labels must be matrices")
        if features.shape[0] != corrected_labels.shape[0]:
            raise ValueError("memory feature/label lengths must match")
        batch_ids = torch.full(
            (features.shape[0],), batch_index, dtype=torch.long
        )
        self._features = (
            features.clone()
            if self._features is None
            else torch.cat((self._features, features), dim=0)
        )
        self._labels = (
            corrected_labels.clone()
            if self._labels is None
            else torch.cat((self._labels, corrected_labels), dim=0)
        )
        self._batch_ids = torch.cat((self._batch_ids, batch_ids), dim=0)
        if len(self) > self.max_len:
            self._features = self._features[-self.max_len :]
            self._labels = self._labels[-self.max_len :]
            self._batch_ids = self._batch_ids[-self.max_len :]
        self.commit_count += 1
