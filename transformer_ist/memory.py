"""Past-only causal IST feature/label memory."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MemorySnapshot:
    features: torch.Tensor
    labels: torch.Tensor
    batch_ids: torch.Tensor


class CausalMemoryBank:
    def __init__(self, max_len: int, feature_dim: int, num_classes: int) -> None:
        self.max_len = int(max_len)
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        if min(self.max_len, self.feature_dim, self.num_classes) <= 0:
            raise ValueError("memory dimensions must be positive")
        self._features = torch.empty((0, self.feature_dim), dtype=torch.float32)
        self._labels = torch.empty((0, self.num_classes), dtype=torch.float32)
        self._batch_ids = torch.empty(0, dtype=torch.long)
        self.commit_count = 0

    def __len__(self) -> int:
        return int(self._batch_ids.numel())

    def snapshot_for(self, batch_index: int, device) -> MemorySnapshot:
        if len(self) and bool(torch.any(self._batch_ids >= int(batch_index)).item()):
            raise RuntimeError("IST memory contains current/future batch state")
        return MemorySnapshot(
            self._features.detach().clone().to(device),
            self._labels.detach().clone().to(device),
            self._batch_ids.detach().clone().to(device),
        )

    def commit(self, batch_index: int, features, corrected_one_hot) -> None:
        if int(batch_index) != self.commit_count:
            raise RuntimeError("memory must commit exactly once in processed-batch order")
        features = features.detach().to(device="cpu", dtype=torch.float32).clone()
        labels = corrected_one_hot.detach().to(device="cpu", dtype=torch.float32).clone()
        if features.ndim != 2 or tuple(features.shape[1:]) != (self.feature_dim,):
            raise ValueError("memory feature shape mismatch")
        if labels.ndim != 2 or tuple(labels.shape[1:]) != (self.num_classes,):
            raise ValueError("memory label shape mismatch")
        if features.shape[0] != labels.shape[0]:
            raise ValueError("memory feature/label lengths differ")
        batch_ids = torch.full((features.shape[0],), int(batch_index), dtype=torch.long)
        self._features = torch.cat((self._features, features), dim=0)[-self.max_len :]
        self._labels = torch.cat((self._labels, labels), dim=0)[-self.max_len :]
        self._batch_ids = torch.cat((self._batch_ids, batch_ids), dim=0)[-self.max_len :]
        self.commit_count += 1

    def state_fingerprint(self) -> tuple:
        return (
            self.commit_count,
            self._features.clone(),
            self._labels.clone(),
            self._batch_ids.clone(),
        )
