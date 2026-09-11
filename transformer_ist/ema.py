"""Exactly-once outer-batch parameter EMA for non-LBI IST."""

from __future__ import annotations

import torch


class OuterBatchEMA:
    def __init__(self, momentum: float) -> None:
        self.momentum = float(momentum)
        if not 0.0 <= self.momentum <= 1.0:
            raise ValueError("EMA momentum must be in [0,1]")
        self._anchor = None
        self._batch_index = None
        self.commit_count = 0

    def begin_batch(self, batch_index: int, named_parameters) -> None:
        if self._anchor is not None or int(batch_index) != self.commit_count:
            raise RuntimeError("EMA begin does not match processed-batch order")
        self._batch_index = int(batch_index)
        self._anchor = {
            name: parameter.detach().clone() for name, parameter in named_parameters
        }

    def commit(self, batch_index: int, named_parameters) -> None:
        if self._anchor is None or int(batch_index) != self._batch_index:
            raise RuntimeError("EMA commit does not match its batch anchor")
        with torch.no_grad():
            for name, parameter in named_parameters:
                parameter.copy_(
                    self.momentum * self._anchor[name]
                    + (1.0 - self.momentum) * parameter.detach()
                )
        self._anchor = None
        self._batch_index = None
        self.commit_count += 1
