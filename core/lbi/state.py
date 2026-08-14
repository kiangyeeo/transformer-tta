"""Local state containers for one online Split-LBI step."""

from dataclasses import dataclass
from typing import Any, Dict

import torch


TensorMap = Dict[str, torch.Tensor]


@dataclass
class LBIState:
    """State that is deliberately recreated for every online target batch."""

    theta_delta: TensorMap
    gamma: TensorMap
    z: TensorMap
    mask: TensorMap

    def clone(self):
        return LBIState(
            theta_delta={
                name: value.detach().clone()
                for name, value in self.theta_delta.items()
            },
            gamma={
                name: value.detach().clone()
                for name, value in self.gamma.items()
            },
            z={
                name: value.detach().clone()
                for name, value in self.z.items()
            },
            mask={
                name: value.detach().clone()
                for name, value in self.mask.items()
            },
        )


@dataclass
class LBIResult:
    """Tensors and lossless scalar statistics produced by one local run."""

    state: LBIState
    base_parameters: TensorMap
    refined_parameters: TensorMap
    applied_parameters: TensorMap
    statistics: Dict[str, Any]
