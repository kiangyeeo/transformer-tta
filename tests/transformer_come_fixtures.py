"""CPU fixtures for the COME-Transformer runner contracts.

The synthetic model carries the *real* candidate tensors
(``blocks.{9,10,11}.{attn.qkv,attn.proj,mlp.fc1,mlp.fc2}.weight`` with their
exact shapes), so the tests exercise the real 6912-group partition, the real
exact-K masks and the real strict masked AdamW step instead of a stand-in.
Blocks 0-8 exist but hold no parameters, which keeps the fixture at the 12
candidate tensors (5,308,416 scalars) plus a small frozen head/LayerNorm/token
set.
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler

from transformer.candidate_dense.config import candidate_parameter_names
from transformer.candidate_dense.model import (
    EXPECTED_SHAPES,
    configure_candidate_dense_scope,
)
from transformer_come.config import (
    CANDIDATE_ADAPTATION,
    FULL_DENSE_ADAPTATION,
)
from transformer_come.config import (
    EXPECTED_COME_BLOCK,
    IMPLEMENTATION_REVISIONS,
    PROTOCOL_REVISIONS,
    come_objective_payload,
)


HIDDEN_DIM = 384
NUM_CLASSES = 31
CLASS_NAMES = [f"class-{index}" for index in range(NUM_CLASSES)]
SAMPLE_COUNT = 32
BATCH_SIZE = 8


class _Attn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(HIDDEN_DIM, 3 * HIDDEN_DIM, bias=False)
        self.proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)


class _Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(HIDDEN_DIM, 4 * HIDDEN_DIM, bias=False)
        self.fc2 = nn.Linear(4 * HIDDEN_DIM, HIDDEN_DIM, bias=False)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _Attn()
        self.mlp = _Mlp()


class CandidateShapedModel(nn.Module):
    """A cheap model with the exact DeiT-S candidate tensors and a Linear head."""

    forward_calls = 0

    def __init__(self, seed: int = 2026) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.blocks = nn.ModuleList(
            [nn.Module() for _ in range(9)] + [_Block() for _ in range(3)]
        )
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.head = nn.Linear(HIDDEN_DIM, NUM_CLASSES)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, HIDDEN_DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, 2, HIDDEN_DIM))
        self.register_buffer("read_only_buffer", torch.ones(1))
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(
                    torch.empty(
                        parameter.shape, dtype=parameter.dtype
                    ).uniform_(-0.05, 0.05, generator=generator)
                )

    def get_classifier(self) -> nn.Linear:
        return self.head

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        type(self).forward_calls += 1
        x = images + self.cls_token.reshape(-1) + self.pos_embed[0, 0]
        for block in self.blocks[9:]:
            qkv = block.attn.qkv(x)
            query, key, value = qkv.split(HIDDEN_DIM, dim=1)
            attended = value * torch.tanh((query * key).sum(dim=1, keepdim=True) / 384.0)
            x = x + torch.tanh(block.attn.proj(attended))
            x = x + torch.tanh(block.mlp.fc2(torch.tanh(block.mlp.fc1(x))))
        return self.head(self.norm(x))


def candidate_tensors():
    """Detached candidate-shaped parameters, for group/mask level checks."""

    return [
        (
            name,
            nn.Parameter(torch.zeros(EXPECTED_SHAPES[name.split(".", maxsplit=2)[2]])),
        )
        for name in candidate_parameter_names()
    ]


def load_candidate_scope_model(config, device):
    """Model loader with the real candidate scope (sparse + candidate dense)."""

    del config
    model = CandidateShapedModel().to(device)
    candidates, frozen, scope_record = configure_candidate_dense_scope(model)
    return model, {"path": "synthetic", "sha256": "0" * 64}, candidates, frozen, {
        **scope_record,
        "candidate_scope": "last_three_blocks_qkv_proj_mlp_weights",
        # Regression guard: a scope helper must never shadow the authoritative
        # structured selection artifact with a descriptive string.
        "selection": "synthetic-scope-selection",
        "host_objective": "come",
    }


def load_full_dense_scope_model(config, device):
    """Model loader with the full-dense scope: everything except the head."""

    del config
    model = CandidateShapedModel().to(device)
    head_parameter_ids = {id(parameter) for parameter in model.head.parameters()}
    model.requires_grad_(True)
    model.head.requires_grad_(False)
    model.eval()
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if id(parameter) not in head_parameter_ids
    ]
    frozen = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if id(parameter) in head_parameter_ids
    ]
    return model, {"path": "synthetic", "sha256": "0" * 64}, trainable, frozen, {
        "update_scope": "all_except_head",
        "trainable_parameter_names": [name for name, _ in trainable],
        "frozen_head_parameter_names": [name for name, _ in frozen],
        "trainable_scalars": sum(parameter.numel() for _, parameter in trainable),
        "host_objective": "come",
    }


class OrderedSampler(Sampler[int]):
    def __init__(self, order) -> None:
        self.order = list(order)

    def __iter__(self):
        return iter(self.order)

    def __len__(self) -> int:
        return len(self.order)


class SyntheticTargetDataset(Dataset):
    """Every fixed class appears at least once so PU/FO metrics are complete."""

    def __init__(self, sample_count: int = SAMPLE_COUNT, seed: int = 7) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.randn(
            sample_count, HIDDEN_DIM, generator=generator, dtype=torch.float32
        )
        labels = [index % NUM_CLASSES for index in range(sample_count)]
        self.labels = torch.tensor(labels, dtype=torch.int64)

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index], index


def build_synthetic_loaders(
    config,
    *,
    sample_count: int = SAMPLE_COUNT,
    batch_size: int = BATCH_SIZE,
    online_order=None,
):
    del config
    dataset = SyntheticTargetDataset(sample_count)
    order = list(online_order) if online_order is not None else list(
        reversed(range(sample_count))
    )
    online = DataLoader(
        dataset, batch_size=batch_size, sampler=OrderedSampler(order), drop_last=False
    )
    final = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=OrderedSampler(range(sample_count)),
        drop_last=False,
    )
    return online, final, {
        "sampler": "fixed_random_permutation",
        "seed": 2026,
        "sample_count": sample_count,
        "batch_count": len(online),
        "online_batch_size": batch_size,
        "fo_batch_count": len(final),
        "fo_batch_size": batch_size,
        "drop_last": False,
        "singleton_tail_merged_into_previous_batch": False,
        "online_order_sha256": "synthetic-online",
        "fo_sampler": "sequential",
        "fo_order_sha256": "synthetic-fo",
    }


def synthetic_config(variant: str, *, output_dir: Path, **overrides) -> dict:
    """A resolved-config stand-in with the real COME objective block."""

    come_block = copy.deepcopy(EXPECTED_COME_BLOCK)
    config = {
        "schema_version": 1,
        "protocol_revision": PROTOCOL_REVISIONS[variant],
        "implementation_revision": IMPLEMENTATION_REVISIONS[variant],
        "method": "come",
        "variant": variant,
        "formal_seed": 2026,
        "experiment_key": f"synthetic-come-{variant}",
        "scientific_config_sha256": "1" * 64,
        "dataset": "office31",
        "source": "dslr",
        "target": "amazon",
        "transfer": "dslr->amazon",
        "num_classes": NUM_CLASSES,
        "class_names": list(CLASS_NAMES),
        "batch_size": BATCH_SIZE,
        "fo_batch_size": BATCH_SIZE,
        "adaptation": copy.deepcopy(
            FULL_DENSE_ADAPTATION if variant == "full_dense" else CANDIDATE_ADAPTATION
        ),
        "optimization": {
            "optimizer": "adamw",
            "lr": 1.0e-3,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "weight_decay": 0.01,
        },
        "come": come_block,
        "come_objective": come_objective_payload(come_block),
        "preprocessing": {"interpolation": "bilinear"},
        "runtime": {
            "device": "cpu",
            "deterministic": True,
            "amp": False,
            "pin_memory": False,
        },
        "output_dir": str(output_dir),
    }
    config.update(overrides)
    return config


def sparse_selection(budget: float, group_count: int, *, per_step: bool = False) -> dict:
    key = "active_candidate_scalars_per_step" if per_step else "active_candidate_scalars"
    return {
        "requested_budget": budget,
        "requested_group_count": group_count,
        key: group_count * 768,
    }


def read_metrics(output_dir: Path) -> list[dict]:
    import json

    return [
        json.loads(line)
        for line in (Path(output_dir) / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


__all__ = [
    "BATCH_SIZE",
    "CLASS_NAMES",
    "CandidateShapedModel",
    "NUM_CLASSES",
    "OrderedSampler",
    "SAMPLE_COUNT",
    "SyntheticTargetDataset",
    "build_synthetic_loaders",
    "candidate_tensors",
    "load_candidate_scope_model",
    "load_full_dense_scope_model",
    "read_metrics",
    "sparse_selection",
    "synthetic_config",
]
