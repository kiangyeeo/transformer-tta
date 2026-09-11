"""Frozen raw-image IST stream, deterministic views, and independent FO."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf

from transformer.source_only.data import (
    FixedOrderSampler,
    MergeSingletonTailBatchSampler,
    TargetImageList,
    _seed_worker,
    build_transforms,
    fixed_random_order,
    read_records,
)


class RawTargetImageList(Dataset):
    """Return paths without decoding; labels remain runner metric metadata."""

    def __init__(self, records) -> None:
        self.records = tuple(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        path, label = self.records[index]
        return path, label, index


def _raw_collate(batch):
    paths, labels, indices = zip(*batch)
    return list(paths), torch.as_tensor(labels), torch.as_tensor(indices)


def _order_sha256(indices) -> str:
    return hashlib.sha256(
        np.asarray(indices, dtype=np.int64).tobytes(order="C")
    ).hexdigest()


def _make_loader(
    dataset,
    order,
    batch_size,
    workers,
    worker_seed,
    pin_memory,
    *,
    raw: bool,
    merge_singleton: bool,
):
    generator = torch.Generator().manual_seed(int(worker_seed))
    sampler = FixedOrderSampler(order)
    common = {
        "dataset": dataset,
        "num_workers": int(workers),
        "pin_memory": bool(pin_memory),
        "persistent_workers": False,
        "worker_init_fn": _seed_worker,
        "generator": generator,
    }
    if raw:
        common["collate_fn"] = _raw_collate
    if merge_singleton:
        return DataLoader(
            **common,
            batch_sampler=MergeSingletonTailBatchSampler(
                sampler, batch_size=int(batch_size), drop_last=False
            ),
        )
    return DataLoader(
        **common,
        batch_size=int(batch_size),
        sampler=sampler,
        drop_last=False,
    )


def build_ist_loaders(config):
    records = read_records(config["target_list"], config["num_classes"])
    online_order = fixed_random_order(len(records), config["formal_seed"])
    fo_order = list(range(len(records)))
    merge_singleton = (
        config["dataset"] == "office31"
        and config["target"].lower() == "amazon"
        and len(records) % int(config["batch_size"]) == 1
    )
    _, fo_transform = build_transforms(config["preprocessing"])
    pin_memory = bool(
        config["runtime"]["pin_memory"]
        and config["runtime"]["device"] == "cuda"
    )
    online = _make_loader(
        RawTargetImageList(records),
        online_order,
        config["batch_size"],
        config["workers"],
        config["formal_seed"] + 1,
        pin_memory,
        raw=True,
        merge_singleton=merge_singleton,
    )
    final = _make_loader(
        TargetImageList(records, fo_transform),
        fo_order,
        config["fo_batch_size"],
        config["workers"],
        config["formal_seed"] + 2,
        pin_memory,
        raw=False,
        merge_singleton=False,
    )
    batch_sizes = [
        len(online_order[start : start + config["batch_size"]])
        for start in range(0, len(online_order), config["batch_size"])
    ]
    if merge_singleton:
        singleton_size = batch_sizes.pop()
        batch_sizes[-1] += singleton_size
    return online, final, {
        "sampler": "fixed_random_permutation",
        "seed": config["formal_seed"],
        "sample_count": len(records),
        "batch_count": len(batch_sizes),
        "batch_size_sequence": batch_sizes,
        "online_order": online_order,
        "online_order_sha256": _order_sha256(online_order),
        "fo_order_sha256": _order_sha256(fo_order),
        "drop_last": False,
        "singleton_tail_merged_into_previous_batch": merge_singleton,
    }


@dataclass(frozen=True)
class MaterializedBatch:
    reference_views: torch.Tensor
    adaptation_views: torch.Tensor
    sample_indices: torch.Tensor
    augmentation_trace_sha256: str
    reference_trace_sha256: str
    adaptation_trace_sha256: str


class ISTViewMaterializer:
    """Decode each path once and materialize 1+8 views with isolated RNGs."""

    def __init__(self, config) -> None:
        ist = config["ist"]
        augmentation = ist["augmentation"]
        rng = ist["rng"]
        seed = int(config["formal_seed"])
        self.extend = int(ist["extend"])
        self.resize_size = int(augmentation["resize_size"])
        self.crop_size = int(augmentation["crop_size"])
        self.flip_probability = float(
            augmentation["horizontal_flip_probability"]
        )
        self.mean = tuple(float(value) for value in augmentation["mean"])
        self.std = tuple(float(value) for value in augmentation["std"])
        self.reference_generator = torch.Generator(device="cpu").manual_seed(
            seed + int(rng["reference_seed_offset"])
        )
        self.adaptation_generator = torch.Generator(device="cpu").manual_seed(
            seed + int(rng["adaptation_seed_offset"])
        )

    @staticmethod
    def _decode(raw_image):
        if torch.is_tensor(raw_image):
            raise TypeError("IST requires raw paths/PIL; normalized tensors are rejected")
        if isinstance(raw_image, Image.Image):
            return raw_image.convert("RGB")
        if isinstance(raw_image, (str, Path)):
            with open(raw_image, "rb") as file_obj:
                with Image.open(file_obj) as image:
                    return image.convert("RGB")
        raise TypeError(f"unsupported raw image type: {type(raw_image)!r}")

    def _view(self, image, generator):
        resized = tvf.resize(
            image,
            [self.resize_size, self.resize_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        maximum = self.resize_size - self.crop_size
        top = int(torch.randint(maximum + 1, (1,), generator=generator).item())
        left = int(torch.randint(maximum + 1, (1,), generator=generator).item())
        flipped = bool(
            torch.rand((), generator=generator).item() < self.flip_probability
        )
        view = tvf.crop(resized, top, left, self.crop_size, self.crop_size)
        if flipped:
            view = tvf.hflip(view)
        tensor = tvf.normalize(tvf.to_tensor(view), self.mean, self.std)
        return tensor, {"top": top, "left": left, "flip": flipped}

    @staticmethod
    def _trace_hash(value) -> str:
        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def materialize(self, raw_images, sample_indices) -> MaterializedBatch:
        indices = torch.as_tensor(sample_indices, dtype=torch.long)
        if indices.ndim != 1 or len(raw_images) != int(indices.numel()):
            raise ValueError("raw image paths and sample indices must align exactly")
        if not raw_images:
            raise ValueError("IST view materialization requires a non-empty batch")
        references, adaptations = [], []
        reference_trace, adaptation_trace = [], []
        for raw_image, sample_index in zip(raw_images, indices.tolist()):
            image = self._decode(raw_image)
            reference, trace = self._view(image, self.reference_generator)
            references.append(reference)
            reference_trace.append({"sample_index": sample_index, **trace})
            for view_index in range(self.extend):
                adaptation, trace = self._view(image, self.adaptation_generator)
                adaptations.append(adaptation)
                adaptation_trace.append(
                    {
                        "sample_index": sample_index,
                        "view_index": view_index,
                        **trace,
                    }
                )
        reference_hash = self._trace_hash(reference_trace)
        adaptation_hash = self._trace_hash(adaptation_trace)
        return MaterializedBatch(
            reference_views=torch.stack(references),
            adaptation_views=torch.stack(adaptations),
            sample_indices=indices,
            augmentation_trace_sha256=self._trace_hash(
                {"reference": reference_trace, "adaptation": adaptation_trace}
            ),
            reference_trace_sha256=reference_hash,
            adaptation_trace_sha256=adaptation_hash,
        )
