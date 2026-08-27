"""Deterministic one-pass target stream and independent FO loader."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def read_records(path: str, num_classes: int) -> list[tuple[str, int]]:
    records: list[tuple[str, int]] = []
    with open(path, "r", encoding="utf-8") as file_obj:
        for line_number, raw_line in enumerate(file_obj, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                image_path, label_text = line.rsplit(maxsplit=1)
                label = int(label_text)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid record at {path}:{line_number}") from error
            if not Path(image_path).is_absolute():
                raise ValueError(f"Image path must be absolute at {path}:{line_number}")
            if not 0 <= label < num_classes:
                raise ValueError(
                    f"Label {label} outside [0, {num_classes}) at {path}:{line_number}"
                )
            records.append((image_path, label))
    if not records:
        raise ValueError(f"Target list is empty: {path}")
    present = {label for _, label in records}
    missing = sorted(set(range(num_classes)) - present)
    if missing:
        raise ValueError(f"Target list is missing fixed classes: {missing}")
    return records


class TargetImageList(Dataset):
    def __init__(self, records: list[tuple[str, int]], transform) -> None:
        self.records = tuple(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        path, label = self.records[index]
        with open(path, "rb") as file_obj:
            with Image.open(file_obj) as image:
                image = image.convert("RGB")
        return self.transform(image), label, index


class FixedOrderSampler(Sampler[int]):
    def __init__(self, indices) -> None:
        self.indices = tuple(int(index) for index in indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class MergeSingletonTailBatchSampler(BatchSampler):
    """Keep every sample while folding a size-one tail into the prior batch."""

    def __iter__(self):
        batches = list(super().__iter__())
        if len(batches) >= 2 and len(batches[-1]) == 1:
            batches[-2].extend(batches.pop())
        yield from batches

    def __len__(self) -> int:
        batch_count = super().__len__()
        has_singleton_tail = len(self.sampler) % self.batch_size == 1
        return batch_count - 1 if batch_count >= 2 and has_singleton_tail else batch_count


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _order_sha256(indices) -> str:
    payload = np.asarray(indices, dtype=np.int64).tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def fixed_random_order(sample_count: int, seed: int) -> list[int]:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.randperm(sample_count, generator=generator).tolist()


def build_transforms(config: dict):
    interpolation = config.get("interpolation")
    if interpolation != "bilinear":
        raise ValueError("Only the frozen FC-aligned bilinear interpolation is supported")
    resize_size = int(config["resize_size"])
    crop_size = int(config["crop_size"])
    normalize = transforms.Normalize(config["mean"], config["std"])
    shared = [
        transforms.Resize(
            (resize_size, resize_size), interpolation=InterpolationMode.BILINEAR
        )
    ]
    online = transforms.Compose(
        [
            *shared,
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    final = transforms.Compose(
        [
            *shared,
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return online, final


def _loader(
    dataset: Dataset,
    *,
    order: list[int],
    batch_size: int,
    workers: int,
    worker_seed: int,
    pin_memory: bool,
    merge_singleton_tail: bool = False,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(worker_seed))
    sampler = FixedOrderSampler(order)
    common = dict(
        dataset=dataset,
        num_workers=int(workers),
        pin_memory=bool(pin_memory),
        persistent_workers=False,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    if merge_singleton_tail:
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


def build_target_loaders(config: dict):
    records = read_records(config["target_list"], config["num_classes"])
    online_transform, fo_transform = build_transforms(config["preprocessing"])
    online_dataset = TargetImageList(records, online_transform)
    fo_dataset = TargetImageList(records, fo_transform)
    online_order = fixed_random_order(len(records), config["formal_seed"])
    fo_order = list(range(len(records)))
    pin_memory = bool(
        config["runtime"]["pin_memory"]
        and config["runtime"]["device"] == "cuda"
    )
    merge_singleton_tail = (
        config["dataset"] == "office31"
        and config["target"].lower() == "amazon"
        and len(records) % int(config["batch_size"]) == 1
    )
    online_loader = _loader(
        online_dataset,
        order=online_order,
        batch_size=config["batch_size"],
        workers=config["workers"],
        worker_seed=config["formal_seed"] + 1,
        pin_memory=pin_memory,
        merge_singleton_tail=merge_singleton_tail,
    )
    fo_loader = _loader(
        fo_dataset,
        order=fo_order,
        batch_size=config["fo_batch_size"],
        workers=config["workers"],
        worker_seed=config["formal_seed"] + 2,
        pin_memory=pin_memory,
    )
    stream_record = {
        "sampler": "fixed_random_permutation",
        "seed": config["formal_seed"],
        "sample_count": len(records),
        "batch_count": len(online_loader),
        "online_batch_size": int(config["batch_size"]),
        "fo_batch_count": len(fo_loader),
        "fo_batch_size": int(config["fo_batch_size"]),
        "drop_last": False,
        "singleton_tail_merged_into_previous_batch": merge_singleton_tail,
        "online_order_sha256": _order_sha256(online_order),
        "fo_sampler": "sequential",
        "fo_order_sha256": _order_sha256(fo_order),
        "online_worker_seed_base": config["formal_seed"] + 1,
        "fo_worker_seed_base": config["formal_seed"] + 2,
    }
    return online_loader, fo_loader, stream_record
