"""Image-list data loading and deterministic source validation splitting."""

import hashlib
import random
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def read_image_records(path, num_classes):
    records = []
    with open(path, "r", encoding="utf-8") as file_obj:
        for line_number, raw_line in enumerate(file_obj, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                image_path, label_text = line.rsplit(maxsplit=1)
                label = int(label_text)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid image-list record at {path}:{line_number}: "
                    f"{line!r}"
                ) from error
            if not 0 <= label < num_classes:
                raise ValueError(
                    f"Label {label} at {path}:{line_number} is outside "
                    f"[0, {num_classes - 1}]"
                )
            records.append((image_path, label))
    if not records:
        raise ValueError(f"Image list contains no records: {path}")
    present_classes = {label for _, label in records}
    missing = sorted(set(range(num_classes)) - present_classes)
    if missing:
        raise ValueError(
            f"Source list {path} is missing labels: {missing}"
        )
    return records


def _indices_sha256(indices):
    payload = ",".join(str(index) for index in indices).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def records_sha256(records):
    payload = "".join(
        f"{path}\t{label}\n" for path, label in records
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stratified_fixed_split(records, validation_fraction, seed):
    """Split every class deterministically while retaining train examples.

    For classes with at least two examples, validation receives at least one
    example and training keeps at least one.  The per-class safeguard is more
    important than attaining exactly 10% on the very small DSLR/Webcam domains.
    """
    by_class = defaultdict(list)
    for index, (_, label) in enumerate(records):
        by_class[label].append(index)
    train_indices = []
    validation_indices = []
    rng = random.Random(int(seed))
    per_class = {}
    for label in sorted(by_class):
        indices = list(by_class[label])
        rng.shuffle(indices)
        if len(indices) < 2:
            raise ValueError(
                f"Class {label} has only {len(indices)} source example; "
                "stratified source validation requires at least 2"
            )
        validation_count = max(
            1, int(round(len(indices) * float(validation_fraction)))
        )
        validation_count = min(validation_count, len(indices) - 1)
        validation_indices.extend(indices[:validation_count])
        train_indices.extend(indices[validation_count:])
        per_class[str(label)] = {
            "total": len(indices),
            "train": len(indices) - validation_count,
            "validation": validation_count,
        }
    train_indices.sort()
    validation_indices.sort()
    if set(train_indices) & set(validation_indices):
        raise RuntimeError("Train and validation source splits overlap")
    if sorted(train_indices + validation_indices) != list(range(len(records))):
        raise RuntimeError("Train and validation source splits are incomplete")
    manifest = {
        "strategy": "stratified_fixed",
        "seed": int(seed),
        "validation_fraction_requested": float(validation_fraction),
        "record_count": len(records),
        "records_sha256": records_sha256(records),
        "train_count": len(train_indices),
        "validation_count": len(validation_indices),
        "validation_fraction_actual": len(validation_indices) / len(records),
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "train_indices_sha256": _indices_sha256(train_indices),
        "validation_indices_sha256": _indices_sha256(validation_indices),
        "per_class": per_class,
    }
    return train_indices, validation_indices, manifest


def build_transforms(config):
    resize_size = config["resize_size"]
    crop_size = config["crop_size"]
    mean = config["mean"]
    std = config["std"]
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (resize_size, resize_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(
                p=config["horizontal_flip_probability"]
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize(
                (resize_size, resize_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, validation_transform


class SourceImageList(Dataset):
    def __init__(self, records, indices, transform):
        self.records = records
        self.indices = tuple(int(index) for index in indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        record_index = self.indices[item]
        path, label = self.records[record_index]
        with open(path, "rb") as file_obj:
            with Image.open(file_obj) as image:
                image = image.convert("RGB")
        return self.transform(image), label, record_index


def _seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(
    dataset,
    *,
    batch_size,
    workers,
    shuffle,
    seed,
    pin_memory,
):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        drop_last=False,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=False,
    )
