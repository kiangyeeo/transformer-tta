"""SHOT image-list datasets, transforms, and deterministic loader order."""

import hashlib
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms


def make_dataset(image_list, labels=None):
    if labels is not None:
        return [
            (image_list[index].strip(), labels[index, :])
            for index in range(len(image_list))
        ]
    if len(image_list[0].split()) > 2:
        return [
            (
                value.split()[0],
                np.array([int(label) for label in value.split()[1:]]),
            )
            for value in image_list
        ]
    return [
        (value.split()[0], int(value.split()[1])) for value in image_list
    ]


def rgb_loader(path):
    with open(path, "rb") as file_obj:
        with Image.open(file_obj) as image:
            return image.convert("RGB")


class ImageListWithIndex(Dataset):
    def __init__(self, image_list, labels=None, transform=None):
        self.images = make_dataset(image_list, labels)
        if not self.images:
            raise RuntimeError("Found 0 images in the supplied image list")
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.images[index]
        image = rgb_loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target, index

    def __len__(self):
        return len(self.images)


class FixedOrderSampler(Sampler):
    def __init__(self, order):
        self.order = tuple(int(index) for index in order)

    def __iter__(self):
        return iter(self.order)

    def __len__(self):
        return len(self.order)


def image_train(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )


def image_test(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def _seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _read_lines(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        return file_obj.readlines()


def _apply_pda_mapping(lines, source_classes, target_classes):
    label_map = {
        original_label: mapped_label
        for mapped_label, original_label in enumerate(source_classes)
    }
    remapped = []
    for record in lines:
        path, label_text = record.strip().split(" ")[:2]
        label = int(label_text)
        if label not in target_classes:
            continue
        mapped = label_map[label] if label in label_map else len(label_map)
        remapped.append(f"{path} {mapped}\n")
    return remapped


def _order_digest(order):
    payload = ",".join(str(index) for index in order).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_loaders(config):
    data_config = config["data"]
    target_lines = _read_lines(data_config["target_list"])
    test_lines = _read_lines(data_config["test_list"])

    if data_config["da"] != "uda":
        target_lines = _apply_pda_mapping(
            target_lines,
            data_config["source_classes"],
            data_config["target_classes"],
        )
        test_lines = list(target_lines)

    target_dataset = ImageListWithIndex(
        target_lines, transform=image_train()
    )
    test_dataset = ImageListWithIndex(test_lines, transform=image_test())

    seed = int(config["seed"])
    if data_config["dataset"] == "VISDA-C":
        from visda_otta.adapter import build_target_order

        target_order, target_order_record = build_target_order(
            seed, len(target_dataset)
        )
    else:
        sampler_generator = torch.Generator()
        sampler_generator.manual_seed(seed)
        target_order = torch.randperm(
            len(target_dataset), generator=sampler_generator
        ).tolist()
        target_order_record = {
            "sampler": "fixed_random_permutation",
            "seed": seed,
        }
    test_order = list(range(len(test_dataset)))

    target_worker_generator = torch.Generator()
    target_worker_generator.manual_seed(seed + 1)
    test_worker_generator = torch.Generator()
    test_worker_generator.manual_seed(seed + 2)

    target_loader = DataLoader(
        target_dataset,
        batch_size=data_config["batch_size"],
        sampler=FixedOrderSampler(target_order),
        num_workers=data_config["workers"],
        drop_last=False,
        worker_init_fn=_seed_worker,
        generator=target_worker_generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config["batch_size"] * 3,
        sampler=FixedOrderSampler(test_order),
        num_workers=data_config["workers"],
        drop_last=False,
        worker_init_fn=_seed_worker,
        generator=test_worker_generator,
    )

    order_record = {
        "target": {
            **target_order_record,
            "indices": target_order,
            "sha256": _order_digest(target_order),
        },
        "test": {
            "sampler": "sequential",
            "indices": test_order,
            "sha256": _order_digest(test_order),
        },
        "target_worker_seed_base": seed + 1,
        "test_worker_seed_base": seed + 2,
    }
    return {"target": target_loader, "test": test_loader}, order_record
