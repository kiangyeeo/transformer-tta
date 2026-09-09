"""Raw-image outer stream and deterministic IST multi-view materialization."""

from dataclasses import dataclass

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf

from shot_otta.data import (
    FixedOrderSampler,
    ImageListWithIndex,
    _apply_pda_mapping,
    _read_lines,
    _seed_worker,
    build_order_record,
    image_test,
    make_dataset,
    resolve_target_order,
    rgb_loader,
)


class RawImageListWithIndex(Dataset):
    """Return image paths, never transformed or normalized tensors."""

    def __init__(self, image_list, labels=None):
        self.images = make_dataset(image_list, labels)
        if not self.images:
            raise RuntimeError("Found 0 images in the supplied image list")

    def __getitem__(self, index):
        path, target = self.images[index]
        return path, target, index

    def __len__(self):
        return len(self.images)


def _raw_collate(batch):
    paths, labels, indices = zip(*batch)
    return list(paths), torch.as_tensor(labels), torch.as_tensor(indices)


def build_ist_loaders(config):
    """Use SHOT's exact outer order/boundaries but keep target images raw."""
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

    target_dataset = RawImageListWithIndex(target_lines)
    test_dataset = ImageListWithIndex(test_lines, transform=image_test())
    seed = int(config["seed"])
    target_order, target_order_record = resolve_target_order(
        config, len(target_dataset)
    )
    test_order = list(range(len(test_dataset)))

    target_generator = torch.Generator().manual_seed(seed + 1)
    test_generator = torch.Generator().manual_seed(seed + 2)
    target_loader = DataLoader(
        target_dataset,
        batch_size=data_config["batch_size"],
        sampler=FixedOrderSampler(target_order),
        num_workers=data_config["workers"],
        drop_last=False,
        worker_init_fn=_seed_worker,
        generator=target_generator,
        collate_fn=_raw_collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config["batch_size"] * 3,
        sampler=FixedOrderSampler(test_order),
        num_workers=data_config["workers"],
        drop_last=False,
        worker_init_fn=_seed_worker,
        generator=test_generator,
    )
    order_record = build_order_record(
        target_order, target_order_record, test_order, seed
    )
    order_record["ist_view_rng"] = dict(config["ist"]["rng"])
    order_record["ist_view_rng"]["formal_seed"] = seed
    return {"target": target_loader, "test": test_loader}, order_record


@dataclass(frozen=True)
class MaterializedBatch:
    reference_views: torch.Tensor
    adaptation_views: torch.Tensor
    labels: torch.Tensor
    sample_indices: torch.Tensor


class ISTViewMaterializer:
    """Create cached PU and independent IST views directly from raw images."""

    def __init__(self, config):
        ist = config["ist"]
        augmentation = ist["augmentation"]
        rng = ist["rng"]
        seed = int(config["seed"])
        self.extend = int(ist["extend"])
        self.resize_size = int(augmentation["resize_size"])
        self.crop_size = int(augmentation["crop_size"])
        self.flip_probability = float(
            augmentation["horizontal_flip_probability"]
        )
        self.mean = tuple(float(value) for value in augmentation["mean"])
        self.std = tuple(float(value) for value in augmentation["std"])
        self.reference_generator = torch.Generator(device="cpu")
        self.reference_generator.manual_seed(
            seed + int(rng["reference_seed_offset"])
        )
        self.adaptation_generator = torch.Generator(device="cpu")
        self.adaptation_generator.manual_seed(
            seed + int(rng["adaptation_seed_offset"])
        )

    def state_dict(self):
        """Return the two independent view-stream RNG states."""
        return {
            "reference_generator": self.reference_generator.get_state(),
            "adaptation_generator": self.adaptation_generator.get_state(),
        }

    def load_state_dict(self, state):
        self.reference_generator.set_state(state["reference_generator"])
        self.adaptation_generator.set_state(state["adaptation_generator"])

    @staticmethod
    def _load_raw_image(raw_image):
        if torch.is_tensor(raw_image):
            raise TypeError(
                "IST views require raw PIL/path inputs; normalized tensors "
                "must never be converted back to PIL"
            )
        if isinstance(raw_image, str):
            return rgb_loader(raw_image)
        if isinstance(raw_image, Image.Image):
            return raw_image.convert("RGB")
        raise TypeError(f"Unsupported raw image type: {type(raw_image)!r}")

    def _view(self, image, generator):
        image = tvf.resize(
            image,
            [self.resize_size, self.resize_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        maximum = self.resize_size - self.crop_size
        top = int(
            torch.randint(maximum + 1, (1,), generator=generator).item()
        )
        left = int(
            torch.randint(maximum + 1, (1,), generator=generator).item()
        )
        image = tvf.crop(
            image, top, left, self.crop_size, self.crop_size
        )
        if float(torch.rand((), generator=generator).item()) < self.flip_probability:
            image = tvf.hflip(image)
        tensor = tvf.to_tensor(image)
        return tvf.normalize(tensor, self.mean, self.std)

    def materialize(self, raw_images, labels, sample_indices):
        references = []
        adaptation = []
        for raw_image in raw_images:
            # Load each raw sample once. Reference/adaptation views still use
            # independent RNG streams, so this is an I/O optimization only.
            image = self._load_raw_image(raw_image)
            references.append(self._view(image, self.reference_generator))
            adaptation.extend(
                self._view(image, self.adaptation_generator)
                for _ in range(self.extend)
            )
        labels = torch.as_tensor(labels, dtype=torch.long)
        return MaterializedBatch(
            reference_views=torch.stack(references),
            adaptation_views=torch.stack(adaptation),
            labels=labels,
            sample_indices=torch.as_tensor(sample_indices, dtype=torch.long),
        )
