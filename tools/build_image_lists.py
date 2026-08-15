#!/usr/bin/env python3
"""Build deterministic absolute-path image lists under the approved data root."""

import argparse
import json
import os
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DATASETS = {
    "office31": ("amazon", "dslr", "webcam"),
    "visda-c": ("train", "validation"),
}
VISDA_CLASS_NAMES = (
    "aeroplane",
    "bicycle",
    "bus",
    "car",
    "horse",
    "knife",
    "motorcycle",
    "person",
    "plant",
    "skateboard",
    "train",
    "truck",
)


def _class_root(domain_root):
    images_root = domain_root / "images"
    return images_root if images_root.is_dir() else domain_root


def discover_domain(domain_root):
    root = _class_root(Path(domain_root))
    if not root.is_dir():
        raise FileNotFoundError(f"Domain directory not found: {root}")
    class_directories = sorted(
        path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")
    )
    if not class_directories:
        raise ValueError(f"No class directories found under {root}")
    by_class = {}
    for class_directory in class_directories:
        images = sorted(
            path.resolve()
            for path in class_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise ValueError(f"Class directory has no images: {class_directory}")
        by_class[class_directory.name] = images
    return by_class


def canonical_class_order(dataset, class_names):
    class_names = list(class_names)
    if dataset == "visda-c" and set(class_names) == set(VISDA_CLASS_NAMES):
        return list(VISDA_CLASS_NAMES)
    if class_names and all(name.isdigit() for name in class_names):
        numeric = sorted(class_names, key=int)
        if [int(name) for name in numeric] != list(range(len(numeric))):
            raise ValueError(
                f"Numeric class directories must be contiguous from 0: {numeric}"
            )
        return numeric
    return sorted(class_names, key=lambda name: name.casefold())


def build_lists(dataset, dataset_root, output_root):
    domains = DATASETS[dataset]
    discovered = {
        domain: discover_domain(Path(dataset_root) / domain)
        for domain in domains
    }
    reference_classes = canonical_class_order(
        dataset, discovered[domains[0]]
    )
    for domain in domains[1:]:
        domain_classes = canonical_class_order(dataset, discovered[domain])
        if set(domain_classes) != set(reference_classes):
            missing = sorted(set(reference_classes) - set(domain_classes))
            extra = sorted(set(domain_classes) - set(reference_classes))
            raise ValueError(
                f"Class directories differ for {domain}: missing={missing}, extra={extra}"
            )
    class_to_idx = {
        class_name: index for index, class_name in enumerate(reference_classes)
    }
    expected = 31 if dataset == "office31" else 12
    if len(class_to_idx) != expected:
        raise ValueError(
            f"{dataset} requires {expected} classes, found {len(class_to_idx)}"
        )
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for domain in domains:
        output_path = output_root / f"{domain}_list.txt"
        temporary_path = output_root / f".{domain}_list.txt.tmp"
        with temporary_path.open("w", encoding="utf-8", newline="\n") as file_obj:
            for class_name in reference_classes:
                label = class_to_idx[class_name]
                for image_path in discovered[domain][class_name]:
                    file_obj.write(f"{image_path} {label}\n")
        os.replace(temporary_path, output_path)
        outputs[domain] = {
            "path": str(output_path.resolve()),
            "image_count": sum(len(values) for values in discovered[domain].values()),
        }
    mapping_path = output_root / "class_to_idx.json"
    with mapping_path.open("w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "dataset": dataset,
                "class_to_idx": class_to_idx,
                "domains": outputs,
            },
            file_obj,
            indent=2,
            ensure_ascii=False,
        )
    return {"mapping": str(mapping_path.resolve()), "domains": outputs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Directory containing the dataset's domain directories.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory for *_list.txt and class_to_idx.json.",
    )
    args = parser.parse_args()
    result = build_lists(args.dataset, args.dataset_root, args.output_root)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
