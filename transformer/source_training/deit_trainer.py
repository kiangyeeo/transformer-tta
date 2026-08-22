"""Deterministic full-model source-domain fine-tuning for DeiT-S."""

import importlib.metadata
import json
import math
import os
import os.path as osp
import platform
import random
import subprocess
import sys
from contextlib import nullcontext
from datetime import datetime, timezone

# CuBLAS reads this before creating a CUDA handle. Set it before importing
# torch so deterministic=True is effective even outside the provided launchers.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from .deit_config import dump_source_config
from .deit_data import (
    SourceImageList,
    build_loader,
    build_transforms,
    read_image_records,
    stratified_fixed_split,
)
from .deit_model import (
    SOURCE_CHECKPOINT_KIND,
    SOURCE_CHECKPOINT_SCHEMA_VERSION,
    build_deit_source_model,
    sha256_file,
)


NO_DECAY_TOKEN_NAMES = {"cls_token", "pos_embed"}


def _unwrap_model(model):
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _json_dump(path, payload):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)


def _jsonl_append(path, payload):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _cpu_state_dict(model):
    return {
        name: tensor.detach().cpu()
        for name, tensor in _unwrap_model(model).state_dict().items()
    }


def _atomic_torch_save(path, payload):
    os.makedirs(osp.dirname(path), exist_ok=True)
    temp_path = osp.join(
        osp.dirname(path),
        f".{osp.basename(path)}.{os.getpid()}.tmp",
    )
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, path)
    finally:
        if osp.exists(temp_path):
            os.remove(temp_path)


def _git_info(project_root):
    def run_git(*arguments):
        completed = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    dirty = run_git("status", "--porcelain")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if dirty is None else bool(dirty),
    }


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _load_and_validate_class_mapping(path, data_config):
    with open(path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if payload.get("dataset") != data_config["dataset"]:
        raise ValueError(
            "class_to_idx.json dataset does not match source config: "
            f"{payload.get('dataset')!r} != {data_config['dataset']!r}"
        )
    mapping = payload.get("class_to_idx")
    if not isinstance(mapping, dict):
        raise ValueError("class_to_idx.json must contain class_to_idx mapping")
    labels = sorted(int(value) for value in mapping.values())
    expected = list(range(data_config["num_classes"]))
    if labels != expected:
        raise ValueError(
            f"class_to_idx labels must be exactly {expected}, got {labels}"
        )
    domain_record = payload.get("domains", {}).get(
        data_config["source_domain"]
    )
    if not isinstance(domain_record, dict):
        raise ValueError(
            "class_to_idx.json has no record for source domain "
            f"{data_config['source_domain']}"
        )
    mapped_list = osp.normpath(osp.abspath(domain_record["path"]))
    configured_list = osp.normpath(osp.abspath(data_config["source_list"]))
    if mapped_list != configured_list:
        raise ValueError(
            "class_to_idx source list does not match configured source list: "
            f"{mapped_list} != {configured_list}"
        )
    return payload


def _load_and_validate_pretrained_config(path, model_config, preprocessing):
    with open(path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    expected_architecture = model_config["name"].split(".", 1)[0]
    checks = {
        "architecture": (payload.get("architecture"), expected_architecture),
        "num_classes": (
            int(payload.get("num_classes", -1)),
            int(model_config["pretrained_num_classes"]),
        ),
        "num_features": (
            int(payload.get("num_features", -1)),
            int(model_config["hidden_dim"]),
        ),
    }
    for field, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(
                f"Local pretrained config {field} mismatch: "
                f"expected {expected!r}, got {actual!r}"
            )
    pretrained_cfg = payload.get("pretrained_cfg", {})
    if pretrained_cfg.get("input_size") != [
        3,
        preprocessing["crop_size"],
        preprocessing["crop_size"],
    ]:
        raise ValueError("Local pretrained input_size does not match crop_size")
    if pretrained_cfg.get("interpolation") != preprocessing["interpolation"]:
        raise ValueError("Local pretrained interpolation does not match protocol")
    for field in ("mean", "std"):
        if [float(value) for value in pretrained_cfg.get(field, [])] != [
            float(value) for value in preprocessing[field]
        ]:
            raise ValueError(
                f"Local pretrained {field} does not match preprocessing"
            )
    if pretrained_cfg.get("classifier") != "head":
        raise ValueError("Local pretrained classifier must be head")
    return payload


def set_reproducibility(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = False
    if deterministic:
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        torch.use_deterministic_algorithms(True, warn_only=False)


def _is_no_decay_parameter(name, parameter):
    leaf_name = name.rsplit(".", 1)[-1]
    return (
        parameter.ndim <= 1
        or leaf_name in NO_DECAY_TOKEN_NAMES
        or name.endswith(".bias")
    )


def build_adamw(model, config):
    """Create explicit backbone/head × decay/no-decay parameter groups."""
    classifier = model.get_classifier()
    head_ids = {id(parameter) for parameter in classifier.parameters()}
    buckets = {
        ("backbone", "decay"): [],
        ("backbone", "no_decay"): [],
        ("head", "decay"): [],
        ("head", "no_decay"): [],
    }
    names = {key: [] for key in buckets}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            raise ValueError(
                f"Full-model source fine-tuning found frozen parameter: {name}"
            )
        scope = "head" if id(parameter) in head_ids else "backbone"
        decay = "no_decay" if _is_no_decay_parameter(name, parameter) else "decay"
        buckets[(scope, decay)].append(parameter)
        names[(scope, decay)].append(name)
    seen = [id(parameter) for values in buckets.values() for parameter in values]
    if len(seen) != len(set(seen)):
        raise RuntimeError("Optimizer parameter groups overlap")
    if len(seen) != len(list(model.parameters())):
        raise RuntimeError("Optimizer parameter groups do not cover the model")

    groups = []
    group_manifest = []
    weight_decay = float(config["weight_decay"])
    for (scope, decay), parameters in buckets.items():
        if not parameters:
            continue
        base_lr = float(
            config["head_lr"] if scope == "head" else config["backbone_lr"]
        )
        group_weight_decay = weight_decay if decay == "decay" else 0.0
        groups.append(
            {
                "params": parameters,
                "lr": base_lr,
                "initial_lr": base_lr,
                "weight_decay": group_weight_decay,
                "group_name": f"{scope}_{decay}",
            }
        )
        group_manifest.append(
            {
                "name": f"{scope}_{decay}",
                "learning_rate": base_lr,
                "weight_decay": group_weight_decay,
                "parameter_names": names[(scope, decay)],
                "parameter_count": sum(p.numel() for p in parameters),
            }
        )
    optimizer = optim.AdamW(
        groups,
        betas=tuple(config["betas"]),
        eps=float(config["eps"]),
    )
    return optimizer, group_manifest


class WarmupCosineSchedule:
    def __init__(self, optimizer, total_steps, warmup_steps, min_lr):
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr = float(min_lr)
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")

    def apply(self, step):
        step = int(step)
        values = []
        for group in self.optimizer.param_groups:
            base_lr = float(group["initial_lr"])
            group_min_lr = min(self.min_lr, base_lr)
            if self.warmup_steps > 0 and step < self.warmup_steps:
                learning_rate = base_lr * (step + 1) / self.warmup_steps
            else:
                denominator = max(1, self.total_steps - self.warmup_steps - 1)
                progress = min(
                    1.0,
                    max(0.0, (step - self.warmup_steps) / denominator),
                )
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                learning_rate = group_min_lr + (
                    base_lr - group_min_lr
                ) * cosine
            group["lr"] = learning_rate
            values.append(learning_rate)
        return values


def _autocast_context(device, enabled):
    if not enabled:
        return nullcontext()
    try:
        return torch.amp.autocast(device_type=device.type, enabled=True)
    except AttributeError:
        return torch.cuda.amp.autocast(enabled=True)


def _grad_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    schedule,
    *,
    device,
    amp,
    scaler,
    global_step_start,
    progress_desc=None,
):
    model.train()
    total_loss = 0.0
    correct = 0
    count = 0
    learning_rates = None
    progress = tqdm(
        loader,
        desc=progress_desc,
        leave=False,
        dynamic_ncols=True,
        disable=progress_desc is None,
    )
    for batch_index, batch in enumerate(progress):
        images, labels = batch[0], batch[1]
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        learning_rates = schedule.apply(global_step_start + batch_index)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, amp):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = labels.size(0)
        total_loss += float(loss.detach().item()) * batch_size
        correct += int((logits.detach().argmax(dim=1) == labels).sum().item())
        count += batch_size
        progress.set_postfix(
            loss=f"{total_loss / count:.4f}",
            acc=f"{100.0 * correct / count:.2f}%",
            lr=f"{learning_rates[0]:.2e}",
        )
    if count == 0:
        raise RuntimeError("Source training loader produced no examples")
    return {
        "loss": total_loss / count,
        "overall_accuracy": 100.0 * correct / count,
        "example_count": count,
        "last_learning_rates": learning_rates,
    }


def evaluate(
    model,
    loader,
    criterion,
    *,
    device,
    amp,
    num_classes,
    progress_desc=None,
):
    model.eval()
    total_loss = 0.0
    count = 0
    correct_by_class = torch.zeros(num_classes, dtype=torch.long)
    count_by_class = torch.zeros(num_classes, dtype=torch.long)
    with torch.no_grad():
        progress = tqdm(
            loader,
            desc=progress_desc,
            leave=False,
            dynamic_ncols=True,
            disable=progress_desc is None,
        )
        for batch in progress:
            images, labels = batch[0], batch[1]
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with _autocast_context(device, amp):
                logits = model(images)
                loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)
            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            count += batch_size
            for class_index in range(num_classes):
                mask = labels == class_index
                class_count = int(mask.sum().item())
                count_by_class[class_index] += class_count
                if class_count:
                    correct_by_class[class_index] += int(
                        (predictions[mask] == labels[mask]).sum().item()
                    )
            progress.set_postfix(
                loss=f"{total_loss / count:.4f}",
                acc=f"{100.0 * correct_by_class.sum().item() / count:.2f}%",
            )
    if count == 0:
        raise RuntimeError("Source validation loader produced no examples")
    if torch.any(count_by_class == 0):
        missing = torch.nonzero(count_by_class == 0).flatten().tolist()
        raise RuntimeError(
            f"Source validation split has no examples for classes: {missing}"
        )
    per_class = 100.0 * correct_by_class.float() / count_by_class.float()
    return {
        "loss": total_loss / count,
        "overall_accuracy": float(
            100.0 * correct_by_class.sum().item() / count
        ),
        "macro_class_accuracy": float(per_class.mean().item()),
        "per_class_accuracy": [float(value) for value in per_class.tolist()],
        "per_class_count": [int(value) for value in count_by_class.tolist()],
        "example_count": count,
    }


def _checkpoint_stem_paths(output_path):
    stem, extension = osp.splitext(output_path)
    if extension.lower() != ".pth":
        raise ValueError("Source checkpoint output must use .pth")
    return {
        "final": output_path,
        "last": f"{stem}.last.pth",
        "manifest": f"{stem}.manifest.json",
        "split": f"{stem}.split.json",
        "metrics": f"{stem}.metrics.jsonl",
        "config": f"{stem}.effective.yaml",
    }


def _source_checkpoint_payload(model, metadata):
    return {
        "schema_version": SOURCE_CHECKPOINT_SCHEMA_VERSION,
        "kind": SOURCE_CHECKPOINT_KIND,
        "state_dict": _cpu_state_dict(model),
        "metadata": metadata,
    }


def _training_state_payload(
    model,
    optimizer,
    scaler,
    *,
    completed_epoch,
    best_metric,
    best_epoch,
    scientific_config_sha256,
):
    return {
        "schema_version": SOURCE_CHECKPOINT_SCHEMA_VERSION,
        "kind": "deit_source_training_state",
        "scientific_config_sha256": scientific_config_sha256,
        "completed_epoch": int(completed_epoch),
        "best_metric": float(best_metric),
        "best_epoch": int(best_epoch),
        "model_state_dict": _cpu_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }


def _restore_training_state(path, model, optimizer, scaler, config_sha256):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("kind") != "deit_source_training_state":
        raise ValueError(f"Not a resumable source training state: {path}")
    if payload.get("scientific_config_sha256") != config_sha256:
        raise ValueError(
            "Resume checkpoint config hash does not match the effective config"
        )
    _unwrap_model(model).load_state_dict(
        payload["model_state_dict"], strict=True
    )
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scaler.load_state_dict(payload["scaler_state_dict"])
    rng = payload["rng_state"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng["cuda"] is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])
    return {
        "start_epoch": int(payload["completed_epoch"]) + 1,
        "best_metric": float(payload["best_metric"]),
        "best_epoch": int(payload["best_epoch"]),
    }


def run_source_training(
    config,
    project_root,
    *,
    resume_path=None,
    model_builder=build_deit_source_model,
):
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    checkpoint_config = config["checkpoint"]
    runtime_config = config["runtime"]
    paths = _checkpoint_stem_paths(checkpoint_config["output_path"])

    if runtime_config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device=cuda but CUDA is not available")
    device = torch.device(runtime_config["device"])
    amp = bool(training_config["amp"] and device.type == "cuda")
    set_reproducibility(
        training_config["seed"], runtime_config["deterministic"]
    )

    if not osp.isfile(data_config["source_list"]):
        raise FileNotFoundError(
            f"Source image list not found: {data_config['source_list']}"
        )
    if not osp.isfile(data_config["class_mapping_path"]):
        raise FileNotFoundError(
            "Class mapping not found: "
            f"{data_config['class_mapping_path']}"
        )
    class_mapping_sha256 = sha256_file(
        data_config["class_mapping_path"]
    )
    class_mapping = _load_and_validate_class_mapping(
        data_config["class_mapping_path"], data_config
    )
    pretrained_path = model_config["pretrained_path"]
    if not osp.isfile(pretrained_path):
        raise FileNotFoundError(
            f"Local pretrained model not found: {pretrained_path}"
        )
    pretrained_sha256 = sha256_file(pretrained_path)
    if not osp.isfile(model_config["pretrained_config_path"]):
        raise FileNotFoundError(
            "Local pretrained config not found: "
            f"{model_config['pretrained_config_path']}"
        )
    pretrained_config_sha256 = sha256_file(
        model_config["pretrained_config_path"]
    )
    pretrained_config = _load_and_validate_pretrained_config(
        model_config["pretrained_config_path"],
        model_config,
        config["preprocessing"],
    )

    records = read_image_records(
        data_config["source_list"], data_config["num_classes"]
    )
    train_indices, validation_indices, split_manifest = (
        stratified_fixed_split(
            records,
            data_config["split"]["validation_fraction"],
            data_config["split"]["seed"],
        )
    )
    train_transform, validation_transform = build_transforms(
        config["preprocessing"]
    )
    train_dataset = SourceImageList(records, train_indices, train_transform)
    validation_dataset = SourceImageList(
        records, validation_indices, validation_transform
    )
    validation_loader = build_loader(
        validation_dataset,
        batch_size=training_config["batch_size"],
        workers=training_config["workers"],
        shuffle=False,
        seed=training_config["seed"] + 100_000,
        pin_memory=runtime_config["pin_memory"],
    )

    model, head = model_builder(
        model_name=model_config["name"],
        num_classes=data_config["num_classes"],
        pretrained_path=pretrained_path,
        pretrained_num_classes=model_config["pretrained_num_classes"],
        head_init_std=model_config["head_init_std"],
        drop_rate=model_config["drop_rate"],
        drop_path_rate=model_config["drop_path_rate"],
    )
    model.to(device)
    optimizer, optimizer_manifest = build_adamw(
        model, training_config["optimizer"]
    )
    visible_cuda_devices = (
        torch.cuda.device_count() if device.type == "cuda" else 0
    )
    data_parallel_effective = bool(
        runtime_config["data_parallel"] and visible_cuda_devices > 1
    )
    if data_parallel_effective:
        model = nn.DataParallel(
            model, device_ids=list(range(visible_cuda_devices))
        )
    steps_per_epoch = math.ceil(
        len(train_dataset) / training_config["batch_size"]
    )
    total_steps = steps_per_epoch * training_config["epochs"]
    warmup_steps = (
        steps_per_epoch * training_config["scheduler"]["warmup_epochs"]
    )
    schedule = WarmupCosineSchedule(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr=training_config["scheduler"]["min_lr"],
    )
    criterion = nn.CrossEntropyLoss(
        label_smoothing=training_config["label_smoothing"]
    )
    scaler = _grad_scaler(amp)

    start_epoch = 0
    best_metric = -math.inf
    best_epoch = -1
    resume_path = resume_path or (
        paths["last"] if osp.isfile(paths["last"]) else None
    )
    if resume_path is not None:
        restored = _restore_training_state(
            resume_path,
            model,
            optimizer,
            scaler,
            config["scientific_config_sha256"],
        )
        start_epoch = restored["start_epoch"]
        best_metric = restored["best_metric"]
        best_epoch = restored["best_epoch"]

    os.makedirs(osp.dirname(paths["final"]), exist_ok=True)
    dump_source_config(paths["config"], config)
    _json_dump(paths["split"], split_manifest)
    if start_epoch == 0 and osp.isfile(paths["metrics"]):
        raise FileExistsError(
            f"Metrics already exist; move them or resume explicitly: {paths['metrics']}"
        )

    run_metadata = {
        "created_at_utc": _utc_now(),
        "command": sys.argv,
        "git": _git_info(project_root),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": _package_version("torchvision"),
            "timm": _package_version("timm"),
            "safetensors": _package_version("safetensors"),
            "cuda": torch.version.cuda,
            "device": str(device),
            "visible_cuda_device_count": visible_cuda_devices,
            "visible_cuda_device_names": [
                torch.cuda.get_device_name(index)
                for index in range(visible_cuda_devices)
            ],
            "data_parallel_requested": runtime_config["data_parallel"],
            "data_parallel_effective": data_parallel_effective,
        },
        "dataset": {
            "name": data_config["dataset"],
            "source_domain": data_config["source_domain"],
            "num_classes": data_config["num_classes"],
            "source_list": data_config["source_list"],
            "class_mapping_path": data_config["class_mapping_path"],
            "class_mapping_sha256": class_mapping_sha256,
            "class_mapping": class_mapping,
            "split_manifest": paths["split"],
            "split": split_manifest,
        },
        "model": {
            "name": model_config["name"],
            "implementation": "timm",
            "non_distilled": True,
            "hidden_dim": model_config["hidden_dim"],
            "num_classes": data_config["num_classes"],
            "head_schema": f"Linear(384,{data_config['num_classes']})",
            "head_parameter_names": [
                name
                for name, parameter in _unwrap_model(model).named_parameters()
                if id(parameter) in {id(value) for value in head.parameters()}
            ],
            "drop_rate": model_config["drop_rate"],
            "drop_path_rate": model_config["drop_path_rate"],
            "parameter_count": sum(
                p.numel() for p in _unwrap_model(model).parameters()
            ),
            "trainable_parameter_count": sum(
                p.numel()
                for p in _unwrap_model(model).parameters()
                if p.requires_grad
            ),
        },
        "pretrained": {
            "path": pretrained_path,
            "sha256": pretrained_sha256,
            "config_path": model_config.get("pretrained_config_path"),
            "config_sha256": pretrained_config_sha256,
            "config": pretrained_config,
            "loaded_locally": True,
            "implicit_download": False,
        },
        "training": {
            **training_config,
            "amp_effective": amp,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "optimizer_parameter_groups": optimizer_manifest,
            "selection_metric": checkpoint_config["selection_metric"],
            "target_labels_used": False,
        },
        "preprocessing": config["preprocessing"],
        "effective_config": config,
        "scientific_config_sha256": config["scientific_config_sha256"],
    }

    if start_epoch >= training_config["epochs"]:
        if not osp.isfile(paths["final"]):
            raise RuntimeError(
                "The resumable state says training is complete, but the "
                f"selected best checkpoint is missing: {paths['final']}"
            )
        return {
            "checkpoint_path": paths["final"],
            "checkpoint_sha256": sha256_file(paths["final"]),
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "already_complete": True,
            "artifact_paths": paths,
        }

    for epoch in range(start_epoch, training_config["epochs"]):
        train_loader = build_loader(
            train_dataset,
            batch_size=training_config["batch_size"],
            workers=training_config["workers"],
            shuffle=True,
            seed=training_config["seed"] + epoch,
            pin_memory=runtime_config["pin_memory"],
        )
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            schedule,
            device=device,
            amp=amp,
            scaler=scaler,
            global_step_start=epoch * steps_per_epoch,
            progress_desc=(
                f"train {data_config['dataset']}/{data_config['source_domain']} "
                f"{epoch + 1:03d}/{training_config['epochs']:03d}"
            ),
        )
        validation_metrics = evaluate(
            model,
            validation_loader,
            criterion,
            device=device,
            amp=amp,
            num_classes=data_config["num_classes"],
            progress_desc=(
                f"valid {data_config['dataset']}/{data_config['source_domain']} "
                f"{epoch + 1:03d}/{training_config['epochs']:03d}"
            ),
        )
        selected_value = validation_metrics[
            checkpoint_config["selection_metric"]
        ]
        improved = selected_value > best_metric
        if improved:
            best_metric = selected_value
            best_epoch = epoch
        epoch_record = {
            "epoch": epoch,
            "epoch_one_based": epoch + 1,
            "timestamp_utc": _utc_now(),
            "train": train_metrics,
            "source_validation": validation_metrics,
            "selection_metric": checkpoint_config["selection_metric"],
            "selection_value": selected_value,
            "is_best": improved,
            "best_epoch": best_epoch,
            "best_metric": best_metric,
        }
        _jsonl_append(paths["metrics"], epoch_record)
        tqdm.write(
            f"epoch={epoch + 1}/{training_config['epochs']} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_overall={validation_metrics['overall_accuracy']:.3f} "
            f"val_macro={validation_metrics['macro_class_accuracy']:.3f} "
            f"best_{checkpoint_config['selection_metric']}={best_metric:.3f}"
        )

        if improved:
            checkpoint_metadata = {
                **run_metadata,
                "best": {
                    "epoch": best_epoch,
                    "epoch_one_based": best_epoch + 1,
                    "selection_metric": checkpoint_config["selection_metric"],
                    "selection_value": best_metric,
                    "source_validation": validation_metrics,
                },
                "saved_at_utc": _utc_now(),
            }
            _atomic_torch_save(
                paths["final"],
                _source_checkpoint_payload(model, checkpoint_metadata),
            )
            _json_dump(
                paths["manifest"],
                {
                    **checkpoint_metadata,
                    "checkpoint": {
                        "path": paths["final"],
                        "sha256": sha256_file(paths["final"]),
                        "schema_version": SOURCE_CHECKPOINT_SCHEMA_VERSION,
                        "kind": SOURCE_CHECKPOINT_KIND,
                    },
                },
            )

        if checkpoint_config["save_last_each_epoch"]:
            _atomic_torch_save(
                paths["last"],
                _training_state_payload(
                    model,
                    optimizer,
                    scaler,
                    completed_epoch=epoch,
                    best_metric=best_metric,
                    best_epoch=best_epoch,
                    scientific_config_sha256=config[
                        "scientific_config_sha256"
                    ],
                ),
            )

    if not osp.isfile(paths["final"]):
        raise RuntimeError("Training finished without producing a best checkpoint")
    return {
        "checkpoint_path": paths["final"],
        "checkpoint_sha256": sha256_file(paths["final"]),
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "artifact_paths": paths,
    }
