"""Faithful SHOT-OTTA baseline training and evaluation loop."""

import copy
import json
import os
import os.path as osp
import random
import signal
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from core.lbi import (
    SplitLBIEngine,
    compute_lbi_run_budget_diagnostics,
    target_support_count,
)
from .artifacts import (
    SCHEMA_VERSION,
    append_jsonl,
    create_run_dir,
    dump_json,
    experiment_output_root,
    write_initial_artifacts,
)
from .data import build_loaders
from .losses import shot_adaptation_loss
from .models import load_source_models


def setup_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


MODULE_CANDIDATE_NAMES = {
    "netB.bottleneck.weight",
    "netB.bottleneck.bias",
}

STREAM_CHECKPOINT_SCHEMA_VERSION = 1


class StreamInterrupted(RuntimeError):
    """Raised after a checkpoint has been safely written for a signal."""


def _stream_checkpoint_paths(output_dir):
    checkpoint_dir = osp.join(output_dir, "checkpoints")
    return {
        "directory": checkpoint_dir,
        "state": osp.join(checkpoint_dir, "stream_state.pt"),
        "metadata": osp.join(checkpoint_dir, "stream_state.json"),
    }


def _atomic_torch_save(path, payload):
    os.makedirs(osp.dirname(path), exist_ok=True)
    temporary_path = f"{path}.tmp"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def _atomic_json_dump(path, payload):
    os.makedirs(osp.dirname(path), exist_ok=True)
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)
    os.replace(temporary_path, path)


def _capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _save_stream_checkpoint(
    output_dir,
    config,
    run_id,
    net_f,
    net_b,
    net_c,
    optimizer,
    iteration,
    loader_batches_processed,
    all_post_predictions,
    all_online_labels,
    dynamic_selection_history,
    started_at_utc,
    runtime_seconds,
):
    """Atomically checkpoint only completed online steps, never LBI mid-step."""
    paths = _stream_checkpoint_paths(output_dir)
    payload = {
        "schema_version": STREAM_CHECKPOINT_SCHEMA_VERSION,
        "run_id": run_id,
        "experiment_key": config["experiment_key"],
        "experiment_config_sha256": config[
            "experiment_config_sha256"
        ],
        "iteration": int(iteration),
        "loader_batches_processed": int(loader_batches_processed),
        "started_at_utc": started_at_utc,
        "runtime_seconds": float(runtime_seconds),
        "net_f": net_f.state_dict(),
        "net_b": net_b.state_dict(),
        "net_c": net_c.state_dict(),
        "optimizer": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "all_post_predictions": all_post_predictions,
        "all_online_labels": all_online_labels,
        "dynamic_selection_history": dynamic_selection_history,
        "rng_state": _capture_rng_state(),
    }
    _atomic_torch_save(paths["state"], payload)
    _atomic_json_dump(
        paths["metadata"],
        {
            "schema_version": STREAM_CHECKPOINT_SCHEMA_VERSION,
            "status": "in_progress",
            "run_id": run_id,
            "experiment_key": config["experiment_key"],
            "experiment_config_sha256": config[
                "experiment_config_sha256"
            ],
            "iteration": int(iteration),
            "loader_batches_processed": int(loader_batches_processed),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return paths


def _load_stream_checkpoint(resume_run_dir, config, net_f, net_b, net_c, optimizer):
    paths = _stream_checkpoint_paths(resume_run_dir)
    if not osp.isfile(paths["state"]) or not osp.isfile(paths["metadata"]):
        raise RuntimeError(
            "resume run directory has no complete stream checkpoint: "
            f"{resume_run_dir}"
        )
    with open(paths["metadata"], "r", encoding="utf-8") as file_obj:
        metadata = json.load(file_obj)
    for field in ("experiment_key", "experiment_config_sha256"):
        if metadata.get(field) != config[field]:
            raise RuntimeError(
                "resume checkpoint identity mismatch for "
                f"{field}: checkpoint={metadata.get(field)!r}, "
                f"requested={config[field]!r}"
            )
    payload = torch.load(
        paths["state"], map_location="cpu", weights_only=False
    )
    if payload.get("schema_version") != STREAM_CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("unsupported stream checkpoint schema")
    for field in ("experiment_key", "experiment_config_sha256"):
        if payload.get(field) != config[field]:
            raise RuntimeError(
                f"stream checkpoint payload identity mismatch for {field}"
            )
    net_f.load_state_dict(payload["net_f"])
    net_b.load_state_dict(payload["net_b"])
    net_c.load_state_dict(payload["net_c"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload, paths


def _existing_artifact_paths(output_dir, summary_filename):
    paths = {
        "config": osp.join(output_dir, "config.yaml"),
        "manifest": osp.join(output_dir, "manifest.json"),
        "metrics": osp.join(output_dir, "metrics.jsonl"),
        "summary": osp.join(output_dir, summary_filename),
    }
    missing = [name for name in ("config", "manifest", "metrics") if not osp.isfile(paths[name])]
    if missing:
        raise RuntimeError(
            "resume run directory is missing artifacts: "
            f"{', '.join(missing)}"
        )
    return paths


def _write_incomplete_run_record(
    artifact_paths,
    config,
    run_id,
    started_at_utc,
    iteration,
    loader_batches_processed,
    reason,
):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "incomplete",
        "run_id": run_id,
        "experiment_key": config["experiment_key"],
        "experiment_config_sha256": config[
            "experiment_config_sha256"
        ],
        "started_at_utc": started_at_utc,
        "interrupted_at_utc": datetime.now(timezone.utc).isoformat(),
        "online_steps_completed": int(iteration),
        "loader_batches_processed": int(loader_batches_processed),
        "reason": reason,
        "stream_checkpoint": _stream_checkpoint_paths(
            osp.dirname(artifact_paths["metrics"])
        ),
    }
    dump_json(
        osp.join(osp.dirname(artifact_paths["metrics"]), "interruption.json"),
        payload,
    )
    dump_json(artifact_paths["summary"], payload)


def _truncate_metrics_to_checkpoint(metrics_path, completed_iteration):
    """Discard rows written after the last atomically committed checkpoint."""
    retained = []
    with open(metrics_path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if not line.strip():
                continue
            record = json.loads(line)
            if (
                record.get("event") == "online_step"
                and int(record.get("iteration", 0)) > completed_iteration
            ):
                continue
            if record.get("event") == "final":
                continue
            retained.append(record)
    temporary_path = f"{metrics_path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as file_obj:
        for record in retained:
            file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary_path, metrics_path)


def _collect_module_candidates(all_named_parameters):
    candidate_parameters = [
        (name, parameter)
        for name, parameter in all_named_parameters
        if name in MODULE_CANDIDATE_NAMES
    ]
    found_names = {name for name, _ in candidate_parameters}
    if found_names != MODULE_CANDIDATE_NAMES:
        missing = sorted(MODULE_CANDIDATE_NAMES - found_names)
        raise RuntimeError(f"module parameters not found: {missing}")
    return candidate_parameters


def _budget_to_num_keep(numel, requested_budget):
    num_keep = int(numel * requested_budget)
    if 0.0 < requested_budget < 1.0 and num_keep == 0 and numel > 0:
        num_keep = 1
    if requested_budget == 0.0:
        return 0
    if requested_budget == 1.0:
        return numel
    return num_keep


def _build_static_random_masks(named_parameters, requested_budget, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    masks = {}
    for name, parameter in named_parameters:
        numel = parameter.numel()
        num_keep = _budget_to_num_keep(numel, requested_budget)

        flat_mask = torch.zeros(numel, dtype=torch.bool, device="cpu")
        if num_keep > 0:
            selected_indices = torch.randperm(
                numel, generator=generator, device="cpu"
            )[:num_keep]
            flat_mask[selected_indices] = True
        masks[name] = flat_mask.reshape(parameter.shape).to(
            device=parameter.device
        )
    return masks


def _build_stable_score_mask(scores, requested_budget, device):
    numel = scores.numel()
    num_keep = _budget_to_num_keep(numel, requested_budget)
    flat_mask = torch.zeros(numel, dtype=torch.bool, device="cpu")
    if num_keep == numel:
        flat_mask.fill_(True)
    elif num_keep > 0:
        flat_scores = scores.detach().reshape(-1).to(device="cpu")
        ranked_indices = torch.argsort(
            flat_scores,
            descending=True,
            stable=True,
        )
        flat_mask[ranked_indices[:num_keep]] = True
    return flat_mask.reshape(scores.shape).to(device=device)


def _build_static_magnitude_masks(named_parameters, requested_budget):
    masks = {}
    for name, parameter in named_parameters:
        scores = parameter.detach().abs()
        masks[name] = _build_stable_score_mask(
            scores,
            requested_budget=requested_budget,
            device=parameter.device,
        )
    return masks


def _compute_saliency_score(name, parameter):
    if parameter.grad is None:
        raise RuntimeError(
            f"Cannot build saliency mask: gradient is None for {name}"
        )
    return torch.abs(parameter.detach() * parameter.grad.detach())


def _build_dynamic_saliency_masks(named_parameters, requested_budget):
    masks = {}
    for name, parameter in named_parameters:
        scores = _compute_saliency_score(name, parameter)
        masks[name] = _build_stable_score_mask(
            scores,
            requested_budget=requested_budget,
            device=parameter.device,
        )
    return masks


def _count_selected_mask_elements(masks):
    return sum(
        int(mask.count_nonzero().item()) for mask in masks.values()
    )


def _compute_mask_selection_stats(
    masks,
    candidate_scope_param_count,
    total_model_param_count,
):
    selected_param_count = _count_selected_mask_elements(masks)
    selected_over_scope_ratio = (
        float(selected_param_count / candidate_scope_param_count)
        if candidate_scope_param_count > 0
        else 0.0
    )
    selected_over_model_ratio = (
        float(selected_param_count / total_model_param_count)
        if total_model_param_count > 0
        else 0.0
    )
    return {
        "selected_param_count": int(selected_param_count),
        "selected_over_scope_ratio": selected_over_scope_ratio,
        "selected_over_model_ratio": selected_over_model_ratio,
    }


def _summarize_dynamic_selection_stats(step_stats):
    fields = (
        "selected_param_count",
        "selected_over_scope_ratio",
        "selected_over_model_ratio",
    )
    summary = {}
    for field in fields:
        values = [step[field] for step in step_stats]
        if values:
            summary[field] = sum(values) / len(values)
            summary[f"{field}_first"] = values[0]
            summary[f"{field}_last"] = values[-1]
            summary[f"{field}_min"] = min(values)
            summary[f"{field}_max"] = max(values)
            summary[f"{field}_mean"] = sum(values) / len(values)
        else:
            summary[field] = None
            summary[f"{field}_first"] = None
            summary[f"{field}_last"] = None
            summary[f"{field}_min"] = None
            summary[f"{field}_max"] = None
            summary[f"{field}_mean"] = None
    return summary


def _summarize_lbi_step_stats(step_stats):
    fields = (
        "selected_param_count",
        "selected_over_scope_ratio",
        "selected_over_model_ratio",
        "effective_delta_nonzero_count",
        "effective_delta_over_scope_ratio",
        "effective_delta_over_model_ratio",
        "applied_update_nonzero_count",
        "applied_update_over_scope_ratio",
        "applied_update_over_model_ratio",
        "stage1_steps_completed",
        "stage2_steps_completed",
    )
    summary = {}
    for field in fields:
        values = [step[field] for step in step_stats]
        mean_value = sum(values) / len(values) if values else None
        summary[field] = mean_value
        summary[f"{field}_first"] = values[0] if values else None
        summary[f"{field}_last"] = values[-1] if values else None
        summary[f"{field}_min"] = min(values) if values else None
        summary[f"{field}_max"] = max(values) if values else None
        summary[f"{field}_mean"] = mean_value
    summary["stage1_steps_mean"] = summary[
        "stage1_steps_completed_mean"
    ]
    summary["stage2_steps_mean"] = summary[
        "stage2_steps_completed_mean"
    ]
    return summary


def _masked_optimizer_step(optimizer, parameter_lookup, masks):
    pre_step_parameters = {}
    with torch.no_grad():
        for name, mask in masks.items():
            parameter = parameter_lookup[name]
            if parameter.grad is not None:
                parameter.grad.mul_(
                    mask.to(
                        device=parameter.grad.device,
                        dtype=parameter.grad.dtype,
                    )
                )
            pre_step_parameters[name] = parameter.detach().clone()

    optimizer.step()

    with torch.no_grad():
        for name, mask in masks.items():
            parameter = parameter_lookup[name]
            parameter.copy_(
                torch.where(
                    mask.to(device=parameter.device),
                    parameter,
                    pre_step_parameters[name],
                )
            )


def _configure_variant(config, net_f, net_b, net_c):
    optimization = config["optimization"]
    named_models = (("netF", net_f), ("netB", net_b), ("netC", net_c))
    batch_norm_modules = [
        module
        for _, model in named_models
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    all_named_parameters = [
        (f"{model_name}.{name}", parameter)
        for model_name, model in named_models
        for name, parameter in model.named_parameters()
    ]
    for _, parameter in all_named_parameters:
        parameter.requires_grad = False

    variant = config["variant"]
    static_masks = {}
    if variant == "source_only":
        candidate_parameters = []
        selected_parameters = []
        selection = "none"
        candidate_scope = "none"
        candidate_scope_type = "none"
        requested_budget = None
        ranking_source = None
        mask_refresh_policy = None
        saliency_score = None
        expected_selected_param_count_per_step = None
        bn_stats_policy = "frozen"
        bn_stats_frozen = True
        selection_seed = None
        mask_static = False
        net_f.eval()
        net_b.eval()
        net_c.eval()
    elif variant == "full_dense":
        candidate_parameters = [
            (name, parameter)
            for name, parameter in all_named_parameters
            if name.startswith("netF.") or name.startswith("netB.")
        ]
        selected_parameters = list(candidate_parameters)
        selection = "dense"
        candidate_scope = "netF+netB"
        candidate_scope_type = "all_parameters"
        requested_budget = 1.0
        ranking_source = None
        mask_refresh_policy = None
        saliency_score = None
        expected_selected_param_count_per_step = None
        bn_stats_policy = "adaptive"
        bn_stats_frozen = False
        selection_seed = None
        mask_static = False
        net_f.train()
        net_b.train()
        net_c.eval()
    elif variant in {
        "module_dense",
        "module_random",
        "module_magnitude",
        "module_saliency",
        "module_lbi",
    }:
        candidate_parameters = _collect_module_candidates(
            all_named_parameters
        )
        selected_parameters = list(candidate_parameters)
        candidate_scope = "netB.bottleneck"
        candidate_scope_type = "fc_parameters"
        bn_stats_policy = "frozen"
        bn_stats_frozen = True
        net_f.train()
        net_b.train()
        net_c.eval()
        for module in batch_norm_modules:
            module.eval()

        if variant == "module_dense":
            selection = "dense"
            requested_budget = 1.0
            selection_seed = None
            mask_static = False
            ranking_source = None
            mask_refresh_policy = None
            saliency_score = None
            expected_selected_param_count_per_step = None
        elif variant == "module_random":
            selection = "random"
            requested_budget = float(config["requested_budget"])
            selection_seed = int(config["selection_seed"])
            mask_static = True
            ranking_source = "independent_random_generator"
            mask_refresh_policy = "once_before_adaptation"
            saliency_score = None
            expected_selected_param_count_per_step = None
            static_masks = _build_static_random_masks(
                candidate_parameters,
                requested_budget=requested_budget,
                seed=selection_seed,
            )
        elif variant == "module_magnitude":
            selection = "magnitude"
            requested_budget = float(config["requested_budget"])
            selection_seed = None
            mask_static = True
            ranking_source = "source_checkpoint_pre_adaptation"
            mask_refresh_policy = "once_before_adaptation"
            saliency_score = None
            expected_selected_param_count_per_step = None
            static_masks = _build_static_magnitude_masks(
                candidate_parameters,
                requested_budget=requested_budget,
            )
        elif variant == "module_saliency":
            selection = "saliency"
            requested_budget = float(config["requested_budget"])
            selection_seed = None
            mask_static = False
            mask_refresh_policy = "every_online_step_after_backward"
            ranking_source = (
                "current_parameter_times_current_gradient"
            )
            saliency_score = "abs_parameter_times_gradient"
            expected_selected_param_count_per_step = sum(
                _budget_to_num_keep(
                    parameter.numel(),
                    requested_budget,
                )
                for _, parameter in candidate_parameters
            )
        else:
            selection = "lbi"
            requested_budget = float(config["requested_budget"])
            selection_seed = None
            mask_static = False
            mask_refresh_policy = "every_online_step_via_split_lbi"
            ranking_source = "split_lbi_gamma_support"
            saliency_score = None
            expected_selected_param_count_per_step = None
    else:
        raise ValueError(f"Unsupported variant: {variant}")

    selected_names = {name for name, _ in selected_parameters}
    for name, parameter in all_named_parameters:
        parameter.requires_grad = name in selected_names

    if variant in {"module_saliency", "module_lbi"}:
        selected_param_count = None
    else:
        selected_param_count = (
            _count_selected_mask_elements(static_masks)
            if mask_static
            else sum(
                parameter.numel()
                for _, parameter in selected_parameters
            )
        )
    candidate_scope_param_count = sum(
        parameter.numel() for _, parameter in candidate_parameters
    )
    total_model_param_count = sum(
        parameter.numel() for _, parameter in all_named_parameters
    )
    if selected_param_count is None:
        selected_over_scope_ratio = None
        selected_over_model_ratio = None
    else:
        selected_over_scope_ratio = (
            float(selected_param_count / candidate_scope_param_count)
            if candidate_scope_param_count > 0
            else 0.0
        )
        selected_over_model_ratio = (
            float(selected_param_count / total_model_param_count)
            if total_model_param_count > 0
            else 0.0
        )
    selection_stats = {
        "selection": selection,
        "candidate_scope": candidate_scope,
        "candidate_scope_type": candidate_scope_type,
        "requested_budget": requested_budget,
        "ranking_source": ranking_source,
        "mask_refresh_policy": mask_refresh_policy,
        "saliency_score": saliency_score,
        "expected_selected_param_count_per_step": (
            expected_selected_param_count_per_step
        ),
        "selected_param_count": (
            int(selected_param_count)
            if selected_param_count is not None
            else None
        ),
        "candidate_scope_param_count": int(candidate_scope_param_count),
        "total_model_param_count": int(total_model_param_count),
        "selected_over_scope_ratio": selected_over_scope_ratio,
        "selected_over_model_ratio": selected_over_model_ratio,
        "bn_stats_policy": bn_stats_policy,
        "bn_stats_frozen": bn_stats_frozen,
        "bn_module_count": len(batch_norm_modules),
        "selection_seed": selection_seed,
        "mask_static": mask_static,
    }
    if variant == "module_lbi":
        lbi = config["lbi"]
        selection_stats.update(
            {
                "lbi_alpha": float(lbi["alpha"]),
                "lbi_kappa": float(lbi["kappa"]),
                "lbi_nu": float(lbi["nu"]),
                "lbi_omega": float(lbi["omega"]),
                "alpha": float(lbi["alpha"]),
                "kappa": float(lbi["kappa"]),
                "nu": float(lbi["nu"]),
                "omega": float(lbi["omega"]),
                "stage1_max_steps": int(lbi["stage1_max_steps"]),
                "budget_tolerance": float(lbi["budget_tolerance"]),
                "stage2_lr": float(lbi["stage2_lr"]),
                "stage2_steps_requested": int(lbi["stage2_steps"]),
                "stage2_optimizer": "sgd",
                "lbi_support_threshold": 1.0e-4,
                "stage2_momentum": 0.9,
                "stage2_weight_decay": 1.0e-3,
                "stage2_nesterov": True,
                "stage2_lr_policy": "shot",
                "stage2_lr_gamma": 10.0,
                "stage2_lr_power": 0.75,
                "delta_nonzero_tolerance": float(
                    lbi["delta_nonzero_tolerance"]
                ),
                "lbi_state_lifecycle": "reset_every_online_step",
                "lbi_initialization": "dense",
                "stage3_mode": "accumulation",
                "stage1_branch_enabled": requested_budget > 0.0,
                "target_support_count": target_support_count(
                    requested_budget, candidate_scope_param_count
                ),
                "support_param_count": None,
                "support_over_scope_ratio": None,
                "support_over_model_ratio": None,
                "effective_delta_nonzero_count": None,
                "effective_delta_over_scope_ratio": None,
                "effective_delta_over_model_ratio": None,
                "effective_delta_l1": None,
                "effective_delta_l2": None,
                "applied_update_nonzero_count": None,
                "applied_update_over_scope_ratio": None,
                "applied_update_over_model_ratio": None,
                "applied_update_l1": None,
                "applied_update_l2": None,
            }
        )

    learning_rate = optimization["lr"]
    parameter_groups = []
    optimizer_parameters = (
        [] if variant == "module_lbi" else selected_parameters
    )
    for name, parameter in optimizer_parameters:
        lr_scale = (
            optimization["lr_decay1"]
            if name.startswith("netF.")
            else optimization["lr_decay2"]
        )
        parameter_groups.append(
            {"params": parameter, "lr": learning_rate * lr_scale}
        )
    return parameter_groups, selection_stats, static_masks


def _build_optimizer(config, parameter_groups):
    if not parameter_groups:
        return None
    optimization = config["optimization"]
    optimizer_name = optimization["optimizer"]
    if optimizer_name == "sgd":
        optimizer = optim.SGD(
            parameter_groups,
            momentum=optimization["momentum"],
            weight_decay=optimization["weight_decay"],
            nesterov=optimization["nesterov"],
        )
    elif optimizer_name == "adam":
        optimizer = optim.Adam(
            parameter_groups,
            weight_decay=optimization["weight_decay"],
            betas=(optimization["momentum"], 0.999),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    for group in optimizer.param_groups:
        group["lr0"] = group["lr"]
    return optimizer


def _schedule_learning_rate(config, optimizer, iteration, max_iterations):
    optimization = config["optimization"]
    decay = (
        1.0
        + optimization["lr_gamma"] * iteration / max_iterations
    ) ** (-optimization["lr_power"])
    for group in optimizer.param_groups:
        group["lr"] = group["lr0"] * decay
        if optimization["optimizer"] == "sgd":
            group["momentum"] = optimization["momentum"]
            group["nesterov"] = optimization["nesterov"]


def _compute_metrics(labels, predictions):
    label_values = np.union1d(labels, predictions)
    label_to_index = {
        int(label): index for index, label in enumerate(label_values)
    }
    matrix = np.zeros(
        (len(label_values), len(label_values)), dtype=np.int64
    )
    true_indices = np.fromiter(
        (label_to_index[int(label)] for label in labels),
        dtype=np.int64,
        count=len(labels),
    )
    predicted_indices = np.fromiter(
        (label_to_index[int(label)] for label in predictions),
        dtype=np.int64,
        count=len(predictions),
    )
    np.add.at(matrix, (true_indices, predicted_indices), 1)
    class_totals = matrix.sum(axis=1)
    per_class = np.divide(
        matrix.diagonal() * 100.0,
        class_totals,
        out=np.zeros_like(class_totals, dtype=float),
        where=class_totals > 0,
    )
    return float(per_class.mean()), [float(value) for value in per_class]


def _compute_dataset_metrics(labels, predictions, dataset, prefix):
    """Keep Office metrics unchanged while using fixed-class VisDA metrics."""
    if dataset == "VISDA-C":
        from visda_otta.evaluator import compute_metrics

        return compute_metrics(labels, predictions, prefix)
    accuracy, per_class = _compute_metrics(labels, predictions)
    return {f"{prefix}-Acc": accuracy, f"{prefix}-Acc-per-class": per_class}


def _evaluate(loader, net_f, net_b, net_c, device, dataset):
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels, _ in loader:
            inputs = inputs.to(device)
            outputs = net_c(net_b(net_f(inputs)))
            all_predictions.append(outputs.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())
    predictions = torch.cat(all_predictions).numpy()
    labels = torch.cat(all_labels).numpy()
    return _compute_dataset_metrics(labels, predictions, dataset, "FO")


def _save_model(output_dir, net_f, net_b, net_c):
    checkpoint_dir = osp.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    paths = {
        "netF": osp.join(checkpoint_dir, "target_F_otta.pt"),
        "netB": osp.join(checkpoint_dir, "target_B_otta.pt"),
        "netC": osp.join(checkpoint_dir, "target_C_otta.pt"),
    }
    torch.save(net_f.state_dict(), paths["netF"])
    torch.save(net_b.state_dict(), paths["netB"])
    torch.save(net_c.state_dict(), paths["netC"])
    return paths


def _run_single_experiment(
    config,
    workspace_root,
    run_id=None,
    output_dir=None,
    summary_filename="summary.json",
    mask_path=None,
    resume_run_dir=None,
    enable_stream_checkpoint=False,
):
    if resume_run_dir and not enable_stream_checkpoint:
        raise ValueError(
            "--resume-run-dir requires --enable-stream-checkpoint"
        )
    if enable_stream_checkpoint and (
        config["data"]["dataset"] != "VISDA-C"
        or config["variant"] != "module_lbi"
    ):
        raise ValueError(
            "stream checkpointing is restricted to VISDA-C module_lbi"
        )
    started_at_utc = datetime.now(timezone.utc).isoformat()
    segment_started_at = time.perf_counter()
    seed = int(config["seed"])
    setup_reproducibility(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, data_order = build_loaders(config)
    (net_f, net_b, net_c), source_paths = load_source_models(config, device)

    parameter_groups, selection_stats, static_masks = _configure_variant(
        config, net_f, net_b, net_c
    )
    optimizer = _build_optimizer(config, parameter_groups)
    masked_update_names = set(static_masks)
    if config["variant"] in {"module_saliency", "module_lbi"}:
        masked_update_names = set(MODULE_CANDIDATE_NAMES)
    masked_parameter_lookup = {
        f"{model_name}.{name}": parameter
        for model_name, model in (
            ("netF", net_f),
            ("netB", net_b),
            ("netC", net_c),
        )
        for name, parameter in model.named_parameters()
        if f"{model_name}.{name}" in masked_update_names
    }
    lbi_engine = None
    if config["variant"] == "module_lbi":
        lbi_engine = SplitLBIEngine(
            selection_stats["total_model_param_count"]
        )

    resume_payload = None
    if resume_run_dir is not None:
        if output_dir is not None or run_id is not None:
            raise ValueError(
                "resume-run-dir cannot be combined with an explicit output run"
            )
        output_dir = osp.abspath(resume_run_dir)
        resume_payload, _ = _load_stream_checkpoint(
            output_dir,
            config,
            net_f,
            net_b,
            net_c,
            optimizer,
        )
        run_id = resume_payload["run_id"]
        artifact_paths = _existing_artifact_paths(
            output_dir, summary_filename
        )
    elif output_dir is None:
        output_root = experiment_output_root(config)
        run_id, output_dir = create_run_dir(
            output_root,
            task_name=config["task_name"],
            run_name=config["output"].get("run_name"),
        )
        artifact_paths = write_initial_artifacts(
            output_dir=output_dir,
            run_id=run_id,
            config=config,
            data_order=data_order,
            source_checkpoint_paths=source_paths,
            selection_stats=selection_stats,
            workspace_root=workspace_root,
            summary_filename=summary_filename,
        )
    elif run_id is None:
        raise ValueError("run_id is required with an explicit output_dir")
    else:
        artifact_paths = write_initial_artifacts(
            output_dir=output_dir,
            run_id=run_id,
            config=config,
            data_order=data_order,
            source_checkpoint_paths=source_paths,
            selection_stats=selection_stats,
            workspace_root=workspace_root,
            summary_filename=summary_filename,
        )
    if mask_path is not None:
        torch.save(
            {
                name: mask.detach().to(device="cpu")
                for name, mask in static_masks.items()
            },
            mask_path,
        )
    open(artifact_paths["metrics"], "a", encoding="utf-8").close()

    all_post_predictions = []
    all_online_labels = []
    iteration = 0
    loader_batches_processed = 0
    runtime_before_resume = 0.0
    max_iterations = len(loaders["target"])
    loss_config = config["loss"]
    dynamic_selection_history = []
    resume_rng_state = None
    if resume_payload is not None:
        iteration = int(resume_payload["iteration"])
        loader_batches_processed = int(
            resume_payload["loader_batches_processed"]
        )
        if not 0 <= loader_batches_processed <= max_iterations:
            raise RuntimeError("invalid loader batch position in checkpoint")
        all_post_predictions = resume_payload["all_post_predictions"]
        all_online_labels = resume_payload["all_online_labels"]
        dynamic_selection_history = resume_payload[
            "dynamic_selection_history"
        ]
        started_at_utc = resume_payload["started_at_utc"]
        runtime_before_resume = float(resume_payload["runtime_seconds"])
        resume_rng_state = resume_payload["rng_state"]
        _truncate_metrics_to_checkpoint(artifact_paths["metrics"], iteration)
    visda_batch_metadata = (
        {
            "batch_size": int(config["data"]["batch_size"]),
            "target_sample_count": int(len(loaders["target"].dataset)),
            "target_batch_count": int(len(loaders["target"])),
            "drop_last": False,
        }
        if config["data"]["dataset"] == "VISDA-C"
        else {}
    )

    interruption = {"signum": None}
    previous_signal_handlers = {}

    def _request_interruption(signum, _frame):
        interruption["signum"] = int(signum)

    if enable_stream_checkpoint:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_signal_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _request_interruption)

    progress = tqdm(
        loaders["target"],
        desc="SHOT-OTTA baseline",
        dynamic_ncols=True,
    )
    for loader_batch_index, (inputs, labels, _) in enumerate(
        progress, start=1
    ):
        if loader_batch_index <= loader_batches_processed:
            continue
        if resume_rng_state is not None:
            _restore_rng_state(resume_rng_state)
            resume_rng_state = None
        if inputs.size(0) == 1:
            loader_batches_processed = loader_batch_index
            continue

        inputs = inputs.to(device)
        labels = labels.to(device)
        iteration += 1
        all_online_labels.append(labels.cpu())
        step_selection_stats = selection_stats
        if config["variant"] == "source_only":
            total_loss_value = None
            classification_loss_value = None
            entropy_loss_value = None
            diversity_loss_value = None
            pseudo_ratio = None
            max_probability_mean = None
            current_lr = None
            with torch.no_grad():
                post_outputs = net_c(net_b(net_f(inputs)))
                post_predictions = post_outputs.argmax(dim=1)
        elif config["variant"] == "module_lbi":
            lbi_config = {
                **config["lbi"],
                "requested_budget": config["requested_budget"],
            }

            def lbi_loss_closure():
                return shot_adaptation_loss(
                    inputs,
                    net_f,
                    net_b,
                    net_c,
                    loss_config,
                )

            lbi_result = lbi_engine.run_step(
                list(masked_parameter_lookup.items()),
                lbi_loss_closure,
                lbi_config,
            )
            lbi_step_stats = lbi_result.statistics
            dynamic_selection_history.append(lbi_step_stats)
            step_selection_stats = {
                **selection_stats,
                **lbi_step_stats,
            }
            total_loss_value = lbi_step_stats["loss"]
            classification_loss_value = lbi_step_stats.get("loss_cls")
            entropy_loss_value = lbi_step_stats.get("loss_ent")
            diversity_loss_value = lbi_step_stats.get("loss_div")
            pseudo_ratio = lbi_step_stats.get("pseudo_ratio")
            max_probability_mean = lbi_step_stats.get("max_prob_mean")
            current_lr = lbi_step_stats["stage2_effective_lr"]
            with torch.no_grad():
                post_outputs = net_c(net_b(net_f(inputs)))
                post_predictions = post_outputs.argmax(dim=1)
        else:
            _schedule_learning_rate(
                config, optimizer, iteration, max_iterations
            )
            optimizer.zero_grad()
            total_loss, loss_parts = shot_adaptation_loss(
                inputs,
                net_f,
                net_b,
                net_c,
                loss_config,
            )

            total_loss.backward()
            step_masks = static_masks
            if config["variant"] == "module_saliency":
                step_masks = _build_dynamic_saliency_masks(
                    list(masked_parameter_lookup.items()),
                    requested_budget=config["requested_budget"],
                )
                dynamic_step_stats = _compute_mask_selection_stats(
                    step_masks,
                    candidate_scope_param_count=selection_stats[
                        "candidate_scope_param_count"
                    ],
                    total_model_param_count=selection_stats[
                        "total_model_param_count"
                    ],
                )
                dynamic_selection_history.append(dynamic_step_stats)
                step_selection_stats = {
                    **selection_stats,
                    **dynamic_step_stats,
                }
            if step_masks:
                _masked_optimizer_step(
                    optimizer, masked_parameter_lookup, step_masks
                )
            else:
                optimizer.step()

            with torch.no_grad():
                post_outputs = net_c(net_b(net_f(inputs)))
                post_predictions = post_outputs.argmax(dim=1)

            total_loss_value = float(total_loss.item())
            classification_loss_value = loss_parts["loss_cls"]
            entropy_loss_value = loss_parts["loss_ent"]
            diversity_loss_value = loss_parts["loss_div"]
            pseudo_ratio = loss_parts["pseudo_ratio"]
            max_probability_mean = loss_parts["max_prob_mean"]
            current_lr = float(optimizer.param_groups[0]["lr"])

        all_post_predictions.append(post_predictions.cpu())

        post_accuracy = (
            (post_predictions == labels).float().mean().item() * 100.0
        )
        metric_record = {
            "schema_version": SCHEMA_VERSION,
            "event": "online_step",
            "run_id": run_id,
            "variant": config["variant"],
            "experiment_key": config["experiment_key"],
            "experiment_config_sha256": config[
                "experiment_config_sha256"
            ],
            "iteration": iteration,
            "acc_post": float(post_accuracy),
            "loss": total_loss_value,
            "loss_cls": classification_loss_value,
            "loss_ent": entropy_loss_value,
            "loss_div": diversity_loss_value,
            "lr": current_lr,
            "pseudo_ratio": pseudo_ratio,
            "max_prob_mean": max_probability_mean,
            **step_selection_stats,
        }
        append_jsonl(artifact_paths["metrics"], metric_record)
        loader_batches_processed = loader_batch_index
        if enable_stream_checkpoint:
            _save_stream_checkpoint(
                output_dir=output_dir,
                config=config,
                run_id=run_id,
                net_f=net_f,
                net_b=net_b,
                net_c=net_c,
                optimizer=optimizer,
                iteration=iteration,
                loader_batches_processed=loader_batches_processed,
                all_post_predictions=all_post_predictions,
                all_online_labels=all_online_labels,
                dynamic_selection_history=dynamic_selection_history,
                started_at_utc=started_at_utc,
                runtime_seconds=(
                    runtime_before_resume
                    + time.perf_counter() - segment_started_at
                ),
            )
        if interruption["signum"] is not None:
            reason = (
                "signal "
                f"{interruption['signum']} received after checkpoint"
            )
            _write_incomplete_run_record(
                artifact_paths,
                config,
                run_id,
                started_at_utc,
                iteration,
                loader_batches_processed,
                reason,
            )
            raise StreamInterrupted(reason)
        if total_loss_value is None:
            progress.set_description(
                f"SHOT-OTTA source-only | Post={post_accuracy:.2f}%"
            )
        else:
            progress.set_description(
                f"SHOT-OTTA | Post={post_accuracy:.2f}% "
                f"Loss={total_loss_value:.4f}"
            )

    if all_post_predictions:
        online_predictions = torch.cat(all_post_predictions).numpy()
        online_labels = torch.cat(all_online_labels).numpy()
        pu_metrics = _compute_dataset_metrics(
            online_labels, online_predictions, config["data"]["dataset"], "PU"
        )
    else:
        pu_metrics = _compute_dataset_metrics(
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
            config["data"]["dataset"],
            "PU",
        )
    pu_accuracy = pu_metrics["PU-Acc"]
    pu_per_class = pu_metrics["PU-Acc-per-class"]

    net_f.eval()
    net_b.eval()
    fo_metrics = _evaluate(
        loaders["test"], net_f, net_b, net_c, device, config["data"]["dataset"]
    )
    fo_accuracy = fo_metrics["FO-Acc"]
    fo_per_class = fo_metrics["FO-Acc-per-class"]
    visda_metrics = (
        {**pu_metrics, **fo_metrics}
        if config["data"]["dataset"] == "VISDA-C"
        else {}
    )

    checkpoint_paths = None
    if config["output"]["save_model"]:
        checkpoint_paths = _save_model(
            output_dir, net_f, net_b, net_c
        )

    runtime = float(
        runtime_before_resume + time.perf_counter() - segment_started_at
    )
    completed_at_utc = datetime.now(timezone.utc).isoformat()
    final_selection_stats = selection_stats
    if config["variant"] == "module_saliency":
        final_selection_stats = {
            **selection_stats,
            **_summarize_dynamic_selection_stats(
                dynamic_selection_history
            ),
        }
    elif config["variant"] == "module_lbi":
        dynamic_summary = _summarize_lbi_step_stats(
            dynamic_selection_history
        )
        budget_summary = compute_lbi_run_budget_diagnostics(
            dynamic_selection_history
        )
        if budget_summary["budget_diagnostics_available"]:
            budget_summary[
                "budget_diagnostics_source"
            ] = "summary_json"
        final_selection_stats = {
            **selection_stats,
            **dynamic_summary,
            **budget_summary,
            "support_param_count": dynamic_summary[
                "selected_param_count"
            ],
            "support_over_scope_ratio": dynamic_summary[
                "selected_over_scope_ratio"
            ],
            "support_over_model_ratio": dynamic_summary[
                "selected_over_model_ratio"
            ],
        }
    append_jsonl(
        artifact_paths["metrics"],
        {
            "schema_version": SCHEMA_VERSION,
            "event": "final",
            "run_id": run_id,
            "variant": config["variant"],
            "experiment_key": config["experiment_key"],
            "experiment_config_sha256": config[
                "experiment_config_sha256"
            ],
            **visda_batch_metadata,
            "PU-Acc": pu_accuracy,
            "FO-Acc": fo_accuracy,
            "PU-Acc-per-class": pu_per_class,
            "FO-Acc-per-class": fo_per_class,
            "online_steps": iteration,
            "runtime": runtime,
            **visda_metrics,
            **final_selection_stats,
        },
    )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "run_id": run_id,
        "output_dir": output_dir,
        "method": config["method"],
        "variant": config["variant"],
        "task": config["task"],
        "dataset": config["data"]["dataset"],
        **visda_batch_metadata,
        "source": config["data"]["source"],
        "target": config["data"]["target"],
        "source-target": (
            f"{config['data']['source_name']}-"
            f"{config['data']['target_name']}"
        ),
        "seed": seed,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "experiment_key": config["experiment_key"],
        "experiment_config_sha256": config[
            "experiment_config_sha256"
        ],
        "PU-Acc": pu_accuracy,
        "FO-Acc": fo_accuracy,
        "PU-Acc-per-class": pu_per_class,
        "FO-Acc-per-class": fo_per_class,
        "runtime": runtime,
        "online_steps": iteration,
        "checkpoints": checkpoint_paths,
        **visda_metrics,
        **final_selection_stats,
    }
    dump_json(artifact_paths["summary"], summary)
    if enable_stream_checkpoint:
        paths = _stream_checkpoint_paths(output_dir)
        _atomic_json_dump(
            paths["metadata"],
            {
                "schema_version": STREAM_CHECKPOINT_SCHEMA_VERSION,
                "status": "completed",
                "run_id": run_id,
                "experiment_key": config["experiment_key"],
                "experiment_config_sha256": config[
                    "experiment_config_sha256"
                ],
                "iteration": int(iteration),
                "loader_batches_processed": int(loader_batches_processed),
                "updated_at_utc": completed_at_utc,
            },
        )
        for signum, handler in previous_signal_handlers.items():
            signal.signal(signum, handler)
    print(f"Run complete: {output_dir}")
    print(f"PU-Acc={pu_accuracy} FO-Acc={fo_accuracy} runtime={runtime}")
    return summary


def _random_mask_seed(run_seed, mask_index):
    return int(run_seed) * 100 + int(mask_index)


def _write_random_summary_markdown(path, summary):
    lines = [
        "# Random baseline summary",
        "",
        f"- run_seed: {summary['run_seed']}",
        f"- mask_seeds: {summary['mask_seeds']}",
        "",
        "| mask | mask_seed | PU-Acc | FO-Acc |",
        "|---|---:|---:|---:|",
    ]
    for result in summary["masks"]:
        lines.append(
            f"| {result['mask_id']} | {result['mask_seed']} | "
            f"{result['PU-Acc']} | {result['FO-Acc']} |"
        )
    lines.extend(
        [
            "",
            f"- mean PU-Acc: {summary['mean_PU-Acc']}",
            f"- std PU-Acc: {summary['std_PU-Acc']}",
            f"- mean FO-Acc: {summary['mean_FO-Acc']}",
            f"- std FO-Acc: {summary['std_FO-Acc']}",
            "",
        ]
    )
    with open(path, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines))


def _run_random_experiment(config, workspace_root):
    started_at_utc = datetime.now(timezone.utc).isoformat()
    started_at = time.perf_counter()
    output_root = experiment_output_root(config)
    run_id, output_dir = create_run_dir(
        output_root,
        task_name=config["task_name"],
        run_name=config["output"].get("run_name"),
    )
    run_seed = int(config["seed"])
    mask_results = []
    for mask_index in range(config["num_random_masks"]):
        mask_id = f"mask_{mask_index:02d}"
        mask_seed = _random_mask_seed(run_seed, mask_index)
        mask_config = copy.deepcopy(config)
        # Only the independent mask generator differs between child runs.
        mask_config["selection_seed"] = mask_seed
        mask_dir = osp.join(output_dir, mask_id)
        os.makedirs(mask_dir)
        result = _run_single_experiment(
            mask_config,
            workspace_root,
            run_id=f"{run_id}/{mask_id}",
            output_dir=mask_dir,
            summary_filename="results.json",
            mask_path=osp.join(mask_dir, "mask.pt"),
        )
        mask_results.append(
            {
                "mask_id": mask_id,
                "mask_seed": mask_seed,
                "PU-Acc": result["PU-Acc"],
                "FO-Acc": result["FO-Acc"],
                "runtime": result["runtime"],
                "started_at_utc": result["started_at_utc"],
                "completed_at_utc": result["completed_at_utc"],
                "result_path": osp.join(mask_dir, "results.json"),
                **(
                    {
                        key: result[key]
                        for key in result
                        if key.startswith("PU-")
                        or key.startswith("FO-")
                        or key == "class-names"
                    }
                    if config["data"]["dataset"] == "VISDA-C"
                    else {}
                ),
            }
        )
    pu_values = [result["PU-Acc"] for result in mask_results]
    fo_values = [result["FO-Acc"] for result in mask_results]
    completed_at_utc = datetime.now(timezone.utc).isoformat()
    total_runtime = float(time.perf_counter() - started_at)
    selection_summary = result
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "run_id": run_id,
        "output_dir": output_dir,
        "method": config["method"],
        "variant": config["variant"],
        "task": config["task"],
        "dataset": config["data"]["dataset"],
        **(
            {
                "batch_size": int(config["data"]["batch_size"]),
                "target_sample_count": result.get("target_sample_count"),
                "target_batch_count": result.get("target_batch_count"),
                "drop_last": False,
            }
            if config["data"]["dataset"] == "VISDA-C"
            else {}
        ),
        "source": config["data"]["source"],
        "target": config["data"]["target"],
        "source-target": (
            f"{config['data']['source_name']}-"
            f"{config['data']['target_name']}"
        ),
        "seed": run_seed,
        "run_seed": run_seed,
        "requested_budget": float(config["requested_budget"]),
        # selection_seed identifies the fixed run-level setup; individual
        # masks and their seeds are recorded separately below.
        "selection_seed": int(config["selection_seed"]),
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "runtime": total_runtime,
        "total_runtime": total_runtime,
        "mask_seeds": [result["mask_seed"] for result in mask_results],
        "experiment_key": config["experiment_key"],
        "experiment_config_sha256": config["experiment_config_sha256"],
        "num_random_masks": config["num_random_masks"],
        "masks": mask_results,
        "mean_PU-Acc": float(np.mean(pu_values)),
        "std_PU-Acc": float(np.std(pu_values, ddof=0)),
        "mean_FO-Acc": float(np.mean(fo_values)),
        "std_FO-Acc": float(np.std(fo_values, ddof=0)),
        # Preserve the standard aggregate columns for existing status and
        # reporting tools; these are explicitly the per-mask means.
        "PU-Acc": float(np.mean(pu_values)),
        "FO-Acc": float(np.mean(fo_values)),
    }
    for field in (
        "selection",
        "mask_static",
        "candidate_scope",
        "candidate_scope_type",
        "ranking_source",
        "mask_refresh_policy",
        "selected_param_count",
        "candidate_scope_param_count",
        "total_model_param_count",
        "selected_over_scope_ratio",
        "selected_over_model_ratio",
        "bn_stats_policy",
        "bn_stats_frozen",
        "bn_module_count",
    ):
        summary[field] = selection_summary[field]
    if config["data"]["dataset"] == "VISDA-C":
        from visda_otta.evaluator import aggregate_random_mask_metrics

        summary.update(aggregate_random_mask_metrics(mask_results))
    dump_json(osp.join(output_dir, "summary.json"), summary)
    _write_random_summary_markdown(osp.join(output_dir, "summary.md"), summary)
    print(f"Random baseline complete: {output_dir}")
    return summary


def run_experiment(
    config,
    workspace_root,
    resume_run_dir=None,
    enable_stream_checkpoint=False,
):
    if config["variant"] == "module_random":
        if resume_run_dir or enable_stream_checkpoint:
            raise ValueError(
                "stream checkpointing is not available for module_random"
            )
        return _run_random_experiment(config, workspace_root)
    return _run_single_experiment(
        config,
        workspace_root,
        resume_run_dir=resume_run_dir,
        enable_stream_checkpoint=enable_stream_checkpoint,
    )
