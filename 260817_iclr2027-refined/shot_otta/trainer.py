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
    max_support_count,
)
from core.lbi.groups import (
    build_random_group_masks,
    global_group_score_mask,
    group_count,
    selected_group_count,
    selected_scalar_count,
)
from protocol_constants import (
    CONV_CANDIDATE_PARAM_COUNT,
    CONV_GROUP_COUNTS,
)
from .artifacts import (
    SCHEMA_VERSION,
    append_jsonl,
    create_run_dir,
    dump_json,
    experiment_output_root,
    write_initial_artifacts,
)
from .candidates import (
    CONV_CANDIDATE_NAMES,
    CONV_CANDIDATE_ORDER,
    CONV_CANDIDATE_SHAPES,
    MODULE_CANDIDATE_NAMES,
    MODULE_CANDIDATE_ORDER,
)
from .data import build_loaders
from .losses import shot_adaptation_loss
from .models import load_source_models
from .efficiency import (
    BATCH_EFFICIENCY_FIELDS,
    EFFICIENCY_PROTOCOL_REVISION,
    aggregate_batch_efficiency,
    aggregate_random_efficiency,
)


def setup_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


CONV_RANDOM_VARIANTS = {"conv_out_random", "conv_filter_random"}
CONV_SALIENCY_VARIANTS = {"conv_out_saliency", "conv_filter_saliency"}
CONV_LBI_VARIANTS = {"conv_out_lbi", "conv_filter_lbi"}
LBI_VARIANTS = {"module_lbi", *CONV_LBI_VARIANTS}
SALIENCY_VARIANTS = {"module_saliency", *CONV_SALIENCY_VARIANTS}
RANDOM_VARIANTS = {"module_random", *CONV_RANDOM_VARIANTS}

STREAM_CHECKPOINT_SCHEMA_VERSION = 1


class StreamInterrupted(RuntimeError):
    """Raised after a checkpoint has been safely written for a signal."""


class RuntimeInstrumentation:
    """Measure formal per-batch online/FO compute without touching science."""

    def __init__(self, device, runtime_comparable=False):
        self.device = device
        self.online_compute_runtime_sec = 0.0
        self.fo_eval_runtime_sec = 0.0
        self.runtime_comparable = bool(runtime_comparable)
        self.batch_efficiency_records = []
        self.runtime_resume_used = False
        self.runtime_segment_count = 1
        self._current_batch = None

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def reset_peak_memory(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def begin_batch(self, batch_index, batch_size):
        """Start a memory interval after input transfer, before adaptation."""
        self.reset_peak_memory()
        self._current_batch = {
            "batch_index": int(batch_index),
            "batch_size": int(batch_size),
            "adapt_runtime_sec": 0.0,
            "pu_runtime_sec": 0.0,
            "online_runtime_sec": 0.0,
            "lbi_stage1_runtime_sec": None,
            "lbi_stage2_runtime_sec": None,
        }

    def _start_timer(self):
        self._synchronize()
        return time.perf_counter()

    def _stop_timer(self, started_at):
        self._synchronize()
        return float(time.perf_counter() - started_at)

    def start_adaptation(self):
        return self._start_timer()

    def finish_adaptation(self, started_at):
        elapsed = self._stop_timer(started_at)
        if self._current_batch is not None:
            self._current_batch["adapt_runtime_sec"] = elapsed
        else:
            self.online_compute_runtime_sec += elapsed
        return elapsed

    def start_pu(self):
        return self._start_timer()

    def finish_pu(self, started_at):
        elapsed = self._stop_timer(started_at)
        if self._current_batch is not None:
            self._current_batch["pu_runtime_sec"] = elapsed
        return elapsed

    def start(self, name):
        """Generic synchronized timer hook used by the LBI engine."""
        del name
        return self._start_timer()

    def stop(self, name, started_at):
        elapsed = self._stop_timer(started_at)
        if self._current_batch is not None:
            self._current_batch[f"{name}_runtime_sec"] = elapsed
        return elapsed

    def finish_batch(self):
        if self._current_batch is None:
            raise RuntimeError("finish_batch called without begin_batch")
        record = dict(self._current_batch)
        record["online_runtime_sec"] = float(
            record["adapt_runtime_sec"] + record["pu_runtime_sec"]
        )
        if self.device.type == "cuda":
            allocated = int(torch.cuda.max_memory_allocated(self.device))
            reserved = int(torch.cuda.max_memory_reserved(self.device))
            record.update(
                {
                    "peak_gpu_memory_allocated_bytes": allocated,
                    "peak_gpu_memory_reserved_bytes": reserved,
                    "peak_gpu_memory_allocated_mb": float(
                        allocated / (1024**2)
                    ),
                    "peak_gpu_memory_reserved_mb": float(
                        reserved / (1024**2)
                    ),
                }
            )
        else:
            record.update(
                {
                    "peak_gpu_memory_allocated_bytes": None,
                    "peak_gpu_memory_reserved_bytes": None,
                    "peak_gpu_memory_allocated_mb": None,
                    "peak_gpu_memory_reserved_mb": None,
                }
            )
        self.batch_efficiency_records.append(record)
        self.online_compute_runtime_sec = float(
            sum(
                item["online_runtime_sec"]
                for item in self.batch_efficiency_records
            )
        )
        self._current_batch = None
        return record

    def restore_batch_efficiency(self, records):
        self.batch_efficiency_records = [dict(record) for record in records]
        self.online_compute_runtime_sec = float(
            sum(
                float(record.get("online_runtime_sec", 0.0))
                for record in self.batch_efficiency_records
            )
        )

    def start_online_step(self):
        return self._start_timer()

    def finish_online_step(self, started_at):
        elapsed = self._stop_timer(started_at)
        self.online_compute_runtime_sec += elapsed
        return elapsed

    def measure_fo(self, function):
        self._synchronize()
        started_at = time.perf_counter()
        result = function()
        self._synchronize()
        self.fo_eval_runtime_sec += time.perf_counter() - started_at
        return result

    def metadata(self):
        batch_aggregate = aggregate_batch_efficiency(
            self.batch_efficiency_records
        )
        metadata = {
            "efficiency_protocol_revision": EFFICIENCY_PROTOCOL_REVISION,
            "online_compute_runtime_sec": float(
                batch_aggregate["online_compute_runtime_sec"]
                if self.batch_efficiency_records
                else self.online_compute_runtime_sec
            ),
            "fo_eval_runtime_sec": float(self.fo_eval_runtime_sec),
            "runtime_comparable": self.runtime_comparable,
            "runtime_resume_used": bool(self.runtime_resume_used),
            "runtime_segment_count": int(self.runtime_segment_count),
            "gpu_name": None,
            "gpu_device_index": None,
            "gpu_total_memory_bytes": None,
            "peak_gpu_memory_allocated_bytes": None,
            "peak_gpu_memory_reserved_bytes": None,
            "peak_gpu_memory_allocated_mb": None,
            "peak_gpu_memory_reserved_mb": None,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }
        metadata.update(batch_aggregate)
        if not self.batch_efficiency_records:
            metadata["online_compute_runtime_sec"] = float(
                self.online_compute_runtime_sec
            )
        if self.device.type != "cuda":
            return metadata
        device_index = self.device.index
        properties = torch.cuda.get_device_properties(self.device)
        allocated = torch.cuda.max_memory_allocated(self.device)
        reserved = torch.cuda.max_memory_reserved(self.device)
        if self.batch_efficiency_records:
            allocated = max(
                int(record["peak_gpu_memory_allocated_bytes"])
                for record in self.batch_efficiency_records
                if record.get("peak_gpu_memory_allocated_bytes") is not None
            )
            reserved = max(
                int(record["peak_gpu_memory_reserved_bytes"])
                for record in self.batch_efficiency_records
                if record.get("peak_gpu_memory_reserved_bytes") is not None
            )
        metadata.update(
            {
                "gpu_name": torch.cuda.get_device_name(self.device),
                "gpu_device_index": int(
                    torch.cuda.current_device()
                    if device_index is None
                    else device_index
                ),
                "gpu_total_memory_bytes": int(properties.total_memory),
                "peak_gpu_memory_allocated_bytes": int(allocated),
                "peak_gpu_memory_reserved_bytes": int(reserved),
                "peak_gpu_memory_allocated_mb": float(allocated / (1024**2)),
                "peak_gpu_memory_reserved_mb": float(reserved / (1024**2)),
            }
        )
        return metadata


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
    online_compute_runtime_sec=0.0,
    wall_runtime_sec=None,
    batch_efficiency_records=None,
    runtime_resume_used=False,
    runtime_segment_count=1,
):
    """Atomically checkpoint only completed online steps, never LBI mid-step."""
    paths = _stream_checkpoint_paths(output_dir)
    payload = {
        "schema_version": STREAM_CHECKPOINT_SCHEMA_VERSION,
        "run_id": run_id,
        "implementation_revision": config["implementation_revision"],
        "experiment_key": config["experiment_key"],
        "experiment_config_sha256": config[
            "experiment_config_sha256"
        ],
        "iteration": int(iteration),
        "loader_batches_processed": int(loader_batches_processed),
        "started_at_utc": started_at_utc,
        "runtime_seconds": float(runtime_seconds),
        "online_compute_runtime_sec": float(online_compute_runtime_sec),
        "wall_runtime_sec": float(
            runtime_seconds if wall_runtime_sec is None else wall_runtime_sec
        ),
        "batch_efficiency_records": list(batch_efficiency_records or []),
        "runtime_resume_used": bool(runtime_resume_used),
        "runtime_segment_count": int(runtime_segment_count),
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
            "implementation_revision": config[
                "implementation_revision"
            ],
            "experiment_key": config["experiment_key"],
            "experiment_config_sha256": config[
                "experiment_config_sha256"
            ],
            "iteration": int(iteration),
            "loader_batches_processed": int(loader_batches_processed),
            "online_compute_runtime_sec": float(
                online_compute_runtime_sec
            ),
            "wall_runtime_sec": float(
                runtime_seconds if wall_runtime_sec is None else wall_runtime_sec
            ),
            "efficiency_protocol_revision": EFFICIENCY_PROTOCOL_REVISION,
            "runtime_resume_used": bool(runtime_resume_used),
            "runtime_segment_count": int(runtime_segment_count),
            "efficiency_batch_count": len(batch_efficiency_records or []),
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
    for field in (
        "implementation_revision",
        "experiment_key",
        "experiment_config_sha256",
    ):
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
    for field in (
        "implementation_revision",
        "experiment_key",
        "experiment_config_sha256",
    ):
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
    missing = [
        name
        for name in ("config", "manifest", "metrics")
        if not osp.isfile(paths[name])
    ]
    if missing:
        raise RuntimeError(
            "resume run directory is missing artifacts: "
            f"{', '.join(missing)}"
        )
    return paths


def _update_manifest_runtime(artifact_paths, runtime_metadata, wall_runtime):
    """Add final runtime metadata after the measured run has completed."""
    manifest_path = artifact_paths.get("manifest")
    if not manifest_path or not osp.isfile(manifest_path):
        return
    with open(manifest_path, "r", encoding="utf-8") as file_obj:
        manifest = json.load(file_obj)
    manifest.update(runtime_metadata)
    manifest["wall_runtime_sec"] = float(wall_runtime)
    dump_json(manifest_path, manifest)


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
        "implementation_revision": config["implementation_revision"],
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
    parameter_lookup = dict(all_named_parameters)
    found_names = set(parameter_lookup).intersection(MODULE_CANDIDATE_NAMES)
    if found_names != MODULE_CANDIDATE_NAMES:
        missing = sorted(MODULE_CANDIDATE_NAMES - found_names)
        raise RuntimeError(f"module parameters not found: {missing}")
    return [
        (name, parameter_lookup[name])
        for name in MODULE_CANDIDATE_ORDER
    ]


def _collect_conv_candidates(all_named_parameters):
    parameter_lookup = dict(all_named_parameters)
    found_names = set(parameter_lookup).intersection(CONV_CANDIDATE_NAMES)
    if found_names != CONV_CANDIDATE_NAMES:
        missing = sorted(CONV_CANDIDATE_NAMES - found_names)
        raise RuntimeError(f"Conv candidate parameters not found: {missing}")
    candidates = [
        (name, parameter_lookup[name]) for name in CONV_CANDIDATE_ORDER
    ]
    candidate_count = sum(parameter.numel() for _, parameter in candidates)
    if candidate_count != CONV_CANDIDATE_PARAM_COUNT:
        raise RuntimeError(
            "layer4 Conv candidate scalar count must be "
            f"{CONV_CANDIDATE_PARAM_COUNT}, got {candidate_count}"
        )
    for name, parameter in candidates:
        expected_shape = CONV_CANDIDATE_SHAPES[name]
        actual_shape = tuple(parameter.shape)
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"Conv candidate {name} must have shape {expected_shape}, "
                f"got {actual_shape}"
            )
    return candidates


def _candidate_offsets(named_parameters):
    """Return deterministic flatten offsets for the global candidate pool."""
    offsets = {}
    total = 0
    for name, parameter in named_parameters:
        if name in offsets:
            raise ValueError(f"duplicate candidate parameter name: {name}")
        offsets[name] = (total, total + parameter.numel())
        total += parameter.numel()
    return offsets, total


def _masks_from_global_indices(named_parameters, selected_indices):
    offsets, total = _candidate_offsets(named_parameters)
    selected = torch.zeros(total, dtype=torch.bool, device="cpu")
    if selected_indices.numel():
        selected[selected_indices.to(device="cpu", dtype=torch.long)] = True
    masks = {}
    for name, parameter in named_parameters:
        start, end = offsets[name]
        masks[name] = selected[start:end].reshape(parameter.shape).to(
            device=parameter.device
        )
    return masks


def _global_score_mask(named_parameters, score_by_name, requested_budget):
    """Select exactly K scores from the deterministic global FC pool."""
    offsets, total = _candidate_offsets(named_parameters)
    max_count = max_support_count(requested_budget, total)
    if max_count == 0:
        selected_indices = torch.empty(0, dtype=torch.long)
    else:
        global_scores = torch.cat(
            [
                score_by_name[name].detach().reshape(-1).to(device="cpu")
                for name, _ in named_parameters
            ]
        )
        selected_indices = torch.argsort(
            global_scores,
            descending=True,
            stable=True,
        )[:max_count]
    return _masks_from_global_indices(named_parameters, selected_indices)


def _build_static_random_masks(named_parameters, requested_budget, seed):
    named_parameters = list(named_parameters)
    _, candidate_count = _candidate_offsets(named_parameters)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    max_count = max_support_count(requested_budget, candidate_count)
    selected_indices = torch.randperm(
        candidate_count, generator=generator, device="cpu"
    )[:max_count]
    return _masks_from_global_indices(named_parameters, selected_indices)


def _build_static_magnitude_masks(named_parameters, requested_budget):
    named_parameters = list(named_parameters)
    return _global_score_mask(
        named_parameters,
        {
            name: parameter.detach().abs()
            for name, parameter in named_parameters
        },
        requested_budget,
    )


def _compute_saliency_score(name, parameter):
    if parameter.grad is None:
        raise RuntimeError(
            f"Cannot build saliency mask: gradient is None for {name}"
        )
    return torch.abs(parameter.detach() * parameter.grad.detach())


def _build_dynamic_saliency_masks(named_parameters, requested_budget):
    named_parameters = list(named_parameters)
    return _global_score_mask(
        named_parameters,
        {
            name: _compute_saliency_score(name, parameter)
            for name, parameter in named_parameters
        },
        requested_budget,
    )


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


def _compute_group_mask_selection_stats(
    masks,
    candidate_parameters,
    group_mode,
    candidate_scope_param_count,
    total_model_param_count,
):
    selected_groups = selected_group_count(
        masks, candidate_parameters, group_mode
    )
    total_groups = group_count(candidate_parameters, group_mode)
    selected_scalars = selected_scalar_count(masks)
    return {
        "selected_param_count": int(selected_scalars),
        "selected_over_scope_ratio": (
            float(selected_scalars / candidate_scope_param_count)
            if candidate_scope_param_count > 0
            else 0.0
        ),
        "selected_over_model_ratio": (
            float(selected_scalars / total_model_param_count)
            if total_model_param_count > 0
            else 0.0
        ),
        "selected_group_count": int(selected_groups),
        "selected_scalar_count": int(selected_scalars),
        "realized_group_ratio": (
            float(selected_groups / total_groups) if total_groups > 0 else 0.0
        ),
        "realized_scalar_ratio": (
            float(selected_scalars / candidate_scope_param_count)
            if candidate_scope_param_count > 0
            else 0.0
        ),
    }


def _summarize_dynamic_selection_stats(step_stats):
    fields = (
        "selected_param_count",
        "selected_over_scope_ratio",
        "selected_over_model_ratio",
        "selected_group_count",
        "selected_scalar_count",
        "realized_group_ratio",
        "realized_scalar_ratio",
    )
    summary = {}
    for field in fields:
        values = [step[field] for step in step_stats if field in step]
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
        "selected_group_count",
        "selected_scalar_count",
        "realized_group_ratio",
        "realized_scalar_ratio",
    )
    summary = {}
    for field in fields:
        values = [step[field] for step in step_stats if field in step]
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
            mask = mask.to(device=parameter.device, dtype=torch.bool)
            momentum_buffer = optimizer.state.get(parameter, {}).get(
                "momentum_buffer"
            )
            if momentum_buffer is not None:
                momentum_buffer.mul_(mask)
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
            momentum_buffer = optimizer.state.get(parameter, {}).get(
                "momentum_buffer"
            )
            if momentum_buffer is not None:
                momentum_buffer.mul_(mask.to(device=parameter.device))


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
    group_mode = None
    expected_selected_group_count_per_step = None
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
            expected_selected_param_count_per_step = None
        else:
            selection = "lbi"
            requested_budget = float(config["requested_budget"])
            selection_seed = None
            mask_static = False
            mask_refresh_policy = "every_online_step_via_split_lbi"
            ranking_source = "split_lbi_gamma_support"
            saliency_score = None
            expected_selected_param_count_per_step = None
    elif variant in {
        "conv_module_dense",
        "conv_out_random",
        "conv_out_magnitude",
        "conv_out_saliency",
        "conv_out_lbi",
        "conv_filter_random",
        "conv_filter_magnitude",
        "conv_filter_saliency",
        "conv_filter_lbi",
    }:
        candidate_parameters = _collect_conv_candidates(all_named_parameters)
        selected_parameters = list(candidate_parameters)
        candidate_scope = "netF.layer4_conv"
        candidate_scope_type = "layer4_conv_weights"
        group_mode = config.get("group_mode")
        bn_stats_policy = "frozen"
        bn_stats_frozen = True
        net_f.train()
        net_b.eval()
        net_c.eval()
        for module in batch_norm_modules:
            module.eval()

        if variant == "conv_module_dense":
            selection = "dense"
            requested_budget = 1.0
            group_mode = None
            selection_seed = None
            mask_static = False
            ranking_source = None
            mask_refresh_policy = None
            saliency_score = None
            expected_selected_param_count_per_step = None
        elif variant.endswith("_random"):
            selection = "random"
            requested_budget = float(config["requested_budget"])
            selection_seed = int(config["selection_seed"])
            mask_static = True
            ranking_source = "independent_random_generator"
            mask_refresh_policy = "once_before_adaptation"
            saliency_score = None
            expected_selected_param_count_per_step = None
            static_masks = build_random_group_masks(
                candidate_parameters,
                group_mode,
                requested_budget,
                selection_seed,
            )
        elif variant.endswith("_magnitude"):
            selection = "magnitude"
            requested_budget = float(config["requested_budget"])
            selection_seed = None
            mask_static = True
            ranking_source = "source_checkpoint_pre_adaptation"
            mask_refresh_policy = "once_before_adaptation"
            saliency_score = None
            expected_selected_param_count_per_step = None
            static_masks = global_group_score_mask(
                candidate_parameters,
                {
                    name: parameter.detach()
                    for name, parameter in candidate_parameters
                },
                group_mode,
                requested_budget,
            )
        elif variant.endswith("_saliency"):
            selection = "saliency"
            requested_budget = float(config["requested_budget"])
            selection_seed = None
            mask_static = False
            ranking_source = "current_parameter_times_current_gradient"
            mask_refresh_policy = "every_online_step_after_backward"
            saliency_score = "group_l2_parameter_times_gradient"
            expected_selected_param_count_per_step = None
        else:
            selection = "lbi"
            requested_budget = float(config["requested_budget"])
            selection_seed = None
            mask_static = False
            ranking_source = "split_lbi_group_gamma_support"
            mask_refresh_policy = "every_online_step_via_split_lbi"
            saliency_score = None
            expected_selected_param_count_per_step = None
    else:
        raise ValueError(f"Unsupported variant: {variant}")

    selected_names = {name for name, _ in selected_parameters}
    for name, parameter in all_named_parameters:
        parameter.requires_grad = name in selected_names

    if variant in SALIENCY_VARIANTS | LBI_VARIANTS:
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
    total_group_count = (
        group_count(candidate_parameters, group_mode)
        if candidate_scope_type == "layer4_conv_weights" and group_mode
        else None
    )
    if total_group_count is not None:
        expected_group_count = CONV_GROUP_COUNTS[group_mode]
        if total_group_count != expected_group_count:
            raise RuntimeError(
                f"Conv group pool for {group_mode} must contain "
                f"{expected_group_count} groups, got {total_group_count}"
            )
    budget_support_count = (
        max_support_count(
            requested_budget,
            total_group_count
            if total_group_count is not None
            else candidate_scope_param_count,
        )
        if requested_budget is not None
        else None
    )
    if variant == "module_saliency":
        expected_selected_param_count_per_step = budget_support_count
    elif variant in CONV_SALIENCY_VARIANTS:
        # Conv budgets groups, not scalar parameters.  Group sizes are not
        # uniform, so a scalar-count expectation would be semantically wrong.
        expected_selected_param_count_per_step = None
        expected_selected_group_count_per_step = budget_support_count
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
        "max_support_count": budget_support_count,
        "budget_semantics": (
            "global_fc_floor_integer"
            if candidate_scope_type == "fc_parameters"
            else (
                "global_conv_group_floor_integer"
                if candidate_scope_type == "layer4_conv_weights" and group_mode
                else None
            )
        ),
        "ranking_source": ranking_source,
        "mask_refresh_policy": mask_refresh_policy,
        "saliency_score": saliency_score,
        "expected_selected_param_count_per_step": (
            expected_selected_param_count_per_step
        ),
        "expected_selected_group_count_per_step": (
            expected_selected_group_count_per_step
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
        "candidate_layer_names": [name for name, _ in candidate_parameters],
        "group_mode": group_mode,
        "total_group_count": total_group_count,
        "max_group_count": budget_support_count if group_mode else None,
        "selected_group_count": (
            selected_group_count(static_masks, candidate_parameters, group_mode)
            if mask_static and group_mode
            else None
        ),
        "realized_group_ratio": (
            float(
                selected_group_count(
                    static_masks, candidate_parameters, group_mode
                )
                / total_group_count
            )
            if mask_static and group_mode and total_group_count > 0
            else None
        ),
        "selected_scalar_count": (
            int(selected_param_count)
            if selected_param_count is not None and group_mode
            else None
        ),
        "realized_scalar_ratio": (
            selected_over_scope_ratio if group_mode else None
        ),
    }
    if variant in LBI_VARIANTS:
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
                "lbi_support_threshold": float(
                    lbi["support_threshold"]
                ),
                "stage2_momentum": 0.9,
                "stage2_weight_decay": 1.0e-3,
                "stage2_nesterov": True,
                "delta_nonzero_tolerance": float(
                    lbi["delta_nonzero_tolerance"]
                ),
                "lbi_state_lifecycle": "reset_every_online_step",
                "lbi_initialization": "masked_delta",
                "stage3_mode": "accumulation",
                "stage1_branch_enabled": budget_support_count > 0,
                "target_support_count": budget_support_count,
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
        [] if variant in LBI_VARIANTS else selected_parameters
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


def _compute_overall_accuracy(labels, predictions):
    """Compute sample-level accuracy as a percentage."""
    labels = np.asarray(labels).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have the same shape")
    return float(np.mean(labels == predictions) * 100.0) if labels.size else 0.0


def _compute_dataset_metrics(labels, predictions, dataset, prefix):
    """Use overall Office accuracy and fixed-class VisDA metrics."""
    if dataset == "VISDA-C":
        from visda_otta.evaluator import compute_metrics

        return compute_metrics(labels, predictions, prefix)
    _, per_class = _compute_metrics(labels, predictions)
    accuracy = _compute_overall_accuracy(labels, predictions)
    return {f"{prefix}-Acc": accuracy, f"{prefix}-Acc-per-class": per_class}


def _snapshot_bn_buffers(*models):
    """Snapshot persistent buffers for every BatchNorm module in ``models``."""
    snapshots = []
    for model in models:
        for module in model.modules():
            if not isinstance(module, nn.modules.batchnorm._BatchNorm):
                continue
            buffers = {}
            for name in (
                "running_mean",
                "running_var",
                "num_batches_tracked",
            ):
                value = getattr(module, name)
                buffers[name] = (
                    value.detach().clone() if value is not None else None
                )
            snapshots.append(
                (module, buffers)
            )
    return snapshots


def _restore_bn_buffers(snapshots):
    """Restore BN buffers without changing module train/eval modes."""
    with torch.no_grad():
        for module, buffers in snapshots:
            for name, snapshot in buffers.items():
                current = getattr(module, name)
                if current is not None and snapshot is not None:
                    current.copy_(snapshot)


def _post_update_forward(inputs, net_f, net_b, net_c, preserve_bn_state=False):
    """Run PU measurement forward, optionally discarding its BN buffer updates."""
    bn_snapshot = (
        _snapshot_bn_buffers(net_f, net_b, net_c)
        if preserve_bn_state
        else None
    )
    try:
        with torch.no_grad():
            return net_c(net_b(net_f(inputs)))
    finally:
        if bn_snapshot is not None:
            _restore_bn_buffers(bn_snapshot)


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
    if enable_stream_checkpoint and config["variant"] not in LBI_VARIANTS:
        raise ValueError(
            "stream checkpointing is restricted to LBI variants"
        )
    if resume_run_dir and config["variant"] not in LBI_VARIANTS:
        raise ValueError("stream resume is restricted to LBI variants")
    if config.get("formal_protocol") and config["output"].get("save_model"):
        raise ValueError("formal protocol requires save_model=false")
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
    if config["variant"] in SALIENCY_VARIANTS | LBI_VARIANTS:
        masked_update_names = set(selection_stats["candidate_layer_names"])
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
    if config["variant"] in LBI_VARIANTS:
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
    online_correct = 0
    online_total = 0
    iteration = 0
    loader_batches_processed = 0
    runtime_before_resume = 0.0
    wall_runtime_before_resume = 0.0
    online_runtime_before_resume = 0.0
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
        for predictions, labels in zip(
            all_post_predictions, all_online_labels
        ):
            online_correct += int(
                torch.count_nonzero(predictions == labels).item()
            )
            online_total += int(labels.numel())
        dynamic_selection_history = resume_payload[
            "dynamic_selection_history"
        ]
        started_at_utc = resume_payload["started_at_utc"]
        runtime_before_resume = float(resume_payload["runtime_seconds"])
        wall_runtime_before_resume = float(
            resume_payload.get("wall_runtime_sec", runtime_before_resume)
        )
        online_runtime_before_resume = float(
            resume_payload.get("online_compute_runtime_sec", 0.0)
        )
        resume_rng_state = resume_payload["rng_state"]
        _truncate_metrics_to_checkpoint(artifact_paths["metrics"], iteration)
    runtime_tracker = RuntimeInstrumentation(
        device,
        runtime_comparable=(
            config.get("runtime", {}).get("runtime_comparable", False)
            and config.get("runtime", {}).get("workers_per_gpu") == 1
        ),
    )
    runtime_tracker.runtime_resume_used = resume_payload is not None
    runtime_tracker.runtime_segment_count = (
        int(resume_payload.get("runtime_segment_count", 1)) + 1
        if resume_payload is not None
        else 1
    )
    if resume_payload is not None:
        restored_records = resume_payload.get("batch_efficiency_records", [])
        if restored_records:
            runtime_tracker.restore_batch_efficiency(restored_records)
        else:
            # Checkpoints created before raw records were introduced retain
            # their legacy total rather than silently reporting zero.
            runtime_tracker.online_compute_runtime_sec = (
                online_runtime_before_resume
            )
    else:
        runtime_tracker.online_compute_runtime_sec = (
            online_runtime_before_resume
        )
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
        runtime_tracker.begin_batch(
            batch_index=loader_batch_index - 1,
            batch_size=inputs.size(0),
        )
        adaptation_started_at = runtime_tracker.start_adaptation()
        step_selection_stats = selection_stats
        if config["variant"] == "source_only":
            total_loss_value = None
            classification_loss_value = None
            entropy_loss_value = None
            diversity_loss_value = None
            pseudo_ratio = None
            max_probability_mean = None
            current_lr = None
        elif config["variant"] in LBI_VARIANTS:
            lbi_config = {
                **config["lbi"],
                "requested_budget": config["requested_budget"],
            }
            if config["variant"] in CONV_LBI_VARIANTS:
                lbi_config["group_mode"] = config["group_mode"]

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
                timing=runtime_tracker,
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
            current_lr = lbi_step_stats["stage2_lr"]
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
            if config["variant"] in SALIENCY_VARIANTS:
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
                else:
                    candidate_parameters = list(masked_parameter_lookup.items())
                    step_masks = global_group_score_mask(
                        candidate_parameters,
                        {
                            name: parameter.detach() * parameter.grad.detach()
                            for name, parameter in candidate_parameters
                        },
                        config["group_mode"],
                        config["requested_budget"],
                    )
                    dynamic_step_stats = _compute_group_mask_selection_stats(
                        step_masks,
                        candidate_parameters,
                        config["group_mode"],
                        selection_stats["candidate_scope_param_count"],
                        selection_stats["total_model_param_count"],
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

            total_loss_value = float(total_loss.item())
            classification_loss_value = loss_parts["loss_cls"]
            entropy_loss_value = loss_parts["loss_ent"]
            diversity_loss_value = loss_parts["loss_div"]
            pseudo_ratio = loss_parts["pseudo_ratio"]
            max_probability_mean = loss_parts["max_prob_mean"]
            current_lr = float(optimizer.param_groups[0]["lr"])

        runtime_tracker.finish_adaptation(adaptation_started_at)
        pu_started_at = runtime_tracker.start_pu()
        post_outputs = _post_update_forward(
            inputs,
            net_f,
            net_b,
            net_c,
            preserve_bn_state=config["variant"] == "full_dense",
        )
        post_predictions = post_outputs.argmax(dim=1)
        runtime_tracker.finish_pu(pu_started_at)
        efficiency_record = runtime_tracker.finish_batch()

        all_post_predictions.append(post_predictions.cpu())

        batch_correct = int(
            torch.count_nonzero(post_predictions == labels).item()
        )
        batch_total = int(labels.numel())
        online_correct += batch_correct
        online_total += batch_total
        post_accuracy = (
            100.0 * batch_correct / batch_total if batch_total else 0.0
        )
        metric_record = {
            "schema_version": SCHEMA_VERSION,
            "event": "online_step",
            "run_id": run_id,
            "variant": config["variant"],
            "implementation_revision": config[
                "implementation_revision"
            ],
            "protocol_revision": config.get("protocol_revision"),
            "conv_protocol_revision": config.get("conv_protocol_revision"),
            "source_checkpoint_revision": config.get(
                "source_checkpoint_revision"
            ),
            "source_checkpoints": source_paths,
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
            **{
                key: efficiency_record[key]
                for key in BATCH_EFFICIENCY_FIELDS
                if key in efficiency_record
            },
            "lbi_stage1_runtime_sec": efficiency_record.get(
                "lbi_stage1_runtime_sec"
            ),
            "lbi_stage2_runtime_sec": efficiency_record.get(
                "lbi_stage2_runtime_sec"
            ),
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
                online_compute_runtime_sec=(
                    runtime_tracker.online_compute_runtime_sec
                ),
                wall_runtime_sec=(
                    wall_runtime_before_resume
                    + time.perf_counter() - segment_started_at
                ),
                batch_efficiency_records=(
                    runtime_tracker.batch_efficiency_records
                ),
                runtime_resume_used=runtime_tracker.runtime_resume_used,
                runtime_segment_count=runtime_tracker.runtime_segment_count,
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
    if config["data"]["dataset"] != "VISDA-C":
        # The Office PU main metric is the same correct/total accumulation
        # represented by all post-update predictions, not a class average.
        pu_accuracy = (
            100.0 * online_correct / online_total
            if online_total
            else 0.0
        )
    pu_per_class = pu_metrics["PU-Acc-per-class"]

    net_f.eval()
    net_b.eval()
    fo_metrics = runtime_tracker.measure_fo(
        lambda: _evaluate(
            loaders["test"],
            net_f,
            net_b,
            net_c,
            device,
            config["data"]["dataset"],
        )
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

    wall_runtime_sec = float(
        wall_runtime_before_resume
        + time.perf_counter() - segment_started_at
    )
    runtime = wall_runtime_sec
    runtime_metadata = runtime_tracker.metadata()
    completed_at_utc = datetime.now(timezone.utc).isoformat()
    final_selection_stats = selection_stats
    if config["variant"] in SALIENCY_VARIANTS:
        final_selection_stats = {
            **selection_stats,
            **_summarize_dynamic_selection_stats(
                dynamic_selection_history
            ),
        }
    elif config["variant"] in LBI_VARIANTS:
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
            "implementation_revision": config[
                "implementation_revision"
            ],
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
            "wall_runtime_sec": wall_runtime_sec,
            **runtime_metadata,
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
        "implementation_revision": config["implementation_revision"],
        "protocol_revision": config.get("protocol_revision"),
        "conv_protocol_revision": config.get("conv_protocol_revision"),
        "source_checkpoint_revision": config.get("source_checkpoint_revision"),
        "source_checkpoints": source_paths,
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
        "wall_runtime_sec": wall_runtime_sec,
        **runtime_metadata,
        "online_steps": iteration,
        "checkpoints": checkpoint_paths,
        **visda_metrics,
        **final_selection_stats,
    }
    dump_json(artifact_paths["summary"], summary)
    _update_manifest_runtime(
        artifact_paths, runtime_metadata, wall_runtime_sec
    )
    if enable_stream_checkpoint:
        paths = _stream_checkpoint_paths(output_dir)
        _atomic_json_dump(
            paths["metadata"],
            {
                "schema_version": STREAM_CHECKPOINT_SCHEMA_VERSION,
                "status": "completed",
                "run_id": run_id,
                "implementation_revision": config[
                    "implementation_revision"
                ],
                "experiment_key": config["experiment_key"],
                "experiment_config_sha256": config[
                    "experiment_config_sha256"
                ],
                "efficiency_protocol_revision": EFFICIENCY_PROTOCOL_REVISION,
                "runtime_resume_used": runtime_tracker.runtime_resume_used,
                "runtime_segment_count": runtime_tracker.runtime_segment_count,
                "efficiency_batch_count": len(
                    runtime_tracker.batch_efficiency_records
                ),
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
                "implementation_revision": config[
                    "implementation_revision"
                ],
                "PU-Acc": result["PU-Acc"],
                "FO-Acc": result["FO-Acc"],
                "runtime": result["runtime"],
                "online_compute_runtime_sec": result.get(
                    "online_compute_runtime_sec"
                ),
                "fo_eval_runtime_sec": result.get("fo_eval_runtime_sec"),
                "wall_runtime_sec": result.get("wall_runtime_sec"),
                "efficiency_protocol_revision": result.get(
                    "efficiency_protocol_revision"
                ),
                "runtime_resume_used": result.get(
                    "runtime_resume_used", False
                ),
                "runtime_segment_count": result.get(
                    "runtime_segment_count", 1
                ),
                "online_batch_runtime_mean_sec": result.get(
                    "online_batch_runtime_mean_sec"
                ),
                "online_batch_runtime_std_sec": result.get(
                    "online_batch_runtime_std_sec"
                ),
                "online_batch_runtime_median_sec": result.get(
                    "online_batch_runtime_median_sec"
                ),
                "online_batch_runtime_p95_sec": result.get(
                    "online_batch_runtime_p95_sec"
                ),
                "adapt_batch_runtime_mean_sec": result.get(
                    "adapt_batch_runtime_mean_sec"
                ),
                "adapt_batch_runtime_std_sec": result.get(
                    "adapt_batch_runtime_std_sec"
                ),
                "adapt_runtime_total_sec": result.get(
                    "adapt_runtime_total_sec"
                ),
                "pu_batch_runtime_mean_sec": result.get(
                    "pu_batch_runtime_mean_sec"
                ),
                "pu_batch_runtime_std_sec": result.get(
                    "pu_batch_runtime_std_sec"
                ),
                "pu_runtime_total_sec": result.get(
                    "pu_runtime_total_sec"
                ),
                "peak_gpu_memory_allocated_mb": result.get(
                    "peak_gpu_memory_allocated_mb"
                ),
                "peak_gpu_memory_reserved_mb": result.get(
                    "peak_gpu_memory_reserved_mb"
                ),
                "peak_gpu_memory_allocated_bytes": result.get(
                    "peak_gpu_memory_allocated_bytes"
                ),
                "peak_gpu_memory_reserved_bytes": result.get(
                    "peak_gpu_memory_reserved_bytes"
                ),
                "gpu_peak_allocated_mean_mb": result.get(
                    "gpu_peak_allocated_mean_mb"
                ),
                "gpu_peak_reserved_mean_mb": result.get(
                    "gpu_peak_reserved_mean_mb"
                ),
                "gpu_peak_allocated_max_mb": result.get(
                    "gpu_peak_allocated_max_mb"
                ),
                "gpu_peak_reserved_max_mb": result.get(
                    "gpu_peak_reserved_max_mb"
                ),
                "gpu_name": result.get("gpu_name"),
                "runtime_comparable": result.get(
                    "runtime_comparable", False
                ),
                "started_at_utc": result["started_at_utc"],
                "completed_at_utc": result["completed_at_utc"],
                "result_path": osp.join(mask_dir, "results.json"),
                "selected_param_count": result.get("selected_param_count"),
                "selected_over_scope_ratio": result.get(
                    "selected_over_scope_ratio"
                ),
                "selected_over_model_ratio": result.get(
                    "selected_over_model_ratio"
                ),
                "selected_group_count": result.get("selected_group_count"),
                "selected_scalar_count": result.get("selected_scalar_count"),
                "realized_group_ratio": result.get("realized_group_ratio"),
                "realized_scalar_ratio": result.get("realized_scalar_ratio"),
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
    child_online_runtime = [
        item["online_compute_runtime_sec"]
        for item in mask_results
        if item["online_compute_runtime_sec"] is not None
    ]
    child_fo_runtime = [
        item["fo_eval_runtime_sec"]
        for item in mask_results
        if item["fo_eval_runtime_sec"] is not None
    ]
    child_allocated = [
        item["peak_gpu_memory_allocated_mb"]
        for item in mask_results
        if item["peak_gpu_memory_allocated_mb"] is not None
    ]
    child_reserved = [
        item["peak_gpu_memory_reserved_mb"]
        for item in mask_results
        if item["peak_gpu_memory_reserved_mb"] is not None
    ]
    child_allocated_bytes = [
        result.get("peak_gpu_memory_allocated_bytes")
        for result in mask_results
        if result.get("peak_gpu_memory_allocated_bytes") is not None
    ]
    child_reserved_bytes = [
        result.get("peak_gpu_memory_reserved_bytes")
        for result in mask_results
        if result.get("peak_gpu_memory_reserved_bytes") is not None
    ]
    random_efficiency = aggregate_random_efficiency(mask_results)
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
        "implementation_revision": config["implementation_revision"],
        "protocol_revision": config.get("protocol_revision"),
        "conv_protocol_revision": config.get("conv_protocol_revision"),
        "source_checkpoint_revision": config.get("source_checkpoint_revision"),
        "source_checkpoints": selection_summary.get("source_checkpoints"),
        "requested_budget": float(config["requested_budget"]),
        # selection_seed identifies the fixed run-level setup; individual
        # masks and their seeds are recorded separately below.
        "selection_seed": int(config["selection_seed"]),
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "runtime": total_runtime,
        "total_runtime": total_runtime,
        **random_efficiency,
        "fo_eval_runtime_sec": random_efficiency[
            "fo_eval_runtime_sec"
        ],
        "fo_eval_runtime_mask_std_sec": random_efficiency[
            "fo_eval_runtime_mask_std_sec"
        ],
        "random_total_fo_eval_runtime_sec": random_efficiency[
            "random_total_fo_eval_runtime_sec"
        ],
        "wall_runtime_sec": total_runtime,
        "peak_gpu_memory_allocated_mb": (
            max(child_allocated) if child_allocated else None
        ),
        "peak_gpu_memory_reserved_mb": (
            max(child_reserved) if child_reserved else None
        ),
        "gpu_name": selection_summary.get("gpu_name"),
        "gpu_device_index": selection_summary.get("gpu_device_index"),
        "gpu_total_memory_bytes": selection_summary.get(
            "gpu_total_memory_bytes"
        ),
        "peak_gpu_memory_allocated_bytes": (
            max(child_allocated_bytes) if child_allocated_bytes else None
        ),
        "peak_gpu_memory_reserved_bytes": (
            max(child_reserved_bytes) if child_reserved_bytes else None
        ),
        "torch_version": selection_summary.get("torch_version"),
        "cuda_version": selection_summary.get("cuda_version"),
        "runtime_comparable": all(
            item.get("runtime_comparable", False) for item in mask_results
        ),
        "efficiency_protocol_revision": config.get("runtime", {}).get(
            "efficiency_protocol_revision", EFFICIENCY_PROTOCOL_REVISION
        ),
        "runtime_resume_used": False,
        "runtime_segment_count": 1,
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
        "expected_selected_group_count_per_step",
        "candidate_scope_param_count",
        "total_model_param_count",
        "selected_over_scope_ratio",
        "selected_over_model_ratio",
        "candidate_layer_names",
        "group_mode",
        "total_group_count",
        "max_group_count",
        "selected_group_count",
        "selected_scalar_count",
        "realized_group_ratio",
        "realized_scalar_ratio",
        "bn_stats_policy",
        "bn_stats_frozen",
        "bn_module_count",
    ):
        summary[field] = selection_summary.get(field)
    if config["variant"] in CONV_RANDOM_VARIANTS:
        scalar_counts = [
            int(item["selected_scalar_count"])
            for item in mask_results
            if item.get("selected_scalar_count") is not None
        ]
        scalar_ratios = [
            float(item["realized_scalar_ratio"])
            for item in mask_results
            if item.get("realized_scalar_ratio") is not None
        ]
        group_counts = [
            int(item["selected_group_count"])
            for item in mask_results
            if item.get("selected_group_count") is not None
        ]
        group_ratios = [
            float(item["realized_group_ratio"])
            for item in mask_results
            if item.get("realized_group_ratio") is not None
        ]
        if group_counts:
            if len(set(group_counts)) != 1:
                raise RuntimeError(
                    "Conv Random child masks disagree on exact selected group count"
                )
            summary["selected_group_count"] = group_counts[0]
            summary["max_group_count"] = group_counts[0]
        if group_ratios:
            summary["realized_group_ratio"] = float(np.mean(group_ratios))
        if scalar_counts:
            scalar_count_mean = float(np.mean(scalar_counts))
            summary["selected_param_count"] = scalar_count_mean
            summary["selected_scalar_count"] = scalar_count_mean
            summary["selected_scalar_count_mean"] = scalar_count_mean
            summary["selected_scalar_count_std"] = float(
                np.std(scalar_counts, ddof=0)
            )
            summary["selected_scalar_count_min"] = int(min(scalar_counts))
            summary["selected_scalar_count_max"] = int(max(scalar_counts))
        if scalar_ratios:
            scalar_ratio_mean = float(np.mean(scalar_ratios))
            summary["selected_over_scope_ratio"] = scalar_ratio_mean
            summary["realized_scalar_ratio"] = scalar_ratio_mean
            summary["realized_scalar_ratio_mean"] = scalar_ratio_mean
            summary["realized_scalar_ratio_std"] = float(
                np.std(scalar_ratios, ddof=0)
            )
            summary["realized_scalar_ratio_min"] = float(min(scalar_ratios))
            summary["realized_scalar_ratio_max"] = float(max(scalar_ratios))
            candidate_count = summary.get("candidate_scope_param_count")
            total_model_count = summary.get("total_model_param_count")
            if candidate_count and total_model_count:
                summary["selected_over_model_ratio"] = float(
                    scalar_ratio_mean * candidate_count / total_model_count
                )

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
    if config["variant"] in RANDOM_VARIANTS:
        if resume_run_dir or enable_stream_checkpoint:
            raise ValueError(
                "stream checkpointing is not available for Random variants"
            )
        return _run_random_experiment(config, workspace_root)
    return _run_single_experiment(
        config,
        workspace_root,
        resume_run_dir=resume_run_dir,
        enable_stream_checkpoint=enable_stream_checkpoint,
    )
