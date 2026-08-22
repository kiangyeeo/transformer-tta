"""Run one Group Split-LBI condition through online PU and final FO."""

from __future__ import annotations

import copy
import json
import os
import random
import signal
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from transformer.candidate_dense.loss import shot_loss
from transformer.group_random.groups import GROUP_KINDS, mask_sha256
from transformer.source_only.data import build_target_loaders
from transformer.source_only.metrics import FixedClassMeter, prefixed
from transformer.source_only.model import hash_model_state
from transformer.source_only.runner import (
    ARTIFACT_SCHEMA_VERSION,
    _append_jsonl,
    _atomic_json,
    _environment,
    _git_info,
    _runtime_stats,
    _sync,
    _validate_indices,
    set_reproducibility,
)

from .config import GROUP_SIZE, TOTAL_GROUPS
from .engine import (
    GroupSplitLBIEngine,
    changed_group_ids,
    group_block_counts,
    group_kind_counts,
)
from .model import frozen_named_state, hash_tensors, load_group_lbi_model


STREAM_CHECKPOINT_SCHEMA_VERSION = 1


class StreamInterrupted(RuntimeError):
    """Raised after a requested signal is checkpointed at a batch boundary."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _primary_name(dataset: str) -> str:
    return (
        "sample_overall_accuracy"
        if dataset == "office31"
        else "fixed_12_class_macro_accuracy"
    )


def _classifier_hash(model) -> str:
    return hash_tensors(model.get_classifier().state_dict().items())


def _named_candidates(model, names):
    lookup = dict(model.named_parameters())
    return [(name, lookup[name]) for name in names]


class BatchTiming:
    """Synchronized Stage-1/Stage-2 timers used inside one online batch."""

    def __init__(self, device: torch.device):
        self.device = device
        self.values: dict[str, float] = {}

    def start(self, name: str):
        _sync(self.device)
        return time.perf_counter()

    def stop(self, name: str, started: float) -> float:
        _sync(self.device)
        elapsed = float(time.perf_counter() - started)
        self.values[name] = elapsed
        return elapsed


def _capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _checkpoint_paths(output_dir: Path) -> dict[str, Path]:
    root = output_dir / ".stream_checkpoint"
    return {"root": root, "state": root / "state.pt", "metadata": root / "metadata.json"}


def _atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _save_checkpoint(
    *,
    output_dir: Path,
    config: dict,
    candidates,
    completed_batches: int,
    pu_meter: FixedClassMeter,
    pu_indices: list[int],
    histories: dict,
    started_at_utc: str,
    prior_wall_runtime_sec: float,
) -> None:
    paths = _checkpoint_paths(output_dir)
    payload = {
        "schema_version": STREAM_CHECKPOINT_SCHEMA_VERSION,
        "experiment_key": config["experiment_key"],
        "scientific_config_sha256": config["scientific_config_sha256"],
        "completed_batches": int(completed_batches),
        "started_at_utc": started_at_utc,
        "prior_wall_runtime_sec": float(prior_wall_runtime_sec),
        "candidate_parameters": {
            name: parameter.detach().cpu().clone() for name, parameter in candidates
        },
        "pu_confusion": pu_meter.confusion.clone(),
        "pu_indices": list(pu_indices),
        "histories": copy.deepcopy(histories),
        "rng_state": _capture_rng_state(),
    }
    _atomic_torch_save(paths["state"], payload)
    _atomic_json(
        paths["metadata"],
        {
            "schema_version": STREAM_CHECKPOINT_SCHEMA_VERSION,
            "status": "in_progress",
            "experiment_key": config["experiment_key"],
            "scientific_config_sha256": config["scientific_config_sha256"],
            "completed_batches": int(completed_batches),
            "updated_at_utc": _utc_now(),
        },
    )


def _load_checkpoint(output_dir: Path, config: dict, candidates) -> dict:
    paths = _checkpoint_paths(output_dir)
    if not paths["state"].is_file() or not paths["metadata"].is_file():
        raise FileNotFoundError(f"No complete stream checkpoint under {output_dir}")
    with open(paths["metadata"], "r", encoding="utf-8") as file_obj:
        metadata = json.load(file_obj)
    for field in ("experiment_key", "scientific_config_sha256"):
        if metadata.get(field) != config[field]:
            raise RuntimeError(f"Resume metadata identity mismatch for {field}")
    payload = torch.load(paths["state"], map_location="cpu", weights_only=False)
    if payload.get("schema_version") != STREAM_CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("Unsupported Group-LBI stream checkpoint schema")
    for field in ("experiment_key", "scientific_config_sha256"):
        if payload.get(field) != config[field]:
            raise RuntimeError(f"Resume payload identity mismatch for {field}")
    saved = payload.get("candidate_parameters", {})
    if set(saved) != {name for name, _ in candidates}:
        raise RuntimeError("Resume candidate tensor names differ from the frozen scope")
    with torch.no_grad():
        for name, parameter in candidates:
            value = saved[name]
            if tuple(value.shape) != tuple(parameter.shape):
                raise RuntimeError(f"Resume tensor shape mismatch for {name}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    return payload


def _remove_checkpoint(output_dir: Path) -> None:
    paths = _checkpoint_paths(output_dir)
    for name in ("state", "metadata"):
        if paths[name].exists():
            paths[name].unlink()
    if paths["root"].exists():
        paths["root"].rmdir()


def _truncate_online_metrics(path: Path, completed_batches: int) -> None:
    if not path.is_file():
        return
    retained = []
    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event") == "online_batch" and int(record["batch_index"]) <= completed_batches:
                retained.append(record)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as file_obj:
            for record in retained:
                file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _empty_histories() -> dict:
    return {
        "adapt_runtimes": [],
        "pu_runtimes": [],
        "online_runtimes": [],
        "stage1_runtimes": [],
        "stage2_runtimes": [],
        "peak_allocated": [],
        "peak_reserved": [],
        "step_records": [],
    }


def _validate_histories(histories: dict, completed_batches: int) -> None:
    expected_fields = set(_empty_histories())
    if set(histories) != expected_fields:
        raise RuntimeError("Resume history schema differs from Group-LBI v1")
    for name, values in histories.items():
        if len(values) != completed_batches:
            raise RuntimeError(
                f"Resume history {name} has {len(values)} records, expected {completed_batches}"
            )


def _selection_summary(step_records: list[dict], *, stage1_step_cap: int) -> dict:
    if not step_records:
        raise ValueError("Group-LBI selection summary requires online batches")
    realized = [int(row["realized_group_count"]) for row in step_records]
    utilization = [float(row["utilization"]) for row in step_records]
    steps = [int(row["stage1_steps_completed"]) for row in step_records]
    batch_count = len(step_records)
    utilization_ge_90_count = sum(value >= 0.90 for value in utilization)
    utilization_ge_95_count = sum(value >= 0.95 for value in utilization)
    step_cap_hit_count = sum(value == int(stage1_step_cap) for value in steps)
    step_3000_hit_count = sum(value == 3000 for value in steps)
    rollback_count = sum(bool(row["stage1_rollback_used"]) for row in step_records)
    budget_violation_count = sum(
        int(row["realized_group_count"]) > int(row["requested_group_count"])
        for row in step_records
    )
    historical = {
        int(group_id)
        for row in step_records
        for group_id in row["selected_group_ids"]
    }
    kind_occurrences = {
        kind: sum(int(row["selected_by_kind"][kind]) for row in step_records)
        for kind in GROUP_KINDS
    }
    block_occurrences = {
        str(block): sum(int(row["selected_by_block"][str(block)]) for row in step_records)
        for block in (9, 10, 11)
    }
    return {
        "online_batch_count": batch_count,
        "selected_groups_sum": sum(realized),
        "average_selected_groups": statistics.fmean(realized),
        "realized_group_count_mean": statistics.fmean(realized),
        "realized_group_count_min": min(realized),
        "realized_group_count_max": max(realized),
        "utilization_sum": sum(utilization),
        "utilization_mean": statistics.fmean(utilization),
        "utilization_min": min(utilization),
        "utilization_max": max(utilization),
        "utilization_ge_90_count": utilization_ge_90_count,
        "utilization_ge_90_rate": utilization_ge_90_count / batch_count,
        "utilization_ge_95_count": utilization_ge_95_count,
        "utilization_ge_95_rate": utilization_ge_95_count / batch_count,
        "stage1_step_cap": int(stage1_step_cap),
        "stage1_step_cap_hit_count": step_cap_hit_count,
        "stage1_step_cap_hit_rate": step_cap_hit_count / batch_count,
        "stage1_3000_step_hit_count": step_3000_hit_count,
        "stage1_3000_step_hit_rate": step_3000_hit_count / batch_count,
        "stage1_steps_sum": sum(steps),
        "stage1_steps_mean": statistics.fmean(steps),
        "stage1_steps_min": min(steps),
        "stage1_steps_max": max(steps),
        "rollback_count": rollback_count,
        "rollback_rate": rollback_count / batch_count,
        "budget_reached_count": sum(row["stage1_stop_reason"] == "budget_reached" for row in step_records),
        "max_steps_reached_count": sum(row["stage1_stop_reason"] == "max_steps_reached" for row in step_records),
        "budget_violation_count": budget_violation_count,
        "budget_violation_rate": budget_violation_count / batch_count,
        "failure_batch_count": 0,
        "failure_rate": 0.0,
        "historical_active_group_union_count": len(historical),
        "historical_active_group_union_ratio": len(historical) / TOTAL_GROUPS,
        "historical_active_group_ids": sorted(historical),
        "selected_group_occurrences_by_kind": kind_occurrences,
        "selected_group_occurrences_by_block": block_occurrences,
        "final_batch_selected_group_ids": list(step_records[-1]["selected_group_ids"]),
        "final_batch_selected_by_kind": copy.deepcopy(step_records[-1]["selected_by_kind"]),
    }


def run_transfer(
    config: dict,
    project_root: Path,
    *,
    show_progress: bool = True,
    resume: bool = False,
    model_loader=load_group_lbi_model,
) -> dict:
    """Run exactly one transfer/budget Group-LBI condition."""
    output_dir = Path(config["output_dir"])
    if resume:
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Resume directory does not exist: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "config": output_dir / "effective_config.yaml",
        "manifest": output_dir / "manifest.json",
        "metrics": output_dir / "metrics.jsonl",
        "summary": output_dir / "summary.json",
    }
    if not resume:
        with open(paths["config"], "w", encoding="utf-8") as file_obj:
            yaml.safe_dump(config, file_obj, sort_keys=False, allow_unicode=True)

    segment_started = time.perf_counter()
    started_at = _utc_now()
    base_manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "running",
        "created_at_utc": started_at,
        "command": list(sys.argv),
        "git": _git_info(project_root),
        "protocol_revision": config["protocol_revision"],
        "experiment_key": config["experiment_key"],
        "scientific_config_sha256": config["scientific_config_sha256"],
        "dataset": config["dataset"],
        "transfer": config["transfer"],
        "formal_seed": config["formal_seed"],
        "method": "shot",
        "variant": "group_lbi",
        "target_labels_usage": "PU/FO metrics only; never passed to loss or engine",
        "adaptation": copy.deepcopy(config["adaptation"]),
        "selection": copy.deepcopy(config["selection"]),
        "lbi": copy.deepcopy(config["lbi"]),
        "stage2_optimization": copy.deepcopy(config["stage2_optimization"]),
        "loss": copy.deepcopy(config["loss"]),
        "resume_used": bool(resume),
    }
    _atomic_json(paths["manifest"], base_manifest)

    failure_context = {
        "phase": "setup",
        "batch_index": None,
        "completed_batches": 0,
        "target_batch_count": None,
    }

    previous_handlers = {}
    interruption = {"signum": None}

    def request_interruption(signum, _frame):
        interruption["signum"] = int(signum)

    try:
        if config["runtime"]["device"] == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable")
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        set_reproducibility(config["formal_seed"], config["runtime"]["deterministic"])
        model, checkpoint_record, candidates, frozen, scope_record = model_loader(config, device)
        if model.training:
            raise RuntimeError("Group-LBI adaptation must keep DeiT in eval mode")
        candidate_names = tuple(name for name, _ in candidates)
        candidate_name_set = set(candidate_names)
        frozen_names = {name for name, _ in frozen}
        all_parameter_names = {name for name, _ in model.named_parameters()}
        if candidate_name_set & frozen_names or candidate_name_set | frozen_names != all_parameter_names:
            raise RuntimeError("Candidate/frozen scopes do not exactly partition the model")

        source_model_hash = hash_model_state(model)
        source_candidate_hash = hash_tensors(candidates)
        source_candidate_values = _clone_candidate_values(candidates)
        frozen_hash_before = hash_tensors(frozen_named_state(model, candidate_name_set))
        classifier_hash_before = _classifier_hash(model)
        online_loader, fo_loader, stream_record = build_target_loaders(config)
        failure_context["target_batch_count"] = len(online_loader)
        set_reproducibility(config["formal_seed"], config["runtime"]["deterministic"])

        pu_meter = FixedClassMeter(config["num_classes"], config["class_names"])
        pu_indices: list[int] = []
        histories = _empty_histories()
        completed_batches = 0
        prior_wall_runtime = 0.0
        resume_rng_state = None
        if resume:
            payload = _load_checkpoint(output_dir, config, candidates)
            completed_batches = int(payload["completed_batches"])
            failure_context["completed_batches"] = completed_batches
            if not 0 <= completed_batches <= len(online_loader):
                raise RuntimeError("Resume batch position is outside the target stream")
            pu_meter.confusion.copy_(payload["pu_confusion"])
            pu_indices = [int(value) for value in payload["pu_indices"]]
            histories = payload["histories"]
            _validate_histories(histories, completed_batches)
            started_at = str(payload["started_at_utc"])
            prior_wall_runtime = float(payload.get("prior_wall_runtime_sec", 0.0))
            resume_rng_state = payload["rng_state"]
            _truncate_online_metrics(paths["metrics"], completed_batches)

        if config["checkpointing"]["enabled"]:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, request_interruption)

        engine = GroupSplitLBIEngine()
        progress = tqdm(
            online_loader,
            desc=(
                f"group-lbi {config['dataset']} {config['transfer']} "
                f"rho={config['selection']['requested_budget']:.3f}"
            ),
            unit="batch",
            dynamic_ncols=True,
            leave=True,
            disable=not show_progress,
        )
        refresh_steps = int(config["diagnostics"]["progress_refresh_steps"])
        for batch_index, (images, labels, indices) in enumerate(progress, start=1):
            if batch_index <= completed_batches:
                continue
            if resume_rng_state is not None:
                _restore_rng_state(resume_rng_state)
                resume_rng_state = None
            failure_context["phase"] = "online_adaptation"
            failure_context["batch_index"] = batch_index
            images = images.to(device, non_blocking=True)
            _sync(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            timing = BatchTiming(device)

            def progress_callback(step: int, support: int, reason: str) -> None:
                if not show_progress:
                    return
                if step == 1 or step % refresh_steps == 0 or reason != "running":
                    progress.set_postfix(
                        stage1=step,
                        groups=f"{support}/{config['selection']['requested_group_count']}",
                        state=reason,
                        refresh=True,
                    )

            def loss_closure():
                return shot_loss(model(images), config["loss"])

            engine_config = {
                **config["lbi"],
                "requested_group_count": config["selection"]["requested_group_count"],
                "stage2_optimization": config["stage2_optimization"],
                "delta_nonzero_tolerance": config["diagnostics"]["delta_nonzero_tolerance"],
            }
            adapt_started = time.perf_counter()
            result = engine.run_batch(
                candidates,
                loss_closure,
                engine_config,
                timing=timing,
                progress_callback=progress_callback,
            )
            _sync(device)
            adapt_runtime = float(time.perf_counter() - adapt_started)
            step_record = copy.deepcopy(result.statistics)
            if step_record["realized_group_count"] > step_record["requested_group_count"]:
                raise RuntimeError("Group-LBI realized support exceeded K")
            if model.training:
                raise RuntimeError("Model left eval mode during Group-LBI")
            if _classifier_hash(model) != classifier_hash_before:
                raise RuntimeError("Frozen classifier changed during Group-LBI")

            pu_started = time.perf_counter()
            failure_context["phase"] = "online_pu"
            with torch.inference_mode():
                post_logits = model(images)
            _sync(device)
            pu_runtime = float(time.perf_counter() - pu_started)
            predictions = post_logits.argmax(dim=1).cpu()
            labels_cpu = torch.as_tensor(labels, dtype=torch.int64).cpu()
            indices_cpu = torch.as_tensor(indices, dtype=torch.int64).cpu()
            pu_meter.update(labels_cpu, predictions)
            pu_indices.extend(int(index) for index in indices_cpu.tolist())
            batch_correct = int((predictions == labels_cpu).sum().item())
            online_runtime = adapt_runtime + pu_runtime
            allocated_mb = (
                torch.cuda.max_memory_allocated(device) / 1048576.0
                if device.type == "cuda"
                else 0.0
            )
            reserved_mb = (
                torch.cuda.max_memory_reserved(device) / 1048576.0
                if device.type == "cuda"
                else 0.0
            )
            stage1_runtime = float(timing.values.get("lbi_stage1", 0.0))
            stage2_runtime = float(timing.values.get("lbi_stage2", 0.0))
            histories["adapt_runtimes"].append(adapt_runtime)
            histories["pu_runtimes"].append(pu_runtime)
            histories["online_runtimes"].append(online_runtime)
            histories["stage1_runtimes"].append(stage1_runtime)
            histories["stage2_runtimes"].append(stage2_runtime)
            histories["peak_allocated"].append(allocated_mb)
            histories["peak_reserved"].append(reserved_mb)
            histories["step_records"].append(step_record)
            completed_batches = batch_index
            failure_context["completed_batches"] = completed_batches
            failure_context["phase"] = "online_bookkeeping"

            _append_jsonl(
                paths["metrics"],
                {
                    "schema_version": ARTIFACT_SCHEMA_VERSION,
                    "event": "online_batch",
                    "batch_index": batch_index,
                    "batch_size": int(labels_cpu.numel()),
                    "correct": batch_correct,
                    "PU-batch-overall-Acc": 100.0 * batch_correct / labels_cpu.numel(),
                    **step_record,
                    "mask_sha256": mask_sha256(step_record["selected_group_ids"]),
                    "adapt_runtime_sec": adapt_runtime,
                    "lbi_stage1_runtime_sec": stage1_runtime,
                    "lbi_stage2_runtime_sec": stage2_runtime,
                    "pu_runtime_sec": pu_runtime,
                    "online_runtime_sec": online_runtime,
                    "pu_is_post_update_same_batch": True,
                    "pu_is_separate_read_only_forward": True,
                    "peak_gpu_memory_allocated_mb": allocated_mb,
                    "peak_gpu_memory_reserved_mb": reserved_mb,
                },
            )
            cumulative = 100.0 * pu_meter.confusion.diagonal().sum().item() / pu_meter.sample_count
            progress.set_postfix(
                steps=step_record["stage1_steps_completed"],
                groups=f"{step_record['realized_group_count']}/{step_record['requested_group_count']}",
                pu=f"{cumulative:.2f}%",
            )

            if config["checkpointing"]["enabled"]:
                _save_checkpoint(
                    output_dir=output_dir,
                    config=config,
                    candidates=candidates,
                    completed_batches=completed_batches,
                    pu_meter=pu_meter,
                    pu_indices=pu_indices,
                    histories=histories,
                    started_at_utc=started_at,
                    prior_wall_runtime_sec=(
                        prior_wall_runtime + time.perf_counter() - segment_started
                    ),
                )
            if interruption["signum"] is not None:
                raise StreamInterrupted(
                    f"signal {interruption['signum']} handled after batch {batch_index} checkpoint"
                )
        progress.close()
        failure_context["phase"] = "post_stream_validation"

        _validate_indices(pu_indices, stream_record["sample_count"], phase="PU", online=True)
        if completed_batches != len(online_loader):
            raise RuntimeError("Online stream did not complete every batch")
        frozen_hash_after_stream = hash_tensors(frozen_named_state(model, candidate_name_set))
        if frozen_hash_after_stream != frozen_hash_before:
            raise RuntimeError("A frozen parameter or model buffer changed")

        current_candidates = _named_candidates(model, candidate_names)
        candidate_hash_after_stream = hash_tensors(current_candidates)
        model_hash_after_stream = hash_model_state(model)
        final_source_relative_groups = changed_group_ids(
            dict(current_candidates),
            source_candidate_values,
            float(config["diagnostics"]["delta_nonzero_tolerance"]),
        )

        for parameter in model.parameters():
            parameter.grad = None
        model.requires_grad_(False)
        model.eval()
        fo_meter = FixedClassMeter(config["num_classes"], config["class_names"])
        fo_indices: list[int] = []
        fo_runtimes: list[float] = []
        fo_progress = tqdm(
            fo_loader,
            desc=(
                f"FO {config['dataset']} {config['transfer']} "
                f"rho={config['selection']['requested_budget']:.3f}"
            ),
            unit="batch",
            dynamic_ncols=True,
            leave=True,
            disable=not show_progress,
        )
        with torch.inference_mode():
            failure_context["phase"] = "fo"
            for batch_index, (images, labels, indices) in enumerate(fo_progress, start=1):
                images = images.to(device, non_blocking=True)
                _sync(device)
                fo_started = time.perf_counter()
                logits = model(images)
                _sync(device)
                fo_runtime = float(time.perf_counter() - fo_started)
                predictions = logits.argmax(dim=1).cpu()
                labels_cpu = torch.as_tensor(labels, dtype=torch.int64).cpu()
                indices_cpu = torch.as_tensor(indices, dtype=torch.int64).cpu()
                fo_meter.update(labels_cpu, predictions)
                fo_indices.extend(int(index) for index in indices_cpu.tolist())
                fo_runtimes.append(fo_runtime)
                batch_correct = int((predictions == labels_cpu).sum().item())
                _append_jsonl(
                    paths["metrics"],
                    {
                        "schema_version": ARTIFACT_SCHEMA_VERSION,
                        "event": "fo_batch",
                        "batch_index": batch_index,
                        "batch_size": int(labels_cpu.numel()),
                        "correct": batch_correct,
                        "FO-batch-overall-Acc": 100.0 * batch_correct / labels_cpu.numel(),
                        "fo_batch_runtime_sec": fo_runtime,
                        "adaptation_steps": 0,
                        "fo_is_read_only": True,
                    },
                )
                cumulative = 100.0 * fo_meter.confusion.diagonal().sum().item() / fo_meter.sample_count
                fo_progress.set_postfix(samples=fo_meter.sample_count, overall=f"{cumulative:.2f}%")
        fo_progress.close()

        _validate_indices(fo_indices, stream_record["sample_count"], phase="FO", online=False)
        model_hash_after_fo = hash_model_state(model)
        if model_hash_after_fo != model_hash_after_stream:
            raise RuntimeError("Read-only FO pass changed final model state")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("FO ended with non-None parameter gradients")

        pu_metrics = prefixed(pu_meter.compute(config["dataset"]), "PU")
        fo_metrics = prefixed(fo_meter.compute(config["dataset"]), "FO")
        failure_context["phase"] = "finalize"
        selection_summary = _selection_summary(
            histories["step_records"],
            stage1_step_cap=int(config["lbi"]["stage1_max_steps"]),
        )
        final_group_record = {
            "group_count": len(final_source_relative_groups),
            "group_ratio": len(final_source_relative_groups) / TOTAL_GROUPS,
            "group_ids": list(final_source_relative_groups),
            "by_kind": group_kind_counts(final_source_relative_groups),
            "by_block": group_block_counts(final_source_relative_groups),
        }
        completed_at = _utc_now()
        summary = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "status": "completed",
            "protocol_revision": config["protocol_revision"],
            "method": "shot",
            "variant": "group_lbi",
            "dataset": config["dataset"],
            "source": config["source"],
            "target": config["target"],
            "transfer": config["transfer"],
            "formal_seed": config["formal_seed"],
            "experiment_key": config["experiment_key"],
            "scientific_config_sha256": config["scientific_config_sha256"],
            "primary_metric": _primary_name(config["dataset"]),
            "class-names": config["class_names"],
            **pu_metrics,
            **fo_metrics,
            "stream": stream_record,
            "checkpoint": checkpoint_record,
            "adaptation": copy.deepcopy(config["adaptation"]),
            "selection": {**copy.deepcopy(config["selection"]), **selection_summary},
            "lbi": copy.deepcopy(config["lbi"]),
            "stage2_optimization": copy.deepcopy(config["stage2_optimization"]),
            "loss": copy.deepcopy(config["loss"]),
            "preprocessing": copy.deepcopy(config["preprocessing"]),
            **scope_record,
            "requested_budget": config["selection"]["requested_budget"],
            "requested_group_count": config["selection"]["requested_group_count"],
            "final_source_relative_delta_support": final_group_record,
            "source_model_state_sha256": source_model_hash,
            "source_candidate_state_sha256": source_candidate_hash,
            "candidate_state_sha256_after_stream": candidate_hash_after_stream,
            "candidate_state_changed": candidate_hash_after_stream != source_candidate_hash,
            "model_state_sha256_after_stream": model_hash_after_stream,
            "model_state_sha256_after_fo": model_hash_after_fo,
            "frozen_state_sha256_before": frozen_hash_before,
            "frozen_state_sha256_after_stream": frozen_hash_after_stream,
            "frozen_state_unchanged": True,
            "frozen_head_sha256_before": classifier_hash_before,
            "frozen_head_sha256_after": _classifier_hash(model),
            "frozen_head_unchanged": _classifier_hash(model) == classifier_hash_before,
            "pu_is_separate_post_update_read_only_forward": True,
            "fo_is_independent_full_target_pass": True,
            "tail_batch_retained": True,
            "stream_checkpoint_enabled": config["checkpointing"]["enabled"],
            "runtime_resume_used": bool(resume),
            **_runtime_stats(histories["adapt_runtimes"], "adapt_batch_runtime"),
            **_runtime_stats(histories["stage1_runtimes"], "lbi_stage1_batch_runtime"),
            **_runtime_stats(histories["stage2_runtimes"], "lbi_stage2_batch_runtime"),
            **_runtime_stats(histories["pu_runtimes"], "pu_batch_runtime"),
            **_runtime_stats(histories["online_runtimes"], "online_batch_runtime"),
            **_runtime_stats(fo_runtimes, "fo_batch_runtime"),
            "online_compute_runtime_sec": float(sum(histories["online_runtimes"])),
            "adapt_runtime_total_sec": float(sum(histories["adapt_runtimes"])),
            "lbi_stage1_runtime_total_sec": float(sum(histories["stage1_runtimes"])),
            "lbi_stage2_runtime_total_sec": float(sum(histories["stage2_runtimes"])),
            "pu_runtime_total_sec": float(sum(histories["pu_runtimes"])),
            "fo_eval_runtime_sec": float(sum(fo_runtimes)),
            "gpu_peak_allocated_mean_mb": float(statistics.fmean(histories["peak_allocated"])),
            "gpu_peak_allocated_max_mb": float(max(histories["peak_allocated"])),
            "gpu_peak_reserved_mean_mb": float(statistics.fmean(histories["peak_reserved"])),
            "gpu_peak_reserved_max_mb": float(max(histories["peak_reserved"])),
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "wall_runtime_sec": float(
                prior_wall_runtime + time.perf_counter() - segment_started
            ),
            "environment": _environment(device),
            "output_dir": str(output_dir),
        }
        _append_jsonl(paths["metrics"], {"schema_version": ARTIFACT_SCHEMA_VERSION, "event": "final", **summary})
        _atomic_json(paths["summary"], summary)
        _atomic_json(
            paths["manifest"],
            {
                **base_manifest,
                "status": "completed",
                "completed_at_utc": completed_at,
                "checkpoint": checkpoint_record,
                "stream": stream_record,
                "selection_summary": selection_summary,
                "environment": summary["environment"],
                "adapted_model_saved": False,
                "stream_checkpoint_removed_after_completion": bool(
                    config["checkpointing"]["enabled"]
                ),
            },
        )
        _remove_checkpoint(output_dir)
        print(
            f"[{config['dataset']} {config['transfer']} "
            f"rho={config['selection']['requested_budget']:.3f}] "
            f"PU-Acc={summary['PU-Acc']:.4f} FO-Acc={summary['FO-Acc']:.4f}",
            flush=True,
        )
        print(
            "  tuning: "
            f"batches={selection_summary['online_batch_count']} "
            f"util(mean/min)={selection_summary['utilization_mean']:.4f}/"
            f"{selection_summary['utilization_min']:.4f} "
            f"u90/u95={selection_summary['utilization_ge_90_rate']:.4f}/"
            f"{selection_summary['utilization_ge_95_rate']:.4f} "
            f"selected-mean={selection_summary['average_selected_groups']:.2f} "
            f"steps(mean/max)={selection_summary['stage1_steps_mean']:.1f}/"
            f"{selection_summary['stage1_steps_max']} "
            f"hit3000={selection_summary['stage1_3000_step_hit_rate']:.4f} "
            f"cap={selection_summary['stage1_step_cap']} "
            f"cap-hit={selection_summary['stage1_step_cap_hit_rate']:.4f} "
            f"rollback={selection_summary['rollback_rate']:.4f} "
            f"violation={selection_summary['budget_violation_rate']:.4f} "
            f"failure={selection_summary['failure_rate']:.4f}",
            flush=True,
        )
        return summary
    except StreamInterrupted as error:
        interrupted = {
            **base_manifest,
            "status": "interrupted",
            "completed_at_utc": _utc_now(),
            "error_type": type(error).__name__,
            "error": str(error),
            "stream_checkpoint_retained": True,
        }
        _atomic_json(paths["manifest"], interrupted)
        _atomic_json(paths["summary"], interrupted)
        raise
    except BaseException as error:
        failed_online_batch = failure_context["phase"] in {
            "online_adaptation",
            "online_pu",
        }
        target_batch_count = failure_context["target_batch_count"]
        failure_batch_count = int(failed_online_batch)
        failure_rate = (
            failure_batch_count / int(target_batch_count)
            if target_batch_count
            else None
        )
        failure_diagnostics = {
            "failure_phase": failure_context["phase"],
            "failure_batch_index": failure_context["batch_index"],
            "completed_online_batch_count": failure_context["completed_batches"],
            "target_batch_count": target_batch_count,
            "failure_batch_count": failure_batch_count,
            "failure_rate": failure_rate,
            "failure_is_nan_or_inf": isinstance(error, FloatingPointError),
        }
        if failed_online_batch:
            _append_jsonl(
                paths["metrics"],
                {
                    "schema_version": ARTIFACT_SCHEMA_VERSION,
                    "event": "online_batch_failure",
                    "batch_index": failure_context["batch_index"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                    **failure_diagnostics,
                },
            )
        failure = {
            **base_manifest,
            "status": "failed",
            "completed_at_utc": _utc_now(),
            "error_type": type(error).__name__,
            "error": str(error),
            "tuning_diagnostics": failure_diagnostics,
            "stream_checkpoint_retained": _checkpoint_paths(output_dir)["state"].is_file(),
        }
        _atomic_json(paths["manifest"], failure)
        _atomic_json(paths["summary"], failure)
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def _clone_candidate_values(candidates) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in candidates}


__all__ = ["BatchTiming", "StreamInterrupted", "run_transfer"]
