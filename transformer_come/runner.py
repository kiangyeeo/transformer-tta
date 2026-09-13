"""COME-OTTA online streams for the five non-LBI variants.

Dense (``full_dense``, ``candidate_dense``), per valid outer batch, protocol
section 8::

    validate batch before all adaptation transitions
    -> eval-mode scope
    -> persistent host AdamW.zero_grad
    -> fresh current model forward
    -> stable COME objective
    -> backward
    -> exactly one optimizer.step
    -> separate read-only post-update PU

Sparse (``group_random`` child, ``group_magnitude``, ``group_saliency``),
protocol section 9: the same sequence with exactly one *strict masked* host
AdamW step.  Off-mask candidate coordinates keep their exact values and their
AdamW moments are cleared.

Constant LR, no scheduler, no Stage-2, no omega/EMA writeback, FP32 throughout.
``group_random`` runs three real children from the same source W0, one per mask
seed, and reports their mean with population std.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import yaml
from tqdm.auto import tqdm

from transformer.candidate_dense.model import frozen_named_state, hash_tensors
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
# The child-config derivation and the three-child aggregation are part of the
# SHOT substrate: identical seeds, identical mean/population-std semantics.
from transformer.group_random.runner import (
    _aggregate_children as _shot_aggregate_children,
    _child_config,
)

from .common import (
    ComeRunInvalid,
    assert_finite_gradients,
    assert_rng_unchanged,
    classifier_hash,
    come_objective_step,
    guard_online_batch,
    invalid_reason_of,
    prediction_diagnostics,
    primary_metric_name,
    rng_snapshot,
    utc_now,
)
from .config import (
    DENSE_VARIANTS,
    GROUP_RANDOM,
    GROUP_SALIENCY,
    GROUP_LBI,
    IMPLEMENTATION_REVISIONS,
    MASK_SEEDS,
    NUM_RANDOM_MASKS,
    PROTOCOL_DOCUMENT,
    budget_key,
    require_supported_variant,
)
from .data import build_target_loaders
# The end-of-stream delta diagnostic uses its own binding of the paired
# group L2 scorer: it is a read-only audit of every sparse variant, not a
# support selection, and must stay countable apart from Magnitude's single
# pre-stream scoring call.
from .groups import compute_group_l2_scores as _delta_group_l2_scores
from .groups import (
    BLOCKS,
    GROUP_KINDS,
    TOTAL_GROUPS,
    build_masks,
    compute_group_l2_scores,
    compute_group_saliency_scores,
    dynamic_mask_record,
    hash_masked_values,
    magnitude_mask_record,
    mask_record,
    select_magnitude_group_ids,
    select_saliency_group_ids,
    selected_group_ids,
)
from .model import MODEL_LOADERS
from .optimizer import assert_off_mask_adam_state_zero, strict_masked_adamw_step


# Diagnostic only: it never replaces the exact off-mask equality checks.
DELTA_NONZERO_TOLERANCE = 1.0e-12

# Parameter/buffer-hash audits around the selection perturb the measured
# adaptation cost (CUDA synchronization, cache) in ways that subtracting the
# measured audit seconds cannot fully undo, so they are OFF by default: a
# formal efficiency run carries no hash instrumentation.  The correctness
# contract is proved instead by the CPU contract tests and by the real-data
# smoke, which switch the audit on through
# ``COME_SELECTION_STATE_AUDIT_BATCHES``.  The cheap CPU/CUDA RNG comparison
# stays on for every batch.  This knob changes no numerics and is deliberately
# not part of the scientific config hash.
SELECTION_AUDIT_BATCHES = int(os.environ.get("COME_SELECTION_STATE_AUDIT_BATCHES", "0"))


# --------------------------------------------------------------------------
# Artifact, manifest and diagnostics helpers
# --------------------------------------------------------------------------


def device_from_config(config: dict) -> torch.device:
    if config["runtime"]["device"] == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda:0")
    return torch.device("cpu")


def manifest_base(
    config: dict, *, variant: str, project_root: Path, started_at: str, extra: dict
) -> dict:
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "running",
        "created_at_utc": started_at,
        "command": list(sys.argv),
        "git": _git_info(project_root),
        "protocol_revision": config["protocol_revision"],
        "protocol_document": config.get("protocol_document", PROTOCOL_DOCUMENT),
        "implementation_revision": config["implementation_revision"],
        "experiment_key": config["experiment_key"],
        "scientific_config_sha256": config["scientific_config_sha256"],
        "dataset": config["dataset"],
        "transfer": config["transfer"],
        "formal_seed": config["formal_seed"],
        "method": "come",
        "variant": variant,
        "target_labels_usage": "PU/FO metrics only",
        "adaptation": copy.deepcopy(config["adaptation"]),
        "optimization": copy.deepcopy(config["optimization"]),
        "come": copy.deepcopy(config["come"]),
        "come_objective": copy.deepcopy(config["come_objective"]),
    }
    manifest.update(extra)
    return manifest


def failure_payload(
    config: dict, *, variant: str, error: BaseException, started_at: str
) -> dict:
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "failed",
        "result_validity": "invalid",
        "invalid_reason": invalid_reason_of(error),
        "method": "come",
        "variant": variant,
        "dataset": config["dataset"],
        "transfer": config["transfer"],
        "formal_seed": config["formal_seed"],
        "experiment_key": config["experiment_key"],
        "error_type": type(error).__name__,
        "error": str(error),
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
    }


def collapse_summary(records: list[dict]) -> dict:
    """Aggregate the per-batch collapse diagnostics of one stream."""

    if not records:
        return {}
    return {
        "collapse_diagnostics_last_batch": records[-1],
        "predicted_class_count_min": min(
            row["predicted_class_count"] for row in records
        ),
        "predicted_class_count_mean": statistics.fmean(
            row["predicted_class_count"] for row in records
        ),
        "dominant_class_ratio_mean": statistics.fmean(
            row["dominant_class_ratio"] for row in records
        ),
        "dominant_class_ratio_max": max(row["dominant_class_ratio"] for row in records),
        "mean_softmax_entropy_mean": statistics.fmean(
            row["mean_softmax_entropy"] for row in records
        ),
        "come_opinion_entropy_mean": statistics.fmean(
            row["come_opinion_entropy"] for row in records
        ),
        "come_mean_uncertainty_mass_mean": statistics.fmean(
            row["come_mean_uncertainty_mass"] for row in records
        ),
    }


def gpu_memory_summary(peak_allocated: list[float], peak_reserved: list[float]) -> dict:
    return {
        "gpu_peak_allocated_mean_mb": float(statistics.fmean(peak_allocated)),
        "gpu_peak_allocated_max_mb": float(max(peak_allocated)),
        "gpu_peak_reserved_mean_mb": float(statistics.fmean(peak_reserved)),
        "gpu_peak_reserved_max_mb": float(max(peak_reserved)),
    }


def run_fo_pass(
    model,
    fo_loader,
    device: torch.device,
    config: dict,
    metrics_path: Path,
    *,
    show_progress: bool,
) -> tuple:
    """Independent read-only full-target CenterCrop evaluation.

    FO never calls the COME objective, never updates parameters and never
    consumes the online augmentation generator.
    """

    fo_meter = FixedClassMeter(config["num_classes"], config["class_names"])
    fo_indices: list[int] = []
    fo_runtimes: list[float] = []
    fo_progress = tqdm(
        fo_loader,
        desc=f"FO {config['dataset']} {config['transfer']}",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
        disable=not show_progress,
    )
    with torch.inference_mode():
        for batch_index, (images, labels, indices) in enumerate(fo_progress, start=1):
            images = images.to(device, non_blocking=True)
            _sync(device)
            fo_started = time.perf_counter()
            logits = model(images)
            _sync(device)
            fo_runtime = float(time.perf_counter() - fo_started)
            predictions = logits.argmax(dim=1).cpu()
            labels = torch.as_tensor(labels, dtype=torch.int64).cpu()
            indices = torch.as_tensor(indices, dtype=torch.int64).cpu()
            fo_meter.update(labels, predictions)
            fo_indices.extend(int(index) for index in indices.tolist())
            fo_runtimes.append(fo_runtime)
            batch_correct = int((predictions == labels).sum().item())
            _append_jsonl(
                metrics_path,
                {
                    "schema_version": ARTIFACT_SCHEMA_VERSION,
                    "event": "fo_batch",
                    "batch_index": batch_index,
                    "batch_size": int(labels.numel()),
                    "correct": batch_correct,
                    "FO-batch-overall-Acc": 100.0 * batch_correct / labels.numel(),
                    "fo_batch_runtime_sec": fo_runtime,
                    "adaptation_steps": 0,
                    "objective_call_count": 0,
                    "fo_is_read_only": True,
                },
            )
            cumulative = (
                100.0
                * fo_meter.confusion.diagonal().sum().item()
                / fo_meter.sample_count
            )
            fo_progress.set_postfix(
                samples=fo_meter.sample_count, overall=f"{cumulative:.2f}%"
            )
    fo_progress.close()
    return fo_meter, fo_indices, fo_runtimes


def named_candidates(model, names):
    parameters = dict(model.named_parameters())
    return [(name, parameters[name]) for name in names]


def mask_history_sha256(mask_hashes) -> str:
    payload = json.dumps(list(mask_hashes), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------
# Support selectors
# --------------------------------------------------------------------------


def _prepare_magnitude_support(*, model, candidates, config, checkpoint_record):
    """Score the source W0 once, on CPU in float64, before the target stream."""

    del model
    budget = config["selection"]["requested_budget"]
    scores = compute_group_l2_scores(candidates)
    group_ids = select_magnitude_group_ids(scores, budget)
    record = magnitude_mask_record(
        group_ids,
        scores=scores,
        budget=budget,
        checkpoint_sha256=checkpoint_record["sha256"],
    )
    return build_masks(candidates, group_ids), record


def _prepare_random_support(*, model, candidates, config, checkpoint_record):
    """Draw this child's exact-K support from the canonical permutation."""

    del model, checkpoint_record
    budget = config["selection"]["requested_budget"]
    group_ids = selected_group_ids(budget, config["mask_seed"])
    record = mask_record(group_ids, budget=budget, seed=config["mask_seed"])
    return build_masks(candidates, group_ids), record


def _select_support(*, candidates, config):
    """Saliency: rank |W * grad(L_COME)| and rebuild the exact-K support."""

    budget = config["selection"]["requested_budget"]
    scores = compute_group_saliency_scores(candidates)
    group_ids = select_saliency_group_ids(scores, budget)
    record = dynamic_mask_record(group_ids, scores=scores, budget=budget)
    return build_masks(candidates, group_ids), record


# --------------------------------------------------------------------------
# Dense stream
# --------------------------------------------------------------------------


def run_dense_transfer(
    config: dict,
    project_root: Path,
    *,
    variant: str,
    model_loader,
    build_target_loaders,
    show_progress: bool = True,
) -> dict:
    """Run exactly one dense COME transfer through PU and independent FO."""

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "config": output_dir / "effective_config.yaml",
        "manifest": output_dir / "manifest.json",
        "metrics": output_dir / "metrics.jsonl",
        "summary": output_dir / "summary.json",
    }
    with open(paths["config"], "w", encoding="utf-8") as file_obj:
        yaml.safe_dump(config, file_obj, sort_keys=False, allow_unicode=True)

    started_at = utc_now()
    wall_started = time.perf_counter()
    base_manifest = manifest_base(
        config,
        variant=variant,
        project_root=project_root,
        started_at=started_at,
        extra={},
    )
    _atomic_json(paths["manifest"], base_manifest)

    try:
        device = device_from_config(config)
        set_reproducibility(config["formal_seed"], config["runtime"]["deterministic"])
        model, checkpoint_record, trainable, frozen, scope_record = model_loader(
            config, device
        )
        if model.training:
            raise RuntimeError(f"{variant} COME adaptation must keep DeiT in eval mode")
        trainable_names = tuple(name for name, _ in trainable)
        trainable_name_set = set(trainable_names)
        frozen_names = {name for name, _ in frozen}
        all_parameter_names = {name for name, _ in model.named_parameters()}
        if len(trainable_name_set) != len(trainable_names):
            raise RuntimeError("Trainable parameter names must be unique")
        if trainable_name_set & frozen_names:
            raise RuntimeError("Trainable and frozen parameter sets overlap")
        if trainable_name_set | frozen_names != all_parameter_names:
            raise RuntimeError("Trainable and frozen parameter sets do not cover the model")

        model_state_before = hash_model_state(model)
        trainable_state_before = hash_tensors(trainable)
        frozen_state_before = hash_tensors(
            frozen_named_state(model, trainable_name_set)
        )
        classifier_state_before = classifier_hash(model)

        optimization = config["optimization"]
        optimizer = torch.optim.AdamW(
            [parameter for _, parameter in trainable],
            lr=float(optimization["lr"]),
            betas=tuple(float(value) for value in optimization["betas"]),
            eps=float(optimization["eps"]),
            weight_decay=float(optimization["weight_decay"]),
        )
        online_loader, fo_loader, stream_record = build_target_loaders(config)
        if online_loader.batch_size not in (None, config["batch_size"]):
            raise RuntimeError("Online loader batch size differs from frozen config")
        if fo_loader.batch_size != config["fo_batch_size"]:
            raise RuntimeError("FO loader batch size differs from frozen config")

        # Model construction may consume RNG; reset so augmentation depends
        # only on the formal seed and the fixed DataLoader worker seeds.
        set_reproducibility(config["formal_seed"], config["runtime"]["deterministic"])
        class_count = int(config["num_classes"])
        pu_meter = FixedClassMeter(config["num_classes"], config["class_names"])
        pu_indices: list[int] = []
        adapt_runtimes: list[float] = []
        pu_runtimes: list[float] = []
        online_runtimes: list[float] = []
        peak_allocated: list[float] = []
        peak_reserved: list[float] = []
        collapse_records: list[dict] = []
        objective_calls = 0
        backward_calls = 0
        optimizer_steps = 0
        scheduler_steps = 0

        progress = tqdm(
            online_loader,
            desc=f"come {variant} {config['dataset']} {config['transfer']}",
            unit="batch",
            dynamic_ncols=True,
            leave=True,
            disable=not show_progress,
        )
        for batch_index, (images, labels, indices) in enumerate(progress, start=1):
            # Before the objective, the optimizer and PU.
            guard_online_batch(int(images.shape[0]))
            images = images.to(device, non_blocking=True)
            _sync(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            adapt_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            total_loss, logits, come_diagnostics = come_objective_step(
                model, images, class_count
            )
            objective_calls += 1
            backward_calls += 1
            if any(parameter.grad is not None for _, parameter in frozen):
                raise RuntimeError("A frozen parameter received a gradient")
            assert_finite_gradients(trainable)
            optimizer.step()
            optimizer_steps += 1
            _sync(device)
            adapt_runtime = float(time.perf_counter() - adapt_started)

            if model.training:
                raise RuntimeError("Model left eval mode during online adaptation")
            if classifier_hash(model) != classifier_state_before:
                raise RuntimeError("Frozen classifier/head changed during adaptation")

            pu_started = time.perf_counter()
            with torch.inference_mode():
                post_logits = model(images)
            _sync(device)
            pu_runtime = float(time.perf_counter() - pu_started)
            predictions = post_logits.argmax(dim=1).cpu()
            labels = torch.as_tensor(labels, dtype=torch.int64).cpu()
            indices = torch.as_tensor(indices, dtype=torch.int64).cpu()
            pu_meter.update(labels, predictions)
            pu_indices.extend(int(index) for index in indices.tolist())

            # Read-only collapse diagnostics from the logits the objective
            # already produced; the objective is not re-entered.
            collapse = {
                **prediction_diagnostics(logits, class_count),
                "come_opinion_entropy": come_diagnostics["come_opinion_entropy"],
                "come_mean_uncertainty_mass": come_diagnostics[
                    "come_mean_uncertainty_mass"
                ],
            }
            collapse_records.append(collapse)

            batch_correct = int((predictions == labels).sum().item())
            online_runtime = adapt_runtime + pu_runtime
            allocated_bytes = (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            )
            reserved_bytes = (
                int(torch.cuda.max_memory_reserved(device))
                if device.type == "cuda"
                else 0
            )
            allocated_mb = allocated_bytes / 1048576.0
            reserved_mb = reserved_bytes / 1048576.0
            adapt_runtimes.append(adapt_runtime)
            pu_runtimes.append(pu_runtime)
            online_runtimes.append(online_runtime)
            peak_allocated.append(allocated_mb)
            peak_reserved.append(reserved_mb)
            _append_jsonl(
                paths["metrics"],
                {
                    "schema_version": ARTIFACT_SCHEMA_VERSION,
                    "event": "online_batch",
                    "batch_index": batch_index,
                    "batch_size": int(labels.numel()),
                    "raw_batch_size": int(labels.numel()),
                    "correct": batch_correct,
                    "PU-batch-overall-Acc": 100.0 * batch_correct / labels.numel(),
                    "loss": float(total_loss.detach().item()),
                    **come_diagnostics,
                    **collapse,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "adapt_runtime_sec": adapt_runtime,
                    "pu_runtime_sec": pu_runtime,
                    "online_runtime_sec": online_runtime,
                    "adaptation_steps": 1,
                    "objective_call_count": 1,
                    "backward_calls": 1,
                    "optimizer_step_count": 1,
                    "scheduler_step_count": 0,
                    "support_selection_count": 0,
                    "model_update_applied": True,
                    "pu_is_post_update_same_batch": True,
                    "pu_is_separate_read_only_forward": True,
                    "peak_gpu_memory_allocated_bytes": allocated_bytes,
                    "peak_gpu_memory_allocated_mb": allocated_mb,
                    "peak_gpu_memory_reserved_bytes": reserved_bytes,
                    "peak_gpu_memory_reserved_mb": reserved_mb,
                },
            )
            cumulative = (
                100.0
                * pu_meter.confusion.diagonal().sum().item()
                / pu_meter.sample_count
            )
            progress.set_postfix(
                loss=f"{float(total_loss.detach().item()):.4f}",
                samples=pu_meter.sample_count,
                pu=f"{cumulative:.2f}%",
            )
        progress.close()

        _validate_indices(
            pu_indices, stream_record["sample_count"], phase="PU", online=True
        )
        batch_count = len(online_loader)
        if not (objective_calls == backward_calls == optimizer_steps == batch_count):
            raise RuntimeError(
                "Each valid online batch must run exactly one COME objective, "
                "one backward and one host optimizer step"
            )
        if scheduler_steps != 0:
            raise RuntimeError("COME dense adaptation must not step any scheduler")
        model_state_after_stream = hash_model_state(model)
        trainable_state_after_stream = hash_tensors(
            [(name, parameter) for name, parameter in model.named_parameters()
             if name in trainable_name_set]
        )
        frozen_state_after_stream = hash_tensors(
            frozen_named_state(model, trainable_name_set)
        )
        if trainable_state_after_stream == trainable_state_before:
            raise RuntimeError("The dense COME stream did not change any trainable tensor")
        if frozen_state_after_stream != frozen_state_before:
            raise RuntimeError("A frozen parameter or model buffer changed")
        if model_state_after_stream == model_state_before:
            raise RuntimeError("The dense COME stream did not change model state")
        if classifier_hash(model) != classifier_state_before:
            raise RuntimeError("Frozen classifier/head changed during the stream")

        optimizer.zero_grad(set_to_none=True)
        model.requires_grad_(False)
        model.eval()
        fo_meter, fo_indices, fo_runtimes = run_fo_pass(
            model,
            fo_loader,
            device,
            config,
            paths["metrics"],
            show_progress=show_progress,
        )
        _validate_indices(
            fo_indices, stream_record["sample_count"], phase="FO", online=False
        )
        model_state_after_fo = hash_model_state(model)
        if model_state_after_fo != model_state_after_stream:
            raise RuntimeError("Read-only FO pass changed final model state")
        if classifier_hash(model) != classifier_state_before:
            raise RuntimeError("Frozen classifier/head changed during FO")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("FO began or ended with non-None parameter gradients")

        pu_metrics = prefixed(pu_meter.compute(config["dataset"]), "PU")
        fo_metrics = prefixed(fo_meter.compute(config["dataset"]), "FO")
        completed_at = utc_now()
        summary = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "status": "completed",
            "result_validity": "valid",
            "protocol_revision": config["protocol_revision"],
            "protocol_document": base_manifest["protocol_document"],
            "implementation_revision": config["implementation_revision"],
            "method": "come",
            "variant": variant,
            "dataset": config["dataset"],
            "source": config["source"],
            "target": config["target"],
            "transfer": config["transfer"],
            "formal_seed": config["formal_seed"],
            "experiment_key": config["experiment_key"],
            "scientific_config_sha256": config["scientific_config_sha256"],
            "primary_metric": primary_metric_name(config["dataset"]),
            "class-names": config["class_names"],
            **pu_metrics,
            **fo_metrics,
            "stream": stream_record,
            "checkpoint": checkpoint_record,
            "adaptation": copy.deepcopy(config["adaptation"]),
            "optimization": copy.deepcopy(config["optimization"]),
            "come": copy.deepcopy(config["come"]),
            "come_objective": copy.deepcopy(config["come_objective"]),
            "preprocessing": copy.deepcopy(config["preprocessing"]),
            "online_batch_size": config["batch_size"],
            "fo_batch_size": config["fo_batch_size"],
            "fo_batch_size_policy": "same_as_online_not_fc_times_three",
            "singleton_batch_policy": "fail_closed_before_all_adaptation",
            **scope_record,
            "model_state_sha256_before": model_state_before,
            "model_state_sha256_after_stream": model_state_after_stream,
            "model_state_sha256_after_fo": model_state_after_fo,
            "candidate_state_sha256_before": trainable_state_before,
            "candidate_state_sha256_after_stream": trainable_state_after_stream,
            "candidate_state_changed": True,
            "frozen_state_sha256_before": frozen_state_before,
            "frozen_state_sha256_after_stream": frozen_state_after_stream,
            "frozen_state_unchanged": True,
            "frozen_head_sha256_before": classifier_state_before,
            "frozen_head_sha256_after": classifier_hash(model),
            "frozen_head_unchanged": True,
            "adaptation_steps": batch_count,
            "objective_call_count": objective_calls,
            "backward_calls": backward_calls,
            "optimizer_step_count": optimizer_steps,
            "scheduler_step_count": scheduler_steps,
            "scheduler": "none",
            "host_optimizer_lifetime": "persistent_across_batches",
            "support_selection_count": 0,
            "native_ema_commit_count": 0,
            "omega_writeback_count": 0,
            "pu_is_post_update_same_batch": True,
            "pu_is_separate_read_only_forward": True,
            "fo_is_independent_full_target_pass": True,
            "adapted_model_saved": False,
            "stream_checkpoint_saved": False,
            "partial_resume_supported": False,
            **collapse_summary(collapse_records),
            **_runtime_stats(adapt_runtimes, "adapt_batch_runtime"),
            **_runtime_stats(pu_runtimes, "pu_batch_runtime"),
            **_runtime_stats(online_runtimes, "online_batch_runtime"),
            **_runtime_stats(fo_runtimes, "fo_batch_runtime"),
            "adapt_runtime_total_sec": float(sum(adapt_runtimes)),
            "pu_runtime_total_sec": float(sum(pu_runtimes)),
            "online_runtime_total_sec": float(sum(online_runtimes)),
            "online_compute_runtime_sec": float(sum(online_runtimes)),
            "fo_eval_runtime_sec": float(sum(fo_runtimes)),
            **gpu_memory_summary(peak_allocated, peak_reserved),
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "wall_runtime_sec": float(time.perf_counter() - wall_started),
            "environment": _environment(device),
            "output_dir": str(output_dir),
        }
        _append_jsonl(
            paths["metrics"],
            {"schema_version": ARTIFACT_SCHEMA_VERSION, "event": "final", **summary},
        )
        _atomic_json(paths["summary"], summary)
        _atomic_json(
            paths["manifest"],
            {
                **base_manifest,
                "status": "completed",
                "completed_at_utc": completed_at,
                "checkpoint": checkpoint_record,
                "stream": stream_record,
                "environment": summary["environment"],
                "frozen_state_unchanged": True,
                "frozen_head_unchanged": True,
                "adapted_model_saved": False,
            },
        )
        print(
            f"[come {variant} {config['dataset']} {config['transfer']}] "
            f"PU-Acc={summary['PU-Acc']:.4f} FO-Acc={summary['FO-Acc']:.4f}",
            flush=True,
        )
        return summary
    except BaseException as error:
        failure = failure_payload(
            config, variant=variant, error=error, started_at=started_at
        )
        _atomic_json(paths["summary"], failure)
        _atomic_json(paths["manifest"], {**base_manifest, **failure})
        raise


# --------------------------------------------------------------------------
# Sparse stream
# --------------------------------------------------------------------------


def run_sparse_transfer(
    config: dict,
    project_root: Path,
    *,
    variant: str,
    model_loader,
    build_target_loaders,
    prepare_selection=None,
    select_per_batch=None,
    selection_audit_batches: int = 0,
    show_progress: bool = True,
    manifest_extra: dict | None = None,
    summary_extra: dict | None = None,
    progress_suffix: str = "",
) -> dict:
    """Run one sparse COME condition (one support policy) through PU and FO."""

    static = prepare_selection is not None
    if static == (select_per_batch is not None):
        raise ValueError("Exactly one of prepare_selection/select_per_batch is required")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "config": output_dir / "effective_config.yaml",
        "manifest": output_dir / "manifest.json",
        "mask": output_dir / "mask.json",
        "metrics": output_dir / "metrics.jsonl",
        "summary": output_dir / "summary.json",
    }
    started_at = utc_now()
    wall_started = time.perf_counter()
    base_manifest = manifest_base(
        config,
        variant=variant,
        project_root=project_root,
        started_at=started_at,
        extra={
            "selection": copy.deepcopy(config["selection"]),
            **(manifest_extra or {}),
        },
    )
    _atomic_json(paths["manifest"], base_manifest)

    try:
        device = device_from_config(config)
        set_reproducibility(config["formal_seed"], config["runtime"]["deterministic"])
        model, checkpoint_record, candidates, frozen, scope_record = model_loader(
            config, device
        )
        if model.training:
            raise RuntimeError(f"{variant} COME adaptation must keep DeiT in eval mode")
        candidate_names = tuple(name for name, _ in candidates)
        candidate_name_set = set(candidate_names)
        frozen_names = {name for name, _ in frozen}
        all_parameter_names = {name for name, _ in model.named_parameters()}
        if len(candidate_name_set) != len(candidate_names):
            raise RuntimeError("Candidate parameter names must be unique")
        if candidate_name_set & frozen_names:
            raise RuntimeError("Candidate and frozen parameter sets overlap")
        if candidate_name_set | frozen_names != all_parameter_names:
            raise RuntimeError("Candidate and frozen parameter sets do not cover the model")

        static_masks = None
        static_record = None
        if static:
            # The only selection point for a static support: source W0,
            # before any target update.
            static_masks, static_record = prepare_selection(
                model=model,
                candidates=candidates,
                config=config,
                checkpoint_record=checkpoint_record,
            )
            if static_record["realized_group_count"] != config["selection"][
                "requested_group_count"
            ]:
                raise RuntimeError("Static selection did not realize exactly K groups")
            _atomic_json(paths["mask"], static_record)
            config = {**config, "mask_sha256": static_record["mask_sha256"]}
            if "score_vector_sha256" in static_record:
                config["score_vector_sha256"] = static_record["score_vector_sha256"]
        with open(paths["config"], "w", encoding="utf-8") as file_obj:
            yaml.safe_dump(config, file_obj, sort_keys=False, allow_unicode=True)

        model_state_before = hash_model_state(model)
        candidate_state_before = hash_tensors(candidates)
        frozen_state_before = hash_tensors(frozen_named_state(model, candidate_name_set))
        classifier_state_before = classifier_hash(model)
        # Kept on CPU: this diagnostic snapshot must not inflate the
        # reported GPU peak of the adaptation itself.
        w0_candidates = {
            name: parameter.detach().to("cpu").clone() for name, parameter in candidates
        }
        if static:
            selected_state_before = hash_masked_values(
                candidates, static_masks, selected=True
            )
            off_mask_state_before = hash_masked_values(
                candidates, static_masks, selected=False
            )

        optimization = config["optimization"]
        optimizer = torch.optim.AdamW(
            [parameter for _, parameter in candidates],
            lr=float(optimization["lr"]),
            betas=tuple(float(value) for value in optimization["betas"]),
            eps=float(optimization["eps"]),
            weight_decay=float(optimization["weight_decay"]),
        )
        online_loader, fo_loader, stream_record = build_target_loaders(config)
        if online_loader.batch_size not in (None, config["batch_size"]):
            raise RuntimeError("Online loader batch size differs from frozen config")
        if fo_loader.batch_size != config["fo_batch_size"]:
            raise RuntimeError("FO loader batch size differs from frozen config")

        # Model construction and any mask generator must not perturb target
        # augmentation RNG: every condition restarts the exact formal stream.
        set_reproducibility(config["formal_seed"], config["runtime"]["deterministic"])
        class_count = int(config["num_classes"])
        pu_meter = FixedClassMeter(config["num_classes"], config["class_names"])
        pu_indices: list[int] = []
        adapt_runtimes: list[float] = []
        adapt_runtimes_with_audit: list[float] = []
        pu_runtimes: list[float] = []
        online_runtimes: list[float] = []
        audit_runtimes: list[float] = []
        audited_batches: list[int] = []
        peak_allocated: list[float] = []
        peak_reserved: list[float] = []
        collapse_records: list[dict] = []
        mask_hashes: list[str] = []
        historical_group_ids: set[int] = set()
        selected_kind_totals = {kind: 0 for kind in GROUP_KINDS}
        selected_block_totals = {str(block): 0 for block in BLOCKS}
        mask_change_count = 0
        previous_group_ids = None
        objective_calls = 0
        backward_calls = 0
        optimizer_steps = 0
        scheduler_steps = 0
        selection_count = 0
        final_masks = static_masks

        budget_text = budget_key(config["selection"]["requested_budget"])
        progress = tqdm(
            online_loader,
            desc=(
                f"come {variant} {config['dataset']} {config['transfer']} "
                f"rho={budget_text}{progress_suffix}"
            ),
            unit="batch",
            dynamic_ncols=True,
            leave=True,
            disable=not show_progress,
        )
        for batch_index, (images, labels, indices) in enumerate(progress, start=1):
            # Before the objective, the selector, the optimizer and PU.
            guard_online_batch(int(images.shape[0]))
            images = images.to(device, non_blocking=True)
            _sync(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            adapt_started = time.perf_counter()
            audit_elapsed = 0.0
            optimizer.zero_grad(set_to_none=True)
            total_loss, logits, come_diagnostics = come_objective_step(
                model, images, class_count
            )
            objective_calls += 1
            backward_calls += 1
            if any(parameter.grad is not None for _, parameter in frozen):
                raise RuntimeError("A frozen non-candidate parameter received a gradient")
            assert_finite_gradients(candidates)

            if static:
                masks = static_masks
                selection_record = None
            else:
                # Saliency: score, select and build the mask from the current
                # state and the gradient that was just computed; the same
                # gradient is then reused for the masked step, so selection
                # must not touch the model, the RNG or the module modes.
                audit_started = time.perf_counter()
                rng_before = rng_snapshot(device)
                full_audit = batch_index <= selection_audit_batches
                if full_audit:
                    candidate_hash_before = hash_tensors(candidates)
                    frozen_hash_before = hash_tensors(
                        frozen_named_state(model, candidate_name_set)
                    )
                    training_before = bool(model.training)
                    audited_batches.append(batch_index)
                audit_elapsed += time.perf_counter() - audit_started

                masks, selection_record = select_per_batch(
                    candidates=candidates, config=config
                )
                selection_count += 1

                audit_started = time.perf_counter()
                assert_rng_unchanged(rng_before, device, stage="Saliency selection")
                if full_audit:
                    if hash_tensors(candidates) != candidate_hash_before:
                        raise ComeRunInvalid(
                            "Saliency selection changed candidate parameters",
                            invalid_reason="selection_mutated_state",
                        )
                    if (
                        hash_tensors(frozen_named_state(model, candidate_name_set))
                        != frozen_hash_before
                    ):
                        raise ComeRunInvalid(
                            "Saliency selection changed frozen parameters or buffers",
                            invalid_reason="selection_mutated_state",
                        )
                    if bool(model.training) != training_before:
                        raise ComeRunInvalid(
                            "Saliency selection changed the module mode",
                            invalid_reason="selection_mutated_state",
                        )
                audit_elapsed += time.perf_counter() - audit_started

                if selection_record["realized_group_count"] != config["selection"][
                    "requested_group_count"
                ]:
                    raise RuntimeError("Saliency selection did not realize exactly K groups")
                group_ids = tuple(selection_record["selected_group_ids"])
                if previous_group_ids is not None and group_ids != previous_group_ids:
                    mask_change_count += 1
                previous_group_ids = group_ids
                historical_group_ids.update(group_ids)
                mask_hashes.append(selection_record["mask_sha256"])
                for kind, count in selection_record["selected_by_kind"].items():
                    selected_kind_totals[kind] += int(count)
                for block, count in selection_record["selected_by_block"].items():
                    selected_block_totals[block] += int(count)
                final_masks = masks

            strict_masked_adamw_step(optimizer, candidates, masks)
            optimizer_steps += 1
            _sync(device)
            adapt_runtime_with_audit = float(time.perf_counter() - adapt_started)
            # Only correctness instrumentation is excluded: the scoring,
            # ranking, top-K and mask construction of the selector itself are
            # adaptation cost and stay inside adapt_runtime_sec.
            adapt_runtime = float(adapt_runtime_with_audit - audit_elapsed)
            adapt_runtimes_with_audit.append(adapt_runtime_with_audit)
            audit_runtimes.append(float(audit_elapsed))

            if model.training:
                raise RuntimeError("Model left eval mode during online adaptation")
            if classifier_hash(model) != classifier_state_before:
                raise RuntimeError("Frozen classifier/head changed during adaptation")

            pu_started = time.perf_counter()
            with torch.inference_mode():
                post_logits = model(images)
            _sync(device)
            pu_runtime = float(time.perf_counter() - pu_started)
            predictions = post_logits.argmax(dim=1).cpu()
            labels = torch.as_tensor(labels, dtype=torch.int64).cpu()
            indices = torch.as_tensor(indices, dtype=torch.int64).cpu()
            pu_meter.update(labels, predictions)
            pu_indices.extend(int(index) for index in indices.tolist())

            collapse = {
                **prediction_diagnostics(logits, class_count),
                "come_opinion_entropy": come_diagnostics["come_opinion_entropy"],
                "come_mean_uncertainty_mass": come_diagnostics[
                    "come_mean_uncertainty_mass"
                ],
            }
            collapse_records.append(collapse)

            batch_correct = int((predictions == labels).sum().item())
            online_runtime = adapt_runtime + pu_runtime
            allocated_bytes = (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            )
            reserved_bytes = (
                int(torch.cuda.max_memory_reserved(device))
                if device.type == "cuda"
                else 0
            )
            allocated_mb = allocated_bytes / 1048576.0
            reserved_mb = reserved_bytes / 1048576.0
            adapt_runtimes.append(adapt_runtime)
            pu_runtimes.append(pu_runtime)
            online_runtimes.append(online_runtime)
            peak_allocated.append(allocated_mb)
            peak_reserved.append(reserved_mb)
            selection_fields = (
                {
                    "requested_group_count": static_record["requested_group_count"],
                    "realized_group_count": static_record["realized_group_count"],
                    "active_candidate_scalars": static_record["active_candidate_scalars"],
                    "mask_sha256": static_record["mask_sha256"],
                }
                if static
                else dict(selection_record)
            )
            _append_jsonl(
                paths["metrics"],
                {
                    "schema_version": ARTIFACT_SCHEMA_VERSION,
                    "event": "online_batch",
                    "batch_index": batch_index,
                    "batch_size": int(labels.numel()),
                    "raw_batch_size": int(labels.numel()),
                    "correct": batch_correct,
                    "PU-batch-overall-Acc": 100.0 * batch_correct / labels.numel(),
                    "loss": float(total_loss.detach().item()),
                    **come_diagnostics,
                    **collapse,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "requested_budget": config["selection"]["requested_budget"],
                    **selection_fields,
                    "adapt_runtime_sec": adapt_runtime,
                    "adapt_runtime_with_audit_sec": adapt_runtime_with_audit,
                    "pu_runtime_sec": pu_runtime,
                    "online_runtime_sec": online_runtime,
                    "selection_audit_runtime_sec": float(audit_elapsed),
                    "adaptation_steps": 1,
                    "objective_call_count": 1,
                    "backward_calls": 1,
                    "optimizer_step_count": 1,
                    "scheduler_step_count": 0,
                    "support_selection_count": 0 if static else 1,
                    "mask_is_dynamic": not static,
                    "gradient_reused_for_masked_step": True,
                    "off_current_mask_values_restored": True,
                    "off_current_mask_adam_state_cleared": True,
                    "pu_is_post_update_same_batch": True,
                    "pu_is_separate_read_only_forward": True,
                    "peak_gpu_memory_allocated_bytes": allocated_bytes,
                    "peak_gpu_memory_allocated_mb": allocated_mb,
                    "peak_gpu_memory_reserved_bytes": reserved_bytes,
                    "peak_gpu_memory_reserved_mb": reserved_mb,
                },
            )
            cumulative = (
                100.0
                * pu_meter.confusion.diagonal().sum().item()
                / pu_meter.sample_count
            )
            progress.set_postfix(
                loss=f"{float(total_loss.detach().item()):.4f}",
                samples=pu_meter.sample_count,
                pu=f"{cumulative:.2f}%",
            )
        progress.close()

        _validate_indices(
            pu_indices, stream_record["sample_count"], phase="PU", online=True
        )
        batch_count = len(online_loader)
        if not (objective_calls == backward_calls == optimizer_steps == batch_count):
            raise RuntimeError(
                "Each valid online batch must run exactly one COME objective, "
                "one backward and one masked host optimizer step"
            )
        if scheduler_steps != 0:
            raise RuntimeError("COME sparse adaptation must not step any scheduler")
        if not static and selection_count != batch_count:
            raise RuntimeError("Dynamic selection must run exactly once per batch")
        if final_masks is None:
            raise RuntimeError("Online loader produced no batches")
        assert_off_mask_adam_state_zero(optimizer, candidates, final_masks)

        model_state_after_stream = hash_model_state(model)
        candidates_after = named_candidates(model, candidate_names)
        candidate_state_after_stream = hash_tensors(candidates_after)
        frozen_state_after_stream = hash_tensors(
            frozen_named_state(model, candidate_name_set)
        )
        if frozen_state_after_stream != frozen_state_before:
            raise RuntimeError("A frozen parameter or model buffer changed")
        if model_state_after_stream == model_state_before:
            raise RuntimeError("The sparse COME stream did not change model state")
        if classifier_hash(model) != classifier_state_before:
            raise RuntimeError("Frozen classifier/head changed during the stream")
        support_summary = {}
        if static:
            selected_state_after_stream = hash_masked_values(
                candidates_after, static_masks, selected=True
            )
            off_mask_state_after_stream = hash_masked_values(
                candidates_after, static_masks, selected=False
            )
            if selected_state_after_stream == selected_state_before:
                raise RuntimeError("No selected structural-group coordinate changed")
            if off_mask_state_after_stream != off_mask_state_before:
                raise RuntimeError("An off-mask candidate coordinate changed")
            support_summary = {
                "selection": static_record,
                "requested_budget": config["selection"]["requested_budget"],
                "requested_group_count": static_record["requested_group_count"],
                "realized_group_count": static_record["realized_group_count"],
                "active_candidate_scalars": static_record["active_candidate_scalars"],
                "selected_state_sha256_before": selected_state_before,
                "selected_state_sha256_after_stream": selected_state_after_stream,
                "selected_state_changed": True,
                "off_mask_state_sha256_before": off_mask_state_before,
                "off_mask_state_sha256_after_stream": off_mask_state_after_stream,
                "off_mask_state_unchanged": True,
            }
        else:
            selection_summary = {
                **copy.deepcopy(config["selection"]),
                "realized_group_count_per_step": config["selection"][
                    "requested_group_count"
                ],
                "active_candidate_scalars_per_step": config["selection"][
                    "active_candidate_scalars_per_step"
                ],
                "online_batch_count": batch_count,
                "support_selection_count": selection_count,
                "mask_change_count": mask_change_count,
                "mask_change_ratio": mask_change_count / max(batch_count - 1, 1),
                "unique_mask_count": len(set(mask_hashes)),
                "mask_history_sha256": mask_history_sha256(mask_hashes),
                "historical_active_group_union_count": len(historical_group_ids),
                "historical_active_group_union_ratio": len(historical_group_ids)
                / TOTAL_GROUPS,
                "selected_group_occurrences_by_kind": selected_kind_totals,
                "selected_group_occurrences_by_block": selected_block_totals,
            }
            support_summary = {
                "selection": selection_summary,
                "requested_budget": config["selection"]["requested_budget"],
                "requested_group_count": config["selection"]["requested_group_count"],
                "realized_group_count_per_step": config["selection"][
                    "requested_group_count"
                ],
                "active_candidate_scalars_per_step": config["selection"][
                    "active_candidate_scalars_per_step"
                ],
                "strict_dynamic_off_mask_value_freezing": True,
                "strict_dynamic_off_mask_adam_state_clearing": True,
                "selection_state_audit_enabled": bool(selection_audit_batches),
                "selection_state_audit_batches": audited_batches,
                "selection_rng_audit_every_batch": True,
                "selection_audit_excluded_from_adapt_runtime": True,
                "selection_audit_excludes_only_instrumentation": True,
                "selector_scoring_included_in_adapt_runtime": True,
                "selection_audit_runtime_total_sec": float(sum(audit_runtimes)),
            }

        delta_norms = _delta_group_l2_scores(
            [
                (name, parameter.detach().to("cpu") - w0_candidates[name])
                for name, parameter in candidates_after
            ]
        )
        delta_group_ids = [
            int(value)
            for value in torch.nonzero(delta_norms > DELTA_NONZERO_TOLERANCE)
            .flatten()
            .tolist()
        ]
        # The comparison basis is the HISTORICAL selected-support union, never
        # the current batch mask: with a persistent host optimizer a group
        # updated at batch t-1 keeps its delta relative to W0 at batch t even
        # though it is no longer selected.  For the static selectors the mask
        # never changes, so the union is that one mask.
        historical_support_union = (
            set(static_record["selected_group_ids"]) if static else historical_group_ids
        )
        if not set(delta_group_ids) <= historical_support_union:
            raise ComeRunInvalid(
                "A group changed relative to the source outside the historical "
                "selected-support union",
                invalid_reason="off_support_parameter_drift",
            )
        delta_scalar_count = sum(
            int(
                torch.count_nonzero(
                    (parameter.detach().to("cpu") - w0_candidates[name]).abs()
                    > DELTA_NONZERO_TOLERANCE
                ).item()
            )
            for name, parameter in candidates_after
        )
        support_summary.update(
            {
                "delta_nonzero_tolerance": DELTA_NONZERO_TOLERANCE,
                "final_delta_support_group_count": len(delta_group_ids),
                "final_delta_support_group_ids": delta_group_ids,
                "final_delta_nonzero_scalar_count": delta_scalar_count,
                "final_delta_support_basis": "historical_selected_support_union",
                "final_delta_support_within_historical_support_union": True,
                "historical_active_group_union_count": len(historical_support_union),
            }
        )
        del w0_candidates

        optimizer.zero_grad(set_to_none=True)
        model.requires_grad_(False)
        model.eval()
        fo_meter, fo_indices, fo_runtimes = run_fo_pass(
            model,
            fo_loader,
            device,
            config,
            paths["metrics"],
            show_progress=show_progress,
        )
        _validate_indices(
            fo_indices, stream_record["sample_count"], phase="FO", online=False
        )
        model_state_after_fo = hash_model_state(model)
        if model_state_after_fo != model_state_after_stream:
            raise RuntimeError("Read-only FO pass changed final model state")
        if classifier_hash(model) != classifier_state_before:
            raise RuntimeError("Frozen classifier/head changed during FO")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("FO began or ended with non-None parameter gradients")

        pu_metrics = prefixed(pu_meter.compute(config["dataset"]), "PU")
        fo_metrics = prefixed(fo_meter.compute(config["dataset"]), "FO")
        completed_at = utc_now()
        summary = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "status": "completed",
            "result_validity": "valid",
            "protocol_revision": config["protocol_revision"],
            "protocol_document": base_manifest["protocol_document"],
            "implementation_revision": config["implementation_revision"],
            "method": "come",
            "variant": variant,
            "dataset": config["dataset"],
            "source": config["source"],
            "target": config["target"],
            "transfer": config["transfer"],
            "formal_seed": config["formal_seed"],
            "experiment_key": config["experiment_key"],
            "scientific_config_sha256": config["scientific_config_sha256"],
            "primary_metric": primary_metric_name(config["dataset"]),
            "class-names": config["class_names"],
            **pu_metrics,
            **fo_metrics,
            "stream": stream_record,
            "checkpoint": checkpoint_record,
            "adaptation": copy.deepcopy(config["adaptation"]),
            "optimization": copy.deepcopy(config["optimization"]),
            "come": copy.deepcopy(config["come"]),
            "come_objective": copy.deepcopy(config["come_objective"]),
            "preprocessing": copy.deepcopy(config["preprocessing"]),
            "online_batch_size": config["batch_size"],
            "fo_batch_size": config["fo_batch_size"],
            "singleton_batch_policy": "fail_closed_before_all_adaptation",
            **scope_record,
            # Keep the authoritative support record last so a scope helper
            # cannot shadow it with the same field name.
            **support_summary,
            "model_state_sha256_before": model_state_before,
            "model_state_sha256_after_stream": model_state_after_stream,
            "model_state_sha256_after_fo": model_state_after_fo,
            "candidate_state_sha256_before": candidate_state_before,
            "candidate_state_sha256_after_stream": candidate_state_after_stream,
            "candidate_state_changed": candidate_state_after_stream
            != candidate_state_before,
            "frozen_state_sha256_before": frozen_state_before,
            "frozen_state_sha256_after_stream": frozen_state_after_stream,
            "frozen_state_unchanged": True,
            "frozen_head_sha256_before": classifier_state_before,
            "frozen_head_sha256_after": classifier_hash(model),
            "frozen_head_unchanged": True,
            "off_mask_adam_state_zero": True,
            "adaptation_steps": batch_count,
            "objective_call_count": objective_calls,
            "backward_calls": backward_calls,
            "optimizer_step_count": optimizer_steps,
            "scheduler_step_count": scheduler_steps,
            "scheduler": "none",
            "host_optimizer_lifetime": "persistent_across_batches",
            "native_ema_commit_count": 0,
            "omega_writeback_count": 0,
            "pu_is_post_update_same_batch": True,
            "pu_is_separate_read_only_forward": True,
            "fo_is_independent_full_target_pass": True,
            "adapted_model_saved": False,
            "stream_checkpoint_saved": False,
            "partial_resume_supported": False,
            **collapse_summary(collapse_records),
            **_runtime_stats(adapt_runtimes, "adapt_batch_runtime"),
            **_runtime_stats(pu_runtimes, "pu_batch_runtime"),
            **_runtime_stats(online_runtimes, "online_batch_runtime"),
            **_runtime_stats(fo_runtimes, "fo_batch_runtime"),
            **_runtime_stats(adapt_runtimes_with_audit, "adapt_batch_runtime_with_audit"),
            "adapt_runtime_total_sec": float(sum(adapt_runtimes)),
            "adapt_runtime_with_audit_total_sec": float(sum(adapt_runtimes_with_audit)),
            "runtime_instrumentation_excluded": (
                "rng_snapshot_compare_and_optional_state_hash_audit"
            ),
            "pu_runtime_total_sec": float(sum(pu_runtimes)),
            "online_compute_runtime_sec": float(sum(online_runtimes)),
            "fo_eval_runtime_sec": float(sum(fo_runtimes)),
            **gpu_memory_summary(peak_allocated, peak_reserved),
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "wall_runtime_sec": float(time.perf_counter() - wall_started),
            "environment": _environment(device),
            "output_dir": str(output_dir),
            **(summary_extra or {}),
        }
        _append_jsonl(
            paths["metrics"],
            {"schema_version": ARTIFACT_SCHEMA_VERSION, "event": "final", **summary},
        )
        _atomic_json(paths["summary"], summary)
        _atomic_json(
            paths["manifest"],
            {
                **base_manifest,
                "status": "completed",
                "completed_at_utc": completed_at,
                "checkpoint": checkpoint_record,
                "stream": stream_record,
                "mask": static_record if static else summary["selection"],
                "environment": summary["environment"],
                "frozen_state_unchanged": True,
                "off_mask_adam_state_zero": True,
                "adapted_model_saved": False,
                "stream_checkpoint_saved": False,
            },
        )
        print(
            f"[come {variant} {config['dataset']} {config['transfer']} "
            f"rho={budget_text}{progress_suffix}] "
            f"PU-Acc={summary['PU-Acc']:.4f} FO-Acc={summary['FO-Acc']:.4f}",
            flush=True,
        )
        return summary
    except BaseException as error:
        failure = failure_payload(
            config, variant=variant, error=error, started_at=started_at
        )
        failure["requested_budget"] = config["selection"]["requested_budget"]
        _atomic_json(paths["summary"], failure)
        _atomic_json(paths["manifest"], {**base_manifest, **failure})
        raise


# --------------------------------------------------------------------------
# Group-Random parent: three real children
# --------------------------------------------------------------------------


def run_mask_child(
    config: dict,
    project_root: Path,
    *,
    show_progress: bool = True,
    model_loader=None,
) -> dict:
    """Run one mask child from a fresh source model through PU and FO."""

    return run_sparse_transfer(
        config,
        project_root,
        variant=GROUP_RANDOM,
        model_loader=model_loader or MODEL_LOADERS[GROUP_RANDOM],
        build_target_loaders=build_target_loaders,
        prepare_selection=_prepare_random_support,
        show_progress=show_progress,
        manifest_extra={
            "parent_experiment_key": config["parent_experiment_key"],
            "random_mask_index": config["random_mask_index"],
            "mask_seed": config["mask_seed"],
        },
        summary_extra={
            "parent_experiment_key": config["parent_experiment_key"],
            "random_mask_index": config["random_mask_index"],
            "mask_seed": config["mask_seed"],
        },
        progress_suffix=f" mask={config['random_mask_index']}",
    )


def _aggregate_children(config: dict, children: list[dict], wall_runtime: float) -> dict:
    summary = _shot_aggregate_children(config, children, wall_runtime)
    summary["method"] = "come"
    summary["protocol_document"] = config.get("protocol_document", PROTOCOL_DOCUMENT)
    summary["implementation_revision"] = config.get(
        "implementation_revision", IMPLEMENTATION_REVISIONS[GROUP_RANDOM]
    )
    summary["result_validity"] = "valid"
    summary["come"] = copy.deepcopy(config["come"])
    summary["come_objective"] = copy.deepcopy(config["come_objective"])
    return summary


def run_random_transfer(
    config: dict,
    project_root: Path,
    *,
    show_progress: bool = True,
    child_runner=None,
) -> dict:
    """Run all three deterministic Random masks for one formal condition.

    Each child restarts from the same source W0 with a fresh optimizer and the
    same target stream; only the support seed differs.  The three children are
    executed for real - the top-level accuracy is their mean with population
    std, never a single run relabelled three times.
    """

    child_runner = child_runner or run_mask_child
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=False)
    with open(output_dir / "effective_config.yaml", "w", encoding="utf-8") as file_obj:
        yaml.safe_dump(config, file_obj, sort_keys=False, allow_unicode=True)
    started_at = utc_now()
    wall_started = time.perf_counter()
    parent_manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "running",
        "created_at_utc": started_at,
        "command": list(sys.argv),
        "git": _git_info(project_root),
        "protocol_revision": config["protocol_revision"],
        "protocol_document": config.get("protocol_document", PROTOCOL_DOCUMENT),
        "implementation_revision": config["implementation_revision"],
        "method": "come",
        "variant": GROUP_RANDOM,
        "dataset": config["dataset"],
        "transfer": config["transfer"],
        "formal_seed": config["formal_seed"],
        "experiment_key": config["experiment_key"],
        "scientific_config_sha256": config["scientific_config_sha256"],
        "selection": copy.deepcopy(config["selection"]),
        "come": copy.deepcopy(config["come"]),
        "come_objective": copy.deepcopy(config["come_objective"]),
    }
    _atomic_json(output_dir / "manifest.json", parent_manifest)
    try:
        children = []
        for mask_index, mask_seed in enumerate(MASK_SEEDS):
            child = _child_config(
                config, mask_index, mask_seed, output_dir / f"mask_{mask_index:02d}"
            )
            children.append(
                child_runner(child, project_root, show_progress=show_progress)
            )
        if len(children) != NUM_RANDOM_MASKS:
            raise RuntimeError("Random parent did not complete exactly three masks")
        completed_at = utc_now()
        summary = _aggregate_children(
            config, children, float(time.perf_counter() - wall_started)
        )
        summary["started_at_utc"] = started_at
        summary["completed_at_utc"] = completed_at
        _atomic_json(output_dir / "summary.json", summary)
        _atomic_json(
            output_dir / "manifest.json",
            {
                **parent_manifest,
                "status": "completed",
                "completed_at_utc": completed_at,
                "mask_seeds": list(MASK_SEEDS),
                "mask_sha256": [child["selection"]["mask_sha256"] for child in children],
            },
        )
        print(
            f"[come group_random {config['dataset']} {config['transfer']} "
            f"rho={budget_key(config['selection']['requested_budget'])}] "
            f"Random mean PU={summary['PU-Acc']:.4f}+-{summary['PU-Acc-mask-std']:.4f} "
            f"FO={summary['FO-Acc']:.4f}+-{summary['FO-Acc-mask-std']:.4f}",
            flush=True,
        )
        return summary
    except BaseException as error:
        failed_at = utc_now()
        failure = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "status": "failed",
            "result_validity": "invalid",
            "invalid_reason": invalid_reason_of(error),
            "method": "come",
            "variant": GROUP_RANDOM,
            "dataset": config["dataset"],
            "transfer": config["transfer"],
            "requested_budget": config["selection"]["requested_budget"],
            "experiment_key": config["experiment_key"],
            "error_type": type(error).__name__,
            "error": str(error),
            "started_at_utc": started_at,
            "completed_at_utc": failed_at,
        }
        _atomic_json(output_dir / "summary.json", failure)
        _atomic_json(output_dir / "manifest.json", {**parent_manifest, **failure})
        raise


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def run_transfer(
    config: dict,
    project_root: Path,
    *,
    show_progress: bool = True,
    model_loader=None,
    child_runner=None,
    resume: bool = False,
) -> dict:
    """Run exactly one protocol-aligned COME condition.

    The variant comes from the resolved config, so a condition cannot be run
    under a scope, support policy or stream other than the one its scientific
    hash was computed over.
    """

    variant = require_supported_variant(config["variant"])
    if child_runner is not None and variant != GROUP_RANDOM:
        raise ValueError("child_runner applies only to group_random")

    if variant == GROUP_LBI:
        from .lbi_runner import run_transfer as run_lbi_transfer

        return run_lbi_transfer(
            config,
            project_root,
            show_progress=show_progress,
            resume=resume,
            model_loader=model_loader or MODEL_LOADERS[GROUP_LBI],
        )

    if resume:
        raise ValueError("Only COME group_lbi supports stream resume")

    if variant == GROUP_RANDOM:
        if model_loader is not None:
            raise ValueError(
                "group_random loads a fresh source model per child; pass the "
                "loader through child_runner instead"
            )
        return run_random_transfer(
            config,
            project_root,
            show_progress=show_progress,
            child_runner=child_runner,
        )

    loader = model_loader or MODEL_LOADERS[variant]
    if variant in DENSE_VARIANTS:
        return run_dense_transfer(
            config,
            project_root,
            variant=variant,
            model_loader=loader,
            build_target_loaders=build_target_loaders,
            show_progress=show_progress,
        )

    if variant == GROUP_SALIENCY:
        return run_sparse_transfer(
            config,
            project_root,
            variant=variant,
            model_loader=loader,
            build_target_loaders=build_target_loaders,
            select_per_batch=_select_support,
            selection_audit_batches=SELECTION_AUDIT_BATCHES,
            show_progress=show_progress,
        )
    return run_sparse_transfer(
        config,
        project_root,
        variant=variant,
        model_loader=loader,
        build_target_loaders=build_target_loaders,
        prepare_selection=_prepare_magnitude_support,
        show_progress=show_progress,
    )


__all__ = [
    "DELTA_NONZERO_TOLERANCE",
    "SELECTION_AUDIT_BATCHES",
    "build_target_loaders",
    "mask_history_sha256",
    "named_candidates",
    "run_dense_transfer",
    "run_mask_child",
    "run_random_transfer",
    "run_sparse_transfer",
    "run_transfer",
]
