"""Unified non-LBI IST outer-batch state machine for DeiT-S."""

from __future__ import annotations

import copy
import hashlib
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from transformer.group_magnitude.groups import (
    compute_group_l2_scores,
    magnitude_mask_record,
    select_magnitude_group_ids,
)
from transformer.group_random.config import (
    MASK_SEEDS,
    NUM_RANDOM_MASKS,
)
from transformer.group_random.groups import (
    build_masks,
    mask_record,
    selected_group_ids,
)
from transformer.group_random.optimizer import (
    assert_off_mask_adam_state_zero,
    strict_masked_adamw_step,
)
from transformer.group_saliency.groups import (
    compute_group_saliency_scores,
    dynamic_mask_record,
    select_saliency_group_ids,
)
from transformer.source_only.metrics import FixedClassMeter, prefixed
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

from .config import (
    GROUP_MAGNITUDE,
    GROUP_RANDOM,
    GROUP_SALIENCY,
    SPARSE_VARIANTS,
    canonical_sha256,
)
from .data import ISTViewMaterializer, build_ist_loaders
from .ema import OuterBatchEMA
from .memory import CausalMemoryBank
from .model import (
    frozen_named_state,
    hash_model_state,
    hash_tensors,
    load_ist_model,
)
from .objective import FixedISTTask, accumulate_full_objective, ist_loss
from .plca import robust_plca


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _primary_name(dataset: str) -> str:
    return (
        "sample_overall_accuracy"
        if dataset == "office31"
        else "fixed_12_class_macro_accuracy"
    )


def _class_diagnostics(probabilities, hard_targets, num_classes: int) -> dict:
    probabilities = probabilities.detach().to(device="cpu", dtype=torch.float32)
    hard_targets = hard_targets.detach().to(device="cpu", dtype=torch.long)
    if probabilities.ndim != 2 or probabilities.shape[0] != hard_targets.numel():
        raise ValueError("diagnostic probabilities/targets do not align")
    predictions = probabilities.argmax(dim=1)
    predicted_histogram = torch.bincount(
        predictions, minlength=int(num_classes)
    )
    corrected_histogram = torch.bincount(
        hard_targets, minlength=int(num_classes)
    )
    entropy = -(
        probabilities
        * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
    ).sum(dim=1)
    return {
        "pre_correction_predicted_class_histogram": predicted_histogram.tolist(),
        "pre_correction_predicted_class_count": int(
            torch.count_nonzero(predicted_histogram).item()
        ),
        "pre_correction_dominant_class_ratio": float(
            predicted_histogram.max().item() / predictions.numel()
        ),
        "pre_correction_mean_softmax_entropy": float(entropy.mean().item()),
        "corrected_target_histogram": corrected_histogram.tolist(),
        "corrected_target_class_count": int(
            torch.count_nonzero(corrected_histogram).item()
        ),
        "corrected_target_dominant_class_ratio": float(
            corrected_histogram.max().item() / hard_targets.numel()
        ),
    }


def _index_trace_hash(indices, extend: int) -> tuple[str, str]:
    indices = torch.as_tensor(indices, dtype=torch.int64).cpu().contiguous()
    sample_hash = hashlib.sha256(indices.numpy().tobytes(order="C")).hexdigest()
    view_rows = torch.stack(
        (
            indices.repeat_interleave(int(extend)),
            torch.arange(int(extend), dtype=torch.int64).repeat(indices.numel()),
        ),
        dim=1,
    ).contiguous()
    view_hash = hashlib.sha256(view_rows.numpy().tobytes(order="C")).hexdigest()
    return sample_hash, view_hash


def _rng_fingerprint(materializer, order_generator) -> str:
    numpy_state = np.random.get_state()
    tensors = [
        ("torch_cpu", torch.get_rng_state()),
        ("reference", materializer.reference_generator.get_state()),
        ("adaptation", materializer.adaptation_generator.get_state()),
        ("inner", order_generator.get_state()),
    ]
    if torch.cuda.is_initialized():
        tensors.extend(
            (f"torch_cuda_{index}", state)
            for index, state in enumerate(torch.cuda.get_rng_state_all())
        )
    return canonical_sha256(
        {
            "python": repr(random.getstate()),
            "numpy": {
                "algorithm": numpy_state[0],
                "keys": numpy_state[1].tolist(),
                "position": int(numpy_state[2]),
                "has_gauss": int(numpy_state[3]),
                "cached_gaussian": float(numpy_state[4]),
            },
            "tensor_hash": hash_tensors(tensors),
        }
    )


def _capture_global_rng() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        ),
    }


def _restore_global_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _optimizer_fingerprint(optimizer, named_parameters) -> str:
    tensors = []
    scalars = []
    for name, parameter in named_parameters:
        state = optimizer.state.get(parameter, {})
        for state_name in sorted(state):
            value = state[state_name]
            if torch.is_tensor(value):
                tensors.append((f"{name}:{state_name}", value))
            else:
                scalars.append((name, state_name, value))
    return canonical_sha256(
        {
            "tensor_hash": hash_tensors(tensors),
            "scalars": scalars,
            "groups": [
                {
                    key: value
                    for key, value in group.items()
                    if key != "params"
                }
                for group in optimizer.param_groups
            ],
        }
    )


def _memory_fingerprint(memory) -> str:
    commit_count, features, labels, batch_ids = memory.state_fingerprint()
    return canonical_sha256(
        {
            "commit_count": commit_count,
            "tensor_hash": hash_tensors(
                (("features", features), ("labels", labels), ("batch_ids", batch_ids))
            ),
        }
    )


def _read_only_state(
    model, optimizer, trainable, memory, ema, materializer, order_generator
) -> dict:
    return {
        "model": hash_model_state(model),
        "optimizer": _optimizer_fingerprint(optimizer, trainable),
        "memory": _memory_fingerprint(memory),
        "ema_commit_count": ema.commit_count,
        "rng": _rng_fingerprint(materializer, order_generator),
    }


def _assert_read_only(before, after, phase: str) -> None:
    if before != after:
        changed = sorted(key for key in before if before[key] != after[key])
        raise RuntimeError(f"{phase} mutated persistent state: {changed}")


def _pre_adaptation_outputs(adapter, inputs, chunk_size: int):
    features, soft_targets = [], []
    physical_forwards = 0
    with torch.inference_mode():
        for start in range(0, inputs.shape[0], int(chunk_size)):
            chunk_features, logits = adapter.features_and_logits(
                inputs[start : start + int(chunk_size)]
            )
            features.append(chunk_features.detach())
            soft_targets.append(torch.softmax(logits, dim=-1).detach())
            physical_forwards += 1
    return torch.cat(features), torch.cat(soft_targets), physical_forwards


def _restore_off_mask(named_parameters, masks, anchors) -> None:
    with torch.no_grad():
        for name, parameter in named_parameters:
            mask = masks[name].to(parameter.device, dtype=torch.bool)
            parameter.copy_(torch.where(mask, parameter, anchors[name]))


def _native_self_training(
    config,
    task,
    adapter,
    optimizer,
    trainable,
    order_generator,
    masks=None,
) -> dict:
    scientific_batch = int(config["batch_size"])
    micro_batch = int(config["ist"]["native_micro_batch_size"])
    loss_sums = {"loss": 0.0, "loss_hard_ce": 0.0, "loss_soft_kl": 0.0}
    sample_count = 0
    optimizer_steps = 0
    physical_backwards = 0
    order_hashes = []
    for _ in range(int(config["ist"]["iters"])):
        order = torch.randperm(task.sample_count, generator=order_generator)
        order_hashes.append(
            hashlib.sha256(order.numpy().tobytes(order="C")).hexdigest()
        )
        for start in range(0, task.sample_count, scientific_batch):
            inner = order[start : start + scientific_batch]
            inner_count = int(inner.numel())
            optimizer.zero_grad(set_to_none=True)
            inner_totals = {
                "loss": 0.0,
                "loss_hard_ce": 0.0,
                "loss_soft_kl": 0.0,
            }
            for micro_start in range(0, inner_count, micro_batch):
                indices = inner[micro_start : micro_start + micro_batch].to(
                    task.views.device
                )
                weight = float(indices.numel() / inner_count)
                total, parts = ist_loss(
                    adapter.logits(task.views[indices]),
                    task.hard_targets[indices],
                    task.soft_targets[indices],
                    config["loss"],
                )
                if not torch.isfinite(total):
                    raise RuntimeError("native IST loss contains NaN or Inf")
                (total * weight).backward()
                inner_totals["loss"] += float(total.detach().item()) * weight
                for name in ("loss_hard_ce", "loss_soft_kl"):
                    inner_totals[name] += parts[name] * weight
                physical_backwards += 1
            if masks is None:
                optimizer.step()
            else:
                strict_masked_adamw_step(optimizer, trainable, masks)
                assert_off_mask_adam_state_zero(optimizer, trainable, masks)
            for name in loss_sums:
                loss_sums[name] += inner_totals[name] * inner_count
            sample_count += inner_count
            optimizer_steps += 1
    return {
        **{name: value / sample_count for name, value in loss_sums.items()},
        "inner_optimizer_steps": optimizer_steps,
        "native_physical_backward_count": physical_backwards,
        "inner_order_sha256": canonical_sha256(order_hashes),
    }


def _child_config(parent: dict, mask_index: int, mask_seed: int, output_dir: Path):
    child = copy.deepcopy(parent)
    child["output_dir"] = str(output_dir.resolve())
    child["random_mask_index"] = int(mask_index)
    child["mask_seed"] = int(mask_seed)
    scientific = {
        key: value
        for key, value in child.items()
        if key
        not in {
            "output_dir",
            "checkpoint_manifest_path",
            "scientific_config_sha256",
            "experiment_key",
        }
    }
    child["scientific_config_sha256"] = canonical_sha256(scientific)
    child["experiment_key"] = (
        parent["experiment_key"]
        + f"_mask-{mask_index}_{child['scientific_config_sha256'][:12]}"
    )
    return child


def run_single(
    config: dict,
    project_root: Path,
    *,
    show_progress: bool = True,
    model_loader=load_ist_model,
    loader_builder=build_ist_loaders,
    materializer_factory=ISTViewMaterializer,
) -> dict:
    """Run one dense/sparse trajectory; Random parents invoke this three times."""
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
    started_at = _utc_now()
    wall_started = time.perf_counter()
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "running",
        "created_at_utc": started_at,
        "command": list(sys.argv),
        "git": _git_info(project_root),
        "protocol_revision": config["protocol_revision"],
        "implementation_revision": config["implementation_revision"],
        "source_checkpoint_revision": config["source_checkpoint_revision"],
        "experiment_key": config["experiment_key"],
        "scientific_config_sha256": config["scientific_config_sha256"],
        "method": "ist",
        "variant": config["variant"],
        "dataset": config["dataset"],
        "transfer": config["transfer"],
        "formal_seed": config["formal_seed"],
        "formal_protocol": config["formal_protocol"],
        "target_labels_usage": "PU/FO metrics only",
    }
    _atomic_json(paths["manifest"], manifest)
    try:
        if config["runtime"]["device"] == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        set_reproducibility(
            config["formal_seed"], config["runtime"]["deterministic"]
        )
        (
            model,
            adapter,
            checkpoint,
            trainable,
            frozen,
            scope_record,
        ) = model_loader(config, device)
        trainable = list(trainable)
        trainable_names = {name for name, _ in trainable}
        frozen_state_before = hash_tensors(
            frozen_named_state(model, trainable_names)
        )
        model_state_before = hash_model_state(model)
        optimizer_config = config["optimization"]
        optimizer = torch.optim.AdamW(
            [parameter for _, parameter in trainable],
            lr=float(optimizer_config["lr"]),
            betas=tuple(float(value) for value in optimizer_config["betas"]),
            eps=float(optimizer_config["eps"]),
            weight_decay=float(optimizer_config["weight_decay"]),
        )
        online_loader, fo_loader, stream_record = loader_builder(config)
        materializer = materializer_factory(config)
        memory = CausalMemoryBank(
            config["ist"]["memory"]["max_len"],
            config["ist"]["feature_dim"],
            config["num_classes"],
        )
        ema = OuterBatchEMA(config["ist"]["ema_momentum"])
        order_generator = torch.Generator(device="cpu").manual_seed(
            int(config["formal_seed"])
            + int(config["ist"]["rng"]["inner_order_seed_offset"])
        )

        static_masks = None
        static_selection = None
        static_selector_started = time.perf_counter()
        budget = (
            None
            if config["selection"] is None
            else config["selection"]["requested_budget"]
        )
        if config["variant"] == GROUP_RANDOM:
            group_ids = selected_group_ids(budget, config["mask_seed"])
            static_masks = build_masks(trainable, group_ids)
            static_selection = mask_record(
                group_ids, budget=budget, seed=config["mask_seed"]
            )
        elif config["variant"] == GROUP_MAGNITUDE:
            scores = compute_group_l2_scores(trainable)
            group_ids = select_magnitude_group_ids(scores, budget)
            static_masks = build_masks(trainable, group_ids)
            static_selection = magnitude_mask_record(
                group_ids,
                scores=scores,
                budget=budget,
                checkpoint_sha256=config["checkpoint_sha256"],
            )
        static_selector_runtime = float(
            time.perf_counter() - static_selector_started
        )

        pu_meter = FixedClassMeter(config["num_classes"], config["class_names"])
        pu_indices = []
        batch_records = []
        adaptation_trace_hashes = []
        inner_order_hashes = []
        historical_groups = set()
        processed_batches = 0
        logical_counts = {
            "pre_inference_task_count": 0,
            "plca_call_count": 0,
            "memory_commit_count": 0,
            "native_ema_commit_count": 0,
            "pu_forward_task_count": 0,
            "saliency_support_selection_count": 0,
            "scheduler_step_count": 0,
        }
        progress = tqdm(
            online_loader,
            desc=f"IST {config['variant']} {config['dataset']} {config['transfer']}",
            unit="outer-batch",
            disable=not show_progress,
            dynamic_ncols=True,
        )
        for loader_batch_index, (raw_images, metric_labels, sample_indices) in enumerate(
            progress
        ):
            if (
                config["debug_max_outer_batches"] is not None
                and processed_batches >= config["debug_max_outer_batches"]
            ):
                break
            if len(raw_images) == 1:
                raise RuntimeError(
                    "stream_protocol_mismatch: singleton reached IST state machine"
                )
            materialization_started = time.perf_counter()
            materialized = materializer.materialize(raw_images, sample_indices)
            view_materialization_runtime = float(
                time.perf_counter() - materialization_started
            )
            h2d_started = time.perf_counter()
            reference_views = materialized.reference_views.to(
                device, non_blocking=True
            )
            adaptation_views = materialized.adaptation_views.to(
                device, non_blocking=True
            )
            _sync(device)
            h2d_runtime = float(time.perf_counter() - h2d_started)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            adapt_started = time.perf_counter()
            ema.begin_batch(processed_batches, trainable)
            sparse_anchors = (
                {
                    name: parameter.detach().clone()
                    for name, parameter in trainable
                }
                if config["variant"] in SPARSE_VARIANTS
                else None
            )

            pre_started = time.perf_counter()
            features, soft_targets, pre_physical_forwards = (
                _pre_adaptation_outputs(
                    adapter,
                    adaptation_views,
                    config["ist"]["pre_inference_chunk_size"],
                )
            )
            pre_runtime = float(time.perf_counter() - pre_started)
            logical_counts["pre_inference_task_count"] += 1

            memory_started = time.perf_counter()
            past_memory = memory.snapshot_for(processed_batches, device)
            memory_snapshot_runtime = float(
                time.perf_counter() - memory_started
            )
            plca_started = time.perf_counter()
            plca_result = robust_plca(
                features, soft_targets, past_memory, config["ist"]["plca"]
            )
            plca_runtime = float(time.perf_counter() - plca_started)
            logical_counts["plca_call_count"] += 1
            commit_started = time.perf_counter()
            memory.commit(
                processed_batches, features, plca_result.corrected_one_hot
            )
            memory_commit_runtime = float(time.perf_counter() - commit_started)
            logical_counts["memory_commit_count"] += 1
            task = FixedISTTask(
                adaptation_views,
                plca_result.corrected_hard_labels,
                soft_targets,
            )
            class_diagnostics = _class_diagnostics(
                soft_targets,
                plca_result.corrected_hard_labels,
                config["num_classes"],
            )

            selector_started = time.perf_counter()
            masks = static_masks
            selection_record = static_selection
            scoring_record = {}
            if config["variant"] == GROUP_SALIENCY:
                scoring_values = accumulate_full_objective(
                    task,
                    adapter.logits,
                    trainable,
                    config["loss"],
                    config["ist"]["objective_chunk_size"],
                )
                scoring_record = {
                    f"saliency_scoring_{name}": value
                    for name, value in scoring_values.items()
                }
                scores = compute_group_saliency_scores(trainable)
                group_ids = select_saliency_group_ids(scores, budget)
                masks = build_masks(trainable, group_ids)
                selection_record = dynamic_mask_record(
                    group_ids, scores=scores, budget=budget
                )
                optimizer.zero_grad(set_to_none=True)
                logical_counts["saliency_support_selection_count"] += 1
            selector_runtime = float(time.perf_counter() - selector_started)
            if selection_record is not None:
                if (
                    selection_record["realized_group_count"]
                    != config["selection"]["requested_group_count"]
                ):
                    raise RuntimeError("sparse selector did not realize exact K")
                historical_groups.update(selection_record["selected_group_ids"])

            self_training_started = time.perf_counter()
            training_record = _native_self_training(
                config,
                task,
                adapter,
                optimizer,
                trainable,
                order_generator,
                masks=masks,
            )
            self_training_runtime = float(
                time.perf_counter() - self_training_started
            )
            inner_order_hashes.append(training_record["inner_order_sha256"])
            for name, parameter in trainable:
                if not torch.isfinite(parameter).all():
                    raise RuntimeError(
                        f"native IST produced non-finite parameter values: {name}"
                    )

            writeback_started = time.perf_counter()
            ema.commit(processed_batches, trainable)
            logical_counts["native_ema_commit_count"] += 1
            if masks is not None:
                _restore_off_mask(trainable, masks, sparse_anchors)
                assert_off_mask_adam_state_zero(optimizer, trainable, masks)
                for name, parameter in trainable:
                    mask = masks[name].to(parameter.device, dtype=torch.bool)
                    if not torch.equal(
                        parameter.detach()[~mask], sparse_anchors[name][~mask]
                    ):
                        raise RuntimeError(
                            f"EMA/writeback changed off-mask values: {name}"
                        )
            writeback_runtime = float(time.perf_counter() - writeback_started)
            if model.training:
                raise RuntimeError("IST Transformer left eval mode")
            frozen_now = hash_tensors(frozen_named_state(model, trainable_names))
            if frozen_now != frozen_state_before:
                raise RuntimeError("off-scope parameters or buffers changed")
            _sync(device)
            adapt_runtime = float(time.perf_counter() - adapt_started)

            pu_state_before = _read_only_state(
                model,
                optimizer,
                trainable,
                memory,
                ema,
                materializer,
                order_generator,
            )
            pu_started = time.perf_counter()
            with torch.inference_mode():
                post_logits = adapter.logits(reference_views)
            _sync(device)
            pu_runtime = float(time.perf_counter() - pu_started)
            pu_state_after = _read_only_state(
                model,
                optimizer,
                trainable,
                memory,
                ema,
                materializer,
                order_generator,
            )
            _assert_read_only(pu_state_before, pu_state_after, "PU")
            logical_counts["pu_forward_task_count"] += 1

            predictions = post_logits.argmax(dim=1).cpu()
            labels = torch.as_tensor(metric_labels, dtype=torch.long).cpu()
            indices = torch.as_tensor(sample_indices, dtype=torch.long).cpu()
            pu_meter.update(labels, predictions)
            pu_indices.extend(int(value) for value in indices.tolist())
            processed_batches += 1
            adaptation_trace_hashes.append(
                materialized.augmentation_trace_sha256
            )
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
            correct = int((predictions == labels).sum().item())
            sample_index_hash, view_index_hash = _index_trace_hash(
                indices, config["ist"]["extend"]
            )
            pu_probabilities = torch.softmax(post_logits.detach().cpu(), dim=1)
            pu_histogram = torch.bincount(
                predictions, minlength=config["num_classes"]
            )
            memory_labels = memory.state_fingerprint()[2]
            record = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "event": "online_outer_batch",
                "loader_batch_index": loader_batch_index,
                "processed_batch_index": processed_batches - 1,
                "raw_batch_size": len(raw_images),
                "adaptation_view_count": task.sample_count,
                "sample_indices": indices.tolist(),
                "sample_index_trace_sha256": sample_index_hash,
                "view_index_trace_sha256": view_index_hash,
                "correct": correct,
                "PU-batch-overall-Acc": 100.0 * correct / labels.numel(),
                "augmentation_trace_sha256": materialized.augmentation_trace_sha256,
                "reference_trace_sha256": materialized.reference_trace_sha256,
                "adaptation_trace_sha256": materialized.adaptation_trace_sha256,
                "view_materialization_runtime_sec": view_materialization_runtime,
                "h2d_runtime_sec": h2d_runtime,
                "pre_adaptation_runtime_sec": pre_runtime,
                "plca_runtime_sec": plca_runtime,
                "memory_runtime_sec": (
                    memory_snapshot_runtime + memory_commit_runtime
                ),
                "selector_runtime_sec": selector_runtime,
                "native_self_training_runtime_sec": self_training_runtime,
                "writeback_runtime_sec": writeback_runtime,
                "adapt_runtime_sec": adapt_runtime,
                "pu_runtime_sec": pu_runtime,
                "online_runtime_sec": adapt_runtime + pu_runtime,
                "peak_gpu_memory_allocated_bytes": allocated_bytes,
                "peak_gpu_memory_reserved_bytes": reserved_bytes,
                "peak_gpu_memory_allocated_mb": allocated_bytes / 1048576.0,
                "peak_gpu_memory_reserved_mb": reserved_bytes / 1048576.0,
                "pre_inference_physical_forward_count": pre_physical_forwards,
                "plca_graph_sample_count": plca_result.graph_sample_count,
                "plca_memory_sample_count": plca_result.memory_sample_count,
                "memory_size": len(memory),
                "memory_commit_index": memory.commit_count - 1,
                "memory_corrected_label_histogram": memory_labels.sum(dim=0).tolist(),
                "PU-predicted-class-histogram": pu_histogram.tolist(),
                "PU-predicted-class-count": int(
                    torch.count_nonzero(pu_histogram).item()
                ),
                "PU-dominant-class-ratio": float(
                    pu_histogram.max().item() / predictions.numel()
                ),
                "PU-mean-softmax-entropy": float(
                    (-(pu_probabilities * pu_probabilities.clamp_min(
                        torch.finfo(pu_probabilities.dtype).tiny
                    ).log()).sum(dim=1)).mean().item()
                ),
                "target_granularity": "view",
                "pre_correction_soft_targets_used": True,
                "pu_is_separate_read_only_forward": True,
                "pu_state_unchanged": True,
                **class_diagnostics,
                **training_record,
                **scoring_record,
                **logical_counts,
            }
            if selection_record is not None:
                record["selection"] = selection_record
            _append_jsonl(paths["metrics"], record)
            batch_records.append(record)
            progress.set_postfix(
                loss=f"{training_record['loss']:.4f}",
                pu=f"{100.0 * pu_meter.confusion.diagonal().sum().item() / pu_meter.sample_count:.2f}%",
            )
        progress.close()
        if processed_batches == 0:
            raise RuntimeError("IST stream processed no outer batches")
        if config["formal_protocol"]:
            _validate_indices(
                pu_indices,
                stream_record["sample_count"],
                phase="PU",
                online=True,
            )
        expected = processed_batches
        for name in (
            "pre_inference_task_count",
            "plca_call_count",
            "memory_commit_count",
            "native_ema_commit_count",
            "pu_forward_task_count",
        ):
            if logical_counts[name] != expected:
                raise RuntimeError(f"{name} must equal processed outer batches")
        if logical_counts["scheduler_step_count"] != 0:
            raise RuntimeError("IST Transformer must not step a scheduler")
        if config["variant"] == GROUP_SALIENCY:
            if logical_counts["saliency_support_selection_count"] != expected:
                raise RuntimeError("Saliency must select exactly once per outer batch")
        elif logical_counts["saliency_support_selection_count"] != 0:
            raise RuntimeError("non-Saliency variant selected saliency support")

        optimizer.zero_grad(set_to_none=True)
        model.requires_grad_(False)
        model.eval()
        fo_state_before = _read_only_state(
            model,
            optimizer,
            trainable,
            memory,
            ema,
            materializer,
            order_generator,
        )
        fo_meter = FixedClassMeter(config["num_classes"], config["class_names"])
        fo_indices = []
        fo_runtimes = []
        fo_global_rng = _capture_global_rng()
        with torch.inference_mode():
            for images, metric_labels, sample_indices in fo_loader:
                images = images.to(device, non_blocking=True)
                _sync(device)
                fo_started = time.perf_counter()
                logits = adapter.logits(images)
                _sync(device)
                fo_runtimes.append(float(time.perf_counter() - fo_started))
                predictions = logits.argmax(dim=1).cpu()
                labels = torch.as_tensor(metric_labels, dtype=torch.long).cpu()
                indices = torch.as_tensor(sample_indices, dtype=torch.long).cpu()
                fo_meter.update(labels, predictions)
                fo_indices.extend(int(value) for value in indices.tolist())
        _restore_global_rng(fo_global_rng)
        _validate_indices(
            fo_indices, stream_record["sample_count"], phase="FO", online=False
        )
        fo_state_after = _read_only_state(
            model,
            optimizer,
            trainable,
            memory,
            ema,
            materializer,
            order_generator,
        )
        _assert_read_only(fo_state_before, fo_state_after, "FO")
        if hash_tensors(frozen_named_state(model, trainable_names)) != frozen_state_before:
            raise RuntimeError("FO changed frozen state")

        pu_metrics = prefixed(pu_meter.compute(config["dataset"]), "PU")
        fo_metrics = prefixed(fo_meter.compute(config["dataset"]), "FO")
        adapt_runtimes = [row["adapt_runtime_sec"] for row in batch_records]
        pu_runtimes = [row["pu_runtime_sec"] for row in batch_records]
        online_runtimes = [row["online_runtime_sec"] for row in batch_records]
        peak_allocated = [
            row["peak_gpu_memory_allocated_mb"] for row in batch_records
        ]
        peak_reserved = [
            row["peak_gpu_memory_reserved_mb"] for row in batch_records
        ]
        peak_allocated_bytes = [
            row["peak_gpu_memory_allocated_bytes"] for row in batch_records
        ]
        peak_reserved_bytes = [
            row["peak_gpu_memory_reserved_bytes"] for row in batch_records
        ]
        completed_at = _utc_now()
        summary = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "status": "completed",
            "result_validity": (
                "valid" if config["formal_protocol"] else "diagnostic_incomplete"
            ),
            "protocol_revision": config["protocol_revision"],
            "implementation_revision": config["implementation_revision"],
            "source_checkpoint_revision": config["source_checkpoint_revision"],
            "method": "ist",
            "variant": config["variant"],
            "dataset": config["dataset"],
            "source": config["source"],
            "target": config["target"],
            "transfer": config["transfer"],
            "formal_seed": config["formal_seed"],
            "formal_protocol": config["formal_protocol"],
            "debug_smoke": config["debug_smoke"],
            "debug_max_outer_batches": config["debug_max_outer_batches"],
            "experiment_key": config["experiment_key"],
            "scientific_config_sha256": config["scientific_config_sha256"],
            "primary_metric": _primary_name(config["dataset"]),
            "class-names": config["class_names"],
            **pu_metrics,
            **fo_metrics,
            "stream": {
                key: value
                for key, value in stream_record.items()
                if key != "online_order"
            },
            "checkpoint": checkpoint,
            "scope": scope_record,
            "optimization": copy.deepcopy(config["optimization"]),
            "loss": copy.deepcopy(config["loss"]),
            "ist": copy.deepcopy(config["ist"]),
            "selection": copy.deepcopy(config["selection"]),
            "requested_budget": (
                None
                if config["selection"] is None
                else config["selection"]["requested_budget"]
            ),
            "requested_group_count": (
                None
                if config["selection"] is None
                else config["selection"]["requested_group_count"]
            ),
            "random_mask_index": config.get("random_mask_index"),
            "mask_seed": config.get("mask_seed"),
            "static_selection": static_selection,
            "static_selector_runtime_sec": static_selector_runtime,
            "processed_outer_batches": processed_batches,
            "adaptation_view_count": sum(
                row["adaptation_view_count"] for row in batch_records
            ),
            "historical_active_group_union": sorted(historical_groups),
            "historical_active_group_union_count": len(historical_groups),
            "augmentation_trace_history_sha256": canonical_sha256(
                adaptation_trace_hashes
            ),
            "inner_order_history_sha256": canonical_sha256(inner_order_hashes),
            **logical_counts,
            "host_optimizer_step_count": sum(
                row["inner_optimizer_steps"] for row in batch_records
            ),
            "model_state_sha256_before": model_state_before,
            "model_state_sha256_after_stream": hash_model_state(model),
            "frozen_state_sha256_before": frozen_state_before,
            "frozen_state_sha256_after": hash_tensors(
                frozen_named_state(model, trainable_names)
            ),
            "off_scope_exact": True,
            "pu_is_post_update_same_batch": True,
            "pu_is_separate_read_only_forward": True,
            "fo_is_independent_full_target_pass": True,
            "pu_state_unchanged": True,
            "fo_state_unchanged": True,
            "adapted_model_saved": False,
            "stream_checkpoint_saved": False,
            **_runtime_stats(adapt_runtimes, "adapt_batch_runtime"),
            **_runtime_stats(pu_runtimes, "pu_batch_runtime"),
            **_runtime_stats(online_runtimes, "online_batch_runtime"),
            **_runtime_stats(fo_runtimes, "fo_batch_runtime"),
            "adapt_runtime_total_sec": float(sum(adapt_runtimes)),
            "pu_runtime_total_sec": float(sum(pu_runtimes)),
            "online_compute_runtime_sec": float(sum(online_runtimes)),
            "fo_eval_runtime_sec": float(sum(fo_runtimes)),
            "gpu_peak_allocated_mean_mb": float(
                statistics.fmean(peak_allocated)
            ),
            "gpu_peak_allocated_max_mb": float(max(peak_allocated)),
            "gpu_peak_allocated_max_bytes": int(max(peak_allocated_bytes)),
            "gpu_peak_reserved_mean_mb": float(
                statistics.fmean(peak_reserved)
            ),
            "gpu_peak_reserved_max_mb": float(max(peak_reserved)),
            "gpu_peak_reserved_max_bytes": int(max(peak_reserved_bytes)),
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "wall_runtime_sec": float(time.perf_counter() - wall_started),
            "environment": _environment(device),
            "output_dir": str(output_dir),
        }
        _append_jsonl(
            paths["metrics"],
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "event": "final",
                **summary,
            },
        )
        _atomic_json(paths["summary"], summary)
        _atomic_json(
            paths["manifest"],
            {
                **manifest,
                "status": "completed",
                "completed_at_utc": completed_at,
                "result_validity": summary["result_validity"],
                "checkpoint": checkpoint,
                "environment": summary["environment"],
            },
        )
        print(
            f"[IST {config['variant']} {config['dataset']} {config['transfer']}] "
            f"PU={summary['PU-Acc']:.4f} FO={summary['FO-Acc']:.4f}",
            flush=True,
        )
        return summary
    except BaseException as error:
        failed_at = _utc_now()
        failure = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "status": "failed",
            "result_validity": "invalid",
            "method": "ist",
            "variant": config["variant"],
            "dataset": config["dataset"],
            "transfer": config["transfer"],
            "formal_seed": config["formal_seed"],
            "experiment_key": config["experiment_key"],
            "error_type": type(error).__name__,
            "error": str(error),
            "started_at_utc": started_at,
            "completed_at_utc": failed_at,
        }
        _atomic_json(paths["summary"], failure)
        _atomic_json(paths["manifest"], {**manifest, **failure})
        raise


def _mean(values):
    return float(statistics.fmean(float(value) for value in values))


def _aggregate_random(config, children, wall_runtime):
    if len(children) != NUM_RANDOM_MASKS:
        raise RuntimeError("Random requires exactly three completed children")
    for field in (
        "stream",
        "augmentation_trace_history_sha256",
        "inner_order_history_sha256",
        "processed_outer_batches",
    ):
        reference = children[0][field]
        if any(child[field] != reference for child in children[1:]):
            raise RuntimeError(f"Random children disagree on {field}")
    summary = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "completed",
        "result_validity": children[0]["result_validity"],
        "method": "ist",
        "variant": GROUP_RANDOM,
        "protocol_revision": config["protocol_revision"],
        "implementation_revision": config["implementation_revision"],
        "source_checkpoint_revision": config["source_checkpoint_revision"],
        "dataset": config["dataset"],
        "source": config["source"],
        "target": config["target"],
        "transfer": config["transfer"],
        "formal_seed": config["formal_seed"],
        "formal_protocol": config["formal_protocol"],
        "debug_smoke": config["debug_smoke"],
        "debug_max_outer_batches": config["debug_max_outer_batches"],
        "experiment_key": config["experiment_key"],
        "scientific_config_sha256": config["scientific_config_sha256"],
        "primary_metric": _primary_name(config["dataset"]),
        "requested_budget": config["selection"]["requested_budget"],
        "requested_group_count": config["selection"]["requested_group_count"],
        "realized_group_count": config["selection"]["requested_group_count"],
        "num_random_masks": NUM_RANDOM_MASKS,
        "mask_seeds": list(MASK_SEEDS),
        "class-names": children[0]["class-names"],
        "checkpoint": children[0]["checkpoint"],
        "scope": children[0]["scope"],
        "optimization": children[0]["optimization"],
        "loss": children[0]["loss"],
        "ist": children[0]["ist"],
        "selection": config["selection"],
        "PU-Acc": _mean(child["PU-Acc"] for child in children),
        "PU-Acc-mask-std": float(
            statistics.pstdev(child["PU-Acc"] for child in children)
        ),
        "FO-Acc": _mean(child["FO-Acc"] for child in children),
        "FO-Acc-mask-std": float(
            statistics.pstdev(child["FO-Acc"] for child in children)
        ),
        "PU-overall-Acc": _mean(
            child["PU-overall-Acc"] for child in children
        ),
        "FO-overall-Acc": _mean(
            child["FO-overall-Acc"] for child in children
        ),
        "online_batch_runtime_mean_sec": _mean(
            child["online_batch_runtime_mean_sec"] for child in children
        ),
        "online_compute_runtime_sec": _mean(
            child["online_compute_runtime_sec"] for child in children
        ),
        "random_total_online_compute_runtime_sec": float(
            sum(child["online_compute_runtime_sec"] for child in children)
        ),
        "fo_eval_runtime_sec": _mean(
            child["fo_eval_runtime_sec"] for child in children
        ),
        "random_total_fo_eval_runtime_sec": float(
            sum(child["fo_eval_runtime_sec"] for child in children)
        ),
        "gpu_peak_allocated_max_mb": max(
            child["gpu_peak_allocated_max_mb"] for child in children
        ),
        "gpu_peak_allocated_max_bytes": max(
            child["gpu_peak_allocated_max_bytes"] for child in children
        ),
        "gpu_peak_reserved_max_mb": max(
            child["gpu_peak_reserved_max_mb"] for child in children
        ),
        "gpu_peak_reserved_max_bytes": max(
            child["gpu_peak_reserved_max_bytes"] for child in children
        ),
        "stream": children[0]["stream"],
        "augmentation_trace_history_sha256": children[0][
            "augmentation_trace_history_sha256"
        ],
        "inner_order_history_sha256": children[0][
            "inner_order_history_sha256"
        ],
        "processed_outer_batches": children[0]["processed_outer_batches"],
        "host_optimizer_step_count": children[0]["host_optimizer_step_count"],
        "pre_inference_task_count": children[0]["pre_inference_task_count"],
        "plca_call_count": children[0]["plca_call_count"],
        "memory_commit_count": children[0]["memory_commit_count"],
        "native_ema_commit_count": children[0]["native_ema_commit_count"],
        "pu_forward_task_count": children[0]["pu_forward_task_count"],
        "off_scope_exact": all(child["off_scope_exact"] for child in children),
        "pu_state_unchanged": all(
            child["pu_state_unchanged"] for child in children
        ),
        "fo_state_unchanged": all(
            child["fo_state_unchanged"] for child in children
        ),
        "masks": [
            {
                "mask_index": child["random_mask_index"],
                "mask_seed": child["mask_seed"],
                "mask_sha256": child["static_selection"]["mask_sha256"],
                "PU-Acc": child["PU-Acc"],
                "FO-Acc": child["FO-Acc"],
                "summary_path": str(
                    Path(child["output_dir"]) / "summary.json"
                ),
            }
            for child in children
        ],
        "wall_runtime_sec": float(wall_runtime),
        "output_dir": config["output_dir"],
    }
    if config["dataset"] == "visda-c":
        names = children[0]["class-names"]
        summary["class-names"] = names
        for prefix in ("PU", "FO"):
            by_name = {
                name: _mean(
                    child[f"{prefix}-Acc-per-class-by-name"][name]
                    for child in children
                )
                for name in names
            }
            summary[f"{prefix}-Acc-per-class-by-name"] = by_name
            summary[f"{prefix}-worst-class-Acc"] = min(by_name.values())
            summary[f"{prefix}-class-std"] = float(
                statistics.pstdev(by_name.values())
            )
    return summary


def run_transfer(config: dict, project_root: Path, *, show_progress=True):
    if config["variant"] != GROUP_RANDOM:
        return run_single(
            config, project_root, show_progress=show_progress
        )
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=False)
    with open(output_dir / "effective_config.yaml", "w", encoding="utf-8") as file_obj:
        yaml.safe_dump(config, file_obj, sort_keys=False, allow_unicode=True)
    parent_manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "running",
        "method": "ist",
        "variant": GROUP_RANDOM,
        "protocol_revision": config["protocol_revision"],
        "implementation_revision": config["implementation_revision"],
        "experiment_key": config["experiment_key"],
        "scientific_config_sha256": config["scientific_config_sha256"],
        "mask_seeds": list(MASK_SEEDS),
        "created_at_utc": _utc_now(),
    }
    _atomic_json(output_dir / "manifest.json", parent_manifest)
    started = time.perf_counter()
    children = []
    try:
        for mask_index, mask_seed in enumerate(MASK_SEEDS):
            child = _child_config(
                config,
                mask_index,
                mask_seed,
                output_dir / f"mask_{mask_index:02d}",
            )
            children.append(
                run_single(child, project_root, show_progress=show_progress)
            )
        summary = _aggregate_random(
            config, children, time.perf_counter() - started
        )
        _atomic_json(output_dir / "summary.json", summary)
        _atomic_json(
            output_dir / "manifest.json",
            {
                **parent_manifest,
                "status": "completed",
                "result_validity": summary["result_validity"],
                "completed_at_utc": _utc_now(),
                "checkpoint": summary["checkpoint"],
            },
        )
        return summary
    except BaseException as error:
        failure = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "status": "failed",
            "result_validity": "invalid",
            "method": "ist",
            "variant": GROUP_RANDOM,
            "dataset": config["dataset"],
            "transfer": config["transfer"],
            "experiment_key": config["experiment_key"],
            "completed_child_count": len(children),
            "error_type": type(error).__name__,
            "error": str(error),
            "completed_at_utc": _utc_now(),
        }
        _atomic_json(output_dir / "summary.json", failure)
        _atomic_json(
            output_dir / "manifest.json", {**parent_manifest, **failure}
        )
        raise
