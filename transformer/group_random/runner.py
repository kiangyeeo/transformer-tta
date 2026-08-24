"""Run three static structural Random masks for one transfer/budget condition."""

from __future__ import annotations

import copy
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from tqdm.auto import tqdm

from transformer.source_only.config import canonical_sha256
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

from .config import MASK_SEEDS, NUM_RANDOM_MASKS, budget_key
from .data import build_target_loaders
from .groups import (
    build_masks,
    hash_masked_values,
    mask_record,
    selected_group_ids,
)
from .loss import shot_loss
from .model import (
    frozen_named_state,
    hash_tensors,
    load_group_random_model,
)
from .optimizer import assert_off_mask_adam_state_zero, strict_masked_adamw_step


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
    parameters = dict(model.named_parameters())
    return [(name, parameters[name]) for name in names]


def _child_config(parent: dict, mask_index: int, mask_seed: int, output_dir: Path) -> dict:
    child = copy.deepcopy(parent)
    child["random_mask_index"] = int(mask_index)
    child["mask_seed"] = int(mask_seed)
    child["parent_experiment_key"] = parent["experiment_key"]
    child["parent_scientific_config_sha256"] = parent["scientific_config_sha256"]
    child["output_dir"] = str(output_dir.resolve())
    scientific = {
        "parent_scientific_config_sha256": parent["scientific_config_sha256"],
        "random_mask_index": int(mask_index),
        "mask_seed": int(mask_seed),
    }
    child["scientific_config_sha256"] = canonical_sha256(scientific)
    child["experiment_key"] = (
        f"{parent['experiment_key']}_mask-{mask_index:02d}_"
        f"{child['scientific_config_sha256'][:12]}"
    )
    return child


def run_mask_child(
    config: dict,
    project_root: Path,
    *,
    show_progress: bool = True,
    model_loader=load_group_random_model,
) -> dict:
    """Run one mask child from a fresh source model through PU and FO."""
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "config": output_dir / "effective_config.yaml",
        "manifest": output_dir / "manifest.json",
        "mask": output_dir / "mask.json",
        "metrics": output_dir / "metrics.jsonl",
        "summary": output_dir / "summary.json",
    }
    started_at = _utc_now()
    wall_started = time.perf_counter()
    base_manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "running",
        "created_at_utc": started_at,
        "command": list(sys.argv),
        "git": _git_info(project_root),
        "protocol_revision": config["protocol_revision"],
        "experiment_key": config["experiment_key"],
        "scientific_config_sha256": config["scientific_config_sha256"],
        "parent_experiment_key": config["parent_experiment_key"],
        "dataset": config["dataset"],
        "transfer": config["transfer"],
        "formal_seed": config["formal_seed"],
        "random_mask_index": config["random_mask_index"],
        "mask_seed": config["mask_seed"],
        "method": "shot",
        "variant": "group_random",
        "target_labels_usage": "PU/FO metrics only",
        "adaptation": copy.deepcopy(config["adaptation"]),
        "selection": copy.deepcopy(config["selection"]),
        "optimization": copy.deepcopy(config["optimization"]),
        "loss": copy.deepcopy(config["loss"]),
    }
    _atomic_json(paths["manifest"], base_manifest)

    try:
        if config["runtime"]["device"] == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable")
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        set_reproducibility(config["formal_seed"], config["runtime"]["deterministic"])
        model, checkpoint_record, candidates, frozen, scope_record = model_loader(
            config, device
        )
        if model.training:
            raise RuntimeError("Group-Random adaptation must keep DeiT in eval mode")
        candidate_names = tuple(name for name, _ in candidates)
        candidate_name_set = set(candidate_names)
        if len(candidate_name_set) != len(candidate_names):
            raise RuntimeError("Candidate parameter names must be unique")
        frozen_names = {name for name, _ in frozen}
        all_parameter_names = {name for name, _ in model.named_parameters()}
        if candidate_name_set & frozen_names:
            raise RuntimeError("Candidate and frozen parameter sets overlap")
        if candidate_name_set | frozen_names != all_parameter_names:
            raise RuntimeError("Candidate and frozen parameter sets do not cover the model")

        group_ids = selected_group_ids(
            config["selection"]["requested_budget"], config["mask_seed"]
        )
        selection_record = mask_record(
            group_ids,
            budget=config["selection"]["requested_budget"],
            seed=config["mask_seed"],
        )
        masks = build_masks(candidates, group_ids)
        if selection_record["realized_group_count"] != config["selection"][
            "requested_group_count"
        ]:
            raise RuntimeError("Random selection did not realize exactly K groups")
        _atomic_json(paths["mask"], selection_record)
        config = {**config, "mask_sha256": selection_record["mask_sha256"]}
        with open(paths["config"], "w", encoding="utf-8") as file_obj:
            yaml.safe_dump(config, file_obj, sort_keys=False, allow_unicode=True)

        model_state_before = hash_model_state(model)
        candidate_state_before = hash_tensors(candidates)
        selected_state_before = hash_masked_values(candidates, masks, selected=True)
        off_mask_state_before = hash_masked_values(candidates, masks, selected=False)
        frozen_state_before = hash_tensors(frozen_named_state(model, candidate_name_set))
        classifier_state_before = _classifier_hash(model)

        optimization = config["optimization"]
        optimizer = torch.optim.AdamW(
            [parameter for _, parameter in candidates],
            lr=float(optimization["lr"]),
            betas=tuple(float(value) for value in optimization["betas"]),
            eps=float(optimization["eps"]),
            weight_decay=float(optimization["weight_decay"]),
        )
        online_loader, fo_loader, stream_record = build_target_loaders(config)
        if online_loader.batch_size != config["batch_size"]:
            raise RuntimeError("Online loader batch size differs from frozen config")
        if fo_loader.batch_size != config["fo_batch_size"]:
            raise RuntimeError("FO loader batch size differs from frozen config")

        # Model construction and the independent mask generator must not perturb
        # target augmentation RNG. Every child restarts the exact formal stream.
        set_reproducibility(config["formal_seed"], config["runtime"]["deterministic"])
        pu_meter = FixedClassMeter(config["num_classes"], config["class_names"])
        pu_indices: list[int] = []
        adapt_runtimes: list[float] = []
        pu_runtimes: list[float] = []
        online_runtimes: list[float] = []
        peak_allocated: list[float] = []
        peak_reserved: list[float] = []
        backward_calls = 0

        progress = tqdm(
            online_loader,
            desc=(
                f"random {config['dataset']} {config['transfer']} "
                f"rho={budget_key(config['selection']['requested_budget'])} "
                f"mask={config['random_mask_index']}"
            ),
            unit="batch",
            dynamic_ncols=True,
            leave=True,
            disable=not show_progress,
        )
        for batch_index, (images, labels, indices) in enumerate(progress, start=1):
            images = images.to(device, non_blocking=True)
            _sync(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            adapt_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            total_loss, loss_parts = shot_loss(logits, config["loss"])
            total_loss.backward()
            backward_calls += 1
            if any(parameter.grad is not None for _, parameter in frozen):
                raise RuntimeError("A frozen non-candidate parameter received a gradient")
            strict_masked_adamw_step(optimizer, candidates, masks)
            _sync(device)
            adapt_runtime = float(time.perf_counter() - adapt_started)

            if model.training:
                raise RuntimeError("Model left eval mode during online adaptation")
            if _classifier_hash(model) != classifier_state_before:
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

            batch_correct = int((predictions == labels).sum().item())
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
                    "correct": batch_correct,
                    "PU-batch-overall-Acc": 100.0 * batch_correct / labels.numel(),
                    "loss": float(total_loss.detach().item()),
                    **loss_parts,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "requested_budget": config["selection"]["requested_budget"],
                    "requested_group_count": config["selection"]["requested_group_count"],
                    "realized_group_count": selection_record["realized_group_count"],
                    "mask_sha256": selection_record["mask_sha256"],
                    "adapt_runtime_sec": adapt_runtime,
                    "pu_runtime_sec": pu_runtime,
                    "online_runtime_sec": online_runtime,
                    "adaptation_steps": 1,
                    "backward_calls": 1,
                    "pu_is_post_update_same_batch": True,
                    "pu_is_separate_read_only_forward": True,
                    "peak_gpu_memory_allocated_mb": allocated_mb,
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

        _validate_indices(pu_indices, stream_record["sample_count"], phase="PU", online=True)
        if backward_calls != len(online_loader):
            raise RuntimeError("Each online batch must receive exactly one backward call")
        assert_off_mask_adam_state_zero(optimizer, candidates, masks)
        model_state_after_stream = hash_model_state(model)
        candidates_after = _named_candidates(model, candidate_names)
        candidate_state_after_stream = hash_tensors(candidates_after)
        selected_state_after_stream = hash_masked_values(candidates_after, masks, selected=True)
        off_mask_state_after_stream = hash_masked_values(candidates_after, masks, selected=False)
        frozen_state_after_stream = hash_tensors(
            frozen_named_state(model, candidate_name_set)
        )
        if selected_state_after_stream == selected_state_before:
            raise RuntimeError("No selected structural-group coordinate changed")
        if off_mask_state_after_stream != off_mask_state_before:
            raise RuntimeError("An off-mask candidate coordinate changed")
        if frozen_state_after_stream != frozen_state_before:
            raise RuntimeError("A frozen parameter or model buffer changed")
        if model_state_after_stream == model_state_before:
            raise RuntimeError("Group-Random stream did not change model state")
        if _classifier_hash(model) != classifier_state_before:
            raise RuntimeError("Frozen classifier/head changed during the stream")

        optimizer.zero_grad(set_to_none=True)
        model.requires_grad_(False)
        model.eval()
        fo_meter = FixedClassMeter(config["num_classes"], config["class_names"])
        fo_indices: list[int] = []
        fo_runtimes: list[float] = []
        fo_progress = tqdm(
            fo_loader,
            desc=(
                f"FO {config['dataset']} {config['transfer']} "
                f"rho={budget_key(config['selection']['requested_budget'])} "
                f"mask={config['random_mask_index']}"
            ),
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
                    paths["metrics"],
                    {
                        "schema_version": ARTIFACT_SCHEMA_VERSION,
                        "event": "fo_batch",
                        "batch_index": batch_index,
                        "batch_size": int(labels.numel()),
                        "correct": batch_correct,
                        "FO-batch-overall-Acc": 100.0 * batch_correct / labels.numel(),
                        "fo_batch_runtime_sec": fo_runtime,
                        "adaptation_steps": 0,
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

        _validate_indices(fo_indices, stream_record["sample_count"], phase="FO", online=False)
        model_state_after_fo = hash_model_state(model)
        if model_state_after_fo != model_state_after_stream:
            raise RuntimeError("Read-only FO pass changed final model state")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("FO began or ended with non-None parameter gradients")

        pu_metrics = prefixed(pu_meter.compute(config["dataset"]), "PU")
        fo_metrics = prefixed(fo_meter.compute(config["dataset"]), "FO")
        completed_at = _utc_now()
        summary = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "status": "completed",
            "protocol_revision": config["protocol_revision"],
            "method": "shot",
            "variant": "group_random",
            "dataset": config["dataset"],
            "source": config["source"],
            "target": config["target"],
            "transfer": config["transfer"],
            "formal_seed": config["formal_seed"],
            "random_mask_index": config["random_mask_index"],
            "mask_seed": config["mask_seed"],
            "experiment_key": config["experiment_key"],
            "scientific_config_sha256": config["scientific_config_sha256"],
            "parent_experiment_key": config["parent_experiment_key"],
            "primary_metric": _primary_name(config["dataset"]),
            "class-names": config["class_names"],
            **pu_metrics,
            **fo_metrics,
            "stream": stream_record,
            "checkpoint": checkpoint_record,
            "adaptation": copy.deepcopy(config["adaptation"]),
            "optimization": copy.deepcopy(config["optimization"]),
            "loss": copy.deepcopy(config["loss"]),
            "preprocessing": copy.deepcopy(config["preprocessing"]),
            "online_batch_size": config["batch_size"],
            "fo_batch_size": config["fo_batch_size"],
            **scope_record,
            # Keep the complete structural support record authoritative even if
            # a future scope helper accidentally reuses the same field name.
            "selection": selection_record,
            "model_state_sha256_before": model_state_before,
            "model_state_sha256_after_stream": model_state_after_stream,
            "model_state_sha256_after_fo": model_state_after_fo,
            "candidate_state_sha256_before": candidate_state_before,
            "candidate_state_sha256_after_stream": candidate_state_after_stream,
            "candidate_state_changed": candidate_state_after_stream != candidate_state_before,
            "selected_state_sha256_before": selected_state_before,
            "selected_state_sha256_after_stream": selected_state_after_stream,
            "selected_state_changed": True,
            "off_mask_state_sha256_before": off_mask_state_before,
            "off_mask_state_sha256_after_stream": off_mask_state_after_stream,
            "off_mask_state_unchanged": True,
            "off_mask_adam_state_zero": True,
            "frozen_state_sha256_before": frozen_state_before,
            "frozen_state_sha256_after_stream": frozen_state_after_stream,
            "frozen_state_unchanged": True,
            "frozen_head_sha256_before": classifier_state_before,
            "frozen_head_sha256_after": _classifier_hash(model),
            "frozen_head_unchanged": True,
            "adaptation_steps": len(online_loader),
            "backward_calls": backward_calls,
            "pu_is_post_update_same_batch": True,
            "pu_is_separate_read_only_forward": True,
            "fo_is_independent_full_target_pass": True,
            "adapted_model_saved": False,
            **_runtime_stats(adapt_runtimes, "adapt_batch_runtime"),
            **_runtime_stats(pu_runtimes, "pu_batch_runtime"),
            **_runtime_stats(online_runtimes, "online_batch_runtime"),
            **_runtime_stats(fo_runtimes, "fo_batch_runtime"),
            "adapt_runtime_total_sec": float(sum(adapt_runtimes)),
            "pu_runtime_total_sec": float(sum(pu_runtimes)),
            "online_compute_runtime_sec": float(sum(online_runtimes)),
            "fo_eval_runtime_sec": float(sum(fo_runtimes)),
            "gpu_peak_allocated_mean_mb": float(statistics.fmean(peak_allocated)),
            "gpu_peak_allocated_max_mb": float(max(peak_allocated)),
            "gpu_peak_reserved_mean_mb": float(statistics.fmean(peak_reserved)),
            "gpu_peak_reserved_max_mb": float(max(peak_reserved)),
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
                "mask": selection_record,
                "environment": summary["environment"],
                "off_mask_state_unchanged": True,
                "off_mask_adam_state_zero": True,
                "frozen_state_unchanged": True,
                "adapted_model_saved": False,
            },
        )
        print(
            f"[{config['dataset']} {config['transfer']} "
            f"rho={budget_key(config['selection']['requested_budget'])} "
            f"mask={config['random_mask_index']}] "
            f"PU-Acc={summary['PU-Acc']:.4f} FO-Acc={summary['FO-Acc']:.4f}",
            flush=True,
        )
        return summary
    except BaseException as error:
        failed_at = _utc_now()
        failure = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "status": "failed",
            "dataset": config["dataset"],
            "transfer": config["transfer"],
            "formal_seed": config["formal_seed"],
            "random_mask_index": config["random_mask_index"],
            "mask_seed": config["mask_seed"],
            "experiment_key": config["experiment_key"],
            "error_type": type(error).__name__,
            "error": str(error),
            "started_at_utc": started_at,
            "completed_at_utc": failed_at,
        }
        _atomic_json(paths["summary"], failure)
        _atomic_json(paths["manifest"], {**base_manifest, **failure})
        raise


def _mean_std(values) -> tuple[float, float]:
    values = [float(value) for value in values]
    return statistics.fmean(values), statistics.pstdev(values)


def _aggregate_children(config: dict, children: list[dict], wall_runtime: float) -> dict:
    pu_mean, pu_std = _mean_std(child["PU-Acc"] for child in children)
    fo_mean, fo_std = _mean_std(child["FO-Acc"] for child in children)
    summary = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "completed",
        "protocol_revision": config["protocol_revision"],
        "method": "shot",
        "variant": "group_random",
        "dataset": config["dataset"],
        "source": config["source"],
        "target": config["target"],
        "transfer": config["transfer"],
        "formal_seed": config["formal_seed"],
        "experiment_key": config["experiment_key"],
        "scientific_config_sha256": config["scientific_config_sha256"],
        "primary_metric": _primary_name(config["dataset"]),
        "requested_budget": config["selection"]["requested_budget"],
        "requested_group_count": config["selection"]["requested_group_count"],
        "realized_group_count": config["selection"]["requested_group_count"],
        "active_candidate_scalars": config["selection"]["active_candidate_scalars"],
        "num_random_masks": NUM_RANDOM_MASKS,
        "mask_seeds": list(MASK_SEEDS),
        "cross_budget_policy": "nested_prefix_same_permutation",
        "PU-Acc": pu_mean,
        "PU-Acc-mask-std": pu_std,
        "FO-Acc": fo_mean,
        "FO-Acc-mask-std": fo_std,
        "masks": [
            {
                "random_mask_index": child["random_mask_index"],
                "mask_seed": child["mask_seed"],
                "mask_sha256": child["selection"]["mask_sha256"],
                "PU-Acc": child["PU-Acc"],
                "FO-Acc": child["FO-Acc"],
                "summary_path": str(Path(child["output_dir"]) / "summary.json"),
            }
            for child in children
        ],
        "online_batch_runtime_mean_sec": statistics.fmean(
            child["online_batch_runtime_mean_sec"] for child in children
        ),
        "online_compute_runtime_sec": statistics.fmean(
            child["online_compute_runtime_sec"] for child in children
        ),
        "random_total_online_compute_runtime_sec": sum(
            child["online_compute_runtime_sec"] for child in children
        ),
        "fo_eval_runtime_sec": statistics.fmean(
            child["fo_eval_runtime_sec"] for child in children
        ),
        "random_total_fo_eval_runtime_sec": sum(
            child["fo_eval_runtime_sec"] for child in children
        ),
        "gpu_peak_allocated_max_mb": max(
            child["gpu_peak_allocated_max_mb"] for child in children
        ),
        "gpu_peak_reserved_max_mb": max(
            child["gpu_peak_reserved_max_mb"] for child in children
        ),
        "wall_runtime_sec": float(wall_runtime),
        "output_dir": config["output_dir"],
    }
    for prefix in ("PU", "FO"):
        overall_mean, overall_std = _mean_std(
            child[f"{prefix}-overall-Acc"] for child in children
        )
        summary[f"{prefix}-overall-Acc"] = overall_mean
        summary[f"{prefix}-overall-Acc-mask-std"] = overall_std
    if config["dataset"] == "visda-c":
        class_names = children[0]["class-names"]
        summary["class-names"] = class_names
        for prefix in ("PU", "FO"):
            per_class_mean = {}
            per_class_std = {}
            for name in class_names:
                values = [child[f"{prefix}-Acc-per-class-by-name"][name] for child in children]
                mean, std = _mean_std(values)
                per_class_mean[name] = mean
                per_class_std[name] = std
            summary[f"{prefix}-Acc-per-class-by-name"] = per_class_mean
            summary[f"{prefix}-Acc-per-class-mask-std-by-name"] = per_class_std
            summary[f"{prefix}-worst-class-Acc"] = min(per_class_mean.values())
            summary[f"{prefix}-class-std"] = statistics.pstdev(per_class_mean.values())
    return summary


def run_transfer(
    config: dict,
    project_root: Path,
    *,
    show_progress: bool = True,
    child_runner=run_mask_child,
) -> dict:
    """Run all three deterministic Random masks for one formal condition."""
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=False)
    with open(output_dir / "effective_config.yaml", "w", encoding="utf-8") as file_obj:
        yaml.safe_dump(config, file_obj, sort_keys=False, allow_unicode=True)
    started_at = _utc_now()
    wall_started = time.perf_counter()
    parent_manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "running",
        "created_at_utc": started_at,
        "command": list(sys.argv),
        "git": _git_info(project_root),
        "protocol_revision": config["protocol_revision"],
        "variant": "group_random",
        "dataset": config["dataset"],
        "transfer": config["transfer"],
        "formal_seed": config["formal_seed"],
        "experiment_key": config["experiment_key"],
        "scientific_config_sha256": config["scientific_config_sha256"],
        "selection": copy.deepcopy(config["selection"]),
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
        completed_at = _utc_now()
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
            f"[{config['dataset']} {config['transfer']} "
            f"rho={budget_key(config['selection']['requested_budget'])}] "
            f"Random mean PU={summary['PU-Acc']:.4f}±{summary['PU-Acc-mask-std']:.4f} "
            f"FO={summary['FO-Acc']:.4f}±{summary['FO-Acc-mask-std']:.4f}",
            flush=True,
        )
        return summary
    except BaseException as error:
        failed_at = _utc_now()
        failure = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "status": "failed",
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
