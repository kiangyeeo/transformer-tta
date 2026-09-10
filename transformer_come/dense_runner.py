"""Dense COME online stream (full_dense and candidate_dense).

Per valid outer batch, protocol section 8:

    validate batch before all adaptation transitions
    -> eval-mode scope
    -> persistent host AdamW.zero_grad
    -> fresh current model forward
    -> stable COME objective
    -> backward
    -> exactly one optimizer.step
    -> separate read-only post-update PU

Constant LR, no scheduler, no EMA/omega, FP32.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import torch
import yaml
from tqdm.auto import tqdm

from transformer.candidate_dense.model import frozen_named_state, hash_tensors
from transformer.source_only.metrics import FixedClassMeter, prefixed
from transformer.source_only.model import hash_model_state
from transformer.source_only.runner import (
    _append_jsonl,
    _atomic_json,
    _environment,
    _runtime_stats,
    _sync,
    _validate_indices,
    set_reproducibility,
)

from .common import (
    assert_finite_gradients,
    classifier_hash,
    come_objective_step,
    guard_online_batch,
    prediction_diagnostics,
    primary_metric_name,
    utc_now,
)
from .runner_common import (
    ARTIFACT_SCHEMA_VERSION,
    collapse_summary,
    device_from_config,
    failure_payload,
    gpu_memory_summary,
    manifest_base,
    run_fo_pass,
)


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


__all__ = ["run_dense_transfer"]
