"""Sequential no-adaptation DeiT OTTA control and structured artifacts."""

import copy
import os.path as osp
import sys
import time

import numpy as np
import torch

from shot_otta.artifacts import (
    append_jsonl,
    config_sha256,
    create_run_dir,
    dump_json,
    git_info,
)
from shot_otta.backbones.deit import hash_model_state, load_frozen_deit_source
from shot_otta.config import dump_yaml
from shot_otta.ttda.config import experiment_output_root
from shot_otta.ttda.source_only import (
    ARTIFACT_SCHEMA_VERSION,
    _autocast_context,
    _environment_record,
    _prediction_sha256,
    _set_reproducibility,
    _utc_now,
    compute_fixed_class_metrics,
)
from source_training.deit_data import (
    SourceImageList,
    build_loader,
    build_transforms,
    read_image_records,
)


def _prefixed_metrics(metrics, prefix):
    """Apply the established PU/FO names to fixed-class metrics."""
    return {
        f"{prefix}-Acc": metrics["Acc"],
        f"{prefix}-mean-class-Acc": metrics["mean-class-Acc"],
        f"{prefix}-overall-Acc": metrics["overall-Acc"],
        f"{prefix}-Acc-per-class": copy.deepcopy(metrics["Acc-per-class"]),
        f"{prefix}-class-count": copy.deepcopy(metrics["class-count"]),
        f"{prefix}-worst-class-Acc": metrics["worst-class-Acc"],
        f"{prefix}-worst-class-id": metrics["worst-class-id"],
        f"{prefix}-worst-class-name": metrics["worst-class-name"],
        f"{prefix}-class-std": metrics["class-std"],
    }


def evaluate_source_only_stream(
    model,
    loader,
    device,
    *,
    amp,
    on_batch=None,
):
    """Run one ordered stream pass, keeping every batch including size one."""
    all_labels, all_predictions, all_indices = [], [], []
    model.eval()
    with torch.inference_mode():
        for stream_batch, (images, labels, indices) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            with _autocast_context(device, amp):
                logits = model(images)
            predictions = logits.argmax(dim=1).cpu()
            labels = torch.as_tensor(labels).cpu()
            indices = torch.as_tensor(indices).cpu()
            all_predictions.append(predictions)
            all_labels.append(labels)
            all_indices.append(indices)
            if on_batch is not None:
                on_batch(stream_batch, labels, predictions, indices)
    if not all_predictions:
        raise RuntimeError("Target stream produced no batches")
    return (
        torch.cat(all_labels).numpy(),
        torch.cat(all_predictions).numpy(),
        torch.cat(all_indices).numpy(),
    )


def run_source_only_otta_experiment(config, project_root, *, model_factory=None):
    """Evaluate a DeiT W0 as a sequential OTTA source-only control."""
    if config.get("task") != "otta":
        raise ValueError("DeiT OTTA runner requires task=otta")
    output_root = experiment_output_root(config)
    run_id, output_dir = create_run_dir(
        output_root,
        config["task_name"],
        run_name=config["output"].get("run_name"),
    )
    paths = {
        "config": osp.join(output_dir, "config.yaml"),
        "manifest": osp.join(output_dir, "manifest.json"),
        "metrics": osp.join(output_dir, "metrics.jsonl"),
        "summary": osp.join(output_dir, "summary.json"),
    }
    dump_yaml(paths["config"], config)
    started_at = _utc_now()
    timer = time.perf_counter()
    base_manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at_utc": started_at,
        "method": "no_tta",
        "task": "otta",
        "protocol": "sequential_target_stream_no_adaptation",
        "variant": "source_only",
        "dataset": config["data"]["dataset"],
        "source": config["data"]["source"],
        "target": config["data"]["target"],
        "source_name": config["data"]["source_name"],
        "target_name": config["data"]["target_name"],
        "seed": config["seed"],
        "experiment_key": config["experiment_key"],
        "experiment_config_sha256": config["experiment_config_sha256"],
        "effective_config_sha256": config_sha256(config),
        "command": list(sys.argv),
        "git": git_info(project_root),
        "source_checkpoint": config["source_checkpoint"],
        "target_stream": {
            key: config["data"][key]
            for key in (
                "target_list",
                "target_list_sha256",
                "target_sample_count",
                "class_mapping_path",
                "class_mapping_sha256",
                "class_names",
            )
        },
        "adaptation": {
            "steps_per_batch": 0,
            "optimizer_created": False,
            "loss_computed": False,
            "backward_calls": 0,
            "state_carried_between_batches": True,
            "target_labels_usage": "evaluation_only",
        },
    }
    dump_json(paths["manifest"], base_manifest)

    try:
        device_type = config["device"]["type"]
        if device_type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")
            device = torch.device("cuda:0")
            torch.cuda.reset_peak_memory_stats(device)
        else:
            device = torch.device("cpu")
        amp_effective = bool(config["evaluation"]["amp"] and device.type == "cuda")
        _set_reproducibility(
            config["seed"], config["evaluation"]["deterministic"]
        )
        model, checkpoint_metadata = load_frozen_deit_source(
            config,
            device,
            model_factory=model_factory,
        )
        state_before = hash_model_state(model)

        records = read_image_records(
            config["data"]["target_list"],
            config["data"]["num_classes"],
        )
        _, transform = build_transforms(
            {
                **config["preprocessing"],
                "horizontal_flip_probability": 0.0,
            }
        )
        dataset = SourceImageList(records, range(len(records)), transform)
        loader = build_loader(
            dataset,
            batch_size=config["evaluation"]["batch_size"],
            workers=config["evaluation"]["workers"],
            shuffle=False,
            seed=config["seed"],
            pin_memory=(
                config["evaluation"]["pin_memory"] and device.type == "cuda"
            ),
        )

        batch_sample_counts = []

        def record_batch(stream_batch, labels, predictions, indices):
            batch_sample_counts.append(int(labels.numel()))
            batch_accuracy = float(
                (predictions == labels).to(torch.float32).mean().item() * 100.0
            )
            batch_prediction_sha256 = _prediction_sha256(
                indices.numpy(), labels.numpy(), predictions.numpy()
            )
            append_jsonl(
                paths["metrics"],
                {
                    "schema_version": ARTIFACT_SCHEMA_VERSION,
                    "event": "online_batch",
                    "run_id": run_id,
                    "experiment_key": config["experiment_key"],
                    "experiment_config_sha256": config[
                        "experiment_config_sha256"
                    ],
                    "stream_batch": int(stream_batch),
                    "sample_count": int(labels.numel()),
                    "first_sample_index": int(indices[0].item()),
                    "last_sample_index": int(indices[-1].item()),
                    "PU-batch-overall-Acc": batch_accuracy,
                    "adaptation_steps": 0,
                    "optimizer_created": False,
                    "loss_computed": False,
                    "backward_calls": 0,
                    "model_update_applied": False,
                    "state_carried_to_next_batch": stream_batch < len(loader),
                    "target_labels_usage": "evaluation_only",
                    "prediction_sha256": batch_prediction_sha256,
                },
            )

        labels, predictions, indices = evaluate_source_only_stream(
            model,
            loader,
            device,
            amp=amp_effective,
            on_batch=record_batch,
        )
        expected_indices = np.arange(len(records), dtype=np.int64)
        if not np.array_equal(indices, expected_indices):
            raise RuntimeError(
                "OTTA target samples were skipped, duplicated, or reordered"
            )
        state_after = hash_model_state(model)
        if state_before != state_after:
            raise RuntimeError("Source-only OTTA stream modified model state")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("Source-only OTTA unexpectedly created gradients")

        base_metrics = compute_fixed_class_metrics(
            labels,
            predictions,
            num_classes=config["data"]["num_classes"],
            class_names=config["data"]["class_names"],
        )
        pu_metrics = _prefixed_metrics(base_metrics, "PU")
        # W_T == W_0, so a separate full-target inference would be identical.
        # Reusing the exact stream predictions makes the control invariant
        # explicit and avoids a redundant second prediction pass.
        fo_metrics = _prefixed_metrics(base_metrics, "FO")
        pu_fo_equal = all(
            pu_metrics[key] == fo_metrics[key.replace("PU-", "FO-", 1)]
            for key in pu_metrics
        )
        if not pu_fo_equal:
            raise RuntimeError("Source-only OTTA PU and FO metrics differ")

        runtime = float(time.perf_counter() - timer)
        peak_memory = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        )
        completed_at = _utc_now()
        prediction_sha256 = _prediction_sha256(indices, labels, predictions)
        invariant_record = {
            "adaptation_steps": 0,
            "adaptation_steps_per_batch": 0,
            "optimizer_created": False,
            "loss_computed": False,
            "backward_calls": 0,
            "target_labels_usage": "evaluation_only",
            "stream_prediction_passes": 1,
            "fo_prediction_passes": 0,
            "fo_predictions_reused": True,
            "fo_reuse_reason": "model_state_unchanged",
            "PU-equals-FO": True,
            "model_state_sha256_before": state_before,
            "model_state_sha256_after": state_after,
            "model_state_unchanged": True,
            "prediction_sha256": prediction_sha256,
            "PU-prediction-sha256": prediction_sha256,
            "FO-prediction-sha256": prediction_sha256,
            "processed_sample_count": int(labels.size),
            "target_batch_count": int(len(loader)),
            "tail_batch_size": batch_sample_counts[-1],
            "tail_batch_size_one_policy": "kept",
            "drop_last": False,
            "state_carried_between_batches": True,
        }
        final_record = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "event": "final",
            "status": "completed",
            "run_id": run_id,
            "experiment_key": config["experiment_key"],
            "experiment_config_sha256": config["experiment_config_sha256"],
            **pu_metrics,
            **fo_metrics,
            **invariant_record,
            "runtime": runtime,
            "peak_gpu_memory_bytes": peak_memory,
        }
        append_jsonl(paths["metrics"], final_record)
        summary = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "status": "completed",
            "run_id": run_id,
            "output_dir": output_dir,
            "method": "no_tta",
            "variant": "source_only",
            "task": "otta",
            "protocol": "sequential_target_stream_no_adaptation",
            "dataset": config["data"]["dataset"],
            "source": config["data"]["source"],
            "target": config["data"]["target"],
            "source_name": config["data"]["source_name"],
            "target_name": config["data"]["target_name"],
            "source-target": (
                f"{config['data']['source_name']}-"
                f"{config['data']['target_name']}"
            ),
            "seed": config["seed"],
            "experiment_key": config["experiment_key"],
            "experiment_config_sha256": config["experiment_config_sha256"],
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "primary_metrics": ["PU-Acc", "FO-Acc"],
            "class-names": config["data"]["class_names"],
            "source_checkpoint_path": config["source_checkpoint"]["path"],
            "source_checkpoint_sha256": config["source_checkpoint"]["sha256"],
            "source_training_seed": config["source_checkpoint"][
                "source_training_seed"
            ],
            "source_best": config["source_checkpoint"]["best"],
            **pu_metrics,
            **fo_metrics,
            **invariant_record,
            "runtime": runtime,
            "peak_gpu_memory_bytes": peak_memory,
            "environment": _environment_record(device, amp_effective),
        }
        dump_json(paths["summary"], summary)
        dump_json(
            paths["manifest"],
            {
                **base_manifest,
                "completed_at_utc": completed_at,
                "environment": summary["environment"],
                "checkpoint_metadata_verified": True,
                "loaded_checkpoint_scientific_config_sha256": (
                    checkpoint_metadata["scientific_config_sha256"]
                ),
                **invariant_record,
            },
        )
        print(
            f"PU-Acc={summary['PU-Acc']} "
            f"FO-Acc={summary['FO-Acc']} runtime={runtime}",
            flush=True,
        )
        return summary
    except BaseException as error:
        dump_json(
            paths["summary"],
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "status": "failed",
                "run_id": run_id,
                "experiment_key": config["experiment_key"],
                "experiment_config_sha256": config[
                    "experiment_config_sha256"
                ],
                "method": "no_tta",
                "variant": "source_only",
                "task": "otta",
                "dataset": config["data"]["dataset"],
                "source": config["data"]["source"],
                "target": config["data"]["target"],
                "seed": config["seed"],
                "error_type": type(error).__name__,
                "error": str(error),
                "started_at_utc": started_at,
                "completed_at_utc": _utc_now(),
            },
        )
        raise
