"""Strict no-adaptation DeiT TTDA evaluation and structured artifacts."""

import os.path as osp
import sys
import time

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
from shot_otta.deit_source_only.config import experiment_output_root
from shot_otta.deit_source_only.runtime import (
    ARTIFACT_SCHEMA_VERSION,
    build_target_loader,
    compute_fixed_class_metrics,
    environment_record,
    evaluate_model,
    prediction_sha256,
    require_sequential_complete,
    set_reproducibility,
    utc_now,
)


def run_source_only_experiment(config, project_root, *, model_factory=None):
    """Evaluate one source checkpoint without any adaptation side effects."""
    if config.get("task") != "ttda":
        raise ValueError("DeiT TTDA runner requires task=ttda")
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
    started_at = utc_now()
    timer = time.perf_counter()
    base_manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at_utc": started_at,
        "method": "no_tta",
        "task": "ttda",
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
        "target_data": {
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
            "steps": 0,
            "optimizer_created": False,
            "loss_computed": False,
            "backward_calls": 0,
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
            # torch 2.4.1 raises "Invalid device argument" when an explicit
            # device index is passed before the CUDA caching allocator has been
            # initialized. Reset the current device's stats instead.
            torch.cuda.reset_peak_memory_stats()
        else:
            device = torch.device("cpu")
        amp_effective = bool(
            config["evaluation"]["amp"] and device.type == "cuda"
        )
        set_reproducibility(
            config["seed"], config["evaluation"]["deterministic"]
        )
        model, checkpoint_metadata = load_frozen_deit_source(
            config,
            device,
            model_factory=model_factory,
        )
        state_before = hash_model_state(model)

        records, loader = build_target_loader(config)
        labels, predictions, indices = evaluate_model(
            model,
            loader,
            device,
            amp=amp_effective,
        )
        require_sequential_complete(indices, len(records), protocol="TTDA")
        state_after = hash_model_state(model)
        if state_before != state_after:
            raise RuntimeError("Source-only evaluation modified model state")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("Source-only evaluation unexpectedly created gradients")

        metric_kwargs = {
            "num_classes": config["data"]["num_classes"],
            "class_names": config["data"]["class_names"],
        }
        metrics = compute_fixed_class_metrics(
            labels, predictions, **metric_kwargs
        )
        runtime = float(time.perf_counter() - timer)
        peak_memory = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        )
        completed_at = utc_now()
        invariant_record = {
            "adaptation_steps": 0,
            "optimizer_created": False,
            "loss_computed": False,
            "backward_calls": 0,
            "target_labels_usage": "evaluation_only",
            "prediction_passes": 1,
            "single_evaluation_pass": True,
            "model_state_sha256_before": state_before,
            "model_state_sha256_after": state_after,
            "model_state_unchanged": True,
            "prediction_sha256": prediction_sha256(
                indices, labels, predictions
            ),
            "processed_sample_count": int(labels.size),
            "target_batch_count": int(len(loader)),
            "drop_last": False,
        }
        final_record = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "event": "final",
            "status": "completed",
            "run_id": run_id,
            "experiment_key": config["experiment_key"],
            "experiment_config_sha256": config[
                "experiment_config_sha256"
            ],
            **metrics,
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
            "task": "ttda",
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
            "experiment_config_sha256": config[
                "experiment_config_sha256"
            ],
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "primary_metric": "macro_class_accuracy",
            "class-names": config["data"]["class_names"],
            "source_checkpoint_path": config["source_checkpoint"]["path"],
            "source_checkpoint_sha256": config["source_checkpoint"]["sha256"],
            "source_training_seed": config["source_checkpoint"][
                "source_training_seed"
            ],
            "source_best": config["source_checkpoint"]["best"],
            **metrics,
            **invariant_record,
            "runtime": runtime,
            "peak_gpu_memory_bytes": peak_memory,
            "environment": environment_record(device, amp_effective),
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
            f"Acc={summary['Acc']} "
            f"overall-Acc={summary['overall-Acc']} "
            f"runtime={runtime}",
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
                "task": "ttda",
                "dataset": config["data"]["dataset"],
                "source": config["data"]["source"],
                "target": config["data"]["target"],
                "seed": config["seed"],
                "error_type": type(error).__name__,
                "error": str(error),
                "started_at_utc": started_at,
                "completed_at_utc": utc_now(),
            },
        )
        raise
