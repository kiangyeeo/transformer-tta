"""Sequential DeiT OTTA saliency structural-group baseline.

After every TTA-loss backward, each paired Q-K / V-O / FFN group is scored by
the L2 norm of its gradient (``gradient_l2_norm``) and the
``active_groups = ceil(budget * total_groups)`` highest-scoring groups are
updated with the SHOT objective + AdamW.  The mask is rebuilt every online
step; positions outside the current step's mask keep their pre-step values
(``per_step_masked_accumulation``), so previously selected groups retain their
accumulated updates without being reset to W0.
"""

import copy
import os
import os.path as osp
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

from shot_otta.adaptation.deit_groups import (
    active_group_count,
    build_group_mask_dict,
    count_active_scalars,
    group_gradient_saliency_scores,
    select_top_groups,
    total_group_count,
)
from shot_otta.artifacts import (
    append_jsonl,
    config_sha256,
    create_run_dir,
    dump_json,
    git_info,
)
from shot_otta.backbones.deit import (
    candidate_parameter_names,
    hash_model_state,
    load_deit_source_for_adaptation,
)
from shot_otta.config import dump_yaml
from shot_otta.deit_source_only.runtime import (
    ARTIFACT_SCHEMA_VERSION,
    autocast_context,
    build_grad_scaler,
    build_target_loader,
    compute_fixed_class_metrics,
    environment_record,
    evaluate_model,
    prediction_sha256,
    prefixed_metrics,
    require_sequential_complete,
    set_reproducibility,
    utc_now,
)
from shot_otta.losses import deit_shot_adaptation_loss
from shot_otta.otta.saliency_config import experiment_output_root

SALIENCY_PROTOCOL = "sequential_target_stream_group_saliency_adaptation"


def _delta_stats(model, initial_state):
    """Count changed scalars and the L2 norm of W_T - W_0."""
    nonzero = 0
    total = 0
    squared = 0.0
    for name, parameter in model.named_parameters():
        difference = parameter.detach() - initial_state[name].to(
            parameter.device
        )
        nonzero += int((difference != 0).sum().item())
        total += int(difference.numel())
        squared += float(difference.pow(2).sum().item())
    return {
        "delta_nonzero_scalars": nonzero,
        "total_parameter_scalars": total,
        "delta_l2_norm": float(squared**0.5),
    }


def _masked_optimizer_step(optimizer, parameter_lookup, masks, scaler=None):
    """AdamW step restricted to the current step's group mask.

    Gradients outside the mask are zeroed before the step and masked-out
    positions are restored to their pre-step values afterwards.  This confines
    the current step's update to the selected groups while preserving any
    accumulated updates on previously selected groups.
    """
    pre_step = {}
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
            pre_step[name] = parameter.detach().clone()

    if scaler is not None:
        scaler.step(optimizer)
    else:
        optimizer.step()

    with torch.no_grad():
        for name, mask in masks.items():
            parameter = parameter_lookup[name]
            parameter.copy_(
                torch.where(
                    mask.to(device=parameter.device),
                    parameter,
                    pre_step[name],
                )
            )


def run_deit_otta_saliency_experiment(
    config, project_root, *, model_factory=None
):
    """Run the DeiT OTTA saliency structural-group baseline.

    Protocol: sequential target stream.  Every online step runs a SHOT-loss
    backward, scores the paired groups by gradient L2 norm, and updates the
    Top-K groups with AdamW; the updated model then predicts the same batch
    (PU).  After the stream the model is frozen and an independent
    full-target pass produces FO metrics.
    """
    variant = config.get("variant")
    if config.get("task") != "otta" or variant != "group_saliency":
        raise ValueError(
            "saliency runner requires task=otta and variant=group_saliency"
        )
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
        "mask_union": osp.join(output_dir, "mask_union.pt"),
    }
    dump_yaml(paths["config"], config)
    started_at = utc_now()
    timer = time.perf_counter()
    steps_per_batch = int(config["adaptation"]["steps_per_batch"])
    candidate_blocks = config["adaptation"]["candidate_blocks"]
    candidate_names = sorted(candidate_parameter_names(candidate_blocks))
    total_groups = total_group_count(candidate_blocks)
    num_active = active_group_count(config["adaptation"]["budget"], total_groups)
    saliency_score = config["adaptation"]["saliency_score"]
    base_manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at_utc": started_at,
        "method": "shot",
        "task": "otta",
        "protocol": SALIENCY_PROTOCOL,
        "variant": "group_saliency",
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
            "update_scope": config["adaptation"]["update_scope"],
            "model_mode": config["adaptation"]["model_mode"],
            "steps_per_batch": steps_per_batch,
            "delta_semantics": config["adaptation"]["delta_semantics"],
            "grouping": config["adaptation"]["grouping"],
            "selection": "saliency",
            "saliency_score": saliency_score,
            "mask_static": False,
            "mask_refresh_policy": "every_online_step_after_backward",
            "ranking_source": "current_gradient_after_backward",
            "budget": config["adaptation"]["budget"],
            "budget_unit": "structural_groups",
            "total_groups": total_groups,
            "active_groups": num_active,
            "candidate_blocks": candidate_blocks,
            "candidate_parameter_names": candidate_names,
            "target_labels_usage": "evaluation_only",
            "state_carried_between_batches": True,
            "tail_batch_size_one_policy": "kept",
        },
        "optimization": copy.deepcopy(config["optimization"]),
        "loss": copy.deepcopy(config["loss"]),
    }
    dump_json(paths["manifest"], base_manifest)

    try:
        device_type = config["device"]["type"]
        if device_type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA evaluation requested but CUDA is unavailable"
                )
            device = torch.device("cuda:0")
            torch.cuda.reset_peak_memory_stats()
        else:
            device = torch.device("cpu")
        amp_effective = bool(
            config["evaluation"]["amp"] and device.type == "cuda"
        )
        set_reproducibility(
            config["seed"], config["evaluation"]["deterministic"]
        )
        print(
            f"[group_saliency] loading source checkpoint: "
            f"{config['source_checkpoint']['path']}",
            flush=True,
        )
        model, checkpoint_metadata = load_deit_source_for_adaptation(
            config, device, model_factory=model_factory
        )
        state_before = hash_model_state(model)
        initial_state = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        candidate_scalars = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name in set(candidate_names)
        )
        named_parameters = dict(model.named_parameters())
        parameter_lookup = {}
        for name in candidate_names:
            parameter = named_parameters.get(name)
            if parameter is None:
                raise RuntimeError(f"Candidate parameter missing: {name}")
            parameter_lookup[name] = parameter
        active_scalars = num_active * 2 * 384
        print(
            f"[group_saliency] active_groups_per_step={num_active}/"
            f"{total_groups} active_scalars_per_step={active_scalars}/"
            f"{candidate_scalars}",
            flush=True,
        )

        optimization = config["optimization"]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(optimization["lr"]),
            betas=tuple(float(value) for value in optimization["betas"]),
            eps=float(optimization["eps"]),
            weight_decay=float(optimization["weight_decay"]),
        )
        scaler = build_grad_scaler(device, amp_effective)
        records, loader = build_target_loader(config)
        loss_config = config["loss"]

        pu_labels, pu_predictions, pu_indices = [], [], []
        batch_sample_counts = []
        backward_calls = 0
        total_steps = 0
        selected_groups_union = set()

        def record_batch(
            stream_batch, labels, predictions, indices, loss, loss_stats
        ):
            batch_sample_counts.append(int(labels.numel()))
            batch_accuracy = float(
                (predictions == labels).to(torch.float32).mean().item()
                * 100.0
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
                    "loss_total": float(loss.item()),
                    **loss_stats,
                    "adaptation_steps": steps_per_batch,
                    "backward_calls": int(backward_calls),
                    "optimizer_created": True,
                    "loss_computed": True,
                    "model_update_applied": True,
                    "mask_static": False,
                    "mask_refresh_policy": "every_online_step_after_backward",
                    "active_groups": num_active,
                    "candidate_groups": total_groups,
                    "active_scalars": active_scalars,
                    "candidate_scalars": candidate_scalars,
                    "state_carried_to_next_batch": (
                        stream_batch < len(loader)
                    ),
                    "target_labels_usage": "evaluation_only",
                    "prediction_sha256": prediction_sha256(
                        indices.numpy(), labels.numpy(), predictions.numpy()
                    ),
                },
            )
            return batch_accuracy

        stream = tqdm(
            loader,
            total=len(loader),
            desc="OTTA adapt saliency",
            file=sys.stdout,
            dynamic_ncols=True,
            mininterval=1.0,
        )
        for stream_batch, (images, labels, indices) in enumerate(
            stream, start=1
        ):
            images = images.to(device, non_blocking=True)
            batch_loss = None
            for _ in range(steps_per_batch):
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, amp_effective):
                    logits = model(images)
                    batch_loss, loss_stats = deit_shot_adaptation_loss(
                        logits, loss_config
                    )
                if scaler is not None:
                    scaler.scale(batch_loss).backward()
                else:
                    batch_loss.backward()

                grads = {}
                for name in candidate_names:
                    gradient = named_parameters[name].grad
                    if gradient is None:
                        raise RuntimeError(
                            f"Candidate parameter has no gradient: {name}"
                        )
                    grads[name] = gradient
                scores = group_gradient_saliency_scores(
                    grads, candidate_blocks=candidate_blocks
                )
                selected_groups = select_top_groups(scores, num_active)
                selected_groups_union.update(int(g) for g in selected_groups)
                masks = build_group_mask_dict(
                    selected_groups, candidate_blocks=candidate_blocks
                )
                masks = {
                    name: mask.to(device=device)
                    for name, mask in masks.items()
                }
                if count_active_scalars(masks) != active_scalars:
                    raise RuntimeError(
                        "Group mask scalar count does not match active groups"
                    )
                if scaler is not None:
                    _masked_optimizer_step(
                        optimizer, parameter_lookup, masks, scaler=scaler
                    )
                    scaler.update()
                else:
                    _masked_optimizer_step(
                        optimizer, parameter_lookup, masks
                    )
                backward_calls += 1
                total_steps += 1
            with torch.inference_mode():
                with autocast_context(device, amp_effective):
                    logits = model(images)
                predictions = logits.argmax(dim=1).cpu()
            labels_cpu = torch.as_tensor(labels).cpu()
            indices_cpu = torch.as_tensor(indices).cpu()
            batch_accuracy = record_batch(
                stream_batch,
                labels_cpu,
                predictions,
                indices_cpu,
                batch_loss,
                loss_stats,
            )
            pu_labels.append(labels_cpu)
            pu_predictions.append(predictions)
            pu_indices.append(indices_cpu)
            stream.set_postfix(
                loss=f"{float(batch_loss.item()):.3f}",
                acc=f"{batch_accuracy:.1f}",
                refresh=False,
            )
        stream.close()

        pu_labels_array = torch.cat(pu_labels).numpy()
        pu_predictions_array = torch.cat(pu_predictions).numpy()
        pu_indices_array = torch.cat(pu_indices).numpy()
        require_sequential_complete(
            pu_indices_array, len(records), protocol="OTTA PU stream"
        )
        state_after_stream = hash_model_state(model)
        if state_after_stream == state_before:
            raise RuntimeError(
                "group-saliency OTTA stream did not modify model state"
            )
        delta_stats = _delta_stats(model, initial_state)
        selected_groups_union = sorted(selected_groups_union)
        union_count = len(selected_groups_union)
        # Only groups selected in at least one step can have changed; each
        # paired group covers exactly 768 scalars.
        if delta_stats["delta_nonzero_scalars"] > union_count * 2 * 384:
            raise RuntimeError(
                "delta_nonzero_scalars exceeds the selected-group union "
                f"bound: {delta_stats['delta_nonzero_scalars']} > "
                f"{union_count} groups x 768 = {union_count * 2 * 384}"
            )
        union_masks = build_group_mask_dict(
            selected_groups_union, candidate_blocks=candidate_blocks
        )
        torch.save(
            {
                name: mask.detach().cpu()
                for name, mask in union_masks.items()
            },
            paths["mask_union"],
        )

        optimizer.zero_grad(set_to_none=True)
        model.requires_grad_(False)
        model.eval()
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError(
                "group-saliency OTTA left non-None gradients after stream"
            )
        fo_labels, fo_predictions, fo_indices = evaluate_model(
            model,
            loader,
            device,
            amp=amp_effective,
            progress_label="OTTA FO",
            use_tqdm=True,
        )
        require_sequential_complete(
            fo_indices, len(records), protocol="OTTA FO evaluation"
        )
        state_after_fo = hash_model_state(model)
        if state_after_fo != state_after_stream:
            raise RuntimeError("OTTA FO evaluation modified final model state")
        if not np.array_equal(pu_indices_array, fo_indices) or not np.array_equal(
            pu_labels_array, fo_labels
        ):
            raise RuntimeError("OTTA PU and FO evaluated different target samples")

        metric_kwargs = {
            "num_classes": config["data"]["num_classes"],
            "class_names": config["data"]["class_names"],
        }
        pu_metrics = prefixed_metrics(
            compute_fixed_class_metrics(
                pu_labels_array, pu_predictions_array, **metric_kwargs
            ),
            "PU",
        )
        fo_metrics = prefixed_metrics(
            compute_fixed_class_metrics(
                fo_labels, fo_predictions, **metric_kwargs
            ),
            "FO",
        )
        pu_prediction_sha256 = prediction_sha256(
            pu_indices_array, pu_labels_array, pu_predictions_array
        )
        fo_prediction_sha256 = prediction_sha256(
            fo_indices, fo_labels, fo_predictions
        )
        predictions_equal = np.array_equal(
            pu_predictions_array, fo_predictions
        )

        runtime = float(time.perf_counter() - timer)
        peak_memory = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        )
        completed_at = utc_now()
        invariant_record = {
            "adaptation_steps": total_steps,
            "adaptation_steps_per_batch": steps_per_batch,
            "optimizer_created": True,
            "loss_computed": True,
            "backward_calls": backward_calls,
            "target_labels_usage": "evaluation_only",
            "prediction_passes": 2,
            "stream_prediction_passes": 1,
            "fo_prediction_passes": 1,
            "fo_predictions_reused": False,
            "fo_evaluation_scope": "full_target_dataset",
            "PU-predictions-equal-FO": predictions_equal,
            "model_state_sha256_before": state_before,
            "model_state_sha256_after_stream": state_after_stream,
            "model_state_sha256_after_fo": state_after_fo,
            "model_state_sha256_after": state_after_fo,
            "model_state_unchanged": False,
            "PU-prediction-sha256": pu_prediction_sha256,
            "FO-prediction-sha256": fo_prediction_sha256,
            "processed_sample_count": int(pu_labels_array.size),
            "stream_processed_sample_count": int(pu_labels_array.size),
            "fo_processed_sample_count": int(fo_labels.size),
            "target_batch_count": int(len(loader)),
            "stream_batch_count": int(len(loader)),
            "fo_batch_count": int(len(loader)),
            "tail_batch_size": batch_sample_counts[-1],
            "tail_batch_size_one_policy": "kept",
            "drop_last": False,
            "state_carried_between_batches": True,
            "delta_nonzero_scalars": delta_stats["delta_nonzero_scalars"],
            "total_parameter_scalars": delta_stats["total_parameter_scalars"],
            "candidate_scalars": candidate_scalars,
            "delta_l2_norm": delta_stats["delta_l2_norm"],
            "active_groups": num_active,
            "candidate_groups": total_groups,
            "active_scalars": active_scalars,
            "selected_groups_union_count": union_count,
            "selected_groups_union": selected_groups_union,
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
            "method": "shot",
            "variant": "group_saliency",
            "task": "otta",
            "protocol": SALIENCY_PROTOCOL,
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
            "run_seed": config["seed"],
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
            "budget": config["adaptation"]["budget"],
            "saliency_score": saliency_score,
            "candidate_groups": total_groups,
            "active_groups": num_active,
            "candidate_scalars": candidate_scalars,
            "active_scalars": active_scalars,
            "selected_groups_union_count": union_count,
            "selected_groups_union": selected_groups_union,
            "adaptation": copy.deepcopy(base_manifest["adaptation"]),
            "optimization": copy.deepcopy(config["optimization"]),
            "loss": copy.deepcopy(config["loss"]),
            **pu_metrics,
            **fo_metrics,
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
            f"[group_saliency] PU-Acc={summary['PU-Acc']} "
            f"FO-Acc={summary['FO-Acc']} "
            f"union_groups={union_count} runtime={runtime}",
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
                "method": "shot",
                "variant": "group_saliency",
                "task": "otta",
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
