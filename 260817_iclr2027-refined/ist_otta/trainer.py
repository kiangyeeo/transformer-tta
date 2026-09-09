"""Dense and FC/Conv sparse IST on the refined common OTTA substrate."""

import hashlib
import json
import os
import os.path as osp
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from protocol_constants import (
    CONV_CANDIDATE_PARAM_COUNT,
    CONV_GROUP_COUNTS,
    FC_CANDIDATE_PARAM_COUNT,
    FORMAL_INTEGER_BUDGETS,
)
from shot_otta.artifacts import (
    SCHEMA_VERSION,
    append_jsonl,
    create_run_dir,
    dump_json,
    experiment_output_root,
    write_initial_artifacts,
)
from shot_otta.candidates import (
    CONV_CANDIDATE_ORDER,
    MODULE_CANDIDATE_ORDER,
)
from shot_otta.efficiency import (
    BATCH_EFFICIENCY_FIELDS,
    aggregate_random_efficiency,
)
from shot_otta.models import load_source_models
from shot_otta.trainer import (
    RuntimeInstrumentation,
    _atomic_json_dump,
    _atomic_torch_save,
    _build_optimizer,
    _capture_rng_state,
    _compute_dataset_metrics,
    _evaluate,
    _existing_artifact_paths,
    _restore_rng_state,
    _save_model,
    _schedule_learning_rate,
    _stream_checkpoint_paths,
    _truncate_metrics_to_checkpoint,
    _update_manifest_runtime,
    setup_reproducibility,
)

from .data import ISTViewMaterializer, build_ist_loaders
from .ema import OuterBatchEMA
from .lbi import ISTLBIExecutor
from .memory import CausalMemoryBank
from .objective import (
    FixedISTTask,
    FullObjectiveGradientAccumulator,
    ist_loss as _ist_loss,
)
from .plca import robust_plca
from .sparse import (
    CONV_VARIANTS,
    FC_VARIANTS,
    LBI_VARIANTS,
    MAGNITUDE_VARIANTS,
    RANDOM_VARIANTS,
    SALIENCY_VARIANTS,
    SPARSE_VARIANTS,
    build_saliency_masks,
    build_static_masks,
    mask_statistics,
    masked_optimizer_step,
    restore_offmask_values,
    variant_selection,
    variant_track,
)


IST_VARIANTS = {
    "ist_full_dense", *FC_VARIANTS, *CONV_VARIANTS,
}

IST_STREAM_CHECKPOINT_SCHEMA_VERSION = 1


def _save_ist_stream_checkpoint(
    output_dir,
    config,
    run_id,
    net_f,
    net_b,
    net_c,
    optimizer,
    materializer,
    memory,
    ema,
    order_generator,
    static_masks,
    source_hashes,
    loader_batches_processed,
    processed_outer_batches,
    singleton_outer_batches_skipped,
    all_post_predictions,
    all_online_labels,
    started_at_utc,
    wall_runtime_sec,
    runtime_tracker,
):
    """Atomically save only state committed at an outer-batch boundary."""
    paths = _stream_checkpoint_paths(output_dir)
    payload = {
        "schema_version": IST_STREAM_CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_semantics": "completed_outer_batch_boundary",
        "run_id": run_id,
        "implementation_revision": config["implementation_revision"],
        "protocol_revision": config["protocol_revision"],
        "experiment_key": config["experiment_key"],
        "experiment_config_sha256": config["experiment_config_sha256"],
        "source_checkpoint_sha256": source_hashes,
        "loader_batches_processed": int(loader_batches_processed),
        "processed_outer_batches": int(processed_outer_batches),
        "singleton_outer_batches_skipped": int(
            singleton_outer_batches_skipped
        ),
        "started_at_utc": started_at_utc,
        "wall_runtime_sec": float(wall_runtime_sec),
        "net_f": net_f.state_dict(),
        "net_b": net_b.state_dict(),
        "net_c": net_c.state_dict(),
        "optimizer": optimizer.state_dict(),
        "materializer": materializer.state_dict(),
        "memory": memory.state_dict(),
        "ema": None if ema is None else ema.state_dict(),
        "order_generator": order_generator.get_state(),
        "static_masks": {
            name: mask.detach().to(device="cpu")
            for name, mask in static_masks.items()
        },
        "all_post_predictions": all_post_predictions,
        "all_online_labels": all_online_labels,
        "batch_efficiency_records": list(
            runtime_tracker.batch_efficiency_records
        ),
        "runtime_resume_used": bool(runtime_tracker.runtime_resume_used),
        "runtime_segment_count": int(runtime_tracker.runtime_segment_count),
        "rng_state": _capture_rng_state(),
    }
    _atomic_torch_save(paths["state"], payload)
    _atomic_json_dump(
        paths["metadata"],
        {
            "schema_version": IST_STREAM_CHECKPOINT_SCHEMA_VERSION,
            "status": "in_progress",
            "checkpoint_semantics": "completed_outer_batch_boundary",
            "run_id": run_id,
            "implementation_revision": config["implementation_revision"],
            "protocol_revision": config["protocol_revision"],
            "experiment_key": config["experiment_key"],
            "experiment_config_sha256": config[
                "experiment_config_sha256"
            ],
            "loader_batches_processed": int(loader_batches_processed),
            "processed_outer_batches": int(processed_outer_batches),
            "runtime_resume_used": bool(runtime_tracker.runtime_resume_used),
            "runtime_segment_count": int(runtime_tracker.runtime_segment_count),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return paths


def _load_ist_stream_checkpoint(
    resume_run_dir,
    config,
    net_f,
    net_b,
    net_c,
    optimizer,
    source_hashes,
):
    paths = _stream_checkpoint_paths(resume_run_dir)
    if not osp.isfile(paths["state"]):
        raise RuntimeError(
            "resume run directory has no committed IST stream checkpoint: "
            f"{resume_run_dir}"
        )
    payload = torch.load(paths["state"], map_location="cpu", weights_only=False)
    if payload.get("schema_version") != IST_STREAM_CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("unsupported IST stream checkpoint schema")
    for field in (
        "implementation_revision",
        "protocol_revision",
        "experiment_key",
        "experiment_config_sha256",
    ):
        if payload.get(field) != config[field]:
            raise RuntimeError(
                f"IST stream checkpoint identity mismatch for {field}"
            )
    if payload.get("source_checkpoint_sha256") != source_hashes:
        raise RuntimeError("IST stream checkpoint source hash mismatch")
    net_f.load_state_dict(payload["net_f"])
    net_b.load_state_dict(payload["net_b"])
    net_c.load_state_dict(payload["net_c"])
    optimizer.load_state_dict(payload["optimizer"])
    return payload, paths


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_checkpoint_hashes(paths):
    return {
        name: {"path": path, "sha256": _sha256_file(path)}
        for name, path in paths.items()
    }


def _all_named_parameters(net_f, net_b, net_c):
    return [
        (f"{model_name}.{name}", parameter)
        for model_name, model in (
            ("netF", net_f), ("netB", net_b), ("netC", net_c)
        )
        for name, parameter in model.named_parameters()
    ]


def configure_ist_variant(config, net_f, net_b, net_c):
    """Apply the frozen IST trainable scope and controlled BN policy."""

    variant = config["variant"]
    if variant not in IST_VARIANTS:
        raise ValueError(f"Unsupported IST variant: {variant}")
    all_parameters = _all_named_parameters(net_f, net_b, net_c)
    lookup = dict(all_parameters)
    for _, parameter in all_parameters:
        parameter.requires_grad = False

    track = variant_track(variant)
    if variant == "ist_full_dense":
        selected_names = {
            name for name in lookup
            if name.startswith("netF.") or name.startswith("netB.")
        }
        candidate_scope = "netF+netB"
        bn_stats_policy = "native_full_train"
        candidate_names = sorted(selected_names)
    elif track == "fc_scalar":
        selected_names = set(MODULE_CANDIDATE_ORDER)
        candidate_scope = "netB.bottleneck"
        bn_stats_policy = "frozen"
        candidate_names = list(MODULE_CANDIDATE_ORDER)
    else:
        selected_names = set(CONV_CANDIDATE_ORDER)
        candidate_scope = "netF.layer4_conv"
        bn_stats_policy = "frozen"
        candidate_names = list(CONV_CANDIDATE_ORDER)

    missing = sorted(selected_names - set(lookup))
    if missing:
        raise RuntimeError(f"IST candidate parameters not found: {missing}")
    selected_parameters = []
    for name, parameter in all_parameters:
        if name in selected_names:
            parameter.requires_grad = True
            selected_parameters.append((name, parameter))

    candidate_count = sum(
        parameter.numel() for _, parameter in selected_parameters
    )
    if track == "conv_out_channel":
        if candidate_count != CONV_CANDIDATE_PARAM_COUNT:
            raise RuntimeError(
                "IST Conv candidate count must match formal full-layer4 "
                f"scope ({CONV_CANDIDATE_PARAM_COUNT}), got {candidate_count}"
            )
        if variant in SPARSE_VARIANTS:
            actual_groups = sum(
                int(parameter.shape[0])
                for _, parameter in selected_parameters
            )
            if actual_groups != CONV_GROUP_COUNTS["out_channel"]:
                raise RuntimeError(
                    "IST Conv out-channel group pool must contain "
                    f"{CONV_GROUP_COUNTS['out_channel']} groups, "
                    f"got {actual_groups}"
                )
    if (
        track == "fc_scalar"
        and variant in SPARSE_VARIANTS
        and candidate_count != FC_CANDIDATE_PARAM_COUNT
    ):
        raise RuntimeError(
            "IST FC candidate count must be "
            f"{FC_CANDIDATE_PARAM_COUNT}, got {candidate_count}"
        )

    optimization = config["optimization"]
    parameter_groups = []
    for name, parameter in selected_parameters:
        multiplier = (
            optimization["lr_decay1"]
            if name.startswith("netF.")
            else optimization["lr_decay2"]
        )
        parameter_groups.append(
            {"params": parameter, "lr": optimization["lr"] * multiplier}
        )
    bn_modules = [
        module
        for model in (net_f, net_b, net_c)
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    requested_budget = float(config["requested_budget"])
    budget_count = (
        int(requested_budget * (
            CONV_GROUP_COUNTS["out_channel"]
            if track == "conv_out_channel"
            else FC_CANDIDATE_PARAM_COUNT
        ))
        if variant in SPARSE_VARIANTS else None
    )
    if variant in SPARSE_VARIANTS and budget_count not in (
        FORMAL_INTEGER_BUDGETS
        if track == "fc_scalar"
        else (4, 9, 18)
    ):
        raise RuntimeError("IST sparse integer budget is not protocol-valid")
    total_model_count = sum(
        parameter.numel() for _, parameter in all_parameters
    )
    return parameter_groups, {
        "selection": variant_selection(variant),
        "candidate_track": track,
        "candidate_scope": candidate_scope,
        "candidate_layer_names": candidate_names,
        "candidate_scope_param_count": int(candidate_count),
        "total_model_param_count": int(total_model_count),
        "selected_param_count": (
            int(candidate_count) if variant not in SPARSE_VARIANTS else None
        ),
        "requested_budget": requested_budget,
        "integer_budget": budget_count,
        "group_mode": (
            "out_channel"
            if track == "conv_out_channel" and variant in SPARSE_VARIANTS
            else None
        ),
        "total_group_count": (
            CONV_GROUP_COUNTS["out_channel"]
            if track == "conv_out_channel" else None
        ),
        "bn_stats_policy": bn_stats_policy,
        "bn_stats_frozen": variant != "ist_full_dense",
        "bn_module_count": len(bn_modules),
        "sparse_selector": (
            None if variant not in SPARSE_VARIANTS
            else variant_selection(variant)
        ),
        "mask_refresh_policy": (
            "once_before_target_stream"
            if variant in RANDOM_VARIANTS | MAGNITUDE_VARIANTS
            else "once_per_outer_batch_full_ist_objective"
            if variant in SALIENCY_VARIANTS
            else "once_per_outer_batch_via_split_lbi"
            if variant in LBI_VARIANTS
            else None
        ),
        "native_ist_ema": variant not in LBI_VARIANTS,
        "off_mask_exact_preservation": (
            True if variant in SPARSE_VARIANTS else None
        ),
        "persistent_writeback": (
            "lbi_omega_only"
            if variant in LBI_VARIANTS else "ist_native_ema"
        ),
    }

def set_inner_train_behavior(variant, net_f, net_b, net_c):
    if variant == "ist_full_dense":
        net_f.train()
        net_b.train()
    elif variant in FC_VARIANTS:
        net_f.train()
        net_b.train()
    elif variant in CONV_VARIANTS:
        net_f.train()
        net_b.eval()
    else:
        raise ValueError(f"Unsupported IST variant: {variant}")
    net_c.eval()
    if variant != "ist_full_dense":
        for model in (net_f, net_b, net_c):
            for module in model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()


@contextmanager
def _temporary_eval(*models):
    modes = [model.training for model in models]
    try:
        for model in models:
            model.eval()
        yield
    finally:
        for model, training in zip(models, modes):
            model.train(training)


def read_only_forward(inputs, net_f, net_b, net_c):
    """Evaluation forward that cannot mutate parameters or BN buffers."""
    with _temporary_eval(net_f, net_b, net_c), torch.no_grad():
        return net_c(net_b(net_f(inputs)))


def _pre_adaptation_outputs(inputs, net_f, net_b, net_c, batch_size):
    all_features = []
    all_soft_targets = []
    with _temporary_eval(net_f, net_b, net_c), torch.no_grad():
        for start in range(0, inputs.shape[0], int(batch_size)):
            batch = inputs[start : start + int(batch_size)]
            features = net_b(net_f(batch))
            logits = net_c(features)
            all_features.append(features.detach())
            all_soft_targets.append(torch.softmax(logits, dim=-1).detach())
    return torch.cat(all_features), torch.cat(all_soft_targets)


def _inner_self_training(
    config,
    inputs,
    hard_targets,
    soft_targets,
    net_f,
    net_b,
    net_c,
    optimizer,
    order_generator,
    candidate_parameters=None,
    fixed_masks=None,
):
    set_inner_train_behavior(config["variant"], net_f, net_b, net_c)
    batch_size = int(config["data"]["batch_size"])
    loss_sums = {"total": 0.0, "hard_ce": 0.0, "soft_kl": 0.0}
    sample_count = 0
    optimizer_steps = 0
    candidate_parameters = list(candidate_parameters or [])
    fixed_masks = fixed_masks or {}
    for _ in range(int(config["ist"]["iters"])):
        order = torch.randperm(inputs.shape[0], generator=order_generator)
        for start in range(0, inputs.shape[0], batch_size):
            indices = order[start : start + batch_size].to(inputs.device)
            optimizer.zero_grad()
            logits = net_c(net_b(net_f(inputs[indices])))
            total, hard_ce, soft_kl = _ist_loss(
                logits,
                hard_targets[indices],
                soft_targets[indices],
                config["loss"],
            )
            total.backward()
            if fixed_masks:
                masked_optimizer_step(
                    optimizer, candidate_parameters, fixed_masks
                )
            else:
                optimizer.step()
            count = int(indices.numel())
            loss_sums["total"] += float(total.item()) * count
            loss_sums["hard_ce"] += float(hard_ce.item()) * count
            loss_sums["soft_kl"] += float(soft_kl.item()) * count
            sample_count += count
            optimizer_steps += 1
    return {
        "loss": loss_sums["total"] / sample_count,
        "loss_hard_ce": loss_sums["hard_ce"] / sample_count,
        "loss_soft_kl": loss_sums["soft_kl"] / sample_count,
        "inner_optimizer_steps": optimizer_steps,
    }


def _should_skip_singleton_outer_batch(batch_size):
    """Match historical SHOT online semantics: skip only outer batches of size 1."""
    return int(batch_size) == 1


def _run(
    config,
    workspace_root,
    run_id=None,
    output_dir=None,
    summary_filename="summary.json",
    random_mask_index=None,
    random_child_mask_seed=None,
    resume_run_dir=None,
    enable_stream_checkpoint=False,
):
    if resume_run_dir and not enable_stream_checkpoint:
        raise ValueError("resume_run_dir requires stream checkpointing")
    if enable_stream_checkpoint and config["variant"] in LBI_VARIANTS:
        raise ValueError(
            "IST stream checkpointing is implemented for non-LBI baselines"
        )
    if config.get("formal_protocol") and config["output"].get("save_model"):
        raise ValueError("formal protocol requires save_model=false")
    debug_max_outer_batches = config.get("runtime", {}).get(
        "debug_max_outer_batches"
    )
    if config.get("formal_protocol") and debug_max_outer_batches is not None:
        raise ValueError("formal protocol rejects runtime.debug_max_outer_batches")
    debug_smoke = debug_max_outer_batches is not None
    started_at_utc = datetime.now(timezone.utc).isoformat()
    segment_started = time.perf_counter()
    setup_reproducibility(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, data_order = build_ist_loaders(config)
    (net_f, net_b, net_c), source_paths = load_source_models(config, device)
    source_hashes = _source_checkpoint_hashes(source_paths)
    parameter_groups, selection_stats = configure_ist_variant(
        config, net_f, net_b, net_c
    )
    optimizer = _build_optimizer(config, parameter_groups)
    all_parameter_lookup = dict(_all_named_parameters(net_f, net_b, net_c))
    candidate_parameters = [
        (name, all_parameter_lookup[name])
        for name in selection_stats["candidate_layer_names"]
    ]
    static_masks = {}
    if config["variant"] in RANDOM_VARIANTS | MAGNITUDE_VARIANTS:
        static_masks = build_static_masks(
            config["variant"],
            candidate_parameters,
            config["requested_budget"],
            seed=(
                random_child_mask_seed
                if random_child_mask_seed is not None
                else config.get("selection_seed")
            ),
        )
        selection_stats.update(
            mask_statistics(
                static_masks,
                candidate_parameters,
                selection_stats["candidate_track"],
                selection_stats["total_model_param_count"],
            )
        )
        selected_support = (
            selection_stats["selected_group_count"]
            if selection_stats["candidate_track"] == "conv_out_channel"
            else selection_stats["selected_param_count"]
        )
        selection_stats["support_utilization"] = float(
            selected_support / selection_stats["integer_budget"]
        )
    materializer = ISTViewMaterializer(config)
    memory = CausalMemoryBank(config["ist"]["memory"]["max_len"])
    ema = (
        None
        if config["variant"] in LBI_VARIANTS
        else OuterBatchEMA(config["ist"]["ema_momentum"])
    )
    lbi_executor = (
        ISTLBIExecutor(selection_stats["total_model_param_count"])
        if config["variant"] in LBI_VARIANTS
        else None
    )
    order_generator = torch.Generator(device="cpu").manual_seed(
        int(config["seed"])
        + int(config["ist"]["rng"]["inner_order_seed_offset"])
    )

    resume_payload = None
    if resume_run_dir is not None:
        if output_dir is not None or run_id is not None:
            raise ValueError(
                "resume_run_dir cannot be combined with an explicit output run"
            )
        output_dir = osp.abspath(resume_run_dir)
        resume_payload, _ = _load_ist_stream_checkpoint(
            output_dir,
            config,
            net_f,
            net_b,
            net_c,
            optimizer,
            source_hashes,
        )
        run_id = resume_payload["run_id"]
    elif output_dir is None:
        output_root = experiment_output_root(config)
        run_id, output_dir = create_run_dir(
            output_root,
            task_name=config["task_name"],
            run_name=config["output"].get("run_name"),
        )
    elif run_id is None:
        raise ValueError("explicit IST output_dir requires an explicit run_id")
    else:
        os.makedirs(output_dir, exist_ok=False)
    selection_stats = {
        **selection_stats,
        "source_checkpoint_sha256": source_hashes,
        "ist_execution_unit": "incoming_outer_batch",
        "ist_extend": config["ist"]["extend"],
        "ist_memory_max_len": config["ist"]["memory"]["max_len"],
        "ist_ema_momentum": (
            None if config["variant"] in LBI_VARIANTS
            else config["ist"]["ema_momentum"]
        ),
        "native_ist_ema": config["variant"] not in LBI_VARIANTS,
        "persistent_writeback": (
            "lbi_omega_only"
            if config["variant"] in LBI_VARIANTS else "ist_native_ema"
        ),
        "random_child_mask_count": (
            config.get("num_random_masks")
            if config["variant"] in RANDOM_VARIANTS else None
        ),
        "random_child_mask_seed": (
            random_child_mask_seed
            if config["variant"] in RANDOM_VARIANTS
            else None
        ),
        "random_mask_index": (
            random_mask_index
            if config["variant"] in RANDOM_VARIANTS
            else None
        ),
        "singleton_outer_batch_policy": "skip_size_1_to_match_shot_history",
        "debug_smoke": debug_smoke,
        "debug_max_outer_batches": debug_max_outer_batches,
    }
    if resume_payload is not None:
        artifact_paths = _existing_artifact_paths(
            output_dir, summary_filename
        )
    else:
        artifact_paths = write_initial_artifacts(
            output_dir,
            run_id,
            config,
            data_order,
            source_paths,
            selection_stats,
            workspace_root,
            summary_filename=summary_filename,
        )
    open(artifact_paths["metrics"], "a", encoding="utf-8").close()

    runtime_tracker = RuntimeInstrumentation(
        device,
        runtime_comparable=(
            config.get("runtime", {}).get("runtime_comparable", False)
            and config.get("runtime", {}).get("workers_per_gpu") == 1
        ),
    )
    all_post_predictions = []
    all_online_labels = []
    max_iterations = len(loaders["target"])
    processed_outer_batches = 0
    singleton_outer_batches_skipped = 0
    loader_batches_processed = 0
    wall_runtime_before_resume = 0.0
    resume_rng_state = None
    runtime_tracker.runtime_resume_used = resume_payload is not None
    if resume_payload is not None:
        loader_batches_processed = int(
            resume_payload["loader_batches_processed"]
        )
        processed_outer_batches = int(
            resume_payload["processed_outer_batches"]
        )
        singleton_outer_batches_skipped = int(
            resume_payload["singleton_outer_batches_skipped"]
        )
        if not 0 <= loader_batches_processed <= max_iterations:
            raise RuntimeError("invalid IST loader position in checkpoint")
        if processed_outer_batches != int(resume_payload["memory"]["commit_count"]):
            raise RuntimeError("IST checkpoint memory commit count mismatch")
        all_post_predictions = resume_payload["all_post_predictions"]
        all_online_labels = resume_payload["all_online_labels"]
        if not (
            len(all_post_predictions)
            == len(all_online_labels)
            == processed_outer_batches
        ):
            raise RuntimeError("IST checkpoint prediction history mismatch")
        materializer.load_state_dict(resume_payload["materializer"])
        memory.load_state_dict(resume_payload["memory"])
        if ema is not None:
            if resume_payload["ema"] is None:
                raise RuntimeError("IST checkpoint is missing EMA state")
            ema.load_state_dict(resume_payload["ema"])
        order_generator.set_state(resume_payload["order_generator"])
        if resume_payload.get("static_masks"):
            candidate_lookup = dict(candidate_parameters)
            static_masks = {
                name: mask.to(candidate_lookup[name].device)
                for name, mask in resume_payload["static_masks"].items()
            }
        restored_records = resume_payload.get("batch_efficiency_records", [])
        runtime_tracker.restore_batch_efficiency(restored_records)
        runtime_tracker.runtime_segment_count = int(
            resume_payload.get("runtime_segment_count", 1)
        ) + 1
        started_at_utc = resume_payload["started_at_utc"]
        wall_runtime_before_resume = float(
            resume_payload.get("wall_runtime_sec", 0.0)
        )
        resume_rng_state = resume_payload["rng_state"]
        _truncate_metrics_to_checkpoint(
            artifact_paths["metrics"], loader_batches_processed
        )
    elif enable_stream_checkpoint:
        _save_ist_stream_checkpoint(
            output_dir=output_dir,
            config=config,
            run_id=run_id,
            net_f=net_f,
            net_b=net_b,
            net_c=net_c,
            optimizer=optimizer,
            materializer=materializer,
            memory=memory,
            ema=ema,
            order_generator=order_generator,
            static_masks=static_masks,
            source_hashes=source_hashes,
            loader_batches_processed=0,
            processed_outer_batches=0,
            singleton_outer_batches_skipped=0,
            all_post_predictions=all_post_predictions,
            all_online_labels=all_online_labels,
            started_at_utc=started_at_utc,
            wall_runtime_sec=time.perf_counter() - segment_started,
            runtime_tracker=runtime_tracker,
        )
    progress = tqdm(loaders["target"], desc="IST-OTTA P1", dynamic_ncols=True)
    for batch_index, (raw_images, labels, sample_indices) in enumerate(progress):
        if batch_index < loader_batches_processed:
            continue
        if resume_rng_state is not None:
            _restore_rng_state(resume_rng_state)
            resume_rng_state = None
        if _should_skip_singleton_outer_batch(len(raw_images)):
            singleton_outer_batches_skipped += 1
            loader_batches_processed = batch_index + 1
            continue
        # State-bearing IST components count only actually processed outer
        # batches.  This remains identical to raw batch_index for all current
        # formal streams before any tail singleton, while staying robust if a
        # skipped singleton ever appears before a later valid batch.
        state_batch_index = processed_outer_batches
        batch = materializer.materialize(raw_images, labels, sample_indices)
        reference_views = batch.reference_views.to(device)
        adaptation_views = batch.adaptation_views.to(device)
        runtime_tracker.begin_batch(batch_index, len(raw_images))
        adaptation_started = runtime_tracker.start_adaptation()

        sparse_base_parameters = (
            {
                name: parameter.detach().clone()
                for name, parameter in candidate_parameters
            }
            if config["variant"] in SPARSE_VARIANTS - LBI_VARIANTS
            else {}
        )
        if ema is not None:
            ema.begin_batch(state_batch_index, (
                ("netF", net_f), ("netB", net_b)
            ))
        features, soft_targets = _pre_adaptation_outputs(
            adaptation_views,
            net_f,
            net_b,
            net_c,
            config["data"]["batch_size"],
        )
        memory_before = memory.snapshot_for(state_batch_index, device=device)
        plca_result = robust_plca(
            features,
            soft_targets,
            memory_before,
            config["ist"]["plca"],
        )
        memory.commit(
            state_batch_index, features, plca_result.corrected_one_hot
        )
        fixed_task = FixedISTTask(
            views=adaptation_views,
            hard_targets=plca_result.corrected_hard_labels.detach(),
            soft_targets=soft_targets.detach(),
        )
        step_selection_stats = selection_stats
        if config["variant"] in LBI_VARIANTS:
            accumulator = FullObjectiveGradientAccumulator(
                fixed_task,
                lambda inputs: net_c(net_b(net_f(inputs))),
                config["loss"],
                config["ist"]["objective_chunk_size"],
                prepare_forward=lambda: set_inner_train_behavior(
                    config["variant"], net_f, net_b, net_c
                ),
            )
            lbi_result = lbi_executor.run_outer_batch(
                candidate_parameters,
                accumulator,
                config["lbi_runtime"],
                timing=runtime_tracker,
            )
            loss_values = dict(lbi_result.statistics)
            step_selection_stats = {
                **selection_stats,
                **lbi_result.statistics,
                "full_objective_accumulator_calls": accumulator.call_count,
                "full_objective_chunk_backwards": (
                    accumulator.chunk_backward_count
                ),
            }
            current_lr = float(config["lbi"]["stage2_lr"])
        else:
            _schedule_learning_rate(
                config, optimizer, batch_index + 1, max_iterations
            )
            step_masks = static_masks
            if config["variant"] in SALIENCY_VARIANTS:
                accumulator = FullObjectiveGradientAccumulator(
                    fixed_task,
                    lambda inputs: net_c(net_b(net_f(inputs))),
                    config["loss"],
                    config["ist"]["objective_chunk_size"],
                    prepare_forward=lambda: set_inner_train_behavior(
                        config["variant"], net_f, net_b, net_c
                    ),
                )
                accumulator(candidate_parameters)
                step_masks = build_saliency_masks(
                    config["variant"],
                    candidate_parameters,
                    config["requested_budget"],
                )
                optimizer.zero_grad()
                step_selection_stats = {
                    **selection_stats,
                    **mask_statistics(
                        step_masks,
                        candidate_parameters,
                        selection_stats["candidate_track"],
                        selection_stats["total_model_param_count"],
                    ),
                    "saliency_support_selections": 1,
                    "support_utilization": 1.0,
                    "full_objective_accumulator_calls": (
                        accumulator.call_count
                    ),
                    "full_objective_chunk_backwards": (
                        accumulator.chunk_backward_count
                    ),
                }
            loss_values = _inner_self_training(
                config,
                fixed_task.views,
                fixed_task.hard_targets,
                fixed_task.soft_targets,
                net_f,
                net_b,
                net_c,
                optimizer,
                order_generator,
                candidate_parameters=candidate_parameters,
                fixed_masks=step_masks,
            )
            ema.commit(state_batch_index, (
                ("netF", net_f), ("netB", net_b)
            ))
            if step_masks:
                restore_offmask_values(
                    candidate_parameters,
                    step_masks,
                    sparse_base_parameters,
                )
            current_lr = float(optimizer.param_groups[0]["lr"])
        runtime_tracker.finish_adaptation(adaptation_started)

        pu_started = runtime_tracker.start_pu()
        post_outputs = read_only_forward(
            reference_views, net_f, net_b, net_c
        )
        post_predictions = post_outputs.argmax(dim=1)
        # Ground-truth labels enter only the read-only evaluation branch.
        labels_device = batch.labels.to(device)
        all_online_labels.append(batch.labels.cpu())
        runtime_tracker.finish_pu(pu_started)
        efficiency = runtime_tracker.finish_batch()
        all_post_predictions.append(post_predictions.cpu())
        processed_outer_batches += 1
        correct = int(torch.count_nonzero(post_predictions == labels_device))
        batch_accuracy = 100.0 * correct / max(1, labels_device.numel())
        append_jsonl(
            artifact_paths["metrics"],
            {
                "schema_version": SCHEMA_VERSION,
                "event": "online_step",
                "run_id": run_id,
                "method": "IST",
                "variant": config["variant"],
                "implementation_revision": config["implementation_revision"],
                "protocol_revision": config["protocol_revision"],
                "source_checkpoint_revision": config["source_checkpoint_revision"],
                "source_checkpoints": source_paths,
                "source_checkpoint_sha256": source_hashes,
                "experiment_key": config["experiment_key"],
                "experiment_config_sha256": config["experiment_config_sha256"],
                "iteration": batch_index + 1,
                "outer_batch_index": batch_index,
                "ist_state_batch_index": state_batch_index,
                "debug_smoke": debug_smoke,
                "debug_max_outer_batches": debug_max_outer_batches,
                "processed_outer_batches": processed_outer_batches,
                "singleton_outer_batches_skipped": (
                    singleton_outer_batches_skipped
                ),
                "sample_indices": batch.sample_indices.tolist(),
                "acc_post": batch_accuracy,
                "lr": current_lr,
                "memory_size": len(memory),
                "memory_commit_count": memory.commit_count,
                "ema_commit_count": (
                    ema.commit_count if ema is not None else 0
                ),
                "native_ist_ema_commit_count": (
                    ema.commit_count if ema is not None else 0
                ),
                "lbi_support_discovery_count": (
                    lbi_executor.support_discovery_count
                    if lbi_executor is not None else 0
                ),
                "lbi_omega_writeback_count": (
                    lbi_executor.omega_writeback_count
                    if lbi_executor is not None else 0
                ),
                "plca_call_count": processed_outer_batches,
                "plca_graph_sample_count": plca_result.graph_sample_count,
                "plca_memory_sample_count": plca_result.memory_sample_count,
                **loss_values,
                **{
                    key: efficiency[key]
                    for key in BATCH_EFFICIENCY_FIELDS
                    if key in efficiency
                },
                **step_selection_stats,
            },
        )
        loader_batches_processed = batch_index + 1
        if enable_stream_checkpoint:
            _save_ist_stream_checkpoint(
                output_dir=output_dir,
                config=config,
                run_id=run_id,
                net_f=net_f,
                net_b=net_b,
                net_c=net_c,
                optimizer=optimizer,
                materializer=materializer,
                memory=memory,
                ema=ema,
                order_generator=order_generator,
                static_masks=static_masks,
                source_hashes=source_hashes,
                loader_batches_processed=loader_batches_processed,
                processed_outer_batches=processed_outer_batches,
                singleton_outer_batches_skipped=(
                    singleton_outer_batches_skipped
                ),
                all_post_predictions=all_post_predictions,
                all_online_labels=all_online_labels,
                started_at_utc=started_at_utc,
                wall_runtime_sec=(
                    wall_runtime_before_resume
                    + time.perf_counter() - segment_started
                ),
                runtime_tracker=runtime_tracker,
            )
        progress.set_description(
            f"IST-OTTA | Post={batch_accuracy:.2f}% "
            f"Loss={loss_values['loss']:.4f}"
        )
        del adaptation_views, reference_views, features, soft_targets
        if (
            debug_max_outer_batches is not None
            and processed_outer_batches >= debug_max_outer_batches
        ):
            break

    if resume_rng_state is not None:
        _restore_rng_state(resume_rng_state)
    online_predictions = torch.cat(all_post_predictions).numpy()
    online_labels = torch.cat(all_online_labels).numpy()
    pu_metrics = _compute_dataset_metrics(
        online_labels, online_predictions, config["data"]["dataset"], "PU"
    )
    net_f.eval()
    net_b.eval()
    net_c.eval()
    fo_metrics = runtime_tracker.measure_fo(
        lambda: _evaluate(
            loaders["test"], net_f, net_b, net_c, device,
            config["data"]["dataset"],
        )
    )
    checkpoints = None
    if config["output"]["save_model"]:
        checkpoints = _save_model(output_dir, net_f, net_b, net_c)
    wall_runtime = float(
        wall_runtime_before_resume + time.perf_counter() - segment_started
    )
    runtime_metadata = runtime_tracker.metadata()
    completed_at_utc = datetime.now(timezone.utc).isoformat()
    final = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "run_id": run_id,
        "output_dir": output_dir,
        "method": "IST",
        "variant": config["variant"],
        "task": config["task"],
        "dataset": config["data"]["dataset"],
        "source": config["data"]["source"],
        "target": config["data"]["target"],
        "source-target": (
            f"{config['data']['source_name']}-{config['data']['target_name']}"
        ),
        "seed": config["seed"],
        "implementation_revision": config["implementation_revision"],
        "protocol_revision": config["protocol_revision"],
        "source_checkpoint_revision": config["source_checkpoint_revision"],
        "source_checkpoints": source_paths,
        "source_checkpoint_sha256": source_hashes,
        "experiment_key": config["experiment_key"],
        "experiment_config_sha256": config["experiment_config_sha256"],
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "online_steps": len(all_post_predictions),
        "debug_smoke": debug_smoke,
        "debug_max_outer_batches": debug_max_outer_batches,
        "processed_outer_batches": processed_outer_batches,
        "singleton_outer_batches_skipped": singleton_outer_batches_skipped,
        "memory_commit_count": memory.commit_count,
        "ema_commit_count": ema.commit_count if ema is not None else 0,
        "native_ist_ema_commit_count": (
            ema.commit_count if ema is not None else 0
        ),
        "lbi_support_discovery_count": (
            lbi_executor.support_discovery_count
            if lbi_executor is not None else 0
        ),
        "lbi_omega_writeback_count": (
            lbi_executor.omega_writeback_count
            if lbi_executor is not None else 0
        ),
        "random_mask_index": random_mask_index,
        "random_child_mask_seed": random_child_mask_seed,
        "plca_call_count": processed_outer_batches,
        "PU-Acc": pu_metrics["PU-Acc"],
        "PU-Acc-per-class": pu_metrics["PU-Acc-per-class"],
        "FO-Acc": fo_metrics["FO-Acc"],
        "FO-Acc-per-class": fo_metrics["FO-Acc-per-class"],
        "wall_runtime_sec": wall_runtime,
        "runtime_resume_used": runtime_tracker.runtime_resume_used,
        "runtime_segment_count": runtime_tracker.runtime_segment_count,
        "stream_checkpoint_enabled": bool(enable_stream_checkpoint),
        "checkpoints": checkpoints,
        "runtime": wall_runtime,
        **pu_metrics,
        **fo_metrics,
        **runtime_metadata,
        **selection_stats,
    }
    append_jsonl(
        artifact_paths["metrics"], {"event": "final", **final}
    )
    dump_json(artifact_paths["summary"], final)
    _update_manifest_runtime(
        artifact_paths,
        {
            **runtime_metadata,
            "debug_smoke": debug_smoke,
            "debug_max_outer_batches": debug_max_outer_batches,
            "processed_outer_batches": processed_outer_batches,
            "singleton_outer_batches_skipped": singleton_outer_batches_skipped,
        },
        wall_runtime,
    )
    if enable_stream_checkpoint:
        checkpoint_paths = _stream_checkpoint_paths(output_dir)
        _atomic_json_dump(
            checkpoint_paths["metadata"],
            {
                "schema_version": IST_STREAM_CHECKPOINT_SCHEMA_VERSION,
                "status": "completed",
                "checkpoint_semantics": "completed_outer_batch_boundary",
                "run_id": run_id,
                "implementation_revision": config[
                    "implementation_revision"
                ],
                "protocol_revision": config["protocol_revision"],
                "experiment_key": config["experiment_key"],
                "experiment_config_sha256": config[
                    "experiment_config_sha256"
                ],
                "loader_batches_processed": int(loader_batches_processed),
                "processed_outer_batches": int(processed_outer_batches),
                "runtime_resume_used": runtime_tracker.runtime_resume_used,
                "runtime_segment_count": runtime_tracker.runtime_segment_count,
                "updated_at_utc": completed_at_utc,
            },
        )
    print(f"Run complete: {output_dir}")
    return final



def _random_mask_seed(selection_seed, mask_index):
    return int(selection_seed) * 100 + int(mask_index)


def _write_ist_random_summary_markdown(path, summary):
    lines = [
        "# IST Random baseline summary",
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


def _run_random_experiment(
    config,
    workspace_root,
    resume_run_dir=None,
    enable_stream_checkpoint=False,
):
    """Run the protocol-required three independent IST Random child masks."""
    started_at_utc = datetime.now(timezone.utc).isoformat()
    started_at = time.perf_counter()
    random_state_path = None
    runtime_before_resume = 0.0
    if resume_run_dir is not None:
        output_dir = osp.abspath(resume_run_dir)
        run_id = osp.basename(output_dir.rstrip(os.sep))
        random_state_path = osp.join(output_dir, "random_stream_state.json")
        if osp.isfile(random_state_path):
            with open(random_state_path, "r", encoding="utf-8") as file_obj:
                random_state = json.load(file_obj)
            started_at_utc = random_state["started_at_utc"]
            runtime_before_resume = float(
                random_state.get("elapsed_runtime_sec", 0.0)
            )
    else:
        output_root = experiment_output_root(config)
        run_id, output_dir = create_run_dir(
            output_root,
            task_name=config["task_name"],
            run_name=config["output"].get("run_name"),
        )
        if enable_stream_checkpoint:
            random_state_path = osp.join(output_dir, "random_stream_state.json")
            _atomic_json_dump(
                random_state_path,
                {
                    "schema_version": IST_STREAM_CHECKPOINT_SCHEMA_VERSION,
                    "status": "in_progress",
                    "run_id": run_id,
                    "experiment_key": config["experiment_key"],
                    "experiment_config_sha256": config[
                        "experiment_config_sha256"
                    ],
                    "started_at_utc": started_at_utc,
                    "elapsed_runtime_sec": 0.0,
                    "completed_masks": [],
                },
            )
    run_seed = int(config["seed"])
    selection_seed = int(config["selection_seed"])
    num_masks = int(config["num_random_masks"])
    if num_masks != 3:
        raise RuntimeError("IST Random protocol requires exactly three masks")

    child_results = []
    mask_records = []
    for mask_index in range(num_masks):
        mask_id = f"mask_{mask_index:02d}"
        mask_seed = _random_mask_seed(selection_seed, mask_index)
        mask_dir = osp.join(output_dir, mask_id)
        result_path = osp.join(mask_dir, "results.json")
        checkpoint_state = _stream_checkpoint_paths(mask_dir)["state"]
        if osp.isfile(result_path):
            with open(result_path, "r", encoding="utf-8") as file_obj:
                child = json.load(file_obj)
            if child.get("status") != "completed":
                raise RuntimeError(
                    f"IST Random child result is not completed: {result_path}"
                )
        elif resume_run_dir is not None and osp.isfile(checkpoint_state):
            child = _run(
                config,
                workspace_root,
                summary_filename="results.json",
                random_mask_index=mask_index,
                random_child_mask_seed=mask_seed,
                resume_run_dir=mask_dir,
                enable_stream_checkpoint=True,
            )
        else:
            if osp.isdir(mask_dir):
                orphan_dir = (
                    f"{mask_dir}.pre_checkpoint_interrupted."
                    f"{int(time.time())}"
                )
                os.replace(mask_dir, orphan_dir)
            child = _run(
                config,
                workspace_root,
                run_id=f"{run_id}/{mask_id}",
                output_dir=mask_dir,
                summary_filename="results.json",
                random_mask_index=mask_index,
                random_child_mask_seed=mask_seed,
                enable_stream_checkpoint=enable_stream_checkpoint,
            )
        child_results.append(child)
        mask_records.append(
            {
                "mask_id": mask_id,
                "mask_index": mask_index,
                "mask_seed": mask_seed,
                "PU-Acc": child["PU-Acc"],
                "FO-Acc": child["FO-Acc"],
                "runtime": child.get("runtime"),
                "online_compute_runtime_sec": child.get(
                    "online_compute_runtime_sec"
                ),
                "fo_eval_runtime_sec": child.get("fo_eval_runtime_sec"),
                "online_batch_runtime_mean_sec": child.get(
                    "online_batch_runtime_mean_sec"
                ),
                "adapt_batch_runtime_mean_sec": child.get(
                    "adapt_batch_runtime_mean_sec"
                ),
                "adapt_batch_runtime_std_sec": child.get(
                    "adapt_batch_runtime_std_sec"
                ),
                "adapt_batch_runtime_median_sec": child.get(
                    "adapt_batch_runtime_median_sec"
                ),
                "adapt_batch_runtime_p95_sec": child.get(
                    "adapt_batch_runtime_p95_sec"
                ),
                "adapt_runtime_total_sec": child.get(
                    "adapt_runtime_total_sec"
                ),
                "pu_batch_runtime_mean_sec": child.get(
                    "pu_batch_runtime_mean_sec"
                ),
                "pu_batch_runtime_std_sec": child.get(
                    "pu_batch_runtime_std_sec"
                ),
                "pu_batch_runtime_median_sec": child.get(
                    "pu_batch_runtime_median_sec"
                ),
                "pu_batch_runtime_p95_sec": child.get(
                    "pu_batch_runtime_p95_sec"
                ),
                "pu_runtime_total_sec": child.get("pu_runtime_total_sec"),
                "gpu_peak_allocated_mean_mb": child.get(
                    "gpu_peak_allocated_mean_mb"
                ),
                "gpu_peak_reserved_mean_mb": child.get(
                    "gpu_peak_reserved_mean_mb"
                ),
                "gpu_peak_allocated_max_mb": child.get(
                    "gpu_peak_allocated_max_mb"
                ),
                "gpu_peak_reserved_max_mb": child.get(
                    "gpu_peak_reserved_max_mb"
                ),
                "peak_gpu_memory_allocated_mb": child.get(
                    "peak_gpu_memory_allocated_mb"
                ),
                "peak_gpu_memory_reserved_mb": child.get(
                    "peak_gpu_memory_reserved_mb"
                ),
                "peak_gpu_memory_allocated_bytes": child.get(
                    "peak_gpu_memory_allocated_bytes"
                ),
                "peak_gpu_memory_reserved_bytes": child.get(
                    "peak_gpu_memory_reserved_bytes"
                ),
                "gpu_name": child.get("gpu_name"),
                "gpu_device_index": child.get("gpu_device_index"),
                "gpu_total_memory_bytes": child.get("gpu_total_memory_bytes"),
                "torch_version": child.get("torch_version"),
                "cuda_version": child.get("cuda_version"),
                "runtime_comparable": child.get("runtime_comparable", False),
                "selected_param_count": child.get("selected_param_count"),
                "selected_group_count": child.get("selected_group_count"),
                "selected_scalar_count": child.get("selected_scalar_count"),
                "realized_group_ratio": child.get("realized_group_ratio"),
                "realized_scalar_ratio": child.get("realized_scalar_ratio"),
                "result_path": osp.join(mask_dir, "results.json"),
                **{
                    key: child[key]
                    for key in child
                    if key.startswith("PU-")
                    or key.startswith("FO-")
                    or key == "class-names"
                },
            }
        )
        if random_state_path is not None:
            _atomic_json_dump(
                random_state_path,
                {
                    "schema_version": IST_STREAM_CHECKPOINT_SCHEMA_VERSION,
                    "status": "in_progress",
                    "run_id": run_id,
                    "experiment_key": config["experiment_key"],
                    "experiment_config_sha256": config[
                        "experiment_config_sha256"
                    ],
                    "started_at_utc": started_at_utc,
                    "elapsed_runtime_sec": float(
                        runtime_before_resume
                        + time.perf_counter() - started_at
                    ),
                    "completed_masks": [
                        item["mask_id"] for item in mask_records
                    ],
                },
            )

    pu_values = [float(item["PU-Acc"]) for item in child_results]
    fo_values = [float(item["FO-Acc"]) for item in child_results]
    random_efficiency = aggregate_random_efficiency(mask_records)
    completed_at_utc = datetime.now(timezone.utc).isoformat()
    total_runtime = float(
        runtime_before_resume + time.perf_counter() - started_at
    )
    reference = child_results[-1]
    mask_seeds = [item["mask_seed"] for item in mask_records]
    if len(set(mask_seeds)) != 3:
        raise RuntimeError("IST Random child mask seeds must be distinct")
    peak_allocated_mb = [
        float(item["peak_gpu_memory_allocated_mb"])
        for item in mask_records
        if item.get("peak_gpu_memory_allocated_mb") is not None
    ]
    peak_reserved_mb = [
        float(item["peak_gpu_memory_reserved_mb"])
        for item in mask_records
        if item.get("peak_gpu_memory_reserved_mb") is not None
    ]
    peak_allocated_bytes = [
        int(item["peak_gpu_memory_allocated_bytes"])
        for item in mask_records
        if item.get("peak_gpu_memory_allocated_bytes") is not None
    ]
    peak_reserved_bytes = [
        int(item["peak_gpu_memory_reserved_bytes"])
        for item in mask_records
        if item.get("peak_gpu_memory_reserved_bytes") is not None
    ]

    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "run_id": run_id,
        "output_dir": output_dir,
        "method": "IST",
        "variant": config["variant"],
        "task": config["task"],
        "dataset": config["data"]["dataset"],
        "source": config["data"]["source"],
        "target": config["data"]["target"],
        "source-target": (
            f"{config['data']['source_name']}-{config['data']['target_name']}"
        ),
        "seed": run_seed,
        "run_seed": run_seed,
        "implementation_revision": config["implementation_revision"],
        "protocol_revision": config["protocol_revision"],
        "source_checkpoint_revision": config["source_checkpoint_revision"],
        "source_checkpoints": reference.get("source_checkpoints"),
        "source_checkpoint_sha256": reference.get("source_checkpoint_sha256"),
        "experiment_key": config["experiment_key"],
        "experiment_config_sha256": config["experiment_config_sha256"],
        "requested_budget": float(config["requested_budget"]),
        "integer_budget": reference.get("integer_budget"),
        "selection_seed": selection_seed,
        "num_random_masks": num_masks,
        "mask_seeds": mask_seeds,
        "masks": mask_records,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "runtime": total_runtime,
        "total_runtime": total_runtime,
        "wall_runtime_sec": total_runtime,
        "runtime_comparable": all(
            item.get("runtime_comparable", False) for item in mask_records
        ),
        "runtime_resume_used": bool(
            resume_run_dir is not None
            or any(item.get("runtime_resume_used", False) for item in child_results)
        ),
        "runtime_segment_count": 1 + int(resume_run_dir is not None),
        "efficiency_protocol_revision": reference.get(
            "efficiency_protocol_revision"
        ),
        "gpu_name": reference.get("gpu_name"),
        "gpu_device_index": reference.get("gpu_device_index"),
        "gpu_total_memory_bytes": reference.get("gpu_total_memory_bytes"),
        "torch_version": reference.get("torch_version"),
        "cuda_version": reference.get("cuda_version"),
        "peak_gpu_memory_allocated_mb": (
            max(peak_allocated_mb) if peak_allocated_mb else None
        ),
        "peak_gpu_memory_reserved_mb": (
            max(peak_reserved_mb) if peak_reserved_mb else None
        ),
        "peak_gpu_memory_allocated_bytes": (
            max(peak_allocated_bytes) if peak_allocated_bytes else None
        ),
        "peak_gpu_memory_reserved_bytes": (
            max(peak_reserved_bytes) if peak_reserved_bytes else None
        ),
        "processed_outer_batches": reference.get("processed_outer_batches"),
        "singleton_outer_batches_skipped": reference.get(
            "singleton_outer_batches_skipped"
        ),
        "debug_smoke": reference.get("debug_smoke", False),
        "debug_max_outer_batches": reference.get("debug_max_outer_batches"),
        "mean_PU-Acc": float(np.mean(pu_values)),
        "std_PU-Acc": float(np.std(pu_values, ddof=0)),
        "mean_FO-Acc": float(np.mean(fo_values)),
        "std_FO-Acc": float(np.std(fo_values, ddof=0)),
        "PU-Acc": float(np.mean(pu_values)),
        "FO-Acc": float(np.mean(fo_values)),
        **random_efficiency,
    }

    # Shared exact-K fields.
    for field in (
        "selection", "candidate_track", "candidate_scope",
        "candidate_scope_param_count", "total_model_param_count",
        "candidate_layer_names", "group_mode", "total_group_count",
        "bn_stats_policy", "bn_stats_frozen", "native_ist_ema",
        "persistent_writeback", "mask_refresh_policy",
    ):
        summary[field] = reference.get(field)

    group_counts = [
        int(item["selected_group_count"])
        for item in child_results
        if item.get("selected_group_count") is not None
    ]
    scalar_counts = [
        int(item["selected_scalar_count"])
        for item in child_results
        if item.get("selected_scalar_count") is not None
    ]
    scalar_ratios = [
        float(item["realized_scalar_ratio"])
        for item in child_results
        if item.get("realized_scalar_ratio") is not None
    ]
    group_ratios = [
        float(item["realized_group_ratio"])
        for item in child_results
        if item.get("realized_group_ratio") is not None
    ]
    if group_counts:
        if len(set(group_counts)) != 1:
            raise RuntimeError(
                "IST Conv Random child masks disagree on exact group count"
            )
        summary["selected_group_count"] = group_counts[0]
        summary["realized_group_ratio"] = float(np.mean(group_ratios))
    if scalar_counts:
        summary["selected_param_count"] = float(np.mean(scalar_counts))
        summary["selected_scalar_count"] = float(np.mean(scalar_counts))
        summary["selected_scalar_count_mean"] = float(np.mean(scalar_counts))
        summary["selected_scalar_count_std"] = float(
            np.std(scalar_counts, ddof=0)
        )
        summary["selected_scalar_count_min"] = int(min(scalar_counts))
        summary["selected_scalar_count_max"] = int(max(scalar_counts))
    if scalar_ratios:
        scalar_mean = float(np.mean(scalar_ratios))
        summary["realized_scalar_ratio"] = scalar_mean
        summary["selected_over_scope_ratio"] = scalar_mean
        summary["realized_scalar_ratio_std"] = float(
            np.std(scalar_ratios, ddof=0)
        )
        summary["realized_scalar_ratio_min"] = float(min(scalar_ratios))
        summary["realized_scalar_ratio_max"] = float(max(scalar_ratios))
        candidate_count = summary.get("candidate_scope_param_count")
        total_count = summary.get("total_model_param_count")
        if candidate_count and total_count:
            summary["selected_over_model_ratio"] = float(
                scalar_mean * candidate_count / total_count
            )

    if config["data"]["dataset"] == "VISDA-C":
        from visda_otta.evaluator import aggregate_random_mask_metrics

        summary.update(aggregate_random_mask_metrics(mask_records))

    dump_json(osp.join(output_dir, "summary.json"), summary)
    _write_ist_random_summary_markdown(
        osp.join(output_dir, "summary.md"), summary
    )
    if random_state_path is not None:
        _atomic_json_dump(
            random_state_path,
            {
                "schema_version": IST_STREAM_CHECKPOINT_SCHEMA_VERSION,
                "status": "completed",
                "run_id": run_id,
                "experiment_key": config["experiment_key"],
                "experiment_config_sha256": config[
                    "experiment_config_sha256"
                ],
                "started_at_utc": started_at_utc,
                "elapsed_runtime_sec": total_runtime,
                "completed_masks": [item["mask_id"] for item in mask_records],
                "updated_at_utc": completed_at_utc,
            },
        )
    print(f"IST Random baseline complete: {output_dir}")
    return summary


def run_experiment(
    config,
    workspace_root,
    resume_run_dir=None,
    enable_stream_checkpoint=False,
):
    if config["variant"] in RANDOM_VARIANTS:
        return _run_random_experiment(
            config,
            workspace_root,
            resume_run_dir=resume_run_dir,
            enable_stream_checkpoint=enable_stream_checkpoint,
        )
    return _run(
        config,
        workspace_root,
        resume_run_dir=resume_run_dir,
        enable_stream_checkpoint=enable_stream_checkpoint,
    )
