"""Execution of controlled COME FC/Conv sparse and LBI trajectories."""

import hashlib
import os
import os.path as osp
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from protocol_constants import (
    COME_IMPLEMENTATION_REVISION,
    COME_PROTOCOL_REVISION,
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
    write_initial_artifacts,
)
from shot_otta.candidates import CONV_CANDIDATE_ORDER, MODULE_CANDIDATE_ORDER
from shot_otta.efficiency import BATCH_EFFICIENCY_FIELDS, aggregate_random_efficiency
from shot_otta.trainer import (
    RuntimeInstrumentation,
    _build_optimizer,
    _compute_dataset_metrics,
    _schedule_learning_rate,
    _update_manifest_runtime,
    setup_reproducibility,
)

from .lbi import COMELBIExecutor
from .objective import come_loss
from .sparse import (
    LBI_VARIANTS,
    MAGNITUDE_VARIANTS,
    RANDOM_VARIANTS,
    SALIENCY_VARIANTS,
    build_saliency_masks,
    build_static_masks,
    mask_statistics,
    masked_optimizer_step,
    variant_selection,
    variant_track,
)
from .trainer import (
    COMEObjectiveError,
    _assert_read_only,
    _bn_state_names,
    _evaluate_with_diagnostics,
    _read_only_guard,
    _should_skip_singleton_outer_batch,
    _snapshot_state,
    _source_checkpoint_hashes,
    _state_transition_audit,
    read_only_post_update_forward,
)


def configure_come_sparse_variant(config, net_f, net_b, net_c):
    variant = config["variant"]
    track = variant_track(variant)
    names = list(
        MODULE_CANDIDATE_ORDER if track == "fc_scalar" else CONV_CANDIDATE_ORDER
    )
    all_named = [
        (f"{prefix}.{name}", parameter)
        for prefix, model in (("netF", net_f), ("netB", net_b), ("netC", net_c))
        for name, parameter in model.named_parameters()
    ]
    lookup = dict(all_named)
    for _, parameter in all_named:
        parameter.requires_grad = False
    missing = sorted(set(names) - set(lookup))
    if missing:
        raise RuntimeError(f"COME sparse candidates not found: {missing}")
    candidates = [(name, lookup[name]) for name in names]
    for _, parameter in candidates:
        parameter.requires_grad = True
    candidate_count = sum(parameter.numel() for _, parameter in candidates)
    expected = (
        FC_CANDIDATE_PARAM_COUNT if track == "fc_scalar" else CONV_CANDIDATE_PARAM_COUNT
    )
    if candidate_count != expected:
        raise RuntimeError(
            f"COME {track} candidate count must be {expected}, got {candidate_count}"
        )
    group_total = (
        None
        if track == "fc_scalar"
        else sum(int(parameter.shape[0]) for _, parameter in candidates)
    )
    if group_total is not None and group_total != CONV_GROUP_COUNTS["out_channel"]:
        raise RuntimeError("COME Conv out-channel pool must contain 9216 groups")
    integer_budget = int(
        float(config["requested_budget"]) * (group_total or candidate_count)
    )
    allowed = FORMAL_INTEGER_BUDGETS if track == "fc_scalar" else (4, 9, 18)
    if integer_budget not in allowed:
        raise RuntimeError("COME sparse integer budget is not protocol-valid")
    optimization = config["optimization"]
    parameter_groups = [
        {
            "params": parameter,
            "lr": optimization["lr"]
            * (
                optimization["lr_decay1"]
                if name.startswith("netF.")
                else optimization["lr_decay2"]
            ),
        }
        for name, parameter in candidates
    ]
    total_count = sum(parameter.numel() for _, parameter in all_named)
    set_sparse_train_behavior(variant, net_f, net_b, net_c)
    selection = variant_selection(variant)
    return (
        parameter_groups,
        candidates,
        {
            "selection": selection,
            "sparse_selector": selection,
            "selector_definition": {
                "random": "uniform_global_exact_budget_child_mask",
                "magnitude": "source_checkpoint_global_magnitude_once",
                "saliency": "current_state_parameter_times_come_gradient_once_per_batch",
                "lbi": "split_lbi_thresholded_gamma_strict_rollback",
            }[selection],
            "candidate_track": track,
            "candidate_scope": (
                "netB.bottleneck" if track == "fc_scalar" else "netF.layer4_conv"
            ),
            "candidate_layer_names": names,
            "candidate_tensor_count": len(candidates),
            "candidate_scope_param_count": candidate_count,
            "candidate_scalar_count": candidate_count,
            "total_model_param_count": total_count,
            "requested_budget": float(config["requested_budget"]),
            "integer_budget": integer_budget,
            "group_mode": "out_channel" if track == "conv_out_channel" else None,
            "total_group_count": group_total,
            "bn_stats_policy": "frozen",
            "bn_stats_frozen": True,
            "mask_refresh_policy": (
                "once_before_target_stream"
                if variant in RANDOM_VARIANTS | MAGNITUDE_VARIANTS
                else "once_per_valid_outer_batch_current_state_come_objective"
                if variant in SALIENCY_VARIANTS
                else "once_per_valid_outer_batch_via_split_lbi"
            ),
            "persistent_writeback": (
                "lbi_omega_only" if variant in LBI_VARIANTS else "masked_host_optimizer"
            ),
            "host_optimizer_persistent_step": variant not in LBI_VARIANTS,
            "off_mask_exact_preservation": True,
            "parent_come_baseline_protocol_revision": COME_PROTOCOL_REVISION,
            "come_lbi_protocol_revision": config["protocol_revision"],
            "come_baseline_implementation_revision": COME_IMPLEMENTATION_REVISION,
            "official_come_commit": config["come"]["official_commit"],
            "lbi_omega": (
                float(config["lbi"]["omega"]) if variant in LBI_VARIANTS else None
            ),
        },
    )


def set_sparse_train_behavior(variant, net_f, net_b, net_c):
    if variant.startswith("come_fc_"):
        net_f.train()
        net_b.train()
    elif variant.startswith("come_conv_"):
        net_f.train()
        net_b.eval()
    else:
        raise ValueError("not a controlled COME sparse variant: " + variant)
    net_c.eval()
    for model in (net_f, net_b, net_c):
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()


def _objective_closure(config, inputs, net_f, net_b, net_c, trace):
    def closure():
        set_sparse_train_behavior(config["variant"], net_f, net_b, net_c)
        logits = net_c(net_b(net_f(inputs)))
        result = come_loss(
            logits,
            config["model"]["class_num"],
            p=config["come"]["p"],
            tau=config["come"]["tau"],
        )
        record = {"call_index": len(trace) + 1, **result.diagnostics}
        trace.append(record)
        if not result.diagnostics["finite_loss"]:
            raise COMEObjectiveError("COME sparse loss is non-finite", record)
        return result.loss, result.diagnostics

    return closure


def adapt_sparse_one_batch(
    config,
    inputs,
    net_f,
    net_b,
    net_c,
    optimizer,
    candidates,
    static_masks,
    lbi_executor,
    iteration,
    max_iterations,
    timing=None,
):
    trace = []
    closure = _objective_closure(config, inputs, net_f, net_b, net_c, trace)
    if config["variant"] in LBI_VARIANTS:
        result = lbi_executor.run_outer_batch(
            candidates, closure, config["lbi_runtime"], timing=timing
        )
        diagnostics = dict(result.statistics)
        diagnostics["objective_current_state_trace"] = trace
        diagnostics["optimizer_step_count"] = 0
        diagnostics["scheduler_step_count"] = 0
        return diagnostics

    _schedule_learning_rate(config, optimizer, iteration, max_iterations)
    optimizer.zero_grad()
    loss, parts = closure()
    loss.backward()
    gradients = [
        parameter.grad for _, parameter in candidates if parameter.grad is not None
    ]
    finite_gradients = len(gradients) == len(candidates) and all(
        bool(gradient.isfinite().all().item()) for gradient in gradients
    )
    if not finite_gradients:
        raise COMEObjectiveError(
            "COME sparse gradients are missing or non-finite",
            {"objective_current_state_trace": trace},
        )
    masks = static_masks
    saliency_count = 0
    saliency_reuse = False
    if config["variant"] in SALIENCY_VARIANTS:
        state_guard = _read_only_guard(net_f, net_b, net_c)
        cpu_rng_before = torch.random.get_rng_state()
        cuda_rng_before = (
            torch.cuda.get_rng_state(inputs.device)
            if inputs.device.type == "cuda"
            else None
        )
        objective_calls_before = len(trace)
        masks = build_saliency_masks(
            config["variant"], candidates, config["requested_budget"]
        )
        _assert_read_only(state_guard, net_f, net_b, net_c)
        if not torch.equal(cpu_rng_before, torch.random.get_rng_state()):
            raise RuntimeError("COME Saliency selection changed CPU RNG state")
        if cuda_rng_before is not None and not torch.equal(
            cuda_rng_before, torch.cuda.get_rng_state(inputs.device)
        ):
            raise RuntimeError("COME Saliency selection changed CUDA RNG state")
        if len(trace) != objective_calls_before:
            raise RuntimeError("COME Saliency selection changed objective state")
        saliency_count = 1
        saliency_reuse = True
    stats = mask_statistics(
        masks,
        candidates,
        variant_track(config["variant"]),
        sum(
            parameter.numel()
            for model in (net_f, net_b, net_c)
            for parameter in model.parameters()
        ),
    )
    if (
        stats.get("selected_group_count", stats["selected_scalar_count"])
        != config["scientific_config"]["integer_budget"]
    ):
        raise RuntimeError("COME selector did not produce the exact global budget")
    masked_optimizer_step(optimizer, candidates, masks)
    if not all(
        bool(parameter.detach().isfinite().all().item()) for _, parameter in candidates
    ):
        raise COMEObjectiveError(
            "COME sparse parameter update is non-finite",
            {"objective_current_state_trace": trace},
        )
    return {
        **parts,
        **stats,
        "loss": float(loss.detach().item()),
        "finite_loss": True,
        "finite_gradients": True,
        "finite_parameter_update": True,
        "objective_call_count": 1,
        "objective_current_state_trace": trace,
        "optimizer_step_count": 1,
        "scheduler_step_count": 1,
        "saliency_support_selections": saliency_count,
        "saliency_gradient_reused_without_state_change": saliency_reuse,
        "lbi_local_restart_count": 0,
        "lbi_support_discovery_count": 0,
        "lbi_omega_writeback_count": 0,
        "stage2_optimizer_instance_count": 0,
        "host_optimizer_persistent_step_count": 1,
        "persistent_writeback": "masked_host_optimizer",
        "support_utilization": 1.0,
        "off_mask_exact_preservation": True,
        "lr": float(optimizer.param_groups[0]["lr"]),
    }


def _output_root(config):
    return osp.join(
        config["output"]["root"],
        config["data"]["dataset"],
        config["task_name"],
        "COME",
        config["variant"],
        f"seed_{config['seed']}",
    )


def _stream_trace_sha256(step_records):
    payload = ",".join(
        str(index) for record in step_records for index in record["sample_indices"]
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _run(
    config,
    workspace_root,
    run_id=None,
    output_dir=None,
    summary_filename="summary.json",
    random_mask_index=None,
    random_child_mask_seed=None,
):
    from . import trainer as base

    formal_run = bool(config["formal_protocol"])
    debug_limit = config.get("runtime", {}).get("debug_max_outer_batches")
    started_at_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    setup_reproducibility(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, data_order = base.build_loaders(config)
    (net_f, net_b, net_c), source_paths = base.load_source_models(config, device)
    source_hashes = _source_checkpoint_hashes(source_paths)
    for name, record in source_hashes.items():
        expected = config["source_checkpoint_identity"][name]
        if (
            record["path"] != expected["resolved_path"]
            or record["sha256"] != expected["sha256"]
        ):
            raise RuntimeError(f"COME sparse source identity mismatch for {name}")
    parameter_groups, candidates, selection_stats = configure_come_sparse_variant(
        config, net_f, net_b, net_c
    )
    optimizer = (
        None
        if config["variant"] in LBI_VARIANTS
        else _build_optimizer(config, parameter_groups)
    )
    static_masks = {}
    if config["variant"] in RANDOM_VARIANTS | MAGNITUDE_VARIANTS:
        static_masks = build_static_masks(
            config["variant"],
            candidates,
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
                candidates,
                selection_stats["candidate_track"],
                selection_stats["total_model_param_count"],
            )
        )
        selected = selection_stats.get(
            "selected_group_count", selection_stats["selected_scalar_count"]
        )
        if selected != selection_stats["integer_budget"]:
            raise RuntimeError("COME static selector violated exact integer budget")
        selection_stats["support_utilization"] = 1.0
    lbi_executor = (
        COMELBIExecutor(selection_stats["total_model_param_count"])
        if config["variant"] in LBI_VARIANTS
        else None
    )
    initial = _snapshot_state(net_f, net_b, net_c)
    bn_names = _bn_state_names(net_f, net_b, net_c)
    selection_stats.update(
        {
            "source_checkpoint_sha256": source_hashes,
            "come_execution_unit": "incoming_outer_batch",
            "random_mask_index": random_mask_index,
            "random_child_mask_seed": random_child_mask_seed,
            "random_child_mask_count": (
                config.get("num_random_masks")
                if config["variant"] in RANDOM_VARIANTS
                else None
            ),
            "debug_only": not formal_run,
            "formal": formal_run,
            "formal_protocol": formal_run,
            "debug_smoke": debug_limit is not None,
            "debug_max_outer_batches": debug_limit,
            "singleton_outer_batch_policy": (
                "skip_size_1_before_selector_objective_lbi_optimizer_and_pu"
            ),
            "come_namespace": dict(config["come"]),
            "lbi_namespace": dict(config["lbi"]) if config.get("lbi") else None,
            "target_stream_identity": config["target_stream_identity"],
            "objective_recomputation": (
                "every_stage1_candidate_and_stage2_current_state"
                if lbi_executor
                else "current_state_once_per_valid_outer_batch"
            ),
        }
    )
    if output_dir is None:
        run_id, output_dir = create_run_dir(
            _output_root(config),
            task_name=config["task_name"],
            run_name=config["output"].get("run_name"),
        )
    else:
        os.makedirs(output_dir, exist_ok=False)
    artifacts = write_initial_artifacts(
        output_dir,
        run_id,
        config,
        data_order,
        source_paths,
        selection_stats,
        workspace_root,
        summary_filename=summary_filename,
    )
    open(artifacts["metrics"], "a", encoding="utf-8").close()
    runtime = RuntimeInstrumentation(
        device,
        runtime_comparable=(
            config.get("runtime", {}).get("runtime_comparable", False)
            and config.get("runtime", {}).get("workers_per_gpu") == 1
        ),
    )
    predictions, labels_all, step_records = [], [], []
    processed = skipped = 0
    max_iterations = len(loaders["target"])
    progress = tqdm(loaders["target"], desc="COME sparse", dynamic_ncols=True)
    for outer_index, (inputs, labels, sample_indices) in enumerate(progress):
        if _should_skip_singleton_outer_batch(inputs.size(0)):
            skipped += 1
            continue
        inputs = inputs.to(device)
        runtime.begin_batch(outer_index, inputs.size(0))
        adapt_started = runtime.start_adaptation()
        try:
            diagnostics = adapt_sparse_one_batch(
                config,
                inputs,
                net_f,
                net_b,
                net_c,
                optimizer,
                candidates,
                static_masks,
                lbi_executor,
                processed + 1,
                max_iterations,
                timing=runtime,
            )
        except COMEObjectiveError as error:
            append_jsonl(
                artifacts["metrics"],
                {"event": "objective_failure", "diagnostics": error.diagnostics},
            )
            raise
        runtime.finish_adaptation(adapt_started)

        pu_started = runtime.start_pu()
        pu_guard = _read_only_guard(net_f, net_b, net_c)
        outputs = read_only_post_update_forward(
            inputs, net_f, net_b, net_c, config["variant"]
        )
        _assert_read_only(pu_guard, net_f, net_b, net_c)
        prediction = outputs.argmax(1).cpu()
        runtime.finish_pu(pu_started)
        efficiency = runtime.finish_batch()
        predictions.append(prediction)
        labels_cpu = labels.cpu()
        labels_all.append(labels_cpu)
        processed += 1
        record = {
            "schema_version": SCHEMA_VERSION,
            "event": "online_step",
            "run_id": run_id,
            "method": "COME",
            "variant": config["variant"],
            "implementation_revision": config["implementation_revision"],
            "protocol_revision": config["protocol_revision"],
            "iteration": processed,
            "outer_batch_index": outer_index,
            "processed_outer_batches": processed,
            "singleton_outer_batches_skipped": skipped,
            "sample_indices": sample_indices.tolist(),
            "acc_post": (
                100.0 * int((prediction == labels_cpu).sum()) / int(labels_cpu.numel())
            ),
            "pu_read_only_verified": True,
            **diagnostics,
            **selection_stats,
            **{
                key: efficiency[key]
                for key in BATCH_EFFICIENCY_FIELDS
                if key in efficiency
            },
        }
        append_jsonl(artifacts["metrics"], record)
        step_records.append(record)
        progress.set_description(
            f"COME sparse | Post={record['acc_post']:.2f}% "
            f"Support={record.get('selected_group_count', record.get('selected_scalar_count'))}"
        )
        if debug_limit is not None and processed >= debug_limit:
            break

    if not predictions:
        raise RuntimeError("COME sparse processed no valid outer batches")
    if debug_limit is not None and processed != debug_limit:
        raise RuntimeError(
            f"COME reviewed smoke processed {processed}, expected {debug_limit}"
        )
    objective_calls = sum(item["objective_call_count"] for item in step_records)
    optimizer_steps = sum(item["optimizer_step_count"] for item in step_records)
    scheduler_steps = sum(item["scheduler_step_count"] for item in step_records)
    stage2_total = sum(item.get("stage2_steps_completed", 0) for item in step_records)
    if config["variant"] in LBI_VARIANTS:
        expected_objectives = sum(
            item["stage1_steps_completed"] + 1 for item in step_records
        )
        if (
            objective_calls != expected_objectives
            or optimizer_steps != 0
            or scheduler_steps != 0
            or stage2_total != processed
            or lbi_executor.omega_writeback_count != processed
            or lbi_executor.local_restart_count != processed
        ):
            raise RuntimeError("COME-LBI aggregate state-transition counts are invalid")
    elif not (
        objective_calls == optimizer_steps == scheduler_steps == processed
        and stage2_total == 0
    ):
        raise RuntimeError("COME sparse host state-transition counts are invalid")

    pu = _compute_dataset_metrics(
        torch.cat(labels_all).numpy(),
        torch.cat(predictions).numpy(),
        config["data"]["dataset"],
        "PU",
    )
    net_f.eval()
    net_b.eval()
    net_c.eval()
    fo_guard = _read_only_guard(net_f, net_b, net_c)
    fo, fo_diagnostics = runtime.measure_fo(
        lambda: _evaluate_with_diagnostics(
            loaders["test"],
            net_f,
            net_b,
            net_c,
            device,
            config["data"]["dataset"],
            config["model"]["class_num"],
        )
    )
    _assert_read_only(fo_guard, net_f, net_b, net_c)
    support_fields = (
        "selected_param_count",
        "selected_group_count",
        "selected_scalar_count",
        "realized_group_ratio",
        "realized_scalar_ratio",
        "support_utilization",
        "stage1_steps_completed",
        "stage1_stop_reason",
        "stage1_support_count",
        "stage1_max_support_count",
        "stage1_cap_hit",
        "overshoot",
        "rollback",
        "stage1_rollback_used",
        "stage2_steps_completed",
        "objective_call_count",
        "off_mask_exact_preservation",
    )
    support_trajectory = [
        {key: record.get(key) for key in support_fields} for record in step_records
    ]
    final_state = _snapshot_state(net_f, net_b, net_c)
    state_audit = _state_transition_audit(
        initial, final_state, selection_stats["candidate_layer_names"], bn_names
    )
    if (
        not state_audit["scope_preserved"]
        or not state_audit["netC_byte_identical"]
        or not state_audit["bn_state_byte_identical"]
    ):
        raise RuntimeError(f"COME sparse scope preservation failure: {state_audit}")
    metadata = runtime.metadata()
    wall = time.perf_counter() - started
    final = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "run_id": run_id,
        "output_dir": output_dir,
        "method": "COME",
        "variant": config["variant"],
        "task": config["task"],
        "dataset": config["data"]["dataset"],
        "source": config["data"]["source"],
        "target": config["data"]["target"],
        "source-target": config["task_name"],
        "seed": config["seed"],
        "implementation_revision": config["implementation_revision"],
        "protocol_revision": config["protocol_revision"],
        "source_checkpoint_revision": config["source_checkpoint_revision"],
        "source_checkpoints": source_paths,
        "source_checkpoint_sha256": source_hashes,
        "experiment_key": config["experiment_key"],
        "experiment_config_sha256": config["experiment_config_sha256"],
        "started_at_utc": started_at_utc,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "processed_outer_batches": processed,
        "singleton_outer_batches_skipped": skipped,
        "target_stream_identity": config["target_stream_identity"],
        "processed_sample_stream_sha256": _stream_trace_sha256(step_records),
        "objective_call_count": objective_calls,
        "support_trajectory": support_trajectory,
        "optimizer_step_count": optimizer_steps,
        "scheduler_step_count": scheduler_steps,
        "saliency_support_selection_count": sum(
            item.get("saliency_support_selections", 0) for item in step_records
        ),
        "stage2_steps_completed_total": stage2_total,
        "stage2_objective_recomputation_count": sum(
            int(item.get("stage2_objective_recomputed", False)) for item in step_records
        ),
        "host_optimizer_persistent_step_count": sum(
            item["host_optimizer_persistent_step_count"] for item in step_records
        ),
        "lbi_local_restart_count": (
            lbi_executor.local_restart_count if lbi_executor else 0
        ),
        "lbi_support_discovery_count": (
            lbi_executor.support_discovery_count if lbi_executor else 0
        ),
        "lbi_omega_writeback_count": (
            lbi_executor.omega_writeback_count if lbi_executor else 0
        ),
        "stage2_optimizer_instance_count": (
            lbi_executor.stage2_optimizer_instance_count if lbi_executor else 0
        ),
        "random_mask_index": random_mask_index,
        "random_child_mask_seed": random_child_mask_seed,
        "pu_read_only_verified": True,
        "fo_read_only_verified": True,
        "debug_only": not formal_run,
        "formal": formal_run,
        "formal_protocol": formal_run,
        "debug_smoke": debug_limit is not None,
        "debug_max_outer_batches": debug_limit,
        "wall_runtime_sec": wall,
        "runtime": wall,
        **pu,
        **fo,
        **fo_diagnostics,
        "primary_metric_name": config["scientific_config"]["primary_metric_name"],
        "PU-primary_metric_value": pu["PU-Acc"],
        "FO-primary_metric_value": fo["FO-Acc"],
        "visda_fixed_class_count": (
            12 if config["data"]["dataset"] == "VISDA-C" else None
        ),
        **metadata,
        **selection_stats,
        **support_trajectory[-1],
        **state_audit,
    }
    append_jsonl(artifacts["metrics"], {"event": "final", **final})
    dump_json(artifacts["summary"], final)
    _update_manifest_runtime(
        artifacts,
        {
            **metadata,
            "processed_outer_batches": processed,
            "objective_call_count": objective_calls,
            "optimizer_step_count": optimizer_steps,
            "scheduler_step_count": scheduler_steps,
            "pu_read_only_verified": True,
            "fo_read_only_verified": True,
            "debug_only": not formal_run,
            "formal": formal_run,
            "formal_protocol": formal_run,
            "debug_smoke": debug_limit is not None,
            "debug_max_outer_batches": debug_limit,
        },
        wall,
    )
    return final


def _run_random_experiment(config, workspace_root):
    started = time.perf_counter()
    run_id, output_dir = create_run_dir(
        _output_root(config),
        task_name=config["task_name"],
        run_name=config["output"].get("run_name"),
    )
    seeds = [202600, 202601, 202602]
    children = []
    for index, seed in enumerate(seeds):
        mask_id = f"mask_{index:02d}"
        child = _run(
            config,
            workspace_root,
            run_id=f"{run_id}/{mask_id}",
            output_dir=osp.join(output_dir, mask_id),
            summary_filename="results.json",
            random_mask_index=index,
            random_child_mask_seed=seed,
        )
        children.append(child)
    stream_hashes = {child["processed_sample_stream_sha256"] for child in children}
    stream_identities = {
        child["target_stream_identity"]["target_order_sha256"] for child in children
    }
    if len(stream_hashes) != 1 or len(stream_identities) != 1:
        raise RuntimeError("COME Random children did not see the same target stream")
    records = [
        {
            "mask_id": f"mask_{index:02d}",
            "mask_index": index,
            "mask_seed": seeds[index],
            "PU-Acc": child["PU-Acc"],
            "FO-Acc": child["FO-Acc"],
            "run_id": child["run_id"],
            "output_dir": child["output_dir"],
            "processed_outer_batches": child["processed_outer_batches"],
            "processed_sample_stream_sha256": child["processed_sample_stream_sha256"],
            "objective_call_count": child["objective_call_count"],
            "optimizer_step_count": child["optimizer_step_count"],
            "host_optimizer_persistent_step_count": child[
                "host_optimizer_persistent_step_count"
            ],
            "scope_preserved": child["scope_preserved"],
            "bn_state_byte_identical": child["bn_state_byte_identical"],
            "netC_byte_identical": child["netC_byte_identical"],
            "source_checkpoint_sha256": child["source_checkpoint_sha256"],
            "support_trajectory": child["support_trajectory"],
            "selected_param_count": child.get("selected_param_count"),
            "selected_group_count": child.get("selected_group_count"),
            "selected_scalar_count": child.get("selected_scalar_count"),
            "realized_group_ratio": child.get("realized_group_ratio"),
            "realized_scalar_ratio": child.get("realized_scalar_ratio"),
            "debug_only": child.get("debug_only"),
            "formal": child.get("formal"),
            "formal_protocol": child.get("formal_protocol"),
            "online_compute_runtime_sec": child.get("online_compute_runtime_sec"),
            "fo_eval_runtime_sec": child.get("fo_eval_runtime_sec"),
            "online_batch_runtime_mean_sec": child.get("online_batch_runtime_mean_sec"),
            "runtime_comparable": child.get("runtime_comparable", False),
        }
        for index, child in enumerate(children)
    ]
    pu_values = [float(child["PU-Acc"]) for child in children]
    fo_values = [float(child["FO-Acc"]) for child in children]
    reference = children[-1]
    children_formal = len(children) == 3 and all(
        child.get("formal") is True and child.get("formal_protocol") is True
        for child in children
    )
    summary = {
        **{
            key: reference.get(key)
            for key in (
                "method",
                "variant",
                "task",
                "dataset",
                "source",
                "target",
                "source-target",
                "implementation_revision",
                "protocol_revision",
                "source_checkpoint_revision",
                "experiment_key",
                "experiment_config_sha256",
                "requested_budget",
                "integer_budget",
                "candidate_track",
                "candidate_scope",
                "candidate_tensor_count",
                "candidate_scope_param_count",
                "candidate_layer_names",
                "group_mode",
                "total_group_count",
                "bn_stats_policy",
                "bn_stats_frozen",
                "persistent_writeback",
                "come_namespace",
                "lbi_namespace",
                "parent_come_baseline_protocol_revision",
                "come_lbi_protocol_revision",
                "official_come_commit",
                "selector_definition",
                "lbi_omega",
                "primary_metric_name",
                "visda_fixed_class_count",
                "target_stream_identity",
            )
        },
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "run_id": run_id,
        "output_dir": output_dir,
        "seed": config["seed"],
        "run_seed": config["seed"],
        "selection_seed": 202600,
        "num_random_masks": 3,
        "mask_seeds": seeds,
        "random_child_execution_count": len(children),
        "fresh_source_model_execution_count": len(children),
        "fresh_host_optimizer_execution_count": len(children),
        "same_target_stream_verified": True,
        "processed_sample_stream_sha256": next(iter(stream_hashes)),
        "child_paths": [child["output_dir"] for child in children],
        "child_supports": [child["support_trajectory"] for child in children],
        "processed_outer_batches_per_child": [
            child["processed_outer_batches"] for child in children
        ],
        "processed_outer_batches_total": sum(
            child["processed_outer_batches"] for child in children
        ),
        "objective_call_count": sum(
            child["objective_call_count"] for child in children
        ),
        "optimizer_step_count": sum(
            child["optimizer_step_count"] for child in children
        ),
        "scheduler_step_count": sum(
            child["scheduler_step_count"] for child in children
        ),
        "host_optimizer_persistent_step_count": sum(
            child["host_optimizer_persistent_step_count"] for child in children
        ),
        "lbi_support_discovery_count": 0,
        "lbi_omega_writeback_count": 0,
        "scope_preserved": all(child["scope_preserved"] for child in children),
        "bn_state_byte_identical": all(
            child["bn_state_byte_identical"] for child in children
        ),
        "netC_byte_identical": all(child["netC_byte_identical"] for child in children),
        "debug_only": not children_formal,
        "formal": children_formal,
        "formal_protocol": children_formal,
        "debug_smoke": config.get("runtime", {}).get("debug_max_outer_batches")
        is not None,
        "debug_max_outer_batches": config.get("runtime", {}).get(
            "debug_max_outer_batches"
        ),
        "masks": records,
        "mean_PU-Acc": float(np.mean(pu_values)),
        "std_PU-Acc": float(np.std(pu_values)),
        "mean_FO-Acc": float(np.mean(fo_values)),
        "std_FO-Acc": float(np.std(fo_values)),
        "PU-Acc": float(np.mean(pu_values)),
        "FO-Acc": float(np.mean(fo_values)),
        "PU-primary_metric_value": float(np.mean(pu_values)),
        "FO-primary_metric_value": float(np.mean(fo_values)),
        "runtime": time.perf_counter() - started,
        **aggregate_random_efficiency(records),
    }
    dump_json(osp.join(output_dir, "summary.json"), summary)
    return summary


def run_sparse_experiment(config, workspace_root):
    if bool(config["formal_protocol"]) and config["variant"] in LBI_VARIANTS:
        raise ValueError(
            "formal COME-LBI launch remains blocked until search protocol "
            "and tuned tuples are frozen"
        )
    if config["variant"] in RANDOM_VARIANTS:
        return _run_random_experiment(config, workspace_root)
    return _run(config, workspace_root)
