"""Execution of controlled NCTTA FC/Conv sparse and LBI trajectories."""

import os
import os.path as osp
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from protocol_constants import (
    CONV_CANDIDATE_PARAM_COUNT,
    CONV_GROUP_COUNTS,
    FC_CANDIDATE_PARAM_COUNT,
    FORMAL_INTEGER_BUDGETS,
    NCTTA_PROTOCOL_REVISION,
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
    _evaluate,
    _schedule_learning_rate,
    _update_manifest_runtime,
    setup_reproducibility,
)

from .lbi import NCTTALBIExecutor
from .objective import NCTTAObjectiveError, nctta_loss
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


def configure_nctta_sparse_variant(config, net_f, net_b, net_c):
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
        raise RuntimeError(f"NCTTA sparse candidates not found: {missing}")
    candidates = [(name, lookup[name]) for name in names]
    for _, parameter in candidates:
        parameter.requires_grad = True
    candidate_count = sum(parameter.numel() for _, parameter in candidates)
    expected = (
        FC_CANDIDATE_PARAM_COUNT if track == "fc_scalar" else CONV_CANDIDATE_PARAM_COUNT
    )
    if candidate_count != expected:
        raise RuntimeError(
            f"NCTTA {track} candidate count must be {expected}, got {candidate_count}"
        )
    groups = (
        None
        if track == "fc_scalar"
        else sum(int(parameter.shape[0]) for _, parameter in candidates)
    )
    if groups is not None and groups != CONV_GROUP_COUNTS["out_channel"]:
        raise RuntimeError("NCTTA Conv out-channel pool must contain 9216 groups")
    integer_budget = int(
        float(config["requested_budget"]) * (groups or candidate_count)
    )
    allowed = FORMAL_INTEGER_BUDGETS if track == "fc_scalar" else (4, 9, 18)
    if integer_budget not in allowed:
        raise RuntimeError("NCTTA sparse integer budget is not protocol-valid")
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
    return (
        parameter_groups,
        candidates,
        {
            "selection": variant_selection(variant),
            "sparse_selector": variant_selection(variant),
            "candidate_track": track,
            "candidate_scope": "netB.bottleneck"
            if track == "fc_scalar"
            else "netF.layer4_conv",
            "candidate_layer_names": names,
            "candidate_scope_param_count": candidate_count,
            "total_model_param_count": total_count,
            "requested_budget": float(config["requested_budget"]),
            "integer_budget": integer_budget,
            "group_mode": "out_channel" if track == "conv_out_channel" else None,
            "total_group_count": groups,
            "bn_stats_policy": "frozen",
            "bn_stats_frozen": True,
            "native_nctta_norm_only_scope": False,
            "mask_refresh_policy": (
                "once_before_target_stream"
                if variant in RANDOM_VARIANTS | MAGNITUDE_VARIANTS
                else "once_per_outer_batch_current_state_nctta_objective"
                if variant in SALIENCY_VARIANTS
                else "once_per_outer_batch_via_split_lbi"
            ),
            "persistent_writeback": "lbi_omega_only"
            if variant in LBI_VARIANTS
            else "masked_host_optimizer",
            "host_optimizer_persistent_step": variant not in LBI_VARIANTS,
            "off_mask_exact_preservation": True,
            "nctta_feature_source": "post_netB_256d",
            "classifier_reference": "effective_frozen_netC.fc.weight.detach",
            "parent_nctta_baseline_protocol_revision": NCTTA_PROTOCOL_REVISION,
            "official_nctta_upstream_commit": "b4d442472a36af6b3f4d6e75f5138ca97d7d8eec",
            "selector_definition": {
                "random": "uniform_global_exact_budget_child_mask",
                "magnitude": "source_checkpoint_global_magnitude",
                "saliency": "current_state_parameter_times_nctta_gradient",
                "lbi": "split_lbi_thresholded_gamma_support",
            }[variant_selection(variant)],
            "lbi_omega": (
                float(config["lbi"]["omega"]) if variant in LBI_VARIANTS else None
            ),
        },
    )


def set_sparse_train_behavior(variant, net_f, net_b, net_c):
    if variant.startswith("nctta_fc_"):
        net_f.train()
        net_b.train()
    elif variant.startswith("nctta_conv_"):
        net_f.train()
        net_b.eval()
    else:
        raise ValueError("not a controlled NCTTA sparse variant: " + variant)
    net_c.eval()
    for model in (net_f, net_b, net_c):
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()


def _snapshot(net_f, net_b, net_c):
    return {
        f"{prefix}.{name}": value.detach().cpu().clone()
        for prefix, model in (("netF", net_f), ("netB", net_b), ("netC", net_c))
        for name, value in model.state_dict().items()
    }


def _audit(before, after, allowed):
    changed = sorted(
        name for name in before if not torch.equal(before[name], after[name])
    )
    unexpected = sorted(set(changed) - set(allowed))
    return {
        "observed_changed_state_names": changed,
        "unexpected_changed_state_names": unexpected,
        "scope_preserved": not unexpected,
        "netC_byte_identical": all(
            torch.equal(before[name], after[name])
            for name in before
            if name.startswith("netC.")
        ),
    }


def _objective_closure(config, inputs, net_f, net_b, net_c, trace):
    def closure():
        set_sparse_train_behavior(config["variant"], net_f, net_b, net_c)
        try:
            result = nctta_loss(inputs, net_f, net_b, net_c, config)
        except NCTTAObjectiveError as error:
            error.diagnostics.update(
                {
                    "objective_call_index": len(trace) + 1,
                    "successful_objective_trace": list(trace),
                }
            )
            raise
        trace.append(
            {
                "call_index": len(trace) + 1,
                **{
                    key: result.diagnostics[key]
                    for key in (
                        "selected_count",
                        "entropy_selected_index_checksum",
                        "topk_index_min",
                        "topk_index_max",
                        "topk_index_checksum",
                        "q_dist_l2",
                        "q_prob_l2",
                        "hybrid_q_min",
                        "hybrid_q_max",
                        "hybrid_q_l2",
                        "entropy_mean",
                        "nc_loss_mean",
                        "loss",
                    )
                },
            }
        )
        return result.loss, {
            **result.diagnostics,
            "loss_ent": result.diagnostics["entropy_mean"],
            "loss_nc": result.diagnostics["nc_loss_mean"],
            "total_loss": result.diagnostics["loss"],
        }

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
        diagnostics.update(
            {
                "objective_call_count": len(trace),
                "objective_current_state_trace": trace,
                "optimizer_step_count": 0,
                "scheduler_step_count": 0,
                "stage2_objective_recomputed": True,
            }
        )
        return diagnostics

    _schedule_learning_rate(config, optimizer, iteration, max_iterations)
    optimizer.zero_grad()
    loss, parts = closure()
    loss.backward()
    gradients = [
        parameter.grad for _, parameter in candidates if parameter.grad is not None
    ]
    if len(gradients) != len(candidates) or not all(
        bool(torch.isfinite(g).all().item()) for g in gradients
    ):
        raise NCTTAObjectiveError(
            "NCTTA sparse gradients are missing or non-finite",
            {"objective_trace": trace},
        )
    masks = static_masks
    saliency_count = 0
    if config["variant"] in SALIENCY_VARIANTS:
        masks = build_saliency_masks(
            config["variant"], candidates, config["requested_budget"]
        )
        saliency_count = 1
    masked_optimizer_step(optimizer, candidates, masks)
    stats = mask_statistics(
        masks,
        candidates,
        variant_track(config["variant"]),
        sum(p.numel() for model in (net_f, net_b, net_c) for p in model.parameters()),
    )
    return {
        **parts,
        **stats,
        "loss": float(loss.detach().item()),
        "objective_call_count": 1,
        "objective_current_state_trace": trace,
        "optimizer_step_count": 1,
        "scheduler_step_count": 1,
        "saliency_support_selections": saliency_count,
        "lbi_support_discovery_count": 0,
        "lbi_omega_writeback_count": 0,
        "host_optimizer_persistent_step_count": 1,
        "persistent_writeback": "masked_host_optimizer",
        "support_utilization": 1.0,
        "lr": float(optimizer.param_groups[0]["lr"]),
    }


def _output_root(config):
    return osp.join(
        config["output"]["root"],
        config["data"]["dataset"],
        config["task_name"],
        "NCTTA",
        config["variant"],
        f"seed_{config['seed']}",
    )


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

    debug_limit = config.get("runtime", {}).get("debug_max_outer_batches")
    started = time.perf_counter()
    setup_reproducibility(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, data_order = base.build_loaders(config)
    (net_f, net_b, net_c), source_paths = base.load_source_models(config, device)
    source_hashes = base._source_checkpoint_hashes(source_paths)
    parameter_groups, candidates, selection_stats = configure_nctta_sparse_variant(
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
            seed=random_child_mask_seed
            if random_child_mask_seed is not None
            else config.get("selection_seed"),
        )
        selection_stats.update(
            mask_statistics(
                static_masks,
                candidates,
                selection_stats["candidate_track"],
                selection_stats["total_model_param_count"],
            )
        )
        selection_stats["support_utilization"] = 1.0
    lbi_executor = (
        NCTTALBIExecutor(selection_stats["total_model_param_count"])
        if config["variant"] in LBI_VARIANTS
        else None
    )
    initial = _snapshot(net_f, net_b, net_c)
    selection_stats.update(
        {
            "source_checkpoint_sha256": source_hashes,
            "nctta_execution_unit": "incoming_outer_batch",
            "random_mask_index": random_mask_index,
            "random_child_mask_seed": random_child_mask_seed,
            "random_child_mask_count": config.get("num_random_masks")
            if config["variant"] in RANDOM_VARIANTS
            else None,
            "debug_only": debug_limit is not None,
            "formal": False,
            "debug_smoke": debug_limit is not None,
            "debug_max_outer_batches": debug_limit,
            "singleton_outer_batch_policy": "skip_size_1_before_all_state_transitions",
            "nctta_namespace": copy_dict(config["nctta"]),
            "lbi_namespace": copy_dict(config["lbi"]) if config.get("lbi") else None,
            "parent_nctta_baseline_protocol_revision": NCTTA_PROTOCOL_REVISION,
            "official_nctta_upstream_commit": "b4d442472a36af6b3f4d6e75f5138ca97d7d8eec",
            "objective_recomputation": "every_stage1_candidate_and_stage2_current_state"
            if lbi_executor
            else "current_state_once_per_outer_batch",
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
        runtime_comparable=config.get("runtime", {}).get("runtime_comparable", False)
        and config.get("runtime", {}).get("workers_per_gpu") == 1,
    )
    predictions, labels_all, step_records = [], [], []
    processed = skipped = 0
    max_iterations = len(loaders["target"])
    progress = tqdm(loaders["target"], desc="NCTTA sparse", dynamic_ncols=True)
    for outer_index, (inputs, labels, sample_indices) in enumerate(progress):
        if base._should_skip_singleton_outer_batch(inputs.size(0)):
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
        except NCTTAObjectiveError as error:
            append_jsonl(
                artifacts["metrics"],
                {"event": "objective_failure", "diagnostics": error.diagnostics},
            )
            raise
        runtime.finish_adaptation(adapt_started)
        pu_started = runtime.start_pu()
        outputs = base.read_only_post_update_forward(
            inputs, net_f, net_b, net_c, config["variant"]
        )
        prediction = outputs.argmax(1).cpu()
        runtime.finish_pu(pu_started)
        efficiency = runtime.finish_batch()
        predictions.append(prediction)
        labels_all.append(labels.cpu())
        processed += 1
        record = {
            "schema_version": SCHEMA_VERSION,
            "event": "online_step",
            "run_id": run_id,
            "method": "NCTTA",
            "variant": config["variant"],
            "implementation_revision": config["implementation_revision"],
            "protocol_revision": config["protocol_revision"],
            "iteration": processed,
            "outer_batch_index": outer_index,
            "processed_outer_batches": processed,
            "singleton_outer_batches_skipped": skipped,
            "sample_indices": sample_indices.tolist(),
            "acc_post": 100.0
            * int((prediction == labels.cpu()).sum())
            / labels.numel(),
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
        if debug_limit is not None and processed >= debug_limit:
            break
    if not predictions:
        raise RuntimeError("NCTTA sparse processed no valid outer batches")
    pu = _compute_dataset_metrics(
        torch.cat(labels_all).numpy(),
        torch.cat(predictions).numpy(),
        config["data"]["dataset"],
        "PU",
    )
    net_f.eval()
    net_b.eval()
    net_c.eval()
    fo = runtime.measure_fo(
        lambda: _evaluate(
            loaders["test"], net_f, net_b, net_c, device, config["data"]["dataset"]
        )
    )
    primary_metric_name = (
        "fixed_12_class_mean_per_class_accuracy"
        if config["data"]["dataset"] == "VISDA-C"
        else "sample_level_overall_accuracy"
    )
    support_trace_fields = (
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
        "stage1_rollback_used",
        "stage2_steps_completed",
    )
    support_trajectory = [
        {key: record.get(key) for key in support_trace_fields}
        for record in step_records
    ]
    objective_selected_count_trajectory = [
        [call["selected_count"] for call in record["objective_current_state_trace"]]
        for record in step_records
    ]
    last_step_diagnostics = {
        key: step_records[-1].get(key) for key in support_trace_fields
    }
    state_audit = _audit(
        initial,
        _snapshot(net_f, net_b, net_c),
        selection_stats["candidate_layer_names"],
    )
    if not state_audit["scope_preserved"] or not state_audit["netC_byte_identical"]:
        raise RuntimeError(f"NCTTA sparse scope preservation failure: {state_audit}")
    metadata = runtime.metadata()
    wall = time.perf_counter() - started
    final = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "run_id": run_id,
        "output_dir": output_dir,
        "method": "NCTTA",
        "variant": config["variant"],
        "task": config["task"],
        "dataset": config["data"]["dataset"],
        "source": config["data"]["source"],
        "target": config["data"]["target"],
        "source-target": f"{config['data']['source_name']}-{config['data']['target_name']}",
        "seed": config["seed"],
        "implementation_revision": config["implementation_revision"],
        "protocol_revision": config["protocol_revision"],
        "source_checkpoint_revision": config["source_checkpoint_revision"],
        "source_checkpoints": source_paths,
        "source_checkpoint_sha256": source_hashes,
        "experiment_key": config["experiment_key"],
        "experiment_config_sha256": config["experiment_config_sha256"],
        "processed_outer_batches": processed,
        "singleton_outer_batches_skipped": skipped,
        "objective_call_count": sum(
            item["objective_call_count"] for item in step_records
        ),
        "objective_selected_count_trajectory": objective_selected_count_trajectory,
        "support_trajectory": support_trajectory,
        "optimizer_step_count": sum(
            item["optimizer_step_count"] for item in step_records
        ),
        "scheduler_step_count": sum(
            item["scheduler_step_count"] for item in step_records
        ),
        "saliency_support_selection_count": sum(
            item.get("saliency_support_selections", 0) for item in step_records
        ),
        "stage2_steps_completed_total": sum(
            item.get("stage2_steps_completed", 0) for item in step_records
        ),
        "stage2_objective_recomputation_count": sum(
            int(item.get("stage2_objective_recomputed", False)) for item in step_records
        ),
        "host_optimizer_persistent_step_count": sum(
            item["host_optimizer_persistent_step_count"] for item in step_records
        ),
        "lbi_support_discovery_count": lbi_executor.support_discovery_count
        if lbi_executor
        else 0,
        "lbi_omega_writeback_count": lbi_executor.omega_writeback_count
        if lbi_executor
        else 0,
        "random_mask_index": random_mask_index,
        "random_child_mask_seed": random_child_mask_seed,
        "wall_runtime_sec": wall,
        "runtime": wall,
        **pu,
        **fo,
        "primary_metric_name": primary_metric_name,
        "PU-primary_metric_value": pu["PU-Acc"],
        "FO-primary_metric_value": fo["FO-Acc"],
        "visda_fixed_class_count": 12
        if config["data"]["dataset"] == "VISDA-C"
        else None,
        **metadata,
        **selection_stats,
        **last_step_diagnostics,
        **state_audit,
    }
    append_jsonl(artifacts["metrics"], {"event": "final", **final})
    dump_json(artifacts["summary"], final)
    _update_manifest_runtime(
        artifacts, {**metadata, "processed_outer_batches": processed}, wall
    )
    return final


def copy_dict(value):
    return {key: value[key] for key in value}


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
    records = [
        {
            "mask_id": f"mask_{i:02d}",
            "mask_index": i,
            "mask_seed": seeds[i],
            "PU-Acc": child["PU-Acc"],
            "FO-Acc": child["FO-Acc"],
            "online_compute_runtime_sec": child.get("online_compute_runtime_sec"),
            "fo_eval_runtime_sec": child.get("fo_eval_runtime_sec"),
            "online_batch_runtime_mean_sec": child.get("online_batch_runtime_mean_sec"),
            "runtime_comparable": child.get("runtime_comparable", False),
            "run_id": child.get("run_id"),
            "output_dir": child.get("output_dir"),
            "processed_outer_batches": child.get("processed_outer_batches"),
            "objective_call_count": child.get("objective_call_count"),
            "optimizer_step_count": child.get("optimizer_step_count"),
            "host_optimizer_persistent_step_count": child.get(
                "host_optimizer_persistent_step_count"
            ),
            "scope_preserved": child.get("scope_preserved"),
            "netC_byte_identical": child.get("netC_byte_identical"),
            "source_checkpoint_sha256": child.get("source_checkpoint_sha256"),
            "selected_param_count": child.get("selected_param_count"),
            "selected_group_count": child.get("selected_group_count"),
            "selected_scalar_count": child.get("selected_scalar_count"),
            "realized_group_ratio": child.get("realized_group_ratio"),
            "realized_scalar_ratio": child.get("realized_scalar_ratio"),
        }
        for i, child in enumerate(children)
    ]
    pu_values = [float(x["PU-Acc"]) for x in children]
    fo_values = [float(x["FO-Acc"]) for x in children]
    reference = children[-1]
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
                "candidate_scope_param_count",
                "candidate_layer_names",
                "group_mode",
                "total_group_count",
                "bn_stats_policy",
                "bn_stats_frozen",
                "persistent_writeback",
                "nctta_namespace",
                "lbi_namespace",
                "parent_nctta_baseline_protocol_revision",
                "official_nctta_upstream_commit",
                "selector_definition",
                "lbi_omega",
                "primary_metric_name",
                "visda_fixed_class_count",
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
        "host_optimizer_persistent_step_count": sum(
            child["host_optimizer_persistent_step_count"] for child in children
        ),
        "lbi_support_discovery_count": 0,
        "lbi_omega_writeback_count": 0,
        "scope_preserved": all(child["scope_preserved"] for child in children),
        "netC_byte_identical": all(child["netC_byte_identical"] for child in children),
        "debug_only": all(bool(child.get("debug_only", False)) for child in children),
        "formal": False,
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
        **(
            __import__(
                "visda_otta.evaluator", fromlist=["aggregate_random_mask_metrics"]
            ).aggregate_random_mask_metrics(children)
            if config["data"]["dataset"] == "VISDA-C"
            else {}
        ),
        **aggregate_random_efficiency(records),
    }
    dump_json(osp.join(output_dir, "summary.json"), summary)
    return summary


def run_sparse_experiment(config, workspace_root):
    if config["variant"] in RANDOM_VARIANTS:
        return _run_random_experiment(config, workspace_root)
    return _run(config, workspace_root)
