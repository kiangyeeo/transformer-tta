"""Formal controlled dense COME baseline trainer."""

import hashlib
import os.path as osp
import time
from datetime import datetime, timezone

import torch
import torch.nn as nn
from tqdm import tqdm

from protocol_constants import (
    CONV_CANDIDATE_PARAM_COUNT,
    FC_CANDIDATE_PARAM_COUNT,
)
from shot_otta.artifacts import (
    SCHEMA_VERSION,
    append_jsonl,
    create_run_dir,
    dump_json,
    write_initial_artifacts,
)
from shot_otta.candidates import CONV_CANDIDATE_ORDER, MODULE_CANDIDATE_ORDER
from shot_otta.data import build_loaders
from shot_otta.efficiency import BATCH_EFFICIENCY_FIELDS
from shot_otta.models import load_source_models
from shot_otta.trainer import (
    RuntimeInstrumentation,
    _build_optimizer,
    _compute_dataset_metrics,
    _post_update_forward,
    _schedule_learning_rate,
    _update_manifest_runtime,
    setup_reproducibility,
)

from .config import COME_VARIANTS
from .objective import come_loss, subjective_opinion


class COMEObjectiveError(RuntimeError):
    """Fail loudly when the fixed official objective is non-finite."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


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
        (f"{prefix}.{name}", parameter)
        for prefix, model in (("netF", net_f), ("netB", net_b), ("netC", net_c))
        for name, parameter in model.named_parameters()
    ]


def _snapshot_state(net_f, net_b, net_c):
    return {
        f"{prefix}.{name}": value.detach().cpu().clone()
        for prefix, model in (("netF", net_f), ("netB", net_b), ("netC", net_c))
        for name, value in model.state_dict().items()
    }


def _bn_state_names(net_f, net_b, net_c):
    names = set()
    for prefix, model in (("netF", net_f), ("netB", net_b), ("netC", net_c)):
        for module_name, module in model.named_modules():
            if not isinstance(module, nn.modules.batchnorm._BatchNorm):
                continue
            stem = f"{prefix}.{module_name}" if module_name else prefix
            local_state_names = [
                name for name, _ in module.named_parameters(recurse=False)
            ] + [name for name, _ in module.named_buffers(recurse=False)]
            for state_name in local_state_names:
                names.add(f"{stem}.{state_name}")
    return names


def _state_delta(before, after):
    changed = sorted(
        name for name in before if not torch.equal(before[name], after[name])
    )
    return changed


def _read_only_guard(*models):
    """Cheap state guard: parameter versions plus exact persistent buffers."""

    parameters = [
        (f"{prefix}.{name}", parameter, parameter._version)
        for prefix, model in zip(("netF", "netB", "netC"), models)
        for name, parameter in model.named_parameters()
    ]
    buffers = {
        f"{prefix}.{name}": value.detach().clone()
        for prefix, model in zip(("netF", "netB", "netC"), models)
        for name, value in model.named_buffers()
    }
    return parameters, buffers


def _assert_read_only(before, *models):
    parameter_versions, buffers = before
    current_parameters = {
        f"{prefix}.{name}": parameter
        for prefix, model in zip(("netF", "netB", "netC"), models)
        for name, parameter in model.named_parameters()
    }
    changed_parameters = [
        name
        for name, parameter, version in parameter_versions
        if current_parameters[name]._version != version
    ]
    current_buffers = {
        f"{prefix}.{name}": value
        for prefix, model in zip(("netF", "netB", "netC"), models)
        for name, value in model.named_buffers()
    }
    changed_buffers = [
        name
        for name, value in buffers.items()
        if not torch.equal(value, current_buffers[name])
    ]
    if changed_parameters or changed_buffers:
        raise RuntimeError(
            "read-only COME branch mutated model state: "
            f"parameters={changed_parameters}, buffers={changed_buffers}"
        )


def set_adaptation_train_behavior(variant, net_f, net_b, net_c):
    """Match the corresponding SHOT full/FC/Conv BN behavior exactly."""

    if variant == "come_full_dense":
        net_f.train()
        net_b.train()
    elif variant == "come_fc_module_dense":
        net_f.train()
        net_b.train()
    elif variant == "come_conv_module_dense":
        net_f.train()
        net_b.eval()
    else:
        raise ValueError(f"Unsupported COME variant: {variant}")
    net_c.eval()
    if variant != "come_full_dense":
        for model in (net_f, net_b, net_c):
            for module in model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()


def configure_come_variant(config, net_f, net_b, net_c):
    """Freeze the model, enable one allowed scope, and build SHOT LR groups."""

    variant = config["variant"]
    if variant not in COME_VARIANTS:
        raise ValueError(f"Unsupported COME variant: {variant}")
    named_parameters = _all_named_parameters(net_f, net_b, net_c)
    lookup = dict(named_parameters)
    for _, parameter in named_parameters:
        parameter.requires_grad = False

    if variant == "come_full_dense":
        selected_names = {
            name
            for name in lookup
            if name.startswith("netF.") or name.startswith("netB.")
        }
        candidate_scope = "netF+netB"
        candidate_names = sorted(selected_names)
        bn_stats_policy = "adaptive"
    elif variant == "come_fc_module_dense":
        selected_names = set(MODULE_CANDIDATE_ORDER)
        candidate_scope = "netB.bottleneck"
        candidate_names = list(MODULE_CANDIDATE_ORDER)
        bn_stats_policy = "frozen"
    else:
        selected_names = set(CONV_CANDIDATE_ORDER)
        candidate_scope = "netF.layer4_conv"
        candidate_names = list(CONV_CANDIDATE_ORDER)
        bn_stats_policy = "frozen"
    missing = sorted(selected_names - set(lookup))
    if missing:
        raise RuntimeError(f"COME candidate parameters not found: {missing}")

    selected_parameters = []
    for name, parameter in named_parameters:
        if name in selected_names:
            parameter.requires_grad = True
            selected_parameters.append((name, parameter))
    candidate_count = sum(parameter.numel() for _, parameter in selected_parameters)
    if (
        variant == "come_fc_module_dense"
        and candidate_count != FC_CANDIDATE_PARAM_COUNT
    ):
        raise RuntimeError(
            "COME FC candidate count must be "
            f"{FC_CANDIDATE_PARAM_COUNT}, got {candidate_count}"
        )
    if variant == "come_conv_module_dense":
        if candidate_count != CONV_CANDIDATE_PARAM_COUNT:
            raise RuntimeError(
                "COME Conv candidate count must be "
                f"{CONV_CANDIDATE_PARAM_COUNT}, got {candidate_count}"
            )
        if len(candidate_names) != 9:
            raise RuntimeError("COME Conv scope must contain exactly 9 weights")

    parameter_groups = []
    optimization = config["optimization"]
    for name, parameter in selected_parameters:
        multiplier = (
            optimization["lr_decay1"]
            if name.startswith("netF.")
            else optimization["lr_decay2"]
        )
        parameter_groups.append(
            {"params": parameter, "lr": optimization["lr"] * multiplier}
        )
    set_adaptation_train_behavior(variant, net_f, net_b, net_c)
    return parameter_groups, {
        "selection": "dense",
        "candidate_scope": candidate_scope,
        "candidate_layer_names": candidate_names,
        "candidate_tensor_count": len(selected_parameters),
        "candidate_scope_param_count": int(candidate_count),
        "selected_param_count": int(candidate_count),
        "total_model_param_count": int(
            sum(parameter.numel() for _, parameter in named_parameters)
        ),
        "requested_budget": 1.0,
        "integer_budget": None,
        "group_mode": None,
        "bn_stats_policy": bn_stats_policy,
        "bn_stats_frozen": variant != "come_full_dense",
        "objective_recomputation": "current_logits_once_per_valid_outer_batch",
    }


def read_only_post_update_forward(inputs, net_f, net_b, net_c, variant):
    """Use the exact corresponding SHOT PU forward without retained writes."""

    return _post_update_forward(
        inputs,
        net_f,
        net_b,
        net_c,
        preserve_bn_state=variant == "come_full_dense",
    )


def _should_skip_singleton_outer_batch(batch_size):
    """Skip before objective/scheduler/optimizer/PU, matching SHOT history."""

    return int(batch_size) == 1


def _prediction_diagnostics(logits, class_count):
    """Detached/read-only collapse diagnostics for a single logits tensor."""

    with torch.no_grad():
        detached = logits.detach()
        probabilities = detached.softmax(dim=1)
        predictions = probabilities.argmax(dim=1)
        histogram = torch.bincount(predictions, minlength=int(class_count)).cpu()
        dominant_count = int(histogram.max().item()) if histogram.numel() else 0
        sample_count = int(predictions.numel())
        softmax_entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum(
            dim=1
        )
        opinion = subjective_opinion(detached, class_count)
    return {
        "predicted_class_histogram": histogram.tolist(),
        "predicted_class_count": int(torch.count_nonzero(histogram).item()),
        "dominant_class_count": dominant_count,
        "dominant_class_ratio": (
            float(dominant_count / sample_count) if sample_count else 0.0
        ),
        "mean_softmax_entropy": float(softmax_entropy.mean().item()),
        "come_opinion_entropy": float(opinion["entropy"].mean().item()),
        "come_mean_uncertainty_mass": float(opinion["uncertainty"].mean().item()),
    }


def adapt_one_batch(
    config,
    inputs,
    net_f,
    net_b,
    net_c,
    optimizer,
    iteration,
    max_iterations,
):
    """Perform exactly one scheduled COME objective and optimizer update."""

    trainable = [
        (name, parameter)
        for name, parameter in _all_named_parameters(net_f, net_b, net_c)
        if parameter.requires_grad
    ]
    parameter_before = {
        name: parameter.detach().clone() for name, parameter in trainable
    }
    set_adaptation_train_behavior(config["variant"], net_f, net_b, net_c)
    _schedule_learning_rate(config, optimizer, iteration, max_iterations)
    optimizer.zero_grad()
    logits = net_c(net_b(net_f(inputs)))
    result = come_loss(
        logits,
        config["model"]["class_num"],
        p=config["come"]["p"],
        tau=config["come"]["tau"],
    )
    diagnostics = {
        **result.diagnostics,
        "objective_call_count": 1,
        "scheduler_step_count": 1,
        "optimizer_step_count": 0,
    }
    if not diagnostics["finite_loss"]:
        raise COMEObjectiveError("COME loss is non-finite", diagnostics)
    result.loss.backward()
    gradients = [
        parameter.grad for _, parameter in trainable if parameter.grad is not None
    ]
    finite_gradients = bool(gradients) and all(
        bool(torch.isfinite(gradient).all().item()) for gradient in gradients
    )
    gradient_norm = (
        float(
            torch.sqrt(
                sum(torch.sum(gradient.detach() ** 2) for gradient in gradients)
            ).item()
        )
        if gradients
        else 0.0
    )
    diagnostics.update(
        {
            "gradient_tensor_count": len(gradients),
            "gradient_norm": gradient_norm,
            "finite_gradients": finite_gradients,
        }
    )
    if not finite_gradients:
        raise COMEObjectiveError("COME gradient is missing or non-finite", diagnostics)
    optimizer.step()
    update_squared = sum(
        torch.sum((parameter.detach() - parameter_before[name]) ** 2)
        for name, parameter in trainable
    )
    parameter_squared = sum(torch.sum(value**2) for value in parameter_before.values())
    update_norm = float(torch.sqrt(update_squared).item())
    parameter_norm = float(torch.sqrt(parameter_squared).item())
    diagnostics.update(
        {
            "optimizer_step_count": 1,
            "parameter_update_norm": update_norm,
            "parameter_norm_before_update": parameter_norm,
            "relative_parameter_update_norm": float(
                update_norm / max(parameter_norm, 1.0e-12)
            ),
            "finite_parameter_update": bool(
                torch.isfinite(update_squared).item()
                and torch.isfinite(parameter_squared).item()
            ),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
    )
    if not diagnostics["finite_parameter_update"]:
        raise COMEObjectiveError("COME parameter update is non-finite", diagnostics)
    return diagnostics


def _evaluate_with_diagnostics(
    loader, net_f, net_b, net_c, device, dataset, class_count
):
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels, _ in loader:
            inputs = inputs.to(device)
            all_logits.append(net_c(net_b(net_f(inputs))).cpu())
            all_labels.append(labels.cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    predictions = logits.argmax(dim=1).numpy()
    metrics = _compute_dataset_metrics(labels, predictions, dataset, "FO")
    diagnostics = {
        f"FO {key}": value
        for key, value in _prediction_diagnostics(logits, class_count).items()
    }
    diagnostics["FO accuracy"] = metrics["FO-Acc"]
    return metrics, diagnostics


def _state_transition_audit(before, after, allowed_names, bn_names):
    changed = _state_delta(before, after)
    unexpected = sorted(set(changed) - set(allowed_names))
    return {
        "observed_changed_state_names": changed,
        "observed_changed_state_count": len(changed),
        "unexpected_changed_state_names": unexpected,
        "scope_preserved": not unexpected,
        "netC_byte_identical": all(
            torch.equal(before[name], after[name])
            for name in before
            if name.startswith("netC.")
        ),
        "bn_state_byte_identical": all(
            torch.equal(before[name], after[name]) for name in bn_names
        ),
    }


def _output_root(config):
    return osp.join(
        config["output"]["root"],
        config["data"]["dataset"],
        config["task_name"],
        "COME",
        config["variant"],
        f"seed_{int(config['seed'])}",
    )


def _run(config, workspace_root):
    started_at_utc = datetime.now(timezone.utc).isoformat()
    wall_started = time.perf_counter()
    setup_reproducibility(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, data_order = build_loaders(config)
    (net_f, net_b, net_c), source_paths = load_source_models(config, device)
    source_hashes = _source_checkpoint_hashes(source_paths)
    for name, record in source_hashes.items():
        expected = config["source_checkpoint_identity"][name]
        if record["path"] != expected["resolved_path"]:
            raise RuntimeError(f"COME source path identity mismatch for {name}")
        if record["sha256"] != expected["sha256"]:
            raise RuntimeError(f"COME source SHA256 identity mismatch for {name}")
    parameter_groups, selection_stats = configure_come_variant(
        config, net_f, net_b, net_c
    )
    optimizer = _build_optimizer(config, parameter_groups)
    initial_state = _snapshot_state(net_f, net_b, net_c)
    bn_names = _bn_state_names(net_f, net_b, net_c)
    allowed_names = (
        {
            name
            for name in initial_state
            if name.startswith("netF.") or name.startswith("netB.")
        }
        if config["variant"] == "come_full_dense"
        else set(selection_stats["candidate_layer_names"])
    )
    debug_limit = config.get("runtime", {}).get("debug_max_outer_batches")
    debug_smoke = debug_limit is not None
    selection_stats = {
        **selection_stats,
        "source_checkpoint_sha256": source_hashes,
        "come_execution_unit": "incoming_outer_batch",
        "singleton_outer_batch_policy": (
            "skip_size_1_before_objective_scheduler_optimizer_and_pu"
        ),
        "formal_protocol": bool(config["formal_protocol"]),
        "debug_smoke": debug_smoke,
        "debug_max_outer_batches": debug_limit,
        "official_come_commit": config["come"]["official_commit"],
        "class_count_k": config["model"]["class_num"],
        "primary_metric_name": config["scientific_config"]["primary_metric_name"],
        "target_stream_identity": config["target_stream_identity"],
    }
    run_id, output_dir = create_run_dir(
        _output_root(config),
        task_name=config["task_name"],
        run_name=config["output"].get("run_name"),
    )
    artifact_paths = write_initial_artifacts(
        output_dir,
        run_id,
        config,
        data_order,
        source_paths,
        selection_stats,
        workspace_root,
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
    trajectory = []
    processed = 0
    skipped = 0
    objective_calls = 0
    optimizer_steps = 0
    scheduler_steps = 0
    max_iterations = len(loaders["target"])
    progress = tqdm(loaders["target"], desc="COME baseline", dynamic_ncols=True)
    for loader_batch_index, (inputs, labels, sample_indices) in enumerate(
        progress, start=1
    ):
        if _should_skip_singleton_outer_batch(inputs.size(0)):
            skipped += 1
            continue
        iteration = processed + 1
        inputs = inputs.to(device)
        runtime_tracker.begin_batch(loader_batch_index - 1, inputs.size(0))
        adaptation_started = runtime_tracker.start_adaptation()
        try:
            diagnostics = adapt_one_batch(
                config,
                inputs,
                net_f,
                net_b,
                net_c,
                optimizer,
                iteration,
                max_iterations,
            )
        except COMEObjectiveError as error:
            append_jsonl(
                artifact_paths["metrics"],
                {
                    "schema_version": SCHEMA_VERSION,
                    "event": "objective_failure",
                    "run_id": run_id,
                    "method": "COME",
                    "variant": config["variant"],
                    "iteration": iteration,
                    "diagnostics": error.diagnostics,
                },
            )
            raise
        runtime_tracker.finish_adaptation(adaptation_started)
        objective_calls += diagnostics["objective_call_count"]
        optimizer_steps += diagnostics["optimizer_step_count"]
        scheduler_steps += diagnostics["scheduler_step_count"]

        pu_started = runtime_tracker.start_pu()
        pu_guard = _read_only_guard(net_f, net_b, net_c)
        post_outputs = read_only_post_update_forward(
            inputs, net_f, net_b, net_c, config["variant"]
        )
        diagnostics.update(
            _prediction_diagnostics(post_outputs, config["model"]["class_num"])
        )
        _assert_read_only(pu_guard, net_f, net_b, net_c)
        diagnostics["pu_read_only_verified"] = True
        post_predictions = post_outputs.argmax(dim=1).cpu()
        runtime_tracker.finish_pu(pu_started)
        efficiency = runtime_tracker.finish_batch()
        labels_cpu = labels.cpu()
        all_post_predictions.append(post_predictions)
        all_online_labels.append(labels_cpu)
        processed += 1
        batch_accuracy = (
            100.0
            * float(torch.count_nonzero(post_predictions == labels_cpu).item())
            / int(labels_cpu.numel())
        )
        diagnostics["PU"] = batch_accuracy
        trajectory_record = {
            "iteration": iteration,
            **{
                key: diagnostics[key]
                for key in (
                    "predicted_class_histogram",
                    "predicted_class_count",
                    "dominant_class_count",
                    "dominant_class_ratio",
                    "mean_softmax_entropy",
                    "come_opinion_entropy",
                    "come_mean_uncertainty_mass",
                    "gradient_norm",
                    "parameter_update_norm",
                    "relative_parameter_update_norm",
                    "PU",
                )
            },
        }
        trajectory.append(trajectory_record)
        append_jsonl(
            artifact_paths["metrics"],
            {
                "schema_version": SCHEMA_VERSION,
                "event": "online_step",
                "run_id": run_id,
                "method": "COME",
                "variant": config["variant"],
                "implementation_revision": config["implementation_revision"],
                "protocol_revision": config["protocol_revision"],
                "source_checkpoint_revision": config["source_checkpoint_revision"],
                "experiment_key": config["experiment_key"],
                "experiment_config_sha256": config["experiment_config_sha256"],
                "iteration": iteration,
                "outer_batch_index": loader_batch_index - 1,
                "processed_outer_batches": processed,
                "singleton_outer_batches_skipped": skipped,
                "formal_protocol": bool(config["formal_protocol"]),
                "debug_smoke": debug_smoke,
                "debug_max_outer_batches": debug_limit,
                "sample_indices": sample_indices.tolist(),
                "acc_post": batch_accuracy,
                **diagnostics,
                **{
                    key: efficiency[key]
                    for key in BATCH_EFFICIENCY_FIELDS
                    if key in efficiency
                },
                **selection_stats,
            },
        )
        progress.set_description(
            f"COME | Post={batch_accuracy:.2f}% "
            f"Loss={diagnostics['loss']:.4f} "
            f"Classes={diagnostics['predicted_class_count']}"
        )
        if debug_limit is not None and processed >= debug_limit:
            break

    if debug_limit is not None and processed != debug_limit:
        raise RuntimeError(
            f"COME reviewed smoke processed {processed}, expected {debug_limit}"
        )
    if not (objective_calls == optimizer_steps == scheduler_steps == processed):
        raise RuntimeError(
            "COME state-transition counts differ from processed batches: "
            f"objective={objective_calls}, optimizer={optimizer_steps}, "
            f"scheduler={scheduler_steps}, processed={processed}"
        )
    online_predictions = torch.cat(all_post_predictions).numpy()
    online_labels = torch.cat(all_online_labels).numpy()
    pu_metrics = _compute_dataset_metrics(
        online_labels, online_predictions, config["data"]["dataset"], "PU"
    )
    net_f.eval()
    net_b.eval()
    net_c.eval()
    fo_guard = _read_only_guard(net_f, net_b, net_c)
    fo_metrics, fo_diagnostics = runtime_tracker.measure_fo(
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
    final_state = _snapshot_state(net_f, net_b, net_c)
    state_audit = _state_transition_audit(
        initial_state, final_state, allowed_names, bn_names
    )
    controlled_scope = config["variant"] != "come_full_dense"
    if (
        not state_audit["scope_preserved"]
        or not state_audit["netC_byte_identical"]
        or (controlled_scope and not state_audit["bn_state_byte_identical"])
    ):
        raise RuntimeError(f"COME scope preservation failure: {state_audit}")
    wall_runtime = float(time.perf_counter() - wall_started)
    runtime_metadata = runtime_tracker.metadata()
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
        "online_steps": processed,
        "processed_outer_batches": processed,
        "singleton_outer_batches_skipped": skipped,
        "objective_call_count": objective_calls,
        "optimizer_step_count": optimizer_steps,
        "scheduler_step_count": scheduler_steps,
        "pu_read_only_verified": all(
            record.get("iteration") is not None for record in trajectory
        ),
        "fo_read_only_verified": True,
        "formal_protocol": bool(config["formal_protocol"]),
        "debug_smoke": debug_smoke,
        "debug_max_outer_batches": debug_limit,
        "primary_metric_name": config["scientific_config"]["primary_metric_name"],
        "primary_PU": pu_metrics["PU-Acc"],
        "primary_FO": fo_metrics["FO-Acc"],
        **pu_metrics,
        **fo_metrics,
        "collapse_diagnostics_trajectory": trajectory,
        **fo_diagnostics,
        "wall_runtime_sec": wall_runtime,
        "runtime": wall_runtime,
        "checkpoints": None,
        **runtime_metadata,
        **selection_stats,
        **state_audit,
    }
    append_jsonl(artifact_paths["metrics"], {"event": "final", **final})
    dump_json(artifact_paths["summary"], final)
    _update_manifest_runtime(
        artifact_paths,
        {
            **runtime_metadata,
            "formal_protocol": bool(config["formal_protocol"]),
            "debug_smoke": debug_smoke,
            "debug_max_outer_batches": debug_limit,
            "processed_outer_batches": processed,
            "singleton_outer_batches_skipped": skipped,
            "objective_call_count": objective_calls,
            "optimizer_step_count": optimizer_steps,
            "scheduler_step_count": scheduler_steps,
            "pu_read_only_verified": True,
            "fo_read_only_verified": True,
        },
        wall_runtime,
    )
    print(f"Run complete: {output_dir}")
    return final


def run_experiment(
    config,
    workspace_root,
    resume_run_dir=None,
    enable_stream_checkpoint=False,
):
    if resume_run_dir is not None or enable_stream_checkpoint:
        raise ValueError("formal COME baseline has no partial resume/checkpoint mode")
    from .sparse import SPARSE_VARIANTS

    if config.get("variant") in SPARSE_VARIANTS:
        from .sparse_trainer import run_sparse_experiment

        return run_sparse_experiment(config, workspace_root)
    return _run(config, workspace_root)
