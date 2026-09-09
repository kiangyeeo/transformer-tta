"""Dense NCTTA baselines on the controlled refined SHOT-OTTA substrate."""

import hashlib
import json
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
    _evaluate,
    _post_update_forward,
    _save_model,
    _schedule_learning_rate,
    _update_manifest_runtime,
    setup_reproducibility,
)

from .config import (
    NCTTA_VARIANTS,
    optimizer_config_for_variant,
)
from .objective import NCTTAObjectiveError, nctta_loss


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
        for model_name, model in (("netF", net_f), ("netB", net_b), ("netC", net_c))
        for name, parameter in model.named_parameters()
    ]


def _module_lookup(net_f, net_b, net_c):
    return {
        f"{model_name}.{name}": module
        for model_name, model in (("netF", net_f), ("netB", net_b), ("netC", net_c))
        for name, module in model.named_modules()
    }


def _trainable_parameter_manifest(net_f, net_b, net_c):
    modules = _module_lookup(net_f, net_b, net_c)
    entries = []
    for name, parameter in _all_named_parameters(net_f, net_b, net_c):
        if not parameter.requires_grad:
            continue
        module_name = name.rsplit(".", 1)[0]
        entries.append(
            {
                "name": name,
                "module_type": type_name(modules[module_name]),
                "shape": list(parameter.shape),
                "scalar_count": int(parameter.numel()),
            }
        )
    return {
        "parameters": entries,
        "tensor_count": len(entries),
        "scalar_count": sum(entry["scalar_count"] for entry in entries),
    }


def type_name(module):
    return module.__class__.__name__


def configure_nctta_variant(config, net_f, net_b, net_c):
    """Apply controlled scopes or the official native normalization selector."""

    variant = config["variant"]
    if variant not in NCTTA_VARIANTS:
        raise ValueError(f"Unsupported NCTTA variant: {variant}")
    named_parameters = _all_named_parameters(net_f, net_b, net_c)
    lookup = dict(named_parameters)
    for _, parameter in named_parameters:
        parameter.requires_grad = False

    if variant == "nctta_native_norm":
        selected_names = set()
        official_types = (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)
        for model_name, model in (("netF", net_f), ("netB", net_b), ("netC", net_c)):
            for module_name, module in model.named_modules():
                if isinstance(module, official_types):
                    for parameter_name, _ in module.named_parameters(recurse=False):
                        if parameter_name in {"weight", "bias"}:
                            prefix = (
                                f"{model_name}.{module_name}"
                                if module_name
                                else model_name
                            )
                            selected_names.add(f"{prefix}.{parameter_name}")
                if isinstance(module, nn.BatchNorm2d):
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None
        candidate_scope = "official_nctta_normalization_selector"
        bn_stats_policy = "native_batch_statistics_no_running_stats"
        candidate_names = sorted(selected_names)
    elif variant == "nctta_full_dense":
        selected_names = {
            name
            for name in lookup
            if name.startswith("netF.") or name.startswith("netB.")
        }
        candidate_scope = "netF+netB"
        bn_stats_policy = "adaptive"
        candidate_names = sorted(selected_names)
    elif variant == "nctta_fc_module_dense":
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
        raise RuntimeError(f"NCTTA candidate parameters not found: {missing}")

    selected_parameters = []
    for name, parameter in named_parameters:
        if name in selected_names:
            parameter.requires_grad = True
            selected_parameters.append((name, parameter))
    candidate_count = sum(parameter.numel() for _, parameter in selected_parameters)
    if variant == "nctta_fc_module_dense" and (
        candidate_count != FC_CANDIDATE_PARAM_COUNT
    ):
        raise RuntimeError(
            "NCTTA FC candidate count must match the controlled scope "
            f"({FC_CANDIDATE_PARAM_COUNT}), got {candidate_count}"
        )
    if variant == "nctta_conv_module_dense" and (
        candidate_count != CONV_CANDIDATE_PARAM_COUNT
    ):
        raise RuntimeError(
            "NCTTA Conv candidate count must match the controlled scope "
            f"({CONV_CANDIDATE_PARAM_COUNT}), got {candidate_count}"
        )
    if variant == "nctta_conv_module_dense" and len(candidate_names) != 9:
        raise RuntimeError("NCTTA Conv dense scope must contain 9 weights")

    if variant == "nctta_native_norm":
        parameter_groups = [
            {
                "params": [parameter for _, parameter in selected_parameters],
                "lr": config["native_optimizer"]["lr"],
            }
        ]
    else:
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
    set_adaptation_train_behavior(variant, net_f, net_b, net_c)
    manifest = _trainable_parameter_manifest(net_f, net_b, net_c)
    total_model_count = sum(parameter.numel() for _, parameter in named_parameters)
    return parameter_groups, {
        "selection": "native" if variant == "nctta_native_norm" else "dense",
        "candidate_scope": candidate_scope,
        "candidate_layer_names": candidate_names,
        "candidate_scope_param_count": int(candidate_count),
        "selected_param_count": int(candidate_count),
        "total_model_param_count": int(total_model_count),
        "requested_budget": 1.0,
        "integer_budget": None,
        "group_mode": None,
        "bn_stats_policy": bn_stats_policy,
        "bn_stats_frozen": variant not in {"nctta_full_dense", "nctta_native_norm"},
        "bn_module_count": len(bn_modules),
        "native_nctta_norm_only_scope": variant == "nctta_native_norm",
        "native_norm_fo_uses_batch_statistics": variant == "nctta_native_norm",
        "trainable_parameter_manifest": manifest,
        "nctta_feature_source": "post_netB_256d",
        "classifier_reference": "effective_frozen_netC.fc.weight.detach",
        "objective_recomputation": "current_state_once_per_outer_batch",
    }


def set_adaptation_train_behavior(variant, net_f, net_b, net_c):
    if variant == "nctta_native_norm":
        net_f.train()
        net_b.eval()
        net_c.eval()
    elif variant == "nctta_full_dense":
        net_f.train()
        net_b.train()
        net_c.eval()
    elif variant == "nctta_fc_module_dense":
        net_f.train()
        net_b.train()
        net_c.eval()
    elif variant == "nctta_conv_module_dense":
        net_f.train()
        net_b.eval()
        net_c.eval()
    else:
        raise ValueError(f"Unsupported NCTTA variant: {variant}")
    if variant not in {"nctta_full_dense", "nctta_native_norm"}:
        for model in (net_f, net_b, net_c):
            for module in model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()


def read_only_post_update_forward(inputs, net_f, net_b, net_c, variant):
    """SHOT-matched PU forward with no parameter or BN-buffer writes."""

    return _post_update_forward(
        inputs,
        net_f,
        net_b,
        net_c,
        preserve_bn_state=variant in {"nctta_full_dense", "nctta_native_norm"},
    )


def _should_skip_singleton_outer_batch(batch_size):
    """Match historical SHOT semantics: skip only outer batches of size one."""

    return int(batch_size) == 1


def _prediction_diagnostics(logits, class_count, prefix=""):
    """Return detached prediction-collapse diagnostics for one forward pass."""

    with torch.no_grad():
        probabilities = logits.detach().softmax(dim=1)
        predictions = probabilities.argmax(dim=1)
        histogram = torch.bincount(predictions, minlength=int(class_count)).cpu()
        dominant_count = int(histogram.max().item()) if histogram.numel() else 0
        sample_count = int(predictions.numel())
        entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum(dim=1)
    stem = f"{prefix} " if prefix else ""
    return {
        f"{stem}predicted_class_histogram": histogram.tolist(),
        f"{stem}predicted_class_count": int(torch.count_nonzero(histogram).item()),
        f"{stem}dominant_class_count": dominant_count,
        f"{stem}dominant_class_ratio": (
            float(dominant_count / sample_count) if sample_count else 0.0
        ),
        f"{stem}mean_prediction_entropy": float(entropy.mean().item()),
    }


def _evaluate_with_prediction_diagnostics(
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
    return metrics, _prediction_diagnostics(logits, class_count, prefix="FO")


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
    """Perform exactly one current-batch objective call and optimizer update."""

    native = config["variant"] == "nctta_native_norm"
    trainable = [
        (name, parameter)
        for name, parameter in _all_named_parameters(net_f, net_b, net_c)
        if parameter.requires_grad
    ]
    parameter_before = {
        name: parameter.detach().clone() for name, parameter in trainable
    }
    before = _snapshot_state(net_f, net_b, net_c) if native else None
    set_adaptation_train_behavior(config["variant"], net_f, net_b, net_c)
    if not native:
        _schedule_learning_rate(config, optimizer, iteration, max_iterations)
    optimizer.zero_grad()
    result = nctta_loss(inputs, net_f, net_b, net_c, config)
    result.loss.backward()
    gradients = [
        parameter.grad for _, parameter in trainable if parameter.grad is not None
    ]
    finite_gradients = all(
        bool(torch.isfinite(gradient).all().item()) for gradient in gradients
    )
    nonzero_gradient_count = sum(
        int(torch.count_nonzero(gradient).item()) for gradient in gradients
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
    diagnostics = {
        **result.diagnostics,
        "loss_ent": result.diagnostics["entropy_mean"],
        "loss_nc": result.diagnostics["nc_loss_mean"],
        "selected_fraction": (
            result.diagnostics["selected_count"] / result.diagnostics["batch_size"]
        ),
        "mean_fca_distance": result.diagnostics[
            "predicted_fca_distance_mean"
        ],
        "total_loss": result.diagnostics["loss"],
        "gradient_tensor_count": len(gradients),
        "gradient_norm": gradient_norm,
        "finite_gradients": finite_gradients,
        "nonzero_gradient_count": nonzero_gradient_count,
        "objective_call_count": 1,
        "optimizer_step_count": 0,
        "scheduler_step_count": 0 if native else 1,
    }
    if not finite_gradients or not gradients or nonzero_gradient_count == 0:
        raise NCTTAObjectiveError(
            "NCTTA declared scope has invalid or zero gradients", diagnostics
        )
    optimizer.step()
    update_squared = sum(
        torch.sum((parameter.detach() - parameter_before[name]) ** 2)
        for name, parameter in trainable
    )
    parameter_squared = sum(
        torch.sum(value ** 2) for value in parameter_before.values()
    )
    parameter_update_norm = float(torch.sqrt(update_squared).item())
    parameter_norm_before_update = float(torch.sqrt(parameter_squared).item())
    diagnostics.update(
        {
            "parameter_update_norm": parameter_update_norm,
            "parameter_norm_before_update": parameter_norm_before_update,
            "relative_parameter_update_norm": float(
                parameter_update_norm / max(parameter_norm_before_update, 1.0e-12)
            ),
            "finite_parameter_update": bool(
                torch.isfinite(update_squared).item()
                and torch.isfinite(parameter_squared).item()
            ),
        }
    )
    diagnostics["optimizer_step_count"] = 1
    diagnostics["lr"] = float(optimizer.param_groups[0]["lr"])
    if native:
        diagnostics.update(
            _state_delta_details(before, _snapshot_state(net_f, net_b, net_c))
        )
    return diagnostics


def _state_delta_details(before, after):
    changed_names = []
    changed_scalars = 0
    for name, before_value in before.items():
        after_value = after[name]
        delta_count = int(torch.count_nonzero(before_value != after_value).item())
        if delta_count:
            changed_names.append(name)
            changed_scalars += delta_count
    return {
        "actual_changed_tensor_names": sorted(changed_names),
        "actual_changed_tensor_count": len(changed_names),
        "actual_changed_scalar_count": changed_scalars,
    }


def _snapshot_state(net_f, net_b, net_c):
    return {
        f"{model_name}.{name}": value.detach().cpu().clone()
        for model_name, model in (("netF", net_f), ("netB", net_b), ("netC", net_c))
        for name, value in model.state_dict().items()
    }


def _state_transition_audit(before, after, allowed_names):
    changed = sorted(
        name for name in before if not torch.equal(before[name], after[name])
    )
    unexpected = sorted(set(changed) - set(allowed_names))
    delta = _state_delta_details(before, after)
    return {
        "observed_changed_state_names": changed,
        **delta,
        "observed_changed_state_count": len(changed),
        "unexpected_changed_state_names": unexpected,
        "scope_preserved": not unexpected,
        "netC_byte_identical": all(
            torch.equal(before[name], after[name])
            for name in before
            if name.startswith("netC.")
        ),
    }


def _allowed_state_names(variant, selection_stats, initial_state):
    if variant == "nctta_full_dense":
        return {
            name
            for name in initial_state
            if name.startswith("netF.") or name.startswith("netB.")
        }
    return set(selection_stats["candidate_layer_names"])


def _build_variant_optimizer(config, parameter_groups):
    if config["variant"] != "nctta_native_norm":
        return _build_optimizer(config, parameter_groups)
    native = optimizer_config_for_variant(config)
    return torch.optim.SGD(
        parameter_groups,
        momentum=native["momentum"],
        dampening=native["dampening"],
        weight_decay=native["weight_decay"],
        nesterov=native["nesterov"],
    )


def _output_root(config):
    return osp.join(
        config["output"]["root"],
        config["data"]["dataset"],
        config["task_name"],
        "NCTTA",
        config["variant"],
        f"seed_{int(config['seed'])}",
    )


def _run(config, workspace_root):
    if config.get("formal_protocol"):
        raise ValueError("NCTTA P1 formal execution is not frozen")
    if config["output"].get("save_model") and config.get("formal_protocol"):
        raise ValueError("formal protocol requires save_model=false")
    started_at_utc = datetime.now(timezone.utc).isoformat()
    wall_started = time.perf_counter()
    setup_reproducibility(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, data_order = build_loaders(config)
    (net_f, net_b, net_c), source_paths = load_source_models(config, device)
    source_hashes = _source_checkpoint_hashes(source_paths)
    parameter_groups, selection_stats = configure_nctta_variant(
        config, net_f, net_b, net_c
    )
    optimizer = _build_variant_optimizer(config, parameter_groups)
    initial_state = _snapshot_state(net_f, net_b, net_c)
    allowed_state_names = _allowed_state_names(
        config["variant"], selection_stats, initial_state
    )
    debug_limit = config.get("runtime", {}).get("debug_max_outer_batches")
    debug_smoke = debug_limit is not None
    selection_stats = {
        **selection_stats,
        "source_checkpoint_sha256": source_hashes,
        "nctta_execution_unit": "incoming_outer_batch",
        "singleton_outer_batch_policy": "skip_size_1_to_match_shot_history",
        "debug_smoke": debug_smoke,
        "debug_max_outer_batches": debug_limit,
        "official_repository_defaults": dict(config["nctta"]),
        "official_nc_type": "infonce",
        "official_metric": "cos",
        "official_tau_align": 1.0,
        "official_margin": 0.2,
        "official_nctta_upstream_commit": "b4d442472a36af6b3f4d6e75f5138ca97d7d8eec",
        "official_nctta_optimizer": dict(config["native_optimizer"]),
        "implemented_variant_optimizer": dict(optimizer_config_for_variant(config)),
        "optimizer_provenance": (
            "official_nctta_ttab_defaults"
            if config["variant"] == "nctta_native_norm"
            else "common_shot_optimizer_substrate"
        ),
    }
    if config["variant"] == "nctta_native_norm":
        print(
            "NCTTA native trainable parameter manifest:\n"
            + json.dumps(selection_stats["trainable_parameter_manifest"], indent=2)
        )
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
    collapse_diagnostics_trajectory = []
    processed_outer_batches = 0
    singleton_outer_batches_skipped = 0
    objective_call_count = 0
    optimizer_step_count = 0
    scheduler_step_count = 0
    max_iterations = len(loaders["target"])
    progress = tqdm(loaders["target"], desc="NCTTA-OTTA P1", dynamic_ncols=True)
    for loader_batch_index, (inputs, labels, sample_indices) in enumerate(
        progress, start=1
    ):
        if _should_skip_singleton_outer_batch(inputs.size(0)):
            singleton_outer_batches_skipped += 1
            continue
        iteration = processed_outer_batches + 1
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
        except NCTTAObjectiveError as error:
            append_jsonl(
                artifact_paths["metrics"],
                {
                    "schema_version": SCHEMA_VERSION,
                    "event": "objective_failure",
                    "run_id": run_id,
                    "method": "NCTTA",
                    "variant": config["variant"],
                    "iteration": iteration,
                    "sample_indices": sample_indices.tolist(),
                    "diagnostics": error.diagnostics,
                },
            )
            dump_json(
                artifact_paths["summary"],
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed",
                    "run_id": run_id,
                    "method": "NCTTA",
                    "variant": config["variant"],
                    "failure": str(error),
                    "diagnostics": error.diagnostics,
                },
            )
            raise
        runtime_tracker.finish_adaptation(adaptation_started)
        objective_call_count += diagnostics["objective_call_count"]
        optimizer_step_count += diagnostics["optimizer_step_count"]
        scheduler_step_count += diagnostics["scheduler_step_count"]

        pu_started = runtime_tracker.start_pu()
        post_outputs = read_only_post_update_forward(
            inputs, net_f, net_b, net_c, config["variant"]
        )
        post_predictions = post_outputs.argmax(dim=1).cpu()
        diagnostics.update(
            _prediction_diagnostics(
                post_outputs, config["model"]["class_num"]
            )
        )
        runtime_tracker.finish_pu(pu_started)
        efficiency = runtime_tracker.finish_batch()
        labels_cpu = labels.cpu()
        all_post_predictions.append(post_predictions)
        all_online_labels.append(labels_cpu)
        processed_outer_batches += 1
        correct = int(torch.count_nonzero(post_predictions == labels_cpu))
        batch_accuracy = 100.0 * correct / int(labels_cpu.numel())
        diagnostics["PU"] = batch_accuracy
        collapse_diagnostics_trajectory.append(
            {
                "iteration": iteration,
                **{
                    key: diagnostics[key]
                    for key in (
                        "predicted_class_histogram",
                        "predicted_class_count",
                        "dominant_class_count",
                        "dominant_class_ratio",
                        "mean_prediction_entropy",
                        "selected_count",
                        "selected_fraction",
                        "mean_fca_distance",
                        "loss_ent",
                        "loss_nc",
                        "total_loss",
                        "gradient_norm",
                        "parameter_update_norm",
                        "relative_parameter_update_norm",
                        "PU",
                    )
                },
            }
        )
        append_jsonl(
            artifact_paths["metrics"],
            {
                "schema_version": SCHEMA_VERSION,
                "event": "online_step",
                "run_id": run_id,
                "method": "NCTTA",
                "variant": config["variant"],
                "implementation_revision": config["implementation_revision"],
                "protocol_revision": config["protocol_revision"],
                "source_checkpoint_revision": config["source_checkpoint_revision"],
                "source_checkpoints": source_paths,
                "source_checkpoint_sha256": source_hashes,
                "experiment_key": config["experiment_key"],
                "experiment_config_sha256": config["experiment_config_sha256"],
                "iteration": iteration,
                "outer_batch_index": loader_batch_index - 1,
                "processed_outer_batches": processed_outer_batches,
                "singleton_outer_batches_skipped": (singleton_outer_batches_skipped),
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
            f"NCTTA-OTTA | Post={batch_accuracy:.2f}% "
            f"Loss={diagnostics['loss']:.4f} "
            f"Selected={diagnostics['selected_count']}"
        )
        if debug_limit is not None and processed_outer_batches >= debug_limit:
            break

    if not all_post_predictions:
        raise RuntimeError("NCTTA processed no valid non-singleton batches")
    online_predictions = torch.cat(all_post_predictions).numpy()
    online_labels = torch.cat(all_online_labels).numpy()
    pu_metrics = _compute_dataset_metrics(
        online_labels, online_predictions, config["data"]["dataset"], "PU"
    )
    net_f.eval()
    net_b.eval()
    net_c.eval()
    fo_metrics, fo_prediction_diagnostics = runtime_tracker.measure_fo(
        lambda: _evaluate_with_prediction_diagnostics(
            loaders["test"],
            net_f,
            net_b,
            net_c,
            device,
            config["data"]["dataset"],
            config["model"]["class_num"],
        )
    )
    fo_prediction_diagnostics["FO accuracy"] = fo_metrics["FO-Acc"]
    final_state = _snapshot_state(net_f, net_b, net_c)
    state_audit = _state_transition_audit(
        initial_state, final_state, allowed_state_names
    )
    if not state_audit["scope_preserved"] or not state_audit["netC_byte_identical"]:
        raise RuntimeError(f"NCTTA scope preservation failure: {state_audit}")
    checkpoints = None
    if config["output"].get("save_model"):
        checkpoints = _save_model(output_dir, net_f, net_b, net_c)
    wall_runtime = float(time.perf_counter() - wall_started)
    runtime_metadata = runtime_tracker.metadata()
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
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "online_steps": processed_outer_batches,
        "processed_outer_batches": processed_outer_batches,
        "singleton_outer_batches_skipped": singleton_outer_batches_skipped,
        "objective_call_count": objective_call_count,
        "optimizer_step_count": optimizer_step_count,
        "scheduler_step_count": scheduler_step_count,
        "debug_smoke": debug_smoke,
        "debug_max_outer_batches": debug_limit,
        "PU-Acc": pu_metrics["PU-Acc"],
        "PU-Acc-per-class": pu_metrics["PU-Acc-per-class"],
        "FO-Acc": fo_metrics["FO-Acc"],
        "FO-Acc-per-class": fo_metrics["FO-Acc-per-class"],
        "collapse_diagnostics_trajectory": collapse_diagnostics_trajectory,
        **fo_prediction_diagnostics,
        "wall_runtime_sec": wall_runtime,
        "runtime": wall_runtime,
        "checkpoints": checkpoints,
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
            "debug_smoke": debug_smoke,
            "debug_max_outer_batches": debug_limit,
            "processed_outer_batches": processed_outer_batches,
            "singleton_outer_batches_skipped": singleton_outer_batches_skipped,
            "objective_call_count": objective_call_count,
            "optimizer_step_count": optimizer_step_count,
            "scheduler_step_count": scheduler_step_count,
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
        raise ValueError("NCTTA P1 does not define stream resume/checkpointing")
    from .sparse import SPARSE_VARIANTS

    if config["variant"] in SPARSE_VARIANTS:
        from .sparse_trainer import run_sparse_experiment

        return run_sparse_experiment(config, workspace_root)
    return _run(config, workspace_root)
