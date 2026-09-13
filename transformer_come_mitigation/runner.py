"""Isolated runner hooks for non-formal COME mitigation pilots.

The production ``transformer_come`` package remains unchanged.  Each pilot is
executed in its own child process; this module temporarily injects a configurable
COME objective and optimizer factory into the already-audited dense stream.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn

from transformer.source_only.config import canonical_sha256
from transformer.source_only.model import load_frozen_source_model
from transformer_come import runner as dense_runner
from transformer_come.common import ComeRunInvalid
from transformer_come.config import (
    NON_SCIENTIFIC_KEYS,
    load_config,
    resolve_transfer_config,
)
from transformer_come.data import build_target_loaders
from transformer_come.model import load_full_dense_model
from transformer_come.model import load_candidate_dense_model
from transformer_come.objective import come_loss

from .experiments import IMPLEMENTATION_REVISION, PROTOCOL_REVISION


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = PROJECT_ROOT / "transformer_come" / "config.yaml"
PROTOCOL_DOCUMENT = "transformer_come_mitigation/README.md"


def configure_layernorm_affine_scope(model):
    """Train exactly the affine weights/biases owned by LayerNorm modules."""

    model.requires_grad_(False)
    trainable_ids = set()
    for module in model.modules():
        if not isinstance(module, nn.LayerNorm):
            continue
        for parameter in module.parameters(recurse=False):
            parameter.requires_grad_(True)
            trainable_ids.add(id(parameter))

    trainable = []
    frozen = []
    for name, parameter in model.named_parameters():
        destination = trainable if id(parameter) in trainable_ids else frozen
        destination.append((name, parameter))
    if not trainable or not frozen:
        raise RuntimeError("LayerNorm-affine scope must have trainable and frozen parts")
    head_ids = {id(parameter) for parameter in model.get_classifier().parameters()}
    if trainable_ids & head_ids:
        raise RuntimeError("LayerNorm-affine scope unexpectedly includes the head")
    model.eval()
    return trainable, frozen, {
        "update_scope": "layernorm_affine_only",
        "host_objective": "come",
        "trainable_parameter_names": [name for name, _ in trainable],
        "trainable_tensor_count": len(trainable),
        "trainable_scalars": sum(parameter.numel() for _, parameter in trainable),
        "frozen_parameter_names": [name for name, _ in frozen],
        "frozen_parameter_scalars": sum(parameter.numel() for _, parameter in frozen),
        "total_parameter_scalars": sum(
            parameter.numel() for parameter in model.parameters()
        ),
    }


def load_layernorm_affine_model(config: dict, device):
    model, checkpoint_record = load_frozen_source_model(config, device)
    trainable, frozen, scope_record = configure_layernorm_affine_scope(model)
    return model, checkpoint_record, trainable, frozen, scope_record


def configurable_objective_step(*, tau: float):
    def objective_step(model, images, class_count: int):
        logits = model(images)
        if logits.ndim != 2 or int(logits.shape[1]) != int(class_count):
            raise ComeRunInvalid(
                f"COME logits/class mismatch: logits={tuple(logits.shape)}, "
                f"C={class_count}",
                invalid_reason="logits_shape_mismatch",
            )
        try:
            result = come_loss(logits, int(class_count), tau=float(tau))
        except RuntimeError as error:
            raise ComeRunInvalid(
                str(error), invalid_reason="nonfinite_objective"
            ) from error
        result.loss.backward()
        return result.loss, logits, dict(result.diagnostics)

    return objective_step


@contextmanager
def experimental_hooks(experiment: dict):
    """Install process-local objective/optimizer hooks and restore them."""

    original_objective = dense_runner.come_objective_step
    original_adamw = torch.optim.AdamW
    original_sgd = torch.optim.SGD
    dense_runner.come_objective_step = configurable_objective_step(
        tau=float(experiment["tau"])
    )

    def optimizer_factory(parameters, *, lr, betas, eps, weight_decay):
        parameters = list(parameters)
        if experiment["optimizer"] == "sgd":
            optimizer = original_sgd(
                parameters,
                lr=float(lr),
                momentum=float(experiment.get("momentum", 0.9)),
                weight_decay=float(experiment.get("weight_decay", 0.0)),
            )
        else:
            optimizer = original_adamw(
                parameters,
                lr=float(lr),
                betas=tuple(float(value) for value in betas),
                eps=float(eps),
                weight_decay=float(weight_decay),
            )

        clip_norm = experiment.get("gradient_clip_norm")
        if experiment.get("reset_adam_moments_each_batch", False):
            if experiment["optimizer"] != "adamw":
                raise ValueError("Adam moment reset requires the AdamW optimizer")
            original_step = optimizer.step

            def moment_reset_step(closure=None):
                # Keep the persistent parameters, optimizer object and Adam
                # step counter.  Only erase the two moment buffers requested
                # by the mechanism control before the current outer-batch step.
                for parameter in parameters:
                    state = optimizer.state.get(parameter)
                    if not state:
                        continue
                    state["exp_avg"].zero_()
                    state["exp_avg_sq"].zero_()
                optimizer._come_moment_reset_calls += 1
                return original_step(closure=closure)

            optimizer._come_moment_reset_calls = 0
            optimizer.step = moment_reset_step

        if clip_norm is not None:
            unclipped_step = optimizer.step

            def clipped_step(closure=None):
                torch.nn.utils.clip_grad_norm_(parameters, float(clip_norm))
                return unclipped_step(closure=closure)

            optimizer.step = clipped_step
        return optimizer

    # The dense runner constructs ``torch.optim.AdamW`` directly.  Child
    # processes isolate this hook from every formal runner and from one another.
    torch.optim.AdamW = optimizer_factory
    try:
        yield
    finally:
        torch.optim.AdamW = original_adamw
        dense_runner.come_objective_step = original_objective


def resolve_experiment_config(experiment: dict, output_dir: Path, *, device: str):
    """Resolve the audited VisDA substrate, then create a new pilot identity."""

    raw = load_config(BASE_CONFIG)
    substrate_variant = (
        "candidate_dense"
        if experiment["scope"] == "candidate_dense"
        else "full_dense"
    )
    config = resolve_transfer_config(
        raw,
        project_root=PROJECT_ROOT,
        dataset="visda-c",
        source="train",
        target="validation",
        variant=substrate_variant,
        device=device,
        output_dir=output_dir,
    )
    config["protocol_revision"] = PROTOCOL_REVISION
    config["protocol_document"] = PROTOCOL_DOCUMENT
    config["implementation_revision"] = IMPLEMENTATION_REVISION
    config["variant"] = experiment["id"]
    update_scopes = {
        "full_dense": "all_except_head",
        "candidate_dense": "last_three_blocks_qkv_proj_mlp_weights",
        "layernorm_affine": "layernorm_affine_only",
    }
    config["adaptation"] = {
        "update_scope": update_scopes[experiment["scope"]],
        "model_mode": "eval",
        "steps_per_online_batch": 1,
        "study": "come_collapse_mitigation_pilot",
    }
    optimization = copy.deepcopy(config["optimization"])
    optimization.update(
        {
            "optimizer": experiment["optimizer"],
            "lr": float(experiment["lr"]),
            "gradient_clip": (
                "none"
                if experiment.get("gradient_clip_norm") is None
                else "global_l2_norm"
            ),
            "gradient_clip_norm": experiment.get("gradient_clip_norm"),
            "optimizer_state_policy": (
                "zero_exp_avg_and_exp_avg_sq_before_each_outer_batch_step"
                if experiment.get("reset_adam_moments_each_batch", False)
                else "persistent"
            ),
            "adam_step_counter_policy": "persistent",
        }
    )
    if experiment["optimizer"] == "sgd":
        optimization["momentum"] = float(experiment.get("momentum", 0.9))
        optimization["weight_decay"] = float(experiment.get("weight_decay", 0.0))
    config["optimization"] = optimization
    config["come"]["tau"] = float(experiment["tau"])
    config["come"]["tunable_hyperparameters"] = ["tau", "host_lr"]
    config["come"]["formal_protocol_compatible"] = False
    config["come_objective"]["come_tau"] = float(experiment["tau"])
    config["come_objective"]["come_tunable_hyperparameters"] = [
        "tau",
        "host_lr",
    ]
    config["come_objective"]["pilot_note"] = (
        "non-formal collapse mitigation; not an official COME reproduction"
    )
    config["mitigation_experiment"] = copy.deepcopy(experiment)

    scientific = {
        key: value
        for key, value in config.items()
        if key not in NON_SCIENTIFIC_KEYS
    }
    config["scientific_config_sha256"] = canonical_sha256(scientific)
    config["experiment_key"] = (
        f"visda-c_train-validation_come-mitigation-{experiment['id']}_"
        f"seed-2026_{config['scientific_config_sha256'][:12]}"
    )
    return config


def run_experiment(experiment: dict, output_dir: Path, *, device: str = "cuda"):
    config = resolve_experiment_config(experiment, output_dir, device=device)
    model_loaders = {
        "full_dense": load_full_dense_model,
        "candidate_dense": load_candidate_dense_model,
        "layernorm_affine": load_layernorm_affine_model,
    }
    model_loader = model_loaders[experiment["scope"]]
    with experimental_hooks(experiment):
        return dense_runner.run_dense_transfer(
            config,
            PROJECT_ROOT,
            variant=experiment["id"],
            model_loader=model_loader,
            build_target_loaders=build_target_loaders,
            show_progress=False,
        )


__all__ = [
    "BASE_CONFIG",
    "PROJECT_ROOT",
    "configure_layernorm_affine_scope",
    "configurable_objective_step",
    "experimental_hooks",
    "load_layernorm_affine_model",
    "resolve_experiment_config",
    "run_experiment",
]
