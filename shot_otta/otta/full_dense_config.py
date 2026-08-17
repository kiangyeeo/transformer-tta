"""Configuration resolution for DeiT OTTA adaptation baselines.

Supports the ``full_dense`` (all parameters) and ``candidate_dense`` (only the
weight tensors of the last-3 candidate blocks) variants.
"""

import os.path as osp

from experiment_identity import resolve_deit_otta_adaptation_identity
from shot_otta.deit_source_only.config import (
    _resolve_deit_common,
    deit_metrics_policy,
)

ALLOWED_LOSS_COMPONENTS = {"ent", "div", "pseudo"}
VARIANTS = {
    "full_dense": "all_except_head",
    "candidate_dense": "last_3_block_weights",
}
NUM_BLOCKS = 12


def _positive_float(value, field, *, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    converted = float(value)
    minimum_ok = converted >= 0.0 if allow_zero else converted > 0.0
    if not minimum_ok:
        raise ValueError(f"{field} must be positive")
    return converted


def _validate_adaptation(config):
    variant = config["variant"]
    adaptation = config.get("adaptation")
    if not isinstance(adaptation, dict):
        raise ValueError("adaptation must be a mapping")
    expected_scope = VARIANTS[variant]
    if adaptation.get("update_scope") != expected_scope:
        raise ValueError(
            f"adaptation.update_scope must be {expected_scope} for "
            f"variant {variant}"
        )
    if adaptation.get("model_mode") != "eval":
        raise ValueError("adaptation.model_mode must be eval")
    steps = adaptation.get("steps_per_batch")
    if isinstance(steps, bool) or int(steps) != steps or int(steps) < 1:
        raise ValueError("adaptation.steps_per_batch must be a positive int")
    adaptation["steps_per_batch"] = int(steps)
    if variant == "candidate_dense":
        blocks = adaptation.get("candidate_blocks")
        if not isinstance(blocks, list) or not blocks:
            raise ValueError("adaptation.candidate_blocks must be a non-empty list")
        converted_blocks = []
        for block in blocks:
            if isinstance(block, bool) or int(block) != block:
                raise ValueError("adaptation.candidate_blocks must be integers")
            converted_blocks.append(int(block))
        if converted_blocks != sorted(converted_blocks):
            raise ValueError("adaptation.candidate_blocks must be sorted")
        if len(set(converted_blocks)) != len(converted_blocks):
            raise ValueError("adaptation.candidate_blocks must be unique")
        if not all(0 <= block < NUM_BLOCKS for block in converted_blocks):
            raise ValueError(
                f"adaptation.candidate_blocks must be in [0, {NUM_BLOCKS})"
            )
        adaptation["candidate_blocks"] = converted_blocks


def _validate_optimization(config):
    optimization = config.get("optimization")
    if not isinstance(optimization, dict):
        raise ValueError("optimization must be a mapping")
    if optimization.get("optimizer") != "adamw":
        raise ValueError("optimization.optimizer must be adamw")
    optimization["lr"] = _positive_float(optimization.get("lr"), "optimization.lr")
    betas = optimization.get("betas")
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError("optimization.betas must be a list of two values")
    converted_betas = [
        _positive_float(value, "optimization.betas", allow_zero=True)
        for value in betas
    ]
    if not all(0.0 <= value < 1.0 for value in converted_betas):
        raise ValueError("optimization.betas values must be in [0, 1)")
    optimization["betas"] = converted_betas
    optimization["eps"] = _positive_float(
        optimization.get("eps"), "optimization.eps"
    )
    optimization["weight_decay"] = _positive_float(
        optimization.get("weight_decay"),
        "optimization.weight_decay",
        allow_zero=True,
    )


def _validate_loss(config):
    loss = config.get("loss")
    if not isinstance(loss, dict):
        raise ValueError("loss must be a mapping")
    components = loss.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("loss.components must be a non-empty list")
    if not set(components) <= ALLOWED_LOSS_COMPONENTS:
        raise ValueError(
            "loss.components must only contain ent, div, and pseudo"
        )
    if len(set(components)) != len(components):
        raise ValueError("loss.components must not contain duplicates")
    loss["cls_par"] = _positive_float(
        loss.get("cls_par"), "loss.cls_par", allow_zero=True
    )
    loss["ent_par"] = _positive_float(loss.get("ent_par"), "loss.ent_par")
    threshold = _positive_float(
        loss.get("threshold"), "loss.threshold", allow_zero=True
    )
    if not 0.0 <= threshold < 1.0:
        raise ValueError("loss.threshold must be in [0, 1)")
    loss["threshold"] = threshold


def resolve_config(
    config,
    project_root,
    *,
    provided_key=None,
    provided_sha256=None,
):
    if config.get("method") != "shot":
        raise ValueError("method must be shot")
    if config.get("task") != "otta":
        raise ValueError("task must be otta")
    variant = config.get("variant")
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {sorted(VARIANTS)}")
    effective = _resolve_deit_common(config, project_root, "otta")
    if effective["metrics"] != deit_metrics_policy("otta"):
        raise ValueError("OTTA adaptation metric policy is frozen")
    _validate_adaptation(effective)
    _validate_optimization(effective)
    _validate_loss(effective)
    identity = resolve_deit_otta_adaptation_identity(
        effective,
        provided_key=provided_key,
        provided_sha256=provided_sha256,
    )
    effective.update(identity)
    return effective


def experiment_output_root(config):
    data = config["data"]
    return osp.join(
        config["output"]["root"],
        config["task"],
        data["dataset"],
        config["task_name"],
        config["variant"],
        f"seed_{int(config['seed'])}",
    )
