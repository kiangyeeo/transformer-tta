"""Configuration resolution for the DeiT OTTA saliency structural-group baseline.

The candidate space is the weight tensors of the last-3 blocks; groups are the
proposal's paired Q-K / V-O / FFN units; ``budget`` is the structural density
``rho_struct = active_groups / total_groups`` and the number of active groups
per online step is ``ceil(budget * total_groups)``.  Selection is dynamic: after
every TTA-loss backward, each group is scored by the L2 norm of its gradient
(``gradient_l2_norm``) and the Top-K groups are updated.
"""

import os.path as osp

from experiment_identity import resolve_deit_otta_group_saliency_identity
from shot_otta.adaptation.deit_groups import (
    active_group_count,
    total_group_count,
)
from shot_otta.deit_source_only.config import (
    _resolve_deit_common,
    deit_metrics_policy,
)
from shot_otta.otta.full_dense_config import (
    _positive_float,
    _validate_loss,
    _validate_optimization,
)

NUM_BLOCKS = 12
VARIANTS = {"group_saliency": "last_3_block_weights"}
FROZEN_GROUPING = "paired_qk_vo_ffn"
FROZEN_DELTA_SEMANTICS = "per_step_masked_accumulation"
FROZEN_SALIENCY_SCORE = "gradient_l2_norm"


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
    if adaptation.get("grouping") != FROZEN_GROUPING:
        raise ValueError(
            f"adaptation.grouping must be {FROZEN_GROUPING}"
        )
    if adaptation.get("delta_semantics") != FROZEN_DELTA_SEMANTICS:
        raise ValueError(
            f"adaptation.delta_semantics must be {FROZEN_DELTA_SEMANTICS}"
        )
    if adaptation.get("saliency_score") != FROZEN_SALIENCY_SCORE:
        raise ValueError(
            f"adaptation.saliency_score must be {FROZEN_SALIENCY_SCORE}"
        )

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

    budget = adaptation.get("budget")
    budget = _positive_float(budget, "adaptation.budget")
    if budget > 1.0:
        raise ValueError("adaptation.budget must be <= 1.0")
    adaptation["budget"] = budget

    total_groups = total_group_count(converted_blocks)
    adaptation["total_groups"] = total_groups
    adaptation["active_groups"] = active_group_count(budget, total_groups)
    return adaptation


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
    identity = resolve_deit_otta_group_saliency_identity(
        effective,
        provided_key=provided_key,
        provided_sha256=provided_sha256,
    )
    effective.update(identity)
    return effective


def experiment_output_root(config):
    data = config["data"]
    budget = config["adaptation"]["budget"]
    return osp.join(
        config["output"]["root"],
        config["task"],
        data["dataset"],
        config["task_name"],
        config["variant"],
        f"budget_{budget}",
        f"seed_{int(config['seed'])}",
    )
