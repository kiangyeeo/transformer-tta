#!/usr/bin/env python3
"""Contract A/B: COME differs from SHOT in its objective and frozen host LR.

All other substrate fields must equal their SHOT counterparts.  The candidate
universe, group definition, integer budget and strict masked optimizer must be
the *same code objects* as the SHOT ones, not merely numerically similar.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import transformer.candidate_dense.config as shot_candidate_config  # noqa: E402
import transformer.candidate_dense.model as shot_candidate_model  # noqa: E402
import transformer.group_magnitude.groups as shot_magnitude_groups  # noqa: E402
import transformer.group_random.config as shot_random_config  # noqa: E402
import transformer.group_random.groups as shot_random_groups  # noqa: E402
import transformer.group_random.optimizer as shot_optimizer  # noqa: E402
import transformer.group_saliency.groups as shot_saliency_groups  # noqa: E402
import transformer.source_only.data as shot_data  # noqa: E402
import transformer.source_only.model as shot_source_model  # noqa: E402

import transformer_come.config as come_config  # noqa: E402
import transformer_come.data as come_data  # noqa: E402
import transformer_come.groups as come_groups  # noqa: E402
import transformer_come.model as come_model  # noqa: E402
import transformer_come.optimizer as come_optimizer  # noqa: E402
from transformer_come.config import (  # noqa: E402
    BASELINE_IMPLEMENTATION_REVISION,
    EXPECTED_COME_BLOCK,
    IMPLEMENTATION_REVISIONS,
    NON_SCIENTIFIC_KEYS,
    PROTOCOL_DOCUMENT,
    GROUP_LBI_PROTOCOL_DOCUMENT,
    PROTOCOL_REVISIONS,
    SPARSE_LBI_IMPLEMENTATION_REVISION,
    finalize_identity,
    load_config,
    validate_raw_config,
    validate_variant_config,
    variant_config_view,
    resolve_lbi_profile,
)


COME_CONFIG = PROJECT_ROOT / "transformer_come" / "config.yaml"

VARIANTS = (
    "full_dense",
    "candidate_dense",
    "group_random",
    "group_magnitude",
    "group_saliency",
)

# The only value inside a shared block that may differ, because it names the
# objective whose gradient is scored.
ALLOWED_SELECTION_DIFFERENCES = {
    "group_saliency": {
        "ranking_source": (
            "current_pre_update_weight_and_current_shot_gradient",
            "current_pre_update_weight_and_current_come_gradient",
        )
    }
}


def _load(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def check_configs_match_shot() -> dict:
    report = {}
    for variant in VARIANTS:
        shot = _load(PROJECT_ROOT / "transformer" / variant / "config.yaml")
        # The flat COME config plus the derived per-variant blocks must still
        # present exactly the config this variant used to own as its own YAML.
        come = variant_config_view(_load(COME_CONFIG), variant)

        assert shot["method"] == "shot"
        assert come["method"] == "come"
        assert come["protocol_revision"] == PROTOCOL_REVISIONS[variant]
        assert come["protocol_revision"] != shot["protocol_revision"]
        assert "loss" in shot and "loss" not in come
        assert come["come"] == EXPECTED_COME_BLOCK
        assert come["output"]["root"].endswith(f"transformer_come_otta_{variant}")
        assert come["output"]["root"] != shot["output"]["root"]

        shot_rest = copy.deepcopy(shot)
        come_rest = copy.deepcopy(come)
        for key in ("protocol_revision", "method", "output"):
            shot_rest.pop(key)
            come_rest.pop(key)
        shot_rest.pop("loss")
        come_rest.pop("come")
        shot_optimization = shot_rest.pop("optimization")
        come_optimization = come_rest.pop("optimization")
        assert shot_optimization == {
            "optimizer": "adamw",
            "lr": 1.0e-5,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "weight_decay": 0.01,
        }
        assert come_optimization == {
            **shot_optimization,
            "lr": 1.0e-7,
        }
        allowed = ALLOWED_SELECTION_DIFFERENCES.get(variant, {})
        for field, (shot_value, come_value) in allowed.items():
            assert shot_rest["selection"][field] == shot_value
            assert come_rest["selection"][field] == come_value
            shot_rest["selection"].pop(field)
            come_rest["selection"].pop(field)
        assert shot_rest == come_rest, variant
        report[variant] = {
            "shared_top_level_keys": sorted(come_rest),
            "allowed_selection_differences": sorted(allowed),
            "shot_lr": shot_optimization["lr"],
            "come_lr": come_optimization["lr"],
        }
    return {"config_substrate_identity": report}


def check_candidate_and_group_universe() -> dict:
    """Contract B: exact names/shapes/groups/budgets, shared code objects."""
    assert (
        come_config.candidate_parameter_names
        is shot_candidate_config.candidate_parameter_names
    )
    names = come_config.candidate_parameter_names()
    assert len(names) == come_config.CANDIDATE_TENSOR_COUNT == 12
    assert names == tuple(
        f"blocks.{block}.{suffix}"
        for block in (9, 10, 11)
        for suffix in (
            "attn.qkv.weight",
            "attn.proj.weight",
            "mlp.fc1.weight",
            "mlp.fc2.weight",
        )
    )
    assert come_config.CANDIDATE_SCALAR_COUNT == 5_308_416
    assert come_model.EXPECTED_SHAPES is shot_candidate_model.EXPECTED_SHAPES
    assert (
        come_model.configure_candidate_dense_scope
        is shot_candidate_model.configure_candidate_dense_scope
    )
    scalar_total = sum(
        shape[0] * shape[1]
        for suffix, shape in come_model.EXPECTED_SHAPES.items()
    ) * 3
    assert scalar_total == 5_308_416

    assert come_groups.STRUCTURAL_GROUPS is shot_random_groups.STRUCTURAL_GROUPS
    assert come_groups.STRUCTURAL_GROUPS is shot_random_groups.STRUCTURAL_GROUPS
    assert come_groups.STRUCTURAL_GROUPS is shot_random_groups.STRUCTURAL_GROUPS
    assert len(come_groups.STRUCTURAL_GROUPS) == come_config.TOTAL_GROUPS == 6912
    assert come_config.GROUP_SIZE == 768
    assert come_config.TOTAL_GROUPS * come_config.GROUP_SIZE == 5_308_416
    assert come_groups.build_masks is shot_random_groups.build_masks
    assert come_groups.selected_group_ids is shot_random_groups.selected_group_ids
    assert (
        come_groups.compute_group_l2_scores
        is shot_magnitude_groups.compute_group_l2_scores
    )
    assert (
        come_groups.compute_group_saliency_scores
        is shot_saliency_groups.compute_group_saliency_scores
    )
    assert (
        come_optimizer.strict_masked_adamw_step is shot_optimizer.strict_masked_adamw_step
    )
    assert (
        come_optimizer.assert_off_mask_adam_state_zero
        is shot_optimizer.assert_off_mask_adam_state_zero
    )
    assert come_data.build_target_loaders is shot_data.build_target_loaders
    assert (
        come_data.MergeSingletonTailBatchSampler
        is shot_data.MergeSingletonTailBatchSampler
    )
    assert "load_frozen_source_model" in dir(shot_source_model)
    assert come_model.hash_tensors is shot_candidate_model.hash_tensors

    assert come_config.budget_group_count is shot_random_config.budget_group_count
    integer_budgets = {
        rho: come_config.budget_group_count(rho) for rho in (0.0005, 0.001, 0.002)
    }
    assert integer_budgets == {0.0005: 3, 0.001: 6, 0.002: 13}
    assert come_config.MASK_SEEDS == (202600, 202601, 202602)
    assert come_config.NUM_RANDOM_MASKS == 3
    return {
        "integer_budgets": {str(key): value for key, value in integer_budgets.items()},
        "active_scalars": {
            str(key): value * 768 for key, value in integer_budgets.items()
        },
    }


def check_revised_optimizer_settings() -> None:
    for variant in VARIANTS:
        come = variant_config_view(_load(COME_CONFIG), variant)
        assert come["optimization"] == {
            "optimizer": "adamw",
            "lr": 1.0e-7,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "weight_decay": 0.01,
        }
        assert come["runtime"] == {
            "deterministic": True,
            "amp": False,
            "pin_memory": True,
        }
        assert come["adaptation"]["model_mode"] == "eval"
        assert come["adaptation"]["steps_per_online_batch"] == 1
        assert "scheduler" not in come


def check_revisions_are_independent() -> None:
    assert PROTOCOL_DOCUMENT.endswith("20260911_v2.md")
    assert (PROJECT_ROOT / PROTOCOL_DOCUMENT).is_file()
    assert len(set(PROTOCOL_REVISIONS.values())) == len(PROTOCOL_REVISIONS)
    assert GROUP_LBI_PROTOCOL_DOCUMENT.endswith("20260913_v1.md")
    assert (PROJECT_ROOT / GROUP_LBI_PROTOCOL_DOCUMENT).is_file()
    for variant, revision in PROTOCOL_REVISIONS.items():
        assert revision.startswith("come_transformer_")
        assert "shot" not in revision
    assert IMPLEMENTATION_REVISIONS["full_dense"] == BASELINE_IMPLEMENTATION_REVISION
    assert IMPLEMENTATION_REVISIONS["candidate_dense"] == BASELINE_IMPLEMENTATION_REVISION
    for variant in ("group_random", "group_magnitude", "group_saliency"):
        assert IMPLEMENTATION_REVISIONS[variant] == SPARSE_LBI_IMPLEMENTATION_REVISION
    assert BASELINE_IMPLEMENTATION_REVISION != SPARSE_LBI_IMPLEMENTATION_REVISION
    # Never reuse the ResNet/FC COME revisions.
    for revision in (
        "come_otta_baseline_20260908_v2",
        "come_otta_sparse_lbi_20260908_v3",
        "nips2026_shot_otta_uda_source_v1",
    ):
        assert revision not in set(IMPLEMENTATION_REVISIONS.values())
        assert revision not in set(PROTOCOL_REVISIONS.values())


def check_scientific_identity_changes_with_objective() -> dict:
    base = {
        "protocol_revision": PROTOCOL_REVISIONS["full_dense"],
        "implementation_revision": IMPLEMENTATION_REVISIONS["full_dense"],
        "method": "come",
        "variant": "full_dense",
        "dataset": "office31",
        "source": "dslr",
        "target": "amazon",
        "transfer": "dslr->amazon",
        "num_classes": 31,
        "come": dict(EXPECTED_COME_BLOCK),
        "output_dir": "/tmp/one",
        "checkpoint_manifest_path": "/tmp/manifest.json",
    }
    first = finalize_identity(copy.deepcopy(base), variant="full_dense")
    relocated = copy.deepcopy(base)
    relocated["output_dir"] = "/tmp/two"
    second = finalize_identity(relocated, variant="full_dense")
    assert first["scientific_config_sha256"] == second["scientific_config_sha256"]

    as_shot = copy.deepcopy(base)
    as_shot["method"] = "shot"
    as_shot.pop("come")
    as_shot["loss"] = {"components": ["ent", "div", "pseudo"]}
    third = finalize_identity(as_shot, variant="full_dense")
    assert third["scientific_config_sha256"] != first["scientific_config_sha256"]

    changed_tau = copy.deepcopy(base)
    changed_tau["come"] = {**EXPECTED_COME_BLOCK, "tau": 0.5}
    fourth = finalize_identity(changed_tau, variant="full_dense")
    assert fourth["scientific_config_sha256"] != first["scientific_config_sha256"]
    assert first["experiment_key"].startswith("office31_dslr-amazon_come-full-dense_")
    assert NON_SCIENTIFIC_KEYS >= {"output_dir", "checkpoint_manifest_path"}
    return {"scientific_config_sha256_full_dense_example": first["scientific_config_sha256"]}


def check_validators_fail_closed() -> dict:
    rejected = 0
    raw = load_config(COME_CONFIG)
    validate_raw_config(raw)
    config = variant_config_view(raw, "full_dense")
    validate_variant_config(config, variant="full_dense")
    for mutate in (
        lambda item: item.update({"method": "shot"}),
        lambda item: item.update({"loss": {"cls_par": 0.3}}),
        lambda item: item["come"].update({"tau": 0.5}),
        lambda item: item["come"].update({"renormalize_after_epsilon": True}),
        lambda item: item["come"].update({"norm_epsilon": 1.0e-6}),
        lambda item: item["come"].update({"opinion_implementation": "direct_exp"}),
        lambda item: item["optimization"].update({"lr": 1.0e-5}),
        lambda item: item["data"]["preprocessing"].update({"interpolation": "bicubic"}),
        lambda item: item["adaptation"].update({"model_mode": "train"}),
    ):
        broken = copy.deepcopy(config)
        mutate(broken)
        try:
            validate_variant_config(broken, variant="full_dense")
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("A frozen-field mutation was accepted")

    saliency = variant_config_view(raw, "group_saliency")
    validate_variant_config(saliency, variant="group_saliency")
    broken = copy.deepcopy(saliency)
    broken["selection"]["integer_rule"] = "ceil"
    try:
        validate_variant_config(broken, variant="group_saliency")
    except ValueError:
        rejected += 1
    else:
        raise AssertionError("A non-floor integer rule was accepted")

    # The shared YAML itself must stay fail-closed: the blocks it still owns
    # cannot drift, and it must not regrow the per-variant keys the view
    # derives or a SHOT loss block.
    for mutate in (
        lambda item: item.update({"method": "shot"}),
        lambda item: item.update({"loss": {"cls_par": 0.3}}),
        lambda item: item.update({"variant": "full_dense"}),
        lambda item: item.update({"selection": {"integer_rule": "ceil"}}),
        lambda item: item.update({"protocol_document": "some_other_protocol_v9"}),
        lambda item: item["come"].update({"tau": 0.5}),
        lambda item: item["optimization"].update({"lr": 1.0e-5}),
        lambda item: item["data"]["preprocessing"].update({"interpolation": "bicubic"}),
        lambda item: item["runtime"].update({"amp": True}),
    ):
        broken_raw = copy.deepcopy(raw)
        mutate(broken_raw)
        try:
            validate_raw_config(broken_raw)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("A frozen shared-config mutation was accepted")

    # Group-LBI is implemented, while provisional tuples remain formally
    # blocked and cannot fall back to a dense identity.
    lbi_view = variant_config_view(raw, "group_lbi")
    validate_variant_config(lbi_view, variant="group_lbi")
    assert come_config.require_supported_variant("group_lbi") == "group_lbi"
    assert lbi_view["selection"]["type"] == "group_split_lbi"
    try:
        resolve_lbi_profile(raw, "office31", 0.0005)
    except ValueError as error:
        assert "formal run is blocked" in str(error)
        rejected += 1
    else:
        raise AssertionError("A provisional Group-LBI tuple passed the formal gate")
    provisional = resolve_lbi_profile(
        raw, "office31", 0.0005, allow_provisional=True
    )
    assert provisional["formal_eligible"] is False
    return {"rejected_frozen_field_mutations": rejected}


def main() -> int:
    report = {"status": "PASS"}
    report.update(check_configs_match_shot())
    report.update(check_candidate_and_group_universe())
    check_revised_optimizer_settings()
    check_revisions_are_independent()
    report.update(check_scientific_identity_changes_with_objective())
    report.update(check_validators_fail_closed())
    print(json.dumps(report, sort_keys=True, indent=2))
    print("Transformer COME substrate identity tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
