#!/usr/bin/env python3
"""CPU contracts for the COME Group-Random baseline (C12, C13, C17)."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys
import tempfile

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(PROJECT_ROOT), str(PROJECT_ROOT / "tests")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import transformer_come.common as come_common  # noqa: E402
import transformer_come.runner as runner_module  # noqa: E402
from transformer.candidate_dense.model import hash_tensors  # noqa: E402
from transformer_come.config import budget_group_count  # noqa: E402
from transformer_come.config import (  # noqa: E402
    SELECTION_BLOCKS,
    FORMAL_BUDGETS,
    IMPLEMENTATION_REVISIONS,
    MASK_SEEDS,
    NUM_RANDOM_MASKS,
    PROTOCOL_REVISIONS,
    validate_variant_config,
    variant_config_view,
    load_config,
)
from transformer_come.groups import (  # noqa: E402
    mask_sha256,
    selected_group_ids,
)

from transformer_come_fixtures import (  # noqa: E402
    BATCH_SIZE,
    CandidateShapedModel,
    SAMPLE_COUNT,
    build_synthetic_loaders,
    load_candidate_scope_model,
    read_metrics,
    sparse_selection,
    synthetic_config,
)


VARIANT = "group_random"
CONFIG_PATH = PROJECT_ROOT / "transformer_come" / "config.yaml"
BUDGET = 0.001


def check_config() -> None:
    config = variant_config_view(load_config(CONFIG_PATH), VARIANT)
    validate_variant_config(config, variant=VARIANT)
    assert config["protocol_revision"] == PROTOCOL_REVISIONS[VARIANT]
    assert config["method"] == "come"
    assert config["selection"] == SELECTION_BLOCKS[VARIANT]
    assert config["selection"]["mask_seeds"] == [202600, 202601, 202602]
    assert config["selection"]["num_random_masks"] == 3
    assert config["selection"]["cross_budget_policy"] == "nested_prefix_same_permutation"


def check_nested_prefix_supports() -> dict:
    """C13: same permutation per seed, nested across budgets, three supports."""
    assert MASK_SEEDS == (202600, 202601, 202602)
    assert NUM_RANDOM_MASKS == 3
    for seed in MASK_SEEDS:
        low, medium, high = (selected_group_ids(rho, seed) for rho in FORMAL_BUDGETS)
        assert (len(low), len(medium), len(high)) == (3, 6, 13)
        assert medium[: len(low)] == low
        assert high[: len(medium)] == medium
        assert selected_group_ids(0.002, seed) == high
        assert mask_sha256(high) == mask_sha256(selected_group_ids(0.002, seed))
    supports = [frozenset(selected_group_ids(0.002, seed)) for seed in MASK_SEEDS]
    assert len(set(supports)) == 3
    return {"distinct_child_supports": len(set(supports))}


def _run_parent():
    CandidateShapedModel.forward_calls = 0
    objective_calls = {"count": 0}
    loaded_models = []
    initial_hashes = []
    original_come_loss = come_common.come_loss
    original_loaders = runner_module.build_target_loaders

    def counting_come_loss(logits, class_count, **kwargs):
        objective_calls["count"] += 1
        return original_come_loss(logits, class_count, **kwargs)

    def capturing_loader(config, device):
        loaded = load_candidate_scope_model(config, device)
        loaded_models.append(loaded[0])
        initial_hashes.append(hash_tensors(loaded[2]))
        return loaded

    def child_runner(config, project_root, *, show_progress=False):
        return runner_module.run_mask_child(
            config,
            project_root,
            show_progress=False,
            model_loader=capturing_loader,
        )

    come_common.come_loss = counting_come_loss
    runner_module.build_target_loaders = build_synthetic_loaders
    try:
        with tempfile.TemporaryDirectory(prefix="come_group_random_") as value:
            output_dir = Path(value) / "run"
            config = synthetic_config(
                "group_random",
                output_dir=output_dir,
                experiment_key="synthetic-come-group-random-parent",
                selection=sparse_selection(BUDGET, budget_group_count(BUDGET)),
            )
            summary = runner_module.run_transfer(
                config, PROJECT_ROOT, show_progress=False, child_runner=child_runner
            )
            children = [
                json.loads(
                    (output_dir / f"mask_{index:02d}" / "summary.json").read_text(
                        encoding="utf-8"
                    )
                )
                for index in range(NUM_RANDOM_MASKS)
            ]
            child_rows = [
                read_metrics(output_dir / f"mask_{index:02d}")
                for index in range(NUM_RANDOM_MASKS)
            ]
            return (
                summary,
                children,
                child_rows,
                objective_calls["count"],
                loaded_models,
                initial_hashes,
            )
    finally:
        come_common.come_loss = original_come_loss
        runner_module.build_target_loaders = original_loaders


def check_three_real_children() -> dict:
    summary, children, child_rows, objective_calls, models, initial_hashes = _run_parent()
    batch_count = SAMPLE_COUNT // BATCH_SIZE
    group_count = budget_group_count(BUDGET)

    # Three real trajectories, not one run relabelled three times.
    assert len(children) == len(models) == NUM_RANDOM_MASKS == 3
    assert objective_calls == NUM_RANDOM_MASKS * batch_count
    assert [child["random_mask_index"] for child in children] == [0, 1, 2]
    assert [child["mask_seed"] for child in children] == list(MASK_SEEDS)
    assert len({child["experiment_key"] for child in children}) == 3
    assert len({child["scientific_config_sha256"] for child in children}) == 3
    assert len({child["selection"]["mask_sha256"] for child in children}) == 3
    assert all(child["parent_experiment_key"] == summary["experiment_key"] for child in children)
    for rows in child_rows:
        online = [row for row in rows if row["event"] == "online_batch"]
        assert len(online) == batch_count
        assert all(row["objective_call_count"] == 1 for row in online)
        assert all(row["optimizer_step_count"] == 1 for row in online)
        assert all(row["scheduler_step_count"] == 0 for row in online)

    # Every child restarts from the same W0 with the same stream.
    reference = CandidateShapedModel()
    reference_hash = hash_tensors(
        [
            (name, parameter)
            for name, parameter in reference.named_parameters()
            if name.startswith(("blocks.9.", "blocks.10.", "blocks.11."))
        ]
    )
    assert set(initial_hashes) == {reference_hash}
    final_hashes = {
        hash_tensors(
            [
                (name, parameter)
                for name, parameter in model.named_parameters()
                if name.startswith(("blocks.9.", "blocks.10.", "blocks.11."))
            ]
        )
        for model in models
    }
    assert len(final_hashes) == 3
    assert reference_hash not in final_hashes

    for child in children:
        assert child["requested_group_count"] == group_count == 6
        assert child["realized_group_count"] == group_count
        assert child["active_candidate_scalars"] == group_count * 768
        assert child["off_mask_state_unchanged"] is True
        assert child["off_mask_adam_state_zero"] is True
        assert child["frozen_state_unchanged"] is True
        assert child["method"] == "come"

    # Top-level accuracy is the child mean with population std.
    for prefix in ("PU", "FO"):
        values = [child[f"{prefix}-Acc"] for child in children]
        assert abs(summary[f"{prefix}-Acc"] - statistics.fmean(values)) < 1e-12
        assert abs(summary[f"{prefix}-Acc-mask-std"] - statistics.pstdev(values)) < 1e-12
    assert summary["method"] == "come"
    assert summary["implementation_revision"] == IMPLEMENTATION_REVISIONS[VARIANT]
    assert summary["num_random_masks"] == 3
    assert summary["mask_seeds"] == list(MASK_SEEDS)
    assert len(summary["masks"]) == 3
    assert summary["random_total_online_compute_runtime_sec"] >= summary[
        "online_compute_runtime_sec"
    ]
    return {
        "children": len(children),
        "objective_calls": objective_calls,
        "requested_group_count": group_count,
    }


def check_child_supports_match_shot_seeds() -> None:
    """The COME child support is the SHOT child support for the same seed."""
    import transformer.group_random.groups as shot_groups

    for seed in MASK_SEEDS:
        assert selected_group_ids(BUDGET, seed) == shot_groups.selected_group_ids(
            BUDGET, seed
        )


def main() -> int:
    check_config()
    report = {"status": "PASS"}
    report.update(check_nested_prefix_supports())
    report.update(check_three_real_children())
    check_child_supports_match_shot_seeds()
    print(json.dumps(report, sort_keys=True))
    print("Transformer COME Group-Random tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
