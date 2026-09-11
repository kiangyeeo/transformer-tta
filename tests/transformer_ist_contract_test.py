#!/usr/bin/env python3
"""CPU contracts for IST mechanism, stream, configuration, and selectors."""

from __future__ import annotations

import copy
import importlib
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformer.group_random.config import MASK_SEEDS  # noqa: E402
from transformer.group_random.groups import selected_group_ids  # noqa: E402
from transformer_ist.config import (  # noqa: E402
    PROTOCOL_REVISION,
    budget_group_count,
    load_config,
    resolve_transfer_config,
)
from transformer_ist.data import ISTViewMaterializer, build_ist_loaders  # noqa: E402
from transformer_ist.ema import OuterBatchEMA  # noqa: E402
from transformer_ist.memory import CausalMemoryBank  # noqa: E402
from transformer_ist.objective import (  # noqa: E402
    FixedISTTask,
    accumulate_full_objective,
    ist_loss,
)
from transformer_ist.plca import robust_plca  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "transformer_ist" / "config.yaml"


def check_config_and_identity() -> None:
    raw = load_config(CONFIG_PATH)
    assert raw["protocol_revision"] == PROTOCOL_REVISION
    dense = resolve_transfer_config(
        raw,
        project_root=PROJECT_ROOT,
        dataset="office31",
        source="amazon",
        target="dslr",
        variant="full_dense",
        budget=None,
        device="cpu",
        output_dir="/tmp/ist-dry-dense",
    )
    sparse = resolve_transfer_config(
        raw,
        project_root=PROJECT_ROOT,
        dataset="office31",
        source="amazon",
        target="dslr",
        variant="group_random",
        budget=0.001,
        device="cpu",
        output_dir="/tmp/ist-dry-sparse",
        debug_max_outer_batches=5,
    )
    assert dense["formal_protocol"] is True
    assert sparse["formal_protocol"] is False
    assert sparse["debug_smoke"] is True
    assert sparse["selection"]["requested_group_count"] == 6
    assert dense["adaptation"]["candidate_tensor_count"] == 12
    assert dense["adaptation"]["candidate_scalar_count"] == 5_308_416
    assert dense["adaptation"]["structural_groups"]["total_groups"] == 6912
    assert dense["adaptation"]["structural_groups"]["group_size"] == 768
    assert [budget_group_count(value) for value in (0.0005, 0.001, 0.002)] == [
        3,
        6,
        13,
    ]
    same_science = resolve_transfer_config(
        raw,
        project_root=PROJECT_ROOT,
        dataset="office31",
        source="amazon",
        target="dslr",
        variant="full_dense",
        budget=None,
        device="cpu",
        output_dir="/tmp/a-different-output-only",
    )
    debug_science = resolve_transfer_config(
        raw,
        project_root=PROJECT_ROOT,
        dataset="office31",
        source="amazon",
        target="dslr",
        variant="full_dense",
        budget=None,
        device="cpu",
        output_dir="/tmp/debug-science",
        debug_max_outer_batches=1,
    )
    assert same_science["scientific_config_sha256"] == dense[
        "scientific_config_sha256"
    ]
    assert debug_science["scientific_config_sha256"] != dense[
        "scientific_config_sha256"
    ]
    changed = copy.deepcopy(raw)
    changed["optimization"]["lr"] = 2.0e-5
    try:
        resolve_transfer_config(
            changed,
            project_root=PROJECT_ROOT,
            dataset="office31",
            source="amazon",
            target="dslr",
            variant="full_dense",
            budget=None,
            device="cpu",
            output_dir="/tmp/rejected",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("non-protocol optimizer was accepted")
    try:
        resolve_transfer_config(
            raw,
            project_root=PROJECT_ROOT,
            dataset="office31",
            source="amazon",
            target="dslr",
            variant="group_lbi",
            budget=0.001,
            device="cpu",
            output_dir="/tmp/rejected-lbi",
        )
    except ValueError as error:
        assert "not implemented" in str(error)
    else:
        raise AssertionError("group_lbi was accepted")
    unexpected = copy.deepcopy(raw)
    unexpected["lbi"] = {"fallback": "dense"}
    try:
        resolve_transfer_config(
            unexpected,
            project_root=PROJECT_ROOT,
            dataset="office31",
            source="amazon",
            target="dslr",
            variant="full_dense",
            budget=None,
            device="cpu",
            output_dir="/tmp/rejected-extra-config",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("unknown/LBI config keys were silently ignored")


class CountingMaterializer(ISTViewMaterializer):
    decode_count = 0

    @staticmethod
    def _decode(raw_image):
        CountingMaterializer.decode_count += 1
        return ISTViewMaterializer._decode(raw_image)


def check_raw_views_and_rng() -> None:
    config = {
        "formal_seed": 2026,
        "ist": load_config(CONFIG_PATH)["ist"],
    }
    with tempfile.TemporaryDirectory(prefix="transformer_ist_views_") as value:
        path = Path(value) / "sample.png"
        Image.new("RGB", (300, 280), color=(12, 34, 56)).save(path)
        CountingMaterializer.decode_count = 0
        first = CountingMaterializer(config).materialize([str(path)], [9])
        second = CountingMaterializer(config).materialize([str(path)], [9])
        assert CountingMaterializer.decode_count == 2
        assert first.reference_views.shape == (1, 3, 224, 224)
        assert first.adaptation_views.shape == (8, 3, 224, 224)
        assert torch.equal(first.reference_views, second.reference_views)
        assert torch.equal(first.adaptation_views, second.adaptation_views)
        assert first.augmentation_trace_sha256 == second.augmentation_trace_sha256
        assert first.reference_trace_sha256 != first.adaptation_trace_sha256
        try:
            CountingMaterializer(config).materialize(
                [torch.zeros(3, 224, 224)], [0]
            )
        except TypeError:
            pass
        else:
            raise AssertionError("normalized tensor input was accepted")
        try:
            CountingMaterializer(config).materialize([str(path)], [0, 1])
        except ValueError:
            pass
        else:
            raise AssertionError("misaligned raw images/sample indices were accepted")


def check_amazon_tail_and_random_prefix() -> None:
    raw = load_config(CONFIG_PATH)
    config = resolve_transfer_config(
        raw,
        project_root=PROJECT_ROOT,
        dataset="office31",
        source="dslr",
        target="amazon",
        variant="candidate_dense",
        budget=None,
        device="cpu",
        output_dir="/tmp/ist-amazon-loader",
        debug_max_outer_batches=1,
    )
    online, _, record = build_ist_loaders(config)
    assert len(online) == 44
    assert record["batch_size_sequence"] == [64] * 43 + [65]
    assert record["singleton_tail_merged_into_previous_batch"] is True
    assert MASK_SEEDS == (202600, 202601, 202602)
    for seed in MASK_SEEDS:
        small = selected_group_ids(0.0005, seed)
        medium = selected_group_ids(0.001, seed)
        large = selected_group_ids(0.002, seed)
        assert medium[: len(small)] == small
        assert large[: len(medium)] == medium


def check_objective_and_full_accumulation() -> None:
    logits = torch.tensor([[2.0, -1.0], [-0.2, 0.8]], requires_grad=True)
    hard = torch.tensor([0, 1])
    soft = torch.tensor([[0.75, 0.25], [0.3, 0.7]])
    config = {"hard_ce_weight": 1.0, "soft_kl_weight": 1.0}
    total, _ = ist_loss(logits, hard, soft, config)
    reference = torch.nn.functional.cross_entropy(
        logits, hard
    ) + torch.nn.functional.kl_div(
        torch.log_softmax(logits, dim=-1), soft, reduction="batchmean"
    )
    assert torch.equal(total, reference)

    torch.manual_seed(4)
    inputs = torch.randn(5, 3)
    hard = torch.tensor([0, 1, 0, 1, 0])
    soft = torch.softmax(torch.randn(5, 2), dim=-1)
    task = FixedISTTask(inputs, hard, soft)
    whole = nn.Linear(3, 2, bias=False)
    chunked = copy.deepcopy(whole)
    accumulate_full_objective(
        task,
        whole,
        [("weight", whole.weight)],
        config,
        chunk_size=5,
    )
    accumulate_full_objective(
        task,
        chunked,
        [("weight", chunked.weight)],
        config,
        chunk_size=2,
    )
    assert torch.allclose(whole.weight.grad, chunked.weight.grad, atol=1.0e-7)
    assert not hasattr(task, "labels")
    original = task.soft_targets.clone()
    soft.add_(1.0)
    assert torch.equal(task.soft_targets, original)


def check_plca_memory_and_ema() -> None:
    memory = CausalMemoryBank(3, feature_dim=2, num_classes=2)
    features = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    labels = torch.eye(2)
    snapshot = memory.snapshot_for(0, "cpu")
    config = load_config(CONFIG_PATH)["ist"]["plca"]
    result = robust_plca(features, labels, snapshot, config)
    assert result.graph_sample_count == 2 and result.memory_sample_count == 0
    memory.commit(0, features, result.corrected_one_hot)
    memory.commit(1, features + 0.1, result.corrected_one_hot)
    assert len(memory) == 3 and memory.commit_count == 2
    assert memory.snapshot_for(2, "cpu").batch_ids.tolist() == [0, 1, 1]
    try:
        memory.commit(1, features, labels)
    except RuntimeError:
        pass
    else:
        raise AssertionError("duplicate memory commit was accepted")

    model = nn.Linear(2, 1)
    ema = OuterBatchEMA(0.9)
    old = model.weight.detach().clone()
    ema.begin_batch(0, [("weight", model.weight)])
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.commit(0, [("weight", model.weight)])
    assert torch.allclose(model.weight, old + 0.1)
    assert ema.commit_count == 1


def check_plca_matches_fc_reference() -> None:
    refined_root = PROJECT_ROOT / "260817_iclr2027-refined"
    sys.path.insert(0, str(refined_root))
    try:
        reference_plca = importlib.import_module("ist_otta.plca")
        reference_memory = importlib.import_module("ist_otta.memory")
    finally:
        sys.path.pop(0)
    generator = torch.Generator().manual_seed(77)
    current_features = torch.randn(9, 4, generator=generator)
    soft_targets = torch.softmax(
        torch.randn(9, 3, generator=generator), dim=1
    )
    prior_features = torch.randn(5, 4, generator=generator)
    prior_labels = torch.nn.functional.one_hot(
        torch.tensor([0, 1, 2, 1, 0]), num_classes=3
    ).float()
    memory_type = type(CausalMemoryBank(1, 1, 1).snapshot_for(0, "cpu"))
    ours_memory = memory_type(
        prior_features, prior_labels, torch.zeros(5, dtype=torch.long)
    )
    theirs_memory = reference_memory.MemorySnapshot(
        prior_features, prior_labels, torch.zeros(5, dtype=torch.long)
    )
    ours_config = copy.deepcopy(load_config(CONFIG_PATH)["ist"]["plca"])
    theirs_config = {
        **ours_config,
        "solver_tolerance": ours_config["solver_rtol"],
    }
    ours = robust_plca(current_features, soft_targets, ours_memory, ours_config)
    theirs = reference_plca.robust_plca(
        current_features, soft_targets, theirs_memory, theirs_config
    )
    assert torch.equal(ours.corrected_hard_labels, theirs.corrected_hard_labels)
    assert torch.equal(ours.corrected_one_hot, theirs.corrected_one_hot)


def main() -> int:
    check_config_and_identity()
    check_raw_views_and_rng()
    check_amazon_tail_and_random_prefix()
    check_objective_and_full_accumulation()
    check_plca_memory_and_ema()
    check_plca_matches_fc_reference()
    print("Transformer IST mechanism/config contracts passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
