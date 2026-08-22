#!/usr/bin/env python3
"""CPU-only protocol checks for Transformer Group-Magnitude SHOT-OTTA."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformer.candidate_dense.config import candidate_parameter_names  # noqa: E402
from transformer.candidate_dense.model import EXPECTED_SHAPES  # noqa: E402
from transformer.group_magnitude.aggregate import aggregate_matrix, write_aggregate  # noqa: E402
from transformer.group_magnitude.config import (  # noqa: E402
    BUDGET_TO_K,
    FORMAL_BUDGETS,
    FORMAL_SEED,
    GROUP_SIZE,
    PROTOCOL_REVISION,
    TOTAL_GROUPS,
    TRANSFERS,
    _validate_frozen_fields,
    budget_group_count,
    budget_tag,
    load_config,
    parse_budgets,
    resolve_transfer_config,
)
from transformer.group_magnitude.groups import (  # noqa: E402
    STRUCTURAL_GROUPS,
    build_masks,
    compute_group_l2_scores,
    magnitude_mask_record,
    score_vector_sha256,
    select_magnitude_group_ids,
)
from transformer.group_magnitude.matrix import validate_devices  # noqa: E402
from transformer.group_magnitude.optimizer import (  # noqa: E402
    assert_off_mask_adam_state_zero,
    strict_masked_adamw_step,
)
from transformer.source_only.metrics import FixedClassMeter  # noqa: E402
import transformer.group_magnitude.runner as runner_module  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "transformer" / "group_magnitude" / "config.yaml"


def check_config_budgets_and_identity() -> None:
    config = load_config(CONFIG_PATH)
    _validate_frozen_fields(config)
    assert config["formal_seed"] == FORMAL_SEED == 2026
    assert config["protocol_revision"] == PROTOCOL_REVISION
    assert FORMAL_BUDGETS == (0.005, 0.01, 0.02)
    assert BUDGET_TO_K == {0.005: 34, 0.01: 69, 0.02: 138}
    assert [budget_group_count(item) for item in FORMAL_BUDGETS] == [34, 69, 138]
    assert parse_budgets("all") == FORMAL_BUDGETS
    assert parse_budgets("0.005,0.02") == (0.005, 0.02)
    validate_devices(["0", "1"])
    for invalid in ([], ["cpu", "0"], ["0", "0"]):
        try:
            validate_devices(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid devices were accepted: {invalid}")

    changed = copy.deepcopy(config)
    changed["selection"]["integer_budgets"] = [35, 70, 139]
    try:
        _validate_frozen_fields(changed)
    except ValueError as error:
        assert "selection" in str(error)
    else:
        raise AssertionError("ceil budgets were accepted")


def _write_fake_resolver_assets(root: Path) -> dict:
    config = load_config(CONFIG_PATH)
    list_root = root / "lists"
    checkpoint_root = root / "checkpoints"
    for target in ("dslr", "webcam"):
        target_list = list_root / "office31" / f"{target}_list.txt"
        target_list.parent.mkdir(parents=True, exist_ok=True)
        target_list.write_text(
            "".join(f"/synthetic/{index}.jpg {index}\n" for index in range(31)),
            encoding="utf-8",
        )
    mapping = {f"class-{index}": index for index in range(31)}
    (list_root / "office31" / "class_to_idx.json").write_text(
        json.dumps(
            {
                "dataset": "office31",
                "class_to_idx": mapping,
                "domains": {
                    target: {
                        "path": str(
                            (list_root / "office31" / f"{target}_list.txt").resolve()
                        )
                    }
                    for target in ("dslr", "webcam")
                },
            }
        ),
        encoding="utf-8",
    )
    checkpoint = checkpoint_root / "office31" / "amazon.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic-checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    checkpoint.with_suffix(".manifest.json").write_text(
        json.dumps({"checkpoint": {"path": str(checkpoint.resolve()), "sha256": digest}}),
        encoding="utf-8",
    )
    config["data"]["list_root"] = str(list_root)
    config["model"]["checkpoint_root"] = str(checkpoint_root)
    return config


def check_resolver_identity_and_same_source() -> None:
    with tempfile.TemporaryDirectory(prefix="group_magnitude_resolver_") as value:
        root = Path(value)
        config = _write_fake_resolver_assets(root)
        dslr = resolve_transfer_config(
            config,
            project_root=PROJECT_ROOT,
            dataset="office31",
            source="amazon",
            target="dslr",
            budget=0.005,
            device="cpu",
            output_dir=root / "dslr",
        )
        webcam = resolve_transfer_config(
            config,
            project_root=PROJECT_ROOT,
            dataset="office31",
            source="amazon",
            target="webcam",
            budget=0.005,
            device="cpu",
            output_dir=root / "webcam",
        )
        high = resolve_transfer_config(
            config,
            project_root=PROJECT_ROOT,
            dataset="office31",
            source="amazon",
            target="dslr",
            budget=0.02,
            device="cpu",
            output_dir=root / "high",
        )
        assert dslr["selection"]["requested_group_count"] == 34
        assert high["selection"]["requested_group_count"] == 138
        assert dslr["checkpoint_sha256"] == webcam["checkpoint_sha256"]
        assert dslr["scientific_config_sha256"] != high["scientific_config_sha256"]


def _candidate_tensors():
    values = []
    for name in candidate_parameter_names():
        suffix = name.split(".", maxsplit=2)[2]
        values.append((name, nn.Parameter(torch.zeros(EXPECTED_SHAPES[suffix]))))
    return values


def _numpy_rng_states_equal(first, second) -> bool:
    return (
        first[0] == second[0]
        and np.array_equal(first[1], second[1])
        and first[2:] == second[2:]
    )


def check_group_l2_ranking_ties_and_rng() -> None:
    assert len(STRUCTURAL_GROUPS) == TOTAL_GROUPS == 6912
    candidates = _candidate_tensors()
    lookup = dict(candidates)
    # QK group 0: L2/L1=3/3; group 1: L2/L1=sqrt(8)/4.
    # A sum(abs(W)) implementation would incorrectly rank group 1 first.
    with torch.no_grad():
        qkv = lookup["blocks.9.attn.qkv.weight"]
        qkv[0, 0] = 3.0
        qkv[1, 0] = 2.0
        qkv[1, 1] = 2.0

    random.seed(2718)
    np.random.seed(2718)
    torch.manual_seed(2718)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    scores = compute_group_l2_scores(candidates)
    low = select_magnitude_group_ids(scores, 0.005)
    medium = select_magnitude_group_ids(scores, 0.01)
    high = select_magnitude_group_ids(scores, 0.02)
    assert random.getstate() == python_before
    assert _numpy_rng_states_equal(np.random.get_state(), numpy_before)
    assert torch.equal(torch.random.get_rng_state(), torch_before)

    assert scores.dtype == torch.float64 and scores.device.type == "cpu"
    assert scores[0].item() == 3.0
    assert abs(scores[1].item() - (8.0**0.5)) < 1.0e-12
    assert low[:4] == (0, 1, 2, 3)
    assert medium[: len(low)] == low
    assert high[: len(medium)] == medium
    assert select_magnitude_group_ids(scores, 0.02) == high
    assert len(low) == 34 and len(medium) == 69 and len(high) == 138
    record = magnitude_mask_record(
        low, scores=scores, budget=0.005, checkpoint_sha256="a" * 64
    )
    assert record["score_vector_sha256"] == score_vector_sha256(scores)
    assert record["selected_group_ids"] == list(low)
    assert record["top_k_min_score"] == 0.0

    focused = build_masks(candidates, [0, 384, 768])
    qkv_mask = focused["blocks.9.attn.qkv.weight"]
    assert qkv_mask[0, :].all() and qkv_mask[384, :].all()
    assert qkv_mask[768, :].all()
    assert focused["blocks.9.attn.proj.weight"][:, 0].all()
    assert focused["blocks.9.mlp.fc1.weight"][0, :].all()
    assert focused["blocks.9.mlp.fc2.weight"][:, 0].all()
    assert sum(int(mask.count_nonzero()) for mask in focused.values()) == 3 * GROUP_SIZE


def check_strict_sparse_adamw() -> None:
    parameter = nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    named = [("parameter", parameter)]
    mask = torch.tensor([True, False, True, False])
    masks = {"parameter": mask}
    optimizer = torch.optim.AdamW([parameter], lr=0.1, weight_decay=0.2)
    off_before = parameter.detach()[~mask].clone()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        parameter.grad = torch.ones_like(parameter)
        strict_masked_adamw_step(optimizer, named, masks)
        assert torch.equal(parameter.detach()[~mask], off_before)
        assert_off_mask_adam_state_zero(optimizer, named, masks)


def check_dataset_metrics() -> None:
    office = FixedClassMeter(2, ["many", "few"])
    labels = torch.tensor([0] * 10 + [1])
    predictions = torch.tensor([0] * 9 + [1, 0])
    office.update(labels[:4], predictions[:4])
    office.update(labels[4:], predictions[4:])
    office_metrics = office.compute("office31")
    assert office_metrics["sample-count"] == 11
    assert abs(office_metrics["Acc"] - 100.0 * 9 / 11) < 1.0e-12

    names = [f"class-{index}" for index in range(12)]
    visda = FixedClassMeter(12, names)
    labels = torch.tensor([0] * 10 + list(range(1, 12)))
    predictions = labels.clone()
    predictions[-1] = 0
    visda.update(labels, predictions)
    visda_metrics = visda.compute("visda-c")
    assert len(visda_metrics["Acc-per-class"]) == 12
    assert abs(visda_metrics["Acc"] - 100.0 * 11 / 12) < 1.0e-12
    assert visda_metrics["Acc"] != visda_metrics["overall-Acc"]


class OrderedSampler(Sampler[int]):
    def __init__(self, order):
        self.order = list(order)

    def __iter__(self):
        return iter(self.order)

    def __len__(self):
        return len(self.order)


class TinyDataset(Dataset):
    def __init__(self):
        self.images = torch.tensor(
            [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.5, 0.0, 1.0]]
        )
        self.labels = torch.tensor([0, 1, 0])

    def __len__(self):
        return 3

    def __getitem__(self, index):
        return self.images[index], self.labels[index], index


class TinyMagnitudeModel(nn.Module):
    forward_calls = 0

    def __init__(self):
        super().__init__()
        self.candidate = nn.Linear(3, 3, bias=False)
        self.head = nn.Linear(3, 2, bias=False)
        self.register_buffer("read_only_buffer", torch.tensor([1.0]))

    def get_classifier(self):
        return self.head

    def forward(self, images):
        type(self).forward_calls += 1
        return self.head(torch.tanh(self.candidate(images)))


def _tiny_model_loader(config, device):
    del config
    model = TinyMagnitudeModel().to(device)
    model.requires_grad_(False)
    model.candidate.weight.requires_grad_(True)
    model.eval()
    candidates = [("candidate.weight", model.candidate.weight)]
    frozen = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    ]
    return model, {"path": "synthetic", "sha256": "0" * 64}, candidates, frozen, {
        "candidate_scope": "synthetic",
        "trainable_tensor_count": 1,
        "trainable_scalars": model.candidate.weight.numel(),
    }


def _tiny_loaders(config):
    del config
    dataset = TinyDataset()
    online = DataLoader(dataset, batch_size=2, sampler=OrderedSampler([2, 0, 1]), drop_last=False)
    final = DataLoader(dataset, batch_size=2, sampler=OrderedSampler([0, 1, 2]), drop_last=False)
    return online, final, {
        "sampler": "fixed_random_permutation",
        "seed": 2026,
        "sample_count": 3,
        "batch_count": 2,
        "online_batch_size": 2,
        "fo_batch_count": 2,
        "fo_batch_size": 2,
        "drop_last": False,
        "online_order_sha256": "synthetic-online",
        "fo_sampler": "sequential",
        "fo_order_sha256": "synthetic-fo",
    }


def check_end_to_end_pu_fo() -> None:
    TinyMagnitudeModel.forward_calls = 0
    config = {
        "output_dir": "",
        "runtime": {"device": "cpu", "deterministic": True, "amp": False, "pin_memory": False},
        "formal_seed": 2026,
        "protocol_revision": PROTOCOL_REVISION,
        "experiment_key": "synthetic-magnitude",
        "scientific_config_sha256": "1" * 64,
        "dataset": "office31",
        "source": "amazon",
        "target": "dslr",
        "transfer": "amazon->dslr",
        "num_classes": 2,
        "class_names": ["zero", "one"],
        "batch_size": 2,
        "fo_batch_size": 2,
        "adaptation": {"update_scope": "synthetic", "model_mode": "eval", "steps_per_online_batch": 1},
        "selection": {"requested_budget": 0.005, "requested_group_count": 1, "active_candidate_scalars": 2},
        "optimization": {"optimizer": "adamw", "lr": 1.0e-2, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.01},
        "loss": {"components": ["ent", "div", "pseudo"], "cls_par": 0.3, "ent_par": 1.0, "threshold": 0.0},
        "preprocessing": {"interpolation": "bilinear"},
    }
    originals = {
        "loaders": runner_module.build_target_loaders,
        "scores": runner_module.compute_group_l2_scores,
        "select": runner_module.select_magnitude_group_ids,
        "record": runner_module.magnitude_mask_record,
        "masks": runner_module.build_masks,
    }
    runner_module.build_target_loaders = _tiny_loaders
    runner_module.compute_group_l2_scores = lambda candidates: torch.tensor([1.0])
    runner_module.select_magnitude_group_ids = lambda scores, budget: (0,)
    runner_module.magnitude_mask_record = lambda ids, scores, budget, checkpoint_sha256: {
        "schema_version": 1,
        "requested_budget": budget,
        "requested_group_count": 1,
        "realized_group_count": 1,
        "active_candidate_scalars": 2,
        "score_vector_sha256": "synthetic-scores",
        "mask_sha256": "synthetic-mask",
        "selected_group_ids": [0],
    }
    runner_module.build_masks = lambda candidates, ids: {
        "candidate.weight": torch.tensor(
            [[True, False, False], [False, True, False], [False, False, False]]
        )
    }
    try:
        with tempfile.TemporaryDirectory(prefix="group_magnitude_runner_") as value:
            config["output_dir"] = str(Path(value) / "condition")
            summary = runner_module.run_transfer(
                config, PROJECT_ROOT, show_progress=False, model_loader=_tiny_model_loader
            )
            assert summary["status"] == "completed"
            assert summary["adaptation_steps"] == 2
            assert summary["backward_calls"] == 2
            assert summary["off_mask_state_unchanged"] is True
            assert summary["off_mask_adam_state_zero"] is True
            assert summary["frozen_head_unchanged"] is True
            assert summary["PU-sample-count"] == 3
            assert summary["FO-sample-count"] == 3
            assert summary["pu_is_post_update_same_batch"] is True
            assert summary["fo_is_independent_full_target_pass"] is True
            assert TinyMagnitudeModel.forward_calls == 6
            rows = [
                json.loads(line)
                for line in (Path(config["output_dir"]) / "metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            online = [row for row in rows if row["event"] == "online_batch"]
            assert [row["batch_size"] for row in online] == [2, 1]
    finally:
        runner_module.build_target_loaders = originals["loaders"]
        runner_module.compute_group_l2_scores = originals["scores"]
        runner_module.select_magnitude_group_ids = originals["select"]
        runner_module.magnitude_mask_record = originals["record"]
        runner_module.build_masks = originals["masks"]


def _fake_summary(dataset, source, target, budget, index, output):
    names = [f"class-{value}" for value in range(12)]
    summary = {
        "status": "completed",
        "variant": "group_magnitude",
        "dataset": dataset,
        "source": source,
        "target": target,
        "transfer": f"{source}->{target}",
        "formal_seed": 2026,
        "primary_metric": "sample_overall_accuracy" if dataset == "office31" else "fixed_12_class_macro_accuracy",
        "requested_budget": budget,
        "requested_group_count": budget_group_count(budget),
        "active_candidate_scalars": budget_group_count(budget) * 768,
        "PU-Acc": float(index),
        "FO-Acc": float(index + 1),
        "PU-overall-Acc": float(index + 2),
        "FO-overall-Acc": float(index + 3),
        "selection": {"mask_sha256": f"mask-{index}"},
        "output_dir": str(output),
    }
    if dataset == "visda-c":
        summary.update(
            {
                "PU-worst-class-Acc": 0.0,
                "FO-worst-class-Acc": 1.0,
                "PU-class-std": 2.0,
                "FO-class-std": 3.0,
                "PU-Acc-per-class-by-name": {name: float(i) for i, name in enumerate(names)},
                "FO-Acc-per-class-by-name": {name: float(i + 1) for i, name in enumerate(names)},
            }
        )
    return summary


def check_aggregate_outputs() -> None:
    with tempfile.TemporaryDirectory(prefix="group_magnitude_aggregate_") as value:
        root = Path(value)
        for budget in FORMAL_BUDGETS:
            for index, (dataset, source, target) in enumerate(TRANSFERS):
                output = root / "results" / budget_tag(budget) / dataset / f"{source}-{target}"
                output.mkdir(parents=True)
                (output / "summary.json").write_text(
                    json.dumps(_fake_summary(dataset, source, target, budget, index, output)),
                    encoding="utf-8",
                )
        aggregate = aggregate_matrix(root)
        assert aggregate["condition_count"] == 21
        assert aggregate["budgets"]["0.005"]["office31"]["PU-Acc"] == 2.5
        assert len(aggregate["budgets"]["0.020"]["visda-c"]["PU-Acc-per-class"]) == 12
        write_aggregate(root, aggregate)
        for name in ("aggregate.json", "results.csv", "visda_per_class.csv"):
            assert (root / name).is_file()
        with open(root / "results.csv", newline="", encoding="utf-8") as file_obj:
            assert len(list(csv.DictReader(file_obj))) == 21
        with open(root / "visda_per_class.csv", newline="", encoding="utf-8") as file_obj:
            assert len(list(csv.DictReader(file_obj))) == 36


def main() -> int:
    check_config_budgets_and_identity()
    check_resolver_identity_and_same_source()
    check_group_l2_ranking_ties_and_rng()
    check_strict_sparse_adamw()
    check_dataset_metrics()
    check_end_to_end_pu_fo()
    check_aggregate_outputs()
    print("Transformer Group-Magnitude tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
