#!/usr/bin/env python3
"""CPU/synthetic protocol checks for Transformer Group-Saliency SHOT-OTTA."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformer.candidate_dense.config import candidate_parameter_names  # noqa: E402
from transformer.candidate_dense.model import EXPECTED_SHAPES  # noqa: E402
from transformer.group_saliency.aggregate import aggregate_matrix, write_aggregate  # noqa: E402
from transformer.group_saliency.config import (  # noqa: E402
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
from transformer.group_saliency.groups import (  # noqa: E402
    STRUCTURAL_GROUPS,
    build_masks,
    compute_group_saliency_scores,
    dynamic_mask_record,
    mask_history_sha256,
    select_saliency_group_ids,
)
from transformer.group_saliency.matrix import validate_devices  # noqa: E402
from transformer.group_saliency.optimizer import (  # noqa: E402
    assert_off_mask_adam_state_zero,
    strict_masked_adamw_step,
)
from transformer.source_only.metrics import FixedClassMeter  # noqa: E402
import transformer.group_saliency.runner as runner_module  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "transformer" / "group_saliency" / "config.yaml"


def check_config_budgets_and_identity() -> None:
    config = load_config(CONFIG_PATH)
    _validate_frozen_fields(config)
    assert config["formal_seed"] == FORMAL_SEED == 2026
    assert config["protocol_revision"] == PROTOCOL_REVISION
    assert FORMAL_BUDGETS == (0.005, 0.01, 0.02)
    assert BUDGET_TO_K == {0.005: 34, 0.01: 69, 0.02: 138}
    assert parse_budgets("all") == FORMAL_BUDGETS
    assert parse_budgets("0.005,0.02") == (0.005, 0.02)
    assert parse_budgets("0.0005") == (0.0005,)
    assert budget_group_count(0.0005) == 3
    assert budget_tag(0.0005) == "rho-0.0005"
    assert budget_tag(0.01) == "rho-0.010"
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
    except ValueError:
        pass
    else:
        raise AssertionError("Ceil budgets were accepted")


def _write_resolver_assets(root: Path) -> dict:
    config = load_config(CONFIG_PATH)
    list_root = root / "lists"
    checkpoint_root = root / "checkpoints"
    datasets = {
        "office31": {
            "classes": 31,
            "domains": ("amazon", "dslr", "webcam"),
            "sources": ("amazon", "dslr", "webcam"),
        },
        "visda-c": {
            "classes": 12,
            "domains": ("train", "validation"),
            "sources": ("train",),
        },
    }
    visda_names = (
        "aeroplane", "bicycle", "bus", "car", "horse", "knife",
        "motorcycle", "person", "plant", "skateboard", "train", "truck",
    )
    for dataset, record in datasets.items():
        dataset_root = list_root / dataset
        dataset_root.mkdir(parents=True, exist_ok=True)
        names = (
            visda_names
            if dataset == "visda-c"
            else tuple(f"class-{index}" for index in range(record["classes"]))
        )
        domains = {}
        for domain in record["domains"]:
            target_list = dataset_root / f"{domain}_list.txt"
            target_list.write_text(
                "".join(
                    f"/synthetic/{dataset}/{domain}/{index}.jpg {index}\n"
                    for index in range(record["classes"])
                ),
                encoding="utf-8",
            )
            domains[domain] = {"path": str(target_list.resolve())}
        (dataset_root / "class_to_idx.json").write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "class_to_idx": {name: index for index, name in enumerate(names)},
                    "domains": domains,
                }
            ),
            encoding="utf-8",
        )
        for source in record["sources"]:
            checkpoint = checkpoint_root / dataset / f"{source}.pth"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f"{dataset}-{source}".encode("ascii"))
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            checkpoint.with_suffix(".manifest.json").write_text(
                json.dumps(
                    {"checkpoint": {"path": str(checkpoint.resolve()), "sha256": digest}}
                ),
                encoding="utf-8",
            )
    config["data"]["list_root"] = str(list_root)
    config["model"]["checkpoint_root"] = str(checkpoint_root)
    return config


def check_four_sources_and_21_identities() -> None:
    with tempfile.TemporaryDirectory(prefix="group_saliency_resolver_") as value:
        root = Path(value)
        config = _write_resolver_assets(root)
        identities = set()
        checkpoints = set()
        for budget in FORMAL_BUDGETS:
            for dataset, source, target in TRANSFERS:
                resolved = resolve_transfer_config(
                    config,
                    project_root=PROJECT_ROOT,
                    dataset=dataset,
                    source=source,
                    target=target,
                    budget=budget,
                    device="cpu",
                    output_dir=root / budget_tag(budget) / dataset / f"{source}-{target}",
                )
                identities.add(resolved["experiment_key"])
                checkpoints.add((dataset, source, resolved["checkpoint_sha256"]))
        assert len(identities) == 21
        assert {(dataset, source) for dataset, source, _ in checkpoints} == {
            ("office31", "amazon"),
            ("office31", "dslr"),
            ("office31", "webcam"),
            ("visda-c", "train"),
        }


def _candidate_tensors():
    values = []
    for name in candidate_parameter_names():
        suffix = name.split(".", maxsplit=2)[2]
        parameter = nn.Parameter(torch.zeros(EXPECTED_SHAPES[suffix]))
        parameter.grad = torch.zeros_like(parameter)
        values.append((name, parameter))
    return values


def check_groups_scores_dynamic_topk_and_coverage() -> None:
    assert len(STRUCTURAL_GROUPS) == TOTAL_GROUPS == 6912
    candidates = _candidate_tensors()
    lookup = dict(candidates)
    qkv = lookup["blocks.9.attn.qkv.weight"]
    with torch.no_grad():
        qkv[0, 0] = 2.0
        qkv[1, 0] = 10.0
    qkv.grad[0, 0] = 3.0
    qkv.grad[1, 0] = 0.5
    scores = compute_group_saliency_scores(candidates)
    assert scores.shape == (TOTAL_GROUPS,)
    assert scores[0].item() == 6.0
    assert scores[1].item() == 5.0
    selected = select_saliency_group_ids(scores, 0.005)
    assert selected[:2] == (0, 1)
    assert len(selected) == 34
    record = dynamic_mask_record(selected, scores=scores, budget=0.005)
    assert record["realized_group_count"] == 34
    assert record["active_candidate_scalars"] == 34 * GROUP_SIZE
    assert len(mask_history_sha256([record["mask_sha256"]])) == 64

    first = selected
    qkv.grad.zero_()
    qkv.grad[1, 0] = 1.0
    second_scores = compute_group_saliency_scores(candidates)
    second = select_saliency_group_ids(second_scores, 0.005)
    assert first != second
    assert second[0] == 1

    focused = build_masks(candidates, [0, 384, 768])
    qkv_mask = focused["blocks.9.attn.qkv.weight"]
    assert qkv_mask[0, :].all() and qkv_mask[384, :].all()
    assert qkv_mask[768, :].all()
    assert focused["blocks.9.attn.proj.weight"][:, 0].all()
    assert focused["blocks.9.mlp.fc1.weight"][0, :].all()
    assert focused["blocks.9.mlp.fc2.weight"][:, 0].all()
    assert sum(int(mask.count_nonzero()) for mask in focused.values()) == 3 * GROUP_SIZE
    coverage = build_masks(candidates, range(TOTAL_GROUPS))
    assert all(mask.all().item() for mask in coverage.values())
    assert sum(int(mask.count_nonzero()) for mask in coverage.values()) == 5_308_416


def check_dynamic_strict_sparse_adamw() -> None:
    parameter = nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    named = [("parameter", parameter)]
    optimizer = torch.optim.AdamW([parameter], lr=0.1, weight_decay=0.2)
    first_mask = torch.tensor([True, True, False, False])
    second_mask = torch.tensor([False, True, True, False])

    parameter.grad = torch.ones_like(parameter)
    strict_masked_adamw_step(optimizer, named, {"parameter": first_mask})
    after_first = parameter.detach().clone()
    first_moment = optimizer.state[parameter]["exp_avg"].detach().clone()
    assert first_moment[1].item() != 0.0

    optimizer.zero_grad(set_to_none=True)
    parameter.grad = torch.ones_like(parameter)
    before_second = parameter.detach().clone()
    strict_masked_adamw_step(optimizer, named, {"parameter": second_mask})
    assert torch.equal(parameter.detach()[~second_mask], before_second[~second_mask])
    assert parameter.detach()[0].item() == after_first[0].item()
    assert parameter.detach()[1].item() != after_first[1].item()
    assert parameter.detach()[2].item() != after_first[2].item()
    assert optimizer.state[parameter]["exp_avg"][1].item() != 0.0
    assert_off_mask_adam_state_zero(
        optimizer, named, {"parameter": second_mask}
    )


def check_dataset_metrics() -> None:
    office = FixedClassMeter(2, ["many", "few"])
    labels = torch.tensor([0] * 10 + [1])
    predictions = torch.tensor([0] * 9 + [1, 0])
    office.update(labels[:4], predictions[:4])
    office.update(labels[4:], predictions[4:])
    assert abs(office.compute("office31")["Acc"] - 100.0 * 9 / 11) < 1.0e-12

    names = [f"class-{index}" for index in range(12)]
    visda = FixedClassMeter(12, names)
    labels = torch.tensor([0] * 10 + list(range(1, 12)))
    predictions = labels.clone()
    predictions[-1] = 0
    visda.update(labels, predictions)
    metrics = visda.compute("visda-c")
    assert len(metrics["Acc-per-class"]) == 12
    assert abs(metrics["Acc"] - 100.0 * 11 / 12) < 1.0e-12
    assert metrics["Acc"] != metrics["overall-Acc"]


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


class TinySaliencyModel(nn.Module):
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
    model = TinySaliencyModel().to(device)
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
    TinySaliencyModel.forward_calls = 0
    config = {
        "output_dir": "",
        "runtime": {"device": "cpu", "deterministic": True, "amp": False, "pin_memory": False},
        "formal_seed": 2026,
        "protocol_revision": PROTOCOL_REVISION,
        "experiment_key": "synthetic-saliency",
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
        "selection": {
            "requested_budget": 0.005,
            "requested_group_count": 1,
            "active_candidate_scalars_per_step": 2,
            "type": "synthetic-dynamic-saliency",
        },
        "optimization": {"optimizer": "adamw", "lr": 1.0e-2, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.01},
        "loss": {"components": ["ent", "div", "pseudo"], "cls_par": 0.3, "ent_par": 1.0, "threshold": 0.0},
        "preprocessing": {"interpolation": "bilinear"},
    }
    originals = {
        "loaders": runner_module.build_target_loaders,
        "scores": runner_module.compute_group_saliency_scores,
        "select": runner_module.select_saliency_group_ids,
        "record": runner_module.dynamic_mask_record,
        "masks": runner_module.build_masks,
    }
    selections = iter(((0,), (1,)))
    runner_module.build_target_loaders = _tiny_loaders
    runner_module.compute_group_saliency_scores = lambda candidates: torch.tensor([2.0, 1.0])
    runner_module.select_saliency_group_ids = lambda scores, budget: next(selections)
    runner_module.dynamic_mask_record = lambda ids, scores, budget: {
        "requested_group_count": 1,
        "realized_group_count": 1,
        "active_candidate_scalars": 2,
        "mask_sha256": f"synthetic-mask-{ids[0]}",
        "top_k_min_score": 1.0,
        "first_excluded_score": 0.0,
        "selected_group_ids": list(ids),
        "selected_group_scores": [1.0],
        "selected_by_kind": {"QK": 1, "VO": 0, "FFN": 0},
        "selected_by_block": {"9": 1, "10": 0, "11": 0},
    }
    runner_module.build_masks = lambda candidates, ids: {
        "candidate.weight": (
            torch.tensor([[True, False, False], [False, True, False], [False, False, False]])
            if ids == (0,)
            else torch.tensor([[False, False, False], [False, True, False], [False, False, True]])
        )
    }
    try:
        with tempfile.TemporaryDirectory(prefix="group_saliency_runner_") as value:
            config["output_dir"] = str(Path(value) / "condition")
            summary = runner_module.run_transfer(
                config, PROJECT_ROOT, show_progress=False, model_loader=_tiny_model_loader
            )
            assert summary["status"] == "completed"
            assert summary["adaptation_steps"] == 2
            assert summary["backward_calls"] == 2
            assert summary["frozen_head_unchanged"] is True
            assert summary["strict_dynamic_off_mask_value_freezing"] is True
            assert summary["PU-sample-count"] == 3
            assert summary["FO-sample-count"] == 3
            assert summary["pu_is_post_update_same_batch"] is True
            assert summary["fo_is_independent_full_target_pass"] is True
            assert summary["selection"]["mask_change_count"] == 1
            assert TinySaliencyModel.forward_calls == 6
            rows = [
                json.loads(line)
                for line in (Path(config["output_dir"]) / "metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            online = [row for row in rows if row["event"] == "online_batch"]
            assert [row["batch_size"] for row in online] == [2, 1]
            assert [row["selected_group_ids"] for row in online] == [[0], [1]]
    finally:
        runner_module.build_target_loaders = originals["loaders"]
        runner_module.compute_group_saliency_scores = originals["scores"]
        runner_module.select_saliency_group_ids = originals["select"]
        runner_module.dynamic_mask_record = originals["record"]
        runner_module.build_masks = originals["masks"]


def _fake_summary(dataset, source, target, budget, index, output):
    names = [f"class-{value}" for value in range(12)]
    summary = {
        "status": "completed",
        "variant": "group_saliency",
        "dataset": dataset,
        "source": source,
        "target": target,
        "transfer": f"{source}->{target}",
        "formal_seed": 2026,
        "primary_metric": "sample_overall_accuracy" if dataset == "office31" else "fixed_12_class_macro_accuracy",
        "requested_budget": budget,
        "requested_group_count": budget_group_count(budget),
        "active_candidate_scalars_per_step": budget_group_count(budget) * GROUP_SIZE,
        "PU-Acc": float(index),
        "FO-Acc": float(index + 1),
        "PU-overall-Acc": float(index + 2),
        "FO-overall-Acc": float(index + 3),
        "selection": {
            "unique_mask_count": 2,
            "historical_active_group_union_count": 3,
            "mask_history_sha256": f"history-{index}",
        },
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
    with tempfile.TemporaryDirectory(prefix="group_saliency_aggregate_") as value:
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
    check_four_sources_and_21_identities()
    check_groups_scores_dynamic_topk_and_coverage()
    check_dynamic_strict_sparse_adamw()
    check_dataset_metrics()
    check_end_to_end_pu_fo()
    check_aggregate_outputs()
    print("Transformer Group-Saliency tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
