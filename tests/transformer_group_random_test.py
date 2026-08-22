#!/usr/bin/env python3
"""CPU-only protocol checks for Transformer structural-group Random SHOT-OTTA."""

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

from transformer.candidate_dense.config import (  # noqa: E402
    candidate_parameter_names,
)
from transformer.candidate_dense.model import EXPECTED_SHAPES  # noqa: E402
from transformer.group_random.aggregate import (  # noqa: E402
    aggregate_matrix,
    write_aggregate,
)
from transformer.group_random.config import (  # noqa: E402
    BUDGET_TO_K,
    FORMAL_BUDGETS,
    FORMAL_SEED,
    GROUP_SIZE,
    MASK_SEEDS,
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
from transformer.group_random.groups import (  # noqa: E402
    STRUCTURAL_GROUPS,
    build_masks,
    mask_sha256,
    selected_group_ids,
)
from transformer.group_random.matrix import validate_devices  # noqa: E402
from transformer.group_random.optimizer import (  # noqa: E402
    assert_off_mask_adam_state_zero,
    strict_masked_adamw_step,
)
import transformer.group_random.runner as runner_module  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "transformer" / "group_random" / "config.yaml"


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
    assert MASK_SEEDS == (202600, 202601, 202602)
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
    target_list = list_root / "office31" / "dslr_list.txt"
    target_list.parent.mkdir(parents=True)
    target_list.write_text(
        "".join(f"/synthetic/{index}.jpg {index}\n" for index in range(31)),
        encoding="utf-8",
    )
    mapping = {f"class-{index}": index for index in range(31)}
    (target_list.parent / "class_to_idx.json").write_text(
        json.dumps(
            {
                "dataset": "office31",
                "class_to_idx": mapping,
                "domains": {"dslr": {"path": str(target_list.resolve())}},
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


def check_resolver_scientific_identity() -> None:
    with tempfile.TemporaryDirectory(prefix="group_random_resolver_") as value:
        root = Path(value)
        config = _write_fake_resolver_assets(root)
        low = resolve_transfer_config(
            config,
            project_root=PROJECT_ROOT,
            dataset="office31",
            source="amazon",
            target="dslr",
            budget=0.005,
            device="cpu",
            output_dir=root / "low",
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
        assert low["selection"]["requested_group_count"] == 34
        assert high["selection"]["requested_group_count"] == 138
        assert low["scientific_config_sha256"] != high["scientific_config_sha256"]
        assert low["checkpoint_sha256"] == high["checkpoint_sha256"]


def _candidate_tensors():
    values = []
    for name in candidate_parameter_names():
        suffix = name.split(".", maxsplit=2)[2]
        values.append((name, nn.Parameter(torch.zeros(EXPECTED_SHAPES[suffix]))))
    return values


def check_structural_groups_and_masks() -> None:
    assert len(STRUCTURAL_GROUPS) == TOTAL_GROUPS == 6912
    assert [group.group_id for group in STRUCTURAL_GROUPS] == list(range(TOTAL_GROUPS))
    assert sum(group.kind == "QK" for group in STRUCTURAL_GROUPS) == 3 * 384
    assert sum(group.kind == "VO" for group in STRUCTURAL_GROUPS) == 3 * 384
    assert sum(group.kind == "FFN" for group in STRUCTURAL_GROUPS) == 3 * 1536

    candidates = _candidate_tensors()
    focused = build_masks(candidates, [0, 384, 768])
    qkv = focused["blocks.9.attn.qkv.weight"]
    proj = focused["blocks.9.attn.proj.weight"]
    fc1 = focused["blocks.9.mlp.fc1.weight"]
    fc2 = focused["blocks.9.mlp.fc2.weight"]
    assert qkv[0, :].all() and qkv[384, :].all()
    assert qkv[768, :].all() and proj[:, 0].all()
    assert fc1[0, :].all() and fc2[:, 0].all()
    assert sum(int(mask.count_nonzero()) for mask in focused.values()) == 3 * GROUP_SIZE

    coverage = build_masks(candidates, range(TOTAL_GROUPS))
    assert all(mask.all().item() for mask in coverage.values())
    assert sum(int(mask.count_nonzero()) for mask in coverage.values()) == 5_308_416


def check_random_exact_k_reproduction_and_nesting() -> None:
    for seed in MASK_SEEDS:
        low = selected_group_ids(0.005, seed)
        medium = selected_group_ids(0.01, seed)
        high = selected_group_ids(0.02, seed)
        assert len(low) == 34 and len(set(low)) == 34
        assert len(medium) == 69 and len(set(medium)) == 69
        assert len(high) == 138 and len(set(high)) == 138
        assert medium[: len(low)] == low
        assert high[: len(medium)] == medium
        assert selected_group_ids(0.02, seed) == high
        assert mask_sha256(high) == mask_sha256(selected_group_ids(0.02, seed))
    supports = [set(selected_group_ids(0.02, seed)) for seed in MASK_SEEDS]
    assert len({tuple(sorted(value)) for value in supports}) == 3


def check_strict_sparse_adamw() -> None:
    first = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    second = nn.Parameter(torch.tensor([5.0, 6.0]))
    named = [("first", first), ("second", second)]
    masks = {
        "first": torch.tensor([[True, False], [False, True]]),
        "second": torch.tensor([False, False]),
    }
    optimizer = torch.optim.AdamW([first, second], lr=0.1, weight_decay=0.2)
    off_first = first.detach()[~masks["first"]].clone()
    off_second = second.detach().clone()
    selected_before = first.detach()[masks["first"]].clone()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        first.grad = torch.ones_like(first)
        second.grad = torch.ones_like(second)
        strict_masked_adamw_step(optimizer, named, masks)
        assert torch.equal(first.detach()[~masks["first"]], off_first)
        assert torch.equal(second.detach(), off_second)
        assert_off_mask_adam_state_zero(optimizer, named, masks)
    assert not torch.equal(first.detach()[masks["first"]], selected_before)
    for parameter, mask in ((first, masks["first"]), (second, masks["second"])):
        state = optimizer.state[parameter]
        assert torch.count_nonzero(state["exp_avg"][~mask]).item() == 0
        assert torch.count_nonzero(state["exp_avg_sq"][~mask]).item() == 0


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


class TinyRandomModel(nn.Module):
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
    model = TinyRandomModel().to(device)
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
        # Regression guard: a scope helper must never replace the structured
        # child selection artifact with this descriptive string.
        "selection": "synthetic-scope-selection",
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
    TinyRandomModel.forward_calls = 0
    config = {
        "output_dir": "",
        "runtime": {"device": "cpu", "deterministic": True, "amp": False, "pin_memory": False},
        "formal_seed": 2026,
        "protocol_revision": PROTOCOL_REVISION,
        "experiment_key": "synthetic-parent-mask-00",
        "scientific_config_sha256": "1" * 64,
        "parent_experiment_key": "synthetic-parent",
        "dataset": "office31",
        "source": "amazon",
        "target": "dslr",
        "transfer": "amazon->dslr",
        "num_classes": 2,
        "class_names": ["zero", "one"],
        "batch_size": 2,
        "fo_batch_size": 2,
        "random_mask_index": 0,
        "mask_seed": 202600,
        "adaptation": {"update_scope": "synthetic", "model_mode": "eval", "steps_per_online_batch": 1},
        "selection": {
            "requested_budget": 0.005,
            "requested_group_count": 1,
            "active_candidate_scalars": 2,
        },
        "optimization": {
            "optimizer": "adamw",
            "lr": 1.0e-2,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "weight_decay": 0.01,
        },
        "loss": {"components": ["ent", "div", "pseudo"], "cls_par": 0.3, "ent_par": 1.0, "threshold": 0.0},
        "preprocessing": {"interpolation": "bilinear"},
    }
    original_loader = runner_module.build_target_loaders
    original_selected = runner_module.selected_group_ids
    original_record = runner_module.mask_record
    original_masks = runner_module.build_masks
    runner_module.build_target_loaders = _tiny_loaders
    runner_module.selected_group_ids = lambda budget, seed: (0,)
    runner_module.mask_record = lambda ids, budget, seed: {
        "schema_version": 1,
        "requested_budget": budget,
        "requested_group_count": 1,
        "realized_group_count": 1,
        "active_candidate_scalars": 2,
        "mask_seed": seed,
        "mask_sha256": "synthetic-mask",
        "selected_group_ids": [0],
        "selected_groups": [],
        "selected_by_kind": {"QK": 1, "VO": 0, "FFN": 0},
        "selected_by_block": {"9": 1, "10": 0, "11": 0},
    }
    runner_module.build_masks = lambda candidates, ids: {
        "candidate.weight": torch.tensor(
            [[True, False, False], [False, True, False], [False, False, False]]
        )
    }
    try:
        with tempfile.TemporaryDirectory(prefix="group_random_runner_") as value:
            config["output_dir"] = str(Path(value) / "mask_00")
            summary = runner_module.run_mask_child(
                config,
                PROJECT_ROOT,
                show_progress=False,
                model_loader=_tiny_model_loader,
            )
            assert summary["status"] == "completed"
            assert summary["adaptation_steps"] == 2
            assert summary["backward_calls"] == 2
            assert summary["off_mask_state_unchanged"] is True
            assert summary["off_mask_adam_state_zero"] is True
            assert isinstance(summary["selection"], dict)
            assert summary["selection"]["mask_sha256"] == "synthetic-mask"
            assert summary["frozen_head_unchanged"] is True
            assert summary["PU-sample-count"] == 3
            assert summary["FO-sample-count"] == 3
            assert summary["pu_is_post_update_same_batch"] is True
            assert summary["fo_is_independent_full_target_pass"] is True
            assert TinyRandomModel.forward_calls == 6
            rows = [
                json.loads(line)
                for line in (Path(config["output_dir"]) / "metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            online = [row for row in rows if row["event"] == "online_batch"]
            assert [row["batch_size"] for row in online] == [2, 1]
    finally:
        runner_module.build_target_loaders = original_loader
        runner_module.selected_group_ids = original_selected
        runner_module.mask_record = original_record
        runner_module.build_masks = original_masks


def _fake_parent_summary(dataset, source, target, budget, index, output):
    names = [f"class-{value}" for value in range(12)]
    summary = {
        "status": "completed",
        "variant": "group_random",
        "dataset": dataset,
        "source": source,
        "target": target,
        "transfer": f"{source}->{target}",
        "formal_seed": 2026,
        "primary_metric": "sample_overall_accuracy" if dataset == "office31" else "fixed_12_class_macro_accuracy",
        "requested_budget": budget,
        "requested_group_count": budget_group_count(budget),
        "active_candidate_scalars": budget_group_count(budget) * 768,
        "num_random_masks": 3,
        "PU-Acc": float(index),
        "PU-Acc-mask-std": 0.5,
        "FO-Acc": float(index + 1),
        "FO-Acc-mask-std": 0.6,
        "PU-overall-Acc": float(index + 2),
        "FO-overall-Acc": float(index + 3),
        "output_dir": str(output),
        "masks": [
            {
                "random_mask_index": mask,
                "mask_seed": MASK_SEEDS[mask],
                "mask_sha256": f"mask-{mask}",
                "PU-Acc": float(index + mask),
                "FO-Acc": float(index + mask + 1),
                "summary_path": str(output / f"mask_{mask:02d}" / "summary.json"),
            }
            for mask in range(3)
        ],
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
                "PU-Acc-per-class-mask-std-by-name": {name: 0.1 for name in names},
                "FO-Acc-per-class-mask-std-by-name": {name: 0.2 for name in names},
            }
        )
    return summary


def check_aggregate_outputs() -> None:
    with tempfile.TemporaryDirectory(prefix="group_random_aggregate_") as value:
        root = Path(value)
        for budget in FORMAL_BUDGETS:
            for index, (dataset, source, target) in enumerate(TRANSFERS):
                output = root / "results" / budget_tag(budget) / dataset / f"{source}-{target}"
                output.mkdir(parents=True)
                summary = _fake_parent_summary(dataset, source, target, budget, index, output)
                (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        aggregate = aggregate_matrix(root)
        assert aggregate["condition_count"] == 21
        assert aggregate["child_run_count"] == 63
        assert aggregate["budgets"]["0.005"]["office31"]["PU-Acc"] == 2.5
        assert len(aggregate["budgets"]["0.020"]["visda-c"]["PU-Acc-per-class"]) == 12
        write_aggregate(root, aggregate)
        for name in ("aggregate.json", "results.csv", "mask_results.csv", "visda_per_class.csv"):
            assert (root / name).is_file()
        with open(root / "results.csv", newline="", encoding="utf-8") as file_obj:
            assert len(list(csv.DictReader(file_obj))) == 21
        with open(root / "mask_results.csv", newline="", encoding="utf-8") as file_obj:
            assert len(list(csv.DictReader(file_obj))) == 63
        with open(root / "visda_per_class.csv", newline="", encoding="utf-8") as file_obj:
            assert len(list(csv.DictReader(file_obj))) == 36


def main() -> int:
    check_config_budgets_and_identity()
    check_resolver_scientific_identity()
    check_structural_groups_and_masks()
    check_random_exact_k_reproduction_and_nesting()
    check_strict_sparse_adamw()
    check_end_to_end_pu_fo()
    check_aggregate_outputs()
    print("Transformer Group-Random tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
