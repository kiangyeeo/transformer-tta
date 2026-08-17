#!/usr/bin/env python3
"""CPU contracts for the DeiT OTTA random structural-group baseline.

Covers group definition (paired Q-K / V-O / FFN), budget ceiling, random
selection determinism, config/identity resolution, and a full end-to-end
multi-mask run with a synthetic candidate DeiT.  No network or real W0.
"""

import copy
import json
import math
import os.path as osp
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image


PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
TEST_ROOT = osp.dirname(osp.abspath(__file__))
for path in (PROJECT_ROOT, TEST_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from deit_source_only_fixtures import (  # noqa: E402
    SOURCE_CHECKPOINT_KIND,
    SOURCE_CHECKPOINT_SCHEMA_VERSION,
    checkpoint_metadata,
    prepare_assets,
    sha256_file,
    write_json,
)
from shot_otta.adaptation.deit_groups import (  # noqa: E402
    GROUPS_PER_BLOCK,
    SCALARS_PER_GROUP,
    TOTAL_GROUP_COUNT,
    active_group_count,
    build_group_mask_dict,
    candidate_group_order,
    count_active_scalars,
    select_random_groups,
    total_group_count,
)
from shot_otta.otta.random_group import (  # noqa: E402
    run_deit_otta_random_group_experiment,
)
from shot_otta.otta.random_group_config import resolve_config  # noqa: E402


CONFIG_PATH = osp.join(PROJECT_ROOT, "configs", "deit_otta_group_random.yaml")
CANDIDATE_SCALARS = 5_308_416  # 6912 groups x 768 scalars


class RandomGroupTinyDeiT(nn.Module):
    """Tiny DeiT whose 12 blocks carry real-size candidate weight tensors.

    ``forward`` reads every candidate weight element of blocks 9, 10, 11 so
    the SHOT loss produces nonzero gradients across the whole candidate
    scope.  Blocks 0-8 exist only so the update scope can freeze them.
    """

    total_forward_calls = 0

    def __init__(self, num_classes, **kwargs):
        super().__init__()
        del kwargs
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 384))
        self.pos_embed = nn.Parameter(torch.zeros(1, 2, 384))
        self.head = nn.Linear(384, num_classes)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": nn.ModuleDict(
                            {
                                "qkv": nn.Linear(384, 1152),
                                "proj": nn.Linear(384, 384),
                            }
                        ),
                        "mlp": nn.ModuleDict(
                            {
                                "fc1": nn.Linear(384, 1536),
                                "fc2": nn.Linear(1536, 384),
                            }
                        ),
                    }
                )
                for _ in range(12)
            ]
        )

    def get_classifier(self):
        return self.head

    def forward(self, images):
        type(self).total_forward_calls += 1
        batch = images.size(0)
        device = images.device
        dtype = images.dtype
        base = images.mean(dim=(1, 2, 3))
        extra = torch.zeros(batch, device=device, dtype=dtype)
        for index in (9, 10, 11):
            block = self.blocks[index]
            extra = extra + base * block["attn"]["qkv"].weight.sum() / 1152.0
            extra = extra + base * block["attn"]["proj"].weight.sum() / 384.0
            extra = extra + base * block["mlp"]["fc1"].weight.sum() / 1536.0
            extra = extra + base * block["mlp"]["fc2"].weight.sum() / 384.0
        features = torch.zeros(batch, 384, device=device, dtype=dtype)
        features[:, 0] = base + 0.01 * extra
        return self.head(features)


def tiny_factory(model_name, **kwargs):
    assert model_name == "deit_small_patch16_224.fb_in1k"
    return RandomGroupTinyDeiT(**kwargs)


def install_fake_checkpoint(root):
    path = Path(root) / "checkpoints" / "office31" / "amazon.pth"
    model = RandomGroupTinyDeiT(num_classes=31)
    metadata = checkpoint_metadata("office31", "amazon", 31)
    torch.save(
        {
            "schema_version": SOURCE_CHECKPOINT_SCHEMA_VERSION,
            "kind": SOURCE_CHECKPOINT_KIND,
            "state_dict": model.state_dict(),
            "metadata": metadata,
        },
        path,
    )
    write_json(
        path.with_suffix(".manifest.json"),
        {
            **metadata,
            "checkpoint": {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "schema_version": SOURCE_CHECKPOINT_SCHEMA_VERSION,
                "kind": SOURCE_CHECKPOINT_KIND,
            },
        },
    )
    image_root = Path(root) / "images" / "office31" / "dslr"
    image_root.mkdir(parents=True, exist_ok=True)
    for label in range(31):
        Image.new("RGB", (8, 8), color=(label, label, label)).save(
            image_root / f"{label}.png"
        )


def _base(root):
    config = prepare_assets(
        root, CONFIG_PATH, runs_directory="group_random_runs"
    )
    # Use a tiny budget so the ceiling is small and deterministic.
    config["adaptation"]["budget"] = 0.005
    return config


def _check_group_definition():
    order = candidate_group_order()
    assert len(order) == TOTAL_GROUP_COUNT == 6912
    assert GROUPS_PER_BLOCK == 2304
    assert SCALARS_PER_GROUP == 768
    assert order[0] == (9, "qk", 0)
    assert order[384] == (9, "vo", 0)
    assert order[768] == (9, "ffn", 0)
    assert order[2304] == (10, "qk", 0)
    assert order[-1] == (11, "ffn", 1535)

    # Full mask covers every candidate scalar exactly once.
    full = build_group_mask_dict(list(range(TOTAL_GROUP_COUNT)))
    assert count_active_scalars(full) == CANDIDATE_SCALARS
    for mask in full.values():
        assert bool(mask.all().item())

    # A single Q-K group masks exactly two rows of the fused qkv.
    qk = build_group_mask_dict([0])
    qkv = qk["blocks.9.attn.qkv.weight"]
    assert qkv[0, :].all()
    assert qkv[384, :].all()
    assert int(qkv.sum()) == 768
    assert count_active_scalars(qk) == 768

    # A single V-O group masks the V row and the proj column.
    vo = build_group_mask_dict([384])
    qkv = vo["blocks.9.attn.qkv.weight"]
    proj = vo["blocks.9.attn.proj.weight"]
    assert qkv[768, :].all()
    assert int(qkv.sum()) == 384
    assert proj[:, 0].all()
    assert int(proj.sum()) == 384
    assert count_active_scalars(vo) == 768

    # A single FFN group masks the fc1 row and the fc2 column.
    ffn = build_group_mask_dict([768])
    fc1 = ffn["blocks.9.mlp.fc1.weight"]
    fc2 = ffn["blocks.9.mlp.fc2.weight"]
    assert fc1[0, :].all()
    assert int(fc1.sum()) == 384
    assert fc2[:, 0].all()
    assert int(fc2.sum()) == 384
    assert count_active_scalars(ffn) == 768

    # Two distinct groups never overlap on a scalar.
    two = build_group_mask_dict([0, 1])
    assert count_active_scalars(two) == 2 * SCALARS_PER_GROUP
    three_kinds = build_group_mask_dict([0, 384, 768])
    assert count_active_scalars(three_kinds) == 3 * SCALARS_PER_GROUP


def _check_budget():
    assert active_group_count(0.005) == 35
    assert active_group_count(0.001) == 7
    assert active_group_count(0.003) == 21
    assert active_group_count(0.01) == 70
    assert active_group_count(0.02) == 139
    for bad in (0.0, -0.1, 1.5, "x"):
        try:
            active_group_count(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"budget {bad!r} should be rejected")


def _check_random_selection():
    total = TOTAL_GROUP_COUNT
    first = select_random_groups(total, 35, 202000)
    second = select_random_groups(total, 35, 202000)
    third = select_random_groups(total, 35, 202001)
    assert first == second
    assert first != third
    assert len(first) == 35
    assert len(set(first)) == 35
    assert all(0 <= index < total for index in first)
    assert select_random_groups(total, 0, 202000) == []
    assert select_random_groups(total, total, 202000) == list(range(total))


def _check_config_contract(base):
    effective = resolve_config(base, PROJECT_ROOT)
    assert effective["method"] == "shot"
    assert effective["variant"] == "group_random"
    assert effective["task"] == "otta"
    assert effective["adaptation"]["update_scope"] == "last_3_block_weights"
    assert effective["adaptation"]["candidate_blocks"] == [9, 10, 11]
    assert effective["adaptation"]["grouping"] == "paired_qk_vo_ffn"
    assert effective["adaptation"]["delta_semantics"] == "strict_masked_delta"
    assert effective["adaptation"]["budget"] == 0.005
    assert effective["adaptation"]["num_random_masks"] == 3
    assert effective["adaptation"]["mask_seeds"] == [202000, 202001, 202002]
    assert effective["adaptation"]["total_groups"] == 6912
    assert effective["adaptation"]["active_groups"] == 35
    scientific = effective["scientific_config"]
    assert scientific["protocol"] == (
        "sequential_target_stream_group_random_adaptation"
    )
    assert scientific["adaptation"]["grouping"] == "paired_qk_vo_ffn"
    assert scientific["adaptation"]["delta_semantics"] == "strict_masked_delta"
    assert scientific["adaptation"]["active_groups"] == 35
    assert "group_random" in effective["experiment_key"]
    return effective


def _check_identity_binds(base):
    first = resolve_config(base, PROJECT_ROOT)
    changed = copy.deepcopy(base)
    changed["adaptation"]["budget"] = 0.01
    second = resolve_config(changed, PROJECT_ROOT)
    assert first["experiment_key"] != second["experiment_key"]
    changed = copy.deepcopy(base)
    changed["adaptation"]["mask_seeds"] = [202000, 202001, 999999]
    third = resolve_config(changed, PROJECT_ROOT)
    assert first["experiment_key"] != third["experiment_key"]


def _expect_value_error(config, message):
    try:
        resolve_config(config, PROJECT_ROOT)
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"Expected ValueError containing: {message}")


def _check_validation(base):
    bad = copy.deepcopy(base)
    bad["method"] = "no_tta"
    _expect_value_error(bad, "method must be shot")
    bad = copy.deepcopy(base)
    bad["variant"] = "source_only"
    _expect_value_error(bad, "variant must be one of")
    bad = copy.deepcopy(base)
    bad["adaptation"]["update_scope"] = "all_except_head"
    _expect_value_error(bad, "update_scope")
    bad = copy.deepcopy(base)
    bad["adaptation"]["grouping"] = "independent"
    _expect_value_error(bad, "grouping")
    bad = copy.deepcopy(base)
    bad["adaptation"]["delta_semantics"] = "unrestricted_accumulation"
    _expect_value_error(bad, "delta_semantics")
    bad = copy.deepcopy(base)
    bad["adaptation"]["budget"] = 0.0
    _expect_value_error(bad, "budget")
    bad = copy.deepcopy(base)
    bad["adaptation"]["candidate_blocks"] = [9, 10, 13]
    _expect_value_error(bad, "in [0, 12)")
    bad = copy.deepcopy(base)
    bad["adaptation"]["mask_seeds"] = [202000, 202000, 202002]
    _expect_value_error(bad, "unique")
    bad = copy.deepcopy(base)
    bad["adaptation"]["mask_seeds"] = [202000, 202001]
    _expect_value_error(bad, "length must equal")


def _check_end_to_end(root, base):
    install_fake_checkpoint(root)
    effective = resolve_config(base, PROJECT_ROOT)
    RandomGroupTinyDeiT.total_forward_calls = 0
    summary = run_deit_otta_random_group_experiment(
        effective, PROJECT_ROOT, model_factory=tiny_factory
    )
    assert summary["status"] == "completed"
    assert summary["variant"] == "group_random"
    assert summary["num_random_masks"] == 3
    assert summary["mask_seeds"] == [202000, 202001, 202002]
    assert summary["candidate_groups"] == 6912
    assert summary["active_groups"] == 35
    assert summary["candidate_scalars"] == CANDIDATE_SCALARS
    assert len(summary["masks"]) == 3
    for mask in summary["masks"]:
        assert mask["active_groups"] == 35
        assert mask["active_scalars"] == 35 * SCALARS_PER_GROUP
        assert 0.0 <= mask["PU-Acc"] <= 100.0
        assert 0.0 <= mask["FO-Acc"] <= 100.0
    assert math.isfinite(summary["mean_PU-Acc"])
    assert math.isfinite(summary["std_PU-Acc"])
    assert math.isfinite(summary["mean_FO-Acc"])
    assert math.isfinite(summary["std_FO-Acc"])
    assert abs(summary["mean_PU-Acc"] - summary["PU-Acc"]) < 1.0e-12
    assert abs(summary["mean_FO-Acc"] - summary["FO-Acc"]) < 1.0e-12
    # Per mask: 2 batches x 2 forwards (loss + PU) + 2 FO forwards = 6.
    # Three masks -> 18 forwards total.
    assert RandomGroupTinyDeiT.total_forward_calls == 18

    output = Path(summary["output_dir"])
    assert (output / "manifest.json").is_file()
    assert (output / "summary.json").is_file()
    assert (output / "random_summary.md").is_file()
    for index in range(3):
        mask_dir = output / f"mask_{index:02d}"
        assert (mask_dir / "summary.json").is_file()
        assert (mask_dir / "mask.pt").is_file()
        assert (mask_dir / "metrics.jsonl").is_file()
        child = json.loads(
            (mask_dir / "summary.json").read_text(encoding="utf-8")
        )
        assert child["status"] == "completed"
        assert child["active_groups"] == 35
        assert child["delta_nonzero_scalars"] == 35 * SCALARS_PER_GROUP
        metrics = [
            json.loads(line)
            for line in (mask_dir / "metrics.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        batch_rows = [row for row in metrics if row["event"] == "online_batch"]
        assert len(batch_rows) == 2
        assert all(row["mask_static"] is True for row in batch_rows)
        assert all(row["active_groups"] == 35 for row in batch_rows)
        final_rows = [row for row in metrics if row["event"] == "final"]
        assert len(final_rows) == 1
        assert final_rows[0]["status"] == "completed"
    return summary


def main():
    with tempfile.TemporaryDirectory(prefix="deit_otta_group_random_") as root:
        base = _base(root)
        _check_group_definition()
        _check_budget()
        _check_random_selection()
        _check_config_contract(base)
        _check_identity_binds(base)
        _check_validation(base)
        _check_end_to_end(root, base)
    print("DeiT OTTA random structural-group baseline tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
