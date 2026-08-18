#!/usr/bin/env python3
"""CPU contracts for the DeiT OTTA saliency structural-group baseline.

Covers the per-group gradient L2-norm scoring, deterministic Top-K selection,
budget ceiling, config/identity resolution, and a full end-to-end run with a
synthetic candidate DeiT.  No network or real W0.
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
    SCALARS_PER_GROUP,
    TOTAL_GROUP_COUNT,
    active_group_count,
    group_gradient_saliency_scores,
    select_top_groups,
)
from shot_otta.otta.saliency import (  # noqa: E402
    run_deit_otta_saliency_experiment,
)
from shot_otta.otta.saliency_config import resolve_config  # noqa: E402


CONFIG_PATH = osp.join(
    PROJECT_ROOT, "configs", "deit_otta_group_saliency.yaml"
)
CANDIDATE_SCALARS = 5_308_416  # 6912 groups x 768 scalars


class SaliencyTinyDeiT(nn.Module):
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
    return SaliencyTinyDeiT(**kwargs)


def install_fake_checkpoint(root):
    path = Path(root) / "checkpoints" / "office31" / "amazon.pth"
    model = SaliencyTinyDeiT(num_classes=31)
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
        root, CONFIG_PATH, runs_directory="saliency_runs"
    )
    config["adaptation"]["budget"] = 0.005
    return config


def _grads_dict():
    grads = {}
    for block in (9, 10, 11):
        grads[f"blocks.{block}.attn.qkv.weight"] = torch.zeros(1152, 384)
        grads[f"blocks.{block}.attn.proj.weight"] = torch.zeros(384, 384)
        grads[f"blocks.{block}.mlp.fc1.weight"] = torch.zeros(1536, 384)
        grads[f"blocks.{block}.mlp.fc2.weight"] = torch.zeros(384, 1536)
    return grads


def _check_saliency_selection():
    grads = _grads_dict()
    # Q-K group 0: |grad| at Q row 0 -> score sqrt(3^2) = 3.
    grads["blocks.9.attn.qkv.weight"][0, 0] = 3.0
    # Q-K group 5: |grad| at K row 5 -> score 4.
    grads["blocks.9.attn.qkv.weight"][384 + 5, 1] = 4.0
    # FFN group 10: global index 768 + 10 = 778 -> score 12.
    grads["blocks.9.mlp.fc1.weight"][10, 2] = 12.0
    # V-O group 0: global index 384 -> |grad| at proj col 0 -> score 5.
    grads["blocks.9.attn.proj.weight"][0, 0] = 5.0

    scores = group_gradient_saliency_scores(grads)
    assert scores.numel() == TOTAL_GROUP_COUNT == 6912
    assert abs(float(scores[0]) - 3.0) < 1.0e-9
    assert abs(float(scores[5]) - 4.0) < 1.0e-9
    assert abs(float(scores[384]) - 5.0) < 1.0e-9
    assert abs(float(scores[778]) - 12.0) < 1.0e-9
    assert float(scores.sum()) == 24.0

    selected = select_top_groups(scores, 4)
    assert selected == [778, 384, 5, 0]
    assert select_top_groups(scores, 4) == selected
    assert len(select_top_groups(scores, 35)) == 35
    assert len(set(select_top_groups(scores, 35))) == 35
    assert select_top_groups(scores, 0) == []
    assert select_top_groups(scores, TOTAL_GROUP_COUNT) == list(
        range(TOTAL_GROUP_COUNT)
    )


def _check_budget():
    assert active_group_count(0.005) == 35
    assert active_group_count(0.001) == 7
    assert active_group_count(0.003) == 21
    assert active_group_count(0.01) == 70
    assert active_group_count(0.02) == 139


def _check_config_contract(base):
    effective = resolve_config(base, PROJECT_ROOT)
    assert effective["method"] == "shot"
    assert effective["variant"] == "group_saliency"
    assert effective["task"] == "otta"
    assert effective["adaptation"]["update_scope"] == "last_3_block_weights"
    assert effective["adaptation"]["candidate_blocks"] == [9, 10, 11]
    assert effective["adaptation"]["grouping"] == "paired_qk_vo_ffn"
    assert (
        effective["adaptation"]["delta_semantics"]
        == "per_step_masked_accumulation"
    )
    assert effective["adaptation"]["saliency_score"] == "gradient_l2_norm"
    assert effective["adaptation"]["budget"] == 0.005
    assert effective["adaptation"]["total_groups"] == 6912
    assert effective["adaptation"]["active_groups"] == 35
    scientific = effective["scientific_config"]
    assert scientific["protocol"] == (
        "sequential_target_stream_group_saliency_adaptation"
    )
    assert scientific["adaptation"]["grouping"] == "paired_qk_vo_ffn"
    assert scientific["adaptation"]["selection"] == "saliency"
    assert scientific["adaptation"]["saliency_score"] == "gradient_l2_norm"
    assert scientific["adaptation"]["mask_static"] is False
    assert scientific["adaptation"]["mask_refresh_policy"] == (
        "every_online_step_after_backward"
    )
    assert scientific["adaptation"]["active_groups"] == 35
    assert "group_saliency" in effective["experiment_key"]
    return effective


def _check_identity_binds(base):
    first = resolve_config(base, PROJECT_ROOT)
    changed = copy.deepcopy(base)
    changed["adaptation"]["budget"] = 0.01
    second = resolve_config(changed, PROJECT_ROOT)
    assert first["experiment_key"] != second["experiment_key"]


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
    bad["adaptation"]["delta_semantics"] = "strict_masked_delta"
    _expect_value_error(bad, "delta_semantics")
    bad = copy.deepcopy(base)
    bad["adaptation"]["saliency_score"] = "abs_parameter_times_gradient"
    _expect_value_error(bad, "saliency_score")
    bad = copy.deepcopy(base)
    bad["adaptation"]["budget"] = 0.0
    _expect_value_error(bad, "budget")
    bad = copy.deepcopy(base)
    bad["adaptation"]["candidate_blocks"] = [9, 10, 13]
    _expect_value_error(bad, "in [0, 12)")


def _check_end_to_end(root, base):
    install_fake_checkpoint(root)
    effective = resolve_config(base, PROJECT_ROOT)
    SaliencyTinyDeiT.total_forward_calls = 0
    summary = run_deit_otta_saliency_experiment(
        effective, PROJECT_ROOT, model_factory=tiny_factory
    )
    assert summary["status"] == "completed"
    assert summary["variant"] == "group_saliency"
    assert summary["budget"] == 0.005
    assert summary["candidate_groups"] == 6912
    assert summary["active_groups"] == 35
    assert summary["candidate_scalars"] == CANDIDATE_SCALARS
    assert summary["active_scalars"] == 35 * SCALARS_PER_GROUP
    # 2 batches x 1 step, each step selects exactly 35 distinct groups, so the
    # union is between 35 and 70.
    union = summary["selected_groups_union_count"]
    assert 35 <= union <= 70
    assert len(summary["selected_groups_union"]) == union
    assert 0 < summary["delta_nonzero_scalars"] <= union * SCALARS_PER_GROUP
    assert 0.0 <= summary["PU-Acc"] <= 100.0
    assert 0.0 <= summary["FO-Acc"] <= 100.0
    assert math.isfinite(summary["PU-Acc"])
    assert math.isfinite(summary["FO-Acc"])
    # 2 batches x 2 forwards (loss + PU) + 2 FO forwards = 6 forwards total.
    assert SaliencyTinyDeiT.total_forward_calls == 6

    output = Path(summary["output_dir"])
    assert (output / "manifest.json").is_file()
    assert (output / "summary.json").is_file()
    assert (output / "mask_union.pt").is_file()
    assert (output / "metrics.jsonl").is_file()
    metrics = [
        json.loads(line)
        for line in (output / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    batch_rows = [row for row in metrics if row["event"] == "online_batch"]
    assert len(batch_rows) == 2
    assert all(row["mask_static"] is False for row in batch_rows)
    assert all(
        row["mask_refresh_policy"] == "every_online_step_after_backward"
        for row in batch_rows
    )
    assert all(row["active_groups"] == 35 for row in batch_rows)
    final_rows = [row for row in metrics if row["event"] == "final"]
    assert len(final_rows) == 1
    assert final_rows[0]["status"] == "completed"
    return summary


def main():
    with tempfile.TemporaryDirectory(prefix="deit_otta_saliency_") as root:
        base = _base(root)
        _check_saliency_selection()
        _check_budget()
        _check_config_contract(base)
        _check_identity_binds(base)
        _check_validation(base)
        _check_end_to_end(root, base)
    print("DeiT OTTA saliency structural-group baseline tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
