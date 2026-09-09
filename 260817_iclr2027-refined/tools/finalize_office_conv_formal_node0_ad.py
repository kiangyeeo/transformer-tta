#!/usr/bin/env python3
"""Verify and report Node 0's completed Office A->D Conv formal runs."""

import json
import math
import os
import os.path as osp
import sys
from pathlib import Path

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.check_experiment_status import check_status


FORMAL_ROOT = osp.join(
    PROJECT_ROOT, "experiment_logs", "office_conv_baselines_seed2026_formal_20260827"
)
PLAN_PATH = osp.join(FORMAL_ROOT, "plans", "node0_ad", "plan.json")
REPORT_DIR = osp.join(FORMAL_ROOT, "reports", "node0_ad")
REPORT_MD = osp.join(REPORT_DIR, "FINAL_REPORT.md")
REPORT_JSON = osp.join(REPORT_DIR, "FINAL_REPORT.json")
FINALIZE_MD = osp.join(FORMAL_ROOT, "phase_records", "node0_ad", "FINALIZE.md")
BASELINE_REVISION = "OTTA_CONV_BASELINE_FORMAL_20260827_v1"
CONV_REVISION = "OTTA_CONV_LBI_PROTOCOL_20260826_v1"
IMPLEMENTATION_REVISION = "iclr2027_refined_conv_20260826_v1"
SOURCE_REVISION = "nips2026_shot_otta_uda_source_v1"
EFFICIENCY_REVISION = "otta_fc_batch_efficiency_20260817_v1"
RANDOM_VARIANTS = {"conv_out_random", "conv_filter_random"}
EXPECTED_K = {
    "out_channel": {0.0005: 4, 0.001: 9, 0.002: 18},
    "filter_connection": {0.0005: 3276, 0.001: 6553, 0.002: 13107},
}
FINITE_FIELDS = (
    "PU-Acc",
    "FO-Acc",
    "runtime",
    "wall_runtime_sec",
    "peak_gpu_memory_allocated_mb",
    "peak_gpu_memory_reserved_mb",
)


def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"non-finite {label}: {value!r}")
    return float(value)


def fmt(value, digits=6):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def verify_and_collect():
    plan = json.loads(Path(PLAN_PATH).read_text(encoding="utf-8"))
    entries = plan.get("experiments", [])
    if plan.get("experiment_count") != 19 or len(entries) != 19:
        raise ValueError("Node 0 plan must contain exactly 19 experiments")
    if plan.get("formal_baseline_protocol_revision") != BASELINE_REVISION:
        raise ValueError("formal baseline protocol revision drifted")
    if plan.get("protocol_revision") != CONV_REVISION:
        raise ValueError("Conv protocol revision drifted")
    if plan.get("implementation_revision") != IMPLEMENTATION_REVISION:
        raise ValueError("implementation revision drifted")
    status = check_status(plan, osp.join(FORMAL_ROOT, "runs"))
    expected_status = {"completed": 19, "missing": 0, "duplicate_completed": 0, "hash_mismatch": 0, "invalid_summary": 0}
    if status.get("status_counts") != expected_status:
        raise ValueError(f"Node 0 status is not complete and unique: {status.get('status_counts')}")
    if status.get("unassociated_invalid_summaries"):
        raise ValueError("unassociated invalid summaries are present")
    status_by_key = {row["experiment_key"]: row for row in status["experiments"]}
    rows = []
    dense_fo = None
    for entry in entries:
        row_status = status_by_key[entry["experiment_key"]]
        paths = row_status.get("matching_summary_paths") or []
        if len(paths) != 1:
            raise ValueError(f"expected exactly one matching summary: {entry['experiment_key']}")
        summary = json.loads(Path(paths[0]).read_text(encoding="utf-8"))
        for field in FINITE_FIELDS:
            finite(summary.get(field), f"{entry['variant']} {field}")
        for field, expected in (
            ("experiment_key", entry["experiment_key"]),
            ("experiment_config_sha256", entry["experiment_config_sha256"]),
            ("implementation_revision", IMPLEMENTATION_REVISION),
            ("protocol_revision", CONV_REVISION),
            ("source_checkpoint_revision", SOURCE_REVISION),
            ("efficiency_protocol_revision", EFFICIENCY_REVISION),
            ("dataset", "office"),
            ("source", 0),
            ("target", 1),
            ("seed", 2026),
            ("status", "completed"),
        ):
            if summary.get(field) != expected:
                raise ValueError(f"summary {field} mismatch for {entry['experiment_key']}")
        variant = entry["variant"]
        mode = entry.get("group_mode")
        rho = entry.get("requested_budget")
        selected_group = summary.get("selected_group_count")
        selected_scalar = summary.get("selected_scalar_count")
        scalar_ratio = summary.get("realized_scalar_ratio")
        if variant == "conv_module_dense":
            dense_fo = finite(summary["FO-Acc"], "dense FO-Acc")
            if any(summary.get(field) is not None for field in ("selected_group_count", "selected_scalar_count", "realized_scalar_ratio")):
                raise ValueError("dense summary unexpectedly reports sparse selection")
        else:
            expected_k = EXPECTED_K[mode][float(rho)]
            if not math.isclose(float(selected_group), expected_k, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(f"selected_group_count != K_G for {entry['experiment_key']}")
            finite(selected_scalar, f"{variant} selected_scalar_count")
            finite(scalar_ratio, f"{variant} realized_scalar_ratio")
        mask_rows = []
        if variant in RANDOM_VARIANTS:
            masks = summary.get("masks")
            if not isinstance(masks, list) or len(masks) != 3:
                raise ValueError(f"Random parent does not have exactly 3 child masks: {entry['experiment_key']}")
            expected_seeds = [202600, 202601, 202602]
            if summary.get("mask_seeds") != expected_seeds:
                raise ValueError(f"Random child mask seeds drifted: {entry['experiment_key']}")
            for mask, expected_seed in zip(masks, expected_seeds):
                if mask.get("mask_seed") != expected_seed:
                    raise ValueError(f"Random child seed mismatch: {entry['experiment_key']}")
                if not math.isclose(float(mask.get("selected_group_count")), expected_k, rel_tol=0.0, abs_tol=1e-9):
                    raise ValueError(f"Random child selected_group_count != K_G: {entry['experiment_key']}")
                for field in ("PU-Acc", "FO-Acc", "runtime", "selected_scalar_count", "realized_scalar_ratio"):
                    finite(mask.get(field), f"Random child {field}")
                mask_rows.append(mask)
            pu_mean = finite(summary.get("mean_PU-Acc"), "Random mean PU-Acc")
            fo_mean = finite(summary.get("mean_FO-Acc"), "Random mean FO-Acc")
            pu_std = finite(summary.get("std_PU-Acc"), "Random std PU-Acc")
            fo_std = finite(summary.get("std_FO-Acc"), "Random std FO-Acc")
        else:
            pu_mean, fo_mean, pu_std, fo_std = finite(summary["PU-Acc"], "PU"), finite(summary["FO-Acc"], "FO"), None, None
        rows.append({
            "variant": variant,
            "group_mode": mode,
            "rho": None if mode is None else float(rho),
            "K_G": None if mode is None else EXPECTED_K[mode][float(rho)],
            "selected_group_count": selected_group,
            "selected_scalar_count": selected_scalar,
            "realized_scalar_ratio": scalar_ratio,
            "PU": pu_mean,
            "FO": fo_mean,
            "runtime": finite(summary["runtime"], "runtime"),
            "peak_allocated_mb": finite(summary["peak_gpu_memory_allocated_mb"], "allocated memory"),
            "peak_reserved_mb": finite(summary["peak_gpu_memory_reserved_mb"], "reserved memory"),
            "PU_std": pu_std,
            "FO_std": fo_std,
            "mask_seeds": summary.get("mask_seeds"),
            "mask_rows": mask_rows,
            "summary_path": paths[0],
        })
    if dense_fo is None:
        raise ValueError("dense anchor missing")
    for row in rows:
        row["FO_minus_dense"] = row["FO"] - dense_fo
    return plan, rows, dense_fo, status


def render_report(plan, rows, dense_fo, status):
    lines = [
        "# Node 0 Office-31 A→D Conv formal baseline report",
        "",
        "- Formal baseline protocol: `OTTA_CONV_BASELINE_FORMAL_20260827_v1`",
        "- Conv protocol / implementation: `OTTA_CONV_LBI_PROTOCOL_20260826_v1` / `iclr2027_refined_conv_20260826_v1`",
        "- Transfer: Office `amazon → dslr` (`source=0`, `target=1`, `AD`); seed `2026`.",
        "- Scope: only Node 0 A→D runs under `experiment_logs/office_conv_baselines_seed2026_formal_20260827`.",
        "",
        "## Verification",
        "",
        "All 19/19 conditions are complete, finite, and hash-matched with one summary each. Status counts: `completed=19`, `missing=0`, `duplicate_completed=0`, `hash_mismatch=0`, `invalid_summary=0`. Every sparse summary and every Random child has exact `selected_group_count == K_G`; each Random parent has child masks `202600/202601/202602`.",
        "",
        "## Results",
        "",
        "| variant | group mode | rho | K_G | selected groups | selected scalars | realized scalar ratio | PU | FO | FO − dense | runtime (s) | peak alloc (MB) | peak reserved (MB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {group_mode} | {rho} | {K_G} | {selected_group_count} | {selected_scalar_count} | {realized_scalar_ratio} | {PU} | {FO} | {FO_minus_dense} | {runtime} | {peak_alloc_mb} | {peak_reserved_mb} |".format(
                variant=row["variant"], group_mode=row["group_mode"] or "n/a", rho=fmt(row["rho"]), K_G=fmt(row["K_G"], 0),
                selected_group_count=fmt(row["selected_group_count"], 3), selected_scalar_count=fmt(row["selected_scalar_count"], 3),
                realized_scalar_ratio=fmt(row["realized_scalar_ratio"], 9), PU=fmt(row["PU"]), FO=fmt(row["FO"]),
                FO_minus_dense=fmt(row["FO_minus_dense"]), runtime=fmt(row["runtime"]), peak_alloc_mb=fmt(row["peak_allocated_mb"]), peak_reserved_mb=fmt(row["peak_reserved_mb"]),
            )
        )
    lines.extend(["", f"Dense anchor FO: `{dense_fo:.6f}`.", "", "## Random 3-mask aggregation", "", "| variant | rho | child seeds | PU mean ± std | FO mean ± std | FO − dense |", "|---|---:|---|---:|---:|---:|"])
    for row in rows:
        if row["variant"] in RANDOM_VARIANTS:
            lines.append(f"| {row['variant']} | {row['rho']:.4f} | `{row['mask_seeds']}` | {row['PU']:.6f} ± {row['PU_std']:.6f} | {row['FO']:.6f} ± {row['FO_std']:.6f} | {row['FO_minus_dense']:.6f} |")
    lines.extend(["", "No additional phase was launched, retried, or rerun during FINALIZE.", ""])
    return "\n".join(lines)


def main():
    plan, rows, dense_fo, status = verify_and_collect()
    report = render_report(plan, rows, dense_fo, status)
    os.makedirs(REPORT_DIR, exist_ok=True)
    Path(REPORT_MD).write_text(report, encoding="utf-8")
    payload = {
        "schema_version": 1,
        "node": "node0_ad",
        "protocol_revision": CONV_REVISION,
        "formal_baseline_protocol_revision": BASELINE_REVISION,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "source_checkpoint_revision": SOURCE_REVISION,
        "transfer": {"dataset": "office", "source": 0, "target": 1, "identifier": "AD"},
        "status_counts": status["status_counts"],
        "dense_FO": dense_fo,
        "rows": rows,
    }
    Path(REPORT_JSON).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    finalize = "# Node 0 — Office-31 A→D Conv formal baselines — FINALIZE\n\nStatus: complete. No launch, retry, rerun, or new phase was performed during FINALIZE.\n\n" + report.split("\n", 1)[1]
    Path(FINALIZE_MD).write_text(finalize, encoding="utf-8")
    print(json.dumps({"valid": True, "completed": 19, "report": REPORT_MD}))


if __name__ == "__main__":
    main()
