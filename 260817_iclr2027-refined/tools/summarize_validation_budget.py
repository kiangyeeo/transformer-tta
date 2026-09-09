#!/usr/bin/env python3
"""Produce isolated validation evidence for one frozen LBI budget plan."""
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from tools.check_experiment_status import check_status, load_plan, scan_run_summaries


def num(value):
    return None if value in (None, "") else float(value)


def mean(values):
    return sum(values) / len(values) if values else None


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_baselines(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return {
            (int(r["source"]), int(r["target"]), int(r["seed"]), float(r["requested_budget"])): float(r["best_sparse_FO"])
            for r in csv.DictReader(handle)
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--runs-root", required=True)
    p.add_argument("--baseline-reference", required=True)
    p.add_argument("--frozen-config", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--phase-dir", required=True)
    p.add_argument("--phase", required=True)
    args = p.parse_args()
    plan = load_plan(args.plan)
    output, phase = Path(args.output_dir), Path(args.phase_dir)
    output.mkdir(parents=True, exist_ok=True); phase.mkdir(parents=True, exist_ok=True)
    baselines = read_baselines(args.baseline_reference)
    summaries, invalid = scan_run_summaries(args.runs_root)
    by_key = {r["experiment_key"]: r["summary"] for r in summaries}
    paths = {r["experiment_key"]: r["summary_path"] for r in summaries}
    rows = []
    for exp in plan["experiments"]:
        key = exp["experiment_key"]; summary = by_key.get(key)
        base_key = (int(exp["source"]), int(exp["target"]), int(exp["seed"]), float(exp["requested_budget"]))
        base = baselines[base_key]
        complete = bool(summary and summary.get("status") == "completed")
        fo = num(summary.get("FO-Acc")) if summary else None
        rows.append({
            "phase": args.phase, "experiment_key": key, "source": exp["source"], "target": exp["target"], "seed": exp["seed"],
            "requested_budget": exp["requested_budget"], "summary_path": paths.get(key, ""), "completed": complete,
            "valid_lbi_run": bool(complete and summary.get("valid_lbi_run") is True) if summary else False,
            "FO_Acc": fo, "best_sparse_FO": base, "FO_margin": fo - base if fo is not None else None,
            "budget_reached_all": bool(summary and summary.get("budget_reached_all_steps")),
            "max_steps_hit_count": int(summary.get("max_steps_hit_count", 0)) if summary else 0,
            "stage1_steps": num(summary.get("stage1_steps_mean")) if summary else None,
            "runtime": num(summary.get("runtime")) if summary else None,
        })
    fields = list(rows[0])
    write_csv(output / "per_run.csv", rows, fields)
    groups = defaultdict(list)
    for row in rows: groups[(row["source"], row["target"])].append(row)
    transfer_rows = []
    for (source, target), group in sorted(groups.items()):
        transfer_rows.append({
            "source": source, "target": target, "planned": len(group), "completed": sum(r["completed"] for r in group),
            "valid": sum(r["valid_lbi_run"] for r in group), "FO_Acc_mean": mean([r["FO_Acc"] for r in group if r["FO_Acc"] is not None]),
            "best_sparse_FO_mean": mean([r["best_sparse_FO"] for r in group]), "FO_margin_mean": mean([r["FO_margin"] for r in group if r["FO_margin"] is not None]),
            "stage1_steps_mean": mean([r["stage1_steps"] for r in group if r["stage1_steps"] is not None]),
            "runtime_sum": sum(r["runtime"] for r in group if r["runtime"] is not None),
        })
    write_csv(output / "per_transfer.csv", transfer_rows, list(transfer_rows[0]))
    fos = [r["FO_Acc"] for r in rows if r["FO_Acc"] is not None]; valid = [r for r in rows if r["valid_lbi_run"]]
    valid_fos = [r["FO_Acc"] for r in valid if r["FO_Acc"] is not None]
    margins = [r["FO_margin"] for r in rows if r["FO_margin"] is not None]
    steps = [r["stage1_steps"] for r in rows if r["stage1_steps"] is not None]
    runtime = [r["runtime"] for r in rows if r["runtime"] is not None]
    status = check_status(plan, args.runs_root)
    frozen_sha = hashlib.sha256(Path(args.frozen_config).read_bytes()).hexdigest()
    report = {
        "phase": args.phase, "plan_path": str(Path(args.plan).resolve()), "result_path": str(output.resolve()), "frozen_config_path": str(Path(args.frozen_config).resolve()), "frozen_config_sha256": frozen_sha,
        "planned": len(rows), "completed": sum(r["completed"] for r in rows), "failed": 0, "missing": status["status_counts"]["missing"], "valid": len(valid), "validity_ratio": len(valid) / len(rows),
        "budget_hits": sum(r["budget_reached_all"] for r in rows), "max_step_hits": sum(r["max_steps_hit_count"] for r in rows),
        "FO_all_runs": mean(fos), "FO_valid_only": mean(valid_fos), "best_sparse_FO": mean([r["best_sparse_FO"] for r in rows]), "mean_margin": mean(margins), "worst_run_margin": min(margins) if margins else None,
        "worst_transfer_mean_margin": min((r["FO_margin_mean"] for r in transfer_rows if r["FO_margin_mean"] is not None), default=None),
        "stage1_steps_min": min(steps) if steps else None, "stage1_steps_mean": mean(steps), "stage1_steps_max": max(steps) if steps else None, "runtime_sum_seconds": sum(runtime), "runtime_mean_seconds": mean(runtime),
        "status_counts": status["status_counts"], "unassociated_invalid_summaries": invalid,
        "resume_command": f"conda run -n SHOT_TTA python iclr2027/tools/run_experiments_multi_gpu.py {args.plan} --runs-root {args.runs_root} --logs-root {Path(args.phase_dir).parents[1] / 'launcher_logs/validation/budget_002'} --workdir . --gpus 0,1,2,3,4,5,6,7 --max-workers 16 --workers-per-gpu 2 --resume",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown = "# budget_002 validation summary\n\n" + "\n".join(f"- {k}: `{v}`" for k, v in report.items() if k not in {"status_counts", "unassociated_invalid_summaries", "resume_command"}) + "\n"
    (output / "SUMMARY.md").write_text(markdown, encoding="utf-8")
    manifest = {"phase": args.phase, "plan_path": report["plan_path"], "result_path": report["result_path"], "frozen_config_sha256": frozen_sha, "status_counts": status["status_counts"], "outputs": [str((output / n).resolve()) for n in ("per_run.csv", "per_transfer.csv", "summary.json", "SUMMARY.md")]}
    (phase / "EXPERIMENT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (phase / "CURRENT_STATUS.md").write_text("# budget_002_validation status\n\nCompleted isolated validation.\n\n" + "\n".join(f"- {k}: `{v}`" for k, v in report.items() if k in {"planned", "completed", "failed", "missing", "valid", "validity_ratio", "frozen_config_sha256"}) + "\n", encoding="utf-8")
    (phase / "MERGE_SUMMARY.md").write_text("# Merge-ready summary: budget_002_validation\n\n" + "\n".join(f"- {k}: `{v}`" for k, v in report.items() if k not in {"status_counts", "unassociated_invalid_summaries"}) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
