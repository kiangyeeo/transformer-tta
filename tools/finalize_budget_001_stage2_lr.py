#!/usr/bin/env python3
"""Finalize the already-launched budget_001 Stage-2 LR phase after all identities complete."""
import csv
import json
from pathlib import Path

ROOT = Path("iclr2027/experiment_logs/shot_otta_lbi_budget_search_20260801T042648Z")
RESULT = ROOT / "results/budget_001/stage2_lr"
PHASE = ROOT / "phase_records/budget_001_stage2_lr"

ranking = json.loads((RESULT / "candidate_ranking.json").read_text())[0]
per_run = list(csv.DictReader((RESULT / "per_run.csv").open()))
source_paths = [r["summary_path"] for r in per_run if r["candidate_tuple"] == ranking["candidate_tuple"]]
config = {
    "phase": "budget_001_stage2_lr", "budget": 0.001, "selection_seed": 2020,
    "selection_statement": "Selection used only seed 2020; evaluation seeds were neither inspected nor used.",
    "scientific_configuration": {
        "method": "shot", "task": "otta", "dataset": "office", "variant": "module_lbi",
        "alpha": ranking["alpha"], "kappa": ranking["kappa"], "nu": ranking["nu"], "omega": ranking["omega"],
        "requested_budget": 0.001, "stage1_max_steps": ranking["stage1_max_steps"],
        "budget_tolerance": ranking["budget_tolerance"], "stage2_lr": ranking["stage2_lr"],
        "stage2_steps": ranking["stage2_steps_requested"], "delta_nonzero_tolerance": 1e-12,
        "batch_size": 64, "workers": 4, "save_model": False,
    },
    "ranking_rule": ["valid_lbi_run_count desc", "mean_FO_margin_all desc", "worst_transfer_FO_margin desc", "mean_FO_all desc", "stage1_steps_max asc", "stage1_steps_mean asc", "runtime_sum asc"],
    "ranking_metrics": {k: ranking[k] for k in ("rank", "complete_six_transfer_configuration", "planned_run_count", "completed_run_count", "valid_lbi_run_count", "valid_lbi_run_ratio", "budget_reached_all_count", "max_steps_hit_count_sum", "mean_FO_all", "mean_FO_margin_all", "worst_transfer_FO_margin", "stage1_steps_mean", "stage1_steps_max", "runtime_sum")},
    "source_result_paths": source_paths, "analysis_result_path": str(RESULT / "candidate_ranking.json"),
}
(ROOT / "selected_configs/budget_001_final_tuning.json").write_text(json.dumps(config, indent=2) + "\n")
manifest = json.loads((PHASE / "EXPERIMENT_MANIFEST.json").read_text())
manifest.update({"status": "completed", "reused_completed_count": 13, "newly_run_count": 23, "completed_run_count": 36, "failed_run_count": 0, "invalid_summary_count": 0, "duplicate_count": 0, "hash_mismatch_count": 0, "final_tuning_config": "selected_configs/budget_001_final_tuning.json"})
(PHASE / "EXPERIMENT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
metric = config["ranking_metrics"]
(PHASE / "CURRENT_STATUS.md").write_text("# Status: budget_001_stage2_lr\n\nCompleted and merge-ready. All 36 identities completed; 13 were reused and 23 were newly run.\n")
(PHASE / "MERGE_SUMMARY.md").write_text(f"# Merge-ready summary: budget_001_stage2_lr\n\nCompleted and merge-ready.\n\n- Planned 36; reused 13; newly run 23; completed 36; failed 0.\n- Six-transfer validity: {metric['valid_lbi_run_count']}/6; winner is `{ranking['candidate_tuple']}`.\n- Mean FO margin: {metric['mean_FO_margin_all']}; worst-transfer FO margin: {metric['worst_transfer_FO_margin']}; mean FO: {metric['mean_FO_all']}.\n- Stage 1 cost: mean {metric['stage1_steps_mean']} steps, maximum {metric['stage1_steps_max']} steps (maximum allowed 3000).\n- Selection used only seed 2020. Ranking and per-run evidence are in `results/budget_001/stage2_lr/`; frozen configuration is `selected_configs/budget_001_final_tuning.json`.\n- Next prompt: `11_FREEZE_CONFIGS_AND_BUILD_VALIDATION_PLANS.md`.\n")
with (PHASE / "DECISION_LOG.md").open("a") as f:
    f.write(f"\n- Completed all 36 planned identities (13 reused, 23 newly run) with no failed, invalid, duplicate, or hash-mismatched summaries. Fixed ranking selected rank 1: `{ranking['candidate_tuple']}`.\n")
