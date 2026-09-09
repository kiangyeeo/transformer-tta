# Codex Prompt — VisDA final formal LBI rerun

Use `MODE=PREPARE` or `MODE=FINALIZE`. Complete only the requested mode.

Project:
`/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined`

Conda: `SHOT_TTA`

Read first:
`protocol/shot-otta_fc/OTTA_FC_LBI_PROTOCOL_20260817_v1.md`

Current refined code is the only algorithm source of truth.

## Hard boundaries

- This is **formal rerun**, not tuning. Do NOT change/search any hyperparameter.
- Do NOT modify algorithm code, frozen protocol, checkpoints, `nips2026/`, or old versions.
- Do NOT reuse tuning run directories/checkpoints as formal results.
- Formal output root must be NEW:
  `experiment_logs/visda_fc_lbi_formal_seed2026_20260825`
- VisDA-C: source=0, target=1, seed=2026, batch_size=256, workers=4.
- Candidate scope: `netB.bottleneck`; controlled-FC BN frozen.
- Stage1 max=3000; Stage2 steps=1; support threshold=1e-4.
- Exact K is NOT required. Scientific validity: completed, finite/error-free, selected_count<=K every batch, zero Stage1 max-step failures.
- Do NOT rerun Source/Random/Magnitude/Saliency/Module-Dense/Full-Dense baselines.
- Formal execution: one experiment/GPU. Use GPU 0,1,2 only; max-workers=3; workers-per-gpu=1.
- Launcher must be foreground/non-interactive and use `--resume --resume-partial-runs`.
- `--resume-partial-runs` is only for interruption recovery inside the NEW formal root, never to resume tuning artifacts.

## Exact frozen configs — create exactly 3 formal runs

```text
F1 budget=.0005  K=262
   alpha=.125  kappa=1  nu=.5
   omega=.00625  stage2_lr=.010

F2 budget=.001   K=524
   alpha=.10   kappa=1  nu=.5
   omega=.003125 stage2_lr=.0025

F3 budget=.002   K=1049
   alpha=.15   kappa=1  nu=.5
   omega=.0015625 stage2_lr=.005
```

No other run/config.

## MODE=PREPARE

1. Verify the three frozen tuples against existing Wave-2/Wave-3 selection/finalize artifacts. If any mismatch, STOP and report it.
2. Create a formal matrix + plan with exactly F1-F3 under the NEW formal root.
3. Ensure the plan has exactly 3 unique identities/config SHAs and contains no tuning rows.
4. Create:
   `tools/run_visda_final_formal_lbi_noninteractive.sh`
5. Launcher requirements:
   - `conda run --no-capture-output -n SHOT_TTA`
   - existing `tools/run_experiments_multi_gpu.py`
   - GPUs `0,1,2`
   - `--max-workers 3`
   - `--workers-per-gpu 1`
   - `--resume --resume-partial-runs`
   - NEW formal runs/logs/command-history paths only
6. Static checks only: plan/config validation, `bash -n`, `git diff --check`. Do NOT launch training or probe GPUs.
7. Write:
   `experiment_logs/visda_fc_lbi_formal_seed2026_20260825/PREPARE.md`

At the end print only:

```bash
cd /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI
bash 260817_iclr2027-refined/tools/run_visda_final_formal_lbi_noninteractive.sh
```

Then stop.

## MODE=FINALIZE

Do NOT launch/retry/rerun.

For F1-F3 verify:
- exact frozen tuple/config SHA;
- 217/217 online batches;
- no NaN/error;
- selected_count<=K every batch;
- zero Stage1 max-step failures;
- scientific-valid=true;
- hard90 utilization (diagnostic);
- `runtime_comparable`;
- `runtime_resume_used`, `runtime_segment_count`, GPU name.

Aggregate for each budget:
- FO mAcc (primary), FO overall;
- all 12 class accuracies;
- worst-class accuracy + class name;
- class-accuracy std;
- PU mAcc / overall (report-only);
- selected count min/mean/max and utilization;
- Stage1 mean/max steps;
- online runtime, Stage1/Stage2 runtime if available, FO eval runtime;
- peak GPU memory.

If `runtime_resume_used=true` for a run, keep its accuracy result but clearly mark its runtime as **not clean final efficiency evidence**; do not silently substitute or rerun it.

Write:
`experiment_logs/visda_fc_lbi_formal_seed2026_20260825/FINALIZE.md`

Also create one compact final VisDA table combining these 3 formal LBI results with the already-existing formal baselines; do not rerun baselines.

Stop after reporting. No new tuning.
