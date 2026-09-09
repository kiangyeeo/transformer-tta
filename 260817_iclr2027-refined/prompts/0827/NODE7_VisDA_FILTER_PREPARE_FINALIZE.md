Project:
`/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined`

Conda: `SHOT_TTA`

Formal baseline protocol:
`protocol/shot-otta_conv/OTTA_CONV_BASELINE_FORMAL_20260827_v1.md`

Conv method protocol:
`protocol/shot-otta_conv/OTTA_CONV_LBI_PROTOCOL_20260826_v1.md`

Use current refined code and the frozen protocols as the only source of truth.

Hard boundaries:
- Only create/modify planning, launcher, report, phase-record files under the current refined repo.
- Do NOT modify algorithm code, frozen protocols, FC code/results, tests, checkpoints, or frozen revisions.
- Formal budgets are exactly `.0005/.001/.002`; `.005` is not formal.
- No `conv_*_lbi`, no tuning, no extra budgets.
- PREPARE: no real training, no GPU probing.
- FINALIZE: no launch/retry/rerun.
- Launcher: foreground, non-interactive, GPUs 0-3, max-workers=4, workers-per-gpu=1, `--resume --resume-partial-runs`.
- All Python/runner commands inside `.sh` must use `conda run --no-capture-output -n SHOT_TTA ...`.
- Do not ask the user to run `conda activate`.


# Node 7 — VisDA-C Filter-Connection Formal Baselines

Search root:
`experiment_logs/visda_conv_baselines_seed2026_formal_20260827`

Run ONLY:
- `conv_filter_random` × 3 budgets
- `conv_filter_magnitude` × 3 budgets
- `conv_filter_saliency` × 3 budgets

Total: **9 scientific conditions**.

`conv_module_dense` is owned by Node 6 and must not be duplicated here.
Random uses formal seed 2026 and the frozen 3 deterministic child masks.

## MODE=PREPARE
1. Verify frozen protocol/revision and current formal VisDA-C configuration; otherwise stop.
2. Create matrix + plan with exactly the 9 conditions above.
3. Verify budgets are only `.0005/.001/.002`; no `.005`, no out variants, no dense, no LBI.
4. Create `tools/run_visda_conv_formal_node7_filter_noninteractive.sh`.
5. Static checks only: count, plan/identity/output uniqueness, no overlap with Node 6 planned identities/outputs, Random 3-mask semantics, `bash -n`, `git diff --check`.
6. Write `experiment_logs/visda_conv_baselines_seed2026_formal_20260827/phase_records/node7_filter/PREPARE.md`.
7. Print only:

```bash
cd /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined
bash tools/run_visda_conv_formal_node7_filter_noninteractive.sh
```

then stop.

## MODE=FINALIZE
1. Aggregate only Node 7 VisDA formal runs.
2. Verify all 9 conditions are complete, finite, hash-matched; sparse baselines must satisfy `selected_group_count == K_G`; Random parents must have 3 child masks.
3. Report: variant, rho, K_G, selected groups/scalars, realized scalar ratio, PU mAcc/overall, FO mAcc/overall, class-wise metrics if already produced by current summary tools, runtime, peak allocated/reserved GPU memory.
4. Report Random mean/std. Use Node 6 dense only as an external reference if it already exists; do not rerun dense here.
5. Write `experiment_logs/visda_conv_baselines_seed2026_formal_20260827/phase_records/node7_filter/FINALIZE.md`.
6. Do NOT launch/retry/rerun or start another phase.
