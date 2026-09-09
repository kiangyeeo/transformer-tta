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


# Node 5 — Office-31 W→D

Search root:
`experiment_logs/office_conv_baselines_seed2026_formal_20260827`

Run ONLY this Office transfer. Read the repository's existing Office transfer mapping and verify the formal source/target identifiers before planning.

Exact scientific conditions:
- `conv_module_dense` × 1
- `conv_out_random` × 3 budgets
- `conv_out_magnitude` × 3 budgets
- `conv_out_saliency` × 3 budgets
- `conv_filter_random` × 3 budgets
- `conv_filter_magnitude` × 3 budgets
- `conv_filter_saliency` × 3 budgets

Total: **19 scientific conditions**.

Random uses formal seed 2026 and the frozen 3 deterministic child masks.
For Node 2 D→A, do NOT reuse pilot artifacts; rerun under the formal root.

## MODE=PREPARE
1. Verify frozen protocol/revision and the exact W→D transfer mapping; otherwise stop.
2. Create matrix + plan containing exactly the 19 conditions above.
3. Verify budgets are only `.0005/.001/.002`; no `.005`, no LBI.
4. Create `tools/run_office_conv_formal_node5_wd_noninteractive.sh`.
5. Static checks only: condition count, plan/identity/output uniqueness, Random 3-mask semantics, `bash -n`, `git diff --check`.
6. Write `experiment_logs/office_conv_baselines_seed2026_formal_20260827/phase_records/node5_wd/PREPARE.md`.
7. Print only the final two-line submit command, then stop:

```bash
cd /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined
bash tools/run_office_conv_formal_node5_wd_noninteractive.sh
```

## MODE=FINALIZE
1. Aggregate only this node's W→D formal runs.
2. Verify all 19 scientific conditions are complete, finite, hash-matched, and all sparse baselines satisfy `selected_group_count == K_G`; verify all Random parents have 3 child masks.
3. Report at least: variant, group mode, rho, K_G, selected groups/scalars, realized scalar ratio, PU, FO, runtime, peak allocated/reserved GPU memory.
4. Report Random mean/std and `FO - conv_module_dense`.
5. Write `experiment_logs/office_conv_baselines_seed2026_formal_20260827/phase_records/node5_wd/FINALIZE.md`.
6. Do NOT launch/retry/rerun or start another phase.
