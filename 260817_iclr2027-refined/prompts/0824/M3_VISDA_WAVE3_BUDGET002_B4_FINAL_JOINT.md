Project:
`/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined`

Conda: `SHOT_TTA`

Protocol:
`protocol/shot-otta_fc/OTTA_FC_LBI_PROTOCOL_20260817_v1.md`

Search root:
`experiment_logs/visda_fc_lbi_seed2026_search_20260819`

Use current refined code as the only algorithm source of truth.

Hard boundaries:
- Only create/modify planning, launcher, report, selected-config files under current refined repo.
- Do NOT modify algorithm code, frozen protocol, checkpoints, `nips2026/`, or old versions.
- seed=2026, VisDA-C, batch_size=256, workers=4, kappa=1, Stage1 max=3000, Stage2 steps=1.
- PREPARE: no real training, no GPU probing.
- FINALIZE: no launch/retry/rerun.
- Launcher: foreground, non-interactive, GPUs 0-7, max-workers=8, workers-per-gpu=1, `--resume --resume-partial-runs`.
- Scientific-valid: completed, no NaN/error, selected_count<=K, zero Stage1 max-step failures. Exact K is NOT required.
- Then require hard90 utilization.
- Rank eligible rows by: FO mAcc desc -> worst-class desc -> class-std asc -> FO overall desc -> S1 max asc -> S1 mean asc.
- PU is report-only.
- Run ONLY the exact new rows listed below. Reuse listed references; do NOT rerun them.
- This is FINAL joint search. Do NOT expand omega/LR/alpha after FINALIZE even if winner is on a boundary.


# M3 — budget=.002, anchor B4

Fixed:
```text
budget=.002, K=1049
alpha=.15, nu=.5
omega in {.0015625,.003125,.00625}
lr2   in {.0025,.005,.010}
```

Existing references; REUSE, do not rerun:
```text
omega=.003125, lr2=.005
omega=.00625,  lr2=.005
```

Create exactly 7 NEW runs:
```text
.0015625/.0025
.0015625/.005
.0015625/.010
.003125 /.0025
.003125 /.010
.00625  /.0025
.00625  /.010
```

## MODE=PREPARE
1. Verify both reference identities/configs exist and match exactly; otherwise stop.
2. Create matrix + plan with exactly the 7 NEW runs.
3. Create `tools/run_visda_wave3_m3_budget002_B4_final_joint_noninteractive.sh`.
4. Static checks only: plan uniqueness, `bash -n`, `git diff --check`.
5. Write `${SEARCH_ROOT}/phase_records/wave3_m3/PREPARE.md`.
6. Print only the final two-line submit command, then stop.

## MODE=FINALIZE
Aggregate the full 3x3 grid = 7 new + 2 references.
Report validity/utilization/S1/FO/PU metrics and rank with the common rule.
Write `${SEARCH_ROOT}/phase_records/wave3_m3/FINALIZE.md`.
Do NOT freeze budget=.002 alone here and do NOT launch extra runs.
