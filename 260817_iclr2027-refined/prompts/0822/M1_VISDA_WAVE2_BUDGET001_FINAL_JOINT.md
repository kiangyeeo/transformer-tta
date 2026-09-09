Project root:
`/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI`

Current repo:
`260817_iclr2027-refined/`

Conda:
`SHOT_TTA`

Normative protocol:
`260817_iclr2027-refined/protocol/shot-otta_fc/OTTA_FC_LBI_PROTOCOL_20260817_v1.md`

Shared search root:
`260817_iclr2027-refined/experiment_logs/visda_fc_lbi_seed2026_search_20260819`

Modes:
`MODE=PREPARE` or `MODE=FINALIZE`. Complete only the requested mode.

Hard boundaries:
- Read the protocol first. Current refined code is the algorithm source of truth.
- Only add experiment matrix/plan/launcher/report files under `260817_iclr2027-refined/`.
- Do NOT modify LBI algorithm code, frozen protocol, source checkpoints, or `nips2026/`.
- Do NOT change seed=2026, VisDA batch_size=256, workers=4, Stage1 cap=3000, Stage2 steps=1, support threshold=1e-4.
- PREPARE: no real training, no GPU probing; only static planning/checks.
- FINALIZE: read existing artifacts only; no launch/retry/rerun.
- Long LBI launcher must be non-interactive, 1 task/GPU, GPUs 0-7, `--resume --resume-partial-runs`.
- Tuning validity: scientific-valid first (`selected_count<=K`, no NaN/error, no Stage1 max-step failure), then hard90 utilization. Exact K is NOT required.
- Primary tuning metric: FO fixed-12-class mean per-class accuracy. PU is report-only.
- Use only the exact grid below. Do NOT add nearby points or post-hoc expand the search.


# Machine 1 — budget 0.001 final local joint search

## Goal

Close refined VisDA `budget=0.001`.

Fixed:
```text
budget=0.001
K=524
kappa=1.0
nu=0.5
stage2_steps=1
stage1_max_steps=3000
```

Main grid:
```text
alpha ∈ {0.075, 0.10}
omega ∈ {0.003125, 0.00625}
stage2_lr ∈ {0.005, 0.010}
```

The row below already exists from Wave 1 and is REFERENCE ONLY; do not rerun:
```text
alpha=0.10, omega=0.00625, stage2_lr=0.005
FO mean-class ≈ 50.269
```

Therefore create exactly these 7 new main-grid runs:
```text
1  alpha=.075 omega=.003125 lr2=.005
2  alpha=.075 omega=.003125 lr2=.010
3  alpha=.075 omega=.00625  lr2=.005
4  alpha=.075 omega=.00625  lr2=.010
5  alpha=.10  omega=.003125 lr2=.005
6  alpha=.10  omega=.003125 lr2=.010
7  alpha=.10  omega=.00625  lr2=.010
```

Add exactly one lower-LR sentinel as run 8:
```text
8  alpha=.10 omega=.003125 lr2=.0025
```

No other runs.

## MODE=PREPARE

1. Locate the existing Wave-1 reference identity above and verify its config exactly. If inconsistent, stop.
2. Create matrix + plan containing exactly the 8 NEW runs above.
3. Verify experiment_count=8, all unique, seed=2026, budget=.001, K=524.
4. Create:
`260817_iclr2027-refined/tools/run_visda_wave2_m1_budget001_final_joint_noninteractive.sh`
5. Reuse the existing current-repo multi-GPU runner. Use GPUs 0-7, max-workers=8, workers-per-gpu=1, `--resume --resume-partial-runs`, and keep shell attached until all jobs finish. No tmux.
6. Run only static checks: `bash -n` on the launcher and `git diff --check`.
7. Write `${SEARCH_ROOT}/phase_records/wave2_m1/PREPARE.md`.

At the end print exactly:
```bash
cd /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI
bash 260817_iclr2027-refined/tools/run_visda_wave2_m1_budget001_final_joint_noninteractive.sh
```
Then stop.

## MODE=FINALIZE

Do not run training.

Aggregate:
- 8 main-grid rows = 7 new + 1 Wave-1 reference;
- plus the 1 sentinel row.

For each report:
`alpha, omega, lr2, scientific_valid, hard90, min/P05/mean utilization, max-step hits, S1 mean/max steps, FO mean-class, worst-class, class-std, FO overall, PU mean-class`.

Rank eligible main-grid rows by:
FO mean-class descending → worst-class descending → class-std ascending → FO overall descending → S1 max ascending → S1 mean ascending.

The sentinel is diagnostic only unless it strictly beats all eligible main-grid rows under the same ranking.

Write the final selected tuple to:
`${SEARCH_ROOT}/selected_configs/budget_001_final_lbi_refined.json`

**STOP RULE: budget=.001 tuning closes after this phase.**
Do NOT propose or run alpha=.05, omega=.0015625, smaller LR, or any new boundary search merely because the winner is on an edge.

Write `${SEARCH_ROOT}/phase_records/wave2_m1/FINALIZE.md` and stop.
