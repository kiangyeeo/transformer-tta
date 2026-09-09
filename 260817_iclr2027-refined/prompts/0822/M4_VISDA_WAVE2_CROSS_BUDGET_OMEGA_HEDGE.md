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


# Machine 4 — cross-budget omega hedge

## Goal

Provide controlled omega comparisons for budgets `.0005` and `.002`.
This machine is diagnostic only; do not freeze final tuples here.

Run exactly 8 configs.

### budget=.0005, K=262, omega=.0125
Fixed `kappa=1, lr2=.005`:
```text
C1 alpha=.075 nu=.5
C2 alpha=.075 nu=1.0
C3 alpha=.10  nu=.5
C4 alpha=.10  nu=1.0
```

### budget=.002, K=1049, omega=.00625
Fixed `kappa=1, lr2=.005`:
```text
C5 alpha=.10 nu=.5
C6 alpha=.10 nu=1.0
C7 alpha=.15 nu=.5
C8 alpha=.15 nu=1.0
```

No other configs.

## MODE=PREPARE

1. Build one matrix + plan with exactly C1-C8.
2. Verify experiment_count=8, seed=2026, correct K for each budget, and exact omega values above.
3. Create:
`260817_iclr2027-refined/tools/run_visda_wave2_m4_cross_budget_omega_hedge_noninteractive.sh`
4. Reuse the existing current-repo multi-GPU runner. GPUs 0-7, max-workers=8, workers-per-gpu=1, `--resume --resume-partial-runs`. No tmux.
5. Static checks only: `bash -n` and `git diff --check`.
6. Write `${SEARCH_ROOT}/phase_records/wave2_m4/PREPARE.md`.

At the end print exactly:
```bash
cd /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI
bash 260817_iclr2027-refined/tools/run_visda_wave2_m4_cross_budget_omega_hedge_noninteractive.sh
```
Then stop.

## MODE=FINALIZE

Do not run training.

Report all 8 rows with:
`budget, alpha, nu, omega, scientific_valid, hard90, min/P05/mean utilization, max-step hits, S1 mean/max steps, FO mean-class, worst-class, class-std, FO overall, PU mean-class`.

Make only direct controlled comparisons:

For `.0005`, compare C1-C4 against matching M2 anchors at omega=.00625.

For `.002`, compare C5-C8 against matching M3 anchors at omega=.003125.

State the omega direction supported by the data for each budget.
Do NOT select/freeze final budget tuples and do NOT create extra runs.

Write `${SEARCH_ROOT}/phase_records/wave2_m4/FINALIZE.md` and stop.
