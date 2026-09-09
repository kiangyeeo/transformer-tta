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


# Machine 3 — budget 0.002 Stage-1 warm-start

## Goal

Search Stage-1 dynamics for refined VisDA `budget=0.002`.

Fixed:
```text
budget=0.002
K=1049
kappa=1.0
omega=0.003125
stage2_lr=0.005
stage2_steps=1
stage1_max_steps=3000
```

Run exactly 8 configs:
```text
B1 alpha=.10 nu=1.0
B2 alpha=.10 nu=.5
B3 alpha=.15 nu=1.0
B4 alpha=.15 nu=.5
B5 alpha=.20 nu=1.0
B6 alpha=.20 nu=.5
B7 alpha=.25 nu=1.0
B8 alpha=.25 nu=.5
```

No alpha=.30, no nu=.25, no kappa changes, no other omega/LR.

## MODE=PREPARE

1. Build matrix + plan with exactly the 8 configs above.
2. Verify experiment_count=8, seed=2026, budget=.002, K=1049, omega=.003125, lr2=.005.
3. Create:
`260817_iclr2027-refined/tools/run_visda_wave2_m3_budget002_stage1_noninteractive.sh`
4. Reuse the existing current-repo multi-GPU runner. GPUs 0-7, max-workers=8, workers-per-gpu=1, `--resume --resume-partial-runs`. No tmux.
5. Static checks only: `bash -n` and `git diff --check`.
6. Write `${SEARCH_ROOT}/phase_records/wave2_m3/PREPARE.md`.

At the end print exactly:
```bash
cd /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI
bash 260817_iclr2027-refined/tools/run_visda_wave2_m3_budget002_stage1_noninteractive.sh
```
Then stop.

## MODE=FINALIZE

Do not run training.

For all 8 rows report:
`alpha, nu, scientific_valid, hard90, min/P05/mean utilization, <95% batch count, max-step hits, S1 mean/max steps, FO mean-class, worst-class, class-std, FO overall, PU mean-class`.

Rank eligible rows by:
FO mean-class ↓ → worst-class ↓ → class-std ascending → FO overall ↓ → S1 max ↓ → S1 mean ↓.

Compare the best eligible row with the current refined `budget=.002` best sparse baseline.

Stage-1 stop rule:
- if at least 2 rows are scientific-valid + hard90 and best eligible > best sparse, mark Stage-1 CLOSED and save top-2 anchors;
- if fewer than 2 eligible rows exist, report that stronger Stage-1 dynamics may be needed later. Do NOT add/run alpha=.30 or nu=.25 in this prompt.

Write top-2 eligible anchors, if available, to:
`${SEARCH_ROOT}/selected_configs/budget_002_stage1_top2_refined.json`

Write `${SEARCH_ROOT}/phase_records/wave2_m3/FINALIZE.md` and stop.
