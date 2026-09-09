# COME Office-31 LBI Tuning Protocol (2026-09-08 v1)

This task-specific protocol instantiates
`OTTA_COME_LBI_PROTOCOL_20260907_v1` without changing its scientific
implementation. The controlling specification is
`prompts/0908/come-lbi/CODEX_COME_OFFICE_LBI_TUNING_KAPPA1_5GPU_20260908.md`.

- Scope: COME, Office-31, FC scalar and Conv out-channel, budgets 0.0005,
  0.001, and 0.002, seed 2026.
- Every run is `formal_protocol=false`, `tuning=true`, uses one target pass,
  batch size 64, four workers, and the frozen SHOT source F/B/C.
- Kappa is fixed at 1. Stage-1 selection searches only `(alpha, nu)` on the
  first five valid D-to-A outer batches, with omega 0.025 and Stage-2 LR
  0.005. Accuracy is excluded from Stage-1 ranking.
- Primary and backup Stage-1 candidates use the exact grids, rescue gates,
  ranking, and six-transfer validation thresholds in the controlling
  specification. Frozen Stage-1 values cannot be changed by later accuracy.
- Downstream calibration performs the registered omega grid, Stage-2 LR grid,
  local-neighbor interaction check, and at most one upper-bound expansion.
  One tuple is shared by all six transfers in each track-budget cell.
- Only GPUs 0, 1, 2, 3, and 4 are allowed. The default is one job per GPU;
  completed slots are refilled immediately. Each failed identity may be
  retried once with identical scientific settings.
- These artifacts are tuning evidence only. They must not pass the formal
  COME-LBI gate and must not be used as paper-final runs.

The task stops after exactly six Stage-1 tuples and six calibrated Office
tuples are frozen and the required summaries are written. It does not run
fresh formal LBI, VisDA, or any new baseline.
