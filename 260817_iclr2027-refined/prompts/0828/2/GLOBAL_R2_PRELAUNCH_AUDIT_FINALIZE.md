# Codex Prompt — Office Conv-LBI Out-channel R2 Global Audit / FINALIZE

Project:
`/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined`

Conda:
`SHOT_TTA`

Normative:
```text
protocol/shot-otta_conv/OTTA_CONV_LBI_PROTOCOL_20260826_v1.md
protocol/shot-otta_conv/OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v1.md
protocol/shot-otta_conv/OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2.md
```

Frozen R1:
```text
experiment_logs/office_conv_lbi_r1_out_stage1_validation_seed2026_20260827/
  selected_configs/OFFICE_OUT_STAGE1_FROZEN.json
  phase_records/R1/FINALIZE.md
```

R2 branch:
```text
experiment_logs/office_conv_lbi_r2_out_joint_sweep_seed2026_20260828
```

R2 control:
```text
experiment_logs/conv_lbi_r2_out_32gpu_20260828
```

This prompt has two modes:
```text
MODE=PRELAUNCH_AUDIT
MODE=FINALIZE
```

Neither mode may launch training, retry, resume, rerun, or probe GPUs.

## Frozen R2 matrix

Initial R2 grid:
```text
omega     ∈ {.00625, .025, .10}
stage2_lr ∈ {.0025, .005, .010}
```

Frozen Stage-1:
```text
.0005 / K_G=4  -> A1 alpha=.05 nu=.50
.001  / K_G=9  -> A1 alpha=.05 nu=.50
.002  / K_G=18 -> A4 alpha=.10 nu=1.00
```

Eight NEW cells:
```text
Node0  omega=.00625  lr=.0025
Node1  omega=.00625  lr=.010
Node2  omega=.025    lr=.0025
Node3  omega=.025    lr=.005
Node4  omega=.025    lr=.010
Node5  omega=.10     lr=.0025
Node6  omega=.10     lr=.005
Node7  omega=.10     lr=.010
```

One REUSED cell:
```text
omega=.00625, lr=.005
```

The reused cell is exactly the already-completed R0/R1 evidence using the R1-frozen Stage-1
anchor per budget:
```text
18 rows = 3 budgets × 6 transfers
```

For the reused cell, locate and validate exactly:
- D→A under `experiment_logs/office_conv_lbi_r0_reachability_seed2026_20260827`;
- the other five transfers under
  `experiment_logs/office_conv_lbi_r1_out_stage1_validation_seed2026_20260827`.

Do not assume a path beyond those roots. Match by scientific configuration:
dataset/transfer/seed/variant/group/budget/alpha/nu/kappa/omega/stage2_lr and completed
summary SHA metadata.

# MODE=PRELAUNCH_AUDIT

Run this only after all 8 node PREPARE steps have completed and before launching the 8 nodes.

1. Read all:
```text
experiment_logs/conv_lbi_r2_out_32gpu_20260828/node_records/node0_PREPARE.md
...
experiment_logs/conv_lbi_r2_out_32gpu_20260828/node_records/node7_PREPARE.md
```
and all eight JSONL plans.
2. Audit the union of NEW rows:
   - exactly 144 rows;
   - exactly 144 unique experiment keys;
   - exactly 144 unique scientific SHA256 values;
   - exactly 144 unique expected output roots;
   - 18 rows per node;
   - exact 8-cell mapping above;
   - no `.00625/.005` central-cell row;
   - each cell has 3 budgets × 6 transfers;
   - exact budget/K and frozen Stage-1 tuple;
   - no filter, baseline, VisDA, or `.005` budget.
3. Audit cell-specific `--output-root` prefixes are eight distinct directories.
4. Audit all launchers use only GPUs 0-3, one process/GPU, max-workers=4,
   workers-per-gpu=1, and `conda run --no-capture-output -n SHOT_TTA`.
5. Audit `bash -n` for all eight launchers.
6. Separately locate the 18 reused central-cell artifacts and confirm:
   - exactly 18 scientific identities;
   - completed and unique;
   - omega=.00625;
   - stage2_lr=.005;
   - correct R1-frozen Stage-1 tuple for each budget.
7. Write:
```text
experiment_logs/conv_lbi_r2_out_32gpu_20260828/GLOBAL_PRELAUNCH_AUDIT.md
```
8. End with exactly:
```text
GLOBAL_PRELAUNCH_AUDIT: PASS/FAIL
NEW_ROWS: X/144
REUSED_ROWS: X/18
GLOBAL_NEW_UNIQUE_KEYS: X/144
GLOBAL_NEW_UNIQUE_SHA: X/144
GLOBAL_NEW_UNIQUE_ROOTS: X/144
SAFE_TO_LAUNCH: YES/NO
```
9. Stop.

# MODE=FINALIZE

Run only after all eight node FINALIZE records exist.

No launch/retry/resume/rerun.

## A. Completeness

1. Read Node0-Node7 FINALIZE records and all underlying summaries/metrics.
2. Verify:
```text
144/144 NEW rows completed
18/18 REUSED central rows available
162 total initial-grid scientific rows
```
3. Re-audit key/SHA/output uniqueness and scientific config.

## B. Per-candidate aggregation

For every:
```text
budget × omega × stage2_lr
```
there must be exactly six Office transfers.

Candidate scientific eligibility requires:
- scientific-valid on all six transfers;
- utilization-eligible on all six transfers.

Out-channel per-transfer utilization gate:
```text
mean(u_t) >= .90
p05(u_t) >= .75
```

PU is report-only and must not select the candidate.

For every candidate report:
```text
budget / K_G
alpha / nu
omega
stage2_lr
six-transfer validity
six-transfer utilization eligibility
FO on AD/AW/DA/DW/WA/WD
FO six-transfer macro
worst-transfer FO
PU six-transfer macro
worst Stage-1 max steps
six-transfer mean Stage-1 steps
mean online runtime
```

## C. Ranking

Within each budget rank eligible candidates exactly by:
1. FO six-transfer macro descending;
2. worst-transfer FO descending;
3. Stage-1 max steps ascending;
4. Stage-1 mean steps ascending;
5. online runtime ascending.

Do not use PU for selection.

## D. Boundary rule

The initial-grid winner is on an upper boundary if:
```text
omega = .10
OR
stage2_lr = .010
```

If a budget winner is on an upper boundary:
- mark that budget `EXPANSION_REQUIRED`;
- do NOT freeze its final tuple yet;
- do NOT launch the expansion;
- do NOT invent the expansion rows beyond the frozen v1 rule;
- record which boundary dimension(s) triggered and the winning cell.

If no upper boundary is hit for a budget:
- mark it `FROZEN_NO_EXPANSION`;
- freeze its selected out-channel tuple.

Write:
```text
experiment_logs/office_conv_lbi_r2_out_joint_sweep_seed2026_20260828/selected_configs/OFFICE_OUT_R2_SELECTION.json
```

If at least one budget requires expansion, also write:
```text
experiment_logs/office_conv_lbi_r2_out_joint_sweep_seed2026_20260828/selected_configs/R2_BOUNDARY_EXPANSION_REQUEST.json
```

The expansion request must contain only the evidence needed to build the next predeclared local
extension prompt; it must not launch or silently generate extra experiments.

## E. Final report

Write:
```text
experiment_logs/office_conv_lbi_r2_out_joint_sweep_seed2026_20260828/phase_records/R2/FINALIZE.md
```

End with exactly:
```text
R2_INITIAL_GRID_COMPLETE: YES/NO
NEW_ROWS: X/144
REUSED_ROWS: X/18
ELIGIBLE_CANDIDATES_p0005: X/9
ELIGIBLE_CANDIDATES_p001: X/9
ELIGIBLE_CANDIDATES_p002: X/9
WINNER_p0005: omega=... stage2_lr=... status=...
WINNER_p001: omega=... stage2_lr=... status=...
WINNER_p002: omega=... stage2_lr=... status=...
BOUNDARY_EXPANSION_REQUIRED: YES/NO
NEXT_PHASE: R2_BOUNDARY_EXPANSION or OUT_TUPLE_FREEZE
```

Then stop.
