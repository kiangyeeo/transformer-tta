# Codex Prompt — Conv Out-LBI VisDA V0 Stage-1 Diagnostic

This prompt controls **one 4-GPU machine only**. Do not execute another node's rows.

Project root:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined
```

Conda environment:

```text
SHOT_TTA
```

Scientific implementation revision:

```text
iclr2027_refined_conv_20260826_v1
```

Normative protocol files — read them first:

```text
protocol/shot-otta_conv/OTTA_CONV_LBI_PROTOCOL_20260826_v1.md
protocol/shot-otta_conv/OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v1.md
protocol/shot-otta_conv/OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2.md
```

Frozen Office Out result — read and verify:

```text
experiment_logs/office_conv_lbi_r2_out_boundary_expansion_seed2026_20260828/
  phase_records/R2_BOUNDARY_EXPANSION/FINALIZE.md
  selected_configs/OFFICE_OUT_LBI_FINAL_TUPLES.json
```

The Office artifacts must say:

```text
OFFICE_OUT_TUNING_CLOSED = YES
NEXT_PHASE = OUT_VISDA_V0
FILTER_STATUS = BLOCKED_V2
SECOND_BOUNDARY_EXPANSION_ALLOWED = NO
```

The frozen Office **Stage-1 anchors** used to seed VisDA V0 must be exactly:

| rho_G | K_G | alpha_office | kappa | nu_office |
|---:|---:|---:|---:|---:|
| .0005 | 4  | .05 | 1 | .50 |
| .001  | 9  | .05 | 1 | .50 |
| .002  | 18 | .10 | 1 | 1.00 |

Abort if the on-disk frozen artifacts disagree.

## V0 protocol

This is **out_channel only**. `filter_connection` remains `BLOCKED_V2` and must not appear in any VisDA V0 plan.

For each budget, the complete predeclared V0 candidate set is:

```text
(alpha_office,       nu_office)
(0.5*alpha_office,   nu_office)
(2.0*alpha_office,   nu_office)
(alpha_office,       0.5*nu_office)
(alpha_office,       2.0*nu_office)
```

Fixed for every V0 row:

```text
dataset = VISDA-C
source = 0
target = 1
seed = 2026
method = shot
variant = conv_out_lbi
group_mode = out_channel
kappa = 1
omega = .003125
stage2_lr = .005
support_threshold = 1e-4
stage1_max_steps = 3000
stage2_steps = 1
runtime_comparable = false
```

Run the **full VisDA target stream**. Do not truncate batches.

Scientific validity:

```text
completed
finite / error-free
selected_group_count <= K_G on every online batch
zero Stage-1 max-step failures
no support / mask / budget contract violation
```

Out-channel utilization eligibility:

```text
mean(selected_group_count / K_G) >= .90
p05(selected_group_count / K_G)  >= .75
```

V0 Stage-1 ranking is **not accuracy-based**. The later global V0 ranking is exactly:

1. scientific-valid;
2. utilization-eligible;
3. p05 utilization descending;
4. mean utilization descending;
5. Stage-1 max steps ascending;
6. Stage-1 mean steps ascending;
7. Stage-1 runtime ascending.

PU/FO may be reported but must not select the V0 Stage-1 anchor.

## Strict code safety

This VisDA V0 task must **not modify tracked source code**.

Read-only:

```text
core/**
shot_otta/**
configs/**
protocol/**
tests/**
tools/**
train.py
```

Do not commit, merge, rebase, checkout, stash, reset, clean, pull, or push.

You may create files only under:

```text
experiment_logs/visda_conv_lbi_v0_out_stage1_seed2026_20260828/
```

Use existing repository config resolution, scientific identity, and launcher code. Do not invent a new scientific identity format.

Before writing node artifacts, require:

```bash
git -C /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI status --short
```

to show no tracked working-tree changes. If tracked source changes already exist, STOP and report them.

## Execution discipline

Supported modes for this same prompt:

```text
MODE=PREPARE
MODE=NODE_FINALIZE
```

PREPARE is static only: create/verify this node's plan and launcher, but do not train and do not probe GPUs.

NODE_FINALIZE is artifact-only: aggregate this node's already-existing runs; do not launch, retry, resume, rerun, or change the matrix.

All Python execution inside generated launchers must use:

```bash
conda run --no-capture-output -n SHOT_TTA ...
```

Use one experiment process per GPU for this burst.


# This machine: NODE 3

Node-specific root:

```text
experiment_logs/visda_conv_lbi_v0_out_stage1_seed2026_20260828/node3/
```

This node owns exactly these 3 rows:

| ID | GPU | rho_G | K_G | alpha | nu |
|---|---:|---:|---:|---:|---:|
| V0022 | 0 | 0.002 | 18 | 0.2 | 1 |
| V0023 | 1 | 0.002 | 18 | 0.1 | 0.5 |
| V0024 | 2 | 0.002 | 18 | 0.1 | 2 |

No other V0 rows are authorized on this machine.

## MODE=PREPARE

1. Read and verify all normative inputs above.
2. Verify this node's rows are an exact subset of the complete 15-row V0 matrix.
3. Create only under the node-specific root:

```text
plan.jsonl
PREPARE.md
run_node3_4gpu.sh
launcher_logs/
command_history/
runs/
```

4. Inspect a successful recent Conv plan/launcher and reuse the repository's canonical effective-config/scientific-identity construction. Do not guess CLI flags.
5. Every plan row must include the correct `requested_budget`, `K_G`, `alpha`, `kappa`, `nu`, `omega`, `stage2_lr`, VisDA source/target, seed, variant, grouping mode, and full-stream settings.
6. Verify before stopping:

```text
planned rows = 3/3
unique experiment keys = 3/3
unique scientific SHA256 = 3/3
unique output roots = 3/3
all rows are exactly NODE 3's allowlisted matrix
runtime_comparable=false
```

7. Generated launcher must use GPUs `0,1,2` with exactly one worker/process per listed GPU. Do not use GPU IDs not assigned above.
8. Launcher must support the existing exact completed-batch-boundary resume semantics (`--resume` / `--resume-partial-runs` if that is the established launcher contract), but PREPARE itself must not launch.
9. Run static checks only:

```text
plan CPU validation
identity/output-root uniqueness validation
bash -n on node launcher
git diff --check
```

10. PREPARE must record source protocol/frozen tuple provenance and all static checks.
11. Print only:

```bash
cd /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined
bash experiment_logs/visda_conv_lbi_v0_out_stage1_seed2026_20260828/node3/run_node3_4gpu.sh
```

Then stop.

## MODE=NODE_FINALIZE

Audit only this node's existing 3 rows.

Require:

```text
completed artifacts = 3/3
no duplicate completed artifacts
experiment key/SHA/config match plan
full VisDA stream completed
```

For each row report validity, mean/p05/min utilization, fraction below .95, selected scalar count/ratio, Stage-1 mean/max steps/runtime, PU mAcc, FO mAcc, FO worst-class, FO class-wise std, FO overall.

Write only:

```text
experiment_logs/visda_conv_lbi_v0_out_stage1_seed2026_20260828/node3/NODE_FINALIZE.md
experiment_logs/visda_conv_lbi_v0_out_stage1_seed2026_20260828/node3/node_rows.csv
```

Do not select the final Stage-1 anchor inside node-local FINALIZE; global selection uses all five candidates for a budget.

End with:

```text
NODE3_COMPLETE: YES/NO
ROWS_COMPLETE: X/3
ROWS_SCIENTIFIC_VALID: X/3
ROWS_UTIL_ELIGIBLE: X/3
GLOBAL_FINALIZE_AUTHORIZED_HERE: YES_AFTER_ALL_NODES_COMPLETE
```


# MODE=GLOBAL_FINALIZE — NODE 3 ONLY

Node 3 is the designated global V0 aggregator, but GLOBAL_FINALIZE may run **only after all four node roots have completed**.

GLOBAL_FINALIZE is artifact-only. Do not launch/retry/resume/rerun anything.

Require the complete V0 matrix:

```text
15/15 unique scientific rows
rho=.0005 : 5/5
rho=.001  : 5/5
rho=.002  : 5/5
filter rows: 0
```

Audit experiment key and scientific SHA uniqueness, config identity, full-stream completion, validity, and utilization for all 15 rows.

For each budget, rank the five rows using only the frozen V0 ranking order above. Accuracy must not select Stage-1.

Freeze exactly one **out-channel VisDA Stage-1 anchor** per budget and write:

```text
experiment_logs/visda_conv_lbi_v0_out_stage1_seed2026_20260828/selected_configs/VISDA_OUT_STAGE1_FROZEN.json
experiment_logs/visda_conv_lbi_v0_out_stage1_seed2026_20260828/selected_configs/VISDA_OUT_STAGE1_FROZEN.md
experiment_logs/visda_conv_lbi_v0_out_stage1_seed2026_20260828/reports/VISDA_OUT_V0_ALL_ROWS.csv
experiment_logs/visda_conv_lbi_v0_out_stage1_seed2026_20260828/reports/VISDA_OUT_V0_ALL_ROWS.md
experiment_logs/visda_conv_lbi_v0_out_stage1_seed2026_20260828/phase_records/V0/FINALIZE.md
```

The frozen JSON/MD must record per budget:

```text
rho_G / K_G
alpha / kappa / nu
fixed V0 omega=.003125
fixed V0 stage2_lr=.005
scientific validity
mean/p05/min utilization
fraction batches below .95
Stage-1 mean/max steps
Stage-1 runtime
PU mAcc / FO mAcc / FO worst-class / FO overall as report-only diagnostics
```

If a budget has zero eligible candidates:

```text
V0_STATUS = BLOCKED
NEXT_PHASE = STOP_PROTOCOL_REVISION_REQUIRED
```

Do not widen the grid.

If all three budgets freeze successfully:

```text
V0_COMPLETE: YES
VISDA_OUT_STAGE1_FROZEN: 3/3
FILTER_STATUS: BLOCKED_V2
NEXT_PHASE: OUT_VISDA_V1
```

Do not launch V1 from this prompt.
