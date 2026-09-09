# Codex Prompt — Office Conv-LBI Out-channel R2 One-Time Boundary Expansion
## PREPARE / FINALIZE
## 24 new runs, 4 GPUs, one final expansion only

Project:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined
```

Conda environment:

```text
SHOT_TTA
```

This prompt is the **separate predeclared local-expansion prompt** required by:

```text
experiment_logs/office_conv_lbi_r2_out_joint_sweep_seed2026_20260828/
  selected_configs/R2_BOUNDARY_EXPANSION_REQUEST.json
```

It authorizes exactly the missing boundary-neighbor rows specified below.

Do not reopen the Cartesian grid.

---

# 0. Normative inputs

Read and verify before doing anything:

```text
protocol/shot-otta_conv/OTTA_CONV_LBI_PROTOCOL_20260826_v1.md
protocol/shot-otta_conv/OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v1.md
protocol/shot-otta_conv/OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2.md
```

Initial R2 artifacts:

```text
experiment_logs/office_conv_lbi_r2_out_joint_sweep_seed2026_20260828/
  phase_records/R2/FINALIZE.md
  selected_configs/OFFICE_OUT_R2_SELECTION.json
  selected_configs/R2_BOUNDARY_EXPANSION_REQUEST.json
```

Frozen Stage-1 source:

```text
experiment_logs/office_conv_lbi_r1_out_stage1_validation_seed2026_20260827/
  selected_configs/OFFICE_OUT_STAGE1_FROZEN.json
```

Formal Conv sparse baseline reference for report-only comparison:

```text
experiment_logs/conv_baseline_formal_global_20260827/
  FINALIZE.md
  reports/OFFICE_ALL_CONDITIONS.csv
  reports/OFFICE_MACRO_PU_FO.csv
```

Scientific implementation:

```text
iclr2027_refined_conv_20260826_v1
```

---

# 1. Frozen protocol facts

The v1 search protocol allows exactly one upper-bound expansion only if the initial R2 winner has:

```text
omega = .10
or
stage2_lr = .010
```

Permitted upper extensions:

```text
omega -> .30
stage2_lr -> .020
```

Only missing boundary-neighbor rows around the current winner may be added.

After this expansion FINALIZE:

```text
STOP even if the winner is still on a boundary.
NO omega > .30.
NO stage2_lr > .020.
NO lower-bound expansion.
NO new Stage-1 anchors.
NO second boundary expansion.
```

The v2 amendment also freezes the current branch state:

```text
out_channel continues normally
filter_connection = BLOCKED_V2
```

Do not fabricate or infer any Filter-LBI tuple.

---

# 2. Initial R2 result that must be verified

`R2_BOUNDARY_EXPANSION_REQUEST.json` must contain exactly:

## Budget .0005

```text
rho_G = .0005
K_G = 4

Stage-1:
anchor = A1
alpha = .05
kappa = 1
nu = .50

initial winner:
omega = .10
stage2_lr = .010

triggered upper boundaries:
omega
stage2_lr
```

## Budget .001

```text
rho_G = .001
K_G = 9

Stage-1:
anchor = A1
alpha = .05
kappa = 1
nu = .50

initial winner:
omega = .10
stage2_lr = .0025

triggered upper boundary:
omega
```

## Budget .002

Initial R2 already froze:

```text
rho_G = .002
K_G = 18

Stage-1:
anchor = A4
alpha = .10
kappa = 1
nu = 1.00

omega = .025
stage2_lr = .0025
```

`.002` is NOT part of this expansion and must not be rerun or retuned.

Abort PREPARE if any of the above disagrees with the on-disk frozen artifacts.

---

# 3. Exact expansion grid

Run exactly four new cells.

| Cell | Budget | K_G | Anchor | alpha | nu | omega | stage2_lr |
|---|---:|---:|---|---:|---:|---:|---:|
| E0 | .0005 | 4 | A1 | .05 | .50 | .30 | .010 |
| E1 | .0005 | 4 | A1 | .05 | .50 | .10 | .020 |
| E2 | .0005 | 4 | A1 | .05 | .50 | .30 | .020 |
| E3 | .001 | 9 | A1 | .05 | .50 | .30 | .0025 |

Every cell runs all six Office transfers:

```text
AD = amazon(0) -> dslr(1)
AW = amazon(0) -> webcam(2)
DA = dslr(1)  -> amazon(0)
DW = dslr(1)  -> webcam(2)
WA = webcam(2)-> amazon(0)
WD = webcam(2)-> dslr(1)
```

Therefore:

```text
4 cells x 6 transfers = 24 new scientific rows
```

There are no other authorized rows.

Forbidden examples:

```text
.001 / omega=.30 / lr=.005
.001 / omega=.30 / lr=.010
.001 / omega=.30 / lr=.020
.002 / any new omega/LR
omega > .30
stage2_lr > .020
rho_G=.005
filter_connection
VisDA
baseline reruns
```

---

# 4. Scientific invariants

This phase changes only the authorized `omega` / `stage2_lr` values.

Keep frozen:

```text
dataset = office
method = shot
variant = conv_out_lbi
group_mode = out_channel
seed = 2026

kappa = 1
support_threshold = 1e-4
stage1_max_steps = 3000
stage2_steps = 1

strict integer global group budget
strict rollback
no LBI top-K trim
batch-local LBI restart
masked-delta Stage-2 initialization
off-mask gradient masking/restoration
controlled Conv BN frozen
SHOT objective and all data semantics
PU / FO semantics
```

Do not modify:

```text
Conv candidate layers
group partition
group prox
support semantics
Stage-1 equation
Stage-2 optimizer semantics
source checkpoint
batch size/workers
transforms
Office metric
```

This is tuning, not final efficiency:

```text
runtime_comparable = false
```

Do not change the scientific implementation revision.

---

# 5. Output root

Use a separate expansion root:

```text
experiment_logs/office_conv_lbi_r2_out_boundary_expansion_seed2026_20260828
```

Do not write new runs into the initial R2 root.

Required layout:

```text
experiment_logs/office_conv_lbi_r2_out_boundary_expansion_seed2026_20260828/
  plans/
  runs/
  launcher_logs/
  command_history/
  phase_records/
  selected_configs/
  reports/
```

Use four distinct cell-specific output roots because the repository's canonical suffix does not
by itself encode both `omega` and `stage2_lr`:

```text
runs/E0_p0005_omega_0p30_lr_0p010
runs/E1_p0005_omega_0p10_lr_0p020
runs/E2_p0005_omega_0p30_lr_0p020
runs/E3_p001_omega_0p30_lr_0p0025
```

Do not allow output collisions across cells.

---

# MODE=PREPARE

PREPARE is static only.

Do not launch training.
Do not probe/query/reserve GPUs.

## A. Verify frozen evidence

1. Read all three normative protocols.
2. Read initial R2 FINALIZE, selection JSON, and boundary request JSON.
3. Verify:

```text
R2_INITIAL_GRID_COMPLETE = YES
NEW_ROWS = 144/144
REUSED_ROWS = 18/18
```

4. Verify boundary request:

```text
status = EXPANSION_REQUIRED
permitted_upper_extensions.omega = .30
permitted_upper_extensions.stage2_lr = .020
```

5. Verify the exact budget/Stage-1/winner facts in Sections 2 and 3 above.
6. Verify `.002` is `FROZEN_NO_EXPANSION`.
7. Verify Office domain mapping in the repository:

```text
amazon=0
dslr=1
webcam=2
```

Abort on any mismatch.

## B. Build exact plans

Create four per-cell plans, each with exactly six rows:

```text
plans/E0_p0005_omega_0p30_lr_0p010.jsonl
plans/E1_p0005_omega_0p10_lr_0p020.jsonl
plans/E2_p0005_omega_0p30_lr_0p020.jsonl
plans/E3_p001_omega_0p30_lr_0p0025.jsonl
```

Use the existing R2 plan/config/hash construction code and the repository's canonical
scientific identity builder.

Do not invent a new identity rule.

`omega` and `stage2_lr` must participate in scientific identity/hash.

Before writing PREPARE, audit the union:

```text
rows = 24 exactly
unique experiment keys = 24
unique scientific SHA256 = 24
unique expected output roots = 24

E0 = 6 transfers exactly
E1 = 6 transfers exactly
E2 = 6 transfers exactly
E3 = 6 transfers exactly
```

Also verify:

```text
18 rows at rho=.0005
6 rows at rho=.001
0 rows at rho=.002

all .0005 rows:
  K_G=4
  alpha=.05
  nu=.50

all .001 rows:
  K_G=9
  alpha=.05
  nu=.50

all rows:
  kappa=1
  variant=conv_out_lbi
  group_mode=out_channel
  seed=2026
```

Reject any row outside the exact four-cell set.

## C. Launcher

Create:

```text
tools/run_conv_lbi_r2_boundary_expansion_4gpu.sh
```

Inspect the already-successful R2 launchers first and reuse their exact
`run_experiments_multi_gpu.py` CLI pattern.

Do not guess command-line flags.

Exact GPU ownership:

```text
GPU 0 -> E0 -> 6 transfers
GPU 1 -> E1 -> 6 transfers
GPU 2 -> E2 -> 6 transfers
GPU 3 -> E3 -> 6 transfers
```

Each GPU processes its six rows sequentially.

Execution policy:

```text
1 experiment process / GPU
4 experiment processes total
no GPU oversubscription
```

Every Python call inside the generated shell script must use:

```bash
conda run --no-capture-output -n SHOT_TTA ...
```

Do not require `conda activate`.

Reuse the existing exact completed-batch-boundary LBI resume behavior only.
Do not introduce a new resume policy.

Launcher must:

```text
set -euo pipefail
fail if PREPARE/plans are absent
use a dedicated lock
CPU-validate all four plans before launch
preserve command history
preserve launcher logs
wait for all four GPU jobs
propagate failure if any cell fails
```

## D. Static checks only

Run:

```text
plan validation
global key/SHA/output-root uniqueness check
bash -n tools/run_conv_lbi_r2_boundary_expansion_4gpu.sh
git diff --check
```

No training.
No GPU probe.

Write:

```text
experiment_logs/office_conv_lbi_r2_out_boundary_expansion_seed2026_20260828/
  phase_records/R2_BOUNDARY_EXPANSION/PREPARE.md
```

PREPARE must record:

```text
source protocol hashes/provenance
source R2 selection/request provenance
exact four authorized cells
24-row count
unique key/SHA/root counts
four plan SHA256 values
launcher path
all static checks
```

End PREPARE by printing only:

```bash
cd /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined
bash tools/run_conv_lbi_r2_boundary_expansion_4gpu.sh
```

Then stop.

---

# MODE=FINALIZE

FINALIZE is artifact-only.

Do not:

```text
launch
retry
resume
rerun
add a row
change a grid
change a protocol
```

## A. Audit the 24 expansion runs

Require:

```text
24/24 planned rows accounted for
no duplicate completed artifacts
experiment key matches plan
scientific SHA matches plan
scientific config matches exact authorized cell
```

For every run verify:

```text
completed
finite/error-free
selected_group_count <= K_G for every online batch
no support/mask/budget contract violation
controlled Conv BN remains frozen
```

Scientific-valid iff:

```text
completed
finite/error-free
selected_group_count <= K_G every batch
zero Stage-1 max-step failures
no support/mask/budget contract violation
```

Per-transfer out-channel utilization-eligible iff:

```text
mean utilization >= .90
p05 utilization >= .75
```

For every cell report:

```text
6-transfer scientific validity
6-transfer utilization eligibility
FO AD/AW/DA/DW/WA/WD
FO six-transfer macro
worst-transfer FO
PU six-transfer macro
worst Stage-1 max steps
six-transfer mean Stage-1 steps
mean Stage-1 runtime
mean online runtime
```

PU is report-only.

## B. Final ranking universe

Do not rank only the new cells in isolation.

For `.0005`, rank:

```text
all 9 initial R2 candidates
+
E0
E1
E2
=
12 candidates
```

For `.001`, rank:

```text
all 9 initial R2 candidates
+
E3
=
10 candidates
```

For `.002`:

```text
do not reopen ranking
preserve the already-frozen initial R2 winner:
omega=.025
stage2_lr=.0025
alpha=.10
nu=1.00
K_G=18
```

If all new expansion candidates for a budget are ineligible, retain the original initial-grid
winner for that budget.

Eligibility requires all six transfers to be:

```text
scientific-valid
and
utilization-eligible
```

Rank eligible candidates exactly by:

1. FO six-transfer macro descending;
2. worst-transfer FO descending;
3. Stage-1 max steps ascending;
4. Stage-1 mean steps ascending;
5. online runtime ascending.

Do not use PU for selection.

## C. Mandatory stop rule

After selecting the final `.0005` and `.001` winners:

```text
STOP.
```

Even if the final winner has:

```text
omega=.30
or
stage2_lr=.020
```

do not create another expansion request.

Do not run:

```text
omega=.4/.5/...
stage2_lr>.020
```

This one-time expansion closes Office out-channel omega/LR tuning.

## D. Same-budget sparse-baseline comparison — report only

Use the frozen formal Conv baseline artifacts.

For each budget separately, compare the final Out-LBI winner only with conventional sparse
baselines at the **same rho_G**:

```text
conv_out_random
conv_out_magnitude
conv_out_saliency
conv_filter_random
conv_filter_magnitude
conv_filter_saliency
```

Compute both:

### 1. Best fixed sparse method at the same budget

For each method:

```text
mean FO over the same six transfers
```

then:

```text
best_fixed_sparse_macro(rho)
=
max over the six conventional sparse methods
```

### 2. Same-budget per-transfer sparse oracle

For each transfer independently:

```text
max FO over the six conventional sparse methods at that same rho
```

then average the six maxima.

Do not mix budgets.

Expected frozen baseline references, to be verified from the CSV rather than blindly copied:

| Budget | Best fixed sparse macro | Same-budget per-transfer sparse oracle |
|---:|---:|---:|
| .0005 | 77.7139085 | 77.8212983 |
| .001 | 77.7822046 | 77.9664636 |
| .002 | 77.8361227 | 77.9351772 |

For every final Out-LBI budget report:

```text
final LBI FO macro
delta vs same-budget best fixed sparse
delta vs same-budget per-transfer sparse oracle
```

These baseline margins are report-only and must not retroactively select hyperparameters.

## E. Final outputs

Write:

```text
experiment_logs/office_conv_lbi_r2_out_boundary_expansion_seed2026_20260828/
  selected_configs/OFFICE_OUT_LBI_FINAL_TUPLES.json
  selected_configs/OFFICE_OUT_LBI_FINAL_TUPLES.md
  reports/OFFICE_OUT_R2_FINAL_COMPARISON.csv
  reports/OFFICE_OUT_R2_FINAL_COMPARISON.json
  reports/OFFICE_OUT_R2_FINAL_COMPARISON.md
  phase_records/R2_BOUNDARY_EXPANSION/FINALIZE.md
```

`OFFICE_OUT_LBI_FINAL_TUPLES.*` must contain exactly three **out-channel** tuples:

```text
rho=.0005
rho=.001
rho=.002
```

Do NOT create a fake six-tuple `OFFICE_CONV_LBI_FINAL_TUPLES.*` because
`filter_connection` remains `BLOCKED_V2`.

Record explicitly:

```text
filter_connection_status = BLOCKED_V2
filter_tuple_fabricated = false
```

## F. FINALIZE footer

End exactly with:

```text
R2_BOUNDARY_EXPANSION_COMPLETE: YES/NO
NEW_ROWS: X/24
VALID_CELLS_p0005: X/3
VALID_CELLS_p001: X/1
FINAL_p0005: alpha=... nu=... omega=... stage2_lr=... FO=...
FINAL_p001: alpha=... nu=... omega=... stage2_lr=... FO=...
FINAL_p002: alpha=.10 nu=1.00 omega=.025 stage2_lr=.0025 FO=...
SAME_BUDGET_ORACLE_MARGIN_p0005: ...
SAME_BUDGET_ORACLE_MARGIN_p001: ...
SAME_BUDGET_ORACLE_MARGIN_p002: ...
SECOND_BOUNDARY_EXPANSION_ALLOWED: NO
FILTER_STATUS: BLOCKED_V2
OFFICE_OUT_TUNING_CLOSED: YES/NO
NEXT_PHASE: OUT_VISDA_V0
```

Then stop.

Do not launch VisDA V0 from this prompt.
