# Office Night Machine 2 — Baselines D→A/D→W + LBI Stage-1 rho=0.001

Project directory:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined
```

Conda environment:

```text
SHOT_TTA
```

Normative protocol:

```text
protocol/shot-otta_fc/OTTA_FC_LBI_PROTOCOL_20260817_v1.md
```

Modes:

```text
MODE=PREPARE
MODE=FINALIZE
```

**Complete only the requested mode.**

Only modify files under:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/
```

Never modify `nips2026/` or any other historical directory.

Do not change the frozen scientific implementation or protocol settings.

Frozen revisions:

```text
IMPLEMENTATION_REVISION = iclr2027_refined_20260817_v1
EFFICIENCY_PROTOCOL_REVISION = otta_fc_batch_efficiency_20260817_v1
SOURCE_CHECKPOINT_REVISION = nips2026_shot_otta_uda_source_v1
```

This machine has 8 GPUs:

```text
0,1,2,3,4,5,6,7
```

This prompt is only for:

```text
Machine tag       : machine2
Baseline transfers: D->A, D->W
LBI Stage-1 budget: 0.001
Integer K         : 524
```

The LBI Stage-1 search must still evaluate every anchor on all six Office transfers.

Use the shared experiment root:

```text
experiment_logs/shot_otta_office_seed2026_stage1_20260818
```

Use machine-specific plan/log/record paths so three machines never overwrite one another.

---

# MODE=PREPARE

**Do not launch real training.**

The purpose of PREPARE is to validate prerequisites, generate the two plans, generate exactly one non-interactive `.sh`, and record reproducibility metadata.

Do not summarize nonexistent results and do not perform FINALIZE work.

## 1. Verify frozen prerequisites

Read and verify:

```text
protocol/shot-otta_fc/OTTA_FC_LBI_PROTOCOL_20260817_v1.md
configs/otta_fc_lbi_protocol_20260817_v1.yaml
experiments/shot_otta_fc_lbi_formal_20260817_v1.yaml
```

Verify at least:

```text
dataset=office
backbone=resnet50
batch_size=64
workers=4
seed=2026

SHOT:
cls_par=0.3
ent_par=1.0
threshold=0.0
lr=0.01
momentum=0.9
weight_decay=0.001
nesterov=true
lr_gamma=10.0
lr_power=0.75

source checkpoint root:
.../nips2026/SHOT-OTTA/ckpt/source
da=uda

FC candidate count=524544
support_threshold=1e-4
stage1_max_steps=3000
stage2_steps=1
budget_tolerance=1e-4 compatibility-only
```

Also verify the current `lbi: null` config plumbing fix:

- unresolved formal LBI still fails closed;
- a fully resolved CLI/planner LBI tuple can resolve successfully;
- a resolved `module_lbi` command with `--dry-run` returns code 0.

If any of these checks fail, stop PREPARE without generating a launchable script.

## 2. Prepare this machine's formal baseline shard

Create a machine-specific baseline matrix/spec and plan under the shared experiment root.

This machine must contain exactly these two transfers:

```text
D->A
D->W
```

Per transfer include exactly:

```text
source_only       x 1
full_dense        x 1
module_dense      x 1

module_random     x budgets 0.0005, 0.001, 0.002
module_magnitude  x budgets 0.0005, 0.001, 0.002
module_saliency   x budgets 0.0005, 0.001, 0.002
```

Do **not** include `module_lbi` in the baseline plan.

Required counts:

```text
12 top-level identities / transfer

source_only       2
full_dense        2
module_dense      2
module_random     6
module_magnitude  6
module_saliency   6
-------------------
total             24
```

Random remains one top-level identity per transfer×budget and internally evaluates 3 deterministic masks.

The baseline plan must use:

```text
seed=2026
save_model=false
no stream checkpoint
```

Generate the plan using the existing planner. Do not handwrite training commands outside the planner/identity system.

Run planner dry-run and launcher dry-run only.

For the launcher dry-run use formal efficiency scheduling:

```text
--gpus 0,1,2,3,4,5,6,7
--max-workers 8
--workers-per-gpu 1
```

Verify every to-be-run baseline command receives `runtime_comparable=true` through the launcher protocol.

## 3. Prepare this machine's LBI Stage-1 anchor plan

This machine owns only:

```text
rho=0.001
K=524
```

Search only Stage-1 dynamics:

```text
alpha / kappa / nu
```

Use exactly these 8 anchors:

| ID | alpha | kappa | nu |
|---|---:|---:|---:|
| A1 | 0.10 | 1.0 | 0.50 |
| A2 | 0.15 | 1.0 | 0.50 |
| A3 | 0.20 | 1.0 | 0.50 |
| A4 | 0.10 | 1.5 | 0.50 |
| A5 | 0.10 | 2.0 | 0.50 |
| A6 | 0.10 | 1.0 | 0.25 |
| A7 | 0.10 | 1.0 | 1.00 |
| A8 | 0.15 | 1.0 | 1.00 |

Each anchor must run all six Office transfers:

```text
A->D
A->W
D->A
D->W
W->A
W->D
```

Therefore the plan must contain exactly:

```text
8 anchors × 6 transfers = 48 experiments
```

Every Stage-1 entry must contain a fully resolved LBI tuple.

Freeze all non-Stage1-tuning values:

```text
requested_budget=0.001
omega=0.2
stage1_max_steps=3000
budget_tolerance=1e-4
stage2_lr=0.02
stage2_steps=1
delta_nonzero_tolerance=1e-12
support_threshold=1e-4
batch_size=64
workers=4
seed=2026
variant=module_lbi
save_model=false
```

`budget_tolerance=1e-4` is metadata only; do not relax the strict:

$$
K=\lfloor \rho N_{\mathrm{FC}}\rfloor
$$

upper bound.

Although this phase tunes only `alpha/kappa/nu`, each online batch must still execute the frozen Stage-2 step and `omega=0.2` persistent writeback. Do not create a Stage1-only altered training algorithm.

Run planner dry-run and launcher dry-run only.

For the Stage-1 launcher dry-run use:

```text
--gpus 0,1,2,3,4,5,6,7
--max-workers 16
--workers-per-gpu 2
--resume-partial-runs
```

Stage-1 tuning runtime is not formal efficiency evidence, so `runtime_comparable` must be false.

## 4. Create exactly one non-interactive shell

Create:

```text
tools/run_office_night_machine2_noninteractive.sh
```

The shell must be suitable for a non-interactive batch runner.

Required shell behavior:

```bash
#!/usr/bin/env bash
set -euo pipefail
```

It must:

- run in the foreground;
- use `conda run --no-capture-output -n SHOT_TTA`;
- not use tmux/screen/nohup;
- not detach;
- not use background `&`;
- remain alive until both launch phases finish;
- return non-zero if either launcher fails;
- keep baseline and Stage-1 launcher logs separate;
- append all real commands to machine-specific command history.

The shell's real execution sequence must be exactly:

```text
1. print environment / paths / machine responsibility
2. run baseline 24-entry plan
   - 8 GPUs
   - 8 max workers
   - 1 worker/GPU
3. only after baseline launcher returns successfully:
   run Stage-1 48-entry plan
   - 8 GPUs
   - 16 max workers
   - 2 workers/GPU
   - resume_partial_runs enabled
4. print that training phases have returned
5. STOP
```

The `.sh` must **not** run FINALIZE aggregation.

The `.sh` must **not** automatically start:

```text
omega search
stage2_lr search
final LBI rerun
```

Do not use `exec` for the first launcher because the shell must continue to the Stage-1 launcher.

## 5. Reproducibility record

Write:

```text
experiment_logs/shot_otta_office_seed2026_stage1_20260818/phase_records/machine2/PREPARE.md
```

Record:

- actual preparation commands;
- conda environment;
- GPU allocation;
- protocol/implementation/source/efficiency revisions;
- baseline transfers;
- Stage-1 budget and integer K;
- all 8 anchors;
- fixed LBI settings;
- matrix paths;
- plan paths;
- 24 baseline identities/hashes;
- 48 Stage-1 identities/hashes;
- launcher dry-run allocation;
- real launcher shell path;
- runs/logs/command-history paths;
- dry-run results.

Do not launch training in PREPARE.

## 6. PREPARE validation

Run only lightweight/static checks:

```text
bash -n tools/run_office_night_machine2_noninteractive.sh
planner dry-run
launcher dry-run
resolved LBI train --dry-run
plan-count checks
git diff --check
relevant lightweight tests if needed
```

Do not run actual training.

At the end of PREPARE print exactly these two user commands:

```bash
cd /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined

bash tools/run_office_night_machine2_noninteractive.sh
```

Then stop.

---

# MODE=FINALIZE

Run FINALIZE **only after the user has submitted the generated `.sh` and the non-interactive launcher has returned**.

Do **not** launch, retry, resume, or rerun any training in FINALIZE.

Do not edit scientific configs or training algorithms.

## 1. Baseline completion check

Load this machine's 24-entry baseline plan.

Using existing status tooling, report at least:

```text
completed
missing
failed
duplicate
hash mismatch
invalid summary
```

Expected success state:

```text
completed=24
missing=0
duplicate=0
hash mismatch=0
invalid summary=0
```

If any planned baseline run is incomplete or invalid:

- report exact identities and paths;
- do not rerun them;
- continue only with summaries that are scientifically safe;
- clearly mark baseline shard incomplete.

Create a machine-local baseline summary containing PU, FO and available formal efficiency aggregates for:

```text
D->A
D->W
```

Do not use any old seed-2020 baseline result.

## 2. Stage-1 completion check

Load the 48-entry Stage-1 plan.

Report:

```text
completed
missing
failed
duplicate
hash mismatch
invalid summary
```

Do not rerun anything.

If the 48-run set is incomplete, report exact missing/failed identities and stop before declaring a selected Stage-1 winner.

## 3. Compute budget-utilization diagnostics from real per-batch records

For every completed Stage-1 run, inspect its real per-batch metrics.

Define:

$$
u_t=\frac{\mathrm{selected\_count}_t}{524}
$$

and:

$$
budget\_gap_t=524-\mathrm{selected\_count}_t.
$$

Scientific safety requires:

```text
no NaN/Inf/error
run complete
selected_count <= K for every batch
no budget violation
no Stage-1 max-steps failure
```

Hard utilization gate:

$$
u_t\ge0.90\quad\text{for every batch}
$$

95% is a strong preference, not a hard failure gate.

For every run report at least:

```text
support_utilization_mean
support_utilization_min
support_utilization_p05

under_95pct_batch_count
under_90pct_batch_count

mean_budget_gap
max_budget_gap

rollback_count
rollback_rate

stage1_steps_mean
stage1_steps_max
stage1_max_steps_hit_count

PU
FO
```

Do not fabricate missing metrics. Derive them from batch records.

## 4. Aggregate each of the 8 candidates over all six transfers

Create a candidate table with one row per anchor.

At minimum report:

```text
candidate_id
alpha
kappa
nu
omega
stage2_lr

completed_transfer_count
scientific_valid_transfer_count
hard_90pct_valid_transfer_count

global_min_utilization
aggregate_p05_utilization
aggregate_mean_utilization
under_95pct_batch_count
under_90pct_batch_count

mean_budget_gap
max_budget_gap
rollback_rate

stage1_steps_mean
stage1_steps_max
max_steps_hit_count

mean_PU
mean_FO
```

If the full new seed-2026 baseline set from all three machines is already present and complete, compute for each transfer:

```text
best_sparse_FO =
max(Random 3-mask mean, Magnitude, Saliency)
```

and add:

```text
mean_FO_margin
worst_transfer_FO_margin
```

If the full 72-baseline set is not yet complete:

```text
baseline_reference_complete=false
```

Do not wait indefinitely and do not use historical/seed-2020 results as a replacement.

## 5. Stage-1 ranking

PU is report-only.

Rank candidates in this order:

```text
1. scientific_valid_transfer_count descending
2. hard_90pct_valid_transfer_count descending
3. global_min_utilization descending
4. aggregate_p05_utilization descending
5. aggregate_mean_utilization descending
6. under_95pct_batch_count ascending
7. max_budget_gap ascending
8. mean_budget_gap ascending
9. if new baseline reference is complete:
      mean_FO_margin descending
10. if new baseline reference is complete:
      worst_transfer_FO_margin descending
11. mean_FO descending
12. stage1_steps_max ascending
13. stage1_steps_mean ascending
```

Do not use PU as a ranking or tie-break field.

Do not automatically promote a candidate that violates the 90% hard utilization gate on any batch simply because its FO is high.

If no candidate satisfies the hard gate across all six transfers, explicitly conclude:

```text
No Stage-1 configuration is ready for omega tuning.
```

and stop. Do not start another search automatically.

## 6. FINALIZE outputs

Write machine/budget-specific outputs under:

```text
experiment_logs/shot_otta_office_seed2026_stage1_20260818/
```

At least create:

```text
results/baseline/machine2/
results/lbi_stage1/budget_001/

reports/machine2/baseline_summary.csv
reports/machine2/baseline_summary.json
reports/machine2/baseline_summary.md

reports/machine2/stage1_per_run.csv
reports/machine2/stage1_candidate_ranking.csv
reports/machine2/stage1_candidate_ranking.json
reports/machine2/stage1_candidate_ranking.md
reports/machine2/stage1_selection.json
```

`stage1_selection.json` must state whether a candidate is ready to advance to omega tuning, but FINALIZE must not start omega tuning.

Also write:

```text
experiment_logs/shot_otta_office_seed2026_stage1_20260818/phase_records/machine2/FINALIZE.md
```

Record:

- all status commands;
- plan/run identities;
- result/log paths;
- completion state;
- utilization definition;
- validity decisions;
- candidate ranking;
- reasons for eliminating candidates;
- selected/preferred Stage-1 candidate if any;
- whether the 72-baseline reference was complete;
- next recommended action.

## 7. FINALIZE hard stop

Do not launch:

```text
omega tuning
Stage-2 LR tuning
final LBI formal runs
baseline retries
LBI retries
```

Finish by clearly stating:

```text
FINALIZE complete for machine2 / budget 0.001.
No additional training was launched.
```

Then stop.
