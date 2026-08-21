# Office LBI Final Joint Sweep — Machine 1

Project directory:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined
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

Never modify `nips2026/` or any historical directory.

Do not change frozen scientific behavior, protocol values, source checkpoints, seeds, metrics, BN semantics, sparse-budget semantics, or experiment-identity semantics.

Frozen revisions:

```text
IMPLEMENTATION_REVISION = iclr2027_refined_20260817_v1
EFFICIENCY_PROTOCOL_REVISION = otta_fc_batch_efficiency_20260817_v1
SOURCE_CHECKPOINT_REVISION = nips2026_shot_otta_uda_source_v1
```

This is part of the **final Office LBI hyperparameter-ablation wave**.

Shared sweep root:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/shot_otta_office_lbi_joint_sweep_20260818
```

Machine tag:

```text
machine1
```

This machine owns exactly these two Stage-1 anchor/budget pairs:

```text
budget=0.0005
K=262
anchor=A6
alpha=0.10
kappa=1.0
nu=0.25
```

```text
budget=0.0005
K=262
anchor=A1
alpha=0.10
kappa=1.0
nu=0.50
```

For each pair, sweep:

```text
omega ∈ {0.05, 0.10, 0.20, 0.30}
stage2_lr ∈ {0.005, 0.010, 0.020}
```

The already-completed reference point:

```text
omega=0.20
stage2_lr=0.020
```

must be reused from the completed Stage-1 experiment and must **not** appear in the new-run plan.

Therefore:

```text
11 new omega×LR configurations / anchor
6 Office transfers / configuration
66 new runs / anchor
132 new runs / machine
```

The full FINALIZE comparison for this machine is:

```text
132 new runs + 12 reused Stage-1 reference runs = 144 run rows
```

Office transfers:

```text
A->D
A->W
D->A
D->W
W->A
W->D
```

Formal seed:

```text
2026
```

Batch size / workers:

```text
64 / 4
```

---

# MODE=PREPARE

## Critical rule: PREPARE is pure planning only

The current machine used for PREPARE may have **no GPU at all**.

Therefore PREPARE must be completely hardware-agnostic.

**Do not do any GPU/runtime/environment execution checks.**

Specifically, in PREPARE do **not** run:

```text
nvidia-smi
CUDA checks
torch.cuda checks
GPU-count checks
GPU-memory checks
conda-run training commands
train.py --dry-run
launcher dry-run
real launcher
real training
```

Do not require the PREPARE machine to have:

```text
GPU
CUDA runtime
working NVIDIA driver
SHOT_TTA runtime activation
dataset mounted for training
source checkpoints mounted for training
```

PREPARE should only inspect repository files, generate matrix/plan/shell files, and perform static consistency checks.

The target execution machines will be separate 8-GPU machines. The generated shell may contain the intended future launcher arguments, but PREPARE must not test those GPUs.

## 1. Read frozen protocol/config/planner code statically

Read:

```text
protocol/shot-otta_fc/OTTA_FC_LBI_PROTOCOL_20260817_v1.md
configs/otta_fc_lbi_protocol_20260817_v1.yaml
experiments/shot_otta_fc_lbi_formal_20260817_v1.yaml
tools/plan_experiments.py
tools/run_experiments_multi_gpu.py
```

Statically verify the intended frozen Office settings from the files:

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

FC candidate count=524544
support_threshold=1e-4
stage1_max_steps=3000
stage2_steps=1
delta_nonzero_tolerance=1e-12
budget_tolerance=1e-4
```

Do not attempt to load model checkpoints or datasets in PREPARE.

Also statically inspect the current `lbi: null` override plumbing and existing regression tests. Do not execute training/config-resolution commands in PREPARE.

## 2. Verify prior Stage-1 references using existing artifacts only

For each owned anchor/budget pair, locate the existing Stage-1 result artifacts under:

```text
experiment_logs/shot_otta_office_seed2026_stage1_20260818
```

for all six transfers at:

```text
omega=0.20
stage2_lr=0.020
```

Use existing plan/status/summary JSON artifacts only.

Require exactly 6 completed scientific identities per pair with no ambiguity in:

```text
experiment_key
scientific hash
transfer
budget
alpha
kappa
nu
omega
stage2_lr
```

Do not execute any training or environment command to verify them.

Record the 12 reused reference identities and paths.

If the repository snapshot does not contain enough existing artifacts to prove these 12 references, stop PREPARE and report the missing artifacts. Do not silently put the reference point back into the new plan.

## 3. Build the joint matrix

Create machine-specific matrix/spec files under:

```text
experiment_logs/shot_otta_office_lbi_joint_sweep_20260818/matrices/machine1/
```

Owned pairs:

| Budget | K | Anchor | alpha | kappa | nu |
|---:|---:|---|---:|---:|---:|
| 0.0005 | 262 | A6 | 0.10 | 1.0 | 0.25 |
| 0.0005 | 262 | A1 | 0.10 | 1.0 | 0.50 |

For every new run freeze:

```text
variant=module_lbi
seed=2026
batch_size=64
workers=4
stage1_max_steps=3000
budget_tolerance=1e-4
stage2_steps=1
delta_nonzero_tolerance=1e-12
support_threshold=1e-4
save_model=false
```

Sweep exactly:

```text
omega = 0.05, 0.10, 0.20, 0.30
stage2_lr = 0.005, 0.010, 0.020
```

Exclude exactly:

```text
omega=0.20 AND stage2_lr=0.020
```

Do not exclude anything else.

Do not add new alpha/kappa/nu candidates.

## 4. Absolute canonical output root

All new runs must use exactly:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/shot_otta_office_lbi_joint_sweep_20260818/runs
```

Use this **absolute path** in the matrix/spec.

This is specifically to prevent recurrence of the previous bug where a relative `output_root` was resolved against `/PJ/Split-LBI` while the launcher used a refined-repo `--runs-root`.

## 5. Generate exactly one 132-entry plan

Generate:

```text
experiment_logs/shot_otta_office_lbi_joint_sweep_20260818/plans/machine1/plan.json
```

using the existing planner/scientific-identity code path.

Planner invocation is allowed only if it is a pure planning operation and does not initialize GPU/training/runtime dependencies.

If the existing planner unexpectedly requires GPU/runtime initialization, do not work around it by touching CUDA. Stop and report the blocker.

Hard static assertions:

```text
experiment_count = 132
unique experiment_key count = 132
duplicate experiment_key count = 0
```

Per owned pair:

```text
66 entries
11 omega×LR combinations
6 transfers each
```

Verify statically:

```text
only Office
only seed=2026
only module_lbi
only the two owned anchor/budget pairs
only allowed omega values
only allowed stage2_lr values
no omega=.20/stage2_lr=.020 entry
```

Run the planner twice if possible and verify deterministic experiment keys/hashes. This is a planning-only determinism check.

## 6. Mandatory static path-consistency validation

Do not invoke the launcher.

Instead, parse the generated `plan.json` directly.

For all 132 entries assert:

```text
expected_output_root is under the exact canonical RUNS_ROOT
expected_output_root is inside 260817_iclr2027-refined
expected_output_root is NOT under /PJ/Split-LBI/experiment_logs/
```

Also statically inspect the generated command/argv stored in the plan and verify its output-root-related arguments are consistent with the plan metadata.

If the plan format does not store enough information, add a small **pure static** validation helper under `tools/` or `tests/` that only parses JSON/paths and never imports CUDA/training/model code.

This path validator is mandatory because the previous real-run failure was caused by a path mismatch.

## 7. Create exactly one future non-interactive execution shell

Create:

```text
tools/run_office_lbi_joint_sweep_machine1_noninteractive.sh
```

This shell is for a **different future 8-GPU execution machine**.

The shell itself should target:

```text
GPUs: 0,1,2,3,4,5,6,7
max-workers: 16
workers-per-gpu: 2
resume-partial-runs: enabled
conda env: SHOT_TTA
```

But PREPARE must **not execute or GPU-test this shell**.

Required shell header:

```bash
#!/usr/bin/env bash
set -euo pipefail
```

The future user command will be:

```bash
cd /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined
bash tools/run_office_lbi_joint_sweep_machine1_noninteractive.sh
```

The shell must:

- remain foreground-attached;
- use the exact generated 132-entry plan;
- use the exact absolute canonical RUNS_ROOT;
- use machine-specific launcher logs and command history;
- enable LBI partial resume;
- not run FINALIZE;
- not start omega/LR searches beyond this predefined 132-run plan;
- not run final formal LBI evaluation.

### Launch-time guard inside the shell

It is fine for the **future execution shell itself**, when the user actually runs it on the GPU machine, to check its runtime environment.

However, the PREPARE step must not execute these checks.

Before launching children, the shell should verify:

```text
plan exists
plan SHA256 matches PREPARE_READY
plan count=132
unique keys=132
each owned pair count=66
all expected_output_root values under canonical RUNS_ROOT
no forbidden omega=.20/lr=.020 reference entry
current project path is correct
```

It may then rely on the actual target execution machine for conda/GPU availability.

### Anti-double-submit lock

Include an atomic machine-specific lock such as:

```text
experiment_logs/shot_otta_office_lbi_joint_sweep_20260818/locks/machine1.lock
```

to prevent accidental double submission of the same machine plan.

Use an atomic mechanism such as `mkdir`.

Record hostname/PID/start time.

Use a trap to remove the lock on normal exit/termination.

Do not remove another live process's lock automatically.

## 8. Static shell validation only

PREPARE may run:

```text
bash -n tools/run_office_lbi_joint_sweep_machine1_noninteractive.sh
```

because this is syntax-only.

Do not execute the shell.

Do not call `conda run`.

Do not call the launcher.

Do not call `train.py`.

## 9. PREPARE_READY

Only after all pure planning/static checks pass, write:

```text
experiment_logs/shot_otta_office_lbi_joint_sweep_20260818/phase_records/machine1/PREPARE_READY.json
```

containing at least:

```text
machine_tag
plan_path
plan_sha256
matrix/spec paths
canonical_runs_root
shell_path
experiment_count=132
unique_experiment_keys=132
pair counts=66+66
reference_run_count=12
static_path_check_passed=true
bash_syntax_passed=true
timestamp
```

Do not include claims such as:

```text
GPU check passed
CUDA check passed
training dry-run passed
launcher runtime check passed
```

because PREPARE does not test those things.

## 10. PREPARE record

Write:

```text
experiment_logs/shot_otta_office_lbi_joint_sweep_20260818/phase_records/machine1/PREPARE.md
```

Record:

- files read;
- planning commands;
- two owned pairs;
- 11 new omega×LR points per pair;
- 12 reused reference identities;
- matrix/spec paths;
- 132 plan identities/hashes;
- absolute output root;
- static path-consistency results;
- deterministic planner result;
- shell syntax result;
- plan SHA256;
- generated future execution command.

Explicitly state:

```text
No GPU checks were performed.
No CUDA checks were performed.
No conda runtime/training commands were executed.
No launcher was executed.
No train.py command was executed.
No real experiment was launched.
```

If all planning/static checks pass, print exactly:

```bash
cd /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined

bash tools/run_office_lbi_joint_sweep_machine1_noninteractive.sh
```

Then stop.

---

# MODE=FINALIZE

Run only after the user has executed the generated shell on the separate GPU machine and copied/synchronized the resulting artifacts back into this repository tree.

FINALIZE must not launch, retry, resume, or rerun training.

FINALIZE should work from saved plans/results/logs only and should not require a GPU.

## 1. Verify completion from artifacts

Load the exact 132-entry plan and use existing status/result artifacts to report:

```text
completed
missing
failed
duplicate
hash mismatch
invalid summary
```

Expected:

```text
completed=132
missing=0
failed=0
duplicate=0
hash mismatch=0
invalid summary=0
```

Do not rerun incomplete entries in FINALIZE.

Also verify all 12 reused Stage-1 reference runs are present.

## 2. Build the full 144-row local result set

Merge:

```text
132 new runs
+
12 existing omega=.20/stage2_lr=.020 reference runs
```

Require exactly:

```text
24 hyperparameter configurations
6 transfers each
144 run rows
```

For every run report:

```text
budget
anchor_id
alpha
kappa
nu
omega
stage2_lr
transfer

scientific_valid
support_utilization_min
support_utilization_p05
support_utilization_mean
under_95pct_batch_count
under_90pct_batch_count
mean_budget_gap
max_budget_gap
rollback_rate
stage1_steps_mean
stage1_steps_max
stage1_max_steps_hit_count

PU
FO
```

Use real per-batch records.

## 3. Eligibility

A configuration is eligible only when all six transfers satisfy:

```text
run completes
no NaN/Inf/error
no Stage-1 max_steps failure
selected_count <= K on every batch
no budget violation
support utilization >= 0.90 on every batch
```

95% utilization remains a strong robustness preference/diagnostic, not a new hard gate.

PU is report-only.

## 4. Aggregate per configuration

For every omega×stage2_lr configuration report:

```text
valid_transfer_count
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
mean_FO
mean_PU
```

Use the complete seed-2026 Office baseline reference from:

```text
experiment_logs/shot_otta_office_seed2026_stage1_20260818
```

and compute:

```text
best_sparse_FO =
max(Random 3-mask mean, Magnitude, Saliency)
```

per transfer.

Report:

```text
mean_FO_margin
worst_transfer_FO_margin
```

Do not use seed-2020 results.

## 5. Local ranking

Within each owned anchor/budget pair, rank eligible configurations by:

```text
1. all 6 transfers scientific-valid
2. all 6 transfers pass 90% utilization hard gate
3. mean_FO_margin descending
4. worst_transfer_FO_margin descending
5. mean_FO descending
6. 95%-utilization robustness diagnostics
7. stage1_steps_max ascending
8. stage1_steps_mean ascending
```

PU is never used for ranking.

Do not launch more search even if the best point lies on a boundary.

## 6. Local FINALIZE outputs

Write:

```text
experiment_logs/shot_otta_office_lbi_joint_sweep_20260818/reports/machine1/
```

with at least:

```text
per_run_144.csv
per_config_24.csv
per_config_24.json
ranking.csv
ranking.md
FINALIZE_SUMMARY.json
```

Also write:

```text
phase_records/machine1/FINALIZE.md
phase_records/machine1/FINALIZE_READY.json
```

Only create `FINALIZE_READY.json` when the local 132-run grid plus 12 references is complete and validly summarized.

## 7. Global selection

This machine does not own global final selection.

Do not freeze final Office tuples from this machine alone.

Machine 4 FINALIZE is the designated global aggregator after all four `FINALIZE_READY.json` markers exist.


Finish with:

```text
FINALIZE complete for machine1.
No training was launched by FINALIZE.
```
