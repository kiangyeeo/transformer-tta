# Office LBI Final Formal Evaluation — Machine 2 — Budget 0.001

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

Do not change frozen scientific behavior, protocol values, source checkpoints, seeds, metrics, BN semantics, sparse-budget semantics, experiment-identity semantics, or the already-frozen Office LBI hyperparameters.

Frozen revisions:

```text
IMPLEMENTATION_REVISION = iclr2027_refined_20260817_v1
EFFICIENCY_PROTOCOL_REVISION = otta_fc_batch_efficiency_20260817_v1
SOURCE_CHECKPOINT_REVISION = nips2026_shot_otta_uda_source_v1
```

The Office LBI hyperparameter search is **closed**.

This prompt is for the **final formal LBI evaluation/efficiency rerun only**.

Machine tag:

```text
machine2
```

This machine owns exactly one Office budget:

```text
budget = 0.001
K      = 524
```

Frozen final LBI tuple:

```text
alpha     = 0.10
kappa     = 1.0
nu        = 0.25
omega     = 0.30
stage2_lr = 0.020
```

Other frozen LBI settings:

```text
stage1_max_steps = 3000
stage2_steps = 1
support_threshold = 1e-4
budget_tolerance = 1e-4
delta_nonzero_tolerance = 1e-12
```

Formal Office setting:

```text
dataset = Office
seed = 2026
batch_size = 64
workers = 4
```

Run exactly the six Office transfers:

```text
A->D
A->W
D->A
D->W
W->A
W->D
```

Therefore this machine must run exactly:

```text
6 final formal LBI experiments
```

These are not tuning runs.

No parameter may be changed based on the results.

Shared final-formal root:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/shot_otta_office_lbi_final_formal_20260818
```

Canonical runs root:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/shot_otta_office_lbi_final_formal_20260818/runs
```

---

# MODE=PREPARE

## Critical rule: PREPARE is pure planning only

The current setup machine may have **no GPU**.

PREPARE must therefore be hardware-agnostic.

Do not run:

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

Do not require the PREPARE machine to have a GPU, NVIDIA driver, CUDA runtime, or active training environment.

PREPARE may only inspect repository files/artifacts, generate formal matrix/plan/shell files, and perform static consistency checks.

## 1. Verify the frozen final tuple from existing global selection artifacts

Read the completed Office joint-sweep global outputs, especially:

```text
experiment_logs/shot_otta_office_lbi_joint_sweep_20260818/selected_configs/OFFICE_FINAL_LBI_TUPLES.json
experiment_logs/shot_otta_office_lbi_joint_sweep_20260818/reports/global/OFFICE_LBI_FINAL_SELECTION.json
experiment_logs/shot_otta_office_lbi_joint_sweep_20260818/reports/global/OFFICE_LBI_FINAL_SELECTION.md
```

Verify that budget `0.001` is frozen exactly as:

```text
alpha     = 0.10
kappa     = 1.0
nu        = 0.25
omega     = 0.30
stage2_lr = 0.020
```

and that the selection provenance says the Office LBI hyperparameter search is closed.

If the artifacts disagree with the tuple above, stop PREPARE and report the mismatch.

Do not choose a different tuple.

## 2. Statically verify protocol/formal configuration

Read:

```text
protocol/shot-otta_fc/OTTA_FC_LBI_PROTOCOL_20260817_v1.md
configs/otta_fc_lbi_protocol_20260817_v1.yaml
experiments/shot_otta_fc_lbi_formal_20260817_v1.yaml
tools/plan_experiments.py
tools/run_experiments_multi_gpu.py
```

Verify from files/artifacts:

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

candidate scope:
netB.bottleneck.weight
netB.bottleneck.bias

candidate count=524544
strict K=floor(rho*N)
support_threshold=1e-4
stage1_max_steps=3000
stage2_steps=1
```

Formal LBI runs must use:

```text
save_model=false
batch-boundary LBI checkpoint/resume allowed
runtime_comparable=true
```

Do not change any scientific value.

## 3. Build one machine-specific final-formal matrix/spec

Create under:

```text
experiment_logs/shot_otta_office_lbi_final_formal_20260818/matrices/machine2/
```

The matrix must contain only:

```text
variant=module_lbi
budget=0.001
alpha=0.10
kappa=1.0
nu=0.25
omega=0.30
stage2_lr=0.020
stage1_max_steps=3000
stage2_steps=1
support_threshold=1e-4
budget_tolerance=1e-4
delta_nonzero_tolerance=1e-12
seed=2026
batch_size=64
workers=4
save_model=false
```

across exactly the six transfers.

Use the absolute canonical output root:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/shot_otta_office_lbi_final_formal_20260818/runs
```

Do not use a relative `output_root`.

## 4. Generate exactly one 6-entry formal plan

Generate:

```text
experiment_logs/shot_otta_office_lbi_final_formal_20260818/plans/machine2/plan.json
```

using the existing planner/scientific-identity machinery.

Pure planning invocation is allowed only if it does not initialize GPU/training dependencies.

Hard static assertions:

```text
experiment_count = 6
unique experiment_key count = 6
duplicate experiment_key count = 0
```

Verify:

```text
exactly one entry for each Office transfer
only budget 0.001
only the frozen tuple
only module_lbi
only seed 2026
runtime_comparable=true
save_model=false
```

Every plan entry must contain a fully resolved LBI tuple; there must be no unresolved `lbi:null`.

Run the planner twice if possible and verify deterministic experiment keys/hashes.

## 5. Mandatory static output-path validation

Parse the generated plan directly.

For all 6 entries require:

```text
expected_output_root is under canonical RUNS_ROOT
expected_output_root is inside 260817_iclr2027-refined
expected_output_root is NOT under /PJ/Split-LBI/experiment_logs/
```

This is mandatory because a previous Office launcher failed due to `output_root` / `--runs-root` mismatch.

Do not invoke the launcher in PREPARE.

## 6. Check for accidental identity overlap

Statically compare the six new final-formal experiment keys against:

```text
experiment_logs/shot_otta_office_seed2026_stage1_20260818
experiment_logs/shot_otta_office_lbi_joint_sweep_20260818
```

These final-formal runs should have their intended formal identity/provenance and must not accidentally point to tuning output directories.

Do not reuse tuning run directories as the final formal result location.

If the scientific identity system intentionally yields the same scientific hash for an identical scientific configuration, that is acceptable only if the run artifact location / formal plan identity is handled exactly according to the repository's existing protocol. Do not modify identity semantics to force uniqueness.

Record what the current identity system does.

## 7. Create exactly one future non-interactive shell

Create:

```text
tools/run_office_lbi_final_formal_budget_001_noninteractive.sh
```

This shell will be executed later on a separate **8-GPU machine**.

The final formal run needs exactly six simultaneous experiments.

Use only:

```text
GPUs: 0,1,2,3,4,5
max-workers: 6
workers-per-gpu: 1
```

Leave GPUs `6,7` unused by this experiment.

Do not fill spare GPUs with unrelated work inside this shell.

The reason is that these are formal efficiency runs and each measured experiment must have one dedicated GPU.

Required shell header:

```bash
#!/usr/bin/env bash
set -euo pipefail
```

The future user command will be exactly:

```bash
cd /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined
bash tools/run_office_lbi_final_formal_budget_001_noninteractive.sh
```

The shell must:

- remain attached in foreground;
- use `conda run --no-capture-output -n SHOT_TTA`;
- use the exact 6-entry plan;
- use the exact absolute canonical RUNS_ROOT;
- use machine-specific logs and command history;
- use `--gpus 0,1,2,3,4,5`;
- use `--max-workers 6`;
- use `--workers-per-gpu 1`;
- enable `--resume-partial-runs` for LBI engineering resilience;
- not run FINALIZE;
- not run baselines;
- not run any tuning;
- not modify the frozen tuple.

## 8. Runtime self-preflight belongs inside the future shell

PREPARE must not execute runtime checks.

However, when the shell is actually submitted on the future GPU machine, it must fail-fast **before child training starts** if the runtime environment is obviously wrong.

At shell runtime check:

```text
conda command exists
SHOT_TTA can invoke Python
at least GPUs 0..5 are visible/addressable
plan exists
plan SHA256 matches PREPARE_READY
plan count=6
unique keys=6
all expected_output_root values under canonical RUNS_ROOT
all six entries are the intended frozen tuple
```

Also record for GPUs 0..5:

```text
hostname
GPU model
GPU index
total memory
PyTorch version
CUDA version
```

Do not perform these checks during PREPARE.

### Formal hardware provenance

These runs are intended to be `runtime_comparable=true`.

At execution time, record the GPU model and environment used.

If the six GPUs on this machine are not the same GPU model, fail before training.

If existing formal baseline hardware provenance is available in the repository, compare the current GPU model against it.

If it clearly differs from the formal baseline hardware model:

- fail before training;
- explain that PU/FO could still be scientifically usable, but runtime comparison would not satisfy the frozen formal efficiency protocol;
- do not silently mark a mismatched hardware run as `runtime_comparable=true`.

Do not change the plan's scientific settings to compensate.

## 9. Anti-double-submit lock

The shell must create an atomic machine-specific lock:

```text
experiment_logs/shot_otta_office_lbi_final_formal_20260818/locks/machine2.lock
```

Use an atomic mechanism such as `mkdir`.

Record hostname/PID/start time.

If the lock already exists, fail before launching.

Use `trap` to remove the lock on normal exit and standard termination signals.

Do not automatically delete another live process's lock.

## 10. PREPARE static validation only

PREPARE may run:

```text
bash -n tools/run_office_lbi_final_formal_budget_001_noninteractive.sh
git diff --check
pure planner generation/determinism checks
static plan count/uniqueness checks
static path-consistency checks
static tuple checks
```

PREPARE must not run:

```text
conda run
train.py
launcher
nvidia-smi
CUDA checks
real experiment
```

## 11. PREPARE_READY

Only after all pure planning/static checks pass, create:

```text
experiment_logs/shot_otta_office_lbi_final_formal_20260818/phase_records/machine2/PREPARE_READY.json
```

containing:

```text
machine_tag
budget
frozen tuple
plan_path
plan_sha256
matrix/spec paths
canonical_runs_root
shell_path
experiment_count=6
unique_experiment_keys=6
static_path_check_passed=true
bash_syntax_passed=true
timestamp
```

Do not claim GPU/runtime checks passed.

## 12. PREPARE reproducibility record

Write:

```text
experiment_logs/shot_otta_office_lbi_final_formal_20260818/phase_records/machine2/PREPARE.md
```

Record:

- exact planning commands;
- frozen tuple provenance;
- protocol/revision IDs;
- matrix/spec paths;
- six planned identities/hashes;
- static path checks;
- plan SHA256;
- shell path;
- future GPU allocation `0..5`;
- final execution command.

Explicitly state:

```text
No GPU checks were performed in PREPARE.
No CUDA checks were performed in PREPARE.
No launcher was executed in PREPARE.
No train.py command was executed in PREPARE.
No real experiment was launched in PREPARE.
```

If all checks pass, print exactly:

```bash
cd /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined

bash tools/run_office_lbi_final_formal_budget_001_noninteractive.sh
```

Then stop.

---

# MODE=FINALIZE

Run only after the user has executed the generated shell on the separate GPU machine and the artifacts are available in the shared repository tree.

FINALIZE must not launch, retry, resume, or rerun training.

No GPU is required for FINALIZE.

## 1. Completion check

Load the exact 6-entry formal plan and report:

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
completed=6
missing=0
failed=0
duplicate=0
hash mismatch=0
invalid summary=0
```

If incomplete, report exact identities/paths and stop before declaring the final budget result.

Do not rerun anything.

## 2. Verify the frozen tuple did not drift

For all six completed runs verify exactly:

```text
budget=0.001
alpha=0.10
kappa=1.0
nu=0.25
omega=0.30
stage2_lr=0.020
stage1_max_steps=3000
stage2_steps=1
support_threshold=1e-4
seed=2026
batch_size=64
workers=4
```

No post-hoc parameter substitution is allowed.

## 3. Formal result aggregation

Aggregate the six transfers and report:

```text
PU overall per transfer
FO overall per transfer
Office mean PU
Office mean FO

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
```

Also aggregate the formal efficiency fields from the batch records:

```text
online_batch_runtime_mean_sec
online_batch_runtime_std_sec
online_batch_runtime_median_sec
online_batch_runtime_p95_sec
online_compute_runtime_sec

adapt_runtime_mean/std/total
pu_runtime_mean/std/total

lbi_stage1_runtime_mean/std/total
lbi_stage2_runtime_mean/std/total

gpu_peak_allocated_mean_mb
gpu_peak_allocated_max_mb
gpu_peak_reserved_mean_mb
gpu_peak_reserved_max_mb

fo_eval_runtime_sec
wall_runtime_sec
runtime_resume_used
runtime_segment_count
```

Use `gpu_peak_allocated_max_mb` as the primary formal GPU-memory number.

## 4. Compare against the already-completed formal baselines

Use only the new seed-2026 formal Office baseline results from:

```text
experiment_logs/shot_otta_office_seed2026_stage1_20260818
```

Do not use seed-2020 results.

For this budget compare final LBI against:

```text
Random 3-mask mean
Magnitude
Saliency
Module dense
Full dense
Source only
```

Report at least:

```text
LBI mean FO
best sparse mean FO
LBI - best sparse FO margin
module dense mean FO
LBI - module dense FO margin
full dense mean FO
LBI - full dense FO margin

LBI mean PU
corresponding PU comparisons

runtime comparison
GPU peak memory comparison
```

Also report per-transfer FO margins against best sparse, Module dense and Full dense.

## 5. Hardware comparability audit

Read the recorded execution hardware provenance.

Confirm whether this formal LBI run used the same GPU model / relevant environment as the formal baseline efficiency runs.

Report explicitly:

```text
runtime_comparable = true/false
reason
```

Do not discard accuracy results solely because runtime hardware differs, but do not present mismatched runtime as a strict apples-to-apples formal comparison.

## 6. Write machine-local final outputs

Create under:

```text
experiment_logs/shot_otta_office_lbi_final_formal_20260818/reports/machine2/
```

at least:

```text
per_transfer.csv
per_transfer.json
final_budget_summary.csv
final_budget_summary.json
final_budget_summary.md
efficiency_summary.json
```

Also write:

```text
phase_records/machine2/FINALIZE.md
phase_records/machine2/FINALIZE_READY.json
```

Only create `FINALIZE_READY.json` when all 6 runs are complete and summarized.

## 7. Global Office final-result aggregation

This machine does not own global aggregation.

Do not construct the final all-budget Office table from one budget alone.

Machine 3 FINALIZE is the designated global aggregator after all three `FINALIZE_READY.json` markers exist.


Finish with:

```text
FINALIZE complete for machine2 / budget 0.001.
No training was launched by FINALIZE.
```
