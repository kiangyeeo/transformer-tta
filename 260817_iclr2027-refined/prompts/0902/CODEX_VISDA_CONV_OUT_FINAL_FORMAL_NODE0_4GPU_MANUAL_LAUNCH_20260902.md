# Codex Prompt — VisDA-C Conv Out-Channel LBI Final Formal
## ONE 4-GPU MACHINE / NON-INTERACTIVE CODEX / PREPARE ONLY
## User manually launches exactly one generated shell command

Project root:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined
```

Conda environment:

```text
SHOT_TTA
```

Frozen V1J root:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_conv_lbi_v1j_out_joint_sweep_seed2026_20260830
```

New final-formal root:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_conv_lbi_final_formal_out_channel_seed2026_20260902
```

Control root:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_conv_lbi_final_formal_out_channel_seed2026_20260902/control/node0
```

Plan revision:

```text
visda_conv_out_final_formal_20260902_v1
```

---

# 0. CRITICAL EXECUTION RULE

This is a **non-interactive Codex PREPARE task**.

Codex MUST NOT launch any GPU experiment.

During PREPARE, Codex MUST NOT:

- execute the generated final-formal launcher;
- run training;
- run a VisDA stream;
- run `nvidia-smi`;
- probe CUDA devices;
- resume any old run;
- retry any scientific run;
- alter any frozen hyperparameter.

Codex MUST:

1. inspect the frozen Conv protocol, the completed V1J GLOBAL FINALIZE, and the selected final tuples;
2. inspect the existing repository-supported Conv LBI plan/runner/identity machinery;
3. inspect the existing **formal VisDA Conv baseline artifacts** and recover the exact formal hardware/software provenance required for runtime comparability;
4. create exactly three fresh final-formal plan rows;
5. create a 4-GPU launcher using GPU0/GPU1/GPU2, with GPU3 intentionally idle;
6. create a node-local artifact-only POSTRUN audit helper;
7. perform static CPU checks only;
8. write `SAFE_TO_LAUNCH: YES` only if every static check passes;
9. print the exact two-line manual launch command specified at the end of this prompt;
10. STOP.

The USER will manually copy/paste the launcher command.

The launcher itself may perform the GPU/hardware preflight and, only if that passes, launch the three final-formal runs.

---

# 1. READ FIRST — SOURCE OF TRUTH

Read:

```text
protocol/shot-otta_conv/OTTA_CONV_LBI_PROTOCOL_20260826_v1.md
protocol/shot-otta_conv/OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v1.md
protocol/shot-otta_conv/OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2.md
configs/otta_conv_lbi_protocol_20260826_v1.yaml
```

Read the completed V1J finalize:

```text
experiment_logs/visda_conv_lbi_v1j_out_joint_sweep_seed2026_20260830/
  phase_records/V1J/FINALIZE.md
  selected_configs/VISDA_OUT_FINAL_TUPLES.json
  selected_configs/VISDA_OUT_FINAL_TUPLES.md
  reports/VISDA_OUT_V1J_ALL_ROWS.csv
  reports/VISDA_OUT_V1J_GRID.md
  reports/VISDA_OUT_V1J_SELECTION.csv
```

Also inspect:

```text
tools/run_experiments_multi_gpu.py
train.py
shot_otta/config.py
```

and the completed V0/V1J launchers/plans plus the completed Office Conv final-formal PREPARE/launcher artifacts if present.

Use only repository-supported CLI/config fields. Do not invent flags.

---

# 2. FINAL-FORMAL SEMANTICS

This is **not tuning**.

The V1J search is closed.

The three tuples are frozen before these formal runs.

These final-formal runs exist only to obtain fresh, independent formal:

```text
PU
FO
12-class PU / FO
worst-class
class-wise std
support utilization
Stage-1 steps
online runtime
Stage-1 runtime
Stage-2 runtime
GPU peak memory
hardware/software provenance
```

The resulting accuracy MUST NOT be used to alter:

```text
alpha
kappa
nu
omega
stage2_lr
budget
K_G
```

No boundary expansion, no sentinel, no V2, no retuning after these runs.

---

# 3. EXACT FROZEN SCIENTIFIC MATRIX

Common scientific settings:

```text
dataset       = VISDA-C
transfer      = TV
source        = 0
target        = 1
seed          = 2026
variant       = conv_out_lbi
group_mode    = out_channel
candidate     = netF.layer4 Conv candidate
batch_size    = 256
workers       = 4
full stream   = 217 online batches
kappa         = 1.0
stage1_max_steps = 3000
support_threshold = 1e-4
budget_tolerance = 1e-4
stage2_steps  = 1
delta_nonzero_tolerance = 1e-12
save_model    = false
runtime_comparable = true
```

Exact three rows:

| Trial | GPU | rho_G | K_G | alpha | kappa | nu | omega | stage2_lr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `F_B0` | 0 | .0005 | 4 | .025 | 1 | .50 | .0125 | .005 |
| `F_B1` | 1 | .001 | 9 | .025 | 1 | .50 | .025 | .010 |
| `F_B2` | 2 | .002 | 18 | .050 | 1 | 1.00 | .025 | .005 |

GPU3:

```text
IDLE
```

Do not move a row to another GPU.
Do not run anything on GPU3.

Before constructing the plan, cross-check all three tuples against:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_conv_lbi_v1j_out_joint_sweep_seed2026_20260830/selected_configs/VISDA_OUT_FINAL_TUPLES.json
```

and:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_conv_lbi_v1j_out_joint_sweep_seed2026_20260830/phase_records/V1J/FINALIZE.md
```

If either file is missing, or if any frozen tuple disagrees with this table, STOP.
Do not resolve the discrepancy yourself.

---

# 4. FRESH-RUN REQUIREMENT

All three rows must be fresh final-formal evaluations.

Forbidden:

```text
reusing a V1J result as final formal
resume-partial-runs
loading a V1J stream checkpoint
continuing from any completed tuning model state
overwriting an existing final-formal run
```

Before `SAFE_TO_LAUNCH: YES`, verify that the new formal root contains no completed/partial scientific run for `F_B0/F_B1/F_B2`.

If any final-formal output already exists, STOP and report it.
Do not delete it and do not overwrite it.

The generated launcher MUST NOT pass:

```text
--resume-partial-runs
```

Each successful formal artifact must eventually report:

```text
runtime_resume_used = false
runtime_segment_count = 1
```

---

# 5. FORMAL HARDWARE / SOFTWARE PRECONDITION

Formal runtime comparison requires the final LBI runs to use the same formal execution environment as the existing matched VisDA Conv formal baselines.

During PREPARE:

1. locate the existing finalized VisDA Conv baseline artifacts for:
   - `conv_module_dense`;
   - out-channel Random / Magnitude / Saliency;
2. read their formal hardware/software provenance;
3. determine and record in:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_conv_lbi_final_formal_out_channel_seed2026_20260902/control/node0/EXPECTED_FORMAL_ENV.json
```

at minimum:

```text
gpu_name
torch_version
cuda_version
batch_size
workers
runtime_comparable
```

Do NOT infer these values from Office if VisDA Conv baseline metadata exists.

If the formal VisDA Conv baseline environment cannot be established unambiguously, STOP before launch.

The generated launcher must perform a **preflight before starting training**:

- inspect GPU0, GPU1, GPU2;
- verify all three are the same GPU model;
- verify GPU model equals the expected VisDA Conv formal baseline GPU model;
- verify PyTorch version equals expected;
- verify CUDA version equals expected;
- verify the intended batch size/workers match;
- verify exactly one experiment process will be launched per used GPU.

If any check fails:

```text
exit non-zero
launch 0 scientific runs
```

Do not silently downgrade to `runtime_comparable=false`.
The goal of this phase is formal evaluation **and** formal efficiency evidence.

GPU3 is intentionally unused.

---

# 6. GIT / REPOSITORY SAFETY

Record:

```bash
git rev-parse --show-toplevel
git branch --show-current
git rev-parse HEAD
git status --short
```

Do not modify tracked scientific code or frozen protocol/config.

Protected paths include at least:

```text
core/lbi/engine.py
core/lbi/groups.py
shot_otta/trainer.py
shot_otta/losses.py
shot_otta/models.py
shot_otta/config.py
train.py
configs/otta_conv_lbi_protocol_20260826_v1.yaml
protocol/shot-otta_conv/**
```

Never run:

```text
git reset
git clean
git stash
git add .
```

All new helper/planning files for this task must live under:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_conv_lbi_final_formal_out_channel_seed2026_20260902/control/node0
```

All scientific run artifacts must live under:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_conv_lbi_final_formal_out_channel_seed2026_20260902/runs/node0/
```

---

# 7. REQUIRED PREPARE ARTIFACTS

Create:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_conv_lbi_final_formal_out_channel_seed2026_20260902/control/node0/plan.jsonl
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_conv_lbi_final_formal_out_channel_seed2026_20260902/control/node0/PREPARE.md
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_conv_lbi_final_formal_out_channel_seed2026_20260902/control/node0/run_node_4gpu.sh
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_conv_lbi_final_formal_out_channel_seed2026_20260902/control/node0/formal_preflight.py
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_conv_lbi_final_formal_out_channel_seed2026_20260902/control/node0/postrun_node.py
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_conv_lbi_final_formal_out_channel_seed2026_20260902/control/node0/EXPECTED_FORMAL_ENV.json
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_conv_lbi_final_formal_out_channel_seed2026_20260902/control/node0/launcher_logs/
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_conv_lbi_final_formal_out_channel_seed2026_20260902/control/node0/command_history/
```

Create:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_conv_lbi_final_formal_out_channel_seed2026_20260902/runs/node0/
```

Do not create run artifacts during PREPARE.

---

# 8. PLAN GENERATION / SCIENTIFIC IDENTITY

Create exactly three plan rows, in this exact order:

```text
line 1 = F_B0
line 2 = F_B1
line 3 = F_B2
```

For each row:

- use the existing canonical effective-config resolver;
- use the existing canonical experiment-key/scientific-SHA builder;
- ensure `omega` and `stage2_lr` are included in scientific identity;
- ensure the frozen Stage-1 tuple is included;
- ensure `runtime_comparable=true`;
- ensure `save_model=false`;
- ensure final-formal phase/revision are explicit;
- output to a unique root under `runs/node0/<trial_id>`;
- do not hand-write a fake scientific SHA.

Suggested phase:

```text
VISDA_OUT_FINAL_FORMAL
```

Suggested plan revision:

```text
visda_conv_out_final_formal_20260902_v1
```

---

# 9. STATIC PREPARE AUDIT

Verify before declaring safe:

```text
3 / 3 plan rows
exact trial IDs F_B0 F_B1 F_B2
exact GPU assignment 0 / 1 / 2
GPU3 idle
3 unique experiment keys
3 unique scientific SHA256
3 unique output roots

VISDA-C only
TV only
source=0 target=1
seed=2026
conv_out_lbi only
out_channel only
no Filter
no Office
no extra budget
no extra omega
no extra LR

all three frozen tuples match V1J FINALIZE exactly
runtime_comparable=true
save_model=false
fresh-run policy
no resume flag
```

Run CPU/static checks:

```text
bash -n /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_conv_lbi_final_formal_out_channel_seed2026_20260902/control/node0/run_node_4gpu.sh
python syntax check for /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_conv_lbi_final_formal_out_channel_seed2026_20260902/control/node0/formal_preflight.py
python syntax check for /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_conv_lbi_final_formal_out_channel_seed2026_20260902/control/node0/postrun_node.py
git diff --check
```

If the repository exposes a supported CPU plan verifier, run it too.

Do not perform training or GPU probing here.

`PREPARE.md` must record:

```text
git branch / HEAD / status
frozen tuple source paths
three-row matrix
experiment keys
scientific SHA256
output roots
expected formal hardware/software provenance
fresh-run checks
static check results
launcher path
exact manual launch command
```

Only if every check passes, end:

```text
SAFE_TO_LAUNCH: YES
```

---

# 10. EXACT LAUNCHER STRUCTURE

Generate:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_conv_lbi_final_formal_out_channel_seed2026_20260902/control/node0/run_node_4gpu.sh
```

It must:

```bash
#!/usr/bin/env bash
set -euo pipefail
```

Use repository-relative/project-root-safe paths.

The launcher sequence is:

```text
1. formal_preflight.py
2. GPU0 -> F_B0
3. GPU1 -> F_B1
4. GPU2 -> F_B2
5. GPU3 -> idle
6. wait for all three
7. if any run failed: exit non-zero, do not retry
8. if all succeeded: run postrun_node.py
9. exit non-zero if POSTRUN audit fails
```

F_B0/F_B1/F_B2 must start concurrently after preflight passes.

Use the established one-row-plan runner pattern.

For each GPU, make a one-row plan from the corresponding line of `plan.jsonl`, then run:

```text
conda run --no-capture-output -n SHOT_TTA python tools/run_experiments_multi_gpu.py <ONE_ROW_PLAN>   --runs-root <RUNS_ROOT>   --logs-root <TRIAL_LOG_ROOT>   --workdir /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined   --gpus <GPU_ID>   --max-workers 1   --workers-per-gpu 1   --command-history <TRIAL_COMMAND_HISTORY>
```

Important:

```text
NO --resume-partial-runs
NO conda activate
NO second process on the same GPU
NO task on GPU3
```

If the actual repository CLI differs, inspect the existing successful V1J/Office-final-formal launcher and use the supported equivalent, preserving all semantics above.

Every Python command must use:

```text
conda run --no-capture-output -n SHOT_TTA
```

---

# 11. POSTRUN ARTIFACT-ONLY AUDIT

After all three scientific runs exit successfully, the launcher automatically runs:

```text
conda run --no-capture-output -n SHOT_TTA python /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_conv_lbi_final_formal_out_channel_seed2026_20260902/control/node0/postrun_node.py
```

`postrun_node.py` MUST NOT launch, retry, resume, or modify a scientific run.

For each of F_B0/F_B1/F_B2 verify:

```text
summary.json exists
status=completed
217 online batches
batch indices 0..216 exactly once
finite / error-free
selected_group_count <= K_G every batch
zero Stage-1 max-step failures
no support/mask/budget contract violation
artifact tuple == frozen tuple
experiment key matches plan
scientific SHA matches plan
runtime_comparable=true
runtime_resume_used=false
runtime_segment_count=1
GPU model matches EXPECTED_FORMAL_ENV
PyTorch version matches EXPECTED_FORMAL_ENV
CUDA version matches EXPECTED_FORMAL_ENV
```

Write:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_conv_lbi_final_formal_out_channel_seed2026_20260902/control/node0/POSTRUN.md
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_conv_lbi_final_formal_out_channel_seed2026_20260902/control/node0/node_results.csv
```

Report for every budget:

```text
PU mAcc
PU overall
PU 12-class accuracy if available
PU worst-class
PU class-wise std

FO mAcc
FO overall
FO 12-class accuracy
FO worst-class
FO class-wise std

min / mean / p05 support utilization
fraction below .95
selected scalar count
realized scalar ratio

Stage-1 mean steps
Stage-1 max steps
Stage-1 runtime
Stage-2 runtime
online runtime
GPU peak allocated memory

runtime_resume_used
runtime_segment_count
hardware/software provenance
```

POSTRUN must not retune.

End `POSTRUN.md` with:

```text
FINAL_FORMAL_COMPLETE: YES/NO
ASSIGNED_RUNS_COMPLETE: X/3
SCIENTIFIC_VALID: X/3
RUNTIME_COMPARABLE: X/3
FRESH_NONRESUMED: X/3
RETUNING_PERFORMED: NO
EXTRA_RUNS_LAUNCHED: 0
READY_FOR_GLOBAL_FINALIZE: YES/NO
```

---

# 12. FAILURE POLICY

If PREPARE finds any disagreement:

```text
STOP
SAFE_TO_LAUNCH: NO
```

If launcher preflight finds hardware/software mismatch:

```text
STOP
launch 0 runs
```

If one final-formal run fails:

```text
do not change hyperparameters
do not launch a substitute scientific cell
do not silently resume
do not delete artifacts
exit non-zero and preserve evidence
```

Do not solve failures by retuning.

---

# 13. REQUIRED FINAL CODEX RESPONSE

After PREPARE only, if and only if:

```text
SAFE_TO_LAUNCH: YES
```

Codex must print exactly:

```bash
cd /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined
bash experiment_logs/visda_conv_lbi_final_formal_out_channel_seed2026_20260902/control/node0/run_node_4gpu.sh
```

Then STOP.

Do not execute the command yourself.
