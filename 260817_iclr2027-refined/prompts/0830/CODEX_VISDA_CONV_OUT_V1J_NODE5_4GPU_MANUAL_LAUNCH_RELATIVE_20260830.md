# Codex Prompt — VisDA-C Conv Out-Channel LBI Full Joint Search
## NON-INTERACTIVE CODEX: PREPARE ONLY
## User manually launches the generated shell command

Project root:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined
```

Conda environment:

```text
SHOT_TTA
```

Frozen V0 root:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_conv_lbi_v0_out_stage1_seed2026_20260828
```

Joint-search root:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_conv_lbi_v1j_out_joint_sweep_seed2026_20260830
```

Plan revision:

```text
visda_conv_out_v1j_full_joint_20260830_v2_manual_launch
```

Expected shared amendment SHA256:

```text
8e110a497ab5689fb753cb5b041295bd476b27990ef8b000a447f38b44afef3c
```

# 0. Critical execution rule

This is a **non-interactive Codex PREPARE task**.

Codex MUST NOT launch any GPU experiment.

Codex MUST NOT:
- execute the generated launcher;
- run training;
- probe GPUs;
- run `nvidia-smi`;
- wait for experiments;
- retry or resume any scientific run during PREPARE.

Codex MUST:
1. inspect the repository and frozen V0 artifacts;
2. create/verify the exact shared amendment;
3. create this node's exact plan and launcher;
4. create a node-local POSTRUN audit helper that the launcher will call automatically after all assigned runs succeed;
5. perform static CPU checks only;
6. if safe, write `SAFE_TO_LAUNCH: YES`;
7. print the exact two-line manual launch command;
8. STOP.

The USER will copy/paste the printed launch command into the shell manually.

The generated launcher itself is allowed to:
- launch the assigned GPU experiments;
- run per-GPU sequential chains;
- wait for all chains;
- after all chains succeed, automatically execute the node-local POSTRUN audit helper.

Therefore the user needs only one manual launcher submission on this node.

# 1. Read first

Read and obey:

```text
protocol/shot-otta_conv/OTTA_CONV_LBI_PROTOCOL_20260826_v1.md
protocol/shot-otta_conv/OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v1.md
protocol/shot-otta_conv/OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2.md
configs/otta_conv_lbi_protocol_20260826_v1.yaml

experiment_logs/visda_conv_lbi_v0_out_stage1_seed2026_20260828/
  phase_records/V0/FINALIZE.md
  selected_configs/VISDA_OUT_STAGE1_FROZEN.json
  selected_configs/VISDA_OUT_STAGE1_FROZEN.md
  reports/VISDA_OUT_V0_ALL_ROWS.csv
```

Also inspect:
- completed V0 `plan.jsonl`;
- completed V0 node launcher(s);
- `tools/run_experiments_multi_gpu.py`;
- `shot_otta/config.py`;
- `train.py`.

Reuse the repository's existing effective-config and scientific-identity machinery.
Do NOT hand-invent experiment keys or scientific SHA256 values.

If the frozen V0 files are absent or disagree with this prompt, STOP before generating a launchable plan.

# 2. Shared search-plan amendment

Create or verify:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_conv_lbi_v1j_out_joint_sweep_seed2026_20260830/SEARCH_PLAN_AMENDMENT.md
```

It must be byte-identical to the exact amendment text included at the end of this prompt and must have SHA256:

```text
8e110a497ab5689fb753cb5b041295bd476b27990ef8b000a447f38b44afef3c
```

Use an atomic temp-file + rename if the file is absent, because eight nodes may PREPARE concurrently.

If the file already exists with a different SHA256, STOP.
Do not overwrite a conflicting shared amendment.

# 3. Scientific scope

Only:

```text
dataset      = VisDA-C
source       = 0
target       = 1
seed         = 2026
variant      = conv_out_lbi
group_mode   = out_channel
batch_size   = 256
workers      = 4
full stream  = 217 online batches
runtime_comparable = false
save_model   = false
```

Frozen Stage-1 anchors:

| rho_G | K_G | alpha | kappa | nu |
|---:|---:|---:|---:|---:|
| .0005 | 4 | .025 | 1 | .50 |
| .001 | 9 | .025 | 1 | .50 |
| .002 | 18 | .050 | 1 | 1.00 |

Full fixed joint grid:

```text
omega     in {.0015625,.003125,.00625,.0125,.025}
stage2_lr in {.0025,.005,.010}
```

Shared LBI settings:

```text
stage1_max_steps = 3000
support_threshold = 1e-4
budget_tolerance = 1e-4
stage2_steps = 1
delta_nonzero_tolerance = 1e-12
```

Do not run:
- Filter-LBI / filter_connection;
- Office;
- V0 retuning;
- extra alpha/nu;
- omega outside the five fixed values;
- LR outside the three fixed values;
- boundary expansion;
- sentinels;
- a later staged V2.

All 45 cells are fresh rows in this phase. Do not substitute a V0 row.

# 4. Validity / eligibility / final ranking contract

Scientific-valid requires:
- completed;
- finite / error-free;
- selected_group_count <= K_G at every batch;
- zero Stage-1 max-step failures;
- no support / mask / budget contract violation.

Out-channel utilization eligibility:

```text
mean(u_t) >= .90
p05(u_t)  >= .75
```

A later GLOBAL FINALIZE, not this node, ranks eligible cells within each budget by:

```text
FO mAcc desc
-> FO worst-class desc
-> FO class-wise std asc
-> FO overall desc
-> Stage-1 max steps asc
-> Stage-1 mean steps asc
```

PU is report-only.

# 5. Git / path safety

Record:

```bash
git rev-parse --show-toplevel
git branch --show-current
git rev-parse HEAD
git status --short
```

Do not modify tracked scientific code, frozen protocols, or config.

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

All node-specific PREPARE/launcher/audit-helper files must remain under this node's `CONTROL_ROOT`.
All run artifacts must remain under this node's `RUNS_ROOT`.

# 6. PREPARE artifacts

Create under the node-specific `CONTROL_ROOT` given later:

```text
plan.jsonl
PREPARE.md
run_node_4gpu.sh
postrun_node.py
launcher_logs/
command_history/
```

Create the node-specific `RUNS_ROOT`.

`plan.jsonl` must contain only this node's exact assigned rows.

For each row:
- use exactly the assigned rho/K/alpha/kappa/nu/omega/stage2_lr;
- resolve the repository canonical effective config;
- generate canonical experiment key and scientific SHA256;
- include actual omega and stage2_lr in scientific identity;
- set `phase=VISDA_OUT_V1J_FULL_JOINT`;
- set `plan_revision=visda_conv_out_v1j_full_joint_20260830_v2_manual_launch`;
- `runtime_comparable=false`;
- `save_model=false`;
- output to `RUNS_ROOT/<trial_id>`.

# 7. Static checks only

Before declaring safe, verify:

```text
exact assigned row count
exact assigned trial IDs
0 duplicate trial IDs
0 duplicate experiment keys
0 duplicate scientific SHA256
0 duplicate output roots
all rows are VisDA-C / seed2026 / conv_out_lbi / out_channel
all K/alpha/kappa/nu match the frozen V0 anchors
all omega/LR exactly match this node's table
runtime_comparable=false
no Filter
no Office
no extra grid point
```

Run only static/CPU checks, including:

```text
bash -n CONTROL_ROOT/run_node_4gpu.sh
python syntax check for CONTROL_ROOT/postrun_node.py
git diff --check
```

If the repository has an existing CPU plan verifier, use it.
Do not invent unsupported CLI flags.

`PREPARE.md` must record:
- Git branch and HEAD;
- assigned matrix;
- experiment keys / SHA / output roots;
- amendment SHA;
- static checks;
- launcher path;
- postrun helper path;
- exact manual launch command.

Only if everything passes, end `PREPARE.md` with:

```text
SAFE_TO_LAUNCH: YES
```

# 8. Launcher contract

The USER, not Codex, will execute the launcher manually.

Model the launcher on the successful V0 tuning launchers.

Every Python command must use:

```text
conda run --no-capture-output -n SHOT_TTA
```

Never require `conda activate`.

Use the current canonical experiment runner and only supported flags that actually exist.
Use the established VisDA tuning resume/checkpoint behavior from V0; do not invent a new resume policy.

Per GPU:
- at most one experiment process at a time;
- if a GPU has two assigned rows, run them sequentially;
- second row starts immediately after the first succeeds;
- no global first-wave barrier.

Different GPU chains run concurrently.

If a run exits non-zero:
- stop that GPU chain;
- launcher eventually exits non-zero;
- do not auto-retry with changed parameters;
- do not add a replacement row.

After all assigned GPU chains succeed, the launcher MUST automatically run:

```text
conda run --no-capture-output -n SHOT_TTA python CONTROL_ROOT/postrun_node.py
```

If POSTRUN audit fails, launcher exits non-zero.

# 9. POSTRUN helper contract

`postrun_node.py` is artifact-only. It must not launch or retry training.

For every assigned row verify:
- `summary.json` exists;
- status=completed;
- 217 online batches;
- batch indices 0..216 exactly once;
- finite/error-free;
- selected_group_count <= K_G every batch;
- zero Stage-1 max-step failures for a scientific-valid row;
- no support/mask/budget violation;
- plan tuple equals artifact tuple;
- experiment key and scientific SHA match plan.

Write:

```text
CONTROL_ROOT/POSTRUN.md
CONTROL_ROOT/node_results.csv
```

Report per row:
- scientific-valid;
- utilization eligible;
- min/mean/p05 utilization;
- fraction below .95;
- selected scalar count / realized scalar ratio;
- Stage-1 mean/max steps;
- PU mAcc / overall;
- FO mAcc / overall;
- FO 12-class accuracy;
- FO worst-class;
- FO class-wise std;
- runtime_resume_used;
- runtime_segment_count.

POSTRUN must not perform global winner selection.

# 10. Required Codex final output

After PREPARE only, Codex must print exactly the final manual command block below and STOP.

Because the first line already `cd`s into the project root, the second line MUST use the repository-relative launcher path. Do NOT print the absolute launcher path after `bash`.

Do not execute it yourself.

# 11. NODE-SPECIFIC ASSIGNMENT — node5

```text
CONTROL_ROOT = /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_conv_lbi_v1j_out_joint_sweep_seed2026_20260830/control/node5
RUNS_ROOT    = /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_conv_lbi_v1j_out_joint_sweep_seed2026_20260830/runs/node5
ASSIGNED_RUN_COUNT = 5
```

Exact rows:

| GPU | Chain | Trial | rho_G | K_G | alpha | kappa | nu | omega | stage2_lr |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | `J_B0_W2_L2` | 0.0005 | 4 | 0.025 | 1 | 0.50 | 0.00625 | 0.01 |
| 0 | 2 | `J_B2_W3_L0` | 0.002 | 18 | 0.050 | 1 | 1.00 | 0.0125 | 0.0025 |
| 1 | 1 | `J_B2_W4_L2` | 0.002 | 18 | 0.050 | 1 | 1.00 | 0.025 | 0.01 |
| 2 | 1 | `J_B0_W3_L0` | 0.0005 | 4 | 0.025 | 1 | 0.50 | 0.0125 | 0.0025 |
| 3 | 1 | `J_B1_W1_L2` | 0.001 | 9 | 0.025 | 1 | 0.50 | 0.003125 | 0.01 |

Exact per-GPU chains:

```text
GPU0: J_B0_W2_L2 -> J_B2_W3_L0
GPU1: J_B2_W4_L2
GPU2: J_B0_W3_L0
GPU3: J_B1_W1_L2
```

Do not move a row to another GPU.
Do not add, omit, or replace a row.

Launcher path relative to project root:

```text
experiment_logs/visda_conv_lbi_v1j_out_joint_sweep_seed2026_20260830/control/node5/run_node_4gpu.sh
```

The final Codex response after PREPARE must be exactly:

```bash
cd /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined
bash experiment_logs/visda_conv_lbi_v1j_out_joint_sweep_seed2026_20260830/control/node5/run_node_4gpu.sh
```

Then STOP. Do not execute that command.

# 12. Exact amendment text

The shared amendment file must contain exactly:

```markdown
# VisDA-C Conv Out-Channel LBI — Full Joint Search Amendment

```text
PLAN_REVISION = visda_conv_out_v1j_full_joint_20260830_v2_manual_launch
DECISION_DATE = 2026-08-30
DECISION_POINT = before observing any V1 omega-search result
DATASET = VISDA-C
METHOD = SHOT-OTTA Conv Out-Channel Group-LBI
SEED = 2026
```

The previously planned staged V1 -> V2 search is replaced, before observing any V1 result, by one fixed full joint grid because 8 x 4 = 32 GPUs became available and omega and Stage-2 LR interact through the persistent sequential OTTA trajectory.

Frozen V0 Stage-1 anchors:

| rho_G | K_G | alpha | kappa | nu |
|---:|---:|---:|---:|---:|
| .0005 | 4 | .025 | 1 | .50 |
| .001 | 9 | .025 | 1 | .50 |
| .002 | 18 | .050 | 1 | 1.00 |

Fixed joint grid for every budget:

```text
omega     in {.0015625, .003125, .00625, .0125, .025}
stage2_lr in {.0025, .005, .010}
```

Total = 45 runs.

Scientific-valid:
- completed;
- finite / error-free;
- selected_group_count <= K_G at every online batch;
- zero Stage-1 max-step failures;
- no support / mask / budget contract violation.

Out-channel utilization eligibility:
- mean(u_t) >= .90;
- p05(u_t) >= .75.

Rank eligible cells within each budget:
FO mAcc desc -> FO worst-class desc -> FO class-wise std asc -> FO overall desc -> Stage-1 max steps asc -> Stage-1 mean steps asc.

PU is report-only.

Stop rule:
- no omega boundary expansion;
- no Stage-2 LR boundary expansion;
- no alpha/nu retuning;
- no post-hoc sentinel;
- no later staged V2.

After all 45 rows are complete, freeze exactly one winner per budget from this grid, then run 3 fresh final-formal VisDA runs.
```
