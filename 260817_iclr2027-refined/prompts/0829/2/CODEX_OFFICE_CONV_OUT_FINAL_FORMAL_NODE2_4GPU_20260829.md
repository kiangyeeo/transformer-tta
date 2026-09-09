# Codex Prompt — SHOT-OTTA × Office-31 × Conv Out-Channel LBI Final Formal

## 0. Task type

This is a **frozen final-formal rerun**, not tuning.

Do not change any scientific setting after seeing accuracy.

The Office out-channel tuning is already closed. The only goal is to fresh-run the frozen tuples under formal efficiency conditions and save auditable artifacts.

Project root:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined
```

Conda environment:

```text
SHOT_TTA
```

Shared final-formal root:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/office_conv_lbi_final_formal_out_channel_seed2026_20260829
```

All Python execution must use:

```text
conda run --no-capture-output -n SHOT_TTA ...
```

Do **not** use `conda activate`.

---

## 1. Read first — normative inputs

Read these files before doing anything:

```text
protocol/shot-otta_conv/OTTA_CONV_LBI_PROTOCOL_20260826_v1.md
protocol/shot-otta_conv/OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v1.md
protocol/shot-otta_conv/OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2.md
protocol/shot-otta_conv/OTTA_CONV_BASELINE_FORMAL_20260827_v1.md
configs/otta_conv_lbi_protocol_20260826_v1.yaml

experiment_logs/office_conv_lbi_r2_out_boundary_expansion_seed2026_20260828/
  selected_configs/OFFICE_OUT_LBI_FINAL_TUPLES.json
  phase_records/R2_BOUNDARY_EXPANSION/FINALIZE.md

experiment_logs/office_conv_lbi_r2_out_joint_sweep_seed2026_20260828/
  phase_records/R2/FINALIZE.md

experiment_logs/conv_baseline_formal_global_20260827/
  FINALIZE.md
```

Also inspect the current canonical experiment planner / identity code and multi-GPU runner before constructing plans. Reuse existing scientific identity generation. **Do not hand-invent experiment SHA256 values.**

---

## 2. Frozen scientific scope

This final formal covers **Office-31 out_channel only**.

Filter-LBI is already `BLOCKED_V2` and has been dropped from the ICLR mainline. Do not fabricate a Filter-LBI tuple and do not run Filter-LBI.

The original v1 search protocol described a later 42-condition out+filter formal stage. The current mainline closure is narrower because the authorized Filter rescue failed. For this task:

```text
dataset      = office
variant      = conv_out_lbi
group_mode   = out_channel
seed         = 2026
transfers    = AD, AW, DA, DW, WA, WD
budgets      = 0.0005, 0.001, 0.002
total formal = 3 × 6 = 18 fresh runs
```

Transfer indices:

```text
AD: source=0 target=1
AW: source=0 target=2
DA: source=1 target=0
DW: source=1 target=2
WA: source=2 target=0
WD: source=2 target=1
```

Frozen tuples — these must exactly match `OFFICE_OUT_LBI_FINAL_TUPLES.json`:

| rho_G | K_G | anchor | alpha | kappa | nu | omega | stage2_lr |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0.0005 | 4 | A1 | 0.05 | 1.0 | 0.50 | 0.10 | 0.010 |
| 0.001 | 9 | A1 | 0.05 | 1.0 | 0.50 | 0.10 | 0.0025 |
| 0.002 | 18 | A4 | 0.10 | 1.0 | 1.00 | 0.025 | 0.0025 |

Shared frozen LBI settings:

```text
support_threshold = 1e-4
stage1_max_steps  = 3000
stage2_steps      = 1
budget_tolerance  = 1e-4
delta_nonzero_tolerance = 1e-12
```

Candidate scope remains full ResNet-50 `netF.layer4` 9 Conv weights, with `out_channel` grouping and frozen controlled-Conv BN.

No `.005` budget. No parameter search. No per-transfer tuning.

---

## 3. Formal execution contract

Formal runs must satisfy:

```text
1 experiment process / GPU at a time
runtime_comparable = true
fresh run
runtime_resume_used = false
GPU = NVIDIA GeForce RTX 4090
PyTorch = 2.4.1
CUDA = 12.4
save_model = false
```

Do not pass `--resume` or `--resume-partial-runs`.

For this fresh Office formal, default to no stream-resume/checkpoint machinery unless the current canonical formal entrypoint requires metadata creation without resuming. In all cases the resulting summaries must report `runtime_resume_used=false`.

Before generating commands, inspect the already completed formal baseline artifacts and current runner/planner to determine the **existing supported mechanism** that makes `runtime_comparable=true`. Reuse that mechanism exactly. Do not invent an unsupported CLI flag.

Tuning / diagnostic runtime must not be copied into this formal root.

---

## 4. Repository safety

PREPARE is static only:

```text
NO training
NO GPU probe
NO nvidia-smi
NO retry
NO resume
NO rerun
```

Do not modify tracked scientific code.

Protected tracked paths include at least:

```text
core/lbi/engine.py
core/lbi/groups.py
shot_otta/trainer.py
shot_otta/losses.py
shot_otta/models.py
train.py
configs/otta_conv_lbi_protocol_20260826_v1.yaml
protocol/shot-otta_conv/**
```

At PREPARE start record:

```bash
git rev-parse --show-toplevel
git branch --show-current
git rev-parse HEAD
git status --short
```

Pre-existing experiment logs / prompts are not a reason to reset the repo. Never run:

```text
git reset
git clean
git stash
git add .
```

If any protected tracked scientific file differs from HEAD, **STOP and report it**. Do not silently overwrite or proceed.

For this task, create files only under the shared final-formal root above.

---

## 5. Plan construction rules

Construct only this node's assigned subset.

Each planned row must preserve the frozen scientific config and must have:

```text
dataset=office
seed=2026
variant=conv_out_lbi
group_mode=out_channel
requested_budget explicitly set
correct frozen tuple for that budget
runtime_comparable=true
save_model=false
unique experiment_key
unique experiment_config_sha256
unique expected output identity
```

Use the repository's canonical identity/config tooling.

Before launch, statically verify:

1. assigned row count exactly matches this prompt;
2. no duplicate `(rho_G, transfer)`;
3. every row belongs to the global 18-condition set;
4. tuple exactly matches frozen JSON;
5. `K_G` is 4 / 9 / 18 respectively;
6. no Filter-LBI;
7. no `.005`;
8. no resume flag;
9. one process per GPU at a time;
10. output root is under this formal root.

---

## 6. Launcher requirements

Create one node launcher under this node's control directory.

The launcher must:

- `cd` to the project root;
- use `conda run --no-capture-output -n SHOT_TTA` for every Python process;
- write stdout/stderr to node-local logs;
- launch one chain per GPU in background;
- within a GPU chain, run assigned jobs **sequentially** using `&&` or explicit exit-code checks;
- never overlap two experiments on the same GPU;
- `wait` for all four GPU chains;
- return non-zero if any assigned run fails.

Do not launch it during PREPARE.

---

## 7. PREPARE output

Write:

```text
<NODE_ROOT>/PREPARE.md
<NODE_ROOT>/plan.jsonl
<NODE_ROOT>/run_node_4gpu.sh
<NODE_ROOT>/logs/      # directory only
```

`PREPARE.md` must state:

```text
node id
assigned conditions
frozen tuple source
git root / branch / HEAD
protected-file audit
plan row count
unique key/SHA/output checks
runtime_comparable mechanism used
resume disabled
launcher path
exact manual launch command
SAFE_TO_LAUNCH = YES/NO
```

Run `bash -n` on the launcher.

Then print the exact manual launch command and STOP.

Do not train in PREPARE.

---

## 8. Post-run node audit mode

If explicitly invoked later with `MODE=FINALIZE`, do artifact-only checking of this node's assigned runs.

No launch / retry / resume / rerun.

Check each completed summary + per-batch metrics for:

```text
identity match
finite metrics
selected_group_count <= K_G every batch
no budget violation
no Stage-1 max-step failure
frozen controlled-Conv BN
runtime_comparable=true
runtime_resume_used=false
hardware/software provenance
PU / FO present
group utilization diagnostics present
selected scalar count / realized scalar ratio present
runtime + GPU memory fields present
```

Write:

```text
<NODE_ROOT>/FINALIZE.md
```

If a run is missing or invalid, report it. Do not repair or rerun it.


---

# NODE-SPECIFIC ASSIGNMENT — node2

Node root:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/office_conv_lbi_final_formal_out_channel_seed2026_20260829/control/node2
```

Node run root:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/office_conv_lbi_final_formal_out_channel_seed2026_20260829/runs/node2
```

This node must create **exactly 4 formal rows**.

| GPU | rho_G | transfer | source | target |
|---:|---:|---|---:|---:|
| 0 | 0.0005 | WA | 2 | 0 |
| 1 | 0.001 | WA | 2 | 0 |
| 2 | 0.002 | WA | 2 | 0 |
| 3 | 0.002 | AW | 0 | 2 |

GPU scheduling:

```text
- GPU0 : .0005 WA
- GPU1 : .001 WA
- GPU2 : .002 WA
- GPU3 : .002 AW
```

Important:

- This node has no sequential GPU chain; each GPU runs exactly one assigned experiment.
- Do not add a second run to any GPU.
- All other GPUs have one assigned run.
- No unassigned `(budget, transfer)` may be added.
- No assigned row may be moved to another node in PREPARE.

Set:

```text
<NODE_ROOT> = /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/office_conv_lbi_final_formal_out_channel_seed2026_20260829/control/node2
<RUNS_ROOT> = /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/office_conv_lbi_final_formal_out_channel_seed2026_20260829/runs/node2
```

The node launcher must write all experiment outputs under `<RUNS_ROOT>`.
