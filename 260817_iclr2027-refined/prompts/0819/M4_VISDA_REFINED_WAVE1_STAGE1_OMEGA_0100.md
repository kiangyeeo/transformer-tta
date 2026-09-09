Project root:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI
```

Current repo:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined
```

Conda:

```text
SHOT_TTA
```

Normative FC protocol:

```text
260817_iclr2027-refined/protocol/shot-otta_fc/OTTA_FC_LBI_PROTOCOL_20260817_v1.md
```

Scientific implementation revision:

```text
iclr2027_refined_20260817_v1
```

Shared VisDA search root for this new refined-code search:

```text
260817_iclr2027-refined/experiment_logs/visda_fc_lbi_seed2026_search_20260819
```

Modes:

```text
MODE=PREPARE
MODE=FINALIZE
```

Complete only the requested mode.

Hard rules:

- Read the protocol above first. For LBI algorithm semantics, the current frozen `260817_iclr2027-refined` code is the source of truth.
- Only add/modify experiment planning, launcher, report, and documentation files under `260817_iclr2027-refined/`.
- Do not modify LBI algorithm logic, the frozen protocol, `nips2026/`, or old-version code.
- Do not reuse old Seed-2020 VisDA numbers as current scientific results.
- Formal VisDA settings are the current protocol settings: ResNet-101, batch size 256, workers 4, seed 2026, one target pass, controlled-FC BN frozen, primary metric fixed-12-class mean per-class accuracy.
- FC sparse candidate is bottleneck weight+bias, with strict budgets `rho={0.0005,0.001,0.002}` and `K={262,524,1049}`.
- PU is report-only for tuning. Validity is checked before accuracy.
- Long LBI launchers must use the existing partial-stream resume mechanism and `--resume-partial-runs`.
- PREPARE is PURE PLAN: do not run training, do not call `nvidia-smi`, `torch.cuda`, GPU-count checks, model/data dry-runs, or launcher dry-runs. Static plan inspection/validation, `bash -n`, and `git diff --check` are allowed.
- FINALIZE is read-only over completed artifacts: do not launch, retry, or rerun training.
- Keep a complete reproducibility record: actual commands, environment/GPU allocation requested by the launcher, every scientific setting, plan/run identities and hashes, log/result paths, validity outcomes, and aggregation decisions.

# Machine 4 — Refined VisDA budget-0.001 Stage-1 grid at omega=0.10

## Goal

Run one 8-point Stage-1 dynamics grid for refined VisDA at a fixed persistent-writeback regime:

```text
budget = 0.001
omega = 0.10
stage2_lr = 0.005
stage2_steps = 1
stage1_max_steps = 3000
kappa = 1.0
```

Search only:

```text
alpha ∈ {0.10, 0.15, 0.20, 0.25}
nu    ∈ {0.5, 1.0}
```

Exactly 8 scientific identities.

This phase is designed to compare Stage-1 dynamics under one fixed omega. Do not expand the grid.

---

# MODE=PREPARE

Do not launch real training.

## 1. Build exact 8-run matrix

Create exactly these 8 configurations:

```text
S1: alpha=0.10, kappa=1.0, nu=1.0
S2: alpha=0.10, kappa=1.0, nu=0.5
S3: alpha=0.15, kappa=1.0, nu=1.0
S4: alpha=0.15, kappa=1.0, nu=0.5
S5: alpha=0.20, kappa=1.0, nu=1.0
S6: alpha=0.20, kappa=1.0, nu=0.5
S7: alpha=0.25, kappa=1.0, nu=1.0
S8: alpha=0.25, kappa=1.0, nu=0.5
```

All share:

```text
dataset=VISDA-C
seed=2026
batch_size=256
workers=4
variant=module_lbi
budget=0.001
K=524
omega=0.10
stage2_lr=0.005
stage2_steps=1
stage1_max_steps=3000
support_threshold=1e-4
```

Use the current refined implementation semantics from code/protocol.

Suggested paths:

```text
${SEARCH_ROOT}/matrices/wave1_m4_stage1_budget001_omega_0100/matrix.yaml
${SEARCH_ROOT}/plans/wave1_m4_stage1_budget001_omega_0100/
```

Statically verify:

```text
experiment_count = 8
all identities are unique
all have budget=0.001 and K=524
all have omega=0.10
all have stage2_lr=0.005
```

Do not import/reuse old Seed-2020 runs.

## 2. Non-interactive launcher

Create:

```text
260817_iclr2027-refined/tools/run_visda_wave1_m4_stage1_omega_0100_noninteractive.sh
```

Requirements:

```text
set -euo pipefail
8 GPUs: 0,1,2,3,4,5,6,7
max-workers=8
workers-per-gpu=1
conda env=SHOT_TTA
```

Reuse the existing multi-GPU launcher in the current repo.

The launcher must use:

```text
--resume
--resume-partial-runs
```

so completed matching identities are skipped and interrupted long LBI runs resume from compatible batch-boundary checkpoints.

Keep the shell attached until all 8 jobs finish. Do not use tmux.

Use the actual existing launcher path confirmed from the repo; do not invent a new runner.

Run only static checks:

```bash
bash -n 260817_iclr2027-refined/tools/run_visda_wave1_m4_stage1_omega_0100_noninteractive.sh
git diff --check
```

Do not execute the launcher.

## 3. Record

Write:

```text
${SEARCH_ROOT}/phase_records/wave1_m4/PREPARE.md
```

Record exact commands, environment, requested GPU allocation, all hyperparameters, matrix/plan paths, scientific identities/hashes, result/log paths, and that PREPARE launched no training.

At the end print exactly:

```bash
cd /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI

bash 260817_iclr2027-refined/tools/run_visda_wave1_m4_stage1_omega_0100_noninteractive.sh
```

Then stop.

---

# MODE=FINALIZE

Run only after the Machine-4 launcher returns.

Do not launch/retry/rerun training.

## 1. Completion

Check all 8 planned runs and report:

```text
completed
missing
failed
duplicate
hash mismatch
invalid summary
```

## 2. Aggregate

For all 8 rows report:

```text
candidate
alpha
kappa
nu
omega
stage2_lr
scientific_valid
hard90_valid
min / P05 / mean support utilization
<95% batch count
max_steps_hit_count
stage1_steps_completed_mean
stage1_steps_completed_max
FO mean-per-class
FO worst-class
FO class-wise std
FO overall
PU mean-per-class
runtime diagnostics if already available
```

Eligibility order:

```text
1. scientific validity
2. hard90 utilization
3. FO mean-per-class descending
4. FO worst-class descending
5. FO class-wise std ascending
6. FO overall descending
7. Stage-1 max steps ascending
8. Stage-1 mean steps ascending
```

PU is report-only.

You may identify this machine's top eligible rows, but do not freeze a global VisDA configuration. Global selection waits for Machines 1–4.

Write:

```text
${SEARCH_ROOT}/phase_records/wave1_m4/FINALIZE.md
```

Stop.
