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

# Machine 1 — Refined VisDA baselines, then budget-0.001 omega bridge

## Goal

Use this machine for two sequential phases in one non-interactive shell:

```text
Phase A: current refined VisDA formal baselines
Phase B: budget=0.001 D4/D5 omega bridge
```

The generated shell must run Phase A first and automatically start Phase B only after Phase A returns successfully. Keep the shell attached until Phase B finishes. Do not use tmux.

---

# MODE=PREPARE

Do not launch real training.

## 1. Phase A baseline matrix

Create a baseline matrix/plan under the shared search root containing exactly these 12 top-level scientific identities:

```text
source_only
full_dense
module_dense

module_random    @ rho=0.0005
module_random    @ rho=0.001
module_random    @ rho=0.002

module_magnitude @ rho=0.0005
module_magnitude @ rho=0.001
module_magnitude @ rho=0.002

module_saliency  @ rho=0.0005
module_saliency  @ rho=0.001
module_saliency  @ rho=0.002
```

`module_random` must use the protocol's 3 deterministic child masks inside each top-level identity.

Use current refined Seed-2026 settings only. Do not import old Seed-2020 baseline results.

Suggested paths:

```text
${SEARCH_ROOT}/matrices/wave1_m1_baselines/matrix.yaml
${SEARCH_ROOT}/plans/wave1_m1_baselines/
```

Statically verify:

```text
top_level_experiment_count = 12
dataset = VISDA-C
seed = 2026
batch_size = 256
```

Also verify the three strict K values are 262 / 524 / 1049.

## 2. Phase B omega bridge matrix

Create exactly 8 LBI runs at:

```text
budget=0.001
stage2_lr=0.005
stage2_steps=1
stage1_max_steps=3000
kappa=1.0
```

Anchors:

```text
D4: alpha=0.15, nu=1.0
D5: alpha=0.15, nu=0.5
```

For each anchor run:

```text
omega ∈ {0.0125, 0.05, 0.20, 0.30}
```

Thus:

```text
2 anchors × 4 omega = 8 runs
```

Suggested paths:

```text
${SEARCH_ROOT}/matrices/wave1_m1_omega_bridge_budget001/matrix.yaml
${SEARCH_ROOT}/plans/wave1_m1_omega_bridge_budget001/
```

These are diagnostic/tuning runs, not final formal LBI evaluation.

## 3. Non-interactive launcher

Create:

```text
260817_iclr2027-refined/tools/run_visda_wave1_m1_baselines_bridge_noninteractive.sh
```

Requirements:

```text
set -euo pipefail
8 GPUs: 0,1,2,3,4,5,6,7
max-workers=8
workers-per-gpu=1
conda env=SHOT_TTA
```

Reuse the existing multi-GPU launcher in the current repo; do not create a new experiment runner.

The shell must:

1. run the 12 baseline identities with one experiment process per GPU;
2. after successful completion, run the 8 bridge LBI identities on the same 8 GPUs;
3. use distinct launcher-log directories for baseline and bridge;
4. use `--resume` so already completed matching identities are skipped;
5. for Phase B additionally use `--resume-partial-runs`;
6. not use `exec` for the first command because the second phase must run afterward; using `exec` for the final bridge command is fine;
7. remain non-interactive and attached until all jobs finish.

Use the actual planner/launcher paths confirmed from the repo. Do not guess a nonexistent path.

Run:

```bash
bash -n 260817_iclr2027-refined/tools/run_visda_wave1_m1_baselines_bridge_noninteractive.sh
git diff --check
```

Do not execute the launcher.

## 4. Record

Write:

```text
${SEARCH_ROOT}/phase_records/wave1_m1/PREPARE.md
```

Record exact matrices, plans, counts, scientific settings, identities/hashes, launcher/log/result paths, and the fact that PREPARE launched no training.

At the end print exactly:

```bash
cd /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI

bash 260817_iclr2027-refined/tools/run_visda_wave1_m1_baselines_bridge_noninteractive.sh
```

Then stop.

---

# MODE=FINALIZE

Run only after the Machine-1 launcher returns.

Do not launch/retry/rerun training.

## 1. Baselines

Check the 12 top-level planned identities and report:

```text
completed
missing
failed
duplicate
hash mismatch
invalid summary
```

Aggregate current refined Seed-2026 PU/FO metrics.

For each sparse budget compute:

```text
best sparse = max(Random 3-mask mean, Magnitude, Saliency)
```

using VisDA primary FO mean-per-class accuracy.

Keep source-only, module-dense, and full-dense separate.

## 2. Omega bridge

Check all 8 bridge runs.

For each row report at least:

```text
anchor
alpha
kappa
nu
omega
stage2_lr
scientific_valid
hard90_valid
min / P05 / mean support utilization
max_steps_hit_count
stage1_steps_completed_mean
stage1_steps_completed_max
FO mean-per-class
FO worst-class
FO class-wise std
FO overall
PU mean-per-class
```

Also summarize early/mid/late stream diagnostics if the current artifacts already expose them; do not invent new metrics or rerun anything.

Do not freeze the final omega or Stage-1 tuple here. This is only Machine-1 evidence; global selection waits for Machines 2–4.

Write:

```text
${SEARCH_ROOT}/phase_records/wave1_m1/FINALIZE.md
```

Stop.
