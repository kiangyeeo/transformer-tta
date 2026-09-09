# CODEX — COME Office LBI Full-Data Tuning, κ=1 — 8 GPU — SUBMIT AND STOP

## Goal

今天完成 **COME × Office-31 × LBI** tuning，并最终冻结 6 个 Office tuples：

```text
FC   rho=.0005/.001/.002
Conv rho=.0005/.001/.002
```

搜索纪律保持不变：

```text
kappa = 1 fixed
Stage-1 GT = (alpha, nu)
Office 六个 directed transfers
完整 target stream
seed=2026
R0 full D->A
R1 all-six full-stream Stage-1 validation
R2 all-six full-stream joint omega × stage2_lr
one-time upper-bound expansion
```

本 prompt 的执行目标不是等待全部实验跑完。

**你只负责：创建/校验可 resume 的持久 launcher → 把任务提交到 GPU 0–7 → 确认 8 张卡都有 active workload → 打印 ETA / resume 信息 → 立即停止 Codex。**

不要等待 tuning 完成。

---

# 0. Current frozen state

```text
COME baseline implementation:
come_otta_baseline_20260908_v2

COME sparse/LBI implementation:
come_otta_sparse_lbi_20260908_v3

source revision:
nips2026_shot_otta_uda_source_v1
```

Do not modify scientific implementation.

---

# 1. Safety under danger-full-access

Codex may run with:

```toml
approval_policy = "never"
sandbox_mode = "danger-full-access"
```

Therefore NEVER run:

```text
rm / rm -rf
git clean
git reset
git reset --hard
git checkout -- .
git restore
git stash
delete/move existing runs
delete/move existing experiment_logs
delete protocols/checkpoints/datasets
broad killall/pkill
```

Do not touch:

```text
shot_otta/**
ist_otta/**
nctta_otta/**
core/lbi/**
come_otta/** scientific implementation
datasets/**
checkpoints/**
```

Only create NEW tuning/launcher/log/summary/config artifacts.

If a scientific implementation bug is found:

```text
do not patch it
record blocker
```

---

# 2. GPU requirement — OCCUPY ALL 8 GPUs

Use ONLY:

```text
GPU 0
GPU 1
GPU 2
GPU 3
GPU 4
GPU 5
GPU 6
GPU 7
```

Before returning control to the user, you MUST verify:

```text
GPU0 has active tuning workload
GPU1 has active tuning workload
GPU2 has active tuning workload
GPU3 has active tuning workload
GPU4 has active tuning workload
GPU5 has active tuning workload
GPU6 has active tuning workload
GPU7 has active tuning workload
```

Initial policy:

```text
>= 1 active LBI job / GPU
```

If one LBI process underutilizes a GPU and memory permits, launcher may use:

```text
max 2 concurrent jobs / GPU
```

Hard global maximum:

```text
16 concurrent jobs
```

The launcher should dynamically refill completed slots.

Do not alter BS/scientific settings to raise utilization.

---

# 3. Persistent background launcher

The tuning must continue after this Codex session stops.

Create a NEW task directory, for example:

```text
260817_iclr2027-refined/experiment_logs/
come_office_lbi_fullsearch_kappa1_20260909/
```

Create a persistent launcher that can run independently after Codex exits.

Allowed mechanisms include a repository/user-environment-appropriate persistent process such as:

```text
nohup
tmux
screen
```

Use the least invasive available mechanism.

Do not modify shell startup/config files.

Launcher must persist:

```text
queue state
running identities
completed identities
failed identities
retry count
PID
GPU assignment
artifact path
stdout/stderr path
timestamps
```

Suggested state file:

```text
launcher_state.json
```

and queue manifest:

```text
tuning_queue.jsonl
```

---

# 4. Resume semantics — REQUIRED

Support safe resume.

Resume means:

```text
launcher / identity-level resume
```

NOT:

```text
partial model-state resume
```

On restart/resume:

### Completed identity

Skip only if all are true:

```text
scientific identity exact match
expected implementation revision exact
run complete
artifact complete
summary/result exists
no correctness failure
```

### Running identity whose process is still alive

Do not duplicate it.

### Interrupted / partial identity

Do NOT resume model/optimizer state from the middle of the target stream.

Instead:

```text
requeue the exact identity
restart from the same source checkpoint
same seed
same stream
same scientific settings
```

Preserve the partial artifact/log; do not delete it.

Write the new attempt to a distinct attempt/run path.

### Failed identity

Retry only according to the predeclared policy.

No silent parameter changes.

---

# 5. Resume command

Create an explicit resume entry point, for example:

```bash
bash experiment_logs/come_office_lbi_fullsearch_kappa1_20260909/resume.sh
```

or:

```bash
python experiment_logs/come_office_lbi_fullsearch_kappa1_20260909/launcher.py --resume
```

The exact command must be printed before Codex stops.

Resume must be idempotent:

```text
safe to invoke multiple times
no duplicate completed identities
no duplicate currently-running identities
```

---

# 6. Full scientific search protocol

## Office transfers

```text
A->D
A->W
D->A
D->W
W->A
W->D
```

Every final tuning evaluation uses the complete target stream.

Frozen:

```text
ResNet-50
BS=64
workers=4
seed=2026
same SHOT source F/B/C
one target pass
```

Tracks:

```text
FC scalar
Conv out-channel
```

Budgets:

```text
rho=.0005/.001/.002
```

Fix:

```text
kappa=1
```

Standard Stage-1 domain:

```text
alpha ∈ {.025,.05,.10,.125,.15,.20}
nu    ∈ {.25,.50,1.00}
```

---

# 7. R0 — FULL D->A Stage-1 reachability

Use complete D->A stream.

Fixed diagnostic downstream values:

```text
omega=.00625
stage2_lr=.005
```

Initial anchors:

```text
A0 (.10,.50)
A1 (.05,.50)
A2 (.20,.50)
A3 (.10,.25)
A4 (.10,1.00)
A5 (.125,.50)
A6 (.15,.50)
A7 (.025,.50)
```

For each:

```text
2 tracks × 3 budgets
```

run all 8.

Accuracy cannot choose Stage-1 anchors.

Scientific-valid requires:

```text
finite
strict budget respected
cap_hit_count=0
off-mask exact
BN/netC exact
objective-call accounting exact
```

Utilization eligibility:

FC:

```text
mean >= .90
p05 >= .90
```

Conv:

```text
mean >= .90
p05 >= .75
```

If fewer than 2 eligible anchors, evaluate missing standard-domain neighbors:

```text
(.05,.25)
(.05,1.00)
(.125,.25)
(.125,1.00)
(.15,.25)
(.15,1.00)
(.20,.25)
(.20,1.00)
(.025,.25)
(.025,1.00)
```

No alpha outside the standard domain.

---

# 8. R1 — ALL SIX FULL-STREAM Stage-1 validation

For every track×budget:

evaluate R0 candidates in Stage-1 ranking order on all six complete Office transfers.

Exact R0 D->A rows may be reused if identity is identical.

Candidate must be:

```text
scientific-valid all six
cap-hit free all six
utilization-eligible all six
```

Rank without accuracy:

```text
worst-transfer p05 utilization
mean utilization
worst S1 max steps
mean S1 steps
runtime
```

Freeze one `(alpha,nu)` per:

```text
FC .0005/.001/.002
Conv .0005/.001/.002
```

No downstream accuracy can change frozen `(alpha,nu)`.

---

# 9. Stage-1 cap definition

Frozen:

```text
stage1_max_steps = 3000
```

If Stage-1 reaches 3000 inner steps before normal strict-budget termination:

```text
cap_hit = true
```

Cap-hit candidate is invalid for formal Stage-1 selection.

Do not increase the cap.

---

# 10. R2 — JOINT omega × Stage2 LR, ALL SIX FULL DATA

After `(alpha,nu)` freeze, for every track×budget run:

```text
omega     ∈ {.00625,.025,.10}
stage2_lr ∈ {.0025,.005,.010}
```

Complete joint 3×3 grid.

Each cell:

```text
all six Office transfers
complete target stream
```

Reuse exact R1 center:

```text
omega=.00625
stage2_lr=.005
```

only if identity is exactly identical.

Eligibility:

```text
6/6 scientific-valid
6/6 cap-hit free
6/6 utilization-eligible
```

Selection:

```text
1. six-transfer equal-weight FO descending
2. worst-transfer FO descending
3. worst-transfer margin vs same-budget sparse baseline descending
4. S1 max ascending
5. S1 mean ascending
6. runtime ascending
```

PU report-only.

---

# 11. Baseline targets

FC:

```text
.0005 sparse ref FO = 77.755480
.001  sparse ref FO = 77.901204
.002  sparse ref FO = 77.928241

FC module-dense FO = 78.931319
```

Conv:

```text
.0005 sparse ref FO = 77.675195
.001  sparse ref FO = 77.818373
.002  sparse ref FO = 77.860458

Conv module-dense FO = 79.152169
```

Goal:

```text
beat same-budget sparse baseline
and maximize six-transfer FO
```

---

# 12. One-time upper-bound expansion

Only if initial R2 winner has:

```text
omega=.10
and/or
stage2_lr=.010
```

allow exactly one local expansion:

```text
omega upper = .30
stage2_lr upper = .020
```

If both upper boundaries trigger, test:

```text
(.30, selected_lr)
(selected_omega, .020)
(.30,.020)
```

All six transfers, full stream.

No second expansion.

---

# 13. Final tuple artifacts

When the full tuning eventually completes, freeze:

```text
FC .0005
FC .001
FC .002
Conv .0005
Conv .001
Conv .002
```

to NEW files:

```text
selected_configs/COME_OFFICE_LBI_FINAL_TUPLES_20260909_v1.json
selected_configs/COME_OFFICE_LBI_FINAL_TUPLES_20260909_v1.md
```

Final-formal LBI is a separate task and must NOT be started by this launcher.

---

# 14. Launcher phase dependency

The launcher must understand dependencies:

```text
R0
↓
select R0 ranked candidates
↓
R1
↓
freeze alpha/nu
↓
R2 3×3
↓
optional boundary expansion
↓
freeze final tuples
↓
STOP
```

Do not queue R2 cells before the relevant Stage-1 tuple is frozen.

The persistent launcher may automatically advance phases when the required artifacts are complete.

---

# 15. GPU scheduling

Use all 8 GPUs from the beginning.

Preferred policy:

```text
start 8 independent R0 jobs immediately
one per GPU
```

Then refill continuously.

If utilization is low and memory permits:

```text
increase to 2 jobs/GPU
```

Do not exceed 2/GPU.

Mix tracks/budgets to avoid all GPUs getting similarly slow rows.

---

# 16. Failure handling

Transient/OOM:

```text
retry exact same identity once
```

No scientific changes.

Scientific failure/cap hit:

```text
record candidate outcome
continue other predeclared candidates
```

Do not patch code.

Do not delete failed/partial artifacts.

---

# 17. SUBMIT-AND-STOP behavior

This is critical.

Once:

```text
persistent launcher is created
queue/state is written
GPU0-7 all have active workload
launcher persistence is confirmed
```

DO NOT wait for experiments to finish.

Before stopping, perform only a quick read-only check:

```text
launcher process alive
8 GPUs assigned active jobs
no immediate launcher crash
state file updating
```

Then immediately return a status report and STOP Codex.

Do not sit in a monitoring loop.

---

# 18. ETA estimation before stopping

Before Codex stops, estimate completion time.

Use the best available evidence in this order:

1. existing previous COME/SHOT LBI tuning artifact runtimes for comparable Office full-stream jobs;
2. elapsed/runtime of any already-started current jobs if enough data exists;
3. queue size + measured completed-job throughput;
4. if insufficient evidence, provide a clearly labeled rough range.

ETA must account for:

```text
R0 workload
possible standard-domain rescue
R1 all-six rows
R2 3×3 joint grid
possible one-time boundary expansion
actual active concurrency
```

Report:

```text
estimated wall-clock remaining
estimated completion timestamp in local/server time
assumptions used
confidence: high / medium / low
```

Do not invent a precise ETA without runtime evidence.

---

# 19. Required immediate final response after submission

Once jobs are submitted, report ONLY the launch/status information below.

## A. Launcher

```text
launcher PID/session
launcher path
state file
queue file
log path
```

## B. GPU occupancy

For each:

```text
GPU0 active jobs
GPU1 active jobs
GPU2 active jobs
GPU3 active jobs
GPU4 active jobs
GPU5 active jobs
GPU6 active jobs
GPU7 active jobs
```

Confirm:

```text
ALL_8_GPUS_OCCUPIED = YES
```

## C. Queue

```text
current phase
pending
running
completed
failed
```

## D. Resume

Print the exact resume command.

Explain:

```text
completed identities are skipped
alive jobs are not duplicated
partial interrupted identities restart from source
no partial model-state resume
```

## E. ETA

Print:

```text
estimated remaining wall-clock
estimated finish timestamp
basis
confidence
```

## F. Safety

Confirm:

```text
scientific implementation modified = false
existing artifacts deleted/overwritten = 0
```

## G. Final status

Print exactly:

```text
COME_OFFICE_LBI_TUNING_SUBMITTED
```

Then STOP.

Do NOT wait for `COME_OFFICE_LBI_TUNING_COMPLETE`.
