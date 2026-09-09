# CODEX — COME Office LBI Tuning, κ=1 — 5 GPU

## Goal

只做 **COME + Office-31 + LBI tuning**。

目标：

```text
1. 固定 kappa = 1
2. 先找到每个 track × budget 的 Stage-1 support-discovery GT:
   (alpha, nu)
3. 冻结 (alpha, nu)
4. 再调 downstream calibration:
   (omega, stage2_lr)
5. 最终冻结 6 个 Office tuples
6. LBI 尽可能打败 same-budget sparse baseline，并让 six-transfer FO 尽可能高
```

本任务：

```text
不跑 VisDA
不跑最终 formal LBI
不改 COME objective
不改 LBI semantics
不改 sparse baselines
不做新的 baseline
```

---

# 0. Current frozen parent state

COME stable objective / baseline：

```text
COME baseline implementation:
come_otta_baseline_20260908_v2

COME sparse/LBI implementation:
come_otta_sparse_lbi_20260908_v3

Source revision:
nips2026_shot_otta_uda_source_v1
```

Parent protocols：

```text
260817_iclr2027-refined/protocol/come-otta/OTTA_COME_BASELINE_PROTOCOL_20260907_v1.md
260817_iclr2027-refined/protocol/come-otta/OTTA_COME_LBI_PROTOCOL_20260907_v1.md
```

Stable COME mathematical semantics不可改变：

```text
p=2
tau=1
Office K=31
opinion_eps=1e-7
stable log-domain opinion evaluation
same constrained-logit norm.detach semantics
```

---

# 1. Danger-full-access safety

Current Codex environment may use：

```toml
approval_policy = "never"
sandbox_mode = "danger-full-access"
```

因此：

### 禁止

```text
rm / rm -rf
git clean
git reset
git reset --hard
git checkout -- .
git restore
git stash
删除/覆盖已有 runs
删除/覆盖 experiment_logs
删除 protocol
删除 checkpoint/dataset
移动已有用户文件
killall/pkill broad pattern
清理其他任务产生的未跟踪文件
```

不要修改：

```text
shot_otta/**
ist_otta/**
nctta_otta/**
core/lbi/**
datasets/**
checkpoints/**
```

不要修改 COME scientific implementation。

本任务原则上只创建：

```text
Office LBI tuning protocol
tuning configs
launcher/scripts
tuning artifacts
selected tuple files
summary files
```

若 tuning execution 暴露 correctness bug，需要修改 scientific implementation：

```text
STOP
报告 blocker
不要现场修算法
```

---

# 2. GPU resources

Only use：

```text
GPU 0
GPU 1
GPU 2
GPU 3
GPU 4
```

禁止使用其他 GPU。

默认：

```text
每张 GPU 先跑 1 个 LBI tuning job
global 5-way concurrency
```

LBI Stage-1 计算密集，优先一张卡一个 job。

如果：

```text
pending queue != empty
且某 GPU 长期 utilization <30%
且显存明显充足
```

允许该 GPU 增加第 2 个 tuning job。

硬上限：

```text
max 2 jobs / GPU
max 10 concurrent jobs
```

不要通过修改：

```text
batch size
model
stage1 cap
algorithm
```

提高利用率。

Job 完成后立即 refill。

仅终止本任务 launcher 明确记录的 PID。

---

# 3. Office matrix

Office transfers：

```text
A->D
A->W
D->A
D->W
W->A
W->D
```

Frozen：

```text
ResNet-50
BS=64
workers=4
seed=2026
one target pass
same SHOT source F/B/C
same target stream
same singleton rule
PU read-only
FO read-only
```

Tracks：

```text
FC scalar
Conv out-channel
```

Budgets：

```text
rho=.0005
rho=.001
rho=.002
```

FC：

```text
candidate scalars = 524544
K = 262 / 524 / 1049
```

Conv：

```text
grouping = out_channel only
groups = 9216
K_G = 4 / 9 / 18
```

`filter_connection` remains forbidden.

---

# 4. Fixed LBI semantics

Do not change：

```text
support_threshold = 1e-4
stage1_max_steps = 3000
strict integer budget
strict rollback
no top-K repair
corrected old-state Z update
masked-delta Stage2 init
stage2_steps = 1
Stage2 SGD
momentum=.9
weight_decay=.001
Nesterov=true
Stage2 scheduler=none
fresh Stage2 optimizer each outer batch
off-mask exact preservation
omega-only persistent writeback
host persistent optimizer step=0
```

Current-state COME objective must be recomputed at every Stage-1 candidate.

---

# 5. Hyperparameter decomposition

This search explicitly defines：

$$
h_{S1}=(\alpha,\nu), \qquad \kappa \equiv 1.
$$

`kappa` is NOT a search axis.

Downstream calibration：

```text
omega
stage2_lr
```

is tuned only after Stage-1 `(alpha,nu)` is frozen.

Do not jointly sweep all four dimensions.

---

# PART A — STAGE-1 SUPPORT DISCOVERY

# 6. Stage-1 diagnostic downstream constants

During Stage-1 `(alpha,nu)` search, fix：

```text
kappa = 1
omega = .025
stage2_lr = .005
```

These values are diagnostic only.

Accuracy is NOT used to select `(alpha,nu)`.

---

# 7. Prefix screen dataset

Use only：

```text
Office D->A
seed=2026
first 5 valid outer batches
formal=false
debug/tuning artifact only
```

This is a tuning prefix, not a correctness smoke.

Do not use other transfers at this stage.

---

# 8. Exact Stage-1 grids

## FC

For every FC budget：

```text
alpha ∈ {.05, .10, .15, .20}
nu    ∈ {.25, .50, 1.00}
kappa = 1
```

Exact Cartesian grid：

```text
12 candidates / budget
36 FC prefix conditions total
```

## Conv

For every Conv budget：

```text
alpha ∈ {.025, .05, .10, .20}
nu    ∈ {.50, 1.00}
kappa = 1
```

Exact Cartesian grid：

```text
8 candidates / budget
24 Conv prefix conditions total
```

Total initial prefix：

```text
60 five-valid-batch conditions
```

Do not add other Stage-1 candidates before evaluating this exact grid.

---

# 9. Prefix validity / ranking

For every batch record：

```text
support count
budget
utilization
stage1 steps
cap hit
rollback
objective call count
finite loss/grad/update
off-mask violations
```

Candidate is prefix-valid only if：

```text
finite/correct on all 5 batches
cap hits = 0
support <= strict budget every batch
off-mask violations = 0
BN/netC violations = 0
objective-call accounting exact
```

Utilization：

FC：

```text
u_t = selected_scalar_count / K
```

Conv：

```text
u_t = selected_group_count / K_G
```

Accuracy is report-only and MUST NOT enter Stage-1 ranking.

Rank valid candidates by：

```text
1. p05 utilization descending
2. mean utilization descending
3. minimum utilization descending
4. max Stage1 steps ascending
5. mean Stage1 steps ascending
6. rollback rate ascending
```

For each：

```text
track × budget
```

freeze：

```text
primary Stage1 candidate
backup Stage1 candidate
```

---

# 10. Pre-registered Stage-1 rescue

Only if a track×budget cell has **no prefix-valid candidate** with：

FC：

```text
mean utilization >= .80
```

Conv：

```text
mean utilization >= .75
```

run one rescue set.

## FC rescue

```text
(alpha=.30, nu=.25)
(alpha=.30, nu=.50)
(alpha=.40, nu=.25)
(alpha=.40, nu=.50)
```

## Conv rescue

```text
(alpha=.30, nu=.50)
(alpha=.30, nu=1.00)
(alpha=.40, nu=.50)
(alpha=.40, nu=1.00)
```

Same 5-batch D->A prefix.

After this rescue：

```text
do not expand alpha/nu again in this protocol
```

If still no valid candidate with the threshold above：

```text
mark STAGE1_BLOCKED
stop that cell
```

---

# 11. Full Office Stage-1 validation

For each of the 6 cells：

```text
FC .0005/.001/.002
Conv .0005/.001/.002
```

run the frozen primary `(alpha,nu)` over **all six Office transfers**.

Keep：

```text
kappa=1
omega=.025
stage2_lr=.005
```

Accuracy remains report-only.

Full validation eligibility：

### FC

On every transfer：

```text
cap hits = 0
mean utilization >= .90
p05 utilization >= .85
scientific validity PASS
```

### Conv

On every transfer：

```text
cap hits = 0
mean utilization >= .85
p05 utilization >= .70
scientific validity PASS
```

If primary fails any transfer：

```text
run backup on all six transfers
```

Freeze the first candidate that passes all six.

If both fail：

```text
STOP that cell
report blocker
```

After this step, the Office Stage-1 GT is frozen：

```text
(alpha, nu), kappa=1
```

for each track×budget.

No later accuracy result may change `(alpha,nu)`.

---

# PART B — DOWNSTREAM CALIBRATION

# 12. Baseline targets

Frozen same-budget sparse baseline FO references：

## FC Office

```text
rho=.0005 -> FC Saliency FO = 77.755480
rho=.001  -> FC Saliency FO = 77.901204
rho=.002  -> FC Saliency FO = 77.928241
```

Matched FC module-dense：

```text
FO = 78.931319
```

## Conv Office

```text
rho=.0005 -> Conv Saliency FO = 77.675195
rho=.001  -> Conv Saliency FO = 77.818373
rho=.002  -> Conv Saliency FO = 77.860458
```

Matched Conv module-dense：

```text
FO = 79.152169
```

Primary scientific goal：

```text
beat same-budget best sparse baseline
```

Among eligible candidates：

```text
maximize six-transfer equal-weight FO
```

Module-dense is a secondary reference, not a hard ceiling.

---

# 13. Omega search

After `(alpha,nu)` is frozen：

```text
stage2_lr = .005
```

## FC omega grid

```text
omega ∈ {.025, .10, .30}
```

## Conv omega grid

```text
omega ∈ {.0125, .025, .05, .10, .20}
```

For every：

```text
track × budget × omega
```

run all six Office transfers.

Do not tune per transfer.

Eligibility：

```text
all six runs scientific-valid
no cap hits
Stage1 full-validation utilization requirements still satisfied
```

Selection metric：

```text
six-transfer equal-weight FO
```

Tie-break：

```text
1. worst-transfer FO higher
2. worst-transfer margin vs same-budget sparse baseline higher
3. fewer Stage1 steps
4. lower runtime
```

PU is report-only.

Freeze one provisional omega per track×budget.

---

# 14. Stage2 LR search

At frozen provisional omega：

```text
stage2_lr ∈ {.0025, .005, .010, .020}
```

Reuse the `.005` omega-search result.

Run only missing LR values over all six transfers.

Same eligibility and ranking.

Freeze provisional：

```text
omega*
stage2_lr*
```

---

# 15. Pre-registered local interaction check

Sequential omega→LR search can miss a local interaction.

Therefore after provisional `(omega*, stage2_lr*)`：

identify the nearest lower and upper omega neighbors from the original omega grid.

At the selected `stage2_lr*`：

```text
run each available adjacent omega neighbor
```

over all six transfers, unless that exact pair was already evaluated.

Examples：

FC omega grid：

```text
.025 < .10 < .30
```

Conv omega grid：

```text
.0125 < .025 < .05 < .10 < .20
```

Maximum two extra omega values per cell.

Rank：

```text
provisional pair
+ local neighbor pairs
```

using the same six-transfer FO rule.

Freeze the final `(omega, stage2_lr)`.

No further interaction search.

---

# 16. One upper-bound expansion

Only if the final selected value is on an upper search boundary：

```text
FC omega=.30
Conv omega=.20
or stage2_lr=.020
```

perform one predeclared expansion.

If FC omega upper-bound：

```text
omega=.50
```

If Conv omega upper-bound：

```text
omega=.30
```

If LR upper-bound：

```text
stage2_lr=.040
```

Evaluate only the needed local expanded pair(s) across all six transfers.

After one expansion：

```text
STOP expansion even if winner remains on boundary
```

No lower-bound expansion.

---

# 17. Final tuple selection

For each track×budget：

final tuple is：

```text
alpha    = frozen Stage1 GT
kappa    = 1
nu       = frozen Stage1 GT
omega    = best calibrated value
stage2_lr= best calibrated value
```

Primary ranking：

```text
six-transfer equal-weight FO descending
```

Require scientific validity across all six transfers.

Always report：

```text
FO
PU
gain vs source
gain vs same-budget best sparse baseline
gap vs matched module-dense
worst-transfer FO
support utilization
Stage1 steps
runtime
```

---

# 18. Final 6 Office tuples

Freeze exactly：

```text
FC .0005
FC .001
FC .002

Conv .0005
Conv .001
Conv .002
```

Create NEW files：

```text
260817_iclr2027-refined/selected_configs/COME_OFFICE_LBI_FINAL_TUPLES_20260908_v1.json
260817_iclr2027-refined/selected_configs/COME_OFFICE_LBI_FINAL_TUPLES_20260908_v1.md
```

Also create：

```text
260817_iclr2027-refined/experiment_logs/come_office_lbi_tuning_20260908/
COME_OFFICE_LBI_TUNING_SUMMARY.md
COME_OFFICE_LBI_TUNING_SUMMARY.csv
COME_OFFICE_LBI_TUNING_MANIFEST.jsonl
```

Do not overwrite existing artifacts.

---

# 19. Search artifacts are NOT final formal

All tuning runs：

```text
formal=false
tuning=true
```

Do not enable final COME-LBI formal gate.

Do not create paper-final LBI results in this task.

After tuple freeze, final formal will be a separate fresh task：

```text
2 tracks × 3 budgets × 6 transfers
= 36 fresh Office formal LBI runs
```

---

# 20. No leakage / fairness

Online adaptation never uses target labels.

Offline tuning may use FO accuracy only in the downstream calibration phase：

```text
omega / stage2_lr
```

Stage-1 `(alpha,nu)` selection：

```text
MUST NOT use accuracy
```

Office has one shared tuple per：

```text
track × budget
```

across all six transfers.

Per-transfer tuning is forbidden.

---

# 21. Launcher / utilization

Create a task-specific launcher under：

```text
experiment_logs/come_office_lbi_tuning_20260908/
```

Queue states：

```text
pending
running
complete
failed
```

Use GPU：

```text
0,1,2,3,4 only
```

Start：

```text
5 jobs immediately
```

Refill immediately.

If utilization policy requires and memory permits：

```text
up to 2 jobs/GPU
```

Record every ~5 minutes：

```text
GPU utilization
memory used
active jobs
pending jobs
```

Do not modify scientific settings for utilization.

---

# 22. Failure policy

Transient/OOM：

```text
retry same identity once
same scientific settings
```

Correctness failure：

```text
record
do not silently repair
continue unrelated healthy cells
```

If same correctness failure occurs in >=2 candidates from same cell：

```text
stop that cell
report blocker
```

Do not modify LBI engine / COME objective during tuning.

---

# 23. Required summary

Final report must include：

## A. Stage-1 GT table

| Track | Budget | alpha | kappa | nu | utilization | Stage1 steps |
|---|---:|---:|---:|---:|---:|---:|

Exactly 6 rows.

## B. Downstream calibration table

For each cell list all tested：

```text
omega
stage2_lr
six-transfer FO
worst-transfer FO
validity
```

## C. Final 6 tuples

Include：

```text
alpha
kappa=1
nu
omega
stage2_lr
FO
same-budget sparse baseline
LBI - sparse baseline
module-dense reference
```

## D. Win summary

For each budget/track：

```text
LBI beats same-budget sparse baseline? yes/no
margin
```

## E. Isolation

Confirm：

```text
SHOT untouched
IST untouched
NCTTA untouched
core/lbi untouched
COME scientific implementation untouched
```

## F. GPU

Report：

```text
GPU 0–4 only
peak concurrency
OOM/retries
utilization log
```

---

# 24. STOP condition

Stop after：

```text
all six Stage1 GTs frozen
all six downstream calibration pairs frozen
six final Office tuples written
tuning summary generated
```

Do NOT continue to：

```text
Office final formal LBI
VisDA LBI tuning
VisDA formal
paper tables
new method development
```

Final verdict only：

```text
COME_OFFICE_LBI_TUNING_COMPLETE
```

or：

```text
COME_OFFICE_LBI_TUNING_INCOMPLETE
```
