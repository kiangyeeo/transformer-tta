# CODEX — IST Office LBI Full-Data Tuning, kappa=1 — 8 GPU — SUBMIT AND STOP

## Goal

今天完成 **IST × Office-31 × LBI** tuning，并最终冻结 6 个 Office tuples：

```text
FC   rho=.0005/.001/.002
Conv rho=.0005/.001/.002
```

严格遵守当前 IST Office LBI tuning 约定：

```text
kappa = 1 fixed
Stage-1 GT = (alpha, nu)
Office 六个 directed transfers
完整 target stream
seed = 2026

R0: full D->A Stage-1 reachability
R1: all-six full-stream Stage-1 validation
R2: all-six full-stream joint omega × stage2_lr
R3: one-time upper-bound expansion（仅触边界时）
```

统一 winner-selection 规则：

```text
Stage-1:
scientific-valid 6/6
-> utilization-eligible 6/6
-> worst-transfer p05 utilization DESC
-> six-transfer mean utilization DESC
-> worst Stage-1 max steps ASC
-> mean Stage-1 steps ASC
-> runtime ASC

Downstream (R2/R3):
scientific-valid 6/6 + utilization-eligible 6/6
-> six-transfer equal-weight FO DESC
-> worst-transfer FO DESC
-> Stage-1 max steps ASC
-> Stage-1 mean steps ASC
-> runtime ASC

PU = report-only
same-budget sparse margin = report-only, NOT a ranking criterion
```

规范来源：

```text
260817_iclr2027-refined/protocol/ist-otta/
  OTTA_IST_BASELINE_PROTOCOL_20260906_v1.md
  OTTA_IST_LBI_PROTOCOL_20260906_v1.md
```

以及用户提供的：

```text
LBI_HYPERPARAMETER_FEASIBLE_DOMAIN_FOR_IST_OFFICE_20260909.md
```

如果该文件未放入 repository，则以本 prompt 中完全展开的搜索域和纪律为准。

---

# 非常重要：本次 Codex 不是等待全部 tuning 跑完

你只负责：

```text
1. 只读核验当前 IST implementation / CLI / config 能否按协议启动
2. 创建可长期运行、可 resume 的持久 launcher
3. 构造完整 phased tuning queue
4. 把初始任务提交到 GPU 0–7
5. 确认八张卡全部有 active IST-LBI workload
6. 确认 launcher 在 Codex 退出后仍会持续自动推进 R0→R1→R2→R3
7. 打印 ETA / resume 信息
8. 立即停止 Codex
```

**不要等待 tuning 完成。**

不要坐在那里持续监控。

---

# 0. Current frozen state

当前应使用：

```text
IST baseline implementation:
ist_otta_p1_baseline_20260906_v4

IST sparse/LBI implementation:
ist_otta_sparse_lbi_20260906_v2

source revision:
nips2026_shot_otta_uda_source_v1
```

当前已经完成：

```text
IST P1 contracts PASS
IST sparse/LBI contracts PASS
Office D->A short smoke PASS
Office dense + sparse formal baselines COMPLETE
```

IST 与 SHOT 使用完全相同 source F/B/C checkpoint。

Do not modify scientific implementation.

---

# 1. Safety under danger-full-access

当前 Codex 可能运行于：

```toml
approval_policy = "never"
sandbox_mode = "danger-full-access"
```

所以安全约束是最高优先级。

## NEVER run destructive commands

禁止：

```text
rm
rm -rf
rmdir existing dirs
find ... -delete
git clean
git reset
git reset --hard
git checkout -- .
git checkout -- <existing-file>
git restore
git stash
git revert
git rebase
broad killall
broad pkill
rsync --delete
sed -i existing scientific files
truncate existing files
mv/cp 覆盖已有 scientific/artifact 文件
```

禁止：

```text
删除已有 runs
删除已有 experiment_logs
移动已有 runs
覆盖已有 summary/artifact
清理 failed/partial attempts
```

## Do not touch scientific code

禁止修改：

```text
shot_otta/**
ist_otta/**
come_otta/**
nctta_otta/**
core/lbi/**
train.py
experiment_identity.py
protocol_constants.py
protocol/**
configs/** existing files
tests/**
datasets/**
checkpoints/**
```

只允许创建新的：

```text
tuning launcher
queue/state files
launcher-local execution configs（若必须）
new tuning runs
new logs
new summary / selected tuple artifacts
```

如果发现 scientific implementation bug：

```text
DO NOT PATCH IT
record blocker
stop launching affected family
report it
```

不要现场修算法。

---

# 2. GPU requirement — OCCUPY ALL 8 GPUs

使用：

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

本机这 8 张 GPU 全部用于 IST tuning。

在提交前只需 read-only：

```bash
nvidia-smi
```

确认 GPU 0–7 可见。

在返回用户前必须验证：

```text
GPU0 has active IST-LBI tuning workload
GPU1 has active IST-LBI tuning workload
GPU2 has active IST-LBI tuning workload
GPU3 has active IST-LBI tuning workload
GPU4 has active IST-LBI tuning workload
GPU5 has active IST-LBI tuning workload
GPU6 has active IST-LBI tuning workload
GPU7 has active IST-LBI tuning workload
```

输出：

```text
ALL_8_GPUS_OCCUPIED = YES
```

## Initial concurrency

先：

```text
>= 1 active LBI job / GPU
```

即至少 8 个并行 scientific jobs。

如果单 job GPU utilization 明显偏低且显存允许，launcher 可动态提升到：

```text
max 2 jobs / GPU
```

Hard cap：

```text
16 concurrent jobs total
```

但是：

> **不得为了利用率修改任何 scientific setting。**

禁止改：

```text
BS
workers
objective
alpha/nu
omega
stage2_lr
Stage1 cap
optimizer
augmentation
PLCA
memory
```

---

# 3. Persistent background launcher

创建新的 task root：

```text
260817_iclr2027-refined/experiment_logs/
ist_office_lbi_fullsearch_kappa1_20260909/
```

推荐至少包含：

```text
launcher.py
resume.sh
launcher_state.json
tuning_queue.jsonl
launcher.log
gpu_scheduler.log
failed_or_retried_runs.jsonl
stage1_selection.json
stage2_selection.json
selected_configs/
configs_runtime/
logs/
```

Launcher 必须在当前 Codex session 退出后继续运行。

允许：

```text
nohup
tmux
screen
```

选择当前服务器最稳定、侵入最小的机制。

不要修改：

```text
~/.bashrc
shell startup files
systemd
cron
global config
```

Launcher 必须持续记录：

```text
phase
queue state
pending identities
running identities
completed identities
failed identities
retry count
PID
GPU assignment
attempt id
artifact path
stdout/stderr path
start/end timestamp
runtime
```

---

# 4. Resume semantics — REQUIRED

Resume 定义为：

```text
launcher / scientific-identity level resume
```

不是：

```text
target stream 中途的 model-state resume
optimizer-state resume
LBI-state partial resume
```

## Completed identity

只有以下全部满足才 skip：

```text
scientific identity exact
track exact
budget exact
alpha/kappa/nu exact
omega/stage2_lr exact
dataset/transfer exact
seed exact
implementation revision exact
source revision / SHA exact
formal/full-stream marker exact
artifact complete
summary complete
scientific-valid result available
```

## Running identity with live PID

```text
do not duplicate
```

## Interrupted / orphan partial identity

不要删除 partial artifact。

不要从中间模型继续。

应：

```text
requeue exact same identity
restart from same source checkpoint
same seed
same full target stream
same scientific settings
new distinct attempt path
```

## Failed identity

按照预注册 policy retry。

不允许：

```text
silent parameter change
change BS
change cap
change source
```

Resume 必须 idempotent：

```text
safe to call repeatedly
completed rows never duplicated
live rows never duplicated
orphan rows safely requeued
```

---

# 5. Exact resume command

必须创建并在 Codex 停止前打印，例如：

```bash
cd /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined
bash experiment_logs/ist_office_lbi_fullsearch_kappa1_20260909/resume.sh
```

或等价的：

```bash
python experiment_logs/ist_office_lbi_fullsearch_kappa1_20260909/launcher.py --resume
```

必须给用户**精确可复制命令**。

---

# 6. Frozen Office scientific settings

Office transfers：

```text
A->D
A->W
D->A
D->W
W->A
W->D
```

Domain mapping 使用当前 frozen Office resolver，不自行重定义。

所有最终 tuning evaluation：

```text
complete target stream
```

禁止：

```text
5-batch prefix
short prefix
partial stream deciding final tuple
```

Fixed：

```text
dataset = Office-31
backbone = ResNet-50
batch_size = 64
workers = 4
seed = 2026
one target pass
drop_last = false
same SHOT source F/B/C
singleton semantics = frozen IST protocol
PU read-only
FO full target read-only
netC frozen
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

对应：

```text
FC:
K = 262 / 524 / 1049

Conv out-channel:
K_G = 4 / 9 / 18
groups = 9216
```

禁止：

```text
filter_connection
```

---

# 7. IST-LBI scientific semantics — DO NOT CHANGE

必须严格沿用当前 approved implementation：

## Outer-batch unit

```text
one valid incoming outer batch
-> fixed 8-view IST task
-> PLCA once
-> memory commit once
-> LBI support discovery once
```

不是：

```text
one LBI support discovery / IST inner mini-batch
```

## Fixed outer-batch IST objective

Stage 1 / Stage 2 均使用完整 fixed outer-batch IST objective。

Gradient accumulation 只允许作为显存 implementation detail：

```text
no optimizer step between chunks
no LBI state update between chunks
one LBI update after complete gradient
```

## Stage 1

每 valid outer batch：

```text
theta_delta = 0
Z = 0
Gamma = 0
```

固定：

```text
support_threshold = 1e-4
stage1_max_steps = 3000
strict integer budget
overshoot -> rollback latest feasible
no budget slack
no top-K repair
correct old-state Z
```

## Stage 2

固定：

```text
masked-delta initialization
exactly one masked SGD optimizer step
momentum=.9
weight_decay=.001
nesterov=true
no Stage2 scheduler
fixed stage2_lr from tuning identity
off-mask exact preservation
```

## Persistent writeback

IST-LBI：

```text
native IST EMA = disabled
LBI omega writeback = exactly once / processed outer batch
```

禁止 double writeback。

---

# 8. Standard Stage-1 domain

当前 constrained tuning：

```text
kappa = 1
```

Stage-1 predictor / GT candidate vocabulary：

```text
alpha ∈ {.025, .05, .10, .125, .15, .20}
nu    ∈ {.25, .50, 1.00}
```

总计：

```text
18 standard (alpha,nu) candidates
```

不要搜索：

```text
kappa
```

不要静默加入：

```text
alpha=.30
alpha=.40
其他 domain 外数值
```

如果整个 standard domain 都不足：

```text
STANDARD_STAGE1_DOMAIN_INSUFFICIENT
```

然后停止对应 track×budget，不自行扩域。

---

# 9. R0 — FULL D->A Stage-1 reachability

R0 使用：

```text
完整 D->A target stream
```

不是 prefix。

固定 diagnostic downstream：

```text
omega = .00625
stage2_lr = .005
```

R0 的目标：

> 只判断 Stage-1 support-discovery dynamics，不用 accuracy 选 Stage-1 anchor。

## Initial anchors

按以下顺序/标签：

```text
A0 (.10,  .50)
A1 (.05,  .50)
A2 (.20,  .50)
A3 (.10,  .25)
A4 (.10, 1.00)
A5 (.125, .50)
A6 (.15,  .50)
A7 (.025, .50)
```

每个 anchor 应用于：

```text
2 tracks × 3 budgets
```

所以初始 R0 candidate identities：

```text
8 anchors × 6 track-budget units
= 48 full D->A jobs
```

这些 job 足够立即占满 8 GPU。

## Scientific-valid requirements

必须：

```text
loss/objective finite
strict budget respected
cap_hit_count = 0
off-mask exact
controlled BN exact
netC exact
objective call accounting exact
PLCA/memory accounting exact
native IST EMA commits = 0
omega writebacks exact
Stage2 step count exact
```

## Utilization eligibility

FC：

```text
mean support utilization >= .90
p05 support utilization  >= .90
```

Conv：

```text
mean group utilization >= .90
p05 group utilization  >= .75
```

其中：

```text
utilization = selected support / strict integer budget
```

Conv 同时记录 realized scalar support，但 eligibility 依据 group budget utilization。

## Rescue inside standard domain

如果某个 `track×budget` 初始 8 anchors 中 eligible 数少于 2：

按缺失标准-domain候选逐步补：

```text
(.05,  .25)
(.05, 1.00)
(.125, .25)
(.125,1.00)
(.15,  .25)
(.15, 1.00)
(.20,  .25)
(.20, 1.00)
(.025, .25)
(.025,1.00)
```

只补当前还没跑过的 identity。

不要超出 18-candidate standard domain。

---

# 10. R0 ranking — NO ACCURACY

对每个：

```text
track × budget
```

按以下顺序排名：

```text
1. scientific-valid
2. utilization-eligible
3. D->A p05 utilization descending
4. D->A mean utilization descending
5. Stage1 max steps ascending
6. Stage1 mean steps ascending
7. runtime ascending
```

FO / PU 在 R0 可记录，但：

```text
DO NOT USE ACCURACY TO SELECT STAGE-1
```

至少保留：

```text
primary
backup
```

eligible anchors 供 R1 验证。

---

# 11. R1 — ALL SIX FULL-STREAM Stage-1 validation

对每个：

```text
FC .0005
FC .001
FC .002
Conv .0005
Conv .001
Conv .002
```

按照 R0 ranking，从最好 Stage-1 candidate 开始，在 Office 六个 transfer 的**完整 target stream**上验证。

R0 的 exact D->A identity：

```text
只有所有 scientific fields 完全一致时允许复用
```

不重复浪费算力。

Candidate 必须：

```text
scientific-valid 6/6
cap-hit free 6/6
utilization-eligible 6/6
```

如果当前 candidate 不通过：

```text
继续下一个预定义标准-domain candidate
```

直到找到 all-six eligible anchor。

禁止：

```text
per-transfer alpha/nu
accuracy驱动换 alpha/nu
```

## R1 ranking

若有多个 all-six eligible：

```text
1. worst-transfer p05 utilization descending
2. six-transfer mean utilization descending
3. worst Stage1 max steps ascending
4. six-transfer mean Stage1 steps ascending
5. total/mean runtime ascending
```

**不要用 accuracy。**

最终冻结一个：

```text
(alpha, nu)
```

for each track×budget。

所以最终得到：

```text
6 frozen Stage-1 GTs
```

并固定：

```text
kappa=1
```

---

# 12. Stage-1 output artifacts

R1 完成后自动生成新的：

```text
experiment_logs/ist_office_lbi_fullsearch_kappa1_20260909/
stage1_selection.json

experiment_logs/ist_office_lbi_fullsearch_kappa1_20260909/
stage1_selection.md
```

必须记录每个 track×budget：

```text
alpha
nu
kappa=1
R0 ranking
R1 six-transfer validation
utilization mean/p05
Stage1 mean/max
cap hits
runtime
selection rationale
```

该文件是后续 hyperparameter fitting 的 GT source。

一旦 frozen：

```text
downstream accuracy 不得改变 alpha/nu
```

---

# 13. R2 — JOINT omega × Stage2 LR, ALL SIX FULL DATA

只有某 `track×budget` 的 Stage-1 `(alpha,nu)` 已冻结后，才能 queue 它的 R2。

禁止提前 queue 未冻结 Stage-1 的 R2 jobs。

Grid：

```text
omega ∈ {.00625, .025, .10}

stage2_lr ∈ {.0025, .005, .010}
```

完整：

```text
3 × 3 = 9 cells
```

每 cell：

```text
6 Office transfers
complete target stream
same frozen alpha/nu
kappa=1
```

所以每个 track×budget：

```text
54 transfer-level jobs
```

6 个 track×budget 全部 initial R2：

```text
6 × 9 × 6
= 324 transfer-level identities
```

但 exact R1 center：

```text
omega=.00625
stage2_lr=.005
```

若与 R1 完全同 identity：

```text
reuse it
```

不要重复运行。

---

# 14. R2 eligibility

每个 3×3 cell 必须在六个 Office transfer 上：

```text
6/6 scientific-valid
6/6 cap-hit free
6/6 utilization-eligible
```

Stage-1 support dynamics 本应与 frozen `(alpha,nu)` 一致。

若出现 divergence：

```text
record scientific failure
do not patch
```

---

# 15. R2 accuracy selection

Stage-1 已冻结后，这时才允许用 accuracy。

Primary：

```text
six-transfer equal-weight FO accuracy
```

不是 sample-weighted pooled accuracy。

Selection：

```text
1. six-transfer equal-weight FO descending
2. worst-transfer FO descending
3. Stage1 max steps ascending
4. Stage1 mean steps ascending
5. runtime ascending
```

说明：

```text
same-budget sparse baseline margin 不参与 winner ranking。
```

原因是对固定：

```text
dataset × track × budget
```

所有 LBI candidates 面对同一个 sparse baseline reference，因此按 FO 排序与按
`FO - FO_sparse` 排序完全等价。最终只报告 winner 是否 beat same-budget sparse baseline
以及领先/落后多少，不把 sparse margin 重复作为 tie-break criterion。

PU：

```text
report only
```

---

# 16. IST Office baseline references — USE THESE, NOT COME/SHOT VALUES

来自当前已完成的 IST Office formal baseline summary。

## FC same-budget best fixed sparse baseline FO

```text
rho=.0005:
best = ist_fc_saliency
FO = 77.9411

rho=.001:
best = ist_fc_saliency
FO = 78.0875

rho=.002:
best = ist_fc_saliency
FO = 78.2075
```

FC module dense：

```text
FO = 82.0906
```

## Conv same-budget best fixed sparse baseline FO

注意：不要假定每个 budget 都是 Saliency。

```text
rho=.0005:
best = ist_conv_out_random
FO = 77.6525

rho=.001:
best = ist_conv_out_random
FO = 77.6486

rho=.002:
best = ist_conv_out_saliency
FO = 77.7957
```

Conv module dense：

```text
FO = 82.4861
```

Shared source-only：

```text
FO = 77.6140
```

IST full dense：

```text
FO = 83.9028
```

R2 goal：

```text
beat same-budget best fixed sparse baseline
and maximize six-transfer equal-weight FO
```

但是：

> 是否打败 sparse baseline 不应作为 Stage-1 eligibility criterion；它只属于 R2/final accuracy interpretation。

---

# 17. Optional stricter sparse oracle reporting

如果现有 baseline summary 能方便地计算：

```text
per-transfer same-budget best sparse oracle
```

可以额外报告 LBI margin。

但：

```text
不要重新定义 R2 primary selection
```

Primary 仍是：

```text
six-transfer equal-weight LBI FO
```

同预算 sparse fixed baseline / oracle 都作为 margin/reporting。

---

# 18. R3 — one-time upper-bound expansion

仅当某 `track×budget` 的 initial R2 winner 触及：

```text
omega=.10
and/or
stage2_lr=.010
```

才允许一次局部 expansion。

Upper values：

```text
omega=.30
stage2_lr=.020
```

规则：

## only omega upper hit

测试：

```text
(.30, selected_stage2_lr)
```

## only LR upper hit

测试：

```text
(selected_omega, .020)
```

## both upper hit

测试：

```text
(.30, selected_lr)
(selected_omega, .020)
(.30, .020)
```

所有 expansion cell：

```text
all six transfers
full stream
same frozen alpha/nu
```

完成后重新用同一 R2 accuracy ranking 比较 initial winner + expansion cells。

**No second expansion.**

禁止：

```text
omega > .30
stage2_lr > .020
alpha domain expansion
nu domain expansion
```

---

# 19. Final tuple freeze

最终冻结：

```text
FC .0005
FC .001
FC .002

Conv .0005
Conv .001
Conv .002
```

每套：

```text
alpha
kappa=1
nu
omega
stage2_lr
```

写到新的：

```text
experiment_logs/ist_office_lbi_fullsearch_kappa1_20260909/
selected_configs/IST_OFFICE_LBI_FINAL_TUPLES_20260909_v1.json

experiment_logs/ist_office_lbi_fullsearch_kappa1_20260909/
selected_configs/IST_OFFICE_LBI_FINAL_TUPLES_20260909_v1.md
```

必须包括：

```text
Stage-1 selection evidence
R2 full 3x3 table
optional R3 rows
six-transfer PU/FO
worst-transfer FO
same-budget sparse margin
source margin
matched-module-dense gap
runtime
scientific validity
```

## Important

Final-formal fresh reruns 是**另一个任务**。

本 launcher 完成 tuple freeze 后：

```text
STOP
```

不要自动启动：

```text
IST final formal LBI
VisDA search
paper integration
```

---

# 20. Launcher phase dependency

Launcher 必须理解依赖关系，而不是一开始把所有 R2 都乱塞进去：

```text
R0
↓
R0 rank / rescue if needed
↓
R1 all-six validation
↓
freeze alpha/nu per track×budget
↓
R2 3×3 all-six
↓
optional R3 boundary expansion
↓
freeze 6 final tuples
↓
STOP
```

不同 `track×budget` 可以独立推进。

例如：

```text
FC .0005 已完成 R1
-> 可以进入 FC .0005 R2

同时 Conv .002 仍在 R0/R1
-> 继续自己的 Stage-1 phase
```

这样可以最大化 8-GPU 利用率。

不要为了等所有 6 个单位同时过 phase 而让 GPU 空闲。

---

# 21. GPU scheduling strategy

目标：

> **今天尽快完成，因此 8 张卡尽量一直有有效 scientific jobs。**

## Initial

立即提交 8 个不同的 R0 jobs：

```text
one per GPU
```

优先混合：

```text
FC / Conv
不同 budgets
不同 anchors
```

不要 8 卡全塞同一类最慢任务。

## Continuous refill

job 结束：

```text
immediately refill
```

Pending queue 非空时，不应有 GPU 长时间空闲。

## Optional 2 jobs/GPU

如果通过当前 job 的：

```text
GPU utilization
memory used
```

确认单 job 明显未吃满 GPU，且不会 OOM：

```text
allow 2 jobs/GPU
```

上限：

```text
2/GPU
16 global
```

不要为了 occupancy 改 scientific settings。

## R1/R2 long-tail avoidance

优先混排：

```text
target=A long transfers
target=D/W short transfers
FC
Conv
不同 budget
```

避免最后只剩一批 D->A / W->A 长任务拖尾。

---

# 22. Runtime / ETA evidence

IST Office non-LBI baseline 的当前 operational runtime 大致：

```text
non-Random:
~6–8 min / top-level Office run

Random 3-mask:
~18–23 min / top-level identity
```

注意：

> LBI Stage-1 是主要额外计算开销，不能直接用 baseline runtime 当 LBI runtime。

已有 SHOT-LBI / IST smoke / 当前正在完成的 R0 job runtime 可以作为更直接依据。

Codex 在提交后给 ETA 时，证据优先级：

```text
1. 当前 IST-LBI R0 实际 completed/near-complete job throughput
2. 当前 IST-LBI smoke / diagnostic 的真实 runtime
3. comparable SHOT/COME Office full-stream LBI runtime
4. baseline runtime + measured Stage1 multiplier
5. 如果还不够，只给 rough range
```

ETA 必须考虑：

```text
R0 48 initial jobs
possible standard-domain rescue
R1 candidate validation
R2 3x3 × all-six
possible R3 expansion
8–16 actual concurrency
不同 Office transfer 长度
```

不要拍脑袋给一个特别精确的分钟数。

---

# 23. ETA reporting requirement

Codex 停止前必须给：

```text
estimated remaining wall-clock
estimated finish timestamp
server/local timezone
basis
assumed effective concurrency
confidence = high / medium / low
```

如果刚提交时还没有足够 runtime evidence：

```text
给 conservative rough range
confidence=low
```

并同时给一个用户之后可以查看实时 ETA 的命令/文件，例如：

```text
tail -n 50 experiment_logs/ist_office_lbi_fullsearch_kappa1_20260909/launcher.log
```

或者 launcher 自己维护：

```text
eta_status.json
```

如果实现方便，推荐 launcher 每次 job 完成后重新估计 ETA 并写入：

```text
eta_status.json
```

但不要因此让 Codex 等待。

---

# 24. Failure handling

## Transient / infrastructure

例如：

```text
CUDA OOM
temporary I/O
worker crash
process crash
```

处理：

```text
preserve attempt
mark failed
refill GPU immediately
retry exact identity once later
```

OOM retry：

```text
same scientific settings
same batch size
same source
same seed
temporarily use one job on that GPU
```

第二次仍失败：

```text
record blocker
do not patch scientific code
```

## Scientific Stage-1 outcome

例如：

```text
cap hit
utilization ineligible
```

这不一定是程序 crash。

记录 candidate outcome，然后继续下一个预定义 candidate。

## Correctness failure

例如：

```text
NaN/Inf
source SHA mismatch
identity mismatch
off-mask violation
BN/netC violation
objective-call mismatch
EMA/omega double-writeback
```

处理：

```text
quarantine
stop launching affected family if systematic
do not patch code
```

若相同 correctness failure 在 >=2 个独立 identities 重现：

```text
stop new jobs for affected family
keep unrelated healthy jobs running
record SYSTEMATIC_CORRECTNESS_BLOCKER
```

---

# 25. No result-driven protocol changes

禁止：

```text
因为 accuracy 低扩 alpha
因为 LBI 没打赢 sparse 就换 Stage1 GT
per-transfer tuning
提高 Stage1 cap
改 support threshold
改 budget
改 grouping
改 Stage2 steps
改 optimizer
改 source
改 seed
只重跑低 accuracy
```

R0/R1：

```text
accuracy cannot choose alpha/nu
```

R2/R3：

```text
accuracy selects only omega/stage2_lr
under frozen alpha/nu
```

---

# 26. Launcher artifacts / summaries

最终 task root 至少维护：

```text
launcher.py
resume.sh
launcher_state.json
tuning_queue.jsonl
launcher.log
gpu_scheduler.log
eta_status.json

R0_RESULTS.csv
R0_RESULTS.md

R1_STAGE1_VALIDATION.csv
R1_STAGE1_VALIDATION.md

stage1_selection.json
stage1_selection.md

R2_DOWNSTREAM_GRID.csv
R2_DOWNSTREAM_GRID.md

R3_BOUNDARY_EXPANSION.csv
R3_BOUNDARY_EXPANSION.md

failed_or_retried_runs.jsonl
quarantined_runs.jsonl

selected_configs/
  IST_OFFICE_LBI_FINAL_TUPLES_20260909_v1.json
  IST_OFFICE_LBI_FINAL_TUPLES_20260909_v1.md
```

未触发 R3 时：

```text
R3 文件可记录 NOT_TRIGGERED
```

---

# 27. SUBMIT-AND-STOP behavior — CRITICAL

当前 Codex session 的停止条件：

只有以下都满足：

```text
persistent launcher created
queue/state created
resume entrypoint created
launcher process/session alive
GPU0 active
GPU1 active
GPU2 active
GPU3 active
GPU4 active
GPU5 active
GPU6 active
GPU7 active
state file is updating
no immediate launcher crash
```

然后：

> **立即返回用户并停止 Codex。**

不要：

```text
wait for first R0 batch to finish
wait for R0 phase
wait for ETA to become high-confidence
poll indefinitely
monitor until tuning complete
```

只需确认 submit 成功。

---

# 28. Quick occupancy verification before stop

只做一次/少量 read-only check：

```bash
nvidia-smi
```

以及：

```text
launcher PID alive
running queue non-empty
state timestamps moving
```

报告每张卡 active job identity / PID。

如果初始某 GPU 没任务：

```text
fix launcher scheduling
submit pending R0 job
recheck
```

必须达到：

```text
ALL_8_GPUS_OCCUPIED = YES
```

才能停止。

---

# 29. Required immediate final response after submission

提交后只给高密度状态。

## A. Launcher

```text
launcher mechanism: nohup/tmux/screen/...
launcher PID/session:
launcher path:
state file:
queue file:
launcher log:
GPU log:
ETA status file:
```

## B. GPU occupancy

```text
GPU0: active job(s) / PID
GPU1: active job(s) / PID
GPU2: active job(s) / PID
GPU3: active job(s) / PID
GPU4: active job(s) / PID
GPU5: active job(s) / PID
GPU6: active job(s) / PID
GPU7: active job(s) / PID
```

必须确认：

```text
ALL_8_GPUS_OCCUPIED = YES
```

## C. Queue

```text
current active phase(s)
pending
running
completed
failed
quarantined
effective concurrency
```

## D. Resume

打印精确 resume 命令。

并说明：

```text
completed exact identities -> skip
alive jobs -> not duplicate
interrupted partial identity -> restart exact identity from source
partial artifacts -> preserved
no partial model-state resume
```

## E. ETA

```text
estimated remaining wall-clock:
estimated finish timestamp:
timezone:
basis:
effective concurrency:
confidence:
```

## F. Safety

明确：

```text
scientific implementation modified = false
existing scientific files overwritten = 0
existing artifacts deleted = 0
existing artifacts overwritten = 0
```

## G. Final submission status

最后打印精确：

```text
IST_OFFICE_LBI_TUNING_SUBMITTED
```

然后 STOP。

不要等待：

```text
IST_OFFICE_LBI_TUNING_COMPLETE
```

---

# 30. Final scientific completion condition（由后台 launcher 后续自动完成）

后台 launcher 最终只有在以下全部完成后才认为 tuning closed：

```text
6/6 track×budget Stage-1 GT frozen
all required R2 cells complete or exact reuse validated
all triggered R3 cells complete
6 final tuples frozen
no unresolved correctness blocker
```

最终状态写入：

```text
launcher_state.json
```

以及：

```text
TUNING_COMPLETE
```

但本 Codex session **不等待这个状态**。

---

# 31. Important reminder

本次只做：

```text
IST × Office LBI tuning
```

不要自动开始：

```text
IST VisDA LBI tuning
IST final formal rerun
COME anything
SHOT anything
NCTTA anything
paper integration
```

最终 6 个 tuple freeze 后停。
