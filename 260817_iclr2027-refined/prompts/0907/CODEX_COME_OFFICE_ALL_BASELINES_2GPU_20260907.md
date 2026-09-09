# CODEX — COME Office All Baselines Formal — 2 GPU

目标：在 **2 张 GPU** 上完成 COME 的 **Office-31 全部 formal baselines**，包括 dense baselines 与 non-LBI sparse baselines。

本任务 **禁止运行 COME-LBI**，禁止做任何 tuning / search，禁止修改已冻结 scientific semantics。

---

## 0. 必读规范

先完整读取：

```text
260817_iclr2027-refined/protocol/come-otta/OTTA_COME_BASELINE_PROTOCOL_20260907_v1.md
260817_iclr2027-refined/protocol/come-otta/OTTA_COME_LBI_PROTOCOL_20260907_v1.md
```

当前冻结 revisions：

```text
COME baseline implementation:
come_otta_baseline_20260907_v1

COME sparse/LBI implementation:
come_otta_sparse_lbi_20260907_v1
```

Official COME audited commit：

```text
409a19b71f62c765b1a5be62347a9455524ec176
```

Source checkpoint revision：

```text
nips2026_shot_otta_uda_source_v1
```

---

## 1. 最高优先级：不要改代码

这是 **formal execution task**，不是 implementation task。

原则：

```text
do not modify scientific code
do not modify protocol
do not modify configs' scientific semantics
do not modify experiment identity semantics
do not modify optimizer/objective/scope
```

尤其禁止修改：

```text
shot_otta/**
ist_otta/**
nctta_otta/**
core/lbi/**
come_otta/**
```

如果发现 formal execution 被现有 correctness bug 阻塞：

```text
STOP
报告 blocker
不要现场改算法/语义
```

开始前记录：

```bash
git status --short
git diff --name-only
```

运行结束再次记录：

```bash
git status --short
git diff --name-only
git diff --check
```

---

## 2. 数据集范围：只跑 Office-31

正式 transfers：

```text
A -> D
A -> W
D -> A
D -> W
W -> A
W -> D
```

固定：

```text
seed = 2026
backbone = ResNet-50
outer batch size = 64
workers = 4
one target pass
drop_last = false
actual BS=1 singleton early skip
primary metric = sample-level overall accuracy
```

所有方法必须使用与 SHOT 完全一致的 source F/B/C checkpoints：

```text
source checkpoint revision =
nips2026_shot_otta_uda_source_v1
```

禁止重新训练 source。

---

## 3. 本次要跑的 COME baselines

### 3.1 Dense baselines

每个 transfer：

```text
come_full_dense
come_fc_module_dense
come_conv_module_dense
```

共：

```text
6 transfers x 3 variants
= 18 formal runs
```

Dense baselines不按 sparse budget重复。

---

### 3.2 FC sparse non-LBI baselines

Variants：

```text
come_fc_random
come_fc_magnitude
come_fc_saliency
```

Budgets：

```text
rho = 0.0005
rho = 0.001
rho = 0.002
```

FC candidate：

```text
netB.bottleneck.weight
netB.bottleneck.bias
candidate scalars = 524544
```

Integer budgets：

```text
0.0005 -> K=262
0.001  -> K=524
0.002  -> K=1049
```

---

### 3.3 Conv sparse non-LBI baselines

Variants：

```text
come_conv_out_random
come_conv_out_magnitude
come_conv_out_saliency
```

Budgets：

```text
rho = 0.0005
rho = 0.001
rho = 0.002
```

Conv candidate：

```text
netF.layer4 nine Conv weights
grouping = out_channel
groups = 9216
candidate scalars = 12845056
```

Integer group budgets：

```text
0.0005 -> K_G=4
0.001  -> K_G=9
0.002  -> K_G=18
```

`filter_connection` 禁止进入 formal。

---

## 4. Random 必须真实跑 3 个 child trajectories

对：

```text
come_fc_random
come_conv_out_random
```

每个：

```text
transfer x budget
```

都必须真实执行：

```text
mask_seed = 202600
mask_seed = 202601
mask_seed = 202602
```

每个 child：

```text
fresh source F/B/C
fresh optimizer state
same target stream seed=2026
same sample order/batch partition
only random support differs
```

禁止：

```text
num_random_masks=3 只写 metadata
但实际只跑 1 child
```

Top-level Random artifact必须聚合：

```text
child PU/FO
mean/std
child mask seed
child artifact path
```

---

## 5. 总任务规模

Top-level formal identities：

```text
Dense:
18

Sparse:
6 transfers
x 2 tracks
x 3 selectors
x 3 budgets
= 108

Total top-level identities:
126
```

实际 executions 因 Random 3-child：

```text
Dense = 18

Magnitude + Saliency:
6 transfers x 2 tracks x 2 selectors x 3 budgets
= 72

Random child executions:
6 transfers x 2 tracks x 3 budgets x 3 children
= 108

Total actual executions:
198
```

---

## 6. 明确禁止 LBI

本任务不得运行：

```text
come_fc_lbi
come_conv_out_lbi
```

不得运行：

```text
LBI search
alpha/kappa/nu search
omega search
stage2_lr search
reachability search
utilization tuning
```

任何已有 LBI smoke artifact都不得混进此次 baseline formal summary。

---

## 7. Formal execution constraints

所有正式 runs 必须：

```text
formal_protocol = true
debug_max_outer_batches = None
save_model = false
```

禁止：

```text
debug run冒充 formal
partial stream run冒充 formal
resume from partial model state
```

若某个 run 中断：

```text
从 source 重新跑该 identity
```

不做 partial resume。

所有 artifacts 必须记录正确：

```text
protocol revision
implementation revision
official COME commit
source checkpoint revision
source F/B/C paths + SHA256
dataset / transfer
seed
target stream identity/hash
variant
track
budget
integer K/K_G
selector
BN semantics
optimizer settings
primary metric
formal=true
debug_only=false
```

---

## 8. 2-GPU 调度

当前只有两张 GPU。

假设可见设备：

```text
GPU 0
GPU 1
```

### 最大并发

默认允许：

```text
GPU0: max 2 jobs
GPU1: max 2 jobs
```

即全局最多：

```text
4 concurrent jobs
```

但必须先做显存安全检查。

### 8.1 先进行最小调度 probe

先用已经冻结的 formal identity做少量正式 job启动检查，例如：

```text
一个 full_dense
一个 FC sparse
一个 Conv sparse
```

只观察：

```text
GPU memory
OOM
system stability
```

不要修改任何 hyperparameter。

如果双并发稳定：

```text
每卡最多 2 jobs
```

如果某类任务双并发 OOM：

```text
该类任务自动降为每卡 1 job
```

特别：

```text
full_dense 可优先单卡单任务
FC/Conv sparse 在显存允许时每卡双任务
```

禁止通过：

```text
改 BS
改模型
改算法
```

解决 OOM。

---

## 9. 调度策略

不要简单按 transfer / variant 串行。

为了减少长尾：

```text
Dense
Magnitude
Saliency
Random
```

混合排队。

特别不要把所有 Random 留到最后，因为 Random 每个 top-level identity需要 3 条 child trajectories。

建议任务队列按估计成本交错：

```text
target=A 的较长任务
target=D/W 的较短任务
Random child
non-Random
```

混合调度，尽量保持两张卡持续利用。

---

## 10. 断点与重复执行保护

在启动任何 identity 前：

1. 计算其 canonical scientific identity；
2. 检查目标 artifact目录；
3. 只在存在 **完整且通过 validation 的 formal summary** 时跳过。

不能仅因为目录存在就跳过。

“已完成”至少要求：

```text
formal=true
debug_only=false
summary.json exists
metrics/artifact complete
scientific identity exact match
source revision exact
implementation revision exact
processed stream complete
PU/FO present
no runtime failure
```

如果 artifact：

```text
partial
corrupted
identity mismatch
debug
old revision
```

则不得复用，重新跑该 formal identity。

### Random

每个 child独立检查。

只有 3 个 child都完整后，top-level Random才算 complete。

---

## 11. Correctness checks during execution

每个 run 至少检查：

```text
finite loss
finite gradient/update
netC unchanged
scope correct
singleton policy correct
PU read-only
FO read-only
processed target stream complete
```

FC/Conv sparse：

```text
BN parameters/buffers frozen
support exact requested K/K_G
off-mask parameters exact unchanged
```

Random：

```text
3 real children
```

Magnitude：

```text
source-fixed support
```

Saliency：

```text
support selected once per valid outer batch
```

如出现 correctness failure：

```text
STOP affected identity
记录 failure
不要调参
不要静默 fallback
```

---

## 12. Accuracy policy

这是 formal baseline execution。

禁止：

```text
根据中间 accuracy 改参数
根据某 transfer 表现改 LR
根据 sparse selector结果改 budget
选择性重跑低 accuracy run
```

只允许因为：

```text
runtime failure
corrupted artifact
identity mismatch
```

重新执行同一 identity。

所有 scientific settings必须保持 frozen。

---

## 13. Source-only

不要为 COME 单独重新跑 source-only。

COME 与 SHOT 使用相同：

```text
source F/B/C
source revision
evaluation substrate
```

若现有 source-only artifact scientific identity与当前 shared reference兼容，则直接在最终 summary中引用已有 shared source-only结果。

不要创建“COME-specific source-only adaptation run”。

---

## 14. 输出目录

正式 artifacts优先使用当前 repo既有 formal naming convention。

要求明确与：

```text
runs_smoke/
runs_diagnostic/
```

区分。

例如保持当前 formal runs体系下的 canonical path。

不要把正式结果写到：

```text
runs_smoke
runs_diagnostic
```

每个 artifact必须可从：

```text
dataset
transfer
method=COME
variant
budget
seed
random child
```

唯一定位。

---

## 15. 运行过程中不要频繁改代码

在开始正式矩阵后：

```text
code/protocol/config scientific semantics视为冻结
```

运行中若发现非阻塞的日志/显示问题：

```text
先记录
不要边跑边改
```

若发现 correctness blocker：

```text
暂停新任务
报告
```

不要自行改变 scientific semantics后继续混跑。

---

## 16. 完成后统一验证

126 个 top-level identities完成后，验证：

### Dense

```text
6 transfers x 3 = 18 complete
```

### FC sparse

```text
6 transfers
x 3 budgets
x 3 selectors
= 54 top-level complete
```

其中 Random：

```text
6 x 3 budgets = 18 top-level
18 x 3 = 54 actual children
```

### Conv sparse

同样：

```text
54 top-level complete
```

Random：

```text
54 actual children
```

最终：

```text
126 / 126 top-level complete
198 / 198 expected executions complete
```

注意 Random child属于 top-level Random内部执行。

---

## 17. 最终结果汇总

生成一个 COME Office baseline summary，例如：

```text
260817_iclr2027-refined/experiment_logs/
come_office_all_baselines_20260907/
```

至少输出：

```text
summary.md
summary.csv
summary.json
run_manifest.jsonl
```

不要修改既有 artifact。

---

## 18. `summary.md` 主表

每个 transfer分别列：

```text
source_only
come_full_dense
come_fc_module_dense
come_conv_module_dense

FC:
  random .0005
  random .001
  random .002
  magnitude .0005
  magnitude .001
  magnitude .002
  saliency .0005
  saliency .001
  saliency .002

Conv:
  random .0005
  random .001
  random .002
  magnitude .0005
  magnitude .001
  magnitude .002
  saliency .0005
  saliency .001
  saliency .002
```

至少报告：

```text
PU
FO
```

Random：

```text
PU mean/std
FO mean/std
3 child values
```

---

## 19. Office aggregate

对六个 directed transfers做等权平均。

分别给：

```text
mean PU
mean FO
```

对每个：

```text
dense variant
track x selector x budget
```

都给 aggregate。

Random aggregation必须先：

```text
within transfer: aggregate 3 child trajectories
```

再：

```text
across six transfers: equal-weight transfer mean
```

不要把 18 个 Random children直接当成 18 个 transfer样本。

---

## 20. Validation summary

最终额外报告：

```text
top-level expected / complete
actual executions expected / complete
failed runs
rerun runs
skipped-existing runs
identity mismatches
corrupted artifacts
singleton counts
source SHA mismatches
scope violations
BN violations
netC violations
```

正常情况下这些 correctness violations都应为 0。

---

## 21. Stop condition

完成：

```text
all COME Office dense baselines
all COME Office FC Random/Magnitude/Saliency baselines
all COME Office Conv Random/Magnitude/Saliency baselines
artifact validation
Office aggregate summary
```

后立即停止。

禁止继续：

```text
COME LBI
COME LBI search
IST runs
VisDA COME runs
paper table integration
new method development
```

---

## 22. 最终 Codex 汇报格式

### A. Git / code state

```text
code modified? yes/no
protocol modified? yes/no
```

预期：

```text
no
```

### B. Hardware / scheduler

```text
GPUs used
max jobs per GPU
whether any class required 1-job/GPU
OOM count
```

### C. Completion

```text
top-level identities:
126 / 126

actual executions:
198 / 198
```

### D. Per-family completeness

```text
Dense
FC Random
FC Magnitude
FC Saliency
Conv Random
Conv Magnitude
Conv Saliency
```

### E. Accuracy summary

给完整 Office 6-transfer表以及 equal-transfer aggregate。

### F. Random child audit

确认每个 Random identity：

```text
3 / 3 children
seeds 202600/202601/202602
```

### G. Correctness audit

```text
source consistency
stream consistency
scope
BN
netC
PU/FO
formal/debug markers
```

### H. Failures / reruns

逐项列明。

### I. Output artifacts

给出：

```text
summary.md
summary.csv
summary.json
run_manifest.jsonl
```

路径。

### J. Final verdict

只能：

```text
COME_OFFICE_BASELINES_COMPLETE
```

或：

```text
COME_OFFICE_BASELINES_INCOMPLETE
```

如果 complete，立即停止，不运行 LBI。
