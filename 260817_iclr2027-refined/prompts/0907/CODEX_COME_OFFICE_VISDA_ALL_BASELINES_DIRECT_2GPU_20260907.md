# CODEX — COME Office + VisDA All Baselines Formal — Direct 2-GPU Run

## Goal

基于**当前代码直接开跑** COME 的全部正式 baseline：

- Office-31：全部 6 个 directed transfers
- VisDA-C：Train/Synthetic -> Validation/Real
- Dense baselines
- FC sparse non-LBI baselines
- Conv sparse non-LBI baselines

**不要 smoke。不要 dry-run。不要 search。不要运行 LBI。不要先跑测试浪费 GPU 时间。**

当前代码已经具备 formal non-LBI sparse support：

```text
COME baseline implementation:
come_otta_baseline_20260907_v1

COME sparse/LBI implementation:
come_otta_sparse_lbi_20260907_v2
```

当前实现语义应为：

```text
Random / Magnitude / Saliency:
formal_protocol=true ALLOWED

come_fc_lbi / come_conv_out_lbi:
formal_protocol=true BLOCKED
```

直接进入正式矩阵。

---

# 1. 硬件约束

**只能使用物理 GPU 0 和 GPU 1。**

禁止使用：

```text
GPU 2+
```

当前机器两张可用 GPU 为约 48GB RTX 4090。

从任务启动开始，尽量保证两张 GPU 持续有计算负载。

用户的资源策略：

> 如果 GPU 在 3 小时窗口内长期低于约 30% utilization，资源可能被 kill。

因此调度优先级：

```text
GPU utilization > 省一点显存/少开一个进程
```

但禁止通过改变 scientific settings 提高 utilization。

---

# 2. 调度策略：从一开始就直接 4-way concurrency

**不要先做 probe。**

默认从正式任务开始就：

```text
GPU 0: 最多同时 2 个 formal jobs
GPU 1: 最多同时 2 个 formal jobs
```

全局：

```text
max concurrent formal jobs = 4
```

启动后始终保持：

```text
只要 pending queue 非空，
每张 GPU 尽量保持 2 个 active jobs。
```

某个 job 一结束：

```text
立即补下一个 job
```

不要等待人工确认，不要在 jobs 之间停顿。

### OOM 例外

若某个具体任务发生 CUDA OOM：

1. 不改变 batch size / optimizer / algorithm；
2. 记录该 identity；
3. 该 identity 允许在同一 GPU 上以“该 GPU 暂时只跑 1 job”方式重试一次；
4. 重试结束后恢复每卡最多 2 jobs。

禁止通过减小 batch size 解决 OOM。

---

# 3. GPU utilization 保活要求

启动一个轻量监控：

```bash
nvidia-smi -i 0,1
```

可每 5 分钟记录一次：

```text
timestamp
GPU index
utilization.gpu
memory.used
active job count
```

输出到本次 launcher log。

如果 pending queue 非空而某 GPU：

```text
active jobs < 2
```

必须立即补任务。

如果某 GPU 上单 job 长时间 utilization <30%，且还有 pending jobs：

```text
优先补第 2 个 job
```

不要为了 utilization 修改：

```text
batch size
workers
learning rate
model
selector
budget
```

### 防长尾

不要把所有长任务或 Random 留到最后。

队列应采用“长短混排 / longest-first + refill”：

优先把下面任务分散到 GPU 0/1：

```text
VisDA jobs
Office target=A jobs: D->A, W->A
Random identities
```

并与：

```text
Office target=D/W 的短任务
Magnitude / Saliency
```

交错。

目标是从开始到结束两张 GPU 都尽量持续有负载。

---

# 4. 禁止修改 scientific code

本任务是 execution task。

不要修改：

```text
come_otta/**
shot_otta/**
ist_otta/**
nctta_otta/**
core/lbi/**
train.py
experiment_identity.py
protocol_constants.py
protocol/**
```

也不要修改 scientific configs semantics。

允许创建：

```text
launcher / queue / log / summary scripts
```

但最好放在：

```text
experiment_logs/come_office_visda_all_baselines_20260907/
```

或 `/tmp`，不要改 scientific implementation。

如果个别 identity runtime 失败：

```text
记录并按原设置重试
```

不要现场改算法。

---

# 5. Source checkpoint

COME 必须继续使用和 SHOT 完全一致的 source F/B/C：

```text
source_checkpoint_revision =
nips2026_shot_otta_uda_source_v1
```

实际 checkpoint root 使用当前 frozen COME config 中的 SHOT source root。

每个 run artifact 已有 source path / SHA256 校验。

禁止：

```text
重新训练 source
换 source checkpoint
COME-specific source
```

---

# 6. Formal dense config

使用当前：

```text
260817_iclr2027-refined/configs/come_baseline_protocol_20260907_v1.yaml
```

正式运行必须：

```text
formal_protocol = true
debug_max_outer_batches = null
save_model = false
seed = 2026
```

Dense variants：

```text
come_full_dense
come_fc_module_dense
come_conv_module_dense
```

---

# 7. Formal sparse config

使用当前：

```text
260817_iclr2027-refined/configs/come_otta_sparse_formal_20260907_v2.yaml
```

当前 revision：

```text
come_otta_sparse_lbi_20260907_v2
```

正式运行必须：

```text
formal_protocol = true
debug_max_outer_batches = null
save_model = false
lbi = null
seed = 2026
```

只允许：

```text
come_fc_random
come_fc_magnitude
come_fc_saliency

come_conv_out_random
come_conv_out_magnitude
come_conv_out_saliency
```

明确禁止：

```text
come_fc_lbi
come_conv_out_lbi
```

---

# 8. Office-31 matrix

Office domain index：

```text
0 = amazon (A)
1 = dslr   (D)
2 = webcam (W)
```

六个 transfers：

```text
A->D = 0->1
A->W = 0->2
D->A = 1->0
D->W = 1->2
W->A = 2->0
W->D = 2->1
```

固定：

```text
dataset = office
backbone = ResNet-50
batch_size = 64
workers = 4
seed = 2026
one target pass
drop_last = false
primary = sample-level overall accuracy
```

---

# 9. VisDA-C matrix

只跑：

```text
dataset = VISDA-C
source = 0 = train
target = 1 = validation
```

即：

```text
Train/Synthetic -> Validation/Real
```

固定 formal resolver 应自动保证：

```text
backbone = ResNet-101
batch_size = 256
workers = 4
base lr = 0.001
K = 12
seed = 2026
```

Primary：

```text
fixed-12-class mAcc
```

Secondary：

```text
overall accuracy
12-class classwise accuracy
worst class
classwise std
```

---

# 10. Dense baseline matrix

对：

```text
6 Office transfers
+ 1 VisDA transfer
= 7 dataset-transfer units
```

分别运行：

```text
come_full_dense
come_fc_module_dense
come_conv_module_dense
```

总计：

```text
7 x 3 = 21 dense formal runs
```

Dense 不按 rho 重复。

---

# 11. FC sparse non-LBI matrix

Variants：

```text
come_fc_random
come_fc_magnitude
come_fc_saliency
```

Candidate：

```text
netB.bottleneck.weight
netB.bottleneck.bias
candidate scalars = 524544
```

Budgets：

```text
rho=0.0005 -> K=262
rho=0.001  -> K=524
rho=0.002  -> K=1049
```

对全部：

```text
7 dataset-transfer units
x 3 selectors
x 3 budgets
= 63 FC sparse top-level identities
```

---

# 12. Conv sparse non-LBI matrix

Variants：

```text
come_conv_out_random
come_conv_out_magnitude
come_conv_out_saliency
```

Candidate：

```text
netF.layer4 nine Conv weights
grouping = out_channel
groups = 9216
candidate scalars = 12845056
```

Budgets：

```text
rho=0.0005 -> K_G=4
rho=0.001  -> K_G=9
rho=0.002  -> K_G=18
```

对全部：

```text
7 dataset-transfer units
x 3 selectors
x 3 budgets
= 63 Conv sparse top-level identities
```

禁止：

```text
filter_connection
```

---

# 13. Random 必须真实 3 children

对：

```text
come_fc_random
come_conv_out_random
```

每个 dataset-transfer-budget identity 必须真实运行：

```text
mask seed 202600
mask seed 202601
mask seed 202602
```

每个 child：

```text
fresh source F/B/C
fresh optimizer
same target stream seed=2026
same target sample order
```

Top-level Random artifact聚合 3 children。

不能把 Random 简化成单 run。

---

# 14. 本次总工作量

## Top-level identities

Dense：

```text
21
```

Sparse：

```text
7 units x 2 tracks x 3 selectors x 3 budgets
= 126
```

Total：

```text
147 top-level identities
```

## Actual executions

Dense：

```text
21
```

Magnitude + Saliency：

```text
7 x 2 tracks x 2 selectors x 3 budgets
= 84
```

Random children：

```text
7 x 2 tracks x 3 budgets x 3 children
= 126
```

Total actual executions：

```text
231
```

完成目标：

```text
147 / 147 top-level identities
231 / 231 actual executions
```

---

# 15. 不重新跑 source-only

COME 与 SHOT 共用完全相同 source checkpoints。

不要创建新的 COME source-only adaptation run。

最终 summary 中可以引用已有 shared SHOT source-only reference，前提是 identity / source checkpoint一致。

Source-only 不计入 147 identities。

---

# 16. 直接正式运行：无 smoke / 无 dry-run

明确禁止在正式矩阵前执行：

```text
5-batch smoke
1-batch smoke
GPU probe
dry-run matrix
hyperparameter probe
accuracy probe
```

当前代码已经审计完成。

**直接启动 formal jobs。**

也不要先等 tests 全跑完。

---

# 17. Job launcher

建立一个持久队列/launcher，至少支持：

```text
pending
running
complete
failed
retry
```

每个 identity 唯一化。

Launcher 必须：

1. 只分配 GPU 0 或 GPU 1；
2. 每 GPU 最多 2 个 active jobs；
3. 默认立即启动 4 个 jobs；
4. job完成后立即 refill；
5. 记录 PID / GPU / identity / log path；
6. 捕获 exit code；
7. 不因单个失败让另一张 GPU 空闲；
8. unaffected jobs继续跑；
9. 同一 identity 不重复并发。

建议每个 job独立 stdout/stderr log。

---

# 18. 已完成 artifact 的处理

如果当前 formal output 中已经存在某 identity：

只有在以下全部满足时才允许 skip：

```text
formal=true
formal_protocol=true
debug_only=false
summary complete
scientific identity exact
implementation revision exact
source checkpoint SHA exact
full stream complete
PU/FO present
exit success
```

否则重新跑该 identity。

不要因为“目录存在”就 skip。

Random 必须 3/3 children完整才算 top-level complete。

---

# 19. Failure policy

为了不让 GPU 因单个失败闲置：

### Infrastructure / transient failure

例如：

```text
CUDA OOM
temporary I/O
process crash
```

处理：

```text
mark failed
immediately refill GPU with another pending job
retry failed identity once later
```

OOM retry：

```text
same scientific settings
same batch size
temporarily one job on that GPU
```

### Scientific/correctness failure

例如：

```text
source SHA mismatch
identity mismatch
NaN/Inf objective
scope violation
formal marker wrong
```

不得修改 setting。

把 identity quarantine，并继续运行其他 independent identities以维持 GPU 使用。

如果同一种 correctness failure连续出现在 >=2 个不同 identities，说明可能是系统性问题：

```text
停止启动新的相关 family jobs
但不要杀掉已经健康运行的其他 family jobs
报告 blocker
```

---

# 20. Formal correctness invariants

所有 runs继续依赖现有代码自动审计：

```text
same SHOT source F/B/C
netC frozen
scope exact
BN semantics exact
singleton early skip
PU read-only
FO read-only
labels不进入 adaptation
```

Sparse：

```text
exact K/K_G
off-mask exact
Random true 3 children
Magnitude source-static
Saliency current-state once/batch
```

不要修改这些语义。

---

# 21. Output roots

统一写到明确的 formal root，避免和 smoke 混淆，例如：

```text
260817_iclr2027-refined/runs/come_formal_baselines_20260907/
```

可分：

```text
dense/
sparse/
```

不得写入：

```text
runs_smoke/
runs_diagnostic/
```

Launcher / monitor logs：

```text
260817_iclr2027-refined/experiment_logs/
come_office_visda_all_baselines_20260907/
```

---

# 22. 运行期间不要做结果驱动决策

禁止：

```text
看某个 transfer accuracy 后修改 LR
换 objective
换 source
换 selector
改 budget
选择性重跑低 accuracy
```

只按 frozen matrix 跑。

即使某个 baseline accuracy低，也照常记录并继续。

---

# 23. 完成后的 artifact validation

**不是运行前 smoke。**

全部 GPU jobs结束后，统一做 artifact completeness audit。

检查：

```text
21/21 dense
63/63 FC sparse top-level
63/63 Conv sparse top-level
147/147 total top-level
231/231 actual executions
```

Random：

```text
每个 identity = 3/3 children
```

检查所有 formal markers。

---

# 24. Office summary

生成 Office 六迁移完整表。

至少列：

```text
source_only reference

come_full_dense
come_fc_module_dense
come_conv_module_dense

FC Random/Magnitude/Saliency:
rho .0005/.001/.002

Conv Random/Magnitude/Saliency:
rho .0005/.001/.002
```

每项：

```text
PU
FO
```

Random：

```text
3 child values
mean
std
```

再给六迁移 equal-transfer aggregate：

```text
mean PU
mean FO
```

Random 聚合顺序：

```text
先 3 child -> within-transfer mean
再 6 transfers equal-weight mean
```

不要把 18 child当 18 transfers。

---

# 25. VisDA summary

同样列：

```text
source_only reference

3 dense
FC Random/Magnitude/Saliency x3 rho
Conv Random/Magnitude/Saliency x3 rho
```

Primary：

```text
FO fixed-12-class mAcc
PU fixed-12-class mAcc
```

同时报告：

```text
overall accuracy
12 classwise accuracies
worst-class accuracy
classwise std
```

Random：

```text
3 child values + mean/std
```

---

# 26. 最终汇总文件

生成：

```text
260817_iclr2027-refined/experiment_logs/
come_office_visda_all_baselines_20260907/
```

至少：

```text
COME_OFFICE_BASELINES_SUMMARY.md
COME_OFFICE_BASELINES_SUMMARY.csv

COME_VISDA_BASELINES_SUMMARY.md
COME_VISDA_BASELINES_SUMMARY.csv

COME_ALL_BASELINES_MANIFEST.jsonl
COME_ALL_BASELINES_COMPLETENESS.json

gpu_scheduler.log
failed_or_retried_runs.jsonl
```

---

# 27. 最终 Git 审计

结束后：

```bash
git status --short
git diff --name-only
git diff --check
```

预期 scientific code/protocol：

```text
unchanged
```

允许新增：

```text
run artifacts
experiment logs
launcher artifacts
```

明确报告：

```text
shot_otta/** modified = false
ist_otta/** modified = false
nctta_otta/** modified = false
core/lbi/** modified = false
come_otta/** scientific implementation modified = false
```

---

# 28. STOP

完成 Office + VisDA baseline 全矩阵和汇总后立即停止。

禁止继续：

```text
COME LBI search
COME LBI formal
IST search
IST formal
paper table integration
new method development
```

---

# 29. Final report

只需要高密度汇报：

## A. Scheduler

```text
GPU 0/1 only
peak concurrent jobs
per-GPU concurrency
GPU utilization log path
OOM/retry count
```

## B. Completeness

```text
top-level: X / 147
actual executions: X / 231
```

## C. Office

```text
dense: X/18
FC sparse: X/54
Conv sparse: X/54
```

注意 Office sparse总计：

```text
108 top-level
```

## D. VisDA

```text
dense: X/3
FC sparse: X/9
Conv sparse: X/9
```

VisDA total：

```text
21 top-level
```

## E. Random audit

```text
all expected random identities 3/3 children? yes/no
```

## F. Failures

只列真实 failure / retry / quarantine。

## G. Results

给 Office aggregate + VisDA主表。

## H. Artifacts

给完整 summary/manifest路径。

## I. Isolation

```text
scientific code modified = false
```

## J. Verdict

只能：

```text
COME_OFFICE_VISDA_BASELINES_COMPLETE
```

或：

```text
COME_OFFICE_VISDA_BASELINES_INCOMPLETE
```

如果 COMPLETE，立即停止，不运行 LBI。
