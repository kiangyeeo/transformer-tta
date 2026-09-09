# CODEX — IST Office + VisDA All Baselines Formal — Direct 4-GPU Run

## Goal

基于**当前已经审计并通过 smoke 的 IST 代码直接开跑**全部正式 baseline：

- Office-31：全部 6 个 directed transfers
- VisDA-C：Train/Synthetic -> Validation/Real
- Dense baselines
- FC sparse non-LBI baselines
- Conv out-channel sparse non-LBI baselines

**不要 smoke。不要 dry-run。不要 parameter search。不要运行 LBI。不要改 scientific code。**

当前冻结语义以：

```text
260817_iclr2027-refined/protocol/ist-otta/OTTA_IST_BASELINE_PROTOCOL_20260906_v1.md
260817_iclr2027-refined/protocol/ist-otta/OTTA_IST_LBI_PROTOCOL_20260906_v1.md
```

为准。

当前 implementation revisions：

```text
IST baseline:
ist_otta_p1_baseline_20260906_v4

IST sparse/LBI:
ist_otta_sparse_lbi_20260906_v2
```

当前代码已经完成：

```text
P1 contracts PASS
Sparse/LBI contracts PASS
FC/Conv regression PASS
Office D->A short smoke PASS
```

因此本任务直接进入 formal baseline matrix。

---

# 0. 最高优先级：danger-full-access 安全边界

当前 Codex 配置：

```toml
approval_policy = "never"
sandbox_mode = "danger-full-access"
```

因此你必须主动遵守以下硬约束。

## 绝对禁止 destructive operations

禁止执行：

```text
rm
rm -rf
rmdir（针对已有目录）
find ... -delete
git clean
git reset --hard
git checkout -- <existing-file>
git restore <existing-file>
git revert
git rebase
git stash
mv 覆盖已有文件
cp -f 覆盖已有 scientific file
rsync --delete
sed -i 修改已有 scientific file
truncate 已有文件
删除 / 清空 / 覆盖任何已有 run artifact
```

不要删除任何失败 run。

不要为了“重跑干净”清目录。

如果已有 artifact 有问题：

```text
保留原 artifact
-> 标记 invalid / incomplete
-> 新建新的 attempt / timestamp run
```

## 禁止修改已有 scientific implementation

本任务只执行实验。

不要编辑、覆盖、格式化或自动修复：

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
configs/** 中任何已有文件
tests/**
```

不要运行会自动修改文件的 formatter / fixer。

允许**新增**：

```text
experiment_logs/ist_office_visda_all_baselines_20260908/**
runs/ist_formal_baselines_20260908/**
```

以及为本次 execution 新建：

```text
launcher script
queue manifest
scheduler state
stdout/stderr logs
summary scripts
summary artifacts
```

这些新文件只能服务于本次 IST formal baseline execution。

## 进程安全

COME baseline 在另一台机器上运行，与本任务无资源竞争。

本机只需要保证：

```text
只管理本 IST launcher 自己启动的 PID
不 kill / pause / renice 任何非本 launcher 进程
不修改任何无关任务的输出
```

---

# 1. Repository / environment

Repository：

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined
```

Conda：

```text
SHOT_TTA
```

进入：

```bash
cd /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined
conda activate SHOT_TTA
```

不要修改 conda env。

不要安装/升级 package。

---

# 2. Output isolation

COME 在另一台机器，不需要考虑 GPU / process isolation。

IST 本次统一写到独立 formal root：

```text
runs/ist_formal_baselines_20260908/
```

launcher / audit logs：

```text
experiment_logs/ist_office_visda_all_baselines_20260908/
```

不要把 IST formal artifact 写入：

```text
runs_smoke/
其他旧 IST run root
```

如果当前训练 CLI 有自己的 timestamp / identity 子目录机制，保留该机制。

如果当前 CLI 不支持 root override：

```text
不要修改 scientific code
```

只读确认其默认 IST output path不会覆盖已有 artifact；若会冲突，则停止并报告，不要现场改 scientific implementation。

---

# 3. GPU 分配：本机 4 张 GPU 全部用于 IST

本机已明确有 4 张 GPU 可用于本次 IST formal baselines。

任务开始时只做一次 read-only 检查：

```bash
nvidia-smi
```

确认 4 张 GPU 可见后，全部用于 IST。

如果物理 GPU 编号为：

```text
0,1,2,3
```

则直接使用：

```text
CUDA_VISIBLE_DEVICES=0,1,2,3
```

如果实际编号不是 0,1,2,3，则使用 `nvidia-smi` 显示的实际 4 张可用卡。

把最终 GPU IDs 记录到：

```text
experiment_logs/ist_office_visda_all_baselines_20260908/selected_gpus.txt
```

不需要检查 COME；COME 在另一台机器。

# 4. 并发调度

使用 4 张 selected GPUs。

默认：

```text
每 GPU 最多同时 2 个 IST formal jobs
global max concurrent IST jobs = 8
```

从正式任务开始直接 refill queue。

不需要额外 smoke/probe。

建议调度：

```text
每卡先启动 2 个不同 identity
-> job结束立即 refill
-> pending 非空时尽量不让 selected GPU 空闲
```

但必须遵守：

```text
scientific setting > utilization
```

绝不通过修改以下参数提升 GPU 利用率：

```text
batch size
workers
learning rate
augmentation
IST extend
PLCA
memory
optimizer
selector
budget
model
```

## OOM policy

如果某个 identity CUDA OOM：

1. 保留失败 artifact / log；
2. 记录 identity；
3. 不修改任何 scientific setting；
4. 不减少 batch size；
5. 将该 identity later retry 一次；
6. retry 时，该 GPU 暂时只运行这一条 job；
7. 成功/失败后恢复正常 refill。

如果 retry 仍 OOM：

```text
mark failed
继续其他独立 identities
最终报告 blocker
```

---

# 5. 防止 CPU / I/O 互相干扰

COME 同时在跑，因此不要无控制地再开额外数据处理进程。

正式数据设置继续使用 frozen：

```text
workers = 4
```

不要改 workers 来抢 CPU。

不要启动额外高 I/O 数据复制、压缩或 checksum 全盘扫描。

每个 job stdout/stderr 独立写日志。

---

# 6. Source checkpoint

IST 必须继续使用和 SHOT 完全一致的 source F/B/C：

```text
source_checkpoint_revision =
nips2026_shot_otta_uda_source_v1
```

使用当前 frozen IST config / resolver 中的 SHOT source root。

必须继续加载同一套：

```text
source_F.pt
source_B.pt
source_C.pt
```

禁止：

```text
重新训练 source
换 checkpoint
复制出 IST-specific source
改变 source revision
```

run artifact 中继续记录：

```text
source path
source revision
source SHA256
```

---

# 7. Dense formal config

优先使用 repository 中当前冻结的 IST baseline config：

```text
configs/ist_otta_p1_baseline_20260905_v1.yaml
```

若实际 repo 中该文件名已有 approved revision 后缀变化：

```text
只读检查当前 config / protocol revision
使用当前 approved IST baseline formal config
不要编辑旧文件
```

正式运行必须解析为：

```text
formal_protocol = true
debug_max_outer_batches = null
save_model = false
seed = 2026
```

Dense variants：

```text
ist_full_dense
ist_fc_module_dense
ist_conv_module_dense
```

---

# 8. Sparse formal config

运行 non-LBI sparse：

```text
ist_fc_random
ist_fc_magnitude
ist_fc_saliency

ist_conv_out_random
ist_conv_out_magnitude
ist_conv_out_saliency
```

必须：

```text
formal_protocol = true
debug_max_outer_batches = null
save_model = false
seed = 2026
lbi = null
```

明确禁止：

```text
ist_fc_lbi
ist_conv_out_lbi
```

## 如果当前 repo 已有 approved formal sparse config

直接使用。

## 如果当前 repo 只有 debug sparse config

**不要编辑 debug config。**

允许在：

```text
experiment_logs/ist_office_visda_all_baselines_20260908/
```

新增一个本次 launcher-local formal execution config。

该新文件必须：

1. 只继承当前 frozen baseline/sparse protocol 已经明确冻结的字段；
2. `formal_protocol=true`；
3. `debug_max_outer_batches=null`；
4. `save_model=false`；
5. `lbi=null`；
6. 不设置任何 LBI tuple；
7. 不改变 optimizer / LR / IST mechanism / source / augmentation / memory；
8. 不覆盖 `configs/**` 中已有文件。

如果 resolver 不接受 launcher-local config：

```text
STOP
报告 config blocker
不要修改 scientific code 解除限制
```

---

# 9. Office-31 matrix

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
```

Primary：

```text
sample-level overall accuracy
```

记录：

```text
PU
FO
```

---

# 10. VisDA-C matrix

只跑：

```text
source = train / synthetic
target = validation / real
```

即：

```text
Train/Synthetic -> Validation/Real
```

正式 resolver 必须保持：

```text
backbone = ResNet-101
batch_size = 256
workers = 4
base lr = 0.001
K = 12 classes
seed = 2026
one target pass
```

Primary：

```text
fixed-12-class mAcc
```

Secondary：

```text
overall accuracy
12 classwise accuracies
worst-class accuracy
classwise std
```

---

# 11. Dense baseline matrix

Dataset-transfer units：

```text
6 Office
+ 1 VisDA
= 7 units
```

每个 unit：

```text
ist_full_dense
ist_fc_module_dense
ist_conv_module_dense
```

所以：

```text
7 x 3 = 21 dense top-level formal runs
```

Dense 不按 rho 重复。

---

# 12. FC sparse non-LBI matrix

Variants：

```text
ist_fc_random
ist_fc_magnitude
ist_fc_saliency
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

总计：

```text
7 units
x 3 selectors
x 3 budgets
= 63 FC sparse top-level identities
```

---

# 13. Conv sparse non-LBI matrix

Variants：

```text
ist_conv_out_random
ist_conv_out_magnitude
ist_conv_out_saliency
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
rho_G=0.0005 -> K_G=4
rho_G=0.001  -> K_G=9
rho_G=0.002  -> K_G=18
```

总计：

```text
7 units
x 3 selectors
x 3 budgets
= 63 Conv sparse top-level identities
```

禁止：

```text
filter_connection
```

---

# 14. Random 必须真实执行 3 children

对：

```text
ist_fc_random
ist_conv_out_random
```

每个 dataset-transfer-budget top-level identity 必须真实执行：

```text
mask seed 202600
mask seed 202601
mask seed 202602
```

每个 child 必须：

```text
fresh same source F/B/C
fresh optimizer
fresh IST memory
same target stream seed=2026
same target sample order
same augmentation protocol
only random support differs
```

Top-level artifact 必须聚合：

```text
3 child PU
3 child FO
mean
std
mean single-child runtime
total 3-child operational runtime
```

不能：

```text
只跑一个 child
但 metadata 写 num_random_masks=3
```

---

# 15. 总工作量

## Top-level identities

Dense：

```text
21
```

FC sparse：

```text
63
```

Conv sparse：

```text
63
```

总计：

```text
147 top-level formal identities
```

## Actual executions

Dense：

```text
21
```

Magnitude + Saliency：

```text
7 units x 2 tracks x 2 selectors x 3 budgets
= 84
```

Random children：

```text
7 units x 2 tracks x 3 budgets x 3 children
= 126
```

总 actual executions：

```text
21 + 84 + 126
= 231
```

完成目标：

```text
147 / 147 top-level identities
231 / 231 actual executions
```

---

# 16. Office / VisDA 分开统计

## Office

Dense：

```text
6 x 3 = 18
```

FC sparse：

```text
6 x 3 selectors x 3 budgets
= 54 top-level
```

Conv sparse：

```text
54 top-level
```

Office total：

```text
126 top-level identities
```

Office actual executions：

```text
198
```

## VisDA

Dense：

```text
3
```

FC sparse：

```text
9
```

Conv sparse：

```text
9
```

VisDA total：

```text
21 top-level identities
```

VisDA actual executions：

```text
33
```

---

# 17. Source-only 不重跑

IST 与 SHOT 使用完全相同 source checkpoints。

不要创建新的：

```text
ist_source_only
```

最终 summary 可以引用已有 shared SHOT source-only reference，但必须核对：

```text
source revision
source SHA
dataset/transfer
metric definition
```

Source-only 不计入 147 identities。

---

# 18. 直接 formal：不要再做 smoke / tests / parameter probe

当前代码已经完成审计与 smoke。

本任务禁止在正式矩阵前重新进行：

```text
1-batch smoke
5-batch smoke
accuracy probe
parameter probe
LBI reachability probe
LBI search
full test suite
```

可以做的 preflight 仅限：

```text
read-only GPU isolation check
read-only CLI/config inspection
output path collision check
```

然后直接启动 formal queue。

---

# 19. Job launcher

在：

```text
experiment_logs/ist_office_visda_all_baselines_20260908/
```

新增 persistent launcher / queue state。

至少维护：

```text
pending
running
complete
failed
retry
quarantined
```

每个 top-level scientific identity 唯一化。

Launcher 必须：

1. 只分配选中的 4 张 physical GPU；
2. 每 GPU 最多 2 个 IST jobs；
3. global max 8 IST jobs；
5. job 完成立即 refill；
6. 记录 PID / physical GPU / identity / log path；
7. 捕获 exit code；
8. 单个 failure 不影响其他 independent identities；
9. 同一 identity 不重复并发；
10. Random top-level identity 内正确管理 3 children；
11. launcher restart 后能读取自己的 state 并避免重复 healthy completed identity；
12. 不删除历史 attempt。

每个 job 独立：

```text
stdout
stderr
exit code
start/end time
runtime
GPU
identity
```

---

# 20. Queue 排序：防止长尾

不要把 VisDA / target=A / Random 全留到最后。

采用：

```text
longest-first + mixed refill
```

优先把以下分散到 4 张 GPU：

```text
VisDA
D->A
W->A
Random identities
```

同时穿插：

```text
A->D
A->W
D->W
W->D
Magnitude
Saliency
dense
```

目标：

> 4 张 selected GPU 从开始到结束尽量都有健康 job。

---

# 21. 已存在 artifact：只读审核，绝不删除

如果 IST formal root 中已经有某 identity：

只有以下全部满足，才允许 skip：

```text
formal=true
formal_protocol=true
debug_only=false
summary complete
scientific identity exact
implementation revision exact
source revision exact
source checkpoint SHA exact
full stream complete
PU present
FO present
exit success
```

Random 还必须：

```text
3/3 children complete
aggregate complete
```

否则：

```text
不要删除旧 artifact
不要覆盖旧 artifact
```

建立新的 attempt / timestamp run。

不要因为“目录存在”就 skip。

---

# 22. Formal correctness invariants

依赖当前 frozen implementation 自动保证：

```text
same SHOT source F/B/C
netC frozen
scope exact
BN semantics exact
singleton early skip
PU read-only
FO read-only
target labels not in adaptation path
```

IST-specific：

```text
8 adaptation views
PLCA once / processed outer batch
memory commit once / processed outer batch
native IST iters=1
native EMA once / processed outer batch
```

Sparse：

```text
FC exact K
Conv exact K_G
off-mask exact
Random true 3 children
Magnitude source-static
Saliency current-state once / outer batch
```

不要修改这些语义。

---

# 23. Failure policy

## Infrastructure / transient failure

例如：

```text
CUDA OOM
temporary I/O
process crash
```

处理：

```text
mark failed
保留 log/artifact
immediately refill GPU with another pending identity
later retry failed identity once
```

OOM retry：

```text
same scientific settings
same batch size
same source
same seed
temporarily one IST job on that GPU
```

## Scientific/correctness failure

例如：

```text
source SHA mismatch
identity mismatch
NaN/Inf
parameter scope violation
formal marker wrong
target-label leakage
Random child count != 3
```

禁止现场改 setting 或 scientific code。

处理：

```text
quarantine identity
继续其他不相关 family jobs
记录 blocker
```

如果相同 correctness failure 在 >=2 个独立 identities 重现：

```text
停止启动新的相关 family jobs
不要杀掉其他已经健康运行的 family
报告 system-level blocker
```

---

# 24. 运行期间禁止结果驱动决策

禁止：

```text
看到 accuracy 低后改 LR
改 augmentation
改 memory
改 PLCA
改 EMA
换 source
改 selector
改 budget
只重跑低 accuracy identity
提前停止“看起来差”的 baseline
```

完整按 frozen matrix 跑。

即使某个 baseline accuracy 很低，也记录并继续。

---

# 25. 全部结束后做 artifact completeness audit

不是运行前 smoke。

全部 formal jobs 结束后检查：

```text
21 / 21 dense
63 / 63 FC sparse top-level
63 / 63 Conv sparse top-level
147 / 147 total top-level
231 / 231 actual executions
```

Office：

```text
126 / 126 top-level
198 / 198 actual
```

VisDA：

```text
21 / 21 top-level
33 / 33 actual
```

Random：

```text
每个 expected identity = 3 / 3 children
```

并验证全部 formal markers / source provenance。

---

# 26. Office summary

生成 Office 六迁移完整 baseline 表。

至少包含：

```text
shared source_only reference

ist_full_dense
ist_fc_module_dense
ist_conv_module_dense

FC:
Random / Magnitude / Saliency
rho=.0005/.001/.002

Conv out-channel:
Random / Magnitude / Saliency
rho_G=.0005/.001/.002
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

Random 聚合顺序必须：

```text
先在 transfer 内 3 child -> mean
再对 6 transfers equal-weight mean
```

不能把 18 Random children 当 18 transfers。

---

# 27. VisDA summary

同样列：

```text
shared source_only reference

3 dense

FC Random/Magnitude/Saliency x3 budgets
Conv Random/Magnitude/Saliency x3 budgets
```

Primary：

```text
PU fixed-12-class mAcc
FO fixed-12-class mAcc
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
3 child values
mean
std
```

---

# 28. 最终输出文件

统一生成到：

```text
experiment_logs/ist_office_visda_all_baselines_20260908/
```

至少：

```text
IST_OFFICE_BASELINES_SUMMARY.md
IST_OFFICE_BASELINES_SUMMARY.csv

IST_VISDA_BASELINES_SUMMARY.md
IST_VISDA_BASELINES_SUMMARY.csv

IST_ALL_BASELINES_MANIFEST.jsonl
IST_ALL_BASELINES_COMPLETENESS.json

selected_gpus.txt
gpu_scheduler.log
failed_or_retried_runs.jsonl
quarantined_runs.jsonl
```

所有文件都必须是新增文件。

不得覆盖 COME summary。

---

# 29. GPU scheduler log

每隔约 5 分钟记录 selected 4 GPUs：

```text
timestamp
physical GPU index
utilization.gpu
memory.used
IST active job count
```

只监控，不调整 scientific settings。

如果 pending queue 非空而某 selected GPU 没有 IST job：

```text
立即 refill
```



---

# 30. 最终 Git / filesystem isolation audit

结束后只做 read-only audit：

```bash
git status --short
git diff --name-only
git diff --check
```

不要执行任何 git write command。

明确报告：

```text
shot_otta/** modified by this task = false
ist_otta/** modified by this task = false
come_otta/** modified by this task = false
nctta_otta/** modified by this task = false
core/lbi/** modified by this task = false
train.py modified by this task = false
protocol/config existing files modified by this task = false

non-IST processes killed by this task = false
```

允许新增：

```text
IST formal run artifacts
IST experiment logs
IST launcher files
IST summary files
```

---

# 31. STOP

完成：

```text
Office + VisDA IST baseline full matrix
artifact audit
summary
```

后立即停止。

禁止继续：

```text
IST LBI search
IST LBI formal
COME LBI search
COME LBI formal
parameter tuning
paper table integration
NCTTA/NSTTA development
任何代码修改
```

---

# 32. Final report

只做高密度汇报：

## A. GPU usage

```text
selected physical GPUs
peak IST concurrent jobs
per-GPU max concurrency
GPU scheduler log
```

## B. Completeness

```text
top-level: X / 147
actual executions: X / 231
```

## C. Office

```text
dense: X / 18
FC sparse: X / 54
Conv sparse: X / 54
Office top-level: X / 126
Office actual: X / 198
```

## D. VisDA

```text
dense: X / 3
FC sparse: X / 9
Conv sparse: X / 9
VisDA top-level: X / 21
VisDA actual: X / 33
```

## E. Random audit

```text
all expected Random identities 3/3 children? YES/NO
```

## F. Failures

只列真实：

```text
OOM
retry
quarantine
incomplete
```

不要把低 accuracy 当 failure。

## G. Results

给：

```text
Office aggregate table
VisDA primary table
```

## H. Artifacts

给 summary / manifest / scheduler log 路径。

## I. Safety / isolation

明确：

```text
existing scientific code edited = NO
existing files deleted = NO
existing run artifacts overwritten = NO
non-IST processes touched = NO
```

## J. Verdict

只能：

```text
IST_OFFICE_VISDA_BASELINES_COMPLETE
```

或：

```text
IST_OFFICE_VISDA_BASELINES_INCOMPLETE
```

如果 COMPLETE：

> 立即停止，不运行 LBI，不开始参数搜索。
