# CODEX：IST FC / Conv Sparse-LBI Short Smoke（2026-09-06）

严格以以下正式协议为规范：

```text
260817_iclr2027-refined/protocol/ist-otta/OTTA_IST_BASELINE_PROTOCOL_20260906_v1.md
260817_iclr2027-refined/protocol/ist-otta/OTTA_IST_LBI_PROTOCOL_20260906_v1.md
```

当前已审查的 sparse/LBI implementation revision：

```text
ist_otta_sparse_lbi_20260906_v2
```

Conda 环境：

```text
SHOT_TTA
```

本任务只做：

> **contract regression + Office D→A short smoke correctness validation**

完成后立即停止。

不要做参数搜索，不要跑 formal experiments，不要开始 NCTTA。

---

# 0. 最高优先级约束

## 禁止修改 SHOT-OTTA

任何情况下都不要修改：

```text
260817_iclr2027-refined/shot_otta/**
```

SHOT 代码只能 read-only。

如果 smoke 暴露 IST-specific correctness bug，优先只修改：

```text
ist_otta/**
tests/ist_otta_*
configs/ist_otta_*
```

只有当确认是 shared objective-agnostic bug 时，才允许最小修改：

```text
core/lbi/**
```

并且必须证明 SHOT regression 仍然 PASS。

禁止为了让 smoke 通过而改变 protocol 科学语义。

---

# 1. 先跑 contracts

```bash
cd /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined
conda activate SHOT_TTA

python tests/ist_otta_p1_contract_test.py
python tests/ist_otta_sparse_lbi_contract_test.py
```

如果 repository 中已有本轮相关的：

```text
shared FC LBI regression
Conv LBI dynamics regression
revision / protocol / config regression
```

也一并运行。

要求全部 PASS。

如果 contract 失败：

1. 先定位 correctness 原因；
2. 只做最小修复；
3. 不修改 `shot_otta/**`；
4. 重跑相关 contracts；
5. 全部通过后再开始 smoke。

---

# 2. Smoke 范围

只跑：

```text
Dataset  : Office-31
Transfer : D -> A
Source   : D
Target   : A
Seed     : 2026
Outer BS : 64
```

Smoke-only：

```text
debug_max_outer_batches = 5
formal_protocol = false
```

这里的 `5` 指：

> **5 个实际 processed、非-singleton outer batches。**

不要用 timeout / kill 人为截断。

必须让 run 正常走 artifact finalize。

输出放到与正式结果完全分开的目录，例如：

```text
runs_smoke/ist_sparse_lbi/
```

不要写入 formal run 目录。

---

# 3. 本次只跑 8 条 sparse/LBI variants

Dense/module-dense 已在 P1 smoke 验证过，本任务不要重复。

## FC

```text
ist_fc_random
ist_fc_magnitude
ist_fc_saliency
ist_fc_lbi
```

## Conv out-channel

```text
ist_conv_out_random
ist_conv_out_magnitude
ist_conv_out_saliency
ist_conv_out_lbi
```

禁止：

```text
filter_connection
```

---

# 4. LBI smoke 参数

正式 tuned tuples 仍然是 TBD。

本次 LBI 只能使用 repository 中：

```text
configs/ist_otta_sparse_lbi_debug_20260906_v1.yaml
```

所明确标记的 **debug-only / untuned** 参数。

如果该 config 中有 FC / Conv debug tuple，原样使用。

如果当前 config 需要 variant/budget override，按已有 config schema 做最小 override。

不要：

```text
搜索参数
比较哪个 tuple accuracy 更高
根据 smoke accuracy 改参数
把 debug tuple 写成 formal tuned tuple
解除 formal LBI launch block
```

本 smoke 的 accuracy 不用于任何参数选择。

为减少 smoke 成本，每个 LBI track 只需要选一个 debug budget。

优先使用：

```text
rho = 0.001
```

即：

```text
FC   K   = 524
Conv K_G = 9
```

除非当前 debug config 已明确指定另一个 smoke budget；若如此，遵循现有 debug config，并在汇报中说明。

---

# 5. Random smoke 必须真实跑 3 个 child masks

`ist_fc_random` 和 `ist_conv_out_random` 必须验证当前 v2 修复后的真实三-mask执行。

每个 Random top-level smoke 必须有：

```text
random_mask_index = 0
random_mask_index = 1
random_mask_index = 2
```

并满足：

```text
3 个 child 都从 fresh same source checkpoint 开始
3 个 child 使用相同 D->A seed-2026 target stream
3 个 child 使用相同 augmentation protocol
只允许 Random support 不同
每个 child 都独立处理 5 个 valid outer batches
```

正式 seed=2026 时，检查 child selection seeds 与当前 frozen implementation 一致。

必须产生：

```text
3-mask child results
3-mask mean
3-mask std
mean single-mask runtime
3-mask total operational runtime
```

不能退化成只跑 1 个 mask、metadata 却写 3 masks。

---

# 6. 每条 smoke 必须检查的 correctness

## 6.1 所有 8 条 run

检查：

```text
processed_outer_batches = 5
singleton skip 计数合理
loss 全部 finite
无 NaN / Inf
netC unchanged
target labels 未进入 adaptation path
PU read-only
artifact finalize 完整
debug_smoke = true
debug_max_outer_batches = 5
formal_protocol = false
```

Office D→A 前 5 个 valid outer batches 若均为 BS64，则每个独立 execution 最终 memory size 应为：

$$
5\times64\times8=2560.
$$

如果不是 2560，不要直接判 bug；先检查是否有 singleton / 数据边界差异，并明确解释。

---

# 7. FC sparse correctness

对：

```text
ist_fc_random
ist_fc_magnitude
ist_fc_saliency
ist_fc_lbi
```

验证 persistent change 只允许发生在：

```text
netB.bottleneck.weight
netB.bottleneck.bias
```

并检查：

```text
all BN parameters/buffers unchanged
netF unchanged
netC unchanged
```

预算：

```text
rho=.001
K=524
```

若使用其他 debug rho，则使用 protocol 对应严格 integer K。

### FC Random

每个 child：

```text
selected_count = exact K
mask 整条 child stream 固定
```

### FC Magnitude

```text
selected_count = exact K
source-checkpoint mask 整条 stream 固定
```

### FC Saliency

每个 processed outer batch：

```text
support selection count = 1
selected_count = exact K
当前 outer batch 的所有 IST inner mini-batches 共用同一 mask
```

### FC LBI

每个 processed outer batch：

```text
LBI support discovery count = 1
theta_delta/Z/Gamma restart once
selected_count <= K
strict budget 无 violation
Stage2 optimizer step count = 1
native IST EMA count = 0
LBI omega writeback count = 1
```

5 个 processed outer batches 总计应有：

```text
LBI outer-batch calls = 5
Stage2 optimizer steps = 5
native IST EMA commits = 0
omega writebacks = 5
```

记录：

```text
selected_count
utilization
Stage-1 steps
rollback/overshoot status
Stage-1 cap hit
```

Smoke 允许 utilization 低；这里只检查 correctness，不用 utilization 选参数。

---

# 8. Conv out-channel sparse correctness

对：

```text
ist_conv_out_random
ist_conv_out_magnitude
ist_conv_out_saliency
ist_conv_out_lbi
```

persistent change 只允许发生在 formal layer4 9 Conv weights：

```text
netF.layer4.0.conv1.weight
netF.layer4.0.conv2.weight
netF.layer4.0.conv3.weight
netF.layer4.1.conv1.weight
netF.layer4.1.conv2.weight
netF.layer4.1.conv3.weight
netF.layer4.2.conv1.weight
netF.layer4.2.conv2.weight
netF.layer4.2.conv3.weight
```

检查：

```text
all BN parameters/buffers unchanged
netB unchanged
netC unchanged
```

Grouping：

```text
out_channel only
num_groups = 9216
```

Budget 若 smoke 用 `.001`：

```text
K_G = 9
```

### Conv Random

每个 child：

```text
selected_group_count = exact K_G
mask 整条 child stream 固定
```

### Conv Magnitude

```text
selected_group_count = exact K_G
source-checkpoint mask 整条 stream 固定
```

### Conv Saliency

每个 processed outer batch：

```text
support selection count = 1
selected_group_count = exact K_G
当前 outer batch 所有 IST inner mini-batches 共用同一 group mask
```

### Conv LBI

每个 processed outer batch：

```text
Group-LBI discovery count = 1
Theta_delta/Z/Gamma restart once
selected_group_count <= K_G
strict budget 无 violation
Stage2 optimizer step count = 1
native IST EMA count = 0
omega writeback count = 1
```

5 个 processed outer batches 总计：

```text
Group-LBI calls = 5
Stage2 optimizer steps = 5
native IST EMA commits = 0
omega writebacks = 5
```

同时必须记录：

```text
selected_group_count
realized_group_ratio
selected_scalar_count
realized_scalar_ratio
group utilization
Stage-1 steps
rollback/overshoot status
Stage-1 cap hit
```

---

# 9. 非-LBI EMA 检查

对于：

```text
Random
Magnitude
Saliency
```

每个独立 execution 的 5 个 processed outer batches 应满足：

```text
native IST EMA commits = 5
LBI omega writebacks = 0
```

Random 是 3 个 child executions，因此每个 child 独立满足上述计数。

---

# 10. Full-objective / PLCA / memory 检查

每个独立 execution：

```text
PLCA calls = 5
memory commits = 5
```

LBI：

```text
Stage 1 内禁止额外 PLCA
Stage 1 内禁止额外 memory commit
Stage 2 内禁止额外 PLCA
Stage 2 内禁止额外 memory commit
```

Saliency：

```text
每 outer batch只做一次 full-objective saliency support selection
不是每 inner mini-batch一次
```

如果 run/artifact 当前没有这些 counters，但 contract 已验证，可以通过日志/trace/diagnostic assertion 核验。

不要为了 smoke 增加会改变 scientific identity 的 debug mechanism。

---

# 11. Smoke 中发现 bug 时怎么办

如果发现 correctness bug：

允许直接修复，但必须遵守：

```text
1. 不改 shot_otta/**
2. 不改变冻结 protocol 科学语义
3. 优先只改 ist_otta/**
4. shared core/lbi 修改必须 objective-agnostic 且有 SHOT regression
5. 修复后重跑 contracts
6. 只重跑受影响 smoke
```

如果只是：

```text
accuracy 低
utilization 低
Stage-1 steps 多但未 hit cap
```

不要修改算法或参数。

这些属于后续 search protocol / tuning 阶段，不是 smoke correctness bug。

---

# 12. 本任务明确禁止

不要：

```text
跑完整 Office 6 transfers
跑 VisDA formal
跑任何 formal experiment
开始 LBI 参数搜索
写 IST_LBI_SEARCH_PROTOCOL
冻结 alpha/kappa/nu/omega/stage2_lr
根据 smoke accuracy 选择参数
修改 SHOT-OTTA
恢复 filter_connection
修改 NCTTA
开始 NCTTA 相关工作
```

本次 smoke 完成并汇报后：

> **立即停止。**

用户下一步会单独处理 NCTTA 代码。

---

# 13. 最终汇报格式

完成后只汇报：

## A. Contracts

```text
P1 contract:
Sparse/LBI contract:
Shared FC LBI regression:
Conv LBI regression:
其他相关 regression:
```

全部写 PASS / FAIL。

## B. 8 条 smoke 状态

逐条：

```text
ist_fc_random
ist_fc_magnitude
ist_fc_saliency
ist_fc_lbi
ist_conv_out_random
ist_conv_out_magnitude
ist_conv_out_saliency
ist_conv_out_lbi
```

每条写：

```text
PASS / FAIL
processed outer batches
final memory size
loss finite?
parameter-scope preservation?
BN unchanged?
netC unchanged?
PU read-only?
```

Random 额外写：

```text
3 child masks 是否真的全部执行
child selection seeds
每 child processed batches
3-mask aggregate artifact 是否生成
```

LBI 额外写：

```text
budget
mean/min/max selected support
mean/min utilization
Stage-1 mean/max steps
cap-hit count
rollback count
native IST EMA count
omega writeback count
Stage2 step count
```

Conv 额外写：

```text
selected scalar support / realized scalar ratio
```

## C. 修改情况

明确回答：

```text
是否修改 shot_otta/: NO
是否因为 smoke 修改代码:
若有，列出文件和 correctness 原因
```

## D. 最终结论

只回答：

```text
IST sparse/LBI smoke 是否可以关闭：YES / NO
是否存在尚未解决 correctness bug：YES / NO
```

然后停止。

不要继续参数搜索，不要继续 formal experiments，不要继续 NCTTA。
