# CODEX — NCTTA Host Objective Fast Diagnostic

目标：在进入任何 LBI parameter search 之前，用最小实验量诊断并稳定 NCTTA host objective。

当前 NCTTA dense/native/sparse/LBI implementation 均已通过 correctness audit。**本任务禁止修改算法代码、feature semantics、candidate scope、optimizer substrate 或 LBI。**

## 1. 只做 Office D->A 5-batch diagnostic

固定：

```text
dataset = Office-31
transfer = D -> A
classes = 31
seed = 2026
BS = 64
debug_max_outer_batches = 5
formal = false
```

只跑：

```text
nctta_native_norm
nctta_fc_module_dense
nctta_conv_module_dense
```

不要跑：

```text
full_dense
Random
Magnitude
Saliency
LBI
VisDA
其他 Office transfers
```

## 2. Optimizer / feature 全部保持现状

禁止修改：

```text
current post-netB 256-d feature
effective frozen netC.fc.weight
current-state NCTTA target reconstruction
controlled FC/Conv optimizer
LR
scheduler
BN semantics
native_norm optimizer semantics
```

尤其禁止：

```text
source-feature anchoring
frozen batch-start feature
LR /10
LR /100
new stabilization loss
```

## 3. 只测试四个 NCTTA objective tuples

设：

```text
H31 = 0.4 * log(31)
H1000 = 0.4 * log(1000)
```

### A — current reference

```text
thre_ent   = H1000
margin_ent = H1000
scale      = 5.0
nctta.nu   = 5.0
```

### B — class-count normalized threshold

```text
thre_ent   = H31
margin_ent = H31
scale      = 5.0
nctta.nu   = 5.0
```

### C — normalized threshold + reduced NC scale

```text
thre_ent   = H31
margin_ent = H31
scale      = 1.0
nctta.nu   = 5.0
```

### D — normalized threshold + reduced NC/reweight strength

```text
thre_ent   = H31
margin_ent = H31
scale      = 1.0
nctta.nu   = 1.0
```

所有 tuples 固定：

```text
reweight_ent = 1.0
eta = 1.0
top_k = 10
mix_prob_weight = 0.3
```

不要扩 grid。

## 4. 增加 collapse diagnostics

每个 processed outer batch记录：

```text
predicted_class_histogram
predicted_class_count
dominant_class_count
dominant_class_ratio
mean_prediction_entropy
selected_count
selected_fraction
mean_fca_distance
loss_ent
loss_nc
total_loss
gradient_norm
parameter_update_norm
relative_parameter_update_norm
PU
```

FO额外记录：

```text
FO predicted_class_histogram
FO predicted_class_count
FO dominant_class_ratio
FO mean_prediction_entropy
FO accuracy
```

这些 diagnostics 必须 detached/read-only，不得改变 objective gradient。

## 5. Selection rule 不允许看 accuracy winner

本轮 accuracy 只用于诊断，不允许：

```text
argmax PU
argmax FO
```

来选择 tuple。

先按无监督 stability 判断是否明显 collapse。

至少报告：

```text
predicted_class_count trajectory
dominant_class_ratio trajectory
selected_fraction trajectory
entropy trajectory
relative update norm trajectory
```

若某 tuple 出现：

```text
non-finite loss/gradient
zero selected samples
near-single-class prediction concentration
rapid prediction-diversity collapse
extreme update-norm explosion
```

标记：

```text
STABILITY_FAIL
```

对于未失败 tuples：

> 选择与 official/current defaults 改动最小的 tuple 作为下一阶段 candidate。

如果无法仅凭预注册 stability diagnostics 唯一选出一个 tuple，不得用 D->A accuracy tie-break；输出 remaining eligible tuples，停止。

## 6. 不修改正式 protocol

这是 diagnostic/search-design 前置实验：

```text
debug_only = true
formal = false
```

不要冻结新的 formal NCTTA objective tuple。

输出独立 summary，例如：

```text
runs_diagnostic/nctta_host_objective/office/DA/...
```

## 7. 最终输出一个 3 x 4 表

行：

```text
native_norm
fc_module_dense
conv_module_dense
```

列：

```text
A
B
C
D
```

每格至少给：

```text
PU
FO
predicted_class_count
dominant_class_ratio
mean entropy
selected fraction
relative update norm
stability verdict
```

并回答：

1. 单纯把 `log(1000)` 改为 `log(31)` 是否解决 instability？
2. 降低 `scale` 是否明显改善？
3. 降低 `nctta.nu` 是否进一步改善？
4. native / FC / Conv 是否表现出同一种 instability？
5. 哪些 tuples 是 `STABILITY_PASS`？
6. 不看 accuracy 时，能否按“minimal deviation from official defaults”唯一选出一个 candidate？

## 8. Stop

完成上述 12 个 5-batch runs 后停止。

禁止：

```text
扩大 grid
VisDA
其他 Office transfer
LBI tuning
formal runs
修改 feature
修改 LR
修改代码语义
```

最终 verdict：

```text
HOST_DIAGNOSTIC_HAS_STABLE_CANDIDATE
```

或：

```text
HOST_DIAGNOSTIC_NO_STABLE_CANDIDATE
```