# CODEX — COME Fast Feasibility Gate

目标：在投入任何 COME sparse/LBI 开发前，用最小代价判断 **COME objective 是否适合当前 matched FC/Conv adaptation spaces**。

这是 feasibility gate，不是正式 COME implementation。

---

## 0. 最高优先级：禁止修改其他方法代码

本任务只能新增/修改 **COME 自己的最小实现、COME 专用配置、COME 专用测试/诊断、以及必要的顶层 method dispatch**。

### 严禁修改

```text
shot_otta/**
ist_otta/**
nctta_otta/**
core/lbi/**
现有 SHOT / IST / NCTTA protocol
现有 SHOT / IST / NCTTA config
现有 SHOT / IST / NCTTA scientific semantics
现有 shared LBI engine
```

除非顶层 method dispatch 必须增加 `COME` 分支，否则不要修改任何 shared 文件。

如果确实必须修改：

```text
train.py
experiment_identity.py
protocol_constants.py
```

只能做 **COME-specific additive extension**，不得改变任何已有 SHOT / IST / NCTTA 分支行为。

开始前保存：

```bash
git status --short
git diff --name-only
```

完成后必须再次运行：

```bash
git status --short
git diff --name-only
git diff --check
```

并明确报告：

```text
shot_otta/** modified = false
ist_otta/** modified = false
nctta_otta/** modified = false
core/lbi/** modified = false
```

如果任务过程中误改了其他方法文件，先恢复这些误改，再继续。

---

## 1. 官方 source audit

先审计官方 repo：

```text
https://github.com/BlueWhaleLab/COME
```

记录 audited official commit SHA。

重点读取：

```text
README.md
tent_come.py
COME objective / Algorithm 1
```

先确认：

1. COME 核心 objective；
2. logit-norm detach semantics；
3. Dirichlet subjective-opinion construction；
4. 类别数 `K` 的定义；
5. `p / tau` 默认值；
6. native Tent-COME trainable scope。

禁止凭记忆实现。

---

## 2. 关键迁移规则：K 是类别数

官方 ImageNet 实现可能直接写 `1000`，但论文中的：

```text
K = total class number
```

所以当前 controlled study 必须是：

```text
Office-31: K = 31
VisDA-C:   K = 12
```

本轮只跑 Office：

```text
K = 31
```

禁止把 ImageNet-specific `1000` 硬编码带入 Office objective。

---

## 3. COME objective

新增独立 COME 模块，优先保持最小：

```text
come_otta/
├── __init__.py
├── config.py
├── objective.py
└── trainer.py
```

不要复用、复制或修改 NCTTA/IST objective。

对 logits `z`，严格按官方实现移植 constrained-logit semantics。概念上：

```python
norm = ||z||_2
z_constrained = z / norm * norm.detach()
```

必须核对并保持官方：

```text
norm dimension
epsilon / numerical safety
p
tau
detach location
```

不要因为 forward value 看起来相同就删除 detach。

然后按官方 COME 定义构造 subjective opinion：

```text
evidence
strength S
belief masses b_k
uncertainty mass u
entropy of opinion
```

对于 Office：

```text
K = 31
```

最终 loss：

```text
mean entropy_of_opinion
```

保持官方 `p=2`、`tau=1`，若官方 audited code/paper有不同，以官方 source 为准并在最终报告说明。

禁止增加：

```text
pseudo-label
feature geometry
teacher
memory
EMA
confidence filter
extra regularization
source-feature anchor
```

---

## 4. 只实现三个临时 controlled dense variants

只允许：

```text
come_full_dense
come_fc_module_dense
come_conv_module_dense
```

不要创建其他 COME variant。

### 4.1 `come_full_dense`

与当前 SHOT controlled full-dense scope 完全一致：

```text
netF + netB trainable
netC frozen
BN semantics = current SHOT full-dense semantics
```

### 4.2 `come_fc_module_dense`

只允许：

```text
netB.bottleneck.weight
netB.bottleneck.bias
```

candidate scalar count：

```text
524544
```

其他全部 frozen。

所有 BN：

```text
parameters frozen
running buffers frozen
```

`netC` frozen。

### 4.3 `come_conv_module_dense`

只允许 formal：

```text
netF.layer4 9 Conv weights
```

candidate scalar count：

```text
12845056
```

其他全部 frozen。

所有 BN：

```text
parameters frozen
running buffers frozen
```

`netC` frozen。

---

## 5. Common substrate 不允许改

三个 COME variants 必须从当前 SHOT formal 的同一 source F/B/C 开始。

固定：

```text
source revision = nips2026_shot_otta_uda_source_v1
same source F/B/C checkpoint
same netF -> netB -> netC
Office backbone = ResNet-50
seed = 2026
BS = 64
workers = 4
same target sample order
same outer-batch partition
one target pass semantics
drop_last = false
same actual-BS=1 singleton skip
same PU definition
same FO definition
same Office metric
same optimizer
same LR
same scheduler
```

特别禁止为了 COME 稳定而：

```text
改 LR
改 scheduler
改 weight decay
改 momentum
改 BN semantics
改 source checkpoint
改 batch size
改 target order
```

本轮就是测试：

> COME objective 在当前 matched substrate 上能否天然支持 full / FC / Conv controlled adaptation。

---

## 6. 只跑 3 个 5-batch tasks

固定：

```text
dataset = Office-31
transfer = D -> A
seed = 2026
BS = 64
debug_max_outer_batches = 5
formal = false
```

只运行：

```text
come_full_dense
come_fc_module_dense
come_conv_module_dense
```

禁止运行：

```text
COME native
Random
Magnitude
Saliency
LBI
VisDA
其他 Office transfers
parameter search
formal experiment
```

---

## 7. Collapse diagnostics

每个 processed outer batch 记录 detached/read-only diagnostics：

```text
predicted_class_histogram
predicted_class_count
dominant_class_count
dominant_class_ratio
mean_softmax_entropy
COME opinion entropy
COME mean uncertainty mass
gradient_norm
parameter_update_norm
relative_parameter_update_norm
PU
```

FO 记录：

```text
FO predicted_class_histogram
FO predicted_class_count
FO dominant_class_ratio
FO mean_softmax_entropy
FO COME mean uncertainty mass
FO accuracy
```

这些 diagnostics：

```text
must be detached
must be read-only
must not change objective graph
must not change optimizer trajectory
```

---

## 8. Correctness contracts

新增 COME 专用 feasibility contract。

至少验证：

1. COME loss 对 synthetic tensor 与独立 direct reference 数值一致。
2. COME gradient 对 direct reference 一致。
3. logit norm detach 位置与官方实现一致。
4. Office objective 确实使用 `K=31`。
5. 代码中不存在 Office 路径硬编码 `1000`。
6. source F/B/C 与 SHOT exact same。
7. adaptation 前 deterministic logits 与 SHOT source branch一致。
8. `netC` byte-identical。
9. full-dense update scope 与 SHOT full-dense一致。
10. FC 只允许 bottleneck weight/bias改变。
11. Conv 只允许 layer4 9 Conv weights改变。
12. FC/Conv BN parameters 和 buffers byte-identical。
13. 每 valid outer batch exactly one objective call。
14. 每 valid outer batch exactly one scheduler update。
15. 每 valid outer batch exactly one optimizer step。
16. singleton 在 adaptation/objective/scheduler/optimizer/PU 前 skip。
17. PU read-only。
18. FO read-only。
19. target labels 不进入 COME objective。
20. existing SHOT / IST / NCTTA tests/regressions仍 PASS。

Contract 必须检查实际 state transition，不只看 `requires_grad=True`。

---

## 9. Feasibility gate

本轮禁止根据 PU/FO accuracy 调参。

先检查：

```text
loss finite
gradient finite
scope correct
prediction diversity trajectory
dominant-class trajectory
softmax entropy trajectory
COME uncertainty trajectory
relative update norm
```

重点判断 FC / Conv 是否复现之前 NCTTA 的：

```text
predicted_class_count -> 1
dominant_class_ratio -> 1
softmax entropy -> 0
```

### GO

如果：

```text
FC 没有快速走向单类 collapse
Conv 没有快速走向单类 collapse
loss/gradient finite
scope contract PASS
PU/FO read-only PASS
```

输出：

```text
COME_FEASIBILITY_GO
```

### STOP

如果 FC 或 Conv 出现明显快速单类 collapse：

```text
COME_FEASIBILITY_STOP
```

然后立即停止。

禁止：

```text
现场调 COME 参数
改 LR
扩大 grid
换 feature
实现 LBI
```

---

## 10. Independent review

Tests PASS 后人工检查：

```text
COME official objective mapping
detach location
K=31
trainable scopes
BN mode/buffers
optimizer param groups
outer scheduler timing
singleton
PU/FO
artifact diagnostics
top-level dispatch
scientific identity
```

重点确认：

```text
没有为了实现 COME 而改变 SHOT / IST / NCTTA 的任何行为
```

---

## 11. Stop condition

完成以下内容后立即停止：

```text
official COME audit
minimal COME objective
3 controlled dense variants
COME feasibility contract
existing regressions
independent review
3 x 5-batch D->A smoke
collapse summary
```

禁止继续：

```text
COME baseline protocol freeze
COME sparse protocol
Random
Magnitude
Saliency
LBI
VisDA
other Office transfers
hyperparameter search
formal runs
```

---

## 12. 最终输出

输出：

1. audited official COME commit SHA；
2. 官方 COME objective 的逐项代码对应；
3. `K=31` 的迁移依据；
4. changed files；
5. 明确列出未修改：
   - `shot_otta/**`
   - `ist_otta/**`
   - `nctta_otta/**`
   - `core/lbi/**`
6. contract results；
7. SHOT / IST / NCTTA regression results；
8. 三条 5-batch smoke collapse diagnostics；
9. 一张 summary table：

```text
variant
PU
FO
predicted_class_count trajectory
dominant_class_ratio trajectory
softmax entropy trajectory
COME uncertainty trajectory
relative update norm trajectory
stability verdict
```

10. 最终 verdict：

```text
COME_FEASIBILITY_GO
```

或：

```text
COME_FEASIBILITY_STOP
```
