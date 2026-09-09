# CODEX — NCTTA P1.5 Native Norm Reference

在当前 `260817_iclr2027-refined/` 上补充 **official/native NCTTA normalization-only reference**。当前 P1 dense implementation 已完成并通过审计；不要重构或修改已有 `nctta_full_dense / nctta_fc_module_dense / nctta_conv_module_dense` 的科学语义。

## 1. 任务目标

新增唯一一个 native-scope reference：

```text
nctta_native_norm
```

它回答的是：

> 在完全相同的 SHOT source F/B/C、target stream、batch size、PU/FO 和 benchmark 上，如果保留官方 NCTTA 的原生 normalization-only adaptation scope，表现如何？

它与现有 controlled variants 的角色必须明确区分：

```text
nctta_native_norm
    = official NCTTA update-scope reference

nctta_full_dense
    = NCTTA objective + controlled full scope

nctta_fc_module_dense
    = NCTTA objective + controlled FC scope

nctta_conv_module_dense
    = NCTTA objective + controlled Conv scope
```

不要创建 `native_full / native_fc / native_conv` 等重复概念。

## 2. Source of truth

严格按以下优先级：

1. 当前 `260817_iclr2027-refined/`
   - source F/B/C
   - Office/VisDA benchmark
   - target stream
   - seed / batch size / singleton
   - PU / FO
   - runtime / provenance

2. 官方 NCTTA：
   - https://github.com/Cevaaa/NCTTA
   - `ttab/model_adaptation/nctta.py`
   - 尤其检查官方：
     - `configure_model`
     - `collect_params`
     - `compute_nc_loss`
     - `compute_loss_function`
     - adaptation step

3. 当前 `nctta_otta/`
   - 已验证的 NCTTA objective port
   - 不得重新发明 loss

禁止凭记忆实现。

## 3. Native trainable scope 必须 faithful

先逐行核对官方 `configure_model()` 和 `collect_params()`，再实现。

当前已知官方语义是：

```text
freeze full model
enable normalization affine parameters
paper 使用 BN
BatchNorm2d:
    weight/bias trainable
    track_running_stats = False
    running_mean = None
    running_var = None
```

尤其注意：

> 官方代码检查的是具体 normalization module type；不要因为我们 `netB` 中存在 BatchNorm1d，就自行扩展官方 scope。

如果当前官方源码仍然只对 `BatchNorm2d` 执行上述 BN adaptation，则 `nctta_native_norm` 必须严格如此：

```text
netF 中 ResNet BatchNorm2d affine:
    trainable

netB 中 BatchNorm1d:
    frozen

all Conv:
    frozen

bottleneck Linear:
    frozen

netC:
    frozen
```

实现前输出一份实际 trainable parameter manifest，包括：

```text
parameter names
tensor count
scalar count
module type
```

禁止笼统写“all BN”。

## 4. Source / architecture 不变

必须继续使用与 SHOT/IST 完全相同的：

```text
source revision = nips2026_shot_otta_uda_source_v1
same source_F / source_B / source_C
same netF -> netB -> netC
Office = ResNet-50
VisDA = ResNet-101
seed = 2026
Office BS = 64
VisDA BS = 256
one target pass
same outer target order
same singleton semantics
```

禁止：
- 使用官方 NCTTA pretrained checkpoint；
- 重训 source；
- 换 backbone；
- 换 classifier/head。

## 5. NCTTA objective 不改

直接复用已经审计通过的当前：

```text
nctta_otta/objective.py
```

不要重新实现一套 native loss。

仍然使用：

$$
h=\mathrm{netB}(\mathrm{netF}(x))
$$

作为 256-d classifier-input feature，并使用：

$$
W_C=\mathrm{netC.fc.weight.detach()}
$$

作为 effective frozen classifier reference。

必须保持现有已验证的：

```text
top-k
q_dist
q_prob
hybrid target
InfoNCE NC loss
cosine geometry
tau_align=1.0
entropy
entropy filtering
entropy reweight
FCA-distance reweight
weighted selected mean
```

因此 native 与 controlled variants 的主要区别是：

```text
update scope + BN behavior
```

而不是换一个 NCTTA loss。

## 6. Native BN behavior 与 PU/FO

这里必须认真检查，不要套 controlled FC/Conv 的 BN frozen 逻辑。

官方 NCTTA 对适应 BN 使用：

```text
track_running_stats = False
running_mean = None
running_var = None
```

因此需要验证 PyTorch 在 adaptation / PU / FO 下的真实行为。

要求：

### Adaptation
使用官方 native BN semantics。

### PU
仍然遵守当前项目定义：

```text
adaptation 后
same current batch/reference view
read-only
不得 optimizer.step()
不得改变 parameter
不得创建额外 persistent update
```

但 native BN 若本身依赖 current batch statistics，应忠实保留这种 forward semantics，不要为了“看起来更像 controlled BN”擅自恢复 source running stats。

### FO
最终模型 parameter 冻结，FO 不得更新 parameter/optimizer/state。

如果 `track_running_stats=False` 导致 FO prediction 仍依赖 evaluation-batch statistics，应：

1. 保持官方 native BN forward semantics；
2. 使用当前固定 FO data order / batch protocol；
3. artifact 明确记录：

```text
native_norm_fo_uses_batch_statistics = true
```

不要偷偷把 native NCTTA 改成 source-running-stat evaluation。

## 7. Optimizer

这里不要默认照搬 controlled full/FC/Conv 的 param groups。

先检查官方 NCTTA optimizer setup。

为了这个 **native reference**，目标是尽量 faithful 地保留官方 NCTTA trainable scope；但 source/network/stream/evaluation 仍共享当前 substrate。

请明确输出：

```text
official NCTTA optimizer type
official LR
official momentum / weight decay if any
current implemented native-reference optimizer
```

如果当前 P0/baseline protocol 已经明确要求 native reference 也使用 common SHOT optimizer substrate，则遵守现有 protocol；否则不要自行猜测，先在 protocol amendment 中明确决定并解释。

无论采用哪种方式：

> `nctta_native_norm` 不参与 FC/Conv matched sparse comparison，它只是 official/native-scope reference，因此 optimizer provenance 必须单独写清楚。

## 8. Hyperparameters

NCTTA objective hyperparameters与当前 P1保持一致，不在本任务调参。

当前 smoke 可以继续使用已记录的 official repository defaults。

禁止根据 native smoke accuracy：
- 改 entropy threshold；
- 改 scale；
- 改 top-k；
- 改 mix weight；
- 改 LR；
- 改 objective。

Smoke 只做 correctness diagnosis。

## 9. Protocol / revision

在现有：

```text
protocol/nctta-otta/OTTA_NCTTA_BASELINE_PROTOCOL_20260906_v1.md
```

基础上新增 native-reference semantics，并升级为：

```text
OTTA_NCTTA_BASELINE_PROTOCOL_20260906_v2.md
```

不要覆盖 v1 历史文件。

新的 implementation revision 建议升级为：

```text
nctta_otta_baseline_20260906_v3
```

更新：
- `protocol_constants.py`
- config
- `experiment_identity.py`
- contracts

确保 artifact 中 protocol revision 指向真实存在的文件。

## 10. Contract tests

新增/扩展 contract，至少逐项验证：

1. 与 SHOT 加载 exact same source F/B/C。
2. adaptation 前 deterministic logits 一致。
3. stream / batch boundaries / singleton 一致。
4. 无 target-label / future-target leakage。
5. NCTTA objective 与已审计 P1 objective 完全复用。
6. post-netB feature = 256-d。
7. `netC` byte-level unchanged。
8. 实际 trainable scope 与官方 normalization selector严格一致。
9. 如果官方当前代码只选 `BatchNorm2d`，则 `netB` 的 `BatchNorm1d` 必须不更新。
10. Conv / FC / classifier 全部不更新。
11. BN running-stat semantics 与官方一致。
12. 每个 valid outer batch exactly one adaptation optimizer step。
13. PU 不做额外 update。
14. FO 不做 parameter/state writeback。
15. protocol / implementation revision / artifact identity 一致。
16. 已有三个 controlled NCTTA dense variants regression PASS。
17. SHOT / IST regression PASS。

特别加一个 **scope snapshot test**：

```text
before outer batch
after outer batch
```

逐 parameter + BN buffer 比较，明确列出真正变化的 tensor。

不要只测 `requires_grad=True`，必须测实际 execution 后哪些 tensor 发生了变化。

## 11. Independent code review

tests PASS 后人工检查：

```text
variant dispatch
configure_model/native BN setup
collect_params
optimizer param groups
BatchNorm2d vs BatchNorm1d
train/eval mode transition
PU
FO
singleton
artifact identity
protocol revision
```

尤其防止：

```text
metadata 说 native BN-only
但 optimizer 实际包含别的 tensor
```

## 12. Short smoke

只跑：

```text
Office D->A
seed=2026
BS=64
5 valid outer batches
variant=nctta_native_norm
```

同时和已有 P1 smoke 放在一起报告：

```text
source-only
nctta_native_norm
nctta_full_dense
nctta_fc_module_dense
nctta_conv_module_dense
```

但不要重跑后三个，直接读取已有 smoke artifacts即可。

记录：

```text
selected samples / batch
loss_ent
loss_nc
total loss
gradient norm
actual changed tensor count
actual changed scalar count
PU
FO
```

Smoke accuracy 只用于诊断，不用于改参数。

## 13. 重点诊断问题

我们目前已有 5-batch smoke：

```text
nctta_full_dense:
PU 60.00 / FO 51.86

nctta_fc_module_dense:
PU 21.56 / FO 3.51

nctta_conv_module_dense:
PU 24.38 / FO 3.48
```

这次 native reference 的目的之一是区分：

### 情况 A
```text
native_norm 正常
FC/Conv collapse
```

则说明当前 NCTTA objective port 大概率没问题，collapse 更可能来自：

```text
NCTTA objective
+
large FC/Conv adaptation space
+
current optimization scale
```

这是后续 tuning/search 问题，不允许在本任务修改 feature 或 objective。

### 情况 B
```text
native_norm 也 collapse
```

则标记为需要进一步 audit：

```text
F/B/C geometry compatibility
BN semantics
NCTTA hyperparameter transfer
Office-specific threshold
optimizer/LR compatibility
```

仍然不要现场“修 accuracy”。

## 14. 禁止事项

本任务禁止：

```text
Random
Magnitude
Saliency
LBI
Conv grouping
parameter search
formal runs
source-feature anchoring
frozen initial feature teacher
new stabilization loss
修改 shot_otta/**
修改 IST scientific semantics
```

尤其不要把 NCTTA feature 改成 source 初始 feature。官方 NCTTA 使用当前模型当前 forward 的 feature；改成 source feature 会变成新的 source-anchor 方法。

## 15. Stop condition

完成：

```text
native_norm implementation
protocol v2
contracts
regressions
independent review
D->A 5-batch smoke
```

然后停止。

输出：

1. changed files；
2. 官方 native NCTTA scope 的逐项源码对应；
3. 实际 trainable parameter manifest；
4. BatchNorm2d / BatchNorm1d 行为；
5. optimizer provenance；
6. contract/regression results；
7. independent review findings；
8. native 5-batch smoke；
9. 与现有 full/FC/Conv smoke 的诊断比较；
10. unresolved items；
11. 最终 verdict：

```text
READY_FOR_NCTTA_SPARSE_PROTOCOL
```

或

```text
NOT_READY_FOR_NCTTA_SPARSE_PROTOCOL
```