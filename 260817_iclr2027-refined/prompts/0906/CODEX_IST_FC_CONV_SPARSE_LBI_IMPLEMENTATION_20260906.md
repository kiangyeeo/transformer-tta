# Codex Prompt：实现 IST FC + Conv Sparse/LBI

严格执行当前项目中的正式协议：

```text
260817_iclr2027-refined/protocol/ist-otta/OTTA_IST_BASELINE_PROTOCOL_20260906_v1.md
260817_iclr2027-refined/protocol/ist-otta/OTTA_IST_LBI_PROTOCOL_20260906_v1.md
```

当前 P1 baseline implementation revision：

```text
ist_otta_p1_baseline_20260906_v4
```

## 最高优先级约束

### 1. 禁止修改 SHOT-OTTA

**任何情况下都不要修改：**

```text
260817_iclr2027-refined/shot_otta/
```

同时禁止改变现有 SHOT 的任何 scientific semantics、config、variant、artifact 或运行行为。

现有 SHOT FC/Conv/LBI 代码只能作为 **read-only reference**。

### 2. 复用当前 corrected LBI semantics

允许读取：

```text
core/lbi/
```

理解当前 refined FC-LBI / Conv Group-LBI 的正确实现。

优先复用现有通用 LBI engine；不要复制旧 `IST-OTTA-LBI`。

如果现有 `core/lbi/` 已足以支持 IST，则不要修改。

如确实必须修改 shared `core/lbi/`，只能做：
- objective-agnostic 的通用扩展；
- 不改变现有 SHOT 输入下任何结果；
- 必须增加 regression test 证明 SHOT semantics 不变。

**严禁修改 `shot_otta/` 来适配 IST。**

---

# 任务

在当前 IST P1 baseline 上实现完整的：

## FC family

```text
ist_fc_module_dense      # 已有，保持不变
ist_fc_random
ist_fc_magnitude
ist_fc_saliency
ist_fc_lbi
```

FC candidate 严格为：

```text
netB.bottleneck.weight
netB.bottleneck.bias
```

并严格使用 protocol 中的：

```text
N_FC = 524544
rho = 0.0005 / 0.001 / 0.002
K = 262 / 524 / 1049
```

---

## Conv family

```text
ist_conv_module_dense    # 已有，保持不变
ist_conv_out_random
ist_conv_out_magnitude
ist_conv_out_saliency
ist_conv_out_lbi
```

Conv candidate 严格为：

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

仅实现：

```text
grouping = out_channel
```

严格使用：

```text
N_conv = 12845056
num_groups = 9216

rho_G = 0.0005 / 0.001 / 0.002
K_G = 4 / 9 / 18
```

**不要实现 / 恢复 filter_connection。**

---

# IST × sparse 的关键语义

所有规则以两个 protocol 为准，尤其保证：

## Random / Magnitude / Saliency

每个 valid outer batch：

```text
8-view
-> pre-adaptation prediction
-> PLCA exactly once
-> memory commit exactly once
-> sparse support
-> native IST iters=1 under fixed mask
-> IST native EMA m=0.9 exactly once
-> PU read-only
```

其中：

### Random
- exact-K / exact-K_G；
- 3 个 deterministic independent child masks；
- mask 整个 target stream 固定。

### Magnitude
- source checkpoint 上计算；
- global top-K / top-K_G；
- mask 整个 target stream 固定。

### Saliency
- 每个 outer batch 只计算一次 support；
- 使用当前 persistent pre-adaptation model；
- 使用 **完整 fixed outer-batch IST objective**；
- FC：

$$
s_j=|\theta_j g_j|
$$

- Conv out-channel：

$$
s_g=\|W_g\odot\nabla_{W_g}\mathcal L_{\mathrm{IST}}\|_2
$$

- 当前 outer batch 的所有 IST inner mini-batches 共用同一 mask；
- 禁止每个 inner mini-batch 重新选 support。

---

# IST-LBI 最关键语义

严格执行 protocol：

```text
一个 valid outer batch
-> 只做一次 LBI support discovery
```

不是每个 IST inner mini-batch 一次。

当前 outer batch：

```text
8-view fixed
hard PL fixed
soft targets fixed
PLCA once
memory once
```

然后 Stage 1 使用：

> 完整 fixed outer-batch IST objective。

显存不足允许 gradient accumulation，但：

```text
chunk 之间：
- 不允许 optimizer step
- 不允许 LBI state update

完整 full-objective gradient 得到后：
- 才允许一次 LBI dynamics update
```

每个 outer batch：

```text
theta_delta = 0
Z = 0
Gamma = 0
```

重新开始 support discovery。

必须保留当前 corrected semantics：

```text
old-state Z update
support threshold = 1e-4
stage1_max_steps = 3000
strict integer budget
overshoot -> rollback latest feasible
no budget slack
no top-K repair
```

Stage 2：

```text
theta_init = theta_base + M * theta_delta
```

禁止 off-mask Stage-1 delta leakage。

Stage 2：

```text
exactly 1 masked SGD step
```

并且这一 step 同样基于完整 fixed outer-batch IST objective。

Stage-2 optimizer：

```text
SGD
momentum = 0.9
weight_decay = 0.001
nesterov = true
fixed stage2_lr
```

---

# IST EMA 与 LBI omega

这一条必须严格实现：

## 非 LBI

```text
dense
random
magnitude
saliency
```

继续：

$$
\theta_{t+1}
=
0.9\theta_t+0.1\widetilde\theta_t
$$

即 native IST EMA。

## LBI

```text
native IST EMA = DISABLED
```

唯一 persistent writeback：

$$
\theta_{t+1}
=
(1-\omega)\theta_t
+
\omega\widetilde\theta_t.
$$

禁止：

```text
LBI omega -> IST EMA
IST EMA -> LBI omega
```

任何 double writeback 都是 correctness bug。

---

# 参数与 config

本阶段只完成实现。

不要拍脑袋确定正式 LBI tuned tuple。

LBI 只暴露：

```text
alpha
kappa
nu
omega
stage2_lr
```

正式 search grid / final tuples 仍为 TBD。

可以提供 debug/test config，但：

```text
不得把未冻结默认值伪装成 formal tuned config
不得跑 formal Office/VisDA experiments
```

---

# Singleton / PU / FO

继续严格保持 P1：

```text
actual outer batch size == 1
-> 在任何 view / PLCA / memory / selector / LBI / EMA / PU 前 skip
```

PU、FO 均 read-only。

Target labels 不能进入 adaptation path。

---

# 实现位置

优先将 IST-specific 实现放在：

```text
ist_otta/
```

可以新增清晰模块，例如：

```text
ist_otta/sparse.py
ist_otta/lbi.py
ist_otta/objective.py
```

具体结构根据当前代码决定，避免重复代码。

允许修改：

```text
ist_otta/
tests/ist_otta_*
protocol_constants.py
experiment_identity.py
train.py
configs/
```

但 shared 文件只能增加 **IST-only dispatch / metadata**，不得改变 SHOT 行为。

明确禁止修改：

```text
shot_otta/**
```

---

# Contract tests

实现后增加/扩展 IST-specific contract tests，至少覆盖 protocol 第 30 节全部要求。

重点必须自动验证：

```text
PLCA once / outer batch
memory once / outer batch

LBI once / outer batch
not once / inner mini-batch

full-objective gradient accumulation correctness

correct old-state Z
strict rollback

FC:
N=524544
K=262/524/1049

Conv:
groups=9216
K_G=4/9/18

Stage2 exactly one optimizer step
off-mask exact preservation

Random/Magnitude/Saliency:
native IST EMA exactly once

LBI:
native IST EMA count = 0
LBI omega writeback count = 1

no double writeback

BN unchanged in controlled FC/Conv
netC unchanged
PU/FO read-only
singleton skip
no target-label leakage
```

同时确认现有：

```text
tests/ist_otta_p1_contract_test.py
```

仍然 PASS。

---

# 本任务不要做的事情

不要：

```text
跑完整 Office formal
跑完整 VisDA formal
调 LBI 超参数
写 search protocol
根据 accuracy 改算法
恢复 filter_connection
修改 SHOT-OTTA
修改已有 SHOT scientific results
```

允许运行：

```text
unit tests
contract tests
必要的 synthetic/local correctness tests
```

暂时不要跑大规模 smoke，代码完成并审查后再统一安排。

---

# 最终汇报

完成后只给我：

1. 修改/新增文件列表；
2. 每个文件做了什么；
3. FC 5 个 variants 的实现状态；
4. Conv 5 个 variants 的实现状态；
5. contract test 结果；
6. P1 contract 是否仍 PASS；
7. 是否修改了 `shot_otta/` —— 必须明确回答 `NO`；
8. 是否存在尚未解决的 correctness 问题；
9. 不要运行 formal experiments。
