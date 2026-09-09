# Codex Prompt — P1 Conv-LBI 代码重构（暂不测试 / 不跑实验）

你现在在项目：

```text
260817_iclr2027-refined/
```

中工作。

本轮目标只有一个：

> **基于当前 refined FC-LBI 框架，完成 Conv-layer Group-LBI 与对应 controlled Conv baselines 的代码重构。**

本轮**不要启动任何 test，不要运行任何 Office / VisDA 实验，不要搜参，不要决定 Conv 的 rho grid**。

---

## 0. 先读规范与当前实现

先完整阅读以下文件，后续实现以这些文件为规范来源。

### Conv 新 protocol

```text
260817_iclr2027-refined/protocol/shot-otta_conv/OTTA_CONV_LBI_PROTOCOL_20260826_DRAFT_v0.md
```

这是本轮 Conv 重构最直接的规范。

### FC frozen protocol

```text
260817_iclr2027-refined/protocol/shot-otta_fc/OTTA_FC_LBI_PROTOCOL_20260817_v1.md
```

Conv 中凡是可以继承的 benchmark / evaluation / corrected LBI semantics，都保持与 FC 一致。

### 当前 refined FC-LBI 实现

```text
260817_iclr2027-refined/core/lbi/engine.py
260817_iclr2027-refined/core/lbi/state.py
260817_iclr2027-refined/core/lbi/diagnostics.py
```

### 当前 SHOT-OTTA pipeline

```text
260817_iclr2027-refined/shot_otta/trainer.py
260817_iclr2027-refined/shot_otta/config.py
260817_iclr2027-refined/shot_otta/artifacts.py
260817_iclr2027-refined/shot_otta/efficiency.py
260817_iclr2027-refined/experiment_identity.py
260817_iclr2027-refined/protocol_constants.py
260817_iclr2027-refined/train.py
```

先理解并保持当前 FC 的这些 refined semantics：

```text
old-state z update
tau = 1e-4 support
strict integer budget
rollback
masked-delta Stage-2 initialization
off-mask value freezing
fixed Stage-2 LR
omega persistent writeback
PU / FO read-only semantics
formal seed = 2026
runtime / GPU-memory protocol
```

---

# 1. 总体原则

不要复制或复活旧版 NIPS Conv-LBI pipeline。

基于当前 refined FC framework 增加 **group-aware sparse adaptation**。

统一关系：

```text
FC   : element-wise / singleton-group sparsity
Conv : group-wise / Group-Lasso sparsity
```

要求：

- 不改变当前 FC scientific semantics；
- 不修改 FC frozen protocol；
- 不因为加入 Conv 而改变已有 FC experiment identity；
- Conv 和 FC protocol / implementation identity 必须能够并存；
- 能复用当前 refined LBI 逻辑就复用，不另建一套散乱的 Conv-LBI 主循环；
- 不为了抽象而大规模重写已经稳定的 FC pipeline。

---

# 2. Conv candidate scope

Controlled Conv family 只允许以下 9 个 Conv weight tensor：

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

不要包含：

```text
其他 layer
Conv bias
BN affine
BN running statistics
netB
netC
```

当前 candidate scalar 总数应为：

```text
12,845,056
```

Controlled Conv family 下：

```text
只有这 9 个 Conv weight 可以 persistent change
其他参数全部 frozen
所有 BN affine frozen
所有 BN running statistics frozen
BN modules 使用 eval / frozen behavior
```

---

# 3. 两种正式 Conv group mode

两种都要实现，之后都会进入论文。

## 3.1 out_channel

对于：

$$
W \in \mathbb{R}^{C_{out}\times C_{in}\times K_h\times K_w}
$$

第 $o$ 个 group：

$$
G_o = W[o,:,:,:].
$$

整个 `layer4` candidate 的总 group 数应为：

```text
9216
```

---

## 3.2 filter_connection

第 $(o,i)$ 个 group：

$$
G_{o,i} = W[o,i,:,:].
$$

整个 `layer4` candidate 的总 group 数应为：

```text
6,553,600
```

---

## 3.3 实现要求

不存在：

```text
相邻 Conv shared 1D gate
physical channel pruning
真正删除 source weight / 改网络 shape
```

未选 group 的语义只是：

> 该 group 不允许发生 target adaptation。

Group 相关计算必须 vectorized。

不要 Python 循环枚举数百万个 group。

建议：

```text
out_channel:
[Cout, Cin, Kh, Kw]
→ [Cout, Cin*Kh*Kw]
→ 沿最后一维算 group norm

filter_connection:
[Cout, Cin, Kh, Kw]
→ [Cout, Cin, Kh*Kw]
→ 沿最后一维算 group norm
```

Group mask 最终 broadcast 回 parameter-level bool mask。

---

# 4. Conv Group-LBI Stage 1

Conv 必须继承 refined FC 的 corrected old-state semantics。

定义：

$$
c^k
=
\frac{\Theta_\Delta^k-\Gamma^k}{\nu}.
$$

更新：

$$
\Theta_\Delta^{k+1}
=
\Theta_\Delta^k
-
\alpha\kappa
\left(
g^k+c^k
\right),
$$

$$
Z^{k+1}
=
Z^k+\alpha c^k.
$$

其中：

$$
g^k
=
\nabla_{\Theta_\Delta}
\mathcal L_{\mathrm{adp}}
(B_t;\Theta_t+\Theta_\Delta^k).
$$

注意：

> `Z^(k+1)` 必须使用 old-state coupling，即 $(\Theta_\Delta^k,\Gamma^k)$，不能使用已经更新后的 $\Theta_\Delta^{k+1}$。

---

# 5. Group-Lasso proximal operator

Conv 与 FC 真正不同的地方是 prox。

对每个 group $g$：

$$
\Gamma_g^{k+1}
=
\kappa
\left(
1-
\frac{1}{\|Z_g^{k+1}\|_2}
\right)_+
Z_g^{k+1}.
$$

要求：

- zero-norm 情况数值安全；
- 不要通过 `eps` 偷偷改变 support / budget 的数学定义；
- out_channel 与 filter_connection 共用同一套 Group-Lasso 逻辑，只改变 group reduction 维度。

---

# 6. Group support semantics

固定：

$$
\tau=10^{-4}.
$$

Group support 定义为：

$$
M_g
=
\mathbf{1}
\left[
\|\Gamma_g\|_2
\ge
\tau
\right].
$$

同一个 thresholded group support 必须同时用于：

```text
Stage-1 group counting
realized group ratio
strict budget checking
final group mask
masked-delta Stage-2 initialization
Stage-2 trainable support
```

不要再使用：

```text
gamma != 0
raw nonzero count
不同阶段不同 threshold
```

---

# 7. Global strict group budget

这轮**不要决定正式 rho grid**。

只实现 budget 的数学语义。

对于显式输入的：

```text
requested_budget = rho
```

定义：

$$
K_G
=
\left\lfloor
\rho
|\mathcal G|
\right\rfloor.
$$

必须是：

> **所有 9 个 Conv layers 合并后的 global group pool。**

禁止：

```text
per-layer K
minimum-one-per-layer
ceil
budget tolerance / slack
LBI overshoot 后 top-K trim
```

Stage-1 逻辑：

```text
n < K_G
→ 当前 state 合法
→ 作为 latest feasible state
→ 继续

n = K_G
→ 接受当前 state
→ 停止

n > K_G
→ 当前 overshoot state 无效
→ 回到最近 feasible state
→ 停止
```

最终必须满足：

$$
n_{\mathrm{selected}}
\le
K_G.
$$

---

# 8. Rollback 实现注意事项

Conv candidate 有约 12.8M scalar。

不要在每一个 Stage-1 step 都无脑深拷贝整套：

```text
Theta_delta
Gamma
Z
```

作为 rollback buffer。

优先考虑：

> candidate next-state → 计算 support → 合法才 commit

这种非破坏式更新方式。

目标是同时满足：

```text
数学上等价于 rollback 到最近 feasible state
又避免每步复制巨大 Conv state
```

如果当前 engine 架构不方便，先做最小必要改动，不要为了优化把整个 LBI engine 重写。

---

# 9. Stage 2

Stage-2 initialization 必须是 masked delta：

$$
\Theta_{\mathrm{init}}
=
\Theta_{\mathrm{base}}
+
M
\odot
\Theta_\Delta.
$$

绝对不能：

$$
\Theta_{\mathrm{base}}
+
\Theta_\Delta
$$

dense apply。

Stage 2：

```text
fresh local SGD
momentum = 0.9
weight_decay = 0.001
nesterov = true
stage2_steps = 1
fixed stage2_lr
不使用 local SHOT LR decay
```

并同时保证：

```text
mask gradient
optimizer step 后 restore off-mask parameter values
off-mask Conv weight 严格保持不变
```

不能只依赖 gradient masking，因为 weight decay / momentum / Nesterov 仍可能改变 off-mask 参数。

---

# 10. Stage 3 persistent writeback

Stage 2 得到 refined state 后：

$$
\Theta_{t+1}
=
(1-\omega)\Theta_t
+
\omega\widetilde{\Theta}_t.
$$

下一个 online batch：

```text
Theta_delta = 0
Gamma = 0
Z = 0
```

重新发现新的 group support。

Group mask 不跨 batch 固定。

---

# 11. Conv controlled baselines

新增以下 variants：

```text
conv_module_dense

conv_out_random
conv_out_magnitude
conv_out_saliency
conv_out_lbi

conv_filter_random
conv_filter_magnitude
conv_filter_saliency
conv_filter_lbi
```

---

## Random

对应 grouping 下，在 **global group pool** 中均匀抽取 exact $K_G$ groups。

继续继承 FC formal semantics：

```text
formal seed = 2026
3 deterministic child masks
accuracy 后续取 3-mask mean
```

---

## Magnitude

每个 group：

$$
s_g=\|W_g\|_2.
$$

全局：

```text
global top-K_G
```

---

## Saliency

每个 current online batch 计算：

$$
s_g
=
\|W_g\odot\nabla_{W_g}\mathcal L\|_2.
$$

全局：

```text
dynamic top-K_G
```

---

## 禁止旧 Conv baseline 行为

```text
per-layer random K
per-layer magnitude top-K
per-layer saliency top-K
minimum-one-per-layer
```

Random / Magnitude / Saliency / LBI 必须共享同一个 global group pool 定义。

---

# 12. Diagnostics / artifacts

Conv sparse runs 需要能记录以下字段。

至少新增 / 明确：

```text
group_mode
candidate_layer_names
candidate_scope_param_count

total_group_count

requested_budget
max_group_count

selected_group_count
realized_group_ratio

selected_scalar_count
realized_scalar_ratio

support_threshold

stage1_steps_completed
stage1_stop_reason
stage1_rollback_used
stage1_last_feasible_step
```

并保留当前已有：

```text
PU / FO
runtime
Stage-1 / Stage-2 runtime
GPU peak memory
scientific provenance
implementation / protocol identity
```

重要：

```text
selected_group_count
```

和：

```text
selected_scalar_count
```

必须区分。

如果已有 `selected_param_count` 字段为了兼容需要继续保留，要明确它究竟表示 scalar count 还是 group count，不允许语义混用。

---

# 13. Config / identity

FC 现有：

```text
OTTA_FC_LBI_PROTOCOL_20260817_v1
iclr2027_refined_20260817_v1
```

必须保持不变。

Conv protocol 当前：

```text
protocol/shot-otta_conv/OTTA_CONV_LBI_PROTOCOL_20260826_DRAFT_v0.md
```

其中 Conv implementation revision 目前仍然允许保持：

```text
TBD_AFTER_CONV_REFACTOR
```

本轮不要擅自命名并冻结最终 Conv implementation revision。

但代码结构必须已经支持：

```text
FC protocol identity
Conv protocol identity
```

并存。

Conv scientific identity 至少需要区分：

```text
candidate scope
group_mode
group semantics
requested_budget
LBI tuple
formal seed
SHOT config
dataset / transfer
```

特别注意：

> 扩展 `experiment_identity.py` 时，不要让已有 FC scientific hashes 发生变化。

---

# 14. 当前 rho 处理

本轮不决定正式 Conv rho grid。

因此：

```text
不要新增正式 rho list
不要复制 FC 的 .0005/.001/.002 当成 Conv 默认值
不要建立 Conv tuning matrix
```

Sparse Conv variant 如果需要 budget：

> 必须显式传入 requested_budget。

若缺失，应 fail closed，而不是偷偷继承 FC budget。

---

# 15. 本轮明确不做的事情

本轮不要：

```text
不要启动任何 tests
不要运行任何 smoke test
不要运行 pytest
不要跑 Office
不要跑 VisDA
不要跑 dry-run adaptation
不要跑 budget pilot
不要搜 rho
不要搜 alpha/kappa/nu/omega/stage2_lr
不要生成正式实验矩阵
不要冻结 Conv final implementation revision
不要修改 FC protocol
```

这一轮只改代码。

---

# 16. 代码完成后的静态自检

只做静态检查，不执行测试。

请自行检查：

```text
是否存在明显 syntax / import / naming 问题
是否误改 FC semantics
是否仍有 per-layer sparse budget
是否仍有 dense Stage-2 initialization
是否仍有 new-state z update
是否仍有 gamma != 0 support
是否仍有 local SHOT Stage-2 scheduler
是否 group_mode 正确进入 config / identity / artifacts
是否 selected_group_count 与 selected_scalar_count 区分
```

不要通过启动测试来验证。

---

# 17. 最终只汇报

完成代码修改后，只给我：

1. 修改 / 新增了哪些文件；
2. 每个文件改了什么；
3. Conv 如何接入当前 refined FC-LBI framework；
4. 两种 group mode 如何实现；
5. Group-Lasso prox 如何实现；
6. global group budget / rollback 如何实现；
7. Stage-2 masked init / off-mask freeze 如何实现；
8. Conv Random / Magnitude / Saliency 如何改成 global group selection；
9. config / identity / artifacts 新增了什么；
10. 当前仍有哪些 TODO；
11. 哪些 TODO 与 `rho grid = TBD` 有关；
12. **明确写：本轮没有运行任何 tests，也没有运行任何实验。**

不要继续进入测试、budget pilot、搜参或正式实验。
