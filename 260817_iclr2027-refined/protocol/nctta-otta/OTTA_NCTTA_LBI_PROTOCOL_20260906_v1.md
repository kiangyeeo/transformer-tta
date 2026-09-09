# OTTA_NCTTA_LBI_PROTOCOL_20260906_v1

**状态：** 科学语义与 sparse/LBI implementation revision 已冻结；在 NCTTA formal objective parameters、search protocol 与 tuned LBI tuples 冻结前，formal sparse/LBI launch 保持 blocked  
**Protocol revision：** `OTTA_NCTTA_LBI_PROTOCOL_20260906_v1`  
**Parent NCTTA baseline protocol：** `OTTA_NCTTA_BASELINE_PROTOCOL_20260906_v2`  
**Parent refined FC protocol：** `OTTA_FC_LBI_PROTOCOL_20260817_v1`  
**Parent refined Conv protocol：** `OTTA_CONV_LBI_PROTOCOL_20260826_v1`  
**Current NCTTA baseline implementation：** `nctta_otta_baseline_20260906_v3`  
**NCTTA sparse/LBI implementation revision：** `nctta_otta_sparse_lbi_20260907_v2`  
**Source checkpoint revision：** `nips2026_shot_otta_uda_source_v1`  
**日期：** 2026-09-06  
**范围：** controlled NCTTA objective + FC scalar sparse adaptation + Conv out-channel structured sparse adaptation

> 本文件冻结 **NCTTA × sparse/LBI 的科学语义**。它不冻结 NCTTA dataset-specific formal objective parameters，也不冻结 LBI search grid、utilization gate、selection rule 或最终 tuned tuples；这些必须在所有代码完成、contract/review/smoke 通过后再单独 preregister。

## 1. 科学问题

本研究要验证同一个 corrected Sparse Delta Learning / LBI optimizer 是否同时具备：

1. **objective generality**：SHOT-OTTA、IST-OTTA、NCTTA objective；
2. **parameter-structure generality**：FC scalar 与 Conv out-channel group。

目标矩阵：

| Adaptation objective | FC scalar | Conv out-channel |
|---|---|---|
| SHOT-OTTA | 已完成 | 已完成 |
| IST-OTTA | 已实现，待统一 search/formal | 已实现，待统一 search/formal |
| NCTTA | 本 protocol | 本 protocol |

核心原则：

> 从 SHOT/IST 换成 NCTTA 时，只替换 adaptation objective；candidate universe、budget semantics、corrected LBI dynamics、Stage-2 与 evaluation protocol 不重新发明。

抽象形式：

$$
g^k
=
\nabla_{\Theta_\Delta}
\mathcal L_{\mathrm{NCTTA}}
\left(
B_t;
\Theta_t+\Theta_\Delta^k
\right).
$$

这里的 NCTTA objective 是一个**current-state objective**：其 top-$k$、hybrid target、entropy filter 与 sample weights 都依赖当前 candidate model state，并非 batch 开头一次生成后固定的 pseudo-target。

## 2. Source-of-truth 优先级

发生冲突时按以下顺序处理：

1. `OTTA_NCTTA_LBI_PROTOCOL_20260906_v1`
2. `OTTA_NCTTA_BASELINE_PROTOCOL_20260906_v2`
3. 当前 corrected refined LBI engine + frozen refined FC/Conv protocol semantics
4. 当前已审计的 `nctta_otta/objective.py`
5. 官方 `Cevaaa/NCTTA` audited commit `b4d442472a36af6b3f4d6e75f5138ca97d7d8eec`
6. NCTTA P0/P1 contracts/audits
7. IST sparse/LBI implementation仅作**结构模板**，不得把 IST-specific state semantics 搬入 NCTTA

禁止为了复用 IST 代码而给 NCTTA 人为加入 fixed pseudo-target、memory、EMA、multi-view 或 inner-loop semantics。

## 3. 正式 variant 范围

### 3.1 FC controlled family

```text
nctta_fc_module_dense
nctta_fc_random
nctta_fc_magnitude
nctta_fc_saliency
nctta_fc_lbi
```

### 3.2 Conv controlled family

```text
nctta_conv_module_dense
nctta_conv_out_random
nctta_conv_out_magnitude
nctta_conv_out_saliency
nctta_conv_out_lbi
```

Conv grouping 主线只允许：

```text
out_channel
```

禁止重新打开：

```text
filter_connection
```

`nctta_native_norm` 是 native-scope reference，不属于 FC/Conv matched sparse family，不按 sparse budget 重复运行。

## 4. 从 NCTTA baseline protocol 继承的规则

所有 NCTTA sparse/LBI variants 继承 parent baseline：

```text
same source F/B/C
source revision = nips2026_shot_otta_uda_source_v1
same netF -> netB -> netC
Office R50 / VisDA R101
seed-2026 fixed outer stream
Office BS64 / VisDA BS256
workers=4
one target pass
drop_last=false
actual-BS=1 singleton skip
same source/evaluation transforms
same PU / FO definitions
same metrics
netC frozen
same artifact / provenance discipline
```

对于 controlled FC/Conv family 额外继承：

```text
all BN parameters frozen
all BN running buffers frozen
same controlled SHOT optimizer/scheduler substrate for Dense/Random/Magnitude/Saliency
```

本文件没有显式 override 的 baseline 规则继续有效。

### 4.1 Dataset-specific benchmark / metric invariants

以下指标必须在本 sparse/LBI protocol 中显式保留，不能只依赖 parent protocol 的隐式继承：

| Item | Office-31 | VisDA-C |
|---|---|---|
| Classes | 31 | 12 |
| Backbone | ResNet-50 | ResNet-101 |
| Outer batch size | 64 | 256 |
| Formal seed | 2026 | 2026 |
| Target passes | 1 | 1 |
| Primary PU metric | sample-level overall accuracy | **fixed-12-class mean per-class accuracy (mAcc)** |
| Primary FO metric | sample-level overall accuracy | **fixed-12-class mean per-class accuracy (mAcc)** |
| Formal aggregation | 6 directed transfers 等权平均 | Synthetic/Train -> Real/Validation 单 transfer |
| Secondary reporting | transfer-wise accuracy | overall accuracy + class-wise accuracy + worst-class + class-wise std |

Office 正式 transfers：

```text
A -> D
A -> W
D -> A
D -> W
W -> A
W -> D
```

VisDA-C 正式 transfer：

```text
Synthetic / Train -> Real / Validation
```

VisDA-C 的 primary metric 必须始终按固定 12 类计算：

$$
\operatorname{mAcc}_{12}
=
\frac{1}{12}
\sum_{c=1}^{12}
\operatorname{Acc}_c.
$$

禁止把 VisDA-C 的 overall sample accuracy 当作 primary PU/FO metric；overall accuracy 只能作为 secondary diagnostic/reporting metric。

## 5. NCTTA objective：每个 model state 都必须重新计算

当前 F/B/C：

$$
f_\Theta(x)=\mathrm{netF}_\Theta(x),
$$

$$
h_\Theta(x)=\mathrm{netB}_\Theta(f_\Theta(x))\in\mathbb R^{256},
$$

$$
z_\Theta(x)=\mathrm{netC}(h_\Theta(x)).
$$

Classifier reference 固定为：

$$
W_C=\mathrm{netC.fc.weight.detach()}.
$$

`netC` 永远 frozen，但 feature $h_\Theta$ 与 logits $z_\Theta$ 随 candidate model 改变。

### 5.1 官方 NCTTA current-state construction

对任意 model state $\Theta$ 与当前 outer batch $B_t$，必须重新计算：

```text
post-netB 256-d features
logits / probabilities
top-k classes
feature-classifier cosine geometry
q_dist
q_prob
hybrid q
NC loss
entropy
entropy filter
entropy coefficient
predicted-class FCA-distance coefficient
weighted selected-sample loss
```

正式 objective 可记为：

$$
\mathcal L_t^{\mathrm{NCTTA}}(\Theta)
=
\operatorname{Mean}_{i:H_i(\Theta)<\gamma_{\mathrm{ent}}}
\left[
\lambda_i(\Theta)
\left(
H_i(\Theta)
+
s\,L_{\mathrm{NC},i}(\Theta)
\right)
\right].
$$

其中：

$$
\lambda_i(\Theta)
=
r_{\mathrm{ent}}
\exp\left[
-\left(H_i(\Theta)-m_{\mathrm{ent}}\right)
\right]
+
\frac{\nu_{\mathrm{NCTTA}}}
{1+\eta_{\mathrm{NCTTA}}d_i(\Theta)}.
$$

### 5.2 最关键的 NCTTA-LBI 规则

与 IST 不同，NCTTA **不存在 LBI 前先冻结的 hard/soft target task**。

Stage-1 第 $k$ 次 candidate：

$$
\Theta^{(k)}
=
\Theta_t+\Theta_\Delta^k.
$$

必须用这个 candidate state 重新 forward，并重新构造完整 NCTTA objective：

$$
g^k
=
\nabla_{\Theta_\Delta}
\mathcal L_t^{\mathrm{NCTTA}}
\left(
\Theta_t+\Theta_\Delta^k
\right).
$$

禁止：

```text
outer batch 开头算一次 source/current feature
-> 固定 top-k
-> 固定 q_dist/q_prob/q
-> 固定 selected samples
-> 在整个 Stage 1/2 复用
```

这会把 NCTTA 改成新的 source-anchor/frozen-target 方法。

### 5.3 Gradient path 必须 faithful

当前 audited NCTTA port 中：

```text
classifier weight is detached
entropy/FCA reweight coefficients are detached
hybrid q 本身按官方 objective path 参与当前-state loss construction
```

不得为了“稳定”额外 detach feature、logits、`q_dist`、`q_prob` 或 hybrid `q`，除非官方 audited objective 已如此定义并经 protocol revision 明确修改。

### 5.4 Empty entropy-filter case

如果某个 candidate state 出现：

```text
selected_count == 0
```

或 loss / gradient 非有限：

```text
NaN / Inf
```

必须 fail loudly 并记录 candidate diagnostics。禁止偷偷：

```text
fallback to all samples
zero loss
reuse previous target/filter
skip LBI iteration as success
```

是否需要新的 formal safe handling，只能在后续 protocol revision 中预注册。

## 6. NCTTA 与 LBI 的状态边界

NCTTA controlled branch没有：

```text
memory
teacher
EMA
momentum encoder
multi-view expansion
PLCA
host-specific persistent auxiliary state
```

每个 outer batch 的 scientific adaptation unit 就是当前 incoming batch $B_t$。

Persistent：

```text
adapted model theta_t
non-LBI host optimizer trajectory
```

LBI local states：

```text
theta_delta
Z
Gamma
Stage-2 local optimizer
```

每个 valid outer batch 全部 reset，不跨 batch。

## 7. LBI restart 粒度

正式冻结：

> **每个 valid incoming outer batch 做一次 support discovery。**

每个 batch 开始：

$$
\Theta_\Delta^0=0,\qquad
Z^0=0,\qquad
\Gamma^0=0.
$$

Persistent base model：

$$
\Theta_t
$$

跨 batch 累积。

不存在 IST 那种 inner mini-batch / multi-view scientific-unit ambiguity。

## 8. Corrected Stage-1 LBI dynamics

对当前 candidate：

$$
g^k
=
\nabla_{\Theta_\Delta}
\mathcal L_t^{\mathrm{NCTTA}}
\left(
\Theta_t+\Theta_\Delta^k
\right).
$$

Coupling：

$$
c^k
=
\frac{\Theta_\Delta^k-\Gamma^k}{\nu_{\mathrm{LBI}}}.
$$

Corrected update：

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

最关键语义：

> $Z^{k+1}$ 必须使用 old-state $(\Theta_\Delta^k,\Gamma^k)$，禁止使用 $\Theta_\Delta^{k+1}$ 参与当前 $Z$ update。

更新 $Z$ 后再按 FC scalar 或 Conv group track 的 prox 得到 $\Gamma^{k+1}$。

### 8.1 `nu` 命名冲突必须彻底消除

NCTTA objective 和 LBI 都有名为 `nu` 的参数，但它们完全不是同一个量：

```text
nctta.nu = NCTTA sample FCA-distance reweight coefficient
lbi.nu   = Split-LBI coupling parameter
```

代码、config、CLI、artifact、experiment identity、paper table 均必须 namespaced。禁止出现无法判定属于哪一层的裸字段：

```text
nu
```

## 9. FC scalar track

### 9.1 Candidate universe

只允许：

```text
netB.bottleneck.weight
netB.bottleneck.bias
```

所有 BN state frozen，`netC` frozen。

Candidate count：

$$
N_{\mathrm{FC}}
=
256\times2048+256
=
524544.
$$

### 9.2 Scalar prox / support

$$
\Gamma_j^{k+1}
=
\kappa
\operatorname{sign}(Z_j^{k+1})
\left[
|Z_j^{k+1}|-1
\right]_+.
$$

Frozen threshold：

$$
\tau=10^{-4}.
$$

Support：

$$
M_j^k
=
\mathbf 1
\left[
|\Gamma_j^k|\ge\tau
\right].
$$

### 9.3 Budgets

$$
\rho\in\{0.0005,\ 0.001,\ 0.002\},
$$

$$
K=\lfloor \rho N_{\mathrm{FC}}\rfloor.
$$

| $\rho$ | $K$ |
|---:|---:|
| 0.0005 | 262 |
| 0.001 | 524 |
| 0.002 | 1049 |

Budget 对 weight+bias 做 **global scalar counting**。禁止 per-tensor quota。

## 10. Conv out-channel track

### 10.1 Candidate universe

只允许 formal full `netF.layer4` 9 Conv weights：

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

所有 BN frozen，`netC` frozen。

$$
N_{\mathrm{conv}}=12,845,056.
$$

### 10.2 Out-channel grouping

对 Conv tensor $W^{(l)}$：

$$
G_{l,o}=W^{(l)}[o,:,:,:].
$$

Global group pool：

$$
|\mathcal G_{\mathrm{out}}|=9216.
$$

### 10.3 Group prox / support

$$
\Gamma_g^{k+1}
=
\kappa
\left(
1-\frac{1}{\|Z_g^{k+1}\|_2}
\right)_+
Z_g^{k+1}.
$$

Zero norm 做数值安全处理，但不能改变 support semantics。

$$
M_g^k
=
\mathbf1
\left[
\|\Gamma_g^k\|_2\ge10^{-4}
\right].
$$

### 10.4 Group budgets

$$
\rho_G\in\{0.0005,\ 0.001,\ 0.002\},
$$

$$
K_G
=
\left\lfloor
\rho_G|\mathcal G_{\mathrm{out}}|
\right\rfloor.
$$

| $\rho_G$ | $K_G$ |
|---:|---:|
| 0.0005 | 4 |
| 0.001 | 9 |
| 0.002 | 18 |

必须同时记录：

```text
selected_group_count
realized_group_ratio
selected_scalar_count
realized_scalar_ratio
```

group ratio 不得被描述成 scalar sparsity。

## 11. Strict budget 与 rollback

设：

```text
FC   : K_star = K
Conv : K_star = K_G
```

Stage-1 每次更新后按 frozen threshold 统计 support：

```text
n < K_star:
    保存当前 state 为 latest feasible state
    继续

n == K_star:
    接受当前 state
    Stage 1 结束

n > K_star:
    当前 overshoot state 非法
    rollback 到 latest feasible state
    Stage 1 结束
```

禁止：

```text
budget slack
K tolerance
ceil-based K
LBI top-K trimming
per-layer quota
per-tensor quota
minimum-one-per-layer repair
```

最终：

$$
|M^\star|<K_\star
$$

是合法结果。

## 12. Stage-1 safety cap

Frozen：

```text
stage1_max_steps = 3000
```

它只是 safety cap，不是 tuning dimension。大量 cap hit 说明当前 LBI tuple 不适合，而不是允许偷偷增大 cap。

## 13. Stage-2 masked refinement

### 13.1 Masked-delta initialization

Stage-1 final：

$$
(\Theta_\Delta^\star,\Gamma^\star,M^\star).
$$

Stage-2 初始 state：

$$
\Theta_{t,2}^0
=
\Theta_t
+
M^\star\odot\Theta_\Delta^\star.
$$

Off-mask Stage-1 delta 禁止泄漏。

Conv group mask broadcast 到对应 scalar coordinates。

### 13.2 Stage 2 恰好一个 optimizer step

Frozen：

```text
stage2_steps = 1
```

Stage-2 使用**同一个 NCTTA objective function**，但必须在 Stage-2 当前 model state 上重新 forward / 重新构造：

```text
features
logits
top-k
q_dist/q_prob/q
entropy filter
weights
loss
```

因此它不是 IST 那种“固定 targets 的同一 objective dataset”，而是：

$$
\nabla_\Theta
\mathcal L_t^{\mathrm{NCTTA}}
(\Theta_{t,2}^0).
$$

### 13.3 Stage-2 optimizer

Frozen：

```text
optimizer = SGD
momentum = 0.9
weight_decay = 0.001
nesterov = true
LR schedule = none
stage2_lr = fixed
```

`stage2_lr` 是 LBI tunable parameter。

每个 outer batch 使用 fresh local Stage-2 optimizer；optimizer state 不跨 outer batches。

### 13.4 Strict off-mask preservation

Stage 2：

```text
off-mask gradients masked
off-mask parameter values restored/frozen
off-mask optimizer state 不得产生 hidden update
non-candidate parameters unchanged
all controlled BN parameters/buffers frozen
netC frozen
```

## 14. Persistent writeback

NCTTA controlled Dense/Random/Magnitude/Saliency 没有 native EMA。它们的 persistent model state就是 host optimizer step 后的 model。

NCTTA-LBI 的唯一 persistent writeback：

$$
\Theta_{t+1}
=
(1-\omega)\Theta_t
+
\omega\widetilde\Theta_t.
$$

其中 $\widetilde\Theta_t$ 是 Stage-2 refined state。

正式规则：

```text
NCTTA Random/Magnitude/Saliency:
    masked host SGD update
    no omega
    no EMA

NCTTA-LBI:
    no host SGD persistent step
    LBI Stage2 local masked step
    exactly one omega writeback
```

禁止：

```text
host optimizer step -> LBI omega
LBI omega -> second host optimizer step
任何 double writeback
```

Off-mask Stage-2 state等于 persistent base，因此 omega writeback 后 off-mask coordinates 仍必须 bit-exact 不变。

## 15. FC matched sparse baselines

所有 FC sparse selectors 使用：

```text
same global FC candidate pool
same exact integer K
same NCTTA objective
same controlled BN freeze
```

### 15.1 `nctta_fc_random`

每个 budget 必须真实执行 3 个 deterministic independent child trajectories。

Random mask seeds：

```text
202600
202601
202602
```

每个 child：

```text
fresh source F/B/C
fresh optimizer state
same formal target stream seed=2026
uniform global exact-K scalar mask
mask fixed for whole target stream
```

Top-level Random artifact 聚合 child PU/FO mean/std。`num_random_masks=3` 不能只存在 metadata；必须真实执行 3 条完整 trajectory。

### 15.2 `nctta_fc_magnitude`

在 source checkpoint：

$$
s_j=|\theta_j^{\mathrm{src}}|.
$$

选择 global top-$K$。Mask 整条 target stream 固定，禁止按 adapted model 重算 magnitude。

### 15.3 `nctta_fc_saliency`

每个 valid outer batch，在任何 adaptation update 前，于当前 persistent pre-adaptation model $\Theta_t$ 计算：

$$
g_j^{(t)}
=
\nabla_{\theta_j}
\mathcal L_t^{\mathrm{NCTTA}}(\Theta_t),
$$

$$
s_j^{(t)}
=
|\theta_j^{(t)}g_j^{(t)}|.
$$

选择 global exact top-$K$。

关键规则：

> support 每个 outer batch 只选一次，当前 batch 的 masked host optimizer step 内固定；下一 outer batch 才允许重选。

如果实现复用同一次 exact gradient 同时完成 saliency selection 与该 batch 的 masked SGD step，必须 contract 证明在 model/RNG/BN/objective state 未变化时与重新 forward/backward 数学等价；否则重新计算 objective gradient。无论实现方式如何，scientific semantics 都是“一次 support selection + 一次 masked host update”。

## 16. Conv matched sparse baselines

所有 Conv sparse selectors 使用：

```text
same 9216 out-channel group pool
same exact K_G
same NCTTA objective
same controlled BN freeze
```

### 16.1 `nctta_conv_out_random`

每个 budget 真实执行 3 个 independent child trajectories，每个 child uniform sample exact $K_G$ groups，mask 整条 stream 固定。

### 16.2 `nctta_conv_out_magnitude`

Source checkpoint：

$$
s_g
=
\|W_g^{\mathrm{src}}\|_2.
$$

选择 global top-$K_G$，mask 整条 stream 固定。

### 16.3 `nctta_conv_out_saliency`

每个 valid outer batch，在当前 pre-adaptation model：

$$
s_g^{(t)}
=
\left\|
W_g^{(t)}
\odot
\nabla_{W_g}
\mathcal L_t^{\mathrm{NCTTA}}(\Theta_t)
\right\|_2.
$$

选择 global exact top-$K_G$。Mask 当前 outer batch 固定，下一 batch 才重选。

## 17. Random/Magnitude/Saliency 的 optimizer semantics

非-LBI sparse variants 继承对应 controlled dense NCTTA host training substrate：

```text
Office base LR = 0.01
VisDA base LR = 0.001
netF LR multiplier = 0.1
netB LR multiplier = 1.0
SGD momentum = 0.9
weight_decay = 0.001
nesterov = true
outer-batch polynomial schedule
exactly one optimizer step per valid outer batch
```

Sparse protection：

```text
only selected coordinates/groups may change
off-mask gradients masked
off-mask values restored/frozen
off-mask momentum/state cleared
continuously selected coordinates may retain legal host optimizer state
controlled BN parameters/buffers frozen
netC frozen
```

Random/Magnitude/Saliency 不使用 LBI Stage-2 optimizer，也不使用 omega。

## 18. NCTTA-LBI 每个 outer batch 的精确执行顺序

对每个 valid non-singleton $B_t$：

```text
1. 从 seed-2026 fixed stream 取得 current B_t
2. 保存 persistent pre-batch base model theta_t
3. theta_delta / Z / Gamma 清零
4. Stage 1 iteration k:
     a. 构造 candidate theta_t + theta_delta^k
     b. candidate state 做 netF -> netB -> netC forward
     c. 读取 effective frozen netC.fc.weight
     d. 重新构造 top-k / q_dist / q_prob / hybrid q
     e. 重新计算 entropy / filter / detached reweights / NCTTA loss
     f. 对 candidate coordinates 求 gradient
     g. corrected LBI update
     h. threshold support
     i. strict integer budget / rollback
5. 得到 final support M*
6. Stage-2 model = theta_t + M* ⊙ theta_delta*
7. 在该 Stage-2 state 上重新完整计算一次 NCTTA objective
8. 恰好 1 个 masked Stage-2 SGD step
9. 恰好 1 个 LBI omega persistent writeback
10. PU on current batch，read-only
11. 丢弃 theta_delta / Z / Gamma / local Stage-2 optimizer
12. 进入下一 outer batch
```

Stream 结束：

```text
freeze final persistent model
-> full-target FO
-> read-only
```

## 19. 非-LBI sparse 每个 outer batch 的顺序

Random / Magnitude / Saliency：

```text
1. current B_t
2. 获取 support：
     Random    = stream-fixed mask
     Magnitude = source-fixed mask
     Saliency  = current-state NCTTA gradient选一次
3. 当前 model state 计算 NCTTA objective
4. under fixed mask 恰好 1 个 host SGD optimizer step
5. PU read-only
6. next batch
```

不存在：

```text
LBI local state
Stage-2 local optimizer
omega writeback
native NCTTA norm-only state
```

## 20. Singleton rule

继承 baseline：

```text
actual outer batch size == 1
-> 在任何 NCTTA/sparse/LBI state transition 前 skip
```

因此 singleton 不触发：

```text
NCTTA objective
selector
host optimizer
LBI Stage 1
Stage 2
omega
PU
```

FO 仍 full-target。

## 21. LBI frozen constants

| Item | Frozen value |
|---|---:|
| support threshold | `1e-4` |
| Stage-1 max steps | `3000` |
| Stage-2 steps | `1` |
| Stage-2 optimizer | SGD |
| Stage-2 momentum | `0.9` |
| Stage-2 weight decay | `0.001` |
| Stage-2 Nesterov | `true` |
| Stage-2 LR schedule | none |
| strict rollback | enabled |
| top-K LBI repair | forbidden |
| masked-delta Stage-2 init | enabled |
| off-mask exact preservation | enabled |
| LBI local restart | every valid outer batch |
| host EMA inside LBI | none / not applicable |

## 22. LBI tunable parameters与 tuning granularity

只允许调：

```text
lbi.alpha
lbi.kappa
lbi.nu
lbi.omega
lbi.stage2_lr
```

Tuning granularity：

$$
(
\mathrm{dataset},
\mathrm{parameter\ track},
\rho_\star
).
$$

其中：

```text
parameter track = FC scalar or Conv out-channel
```

Office 同一 track/budget 的 tuple 必须由 6 transfers 共用。禁止 per-transfer、per-seed、per-batch tuning。

最终需要 12 组 NCTTA-LBI tuples：

### FC scalar

| Dataset | Budget | Tuple |
|---|---:|---|
| Office | .0005 | TBD |
| Office | .001 | TBD |
| Office | .002 | TBD |
| VisDA | .0005 | TBD |
| VisDA | .001 | TBD |
| VisDA | .002 | TBD |

### Conv out-channel

| Dataset | Budget | Tuple |
|---|---:|---|
| Office | .0005 | TBD |
| Office | .001 | TBD |
| Office | .002 | TBD |
| VisDA | .0005 | TBD |
| VisDA | .001 | TBD |
| VisDA | .002 | TBD |

## 23. NCTTA host-objective parameters：formal 前必须先冻结

当前 correctness/smoke 使用的 NCTTA fields：

```text
nctta.thre_ent
nctta.margin_ent
nctta.reweight_ent
nctta.nu
nctta.eta
nctta.scale
nctta.top_k
nctta.mix_prob_weight
```

其中 official repo 的 `0.4*log(1000)` threshold 带明显 ImageNet-1000 语境；因此它们当前只是 correctness defaults，不是 Office/VisDA formal values。

正式 sparse/LBI search 前必须先有独立 preregistered NCTTA objective-parameter resolution，至少满足：

```text
1. 不看 LBI final results 反向调 host objective
2. controlled Dense/Random/Magnitude/Saliency/LBI 在同一 dataset 使用同一 NCTTA objective tuple
3. 不允许按 track/budget/transfer 单独改 NCTTA objective
4. nctta.nu 与 lbi.nu 在 config/artifact/identity 完全分离
```

在 dataset-specific NCTTA objective tuple 未冻结前：

```text
formal Dense = blocked
formal sparse baselines = blocked
formal LBI = blocked
```

但 correctness contracts 与 short smoke 可继续使用明确标记的 debug/default objective values。

## 24. 本 protocol 故意不冻结的 search items

后续单独写 NCTTA search protocol，才冻结：

```text
dataset-specific NCTTA objective tuple
LBI alpha/kappa/nu grid
omega grid
stage2_lr grid
Stage-1 reachability screening
support-utilization eligibility rule
accuracy-selection rule
boundary-expansion rule
search budget
hardware allocation
```

不得直接把 SHOT/IST final winner 当作 NCTTA formal tuple；可以作为 preregistered search anchor，但必须独立验证。

## 25. Support utilization 与 scientific validity

FC 至少记录：

```text
selected_count
budget_K
realized_ratio
utilization
Stage-1 steps
Stage-1 cap hit
overshoot/rollback status
NCTTA selected-sample count trajectory
```

Conv 至少记录：

```text
selected_group_count
budget_K_G
realized_group_ratio
selected_scalar_count
realized_scalar_ratio
group utilization
Stage-1 steps
Stage-1 cap hit
overshoot/rollback status
NCTTA selected-sample count trajectory
```

以下任一情况 scientifically invalid：

```text
NaN / Inf
NCTTA entropy filter zero-selection without explicit failure
runtime correctness failure
selected support > strict integer budget
off-mask parameter changes
controlled BN changes
netC changes
required support trace/artifact missing
Stage-1 cap treated as silent success
future-target leakage
target-label adaptation leakage
double optimizer/writeback path
protocol/identity mismatch
```

Exact utilization gate 留给 search protocol。

## 26. PU / FO 与 state protection

PU：

```text
non-LBI sparse:
    masked host SGD update -> PU

NCTTA-LBI:
    omega writeback -> PU
```

PU 完全 read-only，不得修改：

```text
parameters
BN buffers
optimizer state
selector state
LBI local state
writeback anchor
```

FO：

```text
freeze final persistent model
-> full target evaluation
-> read-only
```

Primary reporting 必须按 dataset 固定：

```text
Office PU/FO primary = sample-level overall accuracy
VisDA  PU/FO primary = fixed-12-class mAcc
```

VisDA 的 overall accuracy / class-wise accuracy / worst-class / class-wise std 只作为 secondary diagnostics，不得替代 primary fixed-12-class mAcc。

Controlled FC/Conv family 的 BN 保持 frozen/source-stat behavior。`nctta_native_norm` 的 batch-stat FO 属于 parent baseline native reference，不进入本 sparse protocol。

## 27. Target-label policy

Target labels只允许 evaluation/reporting。

禁止进入：

```text
NCTTA objective
top-k / hybrid target
entropy filter
Random support
Magnitude support
Saliency support
LBI Stage 1
LBI Stage 2
omega writeback
```

若未来 tuning 使用 labeled validation，必须先在 search protocol 中明确预注册。

## 28. Formal sparse experiment matrix

每个 track、每个 transfer、每个 budget：

```text
Random
Magnitude
Saliency
LBI
```

Budgets：

$$
\rho_\star\in\{0.0005,\ 0.001,\ 0.002\}.
$$

即每个 track：

```text
4 variants x 3 budgets
= 12 budgeted formal identities per transfer
```

Random 的 3 masks 是同一 top-level Random identity 下的 child executions，不是额外 formal seeds。

Formal launch 前必须同时满足：

```text
NCTTA dataset-specific objective tuple frozen
sparse/LBI implementation revision frozen
search protocol frozen
对应 LBI tuned tuple frozen
```

## 29. Scientific identity 与 provenance

至少记录：

```text
parent NCTTA baseline protocol revision
NCTTA-LBI protocol revision
NCTTA sparse/LBI implementation revision
official NCTTA audited commit
source checkpoint revision
source checkpoint paths/SHA256
dataset / transfer / backbone
formal seed / outer batch size
target stream identity/hash
primary_metric_name
primary_metric_value
VisDA fixed class count = 12, if applicable
candidate track / parameter names
grouping mode
candidate scalar count
group count if Conv
variant
requested sparse ratio
integer K or K_G
Random child index/seed
selector definition

NCTTA objective namespace:
  nctta.thre_ent
  nctta.margin_ent
  nctta.reweight_ent
  nctta.nu
  nctta.eta
  nctta.scale
  nctta.top_k
  nctta.mix_prob_weight

LBI namespace:
  lbi.alpha
  lbi.kappa
  lbi.nu
  lbi.omega
  lbi.stage2_lr

support threshold
Stage-1 cap
Stage-2 steps
persistent writeback mode
```

NCTTA-LBI：

```text
persistent_writeback = lbi_omega_only
host_optimizer_persistent_step = false
```

Random/Magnitude/Saliency：

```text
persistent_writeback = masked_host_optimizer
lbi_omega = null
```

## 30. Implementation contracts

正式 sparse/LBI implementation 至少验证：

1. Same source F/B/C、stream、singleton、PU/FO 与 parent baseline。
2. `nctta_native_norm` 不进入 sparse dispatch。
3. FC candidate count=524544，budgets=262/524/1049。
4. Conv out-channel group pool=9216，budgets=4/9/18。
5. Conv realized scalar support被记录。
6. Random top-level 每个 budget真实执行3个 fresh child trajectories；不是只写 metadata。
7. Magnitude support只从 source checkpoint算一次并整条 stream固定。
8. Saliency 每 valid outer batch只选一次 support。
9. Saliency score使用完整 current-state NCTTA objective。
10. NCTTA-LBI 每 valid outer batch只 restart/support-discover一次。
11. Stage-1 每个 candidate iteration重新 forward 并重新构造 top-k / hybrid target / filter / weights。
12. 禁止固定 source feature或 batch-start NCTTA target。
13. Audited NCTTA objective gradient path未被额外 detach。
14. `nctta.nu` 与 `lbi.nu` config/identity/artifact严格分离。
15. corrected old-state $Z$ update保持不变。
16. strict integer rollback保持不变。
17. Stage-2 init不含 off-mask Stage-1 delta。
18. Stage-2 恰好1个 masked optimizer step。
19. Stage-2 objective在 Stage-2 current state重新计算。
20. Off-mask parameters、controlled BN parameters/buffers、netC不变。
21. LBI 每 processed batch omega writeback恰好一次。
22. LBI 不执行额外 host optimizer persistent step。
23. 非-LBI sparse 每 processed batch恰好一次 masked host optimizer step，不执行 omega。
24. PU / FO read-only。
25. Singleton 在 objective/selector/LBI/optimizer/PU 前 skip。
26. Target labels不进入 adaptation。
27. Scientific identity能区分 track/budget/selector/NCTTA tuple/LBI tuple/writeback mode。
28. Existing SHOT/IST/NCTTA dense/native regression全部 PASS。

Contract 要检查**次数和状态转移**，不能只检查 tensor shape / loss finite。

## 31. Independent code review

Tests PASS 后必须人工检查：

```text
variant dispatch
candidate construction
Random child execution multiplicity
source reset between Random children
magnitude source snapshot
saliency support timing
NCTTA current-state objective recomputation
classifier weight extraction
nctta.nu vs lbi.nu namespaces
LBI restart
Stage1 candidate model
strict rollback
Stage2 init
Stage2 step count
off-mask restore
omega writeback
BN/netC protection
singleton
PU/FO
artifact aggregation
experiment identity fallback
```

不得只信“all contracts PASS”。

## 32. Short smoke

Implementation + contracts + review完成后，只做 correctness smoke：

```text
Office D->A
seed=2026
BS64
5 valid outer batches
one debug budget, preferably rho=0.001
all 8 sparse variants
Random = actual 3 child executions
```

LBI 使用明确标记 `debug_only` 的 existing test/smoke tuple；不得根据 smoke accuracy 或 utilization 当场改算法/参数。

Smoke 只验证：

```text
real data/GPU path
real artifacts
scope preservation
objective recomputation counts
support selection counts
strict budget/rollback
Stage2 exactly once
omega exactly once
Random children exactly 3
BN/netC/off-mask protection
PU/FO read-only
```

完成后停止，不搜参。

## 33. 明确禁止的 semantics

禁止：

```text
NCTTA native norm-only scope混入 FC/Conv sparse family
source initial feature anchoring
batch-start feature/target freezing
fixed top-k across LBI Stage1
fixed hybrid q across LBI Stage1
fixed selected entropy subset across LBI Stage1
extra detach改变官方 objective gradient
target-label-driven support
future-target access
per-tensor FC budgets
filter_connection Conv mainline
old-state Z bug
budget slack
LBI top-K repair
dense Stage1 delta leakage
off-mask hidden update
Stage2 > 1 step
host optimizer + omega double writeback
Random metadata=3但只执行1 child
nctta.nu / lbi.nu字段混淆
accuracy-driven smoke tuning
```

## 34. Freeze boundary

### 已冻结

```text
NCTTA sparse/LBI implementation revision = nctta_otta_sparse_lbi_20260907_v2
NCTTA current-state objective for every candidate state
post-netB 256-d feature
effective frozen netC.fc.weight
dynamic top-k/q/filter/reweight recomputation
no frozen NCTTA pseudo-target across LBI iterations
outer-batch LBI restart
corrected old-state dynamics
FC scalar candidate/support/budgets
Conv full-layer4 out-channel grouping/support/budgets
strict rollback
tau=1e-4
Stage1 cap=3000
masked-delta Stage2 initialization
Stage2 exactly one current-state NCTTA masked SGD step
off-mask exact preservation
omega-only persistent writeback for LBI
masked host SGD persistent update for non-LBI sparse
Random/Magnitude/Saliency selector semantics
3 real Random child trajectories
singleton compatibility
PU/FO read-only
filter_connection excluded
LBI tunables limited to alpha/kappa/nu/omega/stage2_lr
NCTTA and LBI nu namespaced separately
```

### 尚未冻结

```text
dataset-specific formal NCTTA objective tuple
LBI search grids
utilization/eligibility gates
selection rule
boundary expansion
12 final tuned LBI tuples
formal efficiency execution package
```

任何已冻结科学规则改变都必须升级 protocol revision。

## 35. 一页式冻结摘要

| Category | Frozen rule |
|---|---|
| Protocol | `OTTA_NCTTA_LBI_PROTOCOL_20260906_v1` |
| Parent | `OTTA_NCTTA_BASELINE_PROTOCOL_20260906_v2` |
| Sparse family | controlled FC / Conv only |
| Native norm | reference only，不进 sparse |
| Office primary metric | sample-level overall accuracy |
| VisDA primary metric | fixed-12-class mAcc |
| LBI unit | 每 valid outer batch 一次 |
| NCTTA feature | current post-netB 256-d |
| Classifier reference | effective frozen `netC.fc.weight` |
| NCTTA target | 每 candidate state 动态重算 |
| Fixed pseudo-target in LBI | forbidden |
| FC candidate | bottleneck weight+bias |
| FC scalars | 524544 |
| FC budgets | 262 / 524 / 1049 |
| Conv candidate | layer4 9 Conv weights |
| Conv grouping | out-channel only |
| Conv groups | 9216 |
| Conv budgets | 4 / 9 / 18 |
| Support threshold | `1e-4` |
| Strict rollback | yes |
| Top-K LBI repair | no |
| Stage-1 cap | 3000 |
| Stage-2 init | base + masked delta |
| Stage-2 | exactly 1 masked SGD step |
| Stage-2 objective | current-state NCTTA recomputation |
| LBI writeback | omega only |
| Non-LBI writeback | masked host SGD |
| Random | exact-K/K_G, 3 real child trajectories |
| Magnitude | source-static global top-K |
| Saliency | current-state dynamic global top-K, once/batch |
| BN | controlled FC/Conv all frozen |
| `netC` | frozen |
| `nu` | `nctta.nu` and `lbi.nu` strictly separate |
| Formal sparse | host objective/search/tuned tuples冻结前 blocked |

**End of protocol.**
