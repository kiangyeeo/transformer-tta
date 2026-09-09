# OTTA_IST_LBI_PROTOCOL_20260906_v1

**状态：** 科学语义已冻结；implementation revision 已冻结为 `ist_otta_sparse_lbi_20260906_v2`；在 search protocol 与 tuned tuples 冻结前，formal LBI launch 保持 blocked  
**Protocol revision：** `OTTA_IST_LBI_PROTOCOL_20260906_v1`  
**Parent IST baseline protocol：** `OTTA_IST_BASELINE_PROTOCOL_20260906_v1`  
**Parent refined FC protocol：** `OTTA_FC_LBI_PROTOCOL_20260817_v1`  
**Parent refined Conv protocol：** `OTTA_CONV_LBI_PROTOCOL_20260826_v1`  
**Current IST baseline implementation：** `ist_otta_p1_baseline_20260906_v4`  
**IST-LBI implementation revision：** `ist_otta_sparse_lbi_20260906_v2`  
**Source checkpoint revision：** `nips2026_shot_otta_uda_source_v1`  
**日期：** 2026-09-06  
**范围：** IST + FC scalar sparse adaptation + Conv out-channel structured sparse adaptation

> 本文件冻结的是 **IST × sparse/LBI 的科学语义**。  
> exact search grid、utilization gate、selection rule 与最终 tuned tuples 不在此文件中拍脑袋决定，而是后续单独写 preregistered `IST_LBI_SEARCH_PROTOCOL`。

---

## 1. 科学问题

本研究要验证同一个 refined Sparse Delta Learning / LBI optimizer 是否同时具备：

1. **objective generality**：SHOT-OTTA 与 IST-OTTA；
2. **parameter-structure generality**：FC scalar 与 Conv out-channel group。

目标矩阵：

| Adaptation objective | FC scalar | Conv out-channel |
|---|---|---|
| SHOT-OTTA | 已完成 | 已完成 |
| IST-OTTA | 本 protocol | 本 protocol |

核心原则：

> 从 SHOT 换成 IST 时，只替换 adaptation objective / signal-generation mechanism，不重新发明一套 LBI 算法。

抽象形式：

$$
g^k
=
\nabla_{\Theta_\Delta}
\mathcal L_{\mathrm{adp}}
\left(
\mathcal D_t;
\Theta_t+\Theta_\Delta^k
\right),
$$

本 protocol 中：

$$
\mathcal L_{\mathrm{adp}}
=
\mathcal L_{\mathrm{IST}}.
$$

---

## 2. Source-of-truth 优先级

发生冲突时：

1. `OTTA_IST_LBI_PROTOCOL_20260906_v1`
2. `OTTA_IST_BASELINE_PROTOCOL_20260906_v1`
3. 当前 corrected refined LBI engine + frozen refined FC/Conv protocol semantics
4. 官方 `JingInAI/IST4TTA`
5. P0/P1 历史 IST contracts / audits
6. 旧 `IST-OTTA`
7. 旧 `IST-OTTA-LBI`

旧 `IST-OTTA-LBI` 只能用于定位历史 bug，不得作为新实现模板。

---

## 3. 正式 variant 范围

### 3.1 FC controlled family

```text
ist_fc_module_dense
ist_fc_random
ist_fc_magnitude
ist_fc_saliency
ist_fc_lbi
```

### 3.2 Conv controlled family

主线只做：

```text
ist_conv_module_dense
ist_conv_out_random
ist_conv_out_magnitude
ist_conv_out_saliency
ist_conv_out_lbi
```

Conv grouping 只保留：

```text
out_channel
```

明确禁止重新打开：

```text
filter_connection
```

`module_dense` 已由 parent baseline protocol 定义，不按 sparse budget 重复。

---

## 4. 从 IST baseline protocol 继承的规则

所有 IST sparse / LBI variants 继承：

```text
exact same source F/B/C
source revision
R50 Office / R101 VisDA
seed-2026 outer stream
Office BS64 / VisDA BS256
workers=4
one target pass
drop_last=false
explicit actual-BS=1 singleton skip
raw-image cached PU reference views
raw-image 8-view IST adaptation views
PLCA settings
causal memory max_len=10000
hard CE + soft KL target construction
target-label boundary
PU / FO read-only semantics
metrics
artifact / provenance discipline
```

本文件没有明确 override 的，都继续沿用 baseline protocol。

---

## 5. Sparse adaptation 前的 IST signal construction

这是 IST-LBI integration 最重要的规则。

对每个有效 incoming outer batch：

$$
B_t,
$$

进入 batch 前 persistent model：

$$
\Theta_t,
$$

causal memory：

$$
\mathcal M_{t-1}.
$$

### 5.1 当前 adaptation views 只 materialize 一次

每张 raw sample：

```text
1 cached PU reference view
8 IST adaptation views
```

完整 adaptation-view set：

$$
X_t
=
\{x_1,\ldots,x_{N_t}\},
$$

其中：

$$
N_t=8|B_t|.
$$

### 5.2 Pre-adaptation outputs 只计算一次

只用：

$$
\Theta_t
$$

在 eval behavior 下计算：

$$
(F_t,Q_t).
$$

这里：

- $F_t$：current features；
- $Q_t$：pre-correction soft targets。

### 5.3 PLCA 恰好一次

PLCA 只使用：

$$
(F_t,Q_t,\mathcal M_{t-1}).
$$

输出：

$$
\hat Y_t.
$$

禁止在 LBI Stage 1 内反复跑 PLCA。

### 5.4 Memory commit 恰好一次

当前 features / corrected pseudo-label distributions 只 commit 一次：

$$
\mathcal M_{t-1}
\rightarrow
\mathcal M_t.
$$

禁止在 LBI Stage 1/2 中再次 commit。

### 5.5 当前 outer batch 的 IST adaptation task 固定

定义：

$$
\mathcal D_t^{\mathrm{IST}}
=
\left\{
(x_i,\hat y_i,q_i)
\right\}_{i=1}^{N_t}.
$$

在当前 outer batch 的 selector / LBI Stage 1 / LBI Stage 2 中：

```text
views fixed
hard targets fixed
soft targets fixed
PLCA not rerun
memory not recommitted
future target inaccessible
```

model prediction 会随参数更新重新计算，但 data / targets 不变。

---

## 6. Full-outer-batch IST objective

对任意 model state：

$$
\Theta,
$$

定义当前 outer batch 的完整 IST objective：

$$
\mathcal L_t^{\mathrm{IST}}(\Theta)
=
\frac{1}{N_t}
\sum_{i=1}^{N_t}
\left[
\operatorname{CE}
\left(
p_\Theta(x_i),\hat y_i
\right)
+
D_{\mathrm{KL}}
\left(
q_i
\;\|\;
p_\Theta(x_i)
\right)
\right].
$$

正式规则：

> **LBI Stage 1 和 Stage 2 的数学 optimization unit 都是整个 fixed outer-batch IST objective，而不是某个 IST inner mini-batch。**

如果所有 views 同时放不进显存，可以做 exact gradient accumulation。

设 chunks：

$$
\mathcal C_r,
$$

大小：

$$
n_r.
$$

则：

$$
\nabla
\mathcal L_t^{\mathrm{IST}}
=
\sum_r
\frac{n_r}{N_t}
\nabla
\mathcal L_{\mathcal C_r}^{\mathrm{IST}}.
$$

实现要求：

```text
accumulation 前只 zero_grad 一次
遍历所有 chunks 累积
每个 chunk 的 mean-gradient 按 n_r / N_t 加权
chunk 之间禁止 optimizer step
chunk 之间禁止 LBI state update
完整 full-objective gradient 得到后，才做一次 LBI update / Stage-2 step
```

如果每个 inner mini-batch 单独做一次 LBI update，属于 protocol violation。

---

## 7. LBI restart 粒度

我们已经正式冻结：

> **每个 valid incoming outer batch 做一次 LBI support discovery。**

不是：

```text
每个 IST inner mini-batch 做一次
```

每个有效：

$$
B_t
$$

开始时：

$$
\Theta_\Delta^0=0,
$$

$$
Z^0=0,
$$

$$
\Gamma^0=0.
$$

Persistent model：

$$
\Theta_t
$$

跨 outer batches 累积。

以下 local states 不跨 batch：

```text
Theta_delta
Z
Gamma
```

FC scalar LBI 与 Conv out-channel Group-LBI 都采用同一 restart 规则。

---

## 8. Corrected Stage-1 LBI dynamics

定义：

$$
g^k
=
\nabla_{\Theta_\Delta}
\mathcal L_t^{\mathrm{IST}}
\left(
\Theta_t+\Theta_\Delta^k
\right),
$$

coupling：

$$
c^k
=
\frac{
\Theta_\Delta^k-\Gamma^k
}{
\nu
}.
$$

Corrected refined update：

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
Z^k
+
\alpha c^k.
$$

最关键的 frozen semantic：

> $Z^{k+1}$ 必须使用 old-state $(\Theta_\Delta^k,\Gamma^k)$。

禁止用：

$$
\Theta_\Delta^{k+1}
$$

参与当前 $Z$ update。

更新 $Z$ 后，再使用对应 track 的 prox 得到：

$$
\Gamma^{k+1}.
$$

---

## 9. FC scalar track

### 9.1 Candidate universe

只允许：

```text
netB.bottleneck.weight
netB.bottleneck.bias
```

所有 BN state frozen。

Candidate scalar count：

$$
N_{\mathrm{FC}}
=
256\times2048+256
=
524544.
$$

FC scalar LBI 可以视为 singleton-group Group-LBI。

### 9.2 FC scalar prox 与 support

Prox：

$$
\Gamma_j^{k+1}
=
\kappa
\operatorname{sign}
\left(
Z_j^{k+1}
\right)
\left[
\left|
Z_j^{k+1}
\right|
-1
\right]_+.
$$

Frozen support threshold：

$$
\tau=10^{-4}.
$$

Support：

$$
M_j^k
=
\mathbf 1
\left[
|\Gamma_j^k|
\ge\tau
\right].
$$

### 9.3 FC budgets

正式 ratios：

$$
\rho
\in
\{0.0005,\ 0.001,\ 0.002\}.
$$

Global integer budget：

$$
K
=
\left\lfloor
\rho N_{\mathrm{FC}}
\right\rfloor.
$$

因此：

| $\rho$ | $K$ |
|---:|---:|
| 0.0005 | 262 |
| 0.001 | 524 |
| 0.002 | 1049 |

Budget 对：

```text
bottleneck.weight + bottleneck.bias
```

做 global counting。

禁止：

```text
per-tensor budget
```

---

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

所有 BN state frozen。

Candidate scalar count：

$$
N_{\mathrm{conv}}
=
12,845,056.
$$

### 10.2 Out-channel groups

对 Conv tensor：

$$
W^{(l)},
$$

group 定义：

$$
G_{l,o}
=
W^{(l)}[o,:,:,:].
$$

Global out-channel group pool：

$$
|\mathcal G_{\mathrm{out}}|
=
9216.
$$

### 10.3 Group prox

对 group：

$$
g,
$$

$$
\Gamma_g^{k+1}
=
\kappa
\left(
1-
\frac{1}{
\|Z_g^{k+1}\|_2
}
\right)_+
Z_g^{k+1}.
$$

zero-norm 要数值安全处理，但不能改变数学 support 语义。

### 10.4 Group support

Frozen threshold：

$$
\tau=10^{-4}.
$$

Group active iff：

$$
M_g^k
=
\mathbf 1
\left[
\|\Gamma_g^k\|_2
\ge\tau
\right].
$$

### 10.5 Conv group budgets

正式：

$$
\rho_G
\in
\{0.0005,\ 0.001,\ 0.002\}.
$$

Integer group budget：

$$
K_G
=
\left\lfloor
\rho_G
|\mathcal G_{\mathrm{out}}|
\right\rfloor.
$$

因此：

| $\rho_G$ | $K_G$ |
|---:|---:|
| 0.0005 | 4 |
| 0.001 | 9 |
| 0.002 | 18 |

Budget 在 9 个 Conv layers 的 global group pool 上统一计数。

注意：

> group ratio 不是 scalar sparsity。

每个 run 还必须报告：

$$
N_{\mathrm{selected\ scalar}}
=
\sum_{g:M_g=1}|g|,
$$

以及：

$$
\rho_{\mathrm{scalar}}
=
\frac{
N_{\mathrm{selected\ scalar}}
}{
N_{\mathrm{conv}}
}.
$$

至少记录：

```text
selected_group_count
realized_group_ratio
selected_scalar_count
realized_scalar_ratio
```

---

## 11. Strict budget 与 rollback

每个 Stage-1 iteration 使用 frozen threshold 统计 support。

设当前 integer budget 为：

```text
FC   : K_star = K
Conv : K_star = K_G
```

严格状态机：

```text
n < K_star:
    当前 state 合法
    保存为 latest feasible state
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
LBI support top-K trimming
per-layer/per-tensor quota
minimum-one-per-layer repair
```

最终 support 允许：

$$
|M^\star|<K_\star.
$$

这是合法结果，不能强行补到 exact-$K$。

---

## 12. Stage-1 safety cap

Frozen：

```text
stage1_max_steps = 3000
```

它是 safety cap，不是 tuning dimension。

如果某个参数配置大量触发 cap，应判定该 config 不适合 formal，而不是偷偷增大 cap。

---

## 13. Stage-2 masked refinement

Stage 2 必须继续保持 refined LBI algorithm 的统一语义。

不能因为 IST native 有 `iters=1`，就把 Stage 2 改成多步 native IST traversal。

### 13.1 Masked-delta initialization

最终 Stage-1 state：

$$
(\Theta_\Delta^\star,\Gamma^\star,M^\star).
$$

Stage 2 初始化：

$$
\Theta_{t,2}^0
=
\Theta_t
+
M^\star
\odot
\Theta_\Delta^\star.
$$

Off-mask Stage-1 delta 不得进入 Stage-2 model。

Conv group mask broadcast 到对应 scalar coordinates。

### 13.2 Stage 2 严格只有 1 个 optimizer step

Frozen：

```text
stage2_steps = 1
```

该 step 使用同一个完整 fixed outer-batch objective：

$$
\mathcal L_t^{\mathrm{IST}}.
$$

显存不够时允许 gradient accumulation，但必须：

```text
完整 objective gradient 累积完
-> 才执行一次 optimizer step
```

禁止把 IST：

```text
iters=1
```

解释成：

```text
Stage 2 把所有 8-view inner mini-batches 逐个 optimizer step 一轮
```

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

每个 outer batch 使用 fresh local Stage-2 optimizer。

### 13.4 Strict off-mask preservation

Stage 2 中：

```text
off-mask gradients masked
off-mask parameter values restored/frozen
off-mask optimizer state 不得造成 hidden update
non-candidate parameters 不变
controlled-family BN state frozen
```

---

## 14. Persistent writeback：LBI omega 替代 IST EMA

这是正式冻结的关键规则。

Stage-2 refined state：

$$
\widetilde\Theta_t.
$$

IST-LBI 的唯一 persistent writeback：

$$
\Theta_{t+1}
=
(1-\omega)\Theta_t
+
\omega\widetilde\Theta_t.
$$

因此 IST-LBI 中：

```text
native IST EMA m=0.9 = DISABLED
LBI omega writeback  = ENABLED
```

明确禁止：

```text
LBI omega -> IST EMA
IST EMA -> LBI omega
任何 double writeback
```

也就是说：

> $\omega$ 是 IST-LBI 唯一 persistent accumulation coefficient。

由于 off-mask Stage-2 state 等于 persistent base state，因此 off-mask coordinates 在 writeback 后仍保持不变。

---

## 15. 非-LBI sparse baselines 的 IST EMA

Random / Magnitude / Saliency 不是 LBI。

它们保留 parent IST baseline 的 native self-training：

```text
PLCA once
memory once
iters=1 native IST traversal
masked sparse optimizer
native IST EMA m=0.9 once
PU read-only
```

因此：

$$
\Theta_{t+1}
=
0.9\Theta_t
+
0.1\widetilde\Theta_t.
$$

它们不使用 LBI：

$$
\omega.
$$

最终 writeback rule：

```text
IST dense / Random / Magnitude / Saliency:
    native IST EMA m=0.9

IST-LBI:
    native IST EMA disabled
    LBI omega only
```

---

## 16. FC matched sparse baseline definitions

所有 FC sparse selector：

```text
same global FC candidate pool
same integer budget K
```

### 16.1 `ist_fc_random`

每个 budget 使用 3 个 deterministic independent child masks。

每个 child：

```text
从 global FC candidate pool 均匀采样 exact K scalars
mask 整个 target stream 固定
```

三个 child：

```text
same source checkpoint
same formal seed
same target stream
only Random support differs
```

Accuracy 报：

```text
3-mask mean
```

Random mask identity 与 formal experiment seed 是两回事。

### 16.2 `ist_fc_magnitude`

在 source checkpoint：

$$
s_j
=
|\theta_j^{\mathrm{src}}|.
$$

Global top-$K$。

Mask 整个 target stream 固定。

禁止根据 adapted model 重算 magnitude。

### 16.3 `ist_fc_saliency`

对每个 valid outer batch：

1. current PLCA targets 固定；
2. native sparse self-training 前；
3. 在当前 persistent pre-adaptation model 上计算完整 outer-batch IST objective gradient：

$$
g_j^{(t)}
=
\nabla_{\theta_j}
\mathcal L_t^{\mathrm{IST}}
(\Theta_t).
$$

Score：

$$
s_j^{(t)}
=
\left|
\theta_j^{(t)}
g_j^{(t)}
\right|.
$$

选择 global top-$K$。

关键规则：

> **Saliency support 每个 outer batch 只选一次。**

该 mask 在当前 outer batch 所有 IST inner mini-batches 中固定。

下一 outer batch 才允许重新选。

禁止：

```text
每个 inner mini-batch 重新选 support
```

---

## 17. Conv matched sparse baseline definitions

所有 Conv sparse selectors：

```text
same 9216 out-channel group pool
same K_G
```

### 17.1 `ist_conv_out_random`

每个 budget 使用 3 个 deterministic independent child masks。

每个 child：

```text
uniformly sample exact K_G out-channel groups
mask 整个 target stream 固定
```

### 17.2 `ist_conv_out_magnitude`

Source checkpoint 上：

$$
s_g
=
\|W_g^{\mathrm{src}}\|_2.
$$

选择 global top-$K_G$。

Mask 整个 target stream 固定。

### 17.3 `ist_conv_out_saliency`

对每个 valid outer batch，在 PLCA targets 固定后，计算完整 outer-batch IST objective gradient。

对 group：

$$
g,
$$

score：

$$
s_g^{(t)}
=
\left\|
W_g^{(t)}
\odot
\nabla_{W_g}
\mathcal L_t^{\mathrm{IST}}
(\Theta_t)
\right\|_2.
$$

选择 global top-$K_G$。

Mask：

```text
当前 outer batch 固定
下一 outer batch 才允许改变
```

---

## 18. Random/Magnitude/Saliency 的 sparse optimizer semantics

所有 controlled sparse baselines：

```text
only selected coordinates/groups may change
off-mask gradients masked
off-mask values restored/frozen
off-mask SGD momentum/state cannot create hidden updates
controlled BN parameters/buffers frozen
netC frozen
```

Persistent support 外必须严格保持不变。

Native IST 仍然：

```text
iters=1
```

遍历当前完整 adaptation-view set。

---

## 19. IST-LBI 每个 outer batch 的精确执行顺序

对每个 valid non-singleton：

$$
B_t
$$

固定：

```text
1. 从 seed-2026 fixed stream 取得 raw B_t
2. materialize cached PU reference views
3. 每张 raw sample 生成 8 个 IST adaptation views
4. 用 persistent theta_t 做 pre-adaptation eval forward
5. 保存 F_t 与 pre-correction soft targets Q_t
6. 读取 causal memory snapshot M_{t-1}
7. PLCA 恰好一次
8. 得到 fixed corrected hard targets
9. memory commit 恰好一次
10. 冻结 D_t^IST = {views, hard targets, soft targets}
11. theta_delta / Z / Gamma 清零
12. Stage 1:
      反复计算 full D_t^IST gradient
      corrected LBI dynamics
      threshold support
      strict budget / rollback
13. 构造最终 mask M*
14. Stage 2 初始化 theta_t + M* ⊙ theta_delta*
15. 累积 full D_t^IST gradient
16. 恰好 1 个 masked Stage-2 SGD step
17. 恰好 1 个 LBI omega persistent writeback
18. 不执行 IST native EMA
19. cached reference views 上 PU，完全 read-only
20. 丢弃当前 IST/LBI batch-local state
21. 进入下一 outer batch
```

Stream 结束：

```text
freeze final persistent model
-> full-target FO
-> read-only
```

---

## 20. 非-LBI sparse 每个 outer batch 的顺序

Random / Magnitude / Saliency：

```text
1. raw + 8-view IST batch
2. pre-adaptation outputs
3. PLCA once
4. memory commit once
5. 获取 sparse mask：
      Random    = fixed stream mask
      Magnitude = fixed source mask
      Saliency  = current outer-batch full-objective mask
6. under fixed mask 做 native IST iters=1
7. native IST EMA m=0.9 once
8. PU read-only
```

这里没有：

```text
LBI local states
omega writeback
```

---

## 21. Singleton rule

继续继承 parent baseline：

```text
actual outer batch size == 1
-> 在任何 IST/LBI state transition 前 skip
```

因此 singleton 不触发：

```text
view materialization
PLCA
memory
selector
LBI Stage 1
Stage 2
native EMA
omega writeback
PU
```

FO 仍 full-target。

---

## 22. LBI frozen constants

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
| off-mask value freezing | enabled |
| LBI local restart | once per valid outer batch |
| IST native EMA inside LBI | disabled |

---

## 23. LBI tunable parameters 与 tuning granularity

只允许调：

```text
alpha
kappa
nu
omega
stage2_lr
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

Office：

> 同一个 track / budget 只能有一组 tuple，6 个 transfers 共用。

禁止：

```text
per-transfer tuning
per-seed tuning
per-target-batch tuning
per-budget tau
per-budget Stage-1 cap
per-budget Stage-2 steps
```

最终需要冻结的 tuples：

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

总计：

$$
6+6=12
$$

组 IST-LBI tuned tuples。

在对应 tuple 冻结前，不允许 formal LBI launch。

---

## 24. 本 protocol 故意不冻结的 search items

正式 tuning 前必须单独写：

```text
IST_LBI_SEARCH_PROTOCOL
```

里面才冻结：

```text
alpha/kappa/nu candidate grid
omega candidate grid
stage2_lr candidate grid
Stage-1 reachability screening
support-utilization eligibility rule
accuracy-selection rule
boundary-expansion rule
search hardware / execution allocation
```

本 protocol 不直接照搬 SHOT 的最终 winner 或 search grid。

原因：

> IST objective 会改变 Stage-1 dynamics；LBI algorithmic semantics 应该共享，但 tuned values 必须独立验证。

禁止根据 final formal result 反向扩大 search space。

---

## 25. Support utilization 与 scientific validity

### FC 至少记录

```text
selected_count
budget_K
realized_ratio
utilization = selected_count / K
Stage-1 steps
Stage-1 cap hit
overshoot/rollback status
```

### Conv 至少记录

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
```

以下任一情况视为 scientifically invalid：

```text
NaN / Inf
runtime correctness failure
selected support > strict integer budget
off-mask parameter changes
required support trace/artifact missing
Stage-1 cap 被当成成功 silent fallback
future-target leakage
target-label adaptation leakage
double EMA/writeback
```

Exact utilization eligibility threshold 留给后续 search protocol。

---

## 26. PU / FO 与 state protection

PU 发生在当前 processed outer batch 的 persistent update 之后：

```text
non-LBI sparse:
  IST native EMA -> PU

IST-LBI:
  LBI omega writeback -> PU
```

PU 完全 read-only。

不得修改：

```text
parameters
BN buffers
memory
optimizer state
selector state
LBI local state
EMA/writeback anchors
```

FO：

```text
freeze final model
-> full target eval
-> read-only
```

PU / FO 都不得影响当前 run 内 support selection。

---

## 27. Target-label policy

Target ground-truth labels 禁止进入：

```text
PLCA
memory
Random support
Magnitude support
Saliency gradient/support
LBI Stage 1
LBI Stage 2
IST native EMA
LBI omega writeback
```

只允许 evaluation/reporting branch 使用。

若未来 tuning 使用 labeled validation，必须先在 search protocol 中显式预注册。

---

## 28. Formal sparse experiment matrix

每个 parameter track、每个 transfer 的 budgeted family：

```text
Random
Magnitude
Saliency
LBI
```

Budgets：

$$
\rho_\star
\in
\{0.0005,\ 0.001,\ 0.002\}.
$$

因此每个 track：

```text
4 variants x 3 budgets
= 12 budgeted formal identities per transfer
```

Random 的 3 个 masks 是一个 top-level identity 下的 child executions。

对应 LBI tuple 未冻结时，formal LBI identity 不得 launch。

---

## 29. Scientific identity 与 provenance

至少包含：

```text
parent IST baseline protocol revision
IST-LBI protocol revision
IST-LBI implementation revision
source checkpoint revision
source checkpoint paths/SHA256
dataset / transfer
backbone
formal seed
outer batch size
target stream identity/hash
IST extend / PLCA / memory settings
candidate track
candidate parameter names
grouping mode
candidate scalar count
group count, if Conv
variant
requested sparse ratio
integer K or K_G
Random mask index/seed
selector definition
LBI alpha
LBI kappa
LBI nu
LBI omega
LBI stage2_lr
support threshold
Stage-1 cap
Stage-2 steps
writeback mode
```

IST-LBI 必须显式记录：

```text
native_ist_ema = false
persistent_writeback = lbi_omega_only
```

Random / Magnitude / Saliency：

```text
native_ist_ema = true
ema_momentum = 0.9
persistent_writeback = ist_native_ema
```

---

## 30. 实验前必须通过的 implementation contracts

正式 sparse/LBI 实现必须至少验证：

1. PLCA 每 processed outer batch 恰好一次。
2. Memory 每 processed outer batch 恰好 commit 一次。
3. LBI support discovery 每 processed outer batch 一次，不是每 inner mini-batch 一次。
4. Stage 1/2 全程 hard/soft targets 不变。
5. full-objective gradient accumulation 与小 synthetic non-chunked reference 一致。
6. accumulation chunks 之间没有 optimizer / LBI state update。
7. corrected old-state $Z$ update 保持不变。
8. strict integer rollback 保持不变。
9. FC candidate count=524544，budgets=262/524/1049。
10. Conv out-channel group pool=9216，budgets=4/9/18。
11. Conv realized scalar support 被记录。
12. Stage-2 initialization 不含 off-mask Stage-1 delta。
13. Stage 2 恰好 1 个 optimizer step。
14. Off-mask parameters 与 controlled BN buffers 不变。
15. 非-LBI sparse 每 processed outer batch native IST EMA 恰好一次。
16. IST-LBI native IST EMA commits=0。
17. IST-LBI 每 processed outer batch LBI $\omega$ writeback 恰好一次。
18. 不存在 double writeback path。
19. PU / FO read-only。
20. Singleton 在 selector/LBI/EMA/PU 前 skip。
21. Target labels 不进入 adaptation objects。
22. Scientific identity 能区分 track / budget / selector / LBI tuple / writeback mode。

这些 contracts 与 short smoke 没过之前，不允许跑 full Office / VisDA formal。

---

## 31. 明确禁止的 legacy semantics

禁止重新引入：

```text
old IST-OTTA shuffle stream
Normalize -> PIL -> IST augmentation
old IST-LBI per-tensor budgets
old-state Z bug
budget slack
LBI top-K repair
dense Stage-1 delta leakage into Stage 2
off-mask hidden updates
per-inner-mini-batch LBI support rediscovery
PLCA rerun inside LBI Stage 1
memory recommit inside LBI Stage 1
native IST EMA + LBI omega double writeback
filter_connection Conv mainline
target-label-driven online adaptation
```

---

## 32. Freeze boundary

### 已冻结

```text
outer-batch LBI support discovery
full fixed outer-batch IST objective for Stage 1
exact gradient accumulation semantics
PLCA once / memory once before LBI
fixed hard/soft targets during LBI
corrected old-state LBI dynamics
FC scalar candidate/support/budgets
Conv full-layer4 out-channel grouping/support/budgets
strict integer rollback
tau=1e-4
Stage-1 cap=3000
masked-delta Stage-2 initialization
Stage-2 exactly one full-objective masked SGD step
Stage-2 optimizer family
off-mask exact preservation
LBI omega as sole IST-LBI persistent writeback
native IST EMA disabled in LBI
native IST EMA retained for Random/Magnitude/Saliency
Random/Magnitude/Saliency selector semantics
singleton compatibility rule
PU/FO read-only semantics
filter_connection excluded
tunables limited to alpha/kappa/nu/omega/stage2_lr
tuning granularity dataset x track x budget
```

### 尚未冻结

```text
IST-LBI implementation revision
search grids
utilization/eligibility gates
selection rule
allowed boundary expansion
12 final tuned tuples
formal efficiency execution package
```

任何已冻结科学规则发生变化，都必须升级 protocol revision。

---

## 33. 一页式冻结摘要

| Category | Frozen rule |
|---|---|
| Protocol | `OTTA_IST_LBI_PROTOCOL_20260906_v1` |
| Parent | `OTTA_IST_BASELINE_PROTOCOL_20260906_v1` |
| LBI unit | 每 processed outer batch 一次 |
| LBI data | 完整 fixed 8-view outer-batch IST objective |
| PLCA | LBI 前一次 |
| Memory | LBI 前 commit 一次 |
| Targets | fixed hard + pre-correction soft |
| Stage-1 local state | 每 outer batch reset |
| Z update | corrected old-state |
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
| Stage-2 | 恰好 1 个 masked SGD step |
| Stage-2 data | 同一个完整 fixed IST objective |
| LBI persistent writeback | omega only |
| IST EMA in LBI | disabled |
| IST EMA in Random/Mag/Saliency | m=.9 |
| Saliency support | 每 outer batch 一次 |
| Filter connection | excluded |
| Tunables | alpha/kappa/nu/omega/stage2_lr |
| Formal LBI | search / tuned tuples 冻结前 blocked |

---

**End of protocol.**
