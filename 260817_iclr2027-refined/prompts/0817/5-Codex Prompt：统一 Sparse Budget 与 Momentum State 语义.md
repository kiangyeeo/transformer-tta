请只修改目录：

`260817_iclr2027-refined/`

旧目录仅作只读参考，不要修改。

本次只处理以下 3 类问题：

1. 统一 LBI strict-budget 的 integer budget semantics；
2. Random / Magnitude / Saliency 改成整个 bottleneck FC candidate pool 上的 global-budget selection；
3. masked sparse optimizer 同时 mask/reset mask 外的 momentum state。

不要修改：
- SHOT-OTTA loss / pseudo-label；
- Office / VisDA metric；
- BN policy；
- target stream/order；
- batch size；
- LBI Eq.(5)；
- LBI support threshold `tau`；
- LBI masked initialization；
- LBI Stage-2 fixed LR；
- LBI omega accumulation；
- 其他实验超参数。

不要运行任何 Office / VisDA / 真实模型实验。

---

# 1. 统一 strict budget 为 integer global budget

当前 candidate scope 是整个 bottleneck FC：

```text
netB.bottleneck.weight
netB.bottleneck.bias
```

统一定义 candidate 参数总数：

$$
N
=
\sum_j \operatorname{numel}(\theta_j)
$$

对于 requested sparsity budget：

$$
\rho_{\max}
$$

统一定义最大合法 support count：

$$
\boxed{
K
=
\left\lfloor
\rho_{\max}N
\right\rfloor
}
$$

这是 strict upper-bound budget：

$$
selected\_count\le K
$$

不要再使用：

$$
\lceil\rho_{\max}N\rceil
$$

也不要使用 floating-point：

```python
support_ratio == requested_budget
```

作为核心 stopping / validity 判断。

## 1.1 建立唯一的 budget-count helper

尽量只保留一个 source of truth，例如：

```python
max_support_count(requested_budget, candidate_param_count)
```

语义严格为：

```python
floor(requested_budget * candidate_param_count)
```

所有以下位置必须复用相同定义：

- LBI Stage-1 stopping；
- strict-budget rollback；
- `target_support_count` / `max_support_count` diagnostics；
- `budget_reached`；
- `valid_lbi_step`；
- `valid_lbi_run`；
- budget-search ranking/filter；
- summary/finalize scripts；
- Random / Magnitude / Saliency sparse baseline；
- experiment metadata 中的 target/max support count。

搜索整个仓库，清除仍然存在的：
- `ceil(rho * N)` budget 语义；
- 每个 tensor 单独计算 budget 的旧逻辑；
- 正 budget 时强制每 tensor 至少选 1 个参数的逻辑。

如果旧字段名 `target_support_count` 被大量下游工具依赖，可以暂时保留字段名，但 README 必须明确它现在表示：

> strict-budget maximum legal support count = floor(rho * N)

不要为了改名做无关的大规模重构。

---

# 2. LBI Stage-1 stopping 改成 integer-count 判断

继续使用已经统一好的：

$$
M=\mathbf 1[|\gamma|\ge\tau]
$$

计算：

$$
support\_count=\|M\|_0
$$

但 stopping 不再依赖 float ratio 与 budget 比较，而直接和 $K$ 比。

要求：

### A. `support_count < K`

- 当前 state 合法；
- 保存为 latest feasible state；
- 继续 Stage 1；
- `stage1_rollback_used=False`。

### B. `support_count == K`

- 当前 state 正好达到最大合法 support；
- 直接停止；
- 保留当前 state；
- `stage1_stop_reason = budget_reached`；
- `stage1_rollback_used=False`。

### C. `support_count > K`

- 当前 step overshoot；
- 回退到 previous feasible state；
- `stage1_stop_reason = strict_budget_rollback`；
- `stage1_rollback_used=True`。

不要 top-k trimming。

继续保持我们之前确定的 strict-budget rollback 设计。

---

# 3. 注意 rollback run 的 validity 语义

这是必须同步检查的地方。

strict-budget rollback 可能出现：

$$
support_{k-1}<K
$$

但下一步：

$$
support_k>K
$$

于是最终 selected support 是：

$$
support_{final}=support_{k-1}<K.
$$

这是我们设计中的**合法 strict-budget termination**，不能因为：

```text
selected_count != K
```

就自动把 run 判 invalid。

因此请检查所有：

- `budget_reached`
- `budget_reached_all_steps`
- `valid_lbi_step`
- `valid_lbi_run`
- search filtering / ranking

的语义。

建议明确区分：

### exact budget reached

```text
stop_reason == budget_reached
selected_count == K
```

### valid strict-budget boundary stop

```text
stop_reason in {
    budget_reached,
    strict_budget_rollback
}
and selected_count <= K
```

`strict_budget_rollback` 是合法结果。

如果现有 downstream 字段必须兼容，不需要大规模改 schema，但必须保证：

> 一个真实发生 overshoot、随后合法 rollback 到 previous feasible state 的 LBI run，不会因为最终 `selected_count < K` 被错误判 invalid。

`max_steps_reached` 的原有语义不要随意改变；只修复与 integer budget / rollback 冲突的部分。

`requested_budget=0` 的已有边界语义保持不变，除非为了兼容统一 helper 必须做最小修改。

---

# 4. Random / Magnitude / Saliency 改为 global FC candidate pool

目前不要再分别对：

```text
bottleneck.weight
bottleneck.bias
```

各自套一次 sparsity ratio。

统一把所有 FC candidate scalar parameter 看成一个 global pool：

$$
\Theta_{\mathrm{cand}}
=
[
\operatorname{vec}(W);
\operatorname{vec}(b)
]
$$

总长度：

$$
N
=
N_W+N_b.
$$

统一使用：

$$
K=\lfloor\rho N\rfloor.
$$

最终三个 sparse baseline 都最多/恰好选择这个 global $K$。

除非 $K=0$，否则不要做“每个 tensor 至少一个”。

若：

$$
K=0,
$$

则应该得到全零 mask，而不是强制选参数。

---

## 4.1 Random

在整个 global FC candidate pool 的 $N$ 个 scalar 中随机选择**恰好 $K$ 个**。

要求：

- 保持已有 seed / generator reproducibility；
- 不允许 weight 和 bias 各自随机 K；
- 不强制 bias 一定被选；
- 将 global selected indices 正确映射回各 parameter tensor mask。

---

## 4.2 Magnitude

构造全局 score：

$$
s_j=|\theta_j|
$$

将所有 candidate tensor score flatten 后视为一个 global vector，在整个 FC pool 上取：

$$
\operatorname{TopK}(s,K).
$$

再映射回 weight / bias mask。

最终：

$$
\sum_j\|M_j\|_0=K
$$

当 $K=0$ 时全部为 0。

不要每个 tensor 分别 top-k。

---

## 4.3 Saliency

保持当前 saliency score 定义不变，只修改 budget selection scope。

如果当前 score 是：

$$
s_j=|\theta_j g_j|,
$$

则继续使用这个定义。

把所有 candidate tensor 的 score flatten 成一个 global pool，然后全局 top-$K$：

$$
M=\operatorname{GlobalTopK}(s,K).
$$

再映射回各 candidate tensor。

不要修改 saliency loss、gradient 来源或其他算法定义。

---

# 5. 建议实现一个 reusable global-mask helper

为了避免 Random / Magnitude / Saliency 再次产生不同的 budget 解释，建议增加最小 reusable helper，负责：

1. 按稳定、确定的 candidate parameter 顺序建立 flatten offsets；
2. 获取总参数数 $N$；
3. 根据 global selected indices / scores；
4. 构造回：

```python
Dict[param_name, BoolTensor]
```

不要依赖无序 `set` 来决定 flatten 顺序。

candidate parameter 顺序必须 deterministic，保证：

- seed reproducibility；
- experiment identity 可复现；
- mask mapping 不随 Python 容器顺序变化。

不要顺手重构无关 trainer 代码。

---

# 6. Mask/reset optimizer momentum state

当前 masked sparse optimizer 已经做到：

1. gradient 乘 mask；
2. `optimizer.step()`；
3. mask 外 parameter value restore。

这个 value-level freezing 保持不变。

但是对于带 momentum / Nesterov 的 SGD，mask 外 coordinate 可能仍保留或累积：

```python
optimizer.state[param]["momentum_buffer"]
```

尤其 Saliency 的 mask 会动态变化，旧 support 的 stale momentum 可能在该 coordinate 后续重新进入 mask 时再次生效。

需要修正。

## 6.1 目标语义

对于当前 mask 外位置：

$$
M_j=0
$$

不仅 parameter value 不变，而且：

$$
\boxed{
momentum_j=0
}
$$

即未选中的 coordinate 不允许跨 step / batch 积累 optimizer momentum state。

对于连续保持 selected 的位置：

$$
M_j=1
$$

正常保留 momentum，不要每一步把所有 momentum 都清零。

---

## 6.2 推荐实现

在通用 masked optimizer step 中，对每个 candidate parameter：

### optimizer.step() 前

如果已经存在：

```python
optimizer.state[param]["momentum_buffer"]
```

则：

```python
momentum_buffer.mul_(mask)
```

确保 stale mask-out momentum 不参与当前 Nesterov / SGD update。

然后：

```python
param.grad.mul_(mask)
optimizer.step()
```

### optimizer.step() 后

继续执行已有的 parameter value restore：

```python
param[~mask] = pre_step_param[~mask]
```

随后如果 `momentum_buffer` 已被 optimizer 创建/更新，再次：

```python
momentum_buffer.mul_(mask)
```

确保最终 persistent optimizer state 满足：

$$
momentum_j=0,\qquad M_j=0.
$$

不要修改 mask 内 momentum。

不要关闭全局 momentum / Nesterov / weight decay。

weight decay 没有独立 persistent buffer；mask 外 parameter value 继续依靠现有 value restore 保证不变。

如果 `_masked_optimizer_step()` 被 Random / Magnitude / Saliency 共用，可以统一处理；这对固定 mask baseline 不应改变 mask 内行为。

LBI Stage-2 optimizer 是另一套逻辑且每 batch local 初始化，**不要因为本项去改 LBI**。

---

# 7. Tests

请增加/修改轻量测试，至少覆盖以下内容。

## 7.1 Integer strict budget

candidate count：

$$
N=524544
$$

应验证：

```text
rho=0.0005 -> K=262
rho=0.001  -> K=524
rho=0.002  -> K=1049
```

即全部使用 floor。

清除原来期待 263 / 525 / 1050 的测试。

---

## 7.2 LBI integer stopping

分别构造：

### support < K
- 保存 feasible；
- 继续；
- rollback=False。

### support == K
- 直接 `budget_reached`；
- selected_count == K；
- rollback=False。

### support > K
- 回到 previous feasible；
- selected_count < K；
- `strict_budget_rollback`；
- rollback=True；
- 此 run 仍应被相关 validity diagnostics 视为合法 strict-budget termination。

不要要求 rollback 后 selected_count 必须等于 K。

---

## 7.3 Global baseline selection

用很小的 weight+bias candidate tensors 构造测试，使 per-tensor budget 与 global budget 会得到不同结果。

例如：

```text
weight numel = 4
bias numel   = 2
N = 6
rho = 0.25
K = floor(1.5) = 1
```

要求：

### Random
总共只选 1 个，不是 weight 1 + bias 1。

### Magnitude
若最大 magnitude 在 weight：
- weight 选 1；
- bias 可以选 0；
- total = 1。

### Saliency
若全局最大 saliency 在某一个 tensor：
- 只选该 coordinate；
- 另一 tensor 可以完全没有 support；
- total = 1。

同时测试：

$$
K=0
$$

时所有 sparse baseline 均为 zero mask，不得强制每 tensor 至少选 1 个。

---

## 7.4 Momentum state masking

构造最小 SGD + momentum + Nesterov case，动态改变 mask。

至少验证：

### Step 1

mask：

```text
[1, 0]
```

- coordinate 0 可以产生 parameter update 和 momentum；
- coordinate 1 parameter 不变；
- coordinate 1 momentum == 0。

### Step 2

mask 切换为：

```text
[0, 1]
```

- coordinate 0 parameter value保持 step 前值；
- coordinate 0 原来的 momentum 被清零；
- coordinate 1 正常更新。

### Step 3

如果 coordinate 0 再进入 support：

```text
[1, 0]
```

它不能带回 Step 1 遗留的 stale momentum。

同时再验证：

- 一个 coordinate 连续两步都在 mask 内时，其 momentum 正常保留并累积；
- 不要把 selected coordinate 的 momentum 每步都 reset。

---

# 8. Regression tests

以下已经确认过的行为必须继续通过：

- corrected LBI Eq.(5) old-coupling update；
- unified `support_threshold=tau`；
- strict-budget rollback；
- correct rollback logging；
- masked-delta initialization；
- LBI Stage-2 mask-out value freezing；
- LBI Stage-2 fixed `stage2_lr`；
- omega accumulation；
- Office overall PU/FO；
- Full SHOT PU BN snapshot/restore；
- VisDA metric 行为；
- existing repository / experiment / launcher smoke tests。

可以运行：
- 所有相关轻量 unit/smoke tests；
- `py_compile`；
- `git diff --check`。

**不要运行任何真实 Office / VisDA 实验。**

---

# 9. README / metadata

同步更新 README，明确说明：

1. 所有 FC sparse methods 的 budget 都基于：

$$
K=\lfloor\rho N_{\mathrm{FC}}\rfloor
$$

2. Random / Magnitude / Saliency 在整个 bottleneck FC `weight+bias` global scalar pool 上选 support；

3. LBI 也使用同一个 global integer budget 上界，但 strict rollback 允许最终：

$$
selected\_count<K
$$

4. `strict_budget_rollback` 是合法的 budget-constrained termination；

5. masked sparse optimizer 对 mask 外同时执行：
   - parameter value freezing；
   - momentum-state zeroing。

如果 experiment identity / artifact metadata 中存在会影响旧结果区分的 budget-selection semantics，请同步更新，使新实验可以明确识别为新的：

- global FC budget；
- floor integer budget；
- masked optimizer momentum-state semantics。

不要修改实验数值超参数。

---

完成后请汇报：

- 修改了哪些文件；
- integer budget 的唯一 source of truth 在哪里；
- LBI stopping / rollback / validity 如何统一；
- Random / Magnitude / Saliency 如何做 global FC selection；
- momentum state 如何 mask/reset；
- 更新/新增了哪些测试；
- 所有测试结果；
- 是否发现 downstream 还有任何残留的 ceil、per-tensor budget、minimum-one-per-tensor 或旧 validity 语义。

不要顺手做无关重构，不要运行真实实验。