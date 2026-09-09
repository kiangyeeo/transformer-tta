请只修改目录：

`260817_iclr2027-refined/`

旧目录只读，不要修改。此次只做下面两个很小的 LBI refinement，不要修改 SHOT-OTTA loss、metric、BN policy、实验超参数或其他无关逻辑。不要运行 Office / VisDA / 真实模型实验。

## 1. Stage-1 budget support counting 改为使用配置中的 tau

当前 Stage-1 support counting 仍有类似：

```python
support_count = sum(abs(gamma) > 1e-4)
```

的 hard-coded threshold。

请删除这个 hard-coded `1e-4`。

我们现在统一定义最终 salient mask 为：

$$
\mathcal M
=
\mathbf 1[|\gamma|\ge\tau]
$$

因此 Stage-1 budget 判断也必须基于完全相同的 mask：

$$
support\_count
=
\|\mathcal M\|_0
$$

$$
support\_ratio
=
\frac{\|\mathcal M\|_0}
{\operatorname{numel}(\gamma)}
$$

要求：

- `tau` 使用 LBI config / engine 已有的 mask threshold，不要重新 hard-code；
- Stage-1 budget counting、最终 Stage-2 mask、`selected_param_count` / `selected_support_ratio` 必须使用完全一致的 threshold 语义；
- 最好复用同一个 helper / mask construction function，避免以后两处实现再次漂移；
- predicate 统一使用：

```python
abs(gamma) >= tau
```

不要一处 `>`、另一处 `>=`。

README 中说明：

- 论文原始 Eq.(6) 字面上使用 `||gamma||_0`；
- 当前 refined implementation 为了让 sparsity budget 与最终实际 selected mask 完全一致，使用 thresholded support：

$$
\|\mathbf 1[|\gamma|\ge\tau]\|_0
$$

来进行 strict-budget stopping。

不要改变现有 strict-budget rollback 策略。

---

## 2. `stage1_rollback_used` 只在真正发生 overshoot 回退时为 True

修正当前 rollback bookkeeping。

期望语义如下。

### 情况 A：当前 support 仍低于 budget

$$
support\_ratio < requested\_budget
$$

- 当前 state 可保存为 latest feasible state；
- 继续 Stage 1；
- `stage1_rollback_used = False`。

### 情况 B：当前 support 恰好达到 budget

$$
support\_ratio = requested\_budget
$$

- 直接保留当前 state；
- 停止 Stage 1；
- 不发生 rollback；
- `stage1_rollback_used = False`。

### 情况 C：当前 support 超过 budget

$$
support\_ratio > requested\_budget
$$

- 恢复到上一个真正的 feasible state；
- 只有此时才：

```python
stage1_rollback_used = True
```

- stop reason 应能反映发生了 budget overshoot / strict-budget rollback。

### 情况 D：因为 `stage1_max_steps` 正常结束

如果没有发生 support overshoot：

- 保留当前合法 state；
- `stage1_rollback_used = False`；
- 不要仅仅因为存在 `feasible_state` 就标记为 rollback。

换言之：

> `stage1_rollback_used=True` 必须严格意味着：当前新 state 因为 support 超过 budget 被拒绝，并且代码确实恢复到了此前的 state。

如果恢复的 state 与当前 state 实际相同，也不应算 rollback。

---

## Tests

更新 `tests/lbi_smoke_test.py` 或相关轻量测试，至少覆盖：

1. 非默认 `tau`（例如不是 `1e-4`）时，Stage-1 support counting 会随 `tau` 正确变化；
2. Stage-1 budget support 与最终 mask 的 `selected_param_count` 完全一致；
3. `abs(gamma) == tau` 时按 `>= tau` 计入 support；
4. `support < budget`：`rollback_used=False`；
5. `support == budget`：直接停止且 `rollback_used=False`；
6. `support > budget`：恢复上一 feasible state 且 `rollback_used=True`；
7. `max_steps_reached` 且没有 overshoot：`rollback_used=False`；
8. 原有 corrected Eq.(5)、masked initialization、Stage-2 value-level freezing 测试继续通过。

可以运行已有轻量 unit/smoke tests 和 `py_compile` / `git diff --check`，但不要运行任何真实 Office / VisDA 实验。

同步更新 README 对应说明。

完成后只汇报：
- 修改文件；
- 两项修改的具体实现；
- 新增/更新的测试；
- 测试结果；
- 是否发现这两项修改引入其他行为变化。