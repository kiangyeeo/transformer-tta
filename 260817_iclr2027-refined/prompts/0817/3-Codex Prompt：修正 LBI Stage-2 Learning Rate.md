请只修改目录：

`260817_iclr2027-refined/`

旧目录仅作只读参考，不要修改。

本次只修正 **OTTA LBI / SDL 的 Stage-2 learning rate 逻辑**。不要修改其他已经确认过的 LBI 行为，不要修改 SHOT-OTTA loss、metric、BN policy、实验超参数或其他无关代码。

## 目标

当前 Stage 2 在每个 OTTA batch 内会按照 local `stage2_index / stage2_steps` 重新走一遍 SHOT-style LR decay，例如：

```python
decay = (
    1.0
    + STAGE2_LR_GAMMA
    * (stage2_index + 1)
    / max(stage2_steps_requested, 1)
) ** (-STAGE2_LR_POWER)

group["lr"] = group["lr0"] * decay
```

这个行为删除。

Stage 2 改为始终使用配置中的固定：

```python
stage2_lr
```

即每一个 Stage-2 optimization step 都使用：

$$
\boxed{
\eta = \texttt{stage2\_lr}
}
$$

不要在每个 OTTA batch 内重新应用 SHOT LR scheduler。

例如：

```yaml
stage2_lr: 0.01
```

则 Stage 2 实际 optimizer LR 就应始终是：

$$
0.01
$$

而不是：

$$
0.01(1+10)^{-0.75}.
$$

## 具体要求

1. 删除/停用 LBI Stage 2 内部的 local LR decay：
   - `STAGE2_LR_GAMMA`
   - `STAGE2_LR_POWER`
   - 基于 `stage2_index / stage2_steps_requested` 的 decay 计算
   - 每一步重新覆盖 optimizer LR 的逻辑

2. Stage-2 optimizer 创建时直接使用：

```python
lr=stage2_lr
```

之后所有 local Stage-2 steps 都保持该 LR 不变。

3. 如果存在仅服务于这套 local scheduler 的常量、metadata、日志字段或 helper，清理掉；但不要顺手重构其他代码。

4. 保持以下已经修好的行为完全不变：
   - $z$ 使用旧的 $\theta_\Delta^k,\gamma^k$；
   - unified `lbi.support_threshold = tau`；
   - strict-budget rollback；
   - `stage1_rollback_used` 只表示真实 overshoot rollback；
   - masked-delta initialization：
   
$$
\theta_{\mathrm{ret}}^0
=
\theta_{\mathrm{base}}
+
M\odot\theta_\Delta
$$

   - Stage-2 optimizer step 后恢复 mask 外 parameter value；
   - final omega accumulation；
   - `effective_delta_nonzero_count <= selected_param_count`。

## Tests

更新/新增轻量测试，至少验证：

1. `stage2_steps = 1` 时，实际 Stage-2 LR 精确等于 `stage2_lr`；
2. `stage2_steps > 1` 时，每一步实际 LR 都保持等于 `stage2_lr`；
3. 不再出现 local SHOT-style LR decay；
4. 原有 LBI smoke tests 全部继续通过，尤其：
   - corrected Eq.(5) update order；
   - support threshold；
   - rollback；
   - masked initialization；
   - mask 外 value freezing；
   - effective support sanity check。

可以运行：
- LBI / repository 现有轻量 smoke/unit tests；
- `py_compile`；
- `git diff --check`。

**不要运行任何 Office、VisDA 或其他真实模型实验。**

同步更新 README，明确说明：

> LBI Stage 2 使用固定 `stage2_lr`；不在每个 streaming batch 内重新执行 SHOT-style local learning-rate schedule。

完成后汇报：
- 修改了哪些文件；
- 删除了哪些旧 LR scheduler 逻辑；
- Stage-2 LR 现在具体如何生效；
- 新增/修改了哪些测试及测试结果；
- 是否发现其他非预期行为变化。