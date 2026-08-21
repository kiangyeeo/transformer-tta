请在新目录 `260817_iclr2027-refined/` 中修改代码。**只允许修改这个目录；旧目录仅作只读参考，不要改。**

目标：修正当前 OTTA LBI / SDL 实现中的以下 4 点。不要修改 SHOT-OTTA 的其他逻辑，不要调整实验超参数，不要运行真实实验。

## 1. 修正 z update：使用旧的 θΔ^k 和 γ^k

当前实现中，`theta_delta` 先更新，然后 `z` 使用了更新后的 `theta_delta`。

改为严格按照：

$$
c^k = \frac{\theta_\Delta^k-\gamma^k}{\nu}
$$

先保存旧状态对应的 coupling，然后：

$$
\theta_\Delta^{k+1}
=
\theta_\Delta^k
-
\alpha\kappa
\left(
\nabla L_{\mathrm{adp}} + c^k
\right)
$$

$$
z^{k+1}
=
z^k+\alpha c^k
$$

最后再：

$$
\gamma^{k+1}
=
\kappa \operatorname{prox}(z^{k+1})
$$

注意：`z` 不能使用已经更新后的 `theta_delta^{k+1}`。

同步修改相关 unit/smoke test，使测试验证论文 Eq.(5) 的这个 update order。

---

## 2. 保留 rollback，但明确它不是论文原始 stopping rule

**不要删除当前 rollback / last feasible state 逻辑。**

我们仍希望 strict budget：

$$
\rho_{\mathrm{selected}}\le\rho_{\max}
$$

即如果一次 ISS step 从 budget 以下跳到 budget 以上，最终采用最后一个不超过 budget 的 feasible state。

但是：

- 不要再把 rollback 描述成论文 Algorithm 2 的原始行为；
- 在代码注释、变量命名和 README 中明确说明：
  - paper 原始规则是第一次满足 `support_ratio >= budget` 后直接停止；
  - 当前实现使用的是我们额外采用的 **strict-budget rollback variant**；
  - rollback 的目的只是避免 support overshoot。

不要引入 top-k trimming。

---

## 3. Stage 2 必须严格保证 mask 外 parameter value 不发生变化

目前即使先执行：

```python
param.grad.mul_(mask)
```

SGD 的 weight decay / momentum / Nesterov 仍可能改变 mask 外参数。

修改 Stage 2：

1. 在每次 `optimizer.step()` 前保存 candidate parameter 当前值；
2. 正常执行 masked gradient + `optimizer.step()`；
3. `optimizer.step()` 后，把 `mask == 0` 的位置恢复成 step 前的值。

最终必须保证：

$$
\theta_j^{k+1}=\theta_j^k,
\qquad \forall j\notin\mathcal M
$$

不要通过关闭全局 weight decay / momentum 来绕过这个问题，因为 mask 内仍需保持原 optimizer 行为。

增加/修改测试，显式检查 mask 外 parameter 在 Stage 2 前后完全不变。

---

## 4. Stage 2 initialization 改成 masked delta initialization

当前如果是：

$$
\theta_{\mathrm{ret}}^0
=
\theta_{\mathrm{ret}}+\theta_\Delta
$$

改为：

$$
\boxed{
\theta_{\mathrm{ret}}^0
=
\theta_{\mathrm{ret}}
+
\mathcal M\odot\theta_\Delta
}
$$

即代码语义应类似：

```python
stage2_init[name] = (
    base_parameters[name]
    + mask[name] * state.theta_delta[name]
)
```

不要使用 dense `theta_delta` initialize，也不要改成用 `gamma` 的数值 initialize。

设计语义是：

- `gamma` / `mask` 决定 **where to adapt**；
- `theta_delta` 在 mask 内提供 **initial adaptation magnitude**；
- mask 外始终保持进入当前 OTTA batch 时的 base parameter。

最终希望满足：

$$
\operatorname{supp}
(
\theta_{\mathrm{final}}-\theta_{\mathrm{base}}
)
\subseteq
\mathcal M
$$

---

## 测试与检查

修改完成后：

1. 更新已有 LBI smoke/unit tests；
2. 至少检查：
   - `z` 使用 old coupling；
   - rollback 仍正常工作；
   - masked initialization 后 mask 外 delta 为 0；
   - Stage 2 optimizer step 后 mask 外 parameter 完全不变；
   - `effective_delta_nonzero_count <= selected_support_count`；
3. 可以运行轻量 unit/smoke tests；
4. **不要运行 Office / VisDA 或任何真实模型实验。**

同时更新 `README` 中对应的 LBI/SDL 实现说明，明确：
- corrected explicit Eq.(5) update order；
- strict-budget rollback 是本实现的额外 variant；
- Stage 2 使用 masked-delta initialization；
- Stage 2 对 mask 外参数执行 value-level freezing。

完成后给我：
- 修改了哪些文件；
- 每个文件改了什么；
- 关键公式与代码如何对应；
- 跑了哪些轻量测试及结果；
- 是否还发现与论文 Algorithm 2 有其他不一致之处。

不要顺手重构无关代码，也不要改 SHOT-OTTA loss、metric、BN policy 或实验 config。