请只修改目录：

`260817_iclr2027-refined/`

旧目录仅作只读参考，不要修改。

本次只修正 **SHOT-OTTA 的两个问题**：

1. Office 的 PU/FO 主指标从 macro per-class accuracy 改为 overall accuracy；
2. Full SHOT 的 PU evaluation 不能额外污染 BatchNorm running statistics。

不要修改其他逻辑。

明确不要修改：
- LBI / SDL 实现；
- SHOT-OTTA loss；
- pseudo-label；
- batch size；
- target stream/order；
- singleton batch 处理；
- BN adaptation policy；
- 实验超参数；
- VisDA 指标定义。

不要运行 Office / VisDA / 任何真实模型实验。

---

## 1. Office PU / FO 主指标改为 overall accuracy

当前 Office 最终 PU/FO 汇总使用了 macro per-class accuracy。

Office 主指标应改为：

$$
\mathrm{Accuracy}
=
100\times
\frac{
\sum_i \mathbf 1[\hat y_i=y_i]
}{
N
}
$$

即普通 sample-level overall classification accuracy。

期望语义：

```text
Office:
    PU-Acc = overall accuracy
    FO-Acc = overall accuracy

VisDA:
    PU-Acc = mean per-class accuracy
    FO-Acc = mean per-class accuracy
```

### 要求

- Office 的最终 `accuracy` / `pu_acc` / `fo_acc` 主结果必须使用 overall accuracy；
- VisDA 保持当前 fixed-class mean per-class accuracy，不要修改；
- 如果 Office 当前还保存 `per_class_accuracy` 作为 diagnostics，可以继续保留；
- 不要因为修改 Office 主指标而删除已有 diagnostic 信息；
- 确认 online 每个 batch 的 post-update correct/total accumulation 与最终 PU 汇总语义一致。

增加轻量测试，至少验证：

- 构造 class-imbalanced prediction example；
- overall accuracy 与 macro accuracy 数值不同；
- Office 返回 overall accuracy；
- VisDA 路径保持原有 per-class mean 行为。

---

## 2. Full SHOT 的 PU forward 不能改变 BN persistent state

当前 Full SHOT 在 adaptation 时保持 `net_f/net_b` 的 BN 处于 train mode，这个 adaptation policy保持不变。

问题是 adaptation step 后为了计算 PU-Acc，又对同一个 batch 做了一次 forward：

```python
with torch.no_grad():
    post_outputs = ...
```

`torch.no_grad()` 不会阻止 train-mode BatchNorm 更新：

- `running_mean`
- `running_var`
- `num_batches_tracked`

因此 PU evaluation 本身会额外改变模型，并影响后续 streaming batch。

### 修改目标

保持 Full SHOT 原有 train-mode PU forward 语义，但要求：

$$
\boxed{
\text{PU evaluation 不得产生 persistent BN state change}
}
$$

推荐实现：

1. PU forward 前 snapshot 所有相关 BatchNorm buffers：
   - `running_mean`
   - `running_var`
   - `num_batches_tracked`

2. 保持当前模型 train/eval mode 不变，正常执行 PU forward；

3. PU forward 后恢复这些 BN buffers 到 snapshot；

4. 然后进入下一个 streaming batch。

伪代码语义：

```python
bn_state = snapshot_bn_buffers(model)

with torch.no_grad():
    post_outputs = model(inputs)

restore_bn_buffers(model, bn_state)
```

不要为了这个问题永久改变 BN adaptation policy。

### 特别要求

- 不要关闭 Full SHOT adaptation 阶段本来应该发生的 BN update；
- 只防止“额外的 PU measurement forward”留下新的 BN state；
- LBI / Random / Magnitude / FC-only 等现有 BN policy 不要改变；
- 如果已有 reusable BN snapshot/restore helper，优先复用；
- 如果没有，可以增加最小、清晰的 helper；
- snapshot/restore 应覆盖所有实际存在 running stats 的 BatchNorm module。

---

## Tests

增加或更新轻量 smoke/unit tests，至少覆盖：

### Office metric
1. Office 使用 overall accuracy；
2. class imbalance 下验证它不是 macro per-class accuracy；
3. VisDA metric 不受影响。

### PU-BN state
1. 构造带 BatchNorm 的简单模型；
2. adaptation-style train forward 后 BN 可以正常变化；
3. PU forward 前 snapshot；
4. PU forward 本身在 train mode 下会临时改变 BN stats；
5. restore 后：
   - `running_mean` 与 PU 前完全一致；
   - `running_var` 与 PU 前完全一致；
   - `num_batches_tracked` 与 PU 前完全一致；
6. PU prediction/result 仍能正常产生；
7. 后续 model mode 不被意外改变。

同时确保已有：
- LBI smoke tests；
- repository smoke tests；
- experiment engineering tests

继续通过。

可以运行：
- 轻量 unit/smoke tests；
- `py_compile`；
- `git diff --check`。

**不要运行任何真实 Office / VisDA 实验。**

---

## README

同步更新 README，简要说明：

- Office PU/FO 主指标采用 overall classification accuracy；
- VisDA 继续采用 mean per-class accuracy；
- Full SHOT 的 post-update PU evaluation 会恢复 BN running buffers，确保 evaluation 不改变后续 OTTA model state。

---

完成后汇报：

- 修改了哪些文件；
- Office metric 如何修改；
- BN snapshot/restore 如何实现；
- 新增/更新了哪些测试；
- 测试结果；
- 是否发现这两个修改引入其他非预期行为。

不要顺手重构或修改本次范围之外的代码。