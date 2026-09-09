# Codex Prompt — P1 IST-OTTA Contract + Short Smoke

当前代码目录：

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined
```

Conda 环境：

```text
SHOT_TTA
```

P0 规范：

```text
protocol/ist-otta/IST_OTTA_P0_REFACTOR_CONTRACT_20260902_v1.md
```

P1 audit：

```text
protocol/ist-otta/IST_OTTA_P1_BASELINE_IMPLEMENTATION_AUDIT_20260905_v3.md
```

本任务只做 **P1 correctness validation + short smoke**。不要开始完整 Office/VisDA baseline，不做 sparse/LBI，不做 tuning。

---

## 1. 先运行 P1 contract tests

进入目录并激活环境：

```bash
cd /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined
conda activate SHOT_TTA
```

运行：

```bash
python tests/ist_otta_p1_contract_test.py
```

预期：

```text
IST-OTTA P1 contracts passed
```

如果 contract test 失败：

- 立即停止；
- 先定位并修 correctness；
- 不要继续 smoke；
- 不要根据 accuracy 调整任何实现。

---

## 2. 增加 smoke-only outer-batch limit

当前正式 trainer 没有 short-smoke batch limit。增加一个最小的 debug-only 开关，例如：

```text
--debug-max-outer-batches 5
```

要求：

1. 默认值为 `None`，现有 SHOT/IST normal/formal run 行为完全不变。
2. 只允许 non-formal/debug smoke 使用；formal run 设置该参数必须直接报错。
3. 不进入 scientific experiment identity。
4. artifact / manifest 必须明确记录：

```text
debug_smoke = true
debug_max_outer_batches = 5
processed_outer_batches = <actual value>
```

5. limit 按 **实际处理的 non-singleton outer batches** 计数。
6. singleton policy 与历史 SHOT formal 完全一致：

```text
outer batch size == 1
-> skip
```

singleton 必须在 adaptation 前跳过，因此不得触发：

```text
IST augmentation
PLCA
memory commit
optimizer update
EMA
PU
```

FO 的正式语义仍是 full-target read-only evaluation；本 smoke 不需要把结果当 formal FO。

7. 处理满 5 个有效 outer batches 后，正常退出 adaptation loop并正常 finalize artifacts。
8. 禁止使用 `timeout`、kill、异常退出或截断进程来制造 short smoke。
9. 这是纯工程/debug开关，不得修改 IST/SHOT scientific semantics。

只做最小必要改动，不要重构 trainer。

---

## 3. 运行 Office D -> A short smoke

Office domain mapping：

```text
amazon = 0
dslr   = 1
webcam = 2
```

因此：

```text
D -> A = source 1, target 0
```

统一设置：

```text
dataset = office
source = 1
target = 0
seed = 2026
valid outer batches = 5
conda env = SHOT_TTA
```

依次跑：

```text
ist_full_dense
ist_fc_module_dense
ist_conv_module_dense
```

输出必须进入独立 smoke 目录，例如：

```text
runs_smoke/ist_p1/
```

不要写入或覆盖任何 formal experiment artifact/log 目录。

如果当前 `train.py` / config interface 已经支持通过 variant/config 启动，直接使用现有正式入口，不要另写一套 smoke trainer。

---

## 4. Smoke 只检查 correctness

每个 variant 只检查以下项目：

### Common

- 正常处理 5 个有效 outer batches；
- singleton policy 与 SHOT 一致；
- raw-image 8-view augmentation 正常；
- PLCA 输出 finite；
- causal memory 正常增长；
- hard CE finite；
- soft KL finite；
- total loss finite；
- optimizer step 正常；
- EMA 每个有效 outer batch 恰好一次；
- PU 正常记录；
- 无 NaN / Inf；
- 无 CUDA error / OOM；
- manifest / metrics / summary 等 artifact 正常落盘；
- source revision、seed、stream identity、IST variant 信息记录正确。

### `ist_full_dense`

确认：

```text
netF + netB adapt
netC frozen
```

### `ist_fc_module_dense`

确认只有：

```text
netB.bottleneck.weight
netB.bottleneck.bias
```

发生 persistent update；其他 parameters 和 BN buffers 不变。

### `ist_conv_module_dense`

确认只有 full `netF.layer4` 的 9 个 Conv weights发生 persistent update；其他 parameters 和 BN buffers 不变。

---

## 5. 不要做的事情

本任务禁止：

```text
完整 D->A
Office 6 transfers
VisDA
hyperparameter search
formal baseline
Random
Magnitude
Saliency
FC-LBI
Conv Group-LBI
filter_connection
修改 P0 protocol
修改现有 SHOT formal semantics
根据 smoke accuracy 调参
重新训练 source model
```

Smoke accuracy 可以记录，但不能用于任何实现选择。

---

## 6. 完成后汇报

只输出：

1. P1 contract test 是否 PASS；
2. smoke-only batch-limit 修改了哪些文件；
3. 三个 variant 各自 PASS / FAIL；
4. 每个 variant 实际 processed outer batch 数；
5. singleton skipped 数；
6. hard CE / soft KL / total loss 是否全部 finite；
7. 最终 memory size；
8. EMA 调用次数；
9. 是否出现 NaN / Inf / OOM / CUDA error；
10. 每个 smoke run 的输出目录；
11. 发现的 correctness 问题及修复；
12. `git diff --stat`；
13. `git status --short`。

如果三条 smoke 都通过，最后只给结论：

```text
P1 baseline implementation is ready for protocol freeze.
```

不要继续启动下一阶段。
