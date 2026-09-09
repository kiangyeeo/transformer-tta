# Codex Prompt — P1 IST-OTTA Baseline Stabilization

## 任务目标

在当前主代码中完成 **P1：IST-OTTA baseline implementation/stabilization**。本阶段只改代码，把 IST baseline 接到现有 refined OTTA framework 中；**不要运行实验、不要跑 smoke、不要跑 contract tests、不要做 sparse/LBI**。代码全部改完后，我会统一 review，再单独安排运行与验证。

当前主代码：

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined
```

P0 冻结规范：

```text
260817_iclr2027-refined/protocol/ist-otta/IST_OTTA_P0_REFACTOR_CONTRACT_20260902_v1.md
```

历史 IST port（只用于参考 IST 机制和旧实现路径）：

```text
nips2026/IST-OTTA/
```

历史 IST-LBI（仅用于历史诊断，禁止作为新实现模板）：

```text
nips2026/IST-OTTA-LBI/
```

如历史代码与 P0/current refined code 冲突，优先级必须是：

```text
P0 contract
> current 260817_iclr2027-refined semantics
> official/original IST mechanism
> old IST-OTTA port
> old IST-OTTA-LBI
```

不要修改 P0 contract。

---

## 1. 先理解边界：什么跟 SHOT 共用，什么必须保留 IST

这次不是复刻一套独立官方 IST 工程，而是把 **IST adaptation mechanism 放到当前已经冻结的 SHOT-OTTA experimental substrate 上**。这样后面比较 SHOT/IST × LBI 时，source、网络、stream、batch size、candidate scope、评测等不会成为混杂变量。

### 与当前 SHOT-OTTA 共用

必须直接复用当前 refined 实现，不要另造一套：

- `netF -> netB -> netC` 网络结构；
- exact same frozen source `F/B/C` checkpoints；
- source revision `nips2026_shot_otta_uda_source_v1`；
- Office-31 / VisDA-C dataset 与 transfer 定义；
- seed=2026 fixed outer target stream；
- Office outer batch size=64；
- VisDA outer batch size=256；
- one-pass target stream；
- PU / FO evaluation semantics；
- Office overall accuracy / VisDA fixed-12-class mAcc；
- current artifact / provenance / scientific identity infrastructure；
- current FC candidate definition；
- current Conv full-layer4 9-Conv candidate definition；
- controlled FC / Conv BN freeze semantics；
- P0 已冻结的 common SHOT optimizer/scheduler substrate。

**不要改现有 SHOT-OTTA 的行为。**新增 IST 路径时避免对现有 SHOT formal configs、trainer semantics、protocol 文件产生 side effect。

### 必须保留 IST 自己的方法机制

以下属于 IST，不得换成 SHOT loss：

- multi-view self-training；
- `extend=8`；
- robust PLCA；
- causal memory bank；
- corrected hard pseudo-label；
- pre-correction soft target；
- `hard CE + soft KL`；
- `iters=1`；
- parameter moving average，`m=0.9`，每个 incoming outer batch 完成后一次。

SHOT 的 entropy/diversity/current-batch argmax objective **不得混入 IST baseline**。

---

## 2. 实现三个 dense IST baseline

P1 只实现：

```text
ist_full_dense
ist_fc_module_dense
ist_conv_module_dense
```

不要实现：

```text
Random
Magnitude
Saliency
FC-LBI
Conv Group-LBI
filter_connection
任何 sparse selector
任何 LBI tuning/search
```

### `ist_full_dense`

作为 IST native/full reference：

```text
trainable: netF + netB
frozen: netC
BN: P0 中定义的 native/full train behavior
```

它的作用是回答：在完全相同 source/network/stream/evaluation 下，完整 IST adaptation 本身表现如何。

### `ist_fc_module_dense`

只允许以下 candidate persistent update：

```text
netB.bottleneck.weight
netB.bottleneck.bias
```

其余参数全部冻结；所有 BN parameters 和 running buffers 都冻结。

这是后续 FC Random/Magnitude/Saliency/LBI 的 matched module-dense reference。

### `ist_conv_module_dense`

只允许当前 formal full-layer4 的 9 个 Conv weights persistent update：

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

其余参数全部冻结；所有 BN parameters 和 running buffers 都冻结。

这是后续 Conv out-channel Random/Magnitude/Saliency/Group-LBI 的 matched module-dense reference。

---

## 3. 每个 incoming outer batch 的固定执行顺序

必须按照 P0 实现，避免把旧 port 的数据流直接复制过来。

设当前 raw target batch 为 `B_t`，进入 batch 前模型为 `theta_t`，历史 memory 为 `M_{t-1}`：

1. 按 frozen seed-2026 outer stream 取得当前 raw sample identities。
2. 从 raw image 生成并缓存当前 batch 的 PU reference view。
3. 从同一 raw image 独立生成 IST 的 8-view adaptation set。
4. 用 adaptation 前的 `theta_t` 在 eval behavior 下计算 IST adaptation views 的 features 与 soft predictions，不允许产生 persistent parameter/BN update。
5. PLCA 只使用 current views + `M_{t-1}` 生成 corrected hard targets，禁止 future-target access。
6. 当前 features / corrected pseudo labels 向 memory commit 一次，得到 `M_t`。
7. 使用当前临时 adaptation set 做 `iters=1` 的 IST self-training；loss 固定为 `hard CE + soft KL`。
8. inner optimization 完成后，对 incoming outer batch 只做一次 IST parameter moving average：
   `theta_{t+1} = 0.9 * theta_t + 0.1 * theta_raw`。
9. 用 `theta_{t+1}` 对已经缓存的 PU reference view 做 read-only PU evaluation。
10. 丢弃当前 temp adaptation set，进入下一个 outer batch；完整 stream 后用 frozen final model做 read-only FO。

Memory、EMA、PU、FO 的 state transition 次数必须以 **incoming outer batch** 为单位，不得因为 `extend=8` 多执行。

---

## 4. 数据与 augmentation：不要复制旧 port 的错误路径

旧 `IST-OTTA` 中曾存在类似：

```text
SHOT transformed/normalized tensor
-> ToPILImage
-> IST augmentation
```

新版禁止这样做。

IST adaptation views 必须从 raw / pre-normalization image 构造：

```text
raw image
-> IST multi-view augmentation
-> network-required preprocessing / normalization
-> model
```

跨 SHOT/IST 需要相同的是：

```text
sample identity
outer stream order
outer batch membership
```

不是要求两种方法产生完全相同的 augmentation pixels。

PU reference view 要在当前 batch materialize 时缓存；adaptation 完成后的 PU 不得重新随机采样。

IST augmentation RNG 必须从 formal seed 派生，并接入当前 provenance/RNG discipline。

---

## 5. Optimizer / LR：按 P0 已冻结的 common substrate

不要自行切回旧 IST parser 默认值，也不要重新设计 optimizer。

Office：

```text
base LR = 0.01
netF LR multiplier = 0.1
netB LR multiplier = 1.0
```

VisDA-C：

```text
base LR = 0.001
netF LR multiplier = 0.1
netB LR multiplier = 1.0
```

共同：

```text
SGD momentum = 0.9
weight_decay = 0.001
Nesterov = true
```

scheduler：

$$
\eta_t
=
\eta_0
\left(
1+10\frac{t}{T}
\right)^{-0.75}.
$$

其中 `t` 按 **outer online batch** 推进。IST 同一个 outer batch 内由 `extend=8` 产生的 inner self-training mini-batches共享同一个 `eta_t`，不能把 scheduler 推进约 8 倍。

---

## 6. EMA 必须修掉旧 shallow-reference 风险

不要直接把可共享 storage 的 `state_dict()` 当作 old-model anchor。

每个 incoming outer batch 开始时，需要保存独立的 old parameter snapshot，例如语义上等价于：

```text
detach().clone()
```

inner adaptation 得到 raw adapted state 后，再执行一次：

$$
\theta_{t+1}
=
0.9\theta_t
+
0.1\widetilde{\theta}_t.
$$

EMA 只能执行一次。PU、FO、memory update、logging 都不能触发额外 EMA。

实现时注意 state dict 中非浮点 buffer，不要对不应做 EMA 的状态做错误浮点插值；BN/state semantics 必须遵循 P0 中 full vs controlled 的定义。

---

## 7. 建议的代码组织

优先把 IST 做成当前 refined framework 的 method-specific adapter/trainer，而不是复制整个旧 `IST-OTTA`。

可以根据当前 repo 架构决定最终文件名，但应保持以下职责分离：

```text
common refined OTTA infrastructure
    |
    +-- SHOT-specific path        # existing, do not change semantics
    |
    +-- IST-specific path
          +-- augmentation
          +-- PLCA
          +-- memory
          +-- hard/soft targets
          +-- IST loss
          +-- outer EMA
```

网络、source loading、stream、evaluation、candidate scope、artifact/provenance 尽量调用已有公共实现。

如果旧 `IST-OTTA/utils/` 中的 PLCA、memory、augmentation/self-training 代码可复用，可以有选择地移植其 IST-specific 部分；不要把旧工程的 loader、network、evaluation、sparse/LBI infrastructure整体复制进来。

---

## 8. 同时写好 contract tests，但本次不要运行

新增 targeted tests，使后续可以验证至少以下内容：

1. IST 与 SHOT 加载 exact same source F/B/C。
2. 同一 deterministic input 上，adaptation 前 IST/SHOT source logits 一致。
3. seed=2026 target sample indices 和 outer batch boundaries 与 current formal stream 一致。
4. PLCA/memory 在 `B_t` 只能看到 current + past，不能看到 future。
5. adaptation pipeline 不存在 `Normalize -> PIL` round-trip。
6. memory 每个 incoming outer batch只 commit 一次。
7. EMA 每个 incoming outer batch只执行一次，且 first-batch EMA 不是 shallow-reference no-op。
8. `netC` 全程不变。
9. `ist_fc_module_dense` 只有 bottleneck FC weight/bias 能改变，其他 parameter / BN buffer 不变。
10. `ist_conv_module_dense` 只有 9 个 layer4 Conv weights 能改变，其他 parameter / BN buffer 不变。
11. PU 前后 parameter、BN buffer、memory、optimizer、EMA anchor 都不变。
12. FO 完全 read-only。
13. source revision、stream hash、method/variant identity、artifact metadata 能进入当前 refined provenance pipeline。

**这次只实现 tests，不运行 tests。**

---

## 9. Config / launcher / artifact 接口

为三个 IST dense variants补齐后续运行所需的 config/entry-point/variant registration，但本次不要启动任何 job。

要求：

- 与现有 SHOT variant 命名不冲突；
- IST artifact/log 路径与 SHOT 区分清楚；
- manifest/config/results 中能明确记录 `method=IST` 和具体 variant；
- source revision、formal seed、stream identity等公共 provenance字段继续沿用；
- IST-specific config（extend/PLCA/memory/EMA）集中定义，不要散落 magic numbers；
- 不要创建任何“旧配置 fallback”导致 formal runner 可以静默走回旧 IST 实现。

---

## 10. 本任务禁止做的事情

本次只完成 P1 代码，不运行。

禁止：

```text
运行 Office/VisDA experiment
运行 smoke
运行 contract tests
运行 hyperparameter search
运行 formal baseline
实现 Random/Magnitude/Saliency
实现 FC-LBI / Conv-LBI
修改 corrected LBI engine
重新打开 filter_connection
修改 SHOT formal scientific semantics
修改 P0 contract
重新训练 source model
```

可以做静态代码检查、阅读、`git diff`、`git status` 等不启动训练/测试的操作。

---

## 11. 完成后的交付

代码修改完成后生成：

```text
260817_iclr2027-refined/protocol/ist-otta/IST_OTTA_P1_BASELINE_IMPLEMENTATION_AUDIT_20260905_v1.md
```

文档不要写成长篇流水账，按下面结构简洁说明：

### A. 实现结果
- 新增/修改文件；
- 三个 IST dense variants 如何接入；
- per-outer-batch execution order。

### B. 基于原 IST 主动改了什么
逐项说明：
- 改了什么；
- 为什么为了 controlled comparison 必须这样改；
- 哪些 IST method semantics 保留不变。

重点至少覆盖：
- SHOT F/B/C + exact same source；
- fixed outer stream / BS / one-pass；
- common optimizer/scheduler substrate；
- raw-image IST augmentation；
- current PU/FO；
- controlled FC/Conv scope与BN；
- provenance。

### C. 相对旧 `IST-OTTA` 修复了什么
至少记录：
-旧 shuffle/stream差异；
- `Normalize -> PIL -> augmentation`；
-历史 LR provenance冲突；
- EMA shallow-reference风险；
-旧工程与当前 refined evaluation/provenance不一致的部分。

### D. Contract tests
列出已经实现的测试文件/测试项，并明确：

```text
implemented, not executed in P1 code-only phase
```

### E. 尚未处理
明确写：

```text
Random/Magnitude/Saliency: not implemented
FC-LBI: not implemented
Conv out-channel Group-LBI: not implemented
IST-LBI restart granularity: deferred
IST EMA vs LBI omega composition: deferred
hyperparameter search/formal runs: not started
```

最后输出：

```text
git diff --stat
git status --short
```

不要 commit，等我 review 后再决定。
