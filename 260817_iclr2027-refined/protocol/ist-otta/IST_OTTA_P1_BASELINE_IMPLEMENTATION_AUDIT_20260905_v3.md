# IST-OTTA P1 Baseline Implementation Audit v2

**Status:** reviewed and corrected; ready for contract-test/smoke on server, not yet frozen for formal baseline runs  
**Date:** 2026-09-05  
**P0 contract:** `protocol/ist-otta/IST_OTTA_P0_REFACTOR_CONTRACT_20260902_v1.md`  
**Implementation revision:** `ist_otta_p1_baseline_20260905_v3`

## 1. Review conclusion

P1 的总体设计正确：当前实现不是重新发明一个“类似 IST”的 loss，而是把官方 IST 的核心 online self-training 机制放到 P0 冻结的 SHOT-OTTA controlled substrate 上。三个 dense variant、IST 的 multi-view / PLCA / causal memory / hard+soft supervision / outer PMA，以及 shared source / stream / candidate / PU-FO / provenance 均已接入。

本轮 review 没有发现需要推翻 P1 架构的问题。发现并修正了四类 implementation-level 问题：删除 adaptation 路径中无用的 target-label expansion，并把 ground-truth label 的 device access 延后到 PU；raw image 每个 sample 从磁盘只 decode 一次；PLCA CG stopping rule 从 absolute tolerance 修正为官方 SciPy `rtol=1e-6, atol=0` 对应的 relative residual；补强 PLCA 数值等价、optimizer-state read-only、raw-I/O 与 checkpoint-hash contract tests。

## 2. 当前实现与官方 IST 的对应关系

| 官方 IST 核心 | 当前实现 | 结论 |
|---|---|---|
| 每个 incoming batch 建立临时 multi-view set | raw image 生成 `extend=8` views | 保留 |
| adaptation 前 `model.eval()` 得到 feature + soft prediction | `_pre_adaptation_outputs()` | 保留 |
| robust PLCA | `ist_otta/plca.py` | 保留 |
| L2 PLCA 使用 `K+2` neighbors | `neighbor_count = K + 2` | 保留 |
| relation `1-d/d_max`，`gamma=3` | `_normalized_affinity()` | 保留 |
| `A+A^T` + symmetric normalization | sparse symmetric normalized affinity | 保留 |
| propagation coefficient `0.99` | `propagation_alpha=0.99` | 保留 |
| CG `rtol=1e-6`, maxiter=20 | relative residual `1e-6`, 20 steps | 保留/本轮修正 |
| past memory joins current graph | `CausalMemoryBank.snapshot_for()` | 保留 |
| PLCA 后 current feature/label 写 memory | `memory.commit()` once/outer batch | 保留并加因果检查 |
| hard pseudo-label self-training | hard CE | 保留 |
| pre-correction soft target consistency | soft KL | 保留 |
| `iters=1` 遍历 temp set | `_inner_self_training()` | 保留 |
| batch end PMA `m=0.9` | `OuterBatchEMA` once/outer batch | 保留并修正 shallow anchor |

官方 `robust_PLCA` 中还计算了一个跨 8 views 的 `hard_labels` 辅助字段，但官方实际 inner training 返回的是 PLCA 后 `pseudo_labels` 与 pre-correction `soft_labels`；该辅助字段不进入优化，也不进入 memory。因此当前实现不复制这个无效中间字段，不改变训练语义。

### PLCA backend 说明

官方代码使用 FAISS GPU 建 KNN，并转 SciPy CG；当前实现使用 chunked PyTorch exact L2 KNN + sparse affinity + PyTorch CG。backend 不同，但图构造和传播公式相同。本轮额外做了随机数值对照：当前实现与按照官方 L2/CG 公式实现的 SciPy reference 在 100 个随机 trial、800 个 current-view labels 上为 `0 / 800` label mismatch。正式 runtime smoke 时仍需观察 `torch.cdist` 的速度；如果性能成为瓶颈，再单独做 backend optimization，不能在不升级 implementation revision 的情况下静默切换。

## 3. 相对原始 IST，P0 主动改变了什么

这些不是 bug，而是为 controlled study 明确做的变更。

| 项目 | 官方 IST | 当前 P0/P1 | 理由 |
|---|---|---|---|
| Network | 官方 benchmark/model wrapper | SHOT `netF -> netB -> netC` | 排除 backbone/head confound |
| Source | 官方 pretrained/source | exact SHOT formal `source_F/B/C` | 排除 source initialization confound |
| Classifier | 官方 optimizer 覆盖整模型 | `netC` frozen | 共享 SHOT source-hypothesis / candidate semantics |
| Benchmark | CIFAR/ImageNet 等官方设置 | Office-31 / VisDA-C | 与当前 ICLR study 对齐 |
| Outer stream | 官方 sampler | seed=2026 fixed stream | 所有方法同 sample order/boundary |
| Outer BS | 官方 default 128 等 | Office 64 / VisDA 256 | 去掉 batch-size confound |
| Optimizer | 官方 plain SGD, LR 1e-3 | SHOT controlled SGD/scheduler substrate | P0 冻结的 controlled-comparison选择 |
| Augmentation source | 原 benchmark transform | raw-image SHOT-size crop/flip multi-view | 保留 multi-view，同时禁止旧 Normalize->PIL port |
| Evaluation | 官方 final/eval flow | PU + FO | 与 SHOT formal 统一 |
| Controlled scopes | 官方整模型 | full / FC module / Conv module | 为后续 same-candidate sparse/LBI 对照 |

因此论文中不能声称这是“官方 IST 原配置逐字复现”。准确表述应是：**IST mechanism on the common SHOT-OTTA controlled substrate**。

## 4. P0 contract 逐项检查

| P0 项 | 状态 | 当前实现 |
|---|---|---|
| same source F/B/C | PASS | common `load_source_models` + SHA256 provenance |
| same R50/R101 | PASS | common model resolver |
| seed-2026 fixed outer stream | PASS | shared `resolve_target_order` |
| Office BS64 / VisDA BS256 | PASS | formal resolver fail-closed |
| one pass / drop_last=false | PASS at loader | raw target loader |
| raw-image IST augmentation | PASS | `ISTViewMaterializer` |
| `extend=8` | PASS | config validation |
| `iters=1` | PASS | config validation |
| PLCA repeat=1/K=50/gamma=3/l2 | PASS | config + PLCA |
| memory max_len=10000 | PASS | causal FIFO memory |
| hard CE + soft KL | PASS | `_ist_loss` |
| PMA m=0.9 once/outer batch | PASS | `OuterBatchEMA` |
| full netF+netB / netC frozen | PASS | `configure_ist_variant` |
| FC only bottleneck weight+bias | PASS | shared candidate constants |
| Conv only layer4 9 Conv weights | PASS | shared candidate constants |
| controlled BN frozen | PASS | BN forced eval |
| common optimizer/scheduler | PASS | shared optimizer + outer-step scheduler |
| PU/FO read-only | PASS in contracts | eval/no-grad + state checks |
| provenance | PASS | method/variant/source/stream/IST config/hash |
| no LBI in P1 | PASS | resolver fail-closed |

## 5. 本轮发现并修正的问题

### 5.1 Adaptation data structure 不应携带展开后的真标签

旧 P1 `MaterializedBatch` 创建了 `adaptation_labels = labels.repeat_interleave(8)`，虽然训练代码没有使用它，但它让 adaptation object 不必要地持有 target ground truth。v2 删除该字段，并将 `labels.to(device)` 延后到 adaptation + EMA 全部结束后的 PU branch。这样代码层面更容易证明 target label 只用于 read-only evaluation。

### 5.2 Raw image 被重复 decode 9 次

旧 P1 每个 sample 的 1 个 reference view + 8 个 adaptation views都会重新调用 `rgb_loader(path)`。这不改变结果，但产生无意义 I/O。v2 每个 raw sample 只 decode 一次，再从同一个 PIL image 用独立 RNG stream生成 1+8 个 stochastic views；随机裁剪/翻转语义不变。

### 5.3 PLCA CG tolerance 与官方 SciPy 语义不完全一致

旧 P1 使用 absolute residual `<=1e-6`；官方代码是 `cg(..., rtol=1e-6, atol=0, maxiter=20)`。v2 改为 relative threshold `1e-6 * ||b||`，与官方 stopping semantics 对齐。

### 5.4 Contract tests 不够强

v2 新增/补强：
- 当前 sparse-PyTorch PLCA vs dense official-equation reference label equivalence；
- `MaterializedBatch` 不存在 adaptation target labels；
- raw path 每 sample 只 decode 一次；
- 在 SGD momentum state 已真实建立后，再验证 PU/FO 不改变 optimizer state；
- FO 后继续检查 optimizer / memory / EMA anchor；
- source checkpoint SHA256 contract。

## 6. Shared SHOT code review

P1 对现有 SHOT shared code 的修改只包括：抽取 FC/Conv candidate constants、抽取 fixed-order helper、注册 IST identity/output/dispatch。与上传前的 refined snapshot 做 diff，没有发现 SHOT loss、SHOT adaptation loop、LBI engine、budget/rollback、candidate scope 或 SHOT evaluation semantics 被 P1 改写。

### 6.1 Historical singleton-batch compatibility rule

P1 review 发现现有 SHOT trainer 在 `drop_last=false` 的 loader 上仍有一条历史逻辑：当实际 outer batch size 恰好为 1 时直接 `continue`。因此 Office Amazon 作为 target 时，最后一个 singleton batch不会参与 online adaptation，也不进入 PU aggregation；但 FO 仍对完整 target test set 做 read-only evaluation。

为避免已经完成的 SHOT formal results 与 IST 使用不同 effective online stream，v3 **不修改 SHOT**，而是让 IST 明确采用同一规则：`batch_size==1 -> skip before materialization/adaptation/PLCA/memory/EMA/PU`。这不是 `drop_last=true`：其他不足完整 batch、但 size>1 的尾 batch仍正常处理。

该规则已加入 provenance 字段 `singleton_outer_batch_policy=skip_size_1_to_match_shot_history`，并加入 contract test。后续 IST formal protocol 应把这一 compatibility rule 显式写出，避免只写 `drop_last=false` 产生歧义。

## 7. 本轮实际验证

本地仅做 correctness/static validation，没有运行 Office/VisDA 数据实验：

```text
Python compile: PASS
3 variants config-only dry-run: PASS
P1 targeted contract tests: PASS
Singleton outer-batch compatibility contract: PASS
PLCA random equivalence vs SciPy reference: 0 / 800 mismatch
```

尚未运行：dataset/model smoke、full baselines、hyperparameter search、LBI。

## 8. 进入下一步的判断

P1 baseline code 本身已经具备上服务器运行 contract tests 和 short smoke 的条件。singleton-batch 语义已在 v3 与 historical SHOT 对齐；下一步先运行 P1 contracts + Office D->A short smoke，再冻结 IST baseline protocol。LBI-specific restart granularity 与 EMA/omega composition仍保持 P0 的 deferred 状态，不在 P1 提前决定。
