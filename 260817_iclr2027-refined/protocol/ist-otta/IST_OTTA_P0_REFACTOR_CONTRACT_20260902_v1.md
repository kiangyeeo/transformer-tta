# IST-OTTA P0 Refactor Contract

**Status:** Frozen for P1 baseline implementation  
**Date:** 2026-09-02  
**Scope:** IST mechanism on the current controlled SHOT-OTTA substrate  
**Next:** P1 = implement/validate dense IST baselines; no sparse/LBI yet.

## 1. P0 结论

当前 ICLR 主线不重新维护一套独立的“官方 IST 工程”，而是把 **IST 的核心方法机制**放到已经冻结的 SHOT-OTTA experimental substrate 上。共享部分包括 source model、F/B/C 网络、target stream、batch size、trainable scope、BN control、PU/FO、metrics、runtime/provenance；IST 保留自己的 multi-view augmentation、robust PLCA、causal memory、hard+soft self-training 和 parameter moving average。这样做的目的，是让后续比较尽量只反映 **adaptation mechanism / objective 与 LBI 的关系**，而不是被 source、backbone、batch size、stream order 或 evaluation protocol 混淆。

> **从 P1 起：共享 SHOT-OTTA 的实验底座；保留 IST 的方法核心；旧 `IST-OTTA` / `IST-OTTA-LBI` 只作历史参考，不是规范来源。**

## 2. Source of truth

冲突时按以下优先级处理：

1. `260817_iclr2027-refined`：benchmark、source、stream、PU/FO、FC/Conv candidate、controlled BN、runtime/provenance；后续 LBI 也只认这里的 corrected semantics。
2. 官方 `JingInAI/IST4TTA`：IST-specific semantics，包括 multi-view self-training、robust PLCA、memory、hard/soft targets、parameter moving average。
3. 旧 `IST-OTTA`：只用于确认此前如何把 IST 搬到 SHOT F/B/C，并定位 porting 问题。
4. 旧 `IST-OTTA-LBI`：只作历史诊断，禁止作为新 LBI 实现模板。

理由：这次不是修旧 ICML 代码，而是在已经验证过的 refined OTTA/LBI 基础设施上重新建立 IST 分支。

## 3. 原 IST 中必须保留什么

设当前 incoming target batch 为 $B_t$，进入 batch 前模型为 $\theta_t$，历史 memory 为 $\mathcal M_{t-1}$。

| IST mechanism | P0 frozen semantics | 理由 |
|---|---|---|
| Multi-view | `extend=8` | IST 用当前 batch 多视图做 self-training，是方法核心 |
| PLCA | `repeat=1`, `K=50`, `gamma=3`, `mode=l2` | 保留官方 robust pseudo-label correction |
| Memory | `max_len=10000`，只含已见 target | IST 允许 causal past memory，但禁止 future-target access |
| Training target | corrected hard PL + pre-correction soft target | 保留 IST hard/soft 双重监督 |
| Inner iterations | `iters=1` | 保留当前 IST 默认 batch self-training 强度 |
| Parameter moving average | $m=0.9$，每个 incoming batch 完成后一次 | 属于 IST 的长期稳定机制 |

当前 IST loss 固定为：

$$
\mathcal L_{\mathrm{IST}}
=
\operatorname{CE}(p_\theta(x_i),\hat y_i)
+
D_{\mathrm{KL}}\left(q_i\,\|\,p_\theta(x_i)\right),
$$

其中 $\hat y_i$ 是 PLCA 后 corrected hard target，$q_i$ 是 PLCA 前保存的 soft prediction。SHOT 的 entropy/diversity/current-batch argmax loss 不得混进 IST baseline。

Memory 的因果顺序固定为：

$$
(B_t,\mathcal M_{t-1})
\rightarrow
(F_t,Q_t)
\rightarrow
\hat Y_t
\rightarrow
\mathcal M_t.
$$

因此论文不要再写“OTTA 禁止访问过去样本”；准确说法是：**OTTA 禁止 future-target access，允许方法显式定义的 causal past memory。**

## 4. 相对原始 IST，我们为了 controlled study 改什么

新的分支应理解为 **IST-OTTA on the common SHOT substrate**，不是官方 IST 在其原模型/原 benchmark 上的逐字复现。

| 项目 | 原始 IST | 当前 IST-OTTA formal | 为什么改 |
|---|---|---|---|
| Network | 官方模型接口 | SHOT `netF → netB → netC` | 消除 backbone/head 差异 |
| Source | 官方 pretrained/source | 与 SHOT formal 完全相同的 `source_F/B/C` | source initialization 不得成为混杂变量 |
| Office / VisDA | 非官方 IST 原 benchmark | R50 / R101 | 与当前 SHOT formal 一致 |
| Bottleneck/head | 官方模型结构 | `Linear(2048,256)+BN` + WN classifier | 共享 FC candidate 与 source hypothesis |
| Classifier | 方法原 head | `netC` 永远 frozen | 与 SHOT source-hypothesis语义一致 |
| Target order | sampler/shuffle | seed=2026 fixed stream | 每个方法看到相同 sample order / batch partition |
| Outer BS | 官方配置 | Office 64 / VisDA 256 | 去掉 batch-size confound |
| Target passes | online | 1 | 当前 formal OTTA invariant |
| Optimizer substrate | 官方 plain SGD | 当前 SHOT controlled optimizer/scheduler | 尽量只让 adaptation mechanism 不同 |
| Evaluation | 官方流程 | current PU / FO | 统一 plasticity/stability 定义 |
| Runtime/provenance | 官方实现口径 | refined artifact protocol | formal 可比、可复现 |

### 4.1 Source 不重训

必须满足：

$$
\theta_{\mathrm{src}}^{\mathrm{IST}}
=
\theta_{\mathrm{src}}^{\mathrm{SHOT}},
$$

source revision 固定为：

```text
nips2026_shot_otta_uda_source_v1
```

P1 记录 exact paths，建议同时记 SHA256，并做同一 deterministic input 的 initialization-logit equality test。为 IST 单独重训 source 会引入 source-training confound，因此禁止。

### 4.2 Network training / update scope 与 SHOT 对齐

P1 只实现三种 dense reference：

| Variant | Persistent trainable scope | BN semantics | `netC` |
|---|---|---|---|
| `ist_full_dense` | `netF + netB` | native/full train behavior | frozen |
| `ist_fc_module_dense` | `netB.bottleneck.weight/bias` | all BN frozen | frozen |
| `ist_conv_module_dense` | `netF.layer4` 9 Conv weights | all BN frozen | frozen |

理由：`full_dense` 是 IST full-reference；FC/Conv module-dense 是后续 Random/Magnitude/Saliency/LBI 的 matched candidate reference。Controlled family 如果让 BN 继续更新，就会出现 candidate mask 外的隐形 adaptation。

### 4.3 Optimizer / LR 也按当前 SHOT training substrate 统一

P1 不是比较“官方 IST optimizer vs 官方 SHOT optimizer”，而是在同一 training substrate 上比较 adaptation mechanism。因此固定：

| Item | Office | VisDA-C |
|---|---:|---:|
| base LR | 0.01 | 0.001 |
| `netF` LR multiplier | 0.1 | 0.1 |
| `netB` LR multiplier | 1.0 | 1.0 |
| SGD momentum | 0.9 | 0.9 |
| weight decay | 0.001 | 0.001 |
| Nesterov | true | true |

并使用当前 SHOT one-pass global schedule：

$$
\eta_t
=
\eta_0\left(1+10\frac{t}{T}\right)^{-0.75},
$$

其中 $t$ 按 **outer online batch** 计数；IST 同一 incoming batch 内的所有 inner self-training mini-batches共享同一个 $\eta_t$，进入下一 outer batch 才推进 scheduler。这样不会因为 `extend=8` 让 LR 时间轴快约 8 倍。

这是一项 deliberate controlled change：官方 IST 的 optimizer recipe 不作为本研究的独立变量；IST 的 PLCA/memory/hard-soft objective/EMA 才是 method-specific mechanism。

## 5. 旧 IST-OTTA 中禁止继承的问题

| 问题 | 旧行为 | 新 P0 决定 | 原因 |
|---|---|---|---|
| Target order | `shuffle=True` | fixed seed-2026 order | 在线顺序不能成为混杂变量 |
| Image path | SHOT `Normalize` 后 tensor → `ToPILImage()` → 再 augmentation | 禁止；IST views 必须从 raw/unnormalized image 产生 | normalized tensor 不是可靠可逆图像，旧路径把两套 augmentation 混在一起 |
| Office LR provenance | parser 默认 `1e-2`，历史文档又写过 `1e-3` | 不继承旧结果，按本合同重跑 | 历史配置无法作为 formal provenance |
| Momentum anchor | 初始化直接保存 `state_dict()` | anchor 必须 `detach().clone()` | 避免 shallow-reference 导致 first-batch EMA 退化 |
| Sparse budget | 旧代码存在 per-tensor ratio/budget | P1 不做 sparse；P3 直接复用 refined global budget | 与当前 SHOT sparse comparison 对齐 |
| Old LBI | 旧 $z$、budget tolerance、dense Stage-2 init 等 | 全部禁止复用 | 当前 corrected engine 已替代 |

### 5.1 IST augmentation 与 common stream 怎么同时满足

跨方法真正必须相同的是：**target sample identity、顺序和 outer batch membership**，不是要求 IST 的 8 个 augmentation pixels 与 SHOT 的 1 个 online view 像素完全一致。新版数据流固定为：

1. outer stream 按 seed=2026 取 raw sample identities；
2. 每个 sample 从 raw image 生成一个 current reference view，缓存给该 batch 的 PU；
3. IST 从同一 raw image 独立生成 $E=8$ adaptation views；
4. adaptation 后 PU 只 forward 缓存的 current reference view，不重新采样、不更新任何 state。

IST augmentation RNG 必须由 formal seed 派生并记录。这样既保留 IST multi-view mechanism，又避免旧 `Normalize → PIL` 问题。

### 5.2 EMA correctness fix

IST moving average 的意图固定为：inner adaptation 得到 $\widetilde\theta_t$ 后，每个 incoming outer batch只执行一次：

$$
\theta_{t+1}=0.9\theta_t+0.1\widetilde\theta_t.
$$

实现必须在 batch 开始时持有真正独立的 $\theta_t$ snapshot；PU、FO、memory update 都不能额外触发 EMA。

## 6. P1 每个 incoming batch 的精确执行顺序

设当前 raw sample batch 为 $B_t$，起始 model 为 $\theta_t$，memory 为 $\mathcal M_{t-1}$。

1. **Materialize current batch**：按 frozen stream 取得 $B_t$；从 raw samples 生成 PU reference view 和 IST 的 8-view adaptation set。
2. **Pre-adaptation prediction**：model 进入 eval behavior；仅用 $\theta_t$ 计算 adaptation views 的 features 和 soft predictions，不允许 parameter/BN persistent update。
3. **Robust PLCA**：只使用当前 views + $\mathcal M_{t-1}$ 生成 corrected hard targets；不得访问未来 batch。
4. **Memory commit**：当前 features / corrected pseudo labels 写入 $\mathcal M_t$，每个 incoming batch只 commit 一次。
5. **IST inner self-training**：`iters=1`，遍历当前临时 adaptation set；loss 始终是 `hard CE + soft KL`，同一 outer batch 内 LR 固定为 $\eta_t$。
6. **Parameter moving average**：inner optimization 得到 $\widetilde\theta_t$ 后，只做一次 $\theta_{t+1}=0.9\theta_t+0.1\widetilde\theta_t$。
7. **PU**：用 $\theta_{t+1}$ 对缓存的 current reference view 做 read-only evaluation；不得改变 parameter、BN、memory、optimizer、EMA anchor。
8. **Next batch / FO**：丢弃当前 temp set 进入 $B_{t+1}$；完整 stream 后冻结 final model，FO 完全 read-only。

## 7. P1 benchmark invariants

| Item | Office-31 | VisDA-C |
|---|---|---|
| Backbone | ResNet-50 | ResNet-101 |
| Source revision | `nips2026_shot_otta_uda_source_v1` | same |
| Outer batch size | 64 | 256 |
| Workers | 4 | 4 |
| Target passes | 1 | 1 |
| Formal seed | 2026 | 2026 |
| `drop_last` | false | false |
| Primary PU/FO metric | sample overall Acc | fixed-12-class mAcc |
| FO | final frozen read-only | final frozen read-only |

Office 仍为 6 个 directed transfers 等权平均；VisDA-C 为 Synthetic/Train → Real/Validation。

## 8. P1 必过的 contract tests

P1 先验收 correctness，不用 accuracy 选实现：

1. IST 与 SHOT 加载 exact same F/B/C checkpoint；初始化 deterministic logits 一致。
2. `(dataset, transfer, seed=2026)` 的 sample indices / outer batch boundaries 与 formal manifest 一致。
3. PLCA/memory 在 $B_t$ 只可见 current + past，不可见 future。
4. IST adaptation views 不存在 `Normalize → PIL` round-trip。
5. Memory 每个 incoming outer batch只 commit 一次。
6. EMA 每个 incoming outer batch只 commit 一次，且 first-batch EMA 非 no-op。
7. `netC` 全程 byte-level 不变。
8. FC module-dense 除 bottleneck FC weight/bias 外，parameters 与 BN buffers 不变。
9. Conv module-dense 除 layer4 9 Conv weights 外，parameters 与 BN buffers 不变。
10. PU 前后 parameters、BN buffers、memory、optimizer state、EMA anchor 完全一致。
11. FO 完全 read-only。
12. source revision、stream hash、scientific identity、artifact metadata 接入当前 refined provenance discipline。

全部通过后，才跑完整 `ist_full_dense / ist_fc_module_dense / ist_conv_module_dense` formal baseline。

## 9. P0 暂不冻结的 LBI-specific 决策

P0 只负责把 **P1 IST baseline** 钉死。以下问题在 P3 写独立 `IST_LBI_PROTOCOL` 前再正式决定，避免未经验证就永久化：

1. LBI restart / support discovery 是按一个 incoming outer batch，还是按 IST inner self-training mini-batch。
2. IST outer EMA 与 refined LBI Stage-3 $\omega$ 如何组合；禁止默认旧代码的双重 writeback。
3. IST-LBI 的 $(\alpha,\kappa,\nu,\omega,\mathrm{stage2\_lr})$ 搜索空间。
4. IST dynamics 下的 utilization gate 是否原样继承 SHOT，还是需要预注册调整。

但以下原则现在已经冻结：**IST-LBI 只能接当前 corrected LBI engine；禁止 old-state bug、budget slack/top-K repair、dense-delta leakage、off-mask hidden update；Conv 主线只做 full `layer4` + `out_channel`，不重新打开 `filter_connection`。**

## 10. 相对原 IST 到当前 IST-OTTA，一句话怎么说

> We retain IST's causal multi-view self-training, PLCA-based pseudo-label correction, memory bank, hard/soft supervision, and parameter moving average, while transplanting them onto the same frozen F/B/C source model, target stream, optimization substrate, controlled update scopes, and PU/FO evaluation protocol used by SHOT-OTTA. This isolates the adaptation mechanism from backbone, source initialization, stream order, and evaluation confounders.

说人话就是：**IST 的“怎么从当前/过去数据产生训练信号”保留；会影响公平比较的网络、source、数据流、训练底座、更新范围和评测全部和 SHOT 对齐。**

## 11. P0 freeze boundary

从 P1 起，除非发现可复现 correctness bug，否则不再修改：

```text
same source F/B/C
same R50/R101 architecture
same seed-2026 outer stream
Office BS64 / VisDA BS256
one pass
same PU / FO semantics
same FC / Conv candidate scopes
common SHOT optimizer/scheduler substrate
IST extend=8, iters=1, repeat=1
PLCA K=50, gamma=3, mode=l2
memory max_len=10000
IST hard CE + soft KL
IST outer parameter moving average m=0.9
P1 dense baseline trainable/BN scopes
```

任何修改都必须升级 protocol revision，并说明原 P0 contract 为什么失效。
