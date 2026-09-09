# OTTA_IST_BASELINE_PROTOCOL_20260906_v1

**状态：** 正式冻结的 IST baseline protocol  
**Protocol revision：** `OTTA_IST_BASELINE_PROTOCOL_20260906_v1`  
**Scientific implementation revision：** `ist_otta_p1_baseline_20260906_v4`  
**Parent contract：** `IST_OTTA_P0_REFACTOR_CONTRACT_20260902_v1`  
**Source checkpoint revision：** `nips2026_shot_otta_uda_source_v1`  
**日期：** 2026-09-06  
**范围：** 在当前 controlled OTTA experimental substrate 上实现 IST；本文件只定义 dense IST references  
**数据集：** Office-31、VisDA-C

> 本文件从现在起作为 **IST formal baseline 的规范来源**。  
> P0 文档保留为历史设计 / refactor contract，不再继续修改其语义。

---

## 1. 目的与冻结边界

本 protocol 冻结 ICLR 2027 主线中的 dense IST baseline。

这里的目标不是复现官方 IST 的整套 benchmark / model stack，而是：

> **保留 IST 的核心 adaptation mechanism，同时把它放到与当前 refined SHOT-OTTA 完全相同的 source / network / stream / optimizer substrate / evaluation framework 上。**

这样可以尽量消除以下混杂变量：

```text
source training
backbone
target order
batch size
candidate scope
evaluation protocol
```

正式 dense IST variants 只有：

```text
ist_full_dense
ist_fc_module_dense
ist_conv_module_dense
```

本文件不定义：

```text
Random
Magnitude
Saliency
LBI
Group-LBI
```

这些统一由：

```text
OTTA_IST_LBI_PROTOCOL_20260906_v1
```

定义。

---

## 2. Source-of-truth 优先级

发生冲突时，优先级固定为：

1. `OTTA_IST_BASELINE_PROTOCOL_20260906_v1`
2. 当前 `260817_iclr2027-refined`，implementation revision 为 `ist_otta_p1_baseline_20260906_v4`
3. `OTTA_FC_LBI_PROTOCOL_20260817_v1` 与当前 refined Conv track 中与 benchmark/evaluation 相关的冻结语义
4. 官方 IST：`https://github.com/JingInAI/IST4TTA`
5. `IST_OTTA_P0_REFACTOR_CONTRACT_20260902_v1`
6. 旧 `nips2026/IST-OTTA/`
7. 旧 `nips2026/IST-OTTA-LBI/`

旧 IST / IST-LBI 只能作为历史参考，不能覆盖本 protocol。

任何会改变科学语义的修改，都必须升级 protocol revision。

---

## 3. 从官方 IST 保留的机制

以下 IST-specific mechanism 正式冻结：

| IST mechanism | 正式值 |
|---|---|
| Multi-view extension | `extend=8` |
| Inner iterations | `iters=1` |
| PLCA repeat | `1` |
| PLCA K | `50` |
| PLCA gamma | `3` |
| PLCA mode | `l2` |
| PLCA propagation alpha | `0.99` |
| PLCA solver max steps | `20` |
| PLCA solver tolerance | relative `1e-6` |
| Memory max length | `10000` |
| Training target | corrected hard PL + pre-correction soft target |
| IST outer parameter moving average | `m=0.9`，每个有效 outer batch 一次 |

也就是说，当前 IST 仍然保留：

```text
8-view self-training
robust PLCA
causal memory
hard + soft supervision
outer-batch parameter moving average
```

但 network / source / stream / optimizer substrate / evaluation 改为当前 controlled study 的统一定义。

---

## 4. 相对官方 IST 的 controlled-study 改动

| 项目 | 当前 formal IST | 原因 |
|---|---|---|
| Network | SHOT `netF -> netB -> netC` | 消除网络/head混杂 |
| Source model | 与 SHOT 完全相同的 source F/B/C | 消除初始化混杂 |
| Office backbone | ResNet-50 | 统一 benchmark |
| VisDA backbone | ResNet-101 | 统一 benchmark |
| Bottleneck | `Linear(2048,256)+BN` | 统一 FC candidate |
| Classifier | weight-normalized `netC`，始终 frozen | 统一 source-hypothesis semantics |
| Target stream | seed=2026 fixed order | 统一 sample order / batch boundaries |
| Outer BS | Office 64 / VisDA 256 | 消除 batch-size 混杂 |
| Target pass | 1 | formal OTTA |
| Optimizer | 当前 SHOT controlled SGD substrate | optimizer recipe 不是本次比较变量 |
| LR schedule | 当前 one-pass polynomial schedule | 统一 optimization timeline |
| Evaluation | PU + FO | 统一 plasticity / stability 口径 |
| Artifact | refined provenance pipeline | 可追溯、可复现 |

因此论文中不要写：

> exact reproduction of the official IST benchmark setup

更准确的表述是：

> We retain IST's multi-view self-training, PLCA-based pseudo-label correction, causal memory, hard/soft supervision and outer-batch parameter moving average, while transplanting them onto the same frozen F/B/C source model, target stream, optimization substrate and PU/FO evaluation protocol used by our SHOT-OTTA study.

---

## 5. Benchmark 与 source invariants

### 5.1 Office-31

| Item | Value |
|---|---|
| Dataset | Office-31 |
| Classes | 31 |
| Backbone | ResNet-50 |
| Outer batch size | 64 |
| Workers | 4 |
| Target passes | 1 |
| DataLoader `drop_last` | `false` |
| Formal seed | 2026 |
| Primary PU/FO metric | sample-level overall accuracy |
| Dataset aggregation | 6 个 directed transfers 等权平均 |

正式 transfers：

```text
A -> D
A -> W
D -> A
D -> W
W -> A
W -> D
```

### 5.2 VisDA-C

| Item | Value |
|---|---|
| Dataset | VisDA-C |
| Classes | 12 |
| Backbone | ResNet-101 |
| Outer batch size | 256 |
| Workers | 4 |
| Target passes | 1 |
| DataLoader `drop_last` | `false` |
| Formal seed | 2026 |
| Primary PU/FO metric | fixed-12-class mean per-class accuracy |

正式 transfer：

```text
Synthetic / Train -> Real / Validation
```

同时保留当前 refined pipeline 中的：

```text
overall accuracy
class-wise accuracy
worst-class accuracy / class name
class-wise standard deviation
```

### 5.3 Source checkpoint identity

IST 必须从与 SHOT formal 完全相同的 F/B/C 开始：

```text
source_checkpoint_revision = nips2026_shot_otta_uda_source_v1
```

Office：

```text
source=A:
  uda/office/A/source_F.pt
  uda/office/A/source_B.pt
  uda/office/A/source_C.pt

source=D:
  uda/office/D/source_F.pt
  uda/office/D/source_B.pt
  uda/office/D/source_C.pt

source=W:
  uda/office/W/source_F.pt
  uda/office/W/source_B.pt
  uda/office/W/source_C.pt
```

VisDA：

```text
uda/VISDA-C/T/source_F.pt
uda/VISDA-C/T/source_B.pt
uda/VISDA-C/T/source_C.pt
```

每个 formal run 必须记录：

```text
resolved source F/B/C paths
source F/B/C SHA256
source_checkpoint_revision
```

禁止为 IST 重新训练单独 source。

---

## 6. 三个 formal dense variants

### 6.1 `ist_full_dense`

Persistent trainable scope：

```text
netF
netB
```

Frozen：

```text
netC
```

BN：

```text
native/full train behavior during IST inner adaptation
```

它是 controlled substrate 上的 IST native/full reference。

### 6.2 `ist_fc_module_dense`

只允许：

```text
netB.bottleneck.weight
netB.bottleneck.bias
```

发生 persistent update。

其余 parameters 全部 frozen。

所有 BN parameters 与 running buffers frozen。

`netC` frozen。

它是后续 FC sparse family 的 matched dense reference。

### 6.3 `ist_conv_module_dense`

只允许 formal `netF.layer4` 9 个 Conv weights：

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

发生 persistent update。

其他 parameters 全部 frozen。

所有 BN parameters 与 running buffers frozen。

`netC` frozen。

它是后续 Conv out-channel sparse family 的 matched dense reference。

---

## 7. Target stream 与 singleton compatibility rule

固定：

```text
(dataset, transfer, seed=2026)
```

后，所有 variants 必须看到完全相同的：

```text
target sample identity
sample order
outer-batch partition
```

DataLoader：

```text
drop_last = false
```

但为了与已经完成的历史 SHOT formal trajectory 保持一致，额外冻结：

```text
if actual outer batch size == 1:
    在任何 adaptation/evaluation state transition 前直接 skip
```

被 skip 的 singleton 不得触发：

```text
IST view materialization
PLCA
memory commit
inner optimization
EMA
PU
```

FO 仍然是 full-target read-only evaluation，因此 FO 仍包含全部 target samples。

Artifact 中必须记录：

```text
singleton_outer_batch_policy = skip_size_1_to_match_shot_history
singleton_outer_batches_skipped
processed_outer_batches
```

不能把这条规则偷偷改成：

```text
drop_last=true
```

因为 loader / stream manifest 语义仍然是 `drop_last=false`。

---

## 8. Raw-image multi-view construction

Outer stream 必须提供 raw image identity/path，而不是已经 Normalize 的 tensor。

对每个有效 outer batch sample：

1. raw RGB image 只 decode 一次；
2. 生成 1 个 cached PU reference view；
3. 从同一个 raw image 独立生成 `extend=8` 个 IST adaptation views。

禁止旧路径：

```text
normalized tensor
-> ToPILImage
-> IST augmentation
```

正式 transform：

```text
Resize(256)
RandomCrop(224)
RandomHorizontalFlip(p=0.5)
ToTensor
ImageNet normalization
```

ImageNet normalization：

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

PU reference view 与 8 个 IST adaptation views 使用独立 deterministic RNG streams。

当前固定 RNG derivation：

```text
formal seed                 = 2026
reference view seed offset  = 10000
adaptation view seed offset = 20000
inner-order seed offset     = 30000
```

跨 SHOT / IST 必须一致的是：

```text
sample identity
sample order
outer-batch membership
```

不是 augmentation pixels 必须一致。

---

## 9. Pre-adaptation prediction、PLCA 与 causal memory

设当前有效 outer batch 为：

$$
B_t,
$$

进入该 batch 前 persistent model 为：

$$
\theta_t,
$$

历史 memory 为：

$$
\mathcal M_{t-1}.
$$

8-view adaptation set 记作：

$$
X_t^{(8)}.
$$

### 9.1 Pre-adaptation outputs

只使用 pre-adaptation model：

$$
\theta_t
$$

在 eval behavior 下计算：

$$
F_t=f_{\theta_t}(X_t^{(8)}),
$$

以及：

$$
Q_t=\operatorname{softmax}(h_{\theta_t}(X_t^{(8)})).
$$

该 forward 不得产生 persistent parameter / BN update。

### 9.2 PLCA

PLCA 只能使用：

$$
(F_t,Q_t,\mathcal M_{t-1}).
$$

禁止 future-target access。

Frozen PLCA：

```text
repeat = 1
K = 50
gamma = 3
mode = l2
propagation_alpha = 0.99
solver_max_steps = 20
solver_rtol = 1e-6
solver_atol = 0
```

L2 graph 保留官方 IST 的 `K+2` neighbor convention（在有效邻居数允许时）。

当前 views 的 corrected hard target 记为：

$$
\hat Y_t.
$$

### 9.3 Causal memory commit

顺序固定为：

$$
(B_t,\mathcal M_{t-1})
\rightarrow
(F_t,Q_t)
\rightarrow
\hat Y_t
\rightarrow
\mathcal M_t.
$$

Memory 每个 processed outer batch 只 commit 一次。

Memory 存：

```text
detached current features
corrected pseudo-label distributions
```

最大长度：

```text
10000
```

超过容量后保留最近 entries。

batch $t$ 能看到的 memory 必须只来自此前已经 processed 的 outer batches。

---

## 10. IST objective

对于 adaptation view $x_i$：

- $\hat y_i$：PLCA corrected hard target；
- $q_i$：PLCA 前保存的 soft prediction；
- $p_\theta(x_i)$：当前模型 self-training 时的 prediction。

正式 IST loss：

$$
\mathcal L_{\mathrm{IST}}
=
\mathcal L_{\mathrm{hard}}
+
\mathcal L_{\mathrm{soft}},
$$

其中：

$$
\mathcal L_{\mathrm{hard}}
=
\operatorname{CE}
\left(
p_\theta(x_i),\hat y_i
\right),
$$

以及：

$$
\mathcal L_{\mathrm{soft}}
=
D_{\mathrm{KL}}
\left(
q_i
\;\|\;
p_\theta(x_i)
\right).
$$

Frozen weights：

```text
hard_ce_weight = 1.0
soft_kl_weight = 1.0
```

禁止混入：

```text
SHOT entropy loss
SHOT diversity loss
SHOT current-batch argmax pseudo objective
```

Target ground-truth labels 不得进入 adaptation path。

---

## 11. Native IST inner self-training

`iters=1` 的正式含义：

> 对当前 outer batch 产生的完整 `8|B_t|` adaptation-view set 遍历一次。

这套临时 view set 可以：

```text
shuffle
split into inner mini-batches
```

inner mini-batch capacity 与正式 outer batch size 相同：

```text
Office = 64
VisDA  = 256
```

inner mini-batch：

```text
drop_last = false
```

同一 outer batch 的所有 inner mini-batches 共享同一个 outer-batch LR。

inner mini-batches 之间：

```text
不得重新跑 PLCA
不得重新 commit memory
```

---

## 12. Optimizer 与 outer-batch LR schedule

### 12.1 Office

```text
base LR = 0.01
netF multiplier = 0.1
netB multiplier = 1.0
```

### 12.2 VisDA-C

```text
base LR = 0.001
netF multiplier = 0.1
netB multiplier = 1.0
```

### 12.3 Common SGD

```text
optimizer = SGD
momentum = 0.9
weight_decay = 0.001
nesterov = true
```

Formal one-pass polynomial schedule：

$$
\eta_t
=
\eta_0
\left(
1+10\frac{t}{T}
\right)^{-0.75}.
$$

这里的：

$$
t
$$

沿 outer target-stream timeline 推进，而不是按 IST inner mini-batch 推进。

同一 processed outer batch 中所有 optimizer steps 共用同一个：

$$
\eta_t.
$$

`extend=8` 不得让 LR timeline 快 8 倍。

---

## 13. Native IST outer EMA

三个 dense baseline 都使用 IST native parameter moving average。

每个 processed outer batch 开始时，保存独立 cloned pre-batch trainable state：

$$
\theta_t.
$$

inner self-training 完成得到：

$$
\widetilde\theta_t.
$$

然后只做一次：

$$
\theta_{t+1}
=
0.9\theta_t
+
0.1\widetilde\theta_t.
$$

要求：

```text
anchor 必须 detach().clone()
禁止 shallow state_dict aliasing
每个 processed outer batch EMA 恰好一次
frozen parameters 不得被平均成新值
BN buffers 按各 variant 的 BN semantics 处理
PU / FO / PLCA / memory 都不能触发 EMA
```

这条 native EMA 适用于：

```text
IST dense
后续非-LBI Random
Magnitude
Saliency
```

不适用于 IST-LBI。

IST-LBI 的唯一 persistent writeback 由 LBI protocol 定义。

---

## 14. 每个 processed outer batch 的精确顺序

对每个有效、非-singleton 的：

$$
B_t
$$

固定执行：

```text
1. 按 seed=2026 frozen stream 取得 raw sample identities
2. decode raw images
3. materialize cached PU reference views
4. 每张 raw image materialize 8 个 IST adaptation views
5. clone pre-batch model state，作为 IST EMA anchor
6. 用 theta_t 做 pre-adaptation eval forward
7. 得到 F_t 与 pre-correction soft targets Q_t
8. 读取 causal memory snapshot M_{t-1}
9. robust PLCA 一次
10. 得到 corrected hard targets
11. memory commit 一次
12. 设置当前 outer-batch LR
13. native IST iters=1，遍历完整临时 adaptation set
14. IST EMA m=0.9 恰好一次
15. 用 cached reference views 做 PU，完全 read-only
16. 丢弃当前 temp views / targets
17. 进入下一个 outer batch
```

完整 target stream 后：

```text
freeze final model/state
-> full target evaluation
-> FO read-only
```

---

## 15. PU 与 FO

### 15.1 PU

PU 定义为：

> same current batch，post-update，read-only evaluation。

即：

$$
\theta_t
\rightarrow
\text{IST adaptation}
\rightarrow
\theta_{t+1}
\rightarrow
\text{PU on cached reference view of }B_t.
$$

PU 不得修改：

```text
parameters
BN buffers
memory
optimizer state
EMA anchor/state
```

被 skip 的 singleton 没有 PU record。

### 15.2 FO

one-pass stream 完成后冻结 final persistent model。

FO 对完整 target evaluation set 做 read-only evaluation。

FO 期间禁止：

```text
adaptation
memory update
optimizer step
EMA
```

---

## 16. Target-label boundary

Target labels 只允许用于：

```text
PU metric
FO metric
final offline reporting / analysis
```

禁止进入：

```text
multi-view adaptation objects
PLCA
memory
hard/soft training targets
optimizer updates
EMA
method-state decisions
```

若未来要使用 labeled validation 做 hyperparameter selection，必须先写独立、预注册的 search/validation protocol。

---

## 17. Formal baseline experiment matrix

三个 dense IST variants 都是 non-budgeted。

不能按 sparse budget 重复运行。

### Office-31

```text
6 transfers x 3 dense IST variants = 18 formal runs
```

### VisDA-C

```text
1 transfer x 3 dense IST variants = 3 formal runs
```

总计：

$$
18+3=21.
$$

`source_only` 是共享 source reference，不属于 IST adaptation variant。

如果 scientific identity 完全一致，可以复用现有 source-only artifact。

---

## 18. Formal execution restrictions

Formal dense baseline 必须：

```text
formal_protocol = true
save_model = false
debug_max_outer_batches = None
```

`--debug-max-outer-batches` 只允许 smoke/debug。

formal run 设置该参数必须直接报错。

P1 dense baseline 不定义 partial resume / stream checkpoint。

中断后：

```text
rerun
```

而不是 partial resume。

禁止根据 final formal accuracy 反向调 baseline。

---

## 19. Scientific identity 与 provenance

至少记录：

```text
protocol_revision
implementation_revision
source_checkpoint_revision
source checkpoint resolved paths
source checkpoint SHA256
method = IST
variant
dataset
source/target transfer
backbone
formal seed
outer batch size
target stream identity/hash
singleton policy
extend
iters
PLCA settings
memory max_len
optimizer settings
outer LR schedule
EMA momentum
candidate/update scope
BN semantics
```

Formal dense IST 的 protocol revision：

```text
OTTA_IST_BASELINE_PROTOCOL_20260906_v1
```

Smoke/debug metadata 不能被当作 formal scientific result。

---

## 20. P1 validation status

Protocol freeze 前，当前 implementation 已通过：

```text
P1 contract suite: PASS
```

Office D->A short smoke：

```text
ist_full_dense        PASS
ist_fc_module_dense   PASS
ist_conv_module_dense PASS
```

每个 variant：

```text
processed valid outer batches = 5
```

Office D->A 最终 memory：

$$
5\times64\times8=2560.
$$

与预期完全一致。

该 smoke 只做 correctness validation，不作为正式 accuracy result。

---

## 21. 正式冻结项

除非发现可复现 correctness bug，否则以下全部冻结：

```text
same source F/B/C as SHOT
source revision nips2026_shot_otta_uda_source_v1
R50 Office / R101 VisDA
F -> B -> C architecture
netC frozen
seed-2026 fixed outer stream
Office BS64 / VisDA BS256
workers=4
one target pass
drop_last=false + explicit size-1 singleton skip
raw-image multi-view path
extend=8
iters=1
PLCA repeat=1, K=50, gamma=3, mode=l2
PLCA alpha=0.99, solver max=20, relative tol=1e-6
causal memory max_len=10000
hard CE + soft KL，weight=1/1
common controlled SGD substrate
outer-batch polynomial LR timeline
native IST EMA m=0.9 once per processed outer batch
three dense update/BN scopes
PU/FO read-only semantics
target-label boundary
baseline non-resume policy
formal debug-limit rejection
```

任何科学语义变化必须升级 protocol revision。

---

## 22. 一页式冻结摘要

| Category | Frozen rule |
|---|---|
| Protocol | `OTTA_IST_BASELINE_PROTOCOL_20260906_v1` |
| Implementation | `ist_otta_p1_baseline_20260906_v4` |
| Source revision | `nips2026_shot_otta_uda_source_v1` |
| Formal seed | 2026 |
| Office backbone / BS | R50 / 64 |
| VisDA backbone / BS | R101 / 256 |
| Workers | 4 |
| Target pass | 1 |
| Singleton | actual BS=1 -> skip before adaptation/PU |
| Multi-view | raw-image `extend=8` |
| IST iterations | `iters=1` |
| PLCA | repeat1 / K50 / gamma3 / l2 |
| PLCA propagation | 0.99 |
| PLCA solver | max20 / rtol1e-6 |
| Memory | causal FIFO，max 10000 |
| Loss | hard CE + soft KL |
| Office base LR | 0.01 |
| VisDA base LR | 0.001 |
| SGD | momentum .9 / wd .001 / Nesterov |
| Scheduler | outer-batch polynomial |
| IST native EMA | m=.9，每 processed outer batch 一次 |
| Full dense | netF+netB；netC frozen |
| FC module dense | bottleneck weight+bias；all BN frozen |
| Conv module dense | layer4 9 Conv weights；all BN frozen |
| PU | cached same-batch post-update，read-only |
| FO | full-target final frozen read-only |
| Formal dense runs | 21 |
| Sparse/LBI | 单独 protocol 定义 |

---

**End of protocol.**
