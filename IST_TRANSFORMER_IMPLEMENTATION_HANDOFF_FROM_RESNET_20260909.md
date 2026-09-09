# IST-OTTA Transformer 版本实现交接说明
## ——基于当前 ResNet IST 实现的迁移逻辑、工程边界与高风险坑点

**项目：** `260817_iclr2027-refined`  
**用途：** 给已完成 SHOT-OTTA Transformer 版本的同学，实现对应的 IST-OTTA Transformer 版本  
**当前参考实现：**
- ResNet IST baseline revision：`ist_otta_p1_baseline_20260906_v4`
- ResNet IST sparse/LBI revision：`ist_otta_sparse_lbi_20260906_v2`
- 当前 LBI 核心：共享 `core/lbi/**`
- 当前 ResNet IST protocol：
  - `protocol/ist-otta/OTTA_IST_BASELINE_PROTOCOL_20260906_v1.md`
  - `protocol/ist-otta/OTTA_IST_LBI_PROTOCOL_20260906_v1.md`

---

# 0. 一句话任务定义

这次不是“把官方 IST repo 原样套到 Transformer”。

正确任务是：

> **把 IST 的 method-specific mechanism 接到当前已经实现好的 SHOT-OTTA Transformer controlled substrate 上，同时保持 SHOT-Transformer 的 source model、target stream、evaluation、candidate scope、sparse/LBI 公平比较语义不被改动。**

也就是说：

```text
已有 SHOT-Transformer
        +
IST-specific mechanism
        +
共享 sparse/LBI machinery
        =
IST-Transformer
```

不要重新造一套 Transformer benchmark，也不要为了 IST 去改已经能跑的 SHOT-Transformer scientific implementation。

---

# 1. Source of Truth：优先级必须先定死

如果出现实现冲突，按下面顺序处理：

```text
1. 当前 refined project / 已冻结 Transformer SHOT substrate
   - source checkpoint
   - model architecture
   - target stream
   - seed / batch
   - PU / FO
   - candidate scope
   - normalization semantics
   - sparse budget
   - corrected LBI semantics
   - artifact / provenance

2. 官方 IST
   - multi-view augmentation
   - PLCA
   - causal past-only memory
   - hard pseudo-label correction
   - pre-correction soft target
   - CE + KL self-training
   - native parameter moving average

3. 当前 ResNet IST
   - 作为“IST 如何接入 refined substrate”的直接实现参考

4. 历史旧 IST-OTTA / IST-OTTA-LBI
   - 仅用于查历史
   - 禁止直接照搬
```

**最重要的原则：**

> Transformer 版应该继承“当前 Transformer SHOT 的实验底座”，而不是继承 ResNet 的结构名字。

例如 ResNet 里有：

```text
netF / netB / netC
bottleneck FC
layer4 Conv
BN
```

Transformer 未必有完全同构的结构。

这些不能机械照搬。

---

# 2. 当前 ResNet IST 到底改了什么

当前 ResNet IST 的核心不是换了一个 loss 就结束，而是加了一整套 **outer-batch state machine**。

原来简单的 OTTA 可以想成：

```text
batch
-> loss
-> backward
-> optimizer.step()
-> prediction
```

IST 实际是：

```text
raw outer batch
-> 构造 reference view + 8 adaptation views
-> pre-adaptation feature / probability
-> PLCA pseudo-label correction
-> causal memory commit
-> 固定当前 outer-batch 的 hard / soft targets
-> IST self-training
-> persistent parameter writeback
-> PU read-only evaluation
```

数学上可以写成：

$$
(B_t,\mathcal M_{t-1})
\rightarrow
(F_t,Q_t)
\rightarrow
\hat Y_t
\rightarrow
\mathcal M_t
\rightarrow
\text{self-training}
\rightarrow
\theta_{t+1}.
$$

这里最关键的是：

> **pseudo-label、memory、adaptation、persistent writeback 的时序是科学语义，不是随便调整的工程细节。**

---

# 3. 当前 ResNet IST 保留的 method-specific 机制

当前实现里真正来自 IST 的核心机制包括：

```text
extend = 8
PLCA repeat = 1
K = 50
gamma = 3
mode = l2
propagation alpha = .99

PLCA solver:
maxiter = 20
rtol = 1e-6
atol = 0

causal memory:
past-only
max_len = 10000

hard target:
corrected pseudo-label

soft target:
pre-correction soft probability

native IST:
iters = 1

native parameter moving average:
m = .9
```

IST objective：

$$
\mathcal L_{\rm IST}
=
\mathrm{CE}(p_\theta(x_i),\hat y_i)
+
D_{\rm KL}(q_i\|p_\theta(x_i)).
$$

其中：

- $\hat y_i$：PLCA 后的 corrected hard pseudo-label；
- $q_i$：correction 前的 soft target；
- 两者在当前 outer batch 内冻结；
- model prediction 随参数变化重新计算。

---

# 4. 当前 ResNet IST 的完整 outer-batch 顺序

对一个有效 incoming outer batch $B_t$：

## Step 0：singleton 先判断

当前 refined 历史语义：

```text
actual outer batch size == 1
-> skip
```

而且要在以下所有操作之前 skip：

```text
augmentation
PLCA
memory
selector
LBI
optimizer
EMA
PU
```

但是：

```text
FO 仍然评估完整 target set
```

这个语义不能因为 Transformer 更容易处理 batch=1 就顺手改掉。

---

## Step 1：raw image 只 decode 一次

同一个 raw sample：

```text
decode once
-> cached reference view
-> 8 adaptation views
```

不要：

```text
reference decode 一次
8 views 又分别重复 decode
```

Transformer 版尤其不要因为数据 pipeline 不同而偷偷改变这个语义。

---

## Step 2：pre-adaptation inference

在 adaptation 之前，只做一次当前 batch 的 pre-adaptation inference，得到：

```text
F_t = features
Q_t = soft predictions
```

这里的 Transformer 对应关系应该是：

```text
F_t:
使用当前 SHOT-Transformer 已经定义/验证的“分类前 feature representation”

Q_t:
当前 classifier 输出的 soft prediction
```

**不要因为 ResNet 叫 `netB` 就强行在 Transformer 里造 bottleneck。**

如果 Transformer SHOT 已经有 feature extraction API，直接复用。

---

## Step 3：PLCA exactly once

每个有效 outer batch：

```text
PLCA calls = 1
```

不要：

```text
每个 view 一次
每个 inner mini-batch 一次
每个 optimizer step 一次
```

PLCA 使用：

```text
current batch pre-adaptation feature/probability
+
past-only memory
```

输出 corrected hard pseudo-label。

---

## Step 4：memory commit exactly once

每个有效 outer batch：

```text
memory commit = 1
```

当前 memory 是 causal：

```text
M_{t-1}
-> process B_t
-> commit
-> M_t
```

严禁：

```text
future sample leakage
重复 commit 当前 batch
inner loop 每轮重新写 memory
```

---

## Step 5：固定当前 outer-batch adaptation task

当前 ResNet IST 会构造：

$$
\mathcal D_t^{IST}
=
\{(x_i,\hat y_i,q_i)\}_{i=1}^{N_t},
$$

其中：

$$
N_t=8|B_t|.
$$

也就是说：

```text
一个 raw sample
-> 8 adaptation views
-> hard corrected label 重复到这些 views
-> pre-correction soft target 重复到这些 views
```

**非常重要：**

当前 outer batch 一旦构造好：

```text
views      fixed
hard target fixed
soft target fixed
```

后面的 native IST / Saliency / LBI 都不能重新跑 PLCA 去改变 target。

---

# 5. Dense IST 的优化逻辑

当前 ResNet dense / module-dense IST 保留 native IST self-training。

整体：

```text
fixed IST outer-batch task
-> native IST iters=1
-> optimizer updates
-> native EMA once
-> PU
```

注意：

> `iters=1` 不等于“整个 outer batch 只有一个 optimizer step”。

当前 ResNet IST 中，完整 8-view adaptation set 可以切成多个 inner mini-batches。

Native IST 的 `iters=1` 是：

```text
完整 adaptation set 遍历一轮
```

因此内部可以有多个 native optimizer steps。

这点后面和 LBI Stage-2 完全不同。

---

# 6. Native IST persistent writeback

非 LBI variant：

$$
\theta_{t+1}
=
0.9\theta_t
+
0.1\widetilde\theta_t.
$$

当前：

```text
Dense
Random
Magnitude
Saliency
```

都保留 native IST EMA / parameter moving average。

每个 processed outer batch：

```text
native EMA commit = exactly 1
```

---

# 7. PU / FO 语义

## PU

当前 batch adaptation 完成以后：

```text
cached reference view
-> post-update
-> read-only evaluation
```

PU 不能：

```text
再跑 optimizer
再更新 LN/BN state
再写 memory
再写 EMA
再改变 model
```

## FO

完整 target stream 结束后：

```text
freeze final model
-> full target read-only evaluation
```

FO 也不能引起任何 adaptation state mutation。

---

# 8. Transformer 版最核心的迁移原则

Transformer 版不要照着 ResNet 的类名/模块名搬。

应该分两层：

## Layer A：Transformer SHOT controlled substrate

直接继承学弟已经做好的 SHOT-Transformer：

```text
source checkpoint
model constructor
feature extraction
classifier
input resolution
target stream
seed
batch size
optimizer substrate
scheduler
evaluation
PU / FO
normalization semantics
candidate scopes
sparse budget
artifact identity
```

这些尽量不因 IST 改变。

---

## Layer B：插入 IST method-specific state machine

新增：

```text
8-view augmentation
PLCA
causal memory
hard pseudo-label correction
pre-correction soft target
CE + KL objective
native IST moving average
IST-specific counters / artifacts
```

目标是：

> **SHOT-Transformer 和 IST-Transformer 的差别主要来自 adaptation mechanism，而不是 source / model / stream / evaluation 被一起换了。**

---

# 9. Transformer 不要机械继承 ResNet 的 F/B/C 拆法

当前 ResNet source 是：

```text
F / B / C
```

但 Transformer 版可能是：

```text
backbone
classifier head
```

或者：

```text
patch embed
transformer blocks
norm
head
```

甚至 DeiT 可能还有：

```text
cls token
dist token
distillation head
```

所以：

> **Transformer IST 应复用当前 SHOT-Transformer 已经验证的 forward / feature / logits contract。**

不要为了“和 ResNet 接口一致”去：

```text
重构 source checkpoint
重新拆模型
额外插 bottleneck
改变 classifier
```

最重要的 contract 是：

```text
同一个 source checkpoint
同一个 raw input
在任何 adaptation 之前

SHOT-Transformer logits
==
IST-Transformer initial logits
```

建议做 exact / near-exact source-logit equality test。

---

# 10. Transformer feature 给 PLCA 时最容易踩的坑

PLCA 需要 feature。

Transformer 版必须先明确：

```text
“feature”到底取哪一层？
```

优先原则：

> **使用当前 SHOT-Transformer 已经定义和验证的分类前 representation。**

例如若现有 SHOT forward 已返回：

```text
features, logits
```

直接复用。

不要临时换成：

```text
logits
attention map
某一层 token 全展开
未定义的中间 hidden state
```

否则 IST 和 SHOT 实际使用了不同 representation contract。

同时确认：

```text
feature shape
feature normalization
batch dimension
cls/dist token semantics
```

在 ResNet→Transformer 后仍与 PLCA 实现兼容。

---

# 11. Transformer 特有坑：train/eval mode、Dropout、DropPath

这是 Transformer 版比 ResNet 更值得警惕的点。

很多 ViT / DeiT 存在：

```text
Dropout
DropPath / stochastic depth
attention dropout
MLP dropout
```

如果为了 adaptation 直接：

```python
model.train()
```

可能导致：

```text
同一 raw sample
同一参数
pre-adaptation feature 仍随机变化
```

进而影响：

```text
PLCA
Saliency
LBI support
PU
```

所以必须明确冻结：

```text
哪些模块 train mode
哪些 stochastic layers active/inactive
```

原则：

> **沿用当前 SHOT-Transformer 已验证的 mode semantics，不要 IST 自己重新定义。**

并增加测试：

```text
固定 seed / 固定 input
pre-adaptation feature/logits determinism contract
```

如果现有 SHOT-Transformer 本身允许 stochastic adaptation，也要保证 IST 与它使用同一 RNG discipline。

---

# 12. Transformer 特有坑：LayerNorm 不是 BatchNorm

ResNet controlled track 很多地方强调：

```text
BN frozen
```

Transformer 通常主要是：

```text
LayerNorm
```

不能看到 “BN frozen” 就以为 Transformer 没事了。

必须显式检查：

```text
LayerNorm weight/bias 是否属于 trainable scope？
LayerNorm 是否进入 optimizer？
candidate 外的 LayerNorm 是否被隐式更新？
full-dense 定义是否包含 LayerNorm？
module-dense / sparse track 是否排除 LayerNorm？
```

这些以 **已有 SHOT-Transformer controlled substrate** 为准。

不要为了 IST 私自改 normalization scope。

---

# 13. Transformer 特有坑：distillation head / token

如果当前 backbone 是带 distillation token/head 的 DeiT variant，必须显式回答：

```text
source inference 用哪个 head？
soft target Q_t 来自哪个 head？
PLCA probability 来自哪个 head？
adaptation loss 作用于哪个 head？
PU / FO 用哪个 head？
两个 head 是否平均？
distillation token 是否参与 feature？
```

如果当前 SHOT-Transformer 已经冻结了这套语义：

> 直接复用，IST 不重新解释。

如果当前用的是 non-distilled DeiT，则不要为了 IST 新增 distillation path。

---

# 14. 8-view 在 Transformer 上的显存问题

ResNet Office：

```text
B=64
8 views
-> 512 adaptation images
```

Transformer 每张图还会展开为 token sequence。

因此显存会比 ResNet 更容易爆。

允许：

```text
micro-batch / chunked forward-backward
```

但必须注意：

> chunking 只能是 memory implementation detail，不能改变数学 objective。

---

# 15. Full-objective gradient accumulation 必须数学等价

对完整 outer-batch adaptation set：

$$
\mathcal L
=
\frac1N\sum_{i=1}^N \ell_i.
$$

若 chunk $r$ 大小 $n_r$，每个 chunk 内 loss 是 mean，则：

$$
\nabla\mathcal L
=
\sum_r
\frac{n_r}{N}
\nabla\mathcal L_r.
$$

因此：

```text
zero_grad once
for each chunk:
    loss_chunk_mean * (n_r / N)
    backward
after all chunks:
    one full-gradient action
```

尤其最后一个 chunk 大小不等时，不能直接把各 chunk mean 等权相加。

---

# 16. LBI 的科学优化单位：outer batch，不是 inner chunk

当前 ResNet IST 已冻结：

```text
LBI support discovery = once per valid outer batch
```

即：

```text
64 raw samples
-> 512 views
-> 一个 fixed IST task
-> 找一次 support
```

不是：

```text
512 views 切成 8 个 mini-batch
-> 找 8 次 support
```

Transformer 因为显存更紧，更容易不小心把 micro-batch 当 scientific batch。

这是高风险错误。

必须一直记住：

> **outer batch 是 scientific adaptation unit；micro-batch 只是 implementation unit。**

---

# 17. LBI Stage-1：输入固定、prediction 重算

当前 fixed outer-batch task：

```text
views fixed
hard targets fixed
soft targets fixed
```

Stage-1 每一步：

```text
model parameters changed
-> predictions recomputed
-> full IST objective gradient recomputed
```

不能为了省 Transformer 算力缓存：

```text
old logits
old loss
old gradient
```

如果参数变了，prediction / gradient 必须重新算。

允许缓存的只是：

```text
input views
hard targets
soft targets
必要的 immutable metadata
```

---

# 18. LBI core 不要复制一份 Transformer 版

当前三个 host method：

```text
SHOT
IST
COME
```

共享 corrected LBI core。

Transformer IST 也应该继续复用：

```text
core/lbi/**
```

不要新建：

```text
transformer_lbi_engine_v2.py
ist_transformer_special_lbi.py
```

去复制 Stage-1 dynamics。

真正 method-specific 的应该只是：

```text
如何计算当前 fixed IST objective
如何把 full objective gradient 提供给 LBI engine
如何做 host state orchestration
```

LBI 本身继续共享：

```text
correct old-state Z
prox
threshold
strict integer budget
overshoot rollback
Stage2 masked-delta init
omega writeback
```

---

# 19. LBI Stage-2：仍然 exactly ONE step

这是 IST 最容易写错的一点。

Native IST：

```text
iters=1
```

可以意味着完整 adaptation set 遍历一轮，内部多个 optimizer steps。

但 refined LBI：

```text
Stage2 = exactly ONE masked SGD step
```

所以 Transformer 版千万不要写成：

```text
for each micro-batch:
    optimizer.step()
```

正确是：

```text
完整 fixed outer-batch objective
-> chunked gradient accumulation if needed
-> 得到 full gradient
-> exactly ONE masked SGD step
```

这是 LBI 跨 host method 的统一算法语义。

---

# 20. LBI 和 native IST EMA 绝对不能 double writeback

非 LBI：

```text
Dense / Random / Magnitude / Saliency
-> native IST EMA m=.9
```

LBI：

```text
native IST EMA = disabled
-> only LBI omega writeback
```

即：

$$
\theta_{t+1}
=
(1-\omega)\theta_t
+
\omega\widetilde\theta_t.
$$

错误写法：

```text
Stage2
-> omega writeback
-> 再 IST EMA
```

这是 double writeback，会完全改变 online trajectory。

建议代码层面直接禁止：

```python
ema = None if variant_is_lbi else OuterBatchEMA(...)
```

而不是靠调用者记忆。

---

# 21. Sparse baseline 不能套 LBI Stage-2

Random / Magnitude / Saliency 是 conventional sparse controls，不是 LBI。

因此当前 ResNet IST 语义：

```text
support
-> native IST self-training (iters=1)
-> native IST EMA
```

不是：

```text
support
-> LBI Stage2 one step
```

否则比较就不公平了。

---

# 22. Random 的正式语义

Random：

```text
global exact-K / exact-K_G
3 deterministic independent masks
whole target stream fixed
```

每个 child：

```text
fresh same source checkpoint
fresh memory
fresh optimizer
fresh EMA state
same target stream
same augmentation discipline
only support seed differs
```

当前正式 child seeds：

```text
202600
202601
202602
```

最后：

```text
aggregate PU / FO mean + std
```

---

# 23. Random 曾经出现过的真实 bug

历史第一次实现：

```text
metadata: num_random_masks=3
artifact: 3 masks
```

但实际只跑了：

```text
1 child trajectory
```

这是非常典型的科研工程 bug：

> metadata 写对了，不等于 execution 真跑对了。

Transformer 版 contract 必须验证：

```text
真实启动 3 个 fresh child executions
```

不要只检查 config field。

---

# 24. Magnitude

当前语义：

```text
score from SOURCE checkpoint
fixed whole stream
```

FC/scalar 类：

$$
s_j=|\theta_j^{src}|.
$$

结构化 group：

$$
s_g=\|W_g^{src}\|_2.
$$

Transformer 具体 scalar/group 定义：

> 复用当前 SHOT-Transformer 已冻结 candidate/grouping。

不要 IST 自己另定义一套。

---

# 25. Saliency

当前 ResNet IST：

```text
每个 valid outer batch
-> 当前 fixed IST full objective
-> 计算一次 saliency
-> 选 support
-> 整个 native IST inner traversal support 固定
-> 下一 outer batch 才重新选
```

scalar：

$$
s_j^{(t)}
=
|\theta_j^{(t)}g_j^{(t)}|.
$$

group：

$$
s_g^{(t)}
=
\|W_g^{(t)}\odot\nabla_{W_g}\mathcal L_t\|_2.
$$

Transformer 版不要：

```text
每个 micro-batch 重选 support
每个 native optimizer step 重选 support
```

---

# 26. Transformer candidate scope：不要从 ResNet 猜

当前 ResNet 有：

```text
FC bottleneck
Conv layer4 out-channel
```

Transformer 不应该直接找：

```text
“最像 layer4 的 block”
```

然后自己定义。

正确做法：

> **完全复用当前 SHOT-Transformer 已经实现/冻结的 sparse candidate scope、grouping 和 budget semantics。**

如果当前 SHOT-Transformer 只是 dense baseline，还没有冻结 sparse candidate：

```text
先停
-> 单独写 Transformer sparse/LBI proposal
-> 冻结 candidate/grouping/budget
-> 再实现 IST sparse/LBI
```

不要在 IST 实现时顺手发明 candidate。

---

# 27. Transformer parameter alias / shared tensor 风险

Transformer 里经常有：

```text
parameter list 更深
module naming 更复杂
可能存在 tied/shared parameter
```

构造 candidate 时必须验证：

```text
unique parameter identity
no duplicate tensor counted twice
candidate scalar/group count exact
budget K derived from exact candidate count
```

尤其不要仅靠字符串 prefix 模糊抓参数，导致：

```text
LayerNorm
head
pos_embed
cls_token
patch_embed
```

被意外纳入。

---

# 28. Positional embedding / cls token / patch embedding 不要偷更新

除非当前 SHOT-Transformer 的对应 dense/sparse scope明确包含，否则：

```text
pos_embed
cls_token
dist_token
patch_embed
classifier head
LayerNorm
```

都不能因为 `model.parameters()` / broad optimizer group 被偷偷更新。

做 scope test：

```text
candidate params changed?
expected YES

all off-scope params bit-exact?
expected YES
```

---

# 29. AMP / mixed precision 不要临时引入

Transformer 显存紧时很容易想：

```text
那就给 IST-LBI 加 AMP
```

这可能改变：

```text
Stage-1 gradient scale
support crossing threshold timing
strict-budget overshoot point
```

所以原则：

> 如果当前 SHOT-Transformer substrate 没有冻结 AMP，就不要为了 IST-LBI 临时加。

如果现有 Transformer SHOT 已经统一使用 AMP，则需要单独验证：

```text
LBI Stage-1 numerics finite
support threshold semantics stable
gradient accumulation mathematically correct
```

不要静默改变 precision protocol。

---

# 30. Augmentation RNG 要和 stream identity 分开

当前 ResNet IST 使用 deterministic child RNG discipline。

已使用的 offset 思路：

```text
reference view RNG   : base + 10000
adaptation views RNG : base + 20000
inner-order RNG      : base + 30000
```

Transformer 版不一定要求相同代码，但必须保证：

```text
formal seed fixed
same scientific identity -> same views
Random support seed 与 augmentation seed 解耦
3 Random children 的 target stream / augmentation一致
only support differs
```

否则 Random comparison 会混进 augmentation variance。

---

# 31. Target label 泄漏：必须零容忍

Adaptation path 绝对不能看到：

```text
ground-truth target label
```

当前 ResNet IST 已经专门修过：

```text
target labels removed from adaptation objects
```

Transformer 版 Dataset / collate 很可能沿用原 SHOT dataloader 返回：

```python
image, label, index
```

如果 adaptation helper 接整个 batch object，非常容易把 label 带进去。

contract 必须验证：

```text
IST adaptation / PLCA / selector / LBI
不依赖 target label
```

label 只允许用于：

```text
PU/FO metric bookkeeping
```

且不能反馈到 adaptation。

---

# 32. Memory 不允许“看未来”

Transformer 版最容易在向量化时写成：

```text
先把整条 stream features 算好
-> 再做 memory/PLCA
```

这会产生 future leakage。

必须保持 online causal：

```text
M_{t-1}
+ current B_t
-> PLCA
-> commit M_t
-> next B_{t+1}
```

绝不能：

```text
future batch feature
```

进入当前 PLCA。

---

# 33. Provenance / scientific identity 是正式实验的一部分

当前 ResNet IST 曾经出现：

```text
sparse implementation
```

绕过正常 resolver 后，revision fallback 错落到 dense revision。

主训练不一定马上出错，但后期整理 artifact 会乱。

Transformer 版 scientific identity 至少应该明确：

```text
host method = IST
architecture / backbone
implementation revision
protocol revision
source checkpoint revision
dataset / transfer
seed
variant
candidate scope
grouping
budget
selector
LBI tuple
writeback mode
```

不要只保证：

```text
train.py 主路径能跑
```

还要保证：

```text
resume
aggregate
finalize
manual builder
```

不会把 identity 写错。

---

# 34. Contract tests：必须测次数和状态，不只测 shape

最低限度建议覆盖以下 contracts。

## A. Source equality

```text
same Transformer source checkpoint
SHOT initial logits == IST initial logits
```

在任何 adaptation 之前成立。

---

## B. Data / augmentation

```text
raw decode once
reference view exactly 1
adaptation views exactly 8
same raw sample
deterministic RNG contract
```

---

## C. PLCA / memory

每 valid outer batch：

```text
PLCA calls = 1
memory commits = 1
```

不能随着：

```text
8 views
micro-batches
optimizer steps
```

增加。

---

## D. Singleton

```text
outer batch size = 1
-> no augmentation
-> no PLCA
-> no memory
-> no optimizer
-> no EMA/omega
-> no PU
```

FO 仍 full target。

---

## E. Dense scope

分别验证：

```text
full dense
module dense / Transformer对应 controlled dense track
```

确保：

```text
expected params change
off-scope exact
classifier policy exact
LayerNorm policy exact
```

---

## F. Native IST

每 processed outer batch：

```text
native EMA commits = 1
```

---

## G. Sparse

Random：

```text
real 3-child execution
exact K / K_G
fresh source/memory/optimizer each child
```

Magnitude：

```text
source-static support
```

Saliency：

```text
1 support selection / outer batch
not per micro-batch
```

---

## H. LBI

每 valid outer batch：

```text
LBI discovery = 1
```

验证：

```text
correct old-state Z
strict integer budget
overshoot rollback
stage1 cap
support threshold
full-objective accumulation
Stage2 exactly 1 optimizer step
off-mask exact
native EMA commits = 0
omega writeback = 1
no double writeback
```

---

## I. Evaluation state

PU / FO：

```text
read-only
```

验证 evaluation 前后：

```text
model params unchanged
memory unchanged
optimizer unchanged
EMA state unchanged
normalization state unchanged
```

---

## J. SHOT regression

实现 IST-Transformer 后：

```text
SHOT-Transformer existing tests / smoke
must remain unchanged and PASS
```

不要因为抽 shared helper 让原 SHOT scientific semantics 漂掉。

---

# 35. 建议的实现顺序

不要一次写完所有 variant。

## P0：先审当前 SHOT-Transformer

整理：

```text
source model
feature/logit API
trainable scope
normalization
optimizer/scheduler
stream
PU/FO
singleton
candidate/grouping（如果已有）
```

输出一个短 design note。

---

## P1：只实现 IST dense baseline

先只做：

```text
ist_transformer_full_dense
```

以及当前 Transformer SHOT 已经定义好的 matched-module dense variant(s)。

不要先碰：

```text
Random
Magnitude
Saliency
LBI
```

先证明：

```text
IST method state machine 在 Transformer 上正确
```

---

## P2：Dense contracts + 5-batch smoke

推荐：

```text
Office D->A
seed=2026
5 valid outer batches
```

只看：

```text
loss finite
PLCA/memory counts
EMA count
scope
PU read-only
no label leakage
```

**不要看 smoke accuracy 调算法。**

---

## P3：再接 sparse baselines

```text
Random
Magnitude
Saliency
```

candidate/grouping 复用 SHOT-Transformer。

---

## P4：再接 LBI adapter

只写：

```text
IST objective adapter
full-objective accumulation
host state orchestration
```

继续复用：

```text
core/lbi/**
```

---

## P5：contracts + independent code review

重点人工看：

```text
outer loop
micro-batch loop
PLCA/memory timing
EMA/omega
candidate builder
Random 3-child
Saliency selection timing
LBI Stage2 step count
LayerNorm scope
train/eval mode
provenance fallback
```

---

## P6：short smoke

全部 variant：

```text
real GPU
real data
5 valid outer batches
```

验证 correctness。

---

## P7：STOP

代码正确以后先停。

不要边写代码边 tune。

Transformer 的参数搜索应等 protocol / search discipline 单独冻结以后再开始。

---

# 36. 推荐代码结构

不要求必须完全照这个命名，但推荐职责分离：

```text
ist_otta/
    mechanism.py
        - PLCA
        - memory
        - target construction

    trainer.py
        - ResNet adapter
        - outer-batch state machine

    transformer_trainer.py   # 或 architecture adapter
        - Transformer forward/feature glue
        - Transformer scope glue
        - 复用同一 IST mechanism

    lbi.py
        - IST objective adapter
        - 复用 core/lbi engine
```

更理想的是：

```text
IST mechanism
        ↑
architecture adapter
   /            \
ResNet        Transformer
```

而不是复制：

```text
ist_resnet_everything.py
ist_transformer_everything_copy.py
```

避免两套 PLCA / memory / target logic 后续漂移。

---

# 37. 哪些逻辑应该共享，哪些必须 architecture-specific

## 应共享

```text
PLCA math
memory semantics
hard/soft target construction
CE + KL loss definition
outer-batch state machine
singleton policy
Random child orchestration
LBI core
Stage1/Stage2 semantics
EMA vs omega rule
PU/FO mutation boundary
artifact schema
```

## architecture-specific

```text
source model loading
feature extraction
classifier forward
trainable scope
normalization policy
candidate tensors/groups
parameter counting
optimizer param groups
possible AMP/memory implementation
```

---

# 38. Transformer 上最可能出现的 12 个坑

按风险排序：

## 1. 把 micro-batch 当成 LBI scientific batch

后果：

```text
一个 outer batch 找多次 support
```

错误。

---

## 2. LBI Stage2 跑了多个 optimizer steps

因为 native IST `iters=1` 被误套到 LBI。

错误。

---

## 3. native EMA + omega double writeback

最危险的 trajectory bug 之一。

---

## 4. model.train() 打开 DropPath / Dropout，导致 PLCA / support 随机漂

Transformer 特别危险。

---

## 5. LayerNorm / cls token / pos_embed 被 broad optimizer 偷更新

scope bug。

---

## 6. PLCA feature 取错

例如用 logits / token matrix 替代当前 SHOT 定义的 representation。

---

## 7. 8-view 显存爆以后，为了省显存改了 scientific objective

例如：

```text
只用部分 views
每 chunk 独立 step
每 chunk 独立 support
```

都不行。

---

## 8. Random metadata=3，但只真正跑一个 child

当前 ResNet 已经踩过一次。

---

## 9. Saliency 每 micro-batch 重算

应该：

```text
once per outer batch
```

---

## 10. memory 提前包含 current/future state

online leakage。

---

## 11. target label 被 dataloader tuple 传进 adaptation

必须隔离。

---

## 12. 为了 Transformer 方便复制一套 LBI engine

后面 corrected semantics 会漂移。

必须共享 `core/lbi/**`。

---

# 39. 验收时建议让实现者给出的报告

完成后不要只说：

```text
all tests passed
```

需要至少报告：

## A. Source equality

```text
checkpoint:
SHOT vs IST initial logits equality:
```

## B. Dense

```text
variants:
trainable scope:
normalization state:
processed outer batches:
PLCA calls:
memory commits:
EMA commits:
PU mutation check:
```

## C. Sparse

```text
candidate count:
budget:
Random child count:
Magnitude source-static:
Saliency selections / outer batch:
```

## D. LBI

```text
LBI discoveries / outer batch:
Stage1 steps:
cap hits:
rollback:
support utilization:
Stage2 steps:
native EMA commits:
omega writebacks:
off-mask exact:
```

## E. Transformer-specific

```text
Dropout/DropPath mode:
LayerNorm trainable/frozen:
pos_embed / cls_token state:
feature extraction point:
AMP:
micro-batch strategy:
```

## F. Regression

```text
SHOT-Transformer modified? no
SHOT-Transformer regression PASS?
core/lbi duplicated? no
```

---

# 40. 最终验收标准

只有同时满足下面这些，才算 Transformer IST 接入完成：

```text
1. 初始 source 与 SHOT-Transformer 完全一致
2. IST outer-batch state machine 正确
3. PLCA once / outer batch
4. memory once / outer batch
5. 8-view task 正确
6. no target-label leakage
7. native EMA exactly once for non-LBI
8. LBI support once / outer batch
9. LBI Stage2 exactly one step
10. LBI native EMA disabled + omega only
11. Transformer scope / LN / stochastic layers语义明确
12. PU/FO read-only
13. Random 真 3-child
14. Saliency once / outer batch
15. shared core/lbi reused
16. SHOT-Transformer regression PASS
17. 5-batch real-data smoke PASS
```

然后：

```text
STOP
```

不要直接进入 parameter search。

---

# 41. 给实现者的一句话版

> **不要把 ResNet IST 的网络结构硬搬到 Transformer；要把已经验证的 IST 状态机（8-view → pre-adaptation feature/probability → PLCA once → causal memory once → fixed hard/soft targets → native IST / sparse / LBI → single persistent writeback → PU）接到现有 SHOT-Transformer substrate 上。Transformer 版最需要防的是 micro-batch 改变 LBI 优化单位、Stage2 多 step、EMA+omega double writeback、DropPath/Dropout stochasticity、LayerNorm/pos_embed/cls token 偷更新，以及 8-view 显存压力下偷偷改变 full-objective 语义。**

---

# 42. 当前最推荐的工作方式

```text
先读当前 SHOT-Transformer
-> 写 1 页 architecture mapping
-> 只做 dense IST
-> contracts
-> 5-batch smoke
-> sparse baselines
-> shared LBI adapter
-> contracts
-> independent review
-> 5-batch smoke
-> stop
```

如果在 Transformer candidate scope / grouping / normalization 上发现当前 SHOT 版本没有明确冻结：

> **先报告，不要在 IST 里自行决定。**

这些属于 Transformer scientific protocol，而不是 IST porting 的实现自由度。
