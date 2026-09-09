# OTTA_COME_BASELINE_PROTOCOL_20260907_v1

**状态：** COME controlled dense scientific semantics 与 implementation revision 已冻结；formal dense launch ready  
**Protocol revision：** `OTTA_COME_BASELINE_PROTOCOL_20260907_v1`  
**Scientific implementation revision：** `come_otta_baseline_20260908_v2`
**Official COME audited commit：** `409a19b71f62c765b1a5be62347a9455524ec176`  
**Source checkpoint revision：** `nips2026_shot_otta_uda_source_v1`  
**日期：** 2026-09-07  

**Numerical-stability status：** same frozen COME mathematical objective; numerically stable log-domain evaluation replaces direct exponential evaluation.
**范围：** 在当前 controlled OTTA substrate 上正式定义 COME dense baselines  
**数据集：** Office-31、VisDA-C

> 本文件从现在起作为 **COME formal dense baseline 的规范来源**。  
> 2026-09-07 的 COME fast-feasibility gate 仅作为实现可行性证据，不作为 formal protocol，也不覆盖本文件。

---

## 1. 目的与冻结边界

本 protocol 冻结 ICLR 2027 主线中的 controlled COME dense baselines。

研究目标不是逐字复现官方 Tent-COME 的整套 benchmark / model / normalization-only adaptation stack，而是：

> **保留 COME 的核心 constrained-logit + subjective-opinion entropy objective，同时把它放到与 refined SHOT-OTTA / IST-OTTA 完全相同的 source F/B/C、target stream、optimizer substrate、candidate scope 和 PU/FO evaluation framework 上。**

这样尽量消除：

```text
source training
backbone
classifier/head
target order
batch size
optimizer recipe
candidate scope
evaluation protocol
```

等混杂变量。

正式 dense COME variants 只有：

```text
come_full_dense
come_fc_module_dense
come_conv_module_dense
```

本文件不定义：

```text
COME native norm-only reference
Random
Magnitude
Saliency
LBI
Group-LBI
```

Random / Magnitude / Saliency / LBI 统一由独立 child protocol：

```text
OTTA_COME_LBI_PROTOCOL_20260907_v1
```

定义。

### 1.1 为什么主线不要求 `come_native_norm`

官方 Tent-COME 的 native usage 主要沿 Tent-style normalization affine adaptation。

但本文主问题不是：

> “能否精确复现官方 COME benchmark stack？”

而是：

> “在相同 source/network/stream/candidate spaces 上，更换 adaptation objective 后，Sparse Delta Learning 是否仍然有效？”

因此主线只要求：

```text
full dense
FC matched dense
Conv matched dense
```

不额外引入 `come_native_norm`，避免增加非必要实验。

若未来需要报告 native Tent-COME，只能作为 supplementary reference，必须另写 protocol revision；不得混入当前 matched sparse family。

---

## 2. Source-of-truth 优先级

发生冲突时：

1. `OTTA_COME_BASELINE_PROTOCOL_20260907_v1`
2. 当前完成 baseline formalization 后冻结的 `come_otta` implementation revision
3. 官方 COME audited commit `409a19b71f62c765b1a5be62347a9455524ec176`
4. 当前 refined SHOT FC/Conv benchmark / optimizer / evaluation frozen semantics
5. COME fast-feasibility gate implementation与 artifacts
6. 其他历史/临时代码

Feasibility gate 可以证明：

```text
objective implementation viable
FC/Conv no rapid class-collapse in 5-batch D->A smoke
```

但不得覆盖 formal benchmark semantics。

任何会改变本文件科学语义的修改必须升级 protocol revision。

---

## 3. 从官方 COME 保留的核心机制

正式保留：

```text
logit-space COME objective
L2 norm constraint
norm detach at the multiplicative norm term
p = 2
tau = 1
evidence = exp(constrained logits)
Dirichlet subjective-opinion construction
belief masses
uncertainty mass
opinion entropy minimization
```

COME 不引入：

```text
pseudo-label
feature-classifier geometry
teacher
memory
EMA
PLCA
multi-view augmentation
confidence filter
source anchor
extra regularization
```

也就是说，当前 COME host objective 是一个纯 current-logit objective。

---

## 4. 相对官方 COME 的 controlled-study 改动

| 项目 | 当前 formal COME | 原因 |
|---|---|---|
| Network | SHOT `netF -> netB -> netC` | 消除架构/head 混杂 |
| Source model | 与 SHOT/IST 完全相同 F/B/C | 消除初始化混杂 |
| Office backbone | ResNet-50 | 统一 benchmark |
| VisDA backbone | ResNet-101 | 统一 benchmark |
| Bottleneck | `Linear(2048,256)+BN` | 统一 FC candidate |
| Classifier | weight-normalized `netC`，始终 frozen | 统一 source hypothesis |
| Target stream | seed=2026 fixed order | 统一 sample order |
| Outer BS | Office 64 / VisDA 256 | 统一 batch boundaries |
| Target pass | 1 | formal OTTA |
| Optimizer | common controlled SGD substrate | optimizer 不作为比较变量 |
| LR schedule | common one-pass polynomial schedule | 统一 timeline |
| Evaluation | PU + FO | 统一 plasticity/stability |
| Class number `K` | dataset class count | COME 论文定义 `K=total class number` |
| Native norm-only | 不作为主线 formal variant | 主线聚焦 matched candidate spaces |

因此论文中不要写：

> exact reproduction of the official Tent-COME benchmark setup

更准确：

> We retain COME's constrained-logit subjective-opinion entropy objective while transplanting it onto the same frozen F/B/C source model, target stream, optimization substrate, parameter scopes, and PU/FO evaluation protocol used in our controlled OTTA study.

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
| Dataset aggregation | 6 directed transfers 等权平均 |

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

同时保留：

```text
overall accuracy
class-wise accuracy
worst-class accuracy / class name
class-wise standard deviation
```

作为 secondary reporting。

VisDA primary metric：

$$
\operatorname{mAcc}_{12}
=
\frac{1}{12}
\sum_{c=1}^{12}
\operatorname{Acc}_c.
$$

禁止把 overall sample accuracy替代 fixed-12-class mAcc。

### 5.3 Source checkpoint identity

COME 必须从与 SHOT/IST formal 完全相同的 F/B/C checkpoint 开始：

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

禁止为 COME 重新训练独立 source model。

---

## 6. COME objective

设当前 model state：

$$
\Theta,
$$

输入 batch：

$$
B_t=\{x_i\}_{i=1}^{n_t},
$$

logits：

$$
z_i
=
\operatorname{netC}
\left(
\operatorname{netB}
\left(
\operatorname{netF}(x_i)
\right)
\right)
\in\mathbb R^K.
$$

其中：

```text
Office: K = 31
VisDA:  K = 12
```

必须满足：

$$
K=\operatorname{dim}(z_i).
$$

### 6.1 Constrained logits

Frozen：

```text
p = 2
tau = 1
norm dimension = class/logit dimension
keepdim = true
```

定义：

$$
r_i
=
\|z_i\|_2.
$$

COME constrained logits：

$$
\widetilde z_i
=
\frac{z_i}{r_i}
\operatorname{sg}(r_i)
\tau.
$$

其中：

$$
\tau=1.
$$

关键规则：

> `detach` 只作用在乘回的 norm `r_i` 上。

禁止：

```text
detach z
detach z / norm
detach entire constrained logits
remove norm detach because forward value looks unchanged
```

虽然当 $\tau=1$ 时 forward 数值近似保持原 logits，但 backward gradient 被 COME constraint 改变；该 gradient semantics 是方法本身的一部分。

### 6.2 Numerical semantics

Official audited path 对 norm 不额外加入 epsilon/clamp。

因此正式实现不得为了“稳定”偷偷：

```text
+ eps inside norm
clamp norm
normalize with different library semantics
```

若出现：

```text
NaN
Inf
zero-norm-induced non-finite
```

必须 fail loudly，并记录 artifact；不能 silent repair。

### 6.3 Evidence 与 subjective opinion

Evidence：

$$
e_{ik}
=
\exp(\widetilde z_{ik}).
$$

Dirichlet strength：

$$
S_i
=
\sum_{k=1}^{K}e_{ik}
+
K.
$$

Belief mass：

$$
b_{ik}
=
\frac{e_{ik}}{S_i}.
$$

Uncertainty mass：

$$
u_i
=
\frac{K}{S_i}.
$$

因此理论上：

$$
\sum_{k=1}^{K} b_{ik}+u_i=1.
$$

构造：

$$
o_i
=
[b_{i1},\ldots,b_{iK},u_i].
$$

Official implementation 在 entropy 计算前使用：

```text
opinion + 1e-7
```

该位置保持不变，不重新 normalize。

### 6.4 COME loss

Per-sample opinion entropy：

$$
H_{\mathrm{COME}}(x_i;\Theta)
=
-
\sum_{j=1}^{K+1}
\left(
o_{ij}+10^{-7}
\right)
\log
\left(
o_{ij}+10^{-7}
\right).
$$

Outer-batch objective：

$$
\mathcal L_t^{\mathrm{COME}}(\Theta)
=
\frac{1}{n_t}
\sum_{i=1}^{n_t}
H_{\mathrm{COME}}(x_i;\Theta).
$$

Formal规则：

```text
no pseudo-label
no filtering
no target labels
no feature cache
no teacher/memory/EMA
```

---

## 7. Dataset-specific class-count semantics

这是从 official ImageNet implementation 移植到当前 benchmark 时最重要的 explicit rule。

Official code中的 `1000` 对应 ImageNet class count，而 COME formulation 中：

$$
K=\text{total number of classes}.
$$

因此：

```text
Office-31 -> K=31
VisDA-C   -> K=12
```

Formal implementation 必须：

1. 从 dataset/classifier config解析 `K`；
2. assert `logits.shape[-1] == K`；
3. artifact记录 `come.class_count`；
4. 禁止 Office/VisDA路径硬编码 `1000`；
5. 禁止把 `K` 当 tunable hyperparameter。

---

## 8. 三个 formal dense variants

### 8.1 `come_full_dense`

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
same native/full train behavior as current SHOT full-dense controlled substrate
```

即 adaptation forward 期间：

```text
netF.train()
netB.train()
netC.eval()
```

BN parameters 可随 trainable scope优化，BN running state 按当前 SHOT full-dense semantics工作。

它是 controlled substrate 上的 COME full-dense reference。

### 8.2 `come_fc_module_dense`

只允许：

```text
netB.bottleneck.weight
netB.bottleneck.bias
```

发生 persistent update。

Candidate scalar count：

$$
256\times2048+256
=
524544.
$$

其他 parameters 全部 frozen。

所有 BN：

```text
parameters frozen
running buffers frozen
module eval behavior
```

`netC` frozen。

这是后续 COME FC sparse family 的 matched dense reference。

### 8.3 `come_conv_module_dense`

只允许 formal `netF.layer4` 9 Conv weights：

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

Candidate scalar count：

$$
12,845,056.
$$

其他 parameters 全 frozen。

所有 BN：

```text
parameters frozen
running buffers frozen
module eval behavior
```

`netB` 的 BN 也必须保持 frozen/eval。

`netC` frozen。

这是后续 COME Conv out-channel sparse family 的 matched dense reference。

---

## 9. Target stream 与 singleton compatibility

固定：

```text
(dataset, transfer, seed=2026)
```

后，所有 SHOT / IST / COME matched variants 必须看到同样的：

```text
target sample identity
sample order
outer-batch partition
```

DataLoader：

```text
drop_last = false
```

为了匹配历史 SHOT trajectory，继续冻结：

```text
if actual outer batch size == 1:
    在任何 COME adaptation/evaluation state transition 前直接 skip
```

被 skip 的 singleton 不触发：

```text
COME objective
scheduler
optimizer
BN adaptation
PU
```

FO 仍是 full-target read-only evaluation，包含完整 target set。

Artifact记录：

```text
singleton_outer_batch_policy = skip_size_1_to_match_shot_history
singleton_outer_batches_skipped
processed_outer_batches
```

禁止改成：

```text
drop_last=true
```

---

## 10. Optimizer 与 outer-batch LR schedule

COME controlled dense 不使用官方 Tent-COME benchmark optimizer作为额外变量，而继承 common controlled substrate。

### 10.1 Office

```text
base LR = 0.01
netF multiplier = 0.1
netB multiplier = 1.0
```

### 10.2 VisDA-C

```text
base LR = 0.001
netF multiplier = 0.1
netB multiplier = 1.0
```

### 10.3 Common SGD

```text
optimizer = SGD
momentum = 0.9
weight_decay = 0.001
nesterov = true
```

### 10.4 Scheduler

Formal one-pass polynomial schedule：

$$
\eta_t
=
\eta_0
\left(
1+10\frac{t}{T}
\right)^{-0.75}.
$$

其中：

- $t$ 按 processed outer target-stream timeline推进；
- $T$ 是完整 formal target loader timeline；
- debug 只跑前若干 batch时，不能把 $T$ 偷偷改成 debug batch count。

每个 valid outer batch：

```text
exactly one scheduler update
exactly one COME objective evaluation for host SGD
exactly one optimizer step
```

---

## 11. 每个 processed outer batch 的精确顺序

对每个 valid non-singleton：

$$
B_t
$$

固定：

```text
1. 从 seed=2026 fixed stream 获取 B_t
2. 若 actual BS==1 -> skip，且无任何 state transition
3. 设置 variant-specific model train/eval behavior
4. 设置当前 outer-batch LR
5. forward: netF -> netB -> netC
6. 从 current logits 构造 constrained logits
7. 用 dataset K 构造 evidence / belief / uncertainty
8. 计算 batch-mean COME opinion entropy
9. backward
10. exactly one optimizer step
11. post-update PU，read-only
12. 进入下一 outer batch
```

完整 stream 后：

```text
freeze final persistent model
-> full-target FO
-> read-only
```

COME 没有：

```text
memory commit
teacher update
EMA
pseudo-label refresh
multi-view inner loop
```

---

## 12. PU 与 FO

### 12.1 PU

定义：

> same current batch, post-update, read-only evaluation.

即：

$$
\Theta_t
\rightarrow
\text{COME update}
\rightarrow
\Theta_{t+1}
\rightarrow
\operatorname{PU}(B_t).
$$

PU不得修改：

```text
parameters
BN buffers
optimizer state
scheduler state
method state
```

对 `come_full_dense`：

> PU 若需要 train-mode-compatible forward，必须使用当前 refined SHOT 的 BN-state preservation semantics，使 PU 本身不产生 persistent BN state change。

对 FC/Conv：

```text
all BN remain frozen/eval
```

被 skip singleton 没有 PU record。

### 12.2 FO

stream 完成后：

```text
netF.eval()
netB.eval()
netC.eval()
```

冻结 final model，对完整 target evaluation set只读计算。

禁止：

```text
adaptation
optimizer
scheduler update
BN update
```

Primary metric：

```text
Office = overall accuracy
VisDA  = fixed-12-class mAcc
```

---

## 13. Target-label boundary

Target labels只允许：

```text
PU metric
FO metric
offline reporting/analysis
```

禁止进入：

```text
COME objective
constrained logits
belief/uncertainty
optimizer
scheduler
BN behavior decisions
method state
```

当前 COME objective本身不需要 target label、pseudo-label或 class-frequency supervision。

若未来 tuning 使用 labeled validation，必须先写独立 preregistered search protocol。

---

## 14. Formal baseline experiment matrix

三个 dense COME variants都是 non-budgeted。

不能按 sparse budget重复运行。

### Office-31

```text
6 transfers x 3 dense COME variants = 18 formal runs
```

### VisDA-C

```text
1 transfer x 3 dense COME variants = 3 formal runs
```

总计：

$$
18+3=21.
$$

`source_only` 是共享 source reference，不属于 COME adaptation variant。

若 scientific identity完全一致，可复用现有 source-only artifacts。

---

## 15. Formal execution restrictions

Formal dense baseline必须：

```text
formal_protocol = true
save_model = false
debug_max_outer_batches = None
```

`--debug-max-outer-batches` 只允许 smoke/debug。

Formal run如设置 debug limit必须 fail closed。

Baseline不定义 partial resume / stream checkpoint。

中断后：

```text
rerun
```

禁止：

```text
根据 final formal accuracy 反向调 COME objective
改变 p/tau/K
改变 common optimizer/LR
```

---

## 16. COME objective parameters：全部 frozen，不需要 host search

正式冻结：

```text
come.p = 2
come.tau = 1
come.class_count = dataset class count
opinion_eps = 1e-7 at official entropy location
```

这些不是 search dimensions。

COME baseline不需要像 NCTTA一样先做 host-objective hyperparameter search。

禁止：

```text
per-dataset tau tuning
per-transfer tau tuning
treat K as hyperparameter
tune opinion_eps
change detach location
```

如果未来要研究这些，只能作为明确 ablation并升级 protocol。

---

## 17. Scientific identity 与 provenance

Formal run至少记录：

```text
protocol_revision
implementation_revision
official_come_commit
source_checkpoint_revision
source F/B/C resolved paths
source F/B/C SHA256
method = COME
variant
dataset
source/target transfer
backbone
formal seed
outer batch size
target stream identity/hash
singleton policy
candidate/update scope
candidate scalar count
BN semantics

COME objective:
  come.p
  come.tau
  come.class_count
  come.opinion_eps
  come.norm_dim
  come.norm_detach_semantics

optimizer:
  family
  momentum
  weight_decay
  nesterov
  base_lr
  netF_multiplier
  netB_multiplier
  scheduler
```

VisDA还必须记录：

```text
primary_metric_name = fixed_12_class_mAcc
fixed_class_count = 12
```

Smoke/debug artifact不能被当成 formal result。

---

## 18. Formal implementation contracts

正式 baseline implementation冻结前至少验证：

1. Official audited commit记录正确。
2. COME loss与 independent direct reference一致。
3. COME gradient与 independent direct reference一致。
4. norm detach位置准确。
5. `p=2`。
6. `tau=1`。
7. `opinion_eps=1e-7`位置与 official一致。
8. Office使用 `K=31`。
9. VisDA使用 `K=12`。
10. logits class dimension与 K exact match。
11. Office/VisDA路径不存在 ImageNet-specific hard-coded 1000。
12. source F/B/C与 SHOT exact same。
13. deterministic pre-adaptation source logits一致。
14. `netC` byte-identical。
15. full-dense update scope与 SHOT full-dense一致。
16. FC exactly 2 bottleneck tensors / 524544 scalars。
17. Conv exactly 9 layer4 Conv weights / 12845056 scalars。
18. FC/Conv所有 BN parameters/buffers byte-identical。
19. 每 valid outer batch objective count=1。
20. 每 valid outer batch scheduler update=1。
21. 每 valid outer batch optimizer step=1。
22. singleton在 objective/scheduler/optimizer/PU前skip。
23. PU read-only。
24. FO read-only。
25. target labels不进入 objective。
26. Office primary metric正确。
27. VisDA primary metric为 fixed-12-class mAcc。
28. Scientific identity完整区分 variant/dataset/scope/objective。
29. Existing SHOT regressions PASS。
30. Existing IST regressions PASS。
31. Existing NCTTA regressions PASS。
32. `shot_otta/**` / `ist_otta/**` / `nctta_otta/**` / `core/lbi/**` 未被修改。

Contracts必须检查真实 state transitions，不只看 requires_grad/shape。

---

## 19. Independent code review

Tests PASS 后必须人工检查：

```text
official objective mapping
detach location
K resolution
class-count assertion
variant dispatch
candidate construction
train/eval behavior
BN buffer behavior
optimizer param groups
scheduler timing
singleton
PU/FO
label boundary
artifact/provenance
experiment identity
shared-file diff
```

重点确认：

```text
COME implementation 没有为了复用代码而改变 SHOT / IST / NCTTA / shared LBI
```

不得只信“all contracts PASS”。

---

## 20. Baseline short smoke

Formal baseline implementation + contracts + review 后，只做 correctness smoke：

```text
Office D->A
seed=2026
BS64
5 valid outer batches

come_full_dense
come_fc_module_dense
come_conv_module_dense
```

这与 feasibility gate形式相同，但必须基于**正式 baseline implementation/config/identity**重新确认，而不能把 gate artifact直接冒充 formal-baseline smoke。

至少检查：

```text
loss finite
gradient finite
update finite
objective/scheduler/optimizer count = 5/5/5
scope exact
BN semantics
netC exact
PU/FO read-only
no class-count mismatch
artifact provenance
```

Collapse diagnostics可以保留为 detached diagnostics：

```text
predicted_class_count
dominant_class_ratio
softmax entropy
COME uncertainty
relative update norm
```

只用于 correctness/stability观察，不用于调参。

完成 smoke 后停止，不跑 full formal。

---

## 21. Feasibility evidence

2026-09-07 fast feasibility gate已经证明：

```text
COME objective direct loss/gradient checks PASS
Office K=31
source/checkpoint/stream alignment PASS
FC/Conv exact scope PASS
FC/Conv BN byte-identical
netC byte-identical
3 x 5-batch D->A smoke finite
no rapid single-class collapse in FC/Conv
existing SHOT/IST/NCTTA regressions PASS
```

该 gate的结论：

```text
COME_FEASIBILITY_GO
```

这只是进入 formal baseline implementation的依据，不是 formal result。

---

## 22. 正式冻结项

除非发现 reproducible correctness bug，否则以下全部冻结：

```text
same source F/B/C as SHOT/IST
source revision nips2026_shot_otta_uda_source_v1
R50 Office / R101 VisDA
F -> B -> C architecture
netC frozen
seed-2026 fixed target stream
Office BS64 / VisDA BS256
workers=4
one target pass
drop_last=false + size-1 singleton skip

COME constrained-logit objective
p=2
tau=1
norm detach semantics
belief/uncertainty formulation
opinion_eps=1e-7 at official location
K = dataset class count
Office K=31
VisDA K=12

common controlled SGD
Office base LR .01
VisDA base LR .001
netF lr multiplier .1
netB lr multiplier 1
momentum .9
wd .001
Nesterov
outer-batch polynomial schedule

three dense variants
implementation revision = come_otta_baseline_20260908_v2
full BN semantics = SHOT full-dense
FC/Conv BN frozen
PU/FO read-only
Office overall-acc primary
VisDA fixed-12-class mAcc primary
target-label boundary
formal non-resume
formal debug-limit rejection
```

任何科学语义变化必须升级 protocol revision。

---

## 23. 尚未冻结

```text
COME sparse/LBI implementation revision
COME LBI search grid
utilization/eligibility gate
LBI tuned tuples
formal efficiency package
```

这些不能阻碍 baseline implementation，但在各自阶段前必须冻结。

---

## 24. 一页式冻结摘要

| Category | Frozen rule |
|---|---|
| Protocol | `OTTA_COME_BASELINE_PROTOCOL_20260907_v1` |
| Official commit | `409a19b71f62c765b1a5be62347a9455524ec176` |
| Implementation | `come_otta_baseline_20260908_v2` |
| Source revision | `nips2026_shot_otta_uda_source_v1` |
| Formal seed | 2026 |
| Office backbone / BS | R50 / 64 |
| VisDA backbone / BS | R101 / 256 |
| Workers | 4 |
| Target pass | 1 |
| Singleton | actual BS=1 -> skip before adaptation/PU |
| COME input | current logits |
| `p` | 2 |
| `tau` | 1 |
| Norm detach | multiply-back norm only |
| Evidence | `exp(constrained_logits)` |
| Strength | `sum(evidence)+K` |
| Belief | `evidence/S` |
| Uncertainty | `K/S` |
| Opinion eps | `1e-7` |
| Office K | 31 |
| VisDA K | 12 |
| Host params search | none |
| Office base LR | .01 |
| VisDA base LR | .001 |
| SGD | momentum .9 / wd .001 / Nesterov |
| Scheduler | outer-batch polynomial |
| Full dense | netF+netB; netC frozen |
| FC module dense | bottleneck weight+bias; all BN frozen |
| Conv module dense | layer4 9 Conv weights; all BN frozen |
| PU | same-batch post-update read-only |
| FO | full-target final frozen read-only |
| Office primary | overall accuracy |
| VisDA primary | fixed-12-class mAcc |
| Dense formal runs | 21 |
| Sparse/LBI | child protocol |

---

**End of protocol.**
