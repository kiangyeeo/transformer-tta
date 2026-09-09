# COME-OTTA Transformer 实现交接说明
## 基于当前稳定 ResNet-COME 版本：修改逻辑、实现边界与易踩坑点

**用途：** 给当前已经实现 SHOT-OTTA Transformer 版本的同学，用于继续实现 COME-OTTA Transformer，包括 dense / sparse baselines / LBI。  
**日期：** 2026-09-09  
**当前 ResNet-COME stable revisions：**

```text
COME dense baseline implementation:
come_otta_baseline_20260908_v2

COME sparse/LBI implementation:
come_otta_sparse_lbi_20260908_v3

Source checkpoint revision:
nips2026_shot_otta_uda_source_v1
```

---

# 0. 最重要的一句话

Transformer-COME 的正确实现思路不是：

> 把 ResNet-COME 整套代码机械搬到 Transformer。

而是：

> **以现有 SHOT-OTTA Transformer 实现为 substrate，只替换 host adaptation objective 为 COME；Transformer 原有的 source model、data stream、candidate universe、QK/VO/FFN grouping、budget、LBI engine、Stage-1/Stage-2、PU/FO 等语义全部保持不变。**

也就是：

$$
\boxed{
\text{SHOT-Transformer substrate}
+
\text{COME objective}
}
$$

而不是重新发明一个 Transformer TTA pipeline。

当前 ResNet 版本的 COME 也是同一个原则：

> 从 SHOT 切到 COME 时，**只替换 adaptation objective**；candidate universe、budget semantics、corrected LBI dynamics、Stage-2、Random/Magnitude/Saliency 和 evaluation discipline 都沿用已有 controlled substrate。

---

# 1. ResNet 版本中，COME 到底改了什么？

本质上只有一件核心科学变化：

```text
SHOT adaptation loss
        ↓
COME current-logit objective
```

其余 controlled TTA substrate 尽量不变。

ResNet-COME 沿用了 SHOT 的：

```text
source checkpoints
target data order
batch size / workers / seed
optimizer family / LR schedule
candidate parameter scope
FC / Conv budget definition
Random / Magnitude / Saliency definitions
corrected LBI engine
strict rollback
Stage-2 semantics
persistent writeback semantics
singleton skip
PU
FO
artifact / provenance
```

因此 Transformer 也应该沿用同样的 transplant philosophy。

---

# 2. COME objective：必须精确实现

## 2.1 输入

COME objective 只依赖：

```text
当前模型
当前 target batch
当前 logits
dataset class count K
```

不需要：

```text
pseudo label
teacher
memory bank
feature geometry
source-logit anchor
class prototype
EMA target
target label
```

这一点与 SHOT 很不一样。

Transformer-COME 中，如果现有 SHOT-Transformer objective 还有：

```text
pseudo-label construction
entropy + diversity
feature-based pseudo labels
```

这些都不应该进入 COME objective。

---

# 3. COME logit constraint

当前 audited official COME semantics：

$$
r=\|z\|_2
$$

$$
\bar z
=
\frac{z}{r}
\operatorname{sg}(r)
\tau
$$

固定：

```text
p = 2
tau = 1
norm dimension = last class dimension
keepdim = true
```

实际代码逻辑：

```python
norm = torch.norm(logits, p=2, dim=-1, keepdim=True)
constrained = logits / norm * norm.detach() * 1.0
```

## 极易踩坑 1：不能因为 tau=1 就直接删掉 constraint

虽然 `tau=1` 时 forward value 看起来基本等于原 logits，但 backward 不一样：

```text
norm.detach()
```

改变了 gradient path。

所以绝对不要写成：

```python
constrained = logits
```

也不要把：

```python
norm.detach()
```

删掉。

---

# 4. 不允许给 norm 自作主张加 epsilon / clamp

当前正式语义中：

```text
no norm epsilon
no norm clamp
```

也就是不要偷偷改成：

```python
norm = norm.clamp_min(1e-6)
```

或：

```python
logits / (norm + 1e-6)
```

这会改变正式 objective。

如果出现 non-finite：

```text
fail loudly
```

而不是偷偷改变数学目标。

---

# 5. Subjective opinion：必须用 stable log-domain 实现

COME 原始直接计算：

$$
e_k=\exp(\bar z_k)
$$

$$
S=\sum_k e_k+K
$$

$$
b_k=\frac{e_k}{S},
\qquad
u=\frac{K}{S}
$$

在 VisDA 上曾真实发生 float32：

```text
exp(constrained_logits) overflow
→ inf
→ inf / inf
→ NaN belief
→ NaN loss
```

因此当前 stable ResNet-COME 已经改成**数学等价的 log-domain 实现**。

必须照这个实现，Transformer 不允许回退到 direct `exp(logits)`。

稳定形式：

$$
\log S
=
\operatorname{logsumexp}
(
\bar z_1,\ldots,\bar z_K,\log K
)
$$

$$
b_k
=
\exp(\bar z_k-\log S)
$$

$$
u
=
\exp(\log K-\log S)
$$

推荐直接复用当前 `come_otta/objective.py` 的实现思想：

```python
log_k = constrained.new_full(
    (constrained.shape[0], 1),
    math.log(float(class_count)),
)

log_strength = torch.logsumexp(
    torch.cat((constrained, log_k), dim=1),
    dim=1,
    keepdim=True,
)

belief = torch.exp(constrained - log_strength)
uncertainty = torch.exp(log_k - log_strength)
opinion = torch.cat((belief, uncertainty), dim=1)
```

---

# 6. COME entropy

当前正式 loss：

$$
o=[b_1,\ldots,b_K,u]
$$

先做：

$$
o_\epsilon=o+10^{-7}
$$

然后：

$$
\mathcal L_{\mathrm{COME}}
=
-\frac1B
\sum_i
\sum_j
o_{\epsilon,ij}
\log o_{\epsilon,ij}
$$

固定：

```text
entropy epsilon = 1e-7
```

## 极易踩坑 2：加 epsilon 后不要 renormalize

正确：

```python
entropy_input = opinion + 1e-7
entropy = -(entropy_input * torch.log(entropy_input)).sum(dim=1)
loss = entropy.mean()
```

错误：

```python
opinion = opinion + 1e-7
opinion = opinion / opinion.sum(...)
```

正式 COME 没有这一步。

---

# 7. K 是 dataset class count，不是 ImageNet 1000

Transformer 尤其容易踩这个坑，因为 backbone 可能来自 ImageNet pretrain。

COME 里的：

$$
K
$$

必须等于**当前 adaptation dataset classifier output dimension**。

例如：

```text
Office-31: K = 31
VisDA-C:   K = 12
```

不是：

```text
1000
```

也不是 backbone pretraining class count。

必须 assert：

```python
logits.shape[1] == class_count
```

---

# 8. Transformer-COME 应该怎么从现有 SHOT-Transformer 改？

推荐最小修改路径：

```text
现有 SHOT-Transformer
│
├── source model loading              保留
├── target loader / order             保留
├── Transformer candidate scope       保留
├── QK / VO / FFN grouping            保留
├── integer budget mapping            保留
├── sparse mask application           保留
├── corrected LBI engine              保留
├── Stage-1 / Stage-2                 保留
├── omega writeback                   保留
├── PU / FO                           保留
├── artifact / identity               保留
│
└── adaptation objective
      SHOT
        ↓
      COME
```

核心建议：

> **不要重新写一套 Transformer LBI engine。**

如果现有 SHOT-Transformer 已经通过了：

```text
candidate grouping
budget
mask
Stage-1
strict rollback
Stage-2
writeback
```

就直接复用。

---

# 9. Transformer candidate scope：以现有 SHOT-Transformer 为唯一真相

当前 Transformer track 已经有自己的：

```text
QK / VO / FFN group
```

以及相应：

```text
candidate parameter universe
group count
integer K
budget mapping
```

COME-Transformer 必须和对应 SHOT-Transformer **完全一致**。

不要从 ResNet 版本照搬：

```text
netB.bottleneck
netF.layer4
FC scalar
Conv out-channel
```

这些都是 ResNet-specific。

Transformer 需要 assert：

```text
candidate tensor names exact
candidate tensor count exact
total group count exact
integer budget exact
```

最好直接调用 SHOT-Transformer 已有 candidate builder。

---

# 10. Source checkpoint 必须与 SHOT-Transformer 完全一致

ResNet-COME 没有重新训练 source model。

而是严格复用 SHOT source：

```text
same source checkpoint path
same checkpoint revision
same SHA256
```

Transformer 也一样。

不要因为实现 COME 就：

```text
重新 train source
换 pretrained checkpoint
换 classifier head
换 normalization
```

正确逻辑：

$$
\boxed{
\theta_{\mathrm{src}}^{COME}
=
\theta_{\mathrm{src}}^{SHOT}
}
$$

最好在 artifact 中保存：

```text
source path
source revision
SHA256
```

并和 SHOT-Transformer 做 exact identity audit。

---

# 11. Dense COME：每个 valid outer batch 就一个 current-state objective

非 LBI dense COME 的每个 valid outer batch：

```text
1. set correct model train/eval behavior
2. schedule LR
3. optimizer.zero_grad()
4. forward current target batch
5. compute COME(current logits)
6. backward
7. exactly one optimizer.step()
8. PU read-only
```

必须记录：

```text
objective_call_count = 1
optimizer_step_count = 1
scheduler_step_count = 1
```

---

# 12. Transformer 的 train/eval 行为不要照抄 ResNet BN 规则

ResNet 中 controlled module sparse/dense 会特别处理 BatchNorm：

```text
full dense: BN adaptive
FC / Conv controlled: BN frozen
```

Transformer 一般不是 BN 主导，而是：

```text
LayerNorm
Dropout
DropPath / stochastic depth
attention dropout
MLP dropout
```

所以 Transformer-COME 的规则应该是：

> **完全复制现有 SHOT-Transformer 在同一 variant 下的 model mode / LayerNorm / dropout / stochastic-depth behavior。**

不要因为看了 ResNet COME 就新增一个“冻结所有 LayerNorm”或“全部 model.eval()”的规则。

科学上应该保证：

```text
SHOT-Transformer vs COME-Transformer
only objective differs
```

---

# 13. Transformer 最大的额外坑：Dropout / DropPath / RNG

这一点比 ResNet 更危险。

LBI Stage-1 在同一个 outer batch 内会反复：

```text
forward
compute current-state objective
backward
update local Delta/Gamma/Z
forward again
...
```

如果 Transformer 处于：

```text
dropout active
DropPath active
stochastic depth active
```

那么即使参数状态相同，两次 forward 也可能因为 RNG 不同产生不同 loss / gradient。

这会污染：

```text
Stage-1 dynamics
saliency selection
objective recomputation audit
reproducibility
```

所以必须先核对 **现有 SHOT-Transformer 的随机层策略**。

原则：

```text
COME 不改变 SHOT-Transformer 的 stochastic-module policy
```

同时建议做 RNG audit。

尤其 Saliency baseline：

```text
current-state COME gradient once
→ build saliency mask
→ reuse same gradient for masked host optimizer step
```

只有在 selection 和 optimizer step 之间：

```text
model state unchanged
RNG state unchanged
normalization state unchanged
objective state unchanged
```

时，gradient reuse 才严格成立。

ResNet-COME 已经显式检查 CPU/CUDA RNG 没变化。

Transformer 里更应该保留这个检查。

---

# 14. LayerNorm 与 classifier head

虽然 LayerNorm 没有 BN running stats，但它的 affine parameters仍然是 persistent parameters。

所以：

- 如果不属于 SHOT-Transformer candidate scope：
  ```text
  requires_grad=False
  ```
- 不应该发生 optimizer update；
- tuning/formal 前后最好做 byte-level / exact state audit。

Classifier head 同理。

COME objective 需要 classifier logits，但：

> **需要 logits ≠ classifier head 必须被更新。**

如果 SHOT-Transformer controlled track 冻结 classifier head，COME 也必须冻结。

---

# 15. Sparse non-LBI baselines：COME 只改 scoring objective，不改 selector 定义

ResNet-COME 已冻结：

## Random

```text
global exact-budget random support
3 real independent child trajectories
fresh source model
fresh optimizer
same target stream
mask seeds = 202600 / 202601 / 202602
```

不能只生成 3 个 metadata mask 然后实际只跑一次。

Transformer 也要真实 3 children。

---

## Magnitude

```text
source-checkpoint static magnitude
once before target stream
```

不是：

```text
current-state magnitude every batch
```

COME objective 不参与 Magnitude support construction。

---

## Saliency

COME Saliency：

$$
\text{score}
=
|\theta\odot\nabla_\theta
\mathcal L_{\mathrm{COME}}(\Theta_t)|
$$

语义：

```text
current persistent state
current valid outer batch
one COME forward/backward
one support selection per batch
```

然后同一次 gradient 可以用于 masked host optimizer step，前提是：

```text
selection 本身没有改变
model
RNG
norm state
objective state
```

Transformer 一定要注意 dropout / DropPath 带来的 RNG 问题。

---

# 16. Non-LBI sparse optimizer 语义

Random / Magnitude / Saliency：

```text
host optimizer persists across outer batches
host LR scheduler persists
one masked optimizer step per valid outer batch
off-mask exact preservation
```

也就是：

```text
persistent_writeback = masked_host_optimizer
```

不要和 LBI 混。

---

# 17. LBI branch：不能使用 persistent host optimizer

COME-LBI 与 non-LBI 最大不同：

```text
host persistent optimizer step = 0
host persistent scheduler step = 0
```

每个 valid outer batch：

```text
local LBI restart once
Stage-1 support discovery once
Stage-2 local optimizer instance once
omega writeback once
```

最后：

```text
persistent_writeback = lbi_omega_only
```

如果 Transformer-COME-LBI 还在调用 SHOT host optimizer：

```python
optimizer.step()
```

就是严重错误。

---

# 18. COME-LBI 每个 outer batch 的正确顺序

每个 actual BS > 1 的 target batch：

```text
1. persistent model = theta_t

2. local LBI restart
   Delta^0 = 0
   Gamma^0 = 0
   Z^0 = 0
   M^0 = 0

3. Stage-1:
   a. construct current candidate model
   b. forward SAME current target batch
   c. obtain current logits
   d. recompute COME loss
   e. backward current candidate gradient
   f. corrected LBI update
   g. threshold support
   h. strict integer budget / rollback
   i. repeat until stop

4. Stage-2 init:
   theta_2^0
   =
   theta_t + M* ⊙ Delta*

5. recompute full COME objective at Stage-2 current state

6. exactly ONE masked local SGD step

7. discard Stage-2 optimizer

8. omega persistent writeback

9. enforce exact off-mask preservation

10. PU read-only

11. discard all local LBI state
```

---

# 19. COME-LBI 最关键的一点：Stage-1 每一步必须重新算 current-state COME

不能做：

```text
在 outer batch 开始算一次 COME gradient
然后整个 Stage-1 重复使用
```

因为 Stage-1 candidate model 在变化。

正确：

$$
g^k
=
\nabla_\Delta
\mathcal L_{\mathrm{COME}}
(
B_t;\theta_t+\Delta^k
)
$$

每个 Stage-1 candidate：

```text
fresh forward
fresh COME loss
fresh backward
```

因此 objective-call contract：

```text
objective_call_count
=
stage1_steps_completed + 1
```

最后的 `+1` 是 Stage-2 objective。

---

# 20. 不允许把 SHOT pseudo-target 固定逻辑带进 COME

SHOT 中可能有：

```text
pseudo label
feature statistics
class centroids
information maximization components
```

COME 都没有。

所以不要出现：

```text
construct pseudo target once
freeze pseudo target for Stage1
```

COME-LBI 的 closure 只需要：

```python
logits = model(inputs)
loss = come_loss(logits, class_count)
```

这反而比 SHOT 简单。

---

# 21. Corrected LBI semantics 必须沿用 shared engine

Transformer-COME 不要单独再实现一遍 LBI 数学。

必须继续使用当前 corrected shared LBI semantics。

关键包括：

$$
c^k
=
\frac{\Delta^k-\Gamma^k}{\nu}
$$

$$
\Delta^{k+1}
=
\Delta^k
-
\alpha\kappa(g^k+c^k)
$$

$$
Z^{k+1}
=
Z^k+\alpha c^k
$$

$$
\Gamma^{k+1}
=
\kappa
\operatorname{prox}(Z^{k+1})
$$

特别注意：

> `Z` update 使用 **old-state coupling** $c^k$。

不要重新引入历史旧实现的错位更新。

---

# 22. Group-LBI 的 prox / support 必须用 Transformer 自己的 group 语义

Transformer 已有：

```text
QK / VO / FFN group
```

就沿用它。

不要把 ResNet Conv：

```text
out_channel
```

的 group norm / group axis 硬搬过来。

COME 只改变：

```text
g^k 来自什么 objective
```

不改变：

```text
group 怎么定义
support 怎么统计
budget 怎么定义
```

---

# 23. Strict integer budget / rollback

当前 refined LBI：

```text
support < K:
    save latest feasible state
    continue

support == K:
    accept
    stop

support > K:
    reject overshoot state
    rollback to latest feasible state
    stop
```

禁止：

```text
top-K trim
overshoot 后切成 K 个
```

因此最终 support 可以：

```text
<K
```

这是合法结果。

Transformer 的 group budget 也一样。

---

# 24. Stage-1 cap

固定：

```text
stage1_max_steps = 3000
```

如果到 3000 inner steps 还没正常停：

```text
stage1_cap_hit = true
```

这是 reachability/tuning 问题。

不要：

```text
提高 cap
偷偷延长
用 top-K 修
```

来救 tuple。

---

# 25. Stage-2 语义

Stage-2 初始化：

$$
\theta_2^0
=
\theta_t
+
M^\star\odot\Delta^\star
$$

然后：

```text
fresh local optimizer
exactly 1 step
current-state COME objective
no scheduler
discard optimizer after batch
```

当前标准 optimizer：

```text
SGD
momentum = .9
weight_decay = .001
Nesterov = true
```

Stage-2 LR 是 LBI tunable。

---

# 26. Omega writeback

Stage-2 refined result记为：

$$
\tilde\theta_t
$$

persistent writeback：

$$
\theta_{t+1}
=
(1-\omega)\theta_t
+
\omega\tilde\theta_t
$$

仅 support 内允许 persistent change。

off-mask 必须最终 exact：

```text
byte/exact equality to theta_t
```

当前 ResNet-COME 甚至在 writeback 后显式用 mask 再恢复 off-mask base parameter，避免浮点 interpolation 引起 1 ULP 偏差。

Transformer 也建议保留 exact guard。

---

# 27. COME tau 和 LBI support threshold 不是一回事

这是非常容易命名撞车的地方。

COME：

```text
come.tau = 1
```

是 logit constraint 的参数。

LBI：

```text
support_threshold = 1e-4
```

是判断 Gamma support 的 threshold。

必须放在两个 namespace：

```text
come.tau
lbi.support_threshold
```

不要都叫 `tau`。

---

# 28. Singleton batch 必须在一切 adaptation 之前 skip

当前规则：

```text
actual batch size == 1
```

则必须在以下任何行为前直接跳过：

```text
COME objective
selector
scheduler
optimizer
LBI restart
Stage-1
Stage-2
omega writeback
PU
```

不要：

```text
先 scheduler.step()
再发现 batch size=1
```

这会让后面 stream identity 漂移。

---

# 29. PU / FO 必须 read-only

## PU

每个 successful adaptation batch 后：

```text
post-update prediction
```

必须 read-only。

不得改：

```text
parameters
LayerNorm state
BN state（若有）
optimizer
scheduler
RNG-sensitive persistent state
LBI state
```

## FO

target stream 结束后：

```text
full target evaluation
```

同样 read-only。

---

# 30. Target label 禁止进入 adaptation

Target label 只能用于：

```text
PU metric computation
FO metric computation
offline tuning selection
```

绝对不能进入：

```text
COME objective
Random
Magnitude
Saliency
LBI Stage-1
Stage-2
support selection
```

建议在代码接口上都不要把 label 传进 objective closure。

---

# 31. 数值稳定 contract：Transformer 实现前必须先过

至少做两个 objective test。

## Moderate logits

对：

```text
K=31
K=12
```

比较：

```text
old direct formula
stable log-domain formula
```

检查：

```text
loss close
belief close
uncertainty close
gradient close
```

允许 float32 rounding 级差异。

---

## Large logits

构造 direct `exp` 会 overflow 的有限 logits。

要求：

```text
old direct path non-finite
stable path finite
stable loss finite
stable belief finite
stable uncertainty finite
stable gradient finite
```

同时 assert：

```text
detach location unchanged
```

这一步一定要在 Transformer full run 前做。

---

# 32. Transformer 特有的 correctness contracts

建议至少额外加：

## Contract A：SHOT vs COME substrate identity

除 objective 之外：

```text
source checkpoint
data order
candidate names
group definition
integer budget
model train/eval policy
optimizer config
LBI runtime constants
PU/FO
```

必须一致。

---

## Contract B：candidate universe

对每个 Transformer track：

```text
candidate tensor list exact
group count exact
K exact
```

不能只比较 parameter count。

---

## Contract C：RNG / stochastic layers

在同一 state 下：

```text
saliency selection
read-only PU
objective diagnostics
```

不能意外改变 RNG/state。

如果 SHOT-Transformer 需要固定 dropout / DropPath behavior，COME 原样继承。

---

## Contract D：LBI objective calls

每个 valid outer batch：

```text
local_restart_count = 1
support_discovery_count = 1
omega_writeback_count = 1
stage2_optimizer_instance_count = 1
host_optimizer_persistent_step_count = 0

objective_call_count
=
stage1_steps_completed + 1
```

---

## Contract E：off-mask exactness

最终：

```text
off-mask persistent parameters exact equal base
```

不是 tolerance-close。

---

# 33. Collapse / instability diagnostics 建议保留

COME ResNet baseline 已经显式记录：

```text
predicted class histogram
predicted class count
dominant class ratio
mean softmax entropy
COME opinion entropy
COME mean uncertainty mass
```

Transformer 也建议保留。

原因：

如果 accuracy 掉了，要区分：

```text
普通 performance degradation
single-class collapse
class imbalance collapse
numerical instability
```

不要只看一个 accuracy。

---

# 34. Dense / sparse / LBI 三条路径不要混在一起

建议保持三套清晰语义。

## Dense

```text
persistent host optimizer
one COME objective / valid batch
one optimizer step
```

## Sparse non-LBI

```text
Random / Magnitude / Saliency support
persistent host optimizer
masked step
```

## LBI

```text
no persistent host optimizer
local Stage1 + Stage2
omega-only writeback
```

不要为了省代码让 LBI 借用 sparse host optimizer state。

---

# 35. Formal gate 要 fail-closed

当前 ResNet-COME 的 LBI formal 在：

```text
search protocol
tuned tuples
```

冻结前是 blocked 的。

Transformer 也建议：

```text
implementation/debug 可以跑
formal=true 必须拒绝
```

直到：

```text
Transformer COME-LBI search protocol frozen
final tuned tuples frozen
```

再只允许精确 frozen identity formal launch。

避免 debug tuple 混成 paper result。

---

# 36. Revision / provenance 必须独立

不要把 Transformer-COME 结果标成 ResNet-COME revision。

至少需要新的：

```text
Transformer COME baseline implementation revision
Transformer COME sparse/LBI implementation revision
Transformer COME protocol revision
```

artifact 中记录：

```text
host objective = COME
official COME commit
source revision
source SHA
dataset
backbone
track
group definition
candidate count
group count
budget rho
integer K
COME p/tau/K/epsilon
LBI alpha/kappa/nu/omega/stage2_lr
support threshold
stage1 cap
stage2 steps
```

---

# 37. 当前 COME host objective 没有要调的超参数

COME host objective 固定：

```text
p = 2
tau = 1
K = dataset class count
entropy epsilon = 1e-7
```

不要加：

```text
COME tau grid
p grid
K grid
evidence clamp grid
```

当前超参数 tuning 是 **LBI tuning**，不是 COME objective tuning。

---

# 38. 后续 Transformer-LBI 超参数主线

当前统一研究口径：

$$
\boxed{
h_{\mathrm{S1}}
=
(\alpha,\nu),
\qquad
\kappa=1
}
$$

标准 Stage-1 candidate domain：

$$
\boxed{
\alpha
\in
\{0.025,0.05,0.10,0.125,0.15,0.20\}
}
$$

$$
\boxed{
\nu
\in
\{0.25,0.50,1.00\}
}
$$

`omega` 和 `stage2_lr`：

```text
仍然需要 downstream calibration
暂时不作为 predictor target
```

不过 Transformer 当前还需要先验证：

> **kappa=1 constrained tuning 是否能基本保持已有 SHOT-Transformer 性能。**

COME-Transformer 建议直接从 `kappa=1` 开始，不主动再开 kappa 维度。

---

# 39. 推荐实际开发顺序

不要一次把 dense/sparse/LBI 全写完再测。

建议：

## P0：Objective

```text
COME stable objective module
moderate-logit equivalence
large-logit finite contract
detach contract
```

PASS 后再继续。

---

## P1：Dense Transformer COME

先只做：

```text
matched Transformer dense / module-dense scope
```

验证：

```text
same source
same stream
one objective call
one optimizer step
finite
PU/FO read-only
no collapse
```

---

## P2：Sparse non-LBI

依次：

```text
Random
Magnitude
Saliency
```

验证：

```text
exact budget
3 real Random children
source-static Magnitude
current-state Saliency
RNG/state audit
off-mask exact
```

---

## P3：COME-LBI

再接：

```text
shared corrected LBI engine
current-state COME closure
Stage1 current objective recomputation
strict rollback
Stage2 one fresh COME step
omega-only writeback
```

---

## P4：Search / formal

最后才做：

```text
search protocol
tuned tuples
formal gate
fresh formal runs
```

---

# 40. 学弟最容易犯的 15 个错误

1. **把 tau=1 当成可以删除 logit constraint。**
2. **把 `norm.detach()` 删除。**
3. **为了稳定给 norm 加 epsilon/clamp。**
4. **重新用 direct `exp(constrained)`，导致 VisDA overflow。**
5. **把 K 写成 ImageNet 1000，而不是 dataset class count。**
6. **opinion +1e-7 后又 renormalize。**
7. **把 SHOT pseudo-label / feature geometry 带进 COME。**
8. **照搬 ResNet BN policy，而不是继承 SHOT-Transformer 的 module-mode policy。**
9. **忽略 Transformer dropout / DropPath 的 RNG 问题。**
10. **Saliency selection 后重新 forward，导致“gradient reuse”已经不等价。**
11. **LBI Stage-1 只算一次 COME gradient然后缓存。**
12. **LBI branch 继续调用 persistent host optimizer.step()。**
13. **overshoot 后 top-K trim，而不是 strict rollback。**
14. **COME `tau` 和 LBI `support_threshold` 混成一个参数。**
15. **debug tuple 没冻结就允许 formal launch。**

---

# 41. 最终验收清单

在认为 Transformer-COME “实现完成”之前，至少逐项回答 YES：

```text
[ ] 以现有 SHOT-Transformer 为 substrate，而不是重新写 pipeline
[ ] source checkpoint 与 SHOT-Transformer exact same
[ ] target stream exact same
[ ] candidate/group universe exact same
[ ] budget / integer K exact same
[ ] COME p=2
[ ] COME tau=1
[ ] dataset class count K correct
[ ] norm.detach exact
[ ] no norm epsilon/clamp
[ ] log-domain stable opinion
[ ] entropy epsilon=1e-7
[ ] no epsilon renormalization
[ ] no pseudo-label / teacher / memory / source anchor
[ ] Dense: one objective + one optimizer step per valid batch
[ ] Random: 3 real children
[ ] Magnitude: source-static
[ ] Saliency: current-state COME once/batch
[ ] Saliency RNG/state contract PASS
[ ] LBI: local restart once/batch
[ ] LBI: Stage1 objective fresh every candidate
[ ] corrected old-state Z coupling
[ ] strict integer budget
[ ] strict rollback
[ ] no top-K repair
[ ] Stage1 cap=3000
[ ] Stage2 masked-delta initialization
[ ] Stage2 exactly one fresh local SGD step
[ ] Stage2 current-state COME recomputed
[ ] LBI host persistent optimizer step=0
[ ] omega writeback exactly once
[ ] off-mask persistent state exact
[ ] classifier/noncandidate state protected
[ ] Transformer stochastic-layer policy matches SHOT-Transformer
[ ] actual BS=1 skipped before all adaptation transitions
[ ] PU read-only
[ ] FO read-only
[ ] labels excluded from adaptation
[ ] collapse diagnostics available
[ ] objective large-logit regression PASS
[ ] formal gate remains blocked before tuple freeze
[ ] new Transformer-specific revision/provenance recorded
```

---

# 42. 建议代码组织

如果现有 SHOT-Transformer 结构允许，推荐避免复制大量 ResNet COME trainer。

更好的拆法：

```text
come objective
    ↓
generic current-logit closure

SHOT-Transformer substrate
    ↓
candidate/group/budget/model-mode
    ↓
dense runner
sparse runner
LBI runner
```

也就是说：

> **COME objective 尽量做成 backbone-agnostic；Transformer-specific 的东西只存在 candidate/group/model-forward/model-mode 层。**

这样以后：

```text
ResNet
Transformer
```

共享的 COME 数学目标只有一份，不会两边越改越不一致。

---

# 43. 最后一句交接原则

Transformer-COME 的 implementation review 应该始终问：

> **如果把 `COME loss` 换回 `SHOT loss`，除了 objective-specific 部分之外，是否能恢复成当前已经验证过的 SHOT-Transformer execution semantics？**

如果答案不是“基本可以”，说明实现过程中很可能不小心重新定义了：

```text
candidate scope
state transition
optimizer
grouping
budget
LBI
evaluation
```

这就是需要优先检查的地方。

最终目标是：

$$
\boxed{
\text{same Transformer substrate}
+
\text{different host objective}
}
$$

这样后面我们才能把 SHOT / IST / COME 的结果解释为真正的 **objective generality**，而不是三个不同 implementation pipeline 的混合比较。
