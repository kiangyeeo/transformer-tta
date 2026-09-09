# OTTA_COME_LBI_PROTOCOL_20260907_v1

**状态：** COME sparse/LBI scientific semantics 与 implementation revision 已冻结；non-LBI sparse baselines (Random/Magnitude/Saliency) allow formal launch；COME-LBI remains blocked until search protocol and tuned tuples are frozen
**Protocol revision：** `OTTA_COME_LBI_PROTOCOL_20260907_v1`  
**Parent COME baseline protocol：** `OTTA_COME_BASELINE_PROTOCOL_20260907_v1`  
**Parent refined FC protocol：** `OTTA_FC_LBI_PROTOCOL_20260817_v1`  
**Parent refined Conv protocol：** `OTTA_CONV_LBI_PROTOCOL_20260826_v1`  
**COME baseline implementation revision：** `come_otta_baseline_20260908_v2`
**COME sparse/LBI implementation revision：** `come_otta_sparse_lbi_20260908_v3`
**Official COME audited commit：** `409a19b71f62c765b1a5be62347a9455524ec176`  
**Source checkpoint revision：** `nips2026_shot_otta_uda_source_v1`  
**日期：** 2026-09-07  

**Numerical-stability status：** same frozen COME mathematical objective; numerically stable log-domain evaluation replaces direct exponential evaluation.
**范围：** COME + FC scalar sparse adaptation + Conv out-channel structured sparse adaptation

> 本文件冻结 **COME × sparse/LBI 的科学语义**。  
> exact LBI search grid、utilization gate、selection rule 与最终 tuned tuples 后续单独 preregister。  
> COME host objective本身无 dataset-specific tuning：`p=2`、`tau=1`、`K=dataset class count` 已由 parent baseline protocol冻结。

---

## 1. 科学问题

本研究要验证同一个 corrected Sparse Delta Learning / LBI optimizer 是否具备：

1. **objective generality**：SHOT-OTTA、IST-OTTA、COME；
2. **parameter-structure generality**：FC scalar 与 Conv out-channel group。

目标矩阵：

| Adaptation objective | FC scalar | Conv out-channel |
|---|---|---|
| SHOT-OTTA | 已完成 | 已完成 |
| IST-OTTA | 已实现 | 已实现 |
| COME | 本 protocol | 本 protocol |

核心原则：

> 从 SHOT/IST 换成 COME 时，只替换 adaptation objective；candidate universe、budget semantics、corrected LBI dynamics、Stage-2、Random/Magnitude/Saliency 和 evaluation discipline 不重新发明。

抽象：

$$
g^k
=
\nabla_{\Theta_\Delta}
\mathcal L_t^{\mathrm{COME}}
\left(
\Theta_t+\Theta_\Delta^k
\right).
$$

COME objective只依赖当前 candidate logits，不需要 pseudo-target、feature geometry、memory或 teacher。

---

## 2. Source-of-truth 优先级

发生冲突时：

1. `OTTA_COME_LBI_PROTOCOL_20260907_v1`
2. `OTTA_COME_BASELINE_PROTOCOL_20260907_v1`
3. corrected refined LBI engine + frozen refined FC/Conv semantics
4. official COME audited commit
5. current audited `come_otta/objective.py`
6. feasibility gate / temporary implementation
7. IST/NCTTA sparse implementations仅作工程模板

禁止把 IST/NCTTA host-specific state语义带入 COME。

---

## 3. 正式 variant 范围

### 3.1 FC controlled family

```text
come_fc_module_dense
come_fc_random
come_fc_magnitude
come_fc_saliency
come_fc_lbi
```

### 3.2 Conv controlled family

```text
come_conv_module_dense
come_conv_out_random
come_conv_out_magnitude
come_conv_out_saliency
come_conv_out_lbi
```

Conv grouping只允许：

```text
out_channel
```

禁止：

```text
filter_connection
```

`module_dense` 由 parent baseline protocol定义，不按 sparse budget重复。

---

## 4. 从 COME baseline继承

所有 sparse/LBI variants继承：

```text
same source F/B/C
source revision
R50 Office / R101 VisDA
seed-2026 fixed outer stream
Office BS64 / VisDA BS256
workers=4
one target pass
drop_last=false
actual-BS=1 singleton skip
same input transform
same PU/FO
same metrics
same artifact/provenance discipline

COME p=2
COME tau=1
K=dataset class count
Office K=31
VisDA K=12
official norm-detach semantics
opinion_eps=1e-7
```

Controlled FC/Conv额外继承：

```text
all BN parameters frozen
all BN running buffers frozen
netC frozen
common controlled optimizer/scheduler for Dense/Random/Magnitude/Saliency
```

---

## 5. Dataset / metric invariants

| Item | Office-31 | VisDA-C |
|---|---|---|
| Classes | 31 | 12 |
| Backbone | R50 | R101 |
| BS | 64 | 256 |
| Seed | 2026 | 2026 |
| Passes | 1 | 1 |
| Primary PU | overall accuracy | fixed-12-class mAcc |
| Primary FO | overall accuracy | fixed-12-class mAcc |

Office transfers：

```text
A->D
A->W
D->A
D->W
W->A
W->D
```

VisDA：

```text
Synthetic/Train -> Real/Validation
```

---

## 6. COME current-state objective inside sparse/LBI

对任意 candidate model：

$$
\Theta,
$$

当前 batch logits：

$$
z_\Theta(x).
$$

必须在该 current state重新构造：

```text
L2 logit norm
constrained logits
evidence
strength
belief
uncertainty
opinion
opinion entropy
```

定义：

$$
r=\|z\|_2,
$$

$$
\widetilde z
=
\frac{z}{r}\operatorname{sg}(r)\tau,
\qquad \tau=1.
$$

$$
e_k=\exp(\widetilde z_k),
$$

$$
S=\sum_k e_k+K,
$$

$$
b_k=\frac{e_k}{S},
\qquad
u=\frac{K}{S}.
$$

$$
\mathcal L_{\mathrm{COME}}
=
-\operatorname{Mean}
\sum_j
(o_j+10^{-7})
\log(o_j+10^{-7}).
$$

### 6.1 最关键规则

Stage-1第 $k$ 个 candidate：

$$
\Theta^{(k)}
=
\Theta_t+\Theta_\Delta^k.
$$

必须：

$$
g^k
=
\nabla_{\Theta_\Delta}
\mathcal L_t^{\mathrm{COME}}
(\Theta_t+\Theta_\Delta^k).
$$

即每个 candidate重新 forward得到 current logits。

禁止：

```text
cache batch-start logits
reuse batch-start constrained logits
reuse batch-start opinion
source-logit anchoring
detach entire objective
```

因为 COME 没有 pseudo-target，所以没有 IST式“固定 target”问题；也没有 NCTTA式 feature/target construction。

---

## 7. LBI state boundary

COME没有：

```text
memory
teacher
EMA
pseudo-label state
multi-view inner state
```

Persistent：

```text
theta_t
non-LBI host optimizer trajectory
```

LBI local：

```text
theta_delta
Z
Gamma
Stage2 local optimizer
```

每 valid outer batch：

```text
theta_delta = 0
Z = 0
Gamma = 0
```

不跨 batch。

---

## 8. Corrected Stage-1 LBI dynamics

$$
g^k
=
\nabla_{\Theta_\Delta}
\mathcal L_t^{\mathrm{COME}}
(\Theta_t+\Theta_\Delta^k).
$$

Coupling：

$$
c^k
=
\frac{\Theta_\Delta^k-\Gamma^k}{\nu}.
$$

Update：

$$
\Theta_\Delta^{k+1}
=
\Theta_\Delta^k
-
\alpha\kappa(g^k+c^k),
$$

$$
Z^{k+1}
=
Z^k+\alpha c^k.
$$

Frozen semantic：

> $Z^{k+1}$ 使用 old-state $(\Theta_\Delta^k,\Gamma^k)$。

禁止使用：

$$
\Theta_\Delta^{k+1}
$$

参与当前 Z update。

再由对应 prox得到：

$$
\Gamma^{k+1}.
$$

---

## 9. FC scalar track

Candidate：

```text
netB.bottleneck.weight
netB.bottleneck.bias
```

BN frozen，netC frozen。

$$
N_{\mathrm{FC}}=524544.
$$

Scalar prox：

$$
\Gamma_j^{k+1}
=
\kappa
\operatorname{sign}(Z_j^{k+1})
\left[
|Z_j^{k+1}|-1
\right]_+.
$$

Threshold：

$$
\tau_{\mathrm{support}}=10^{-4}.
$$

注意：

> 这里的 support threshold $\tau_{\mathrm{support}}$ 与 COME logit constraint 的 `come.tau=1` 不是同一个量。

实现、config、artifact必须区分：

```text
come.tau = 1
lbi.support_threshold = 1e-4
```

Support：

$$
M_j^k
=
\mathbf1[
|\Gamma_j^k|\ge10^{-4}
].
$$

Budgets：

$$
\rho\in\{0.0005,0.001,0.002\},
$$

$$
K=\lfloor \rho\cdot524544\rfloor.
$$

| ratio | K |
|---:|---:|
| .0005 | 262 |
| .001 | 524 |
| .002 | 1049 |

Global across weight+bias。

---

## 10. Conv out-channel track

Candidate仅：

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

$$
N_{\mathrm{conv}}=12,845,056.
$$

Group：

$$
G_{l,o}=W^{(l)}[o,:,:,:].
$$

$$
|\mathcal G_{\mathrm{out}}|=9216.
$$

Group prox：

$$
\Gamma_g^{k+1}
=
\kappa
\left(
1-\frac{1}{\|Z_g^{k+1}\|_2}
\right)_+
Z_g^{k+1}.
$$

Support：

$$
M_g^k
=
\mathbf1[
\|\Gamma_g^k\|_2\ge10^{-4}
].
$$

Budgets：

$$
\rho_G\in\{0.0005,0.001,0.002\},
$$

$$
K_G=\lfloor \rho_G\cdot9216\rfloor.
$$

| ratio | K_G |
|---:|---:|
| .0005 | 4 |
| .001 | 9 |
| .002 | 18 |

必须同时记录：

```text
selected_group_count
realized_group_ratio
selected_scalar_count
realized_scalar_ratio
```

group ratio不是 scalar sparsity。

---

## 11. Strict budget / rollback

```text
FC   K_star = K
Conv K_star = K_G
```

每个 Stage-1 iteration后：

```text
n < K_star:
    save latest feasible state
    continue

n == K_star:
    accept current state
    stop Stage1

n > K_star:
    reject overshoot state
    rollback latest feasible state
    stop Stage1
```

禁止：

```text
budget slack
K tolerance
ceil K
LBI top-K trimming
per-layer quota
per-tensor quota
minimum-one repair
```

允许：

$$
|M^\star|<K_\star.
$$

---

## 12. Stage-1 cap

```text
stage1_max_steps = 3000
```

只是 safety cap。

大量 cap hit意味着 tuple不适合，不允许偷偷增大 cap。

---

## 13. Stage-2

Final Stage1：

$$
(\Theta_\Delta^\star,\Gamma^\star,M^\star).
$$

Init：

$$
\Theta_{t,2}^0
=
\Theta_t
+
M^\star\odot\Theta_\Delta^\star.
$$

Off-mask delta不能进入 Stage2。

### 13.1 Exactly one step

```text
stage2_steps = 1
```

Stage2 objective必须在 Stage2 current state重新 forward：

$$
\nabla_\Theta
\mathcal L_t^{\mathrm{COME}}
(\Theta_{t,2}^0).
$$

不能复用 Stage1最后一次 gradient/loss，因为 model state不同。

### 13.2 Stage2 optimizer

```text
SGD
momentum=.9
weight_decay=.001
nesterov=true
schedule=none
stage2_lr = tunable
```

每 outer batch fresh local Stage2 optimizer。

### 13.3 Off-mask protection

```text
mask gradients
restore/freeze off-mask values
off-mask local optimizer state must not create hidden persistent parameter updates
fresh local Stage-2 optimizer/state is discarded after the single step
noncandidate unchanged
BN frozen
netC frozen
```

---

## 14. Persistent writeback

COME non-LBI sparse：

```text
masked host SGD
no EMA
no omega
```

COME-LBI：

$$
\Theta_{t+1}
=
(1-\omega)\Theta_t
+
\omega\widetilde\Theta_t.
$$

唯一 writeback：

```text
LBI omega
```

LBI branch禁止 host persistent optimizer step。

禁止：

```text
host SGD -> omega
omega -> host SGD
double writeback
```

Off-mask在 omega后仍 exact unchanged。

---

## 15. FC Random / Magnitude / Saliency

同一 candidate pool、同一 exact K。

### Random

每 budget真实运行3条 independent child trajectories：

```text
202600
202601
202602
```

每 child：

```text
fresh source F/B/C
fresh optimizer state
same target stream
uniform exact-K scalar mask
mask fixed entire stream
```

Top-level聚合 mean/std。

### Magnitude

source checkpoint：

$$
s_j=|\theta_j^{src}|.
$$

Global top-K，一次计算，整条 stream固定。

### Saliency

每 valid outer batch、pre-update current state：

$$
g_j^{(t)}
=
\nabla_{\theta_j}
\mathcal L_t^{\mathrm{COME}}(\Theta_t),
$$

$$
s_j^{(t)}
=
|\theta_j^{(t)}g_j^{(t)}|.
$$

Global exact top-K。

Support每 batch只选一次。

若复用同一次 saliency gradient执行 masked host step，必须证明 model/RNG/BN/objective state中间无变化，数学上等价。

---

## 16. Conv Random / Magnitude / Saliency

同一 9216 out-channel group pool、同一 exact K_G。

### Random

3 real child trajectories；uniform exact-K_G groups；stream-fixed mask。

### Magnitude

source：

$$
s_g=\|W_g^{src}\|_2.
$$

Global top-K_G，stream fixed。

### Saliency

current pre-update model：

$$
s_g^{(t)}
=
\left\|
W_g^{(t)}
\odot
\nabla_{W_g}
\mathcal L_t^{\mathrm{COME}}(\Theta_t)
\right\|_2.
$$

Global exact top-K_G，每 outer batch只选一次。

---

## 17. Non-LBI sparse optimizer semantics

Random/Magnitude/Saliency继承 controlled COME dense host optimizer：

```text
Office base LR=.01
VisDA base LR=.001
netF multiplier=.1
netB multiplier=1
SGD momentum=.9
wd=.001
nesterov=true
outer polynomial scheduler
exactly one optimizer step / valid outer batch
```

Sparse protection：

```text
only selected coordinates/groups change
off-mask gradients masked
off-mask values restored
off-mask momentum/state cleared
selected coordinates may retain legal host optimizer state
BN frozen
netC frozen
```

不使用：

```text
LBI Stage2 optimizer
omega
```

---

## 18. COME-LBI 每 outer batch精确顺序

```text
1. 取得 current valid B_t
2. save persistent pre-batch theta_t
3. reset theta_delta/Z/Gamma
4. Stage1 iteration k:
     a. candidate = theta_t + theta_delta^k
     b. candidate forward -> current logits
     c. recompute norm/constrained logits/evidence/opinion
     d. compute COME loss
     e. gradient wrt candidate delta
     f. corrected LBI update
     g. threshold support
     h. strict budget / rollback
5. final M*
6. Stage2 model = theta_t + M* ⊙ theta_delta*
7. recompute full COME objective on Stage2 state
8. exactly one masked Stage2 SGD step
9. exactly one omega persistent writeback
10. PU read-only
11. discard all local LBI state
```

Stream end：

```text
freeze final model
-> full-target FO
-> read-only
```

---

## 19. Non-LBI sparse每 outer batch顺序

Random / Magnitude / Saliency：

```text
1. current B_t
2. get support:
     Random    = stream-fixed
     Magnitude = source-fixed
     Saliency  = current-state COME gradient once
3. current-state COME objective
4. exactly one masked host SGD step
5. PU read-only
6. next batch
```

不存在：

```text
LBI local states
Stage2 local optimizer
omega
EMA
```

---

## 20. Singleton

```text
actual BS == 1
-> skip before any COME/sparse/LBI state transition
```

不得触发：

```text
COME objective
selector
scheduler
host optimizer
LBI Stage1
Stage2
omega
PU
```

FO仍 full target。

---

## 21. LBI frozen constants

| Item | Value |
|---|---:|
| support threshold | `1e-4` |
| Stage1 max | `3000` |
| Stage2 steps | `1` |
| Stage2 optimizer | SGD |
| momentum | .9 |
| wd | .001 |
| Nesterov | true |
| Stage2 schedule | none |
| strict rollback | yes |
| LBI top-K repair | forbidden |
| masked-delta init | yes |
| off-mask preservation | yes |
| restart | every valid outer batch |
| host EMA | none |

---

## 22. Tunables与 granularity

COME host objective没有 search dimension。

只调：

```text
lbi.alpha
lbi.kappa
lbi.nu
lbi.omega
lbi.stage2_lr
```

Granularity：

$$
(dataset,\ track,\ budget).
$$

Track：

```text
FC scalar
Conv out-channel
```

Office同 track/budget 6 transfers共用一组 tuple。

禁止：

```text
per-transfer tuning
per-seed tuning
per-batch tuning
per-budget support threshold
per-budget Stage1 cap
per-budget Stage2 steps
COME tau tuning
COME K tuning
```

12组 tuned tuples：

### FC

| Dataset | Budget | Tuple |
|---|---:|---|
| Office | .0005 | TBD |
| Office | .001 | TBD |
| Office | .002 | TBD |
| VisDA | .0005 | TBD |
| VisDA | .001 | TBD |
| VisDA | .002 | TBD |

### Conv

| Dataset | Budget | Tuple |
|---|---:|---|
| Office | .0005 | TBD |
| Office | .001 | TBD |
| Office | .002 | TBD |
| VisDA | .0005 | TBD |
| VisDA | .001 | TBD |
| VisDA | .002 | TBD |

---

## 23. Search items intentionally not frozen here

后续 `COME_LBI_SEARCH_PROTOCOL` 冻结：

```text
alpha/kappa/nu grid
omega grid
stage2_lr grid
Stage1 reachability screening
support-utilization eligibility
accuracy-selection rule
boundary-expansion rule
hardware allocation
```

不得根据 formal result反向扩 grid。

---

## 24. Support utilization / validity

FC至少记录：

```text
selected_count
budget_K
realized_ratio
utilization
stage1_steps
cap_hit
overshoot/rollback
COME loss trajectory
```

Conv：

```text
selected_group_count
budget_K_G
realized_group_ratio
selected_scalar_count
realized_scalar_ratio
group utilization
stage1_steps
cap_hit
overshoot/rollback
COME loss trajectory
```

Invalid：

```text
NaN/Inf
runtime correctness failure
support > strict budget
off-mask changes
BN changes
netC changes
missing support artifact
cap treated as silent success
target-label leakage
future-target leakage
double writeback
protocol/identity mismatch
```

---

## 25. PU / FO

PU：

```text
non-LBI:
  masked host SGD -> PU

LBI:
  omega writeback -> PU
```

PU read-only，不得改：

```text
parameters
BN
optimizer
scheduler
selector
LBI state
writeback anchor
```

FO：

```text
freeze final model
-> full target eval
-> read-only
```

Primary：

```text
Office overall accuracy
VisDA fixed-12-class mAcc
```

---

## 26. Target-label policy

Target GT只用于 evaluation/reporting。

禁止进入：

```text
COME objective
Random
Magnitude
Saliency
LBI Stage1
Stage2
omega
```

---

## 27. Formal sparse matrix

每 track / transfer / budget：

```text
Random
Magnitude
Saliency
LBI
```

Budgets：

```text
.0005
.001
.002
```

每 track：

```text
4 x 3 = 12 budgeted identities / transfer
```

Random 3 masks是一个 top-level identity下的 child executions。

Random/Magnitude/Saliency formal allowed。

LBI tuned tuple未冻结前 formal LBI blocked。

---

## 28. Scientific identity

至少：

```text
parent COME baseline protocol
COME-LBI protocol
COME baseline implementation revision
COME sparse implementation revision
official COME commit
source revision / source paths / SHA256
dataset / transfer / backbone
formal seed / BS
stream identity/hash
primary metric
candidate track / names
grouping
candidate scalar count
group count
variant
ratio
K / K_G
Random child index/seed
selector

COME:
  p
  tau
  class_count
  opinion_eps
  norm_detach_semantics

LBI:
  alpha
  kappa
  nu
  omega
  stage2_lr
  support_threshold
  stage1_cap
  stage2_steps

writeback mode
```

LBI：

```text
persistent_writeback = lbi_omega_only
host_optimizer_persistent_step = false
```

Non-LBI sparse：

```text
persistent_writeback = masked_host_optimizer
lbi_omega = null
```

---

## 29. Implementation contracts

至少验证：

1. same source/stream/singleton/PU/FO as parent。
2. COME objective direct loss/gradient一致。
3. COME p=2/tau=1/K semantics正确。
4. FC count=524544，budgets=262/524/1049。
5. Conv group pool=9216，budgets=4/9/18。
6. Conv realized scalar support记录。
7. Random真实3 fresh children。
8. Magnitude source-only once。
9. Saliency once per valid outer batch。
10. Saliency使用 current-state COME objective。
11. LBI restart once per valid outer batch。
12. 每 Stage1 candidate重新 forward current logits。
13. 禁止 batch-start logits/opinion cache。
14. corrected old-state Z。
15. strict rollback。
16. Stage2 init无 off-mask Stage1 delta。
17. Stage2 exactly 1 optimizer step。
18. Stage2重新算 current-state COME objective。
19. off-mask exact。
20. controlled BN exact frozen。
21. netC exact frozen。
22. LBI omega exactly once。
23. LBI host persistent optimizer step=0。
24. non-LBI masked host step exactly once。
25. PU/FO read-only。
26. singleton early skip。
27. labels不进 adaptation。
28. identity/provenance完整。
29. SHOT regressions PASS。
30. IST regressions PASS。
31. NCTTA regressions PASS。
32. COME baseline regressions PASS。
33. other-method code untouched。

---

## 30. Independent review

人工检查：

```text
dispatch
candidate pools
Random child execution/reset
Magnitude snapshot
Saliency timing
COME closure
current candidate binding
K resolution
COME tau vs LBI support threshold naming
LBI restart
old-state Z
rollback
Stage2 init
Stage2 step count
omega
BN/netC/off-mask
singleton
PU/FO
artifact aggregation
identity fallback
shared-file diff
```

---

## 31. Short smoke

实现/contract/review后：

```text
Office D->A
seed=2026
BS64
5 valid outer batches
rho=.001
all 8 sparse variants
Random = 3 real children
```

LBI使用明确 `debug_only` test tuple。

Smoke只验 correctness，不调参：

```text
GPU/data path
artifact
support
budget/rollback
objective calls
Stage2 count
omega count
Random children
scope
BN/netC/off-mask
PU/FO
```

完成后停止。

---

## 32. 明确禁止

```text
COME native norm-only 混入 sparse family
K=1000 on Office/VisDA
tune K
tune COME tau
remove norm detach
batch-start logits cache
source-logit anchor
target-label selector
per-tensor FC budget
filter_connection
old-state Z bug
budget slack
LBI top-K repair
dense Stage1 delta leakage
off-mask hidden update
Stage2 >1
host SGD + omega
Random metadata-only 3 masks
accuracy-driven smoke tuning
modify SHOT/IST/NCTTA/shared LBI semantics
```

---

## 33. Freeze boundary

### Frozen

```text
COME baseline implementation revision = come_otta_baseline_20260908_v2
COME current-logit objective
p=2
tau=1
dataset K
norm detach
opinion construction
outer-batch LBI restart
corrected old-state dynamics
FC candidates/budgets
Conv out-channel groups/budgets
strict rollback
support threshold 1e-4
Stage1 cap 3000
masked-delta Stage2 init
Stage2 exactly one current-state COME step
off-mask preservation
omega-only LBI writeback
masked host SGD non-LBI writeback
Random/Magnitude/Saliency semantics
3 real Random children
singleton
PU/FO
filter_connection excluded
LBI tunables only alpha/kappa/nu/omega/stage2_lr
```

### Not frozen

```text
LBI search grids
utilization gates
selection rule
boundary expansion
12 final tuned tuples
formal efficiency package
```

---

## 34. 一页式摘要

| Category | Frozen rule |
|---|---|
| Protocol | `OTTA_COME_LBI_PROTOCOL_20260907_v1` |
| Parent | `OTTA_COME_BASELINE_PROTOCOL_20260907_v1` |
| Baseline implementation | `come_otta_baseline_20260908_v2` |
| Sparse/LBI implementation | `come_otta_sparse_lbi_20260908_v3` |
| Official commit | `409a19...176` |
| LBI unit | each valid outer batch |
| Host input | current logits |
| COME p/tau | 2 / 1 |
| K | Office31 / VisDA12 |
| Fixed pseudo-target | none |
| FC candidate | bottleneck weight+bias |
| FC scalars | 524544 |
| FC budgets | 262 / 524 / 1049 |
| Conv candidate | layer4 9 Conv |
| Conv grouping | out-channel |
| Conv groups | 9216 |
| Conv budgets | 4 / 9 / 18 |
| Support threshold | 1e-4 |
| Strict rollback | yes |
| Stage1 cap | 3000 |
| Stage2 init | base + masked delta |
| Stage2 | exactly 1 |
| Stage2 objective | recompute current COME |
| LBI writeback | omega only |
| Non-LBI writeback | masked host SGD |
| Random | exact budget, 3 real children |
| Magnitude | source-static |
| Saliency | current-state, once/batch |
| BN | controlled FC/Conv frozen |
| netC | frozen |
| Host objective search | none |
| Formal LBI | search/tuned tuples前 blocked |

---

**End of protocol.**
