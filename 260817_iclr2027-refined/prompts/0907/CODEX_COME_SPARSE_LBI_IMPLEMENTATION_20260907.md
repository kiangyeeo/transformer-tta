# CODEX — COME Sparse Baselines + LBI Implementation

目标：严格按照已经冻结的 COME baseline / sparse-LBI protocols，一次性完成 COME 的 FC scalar 与 Conv out-channel 两条 sparse family：

```text
FC:
come_fc_random
come_fc_magnitude
come_fc_saliency
come_fc_lbi

Conv:
come_conv_out_random
come_conv_out_magnitude
come_conv_out_saliency
come_conv_out_lbi
```

完成 implementation → contracts → independent review → Office D->A、rho=0.001、5-valid-batch reviewed smoke 后立即停止。

**本任务不做 LBI search，不跑 formal，不修改任何已冻结 scientific semantics。**

---

## 0. 必读规范

开始前完整读取：

```text
260817_iclr2027-refined/protocol/come-otta/OTTA_COME_BASELINE_PROTOCOL_20260907_v1.md
260817_iclr2027-refined/protocol/come-otta/OTTA_COME_LBI_PROTOCOL_20260907_v1.md
```

当前 COME baseline implementation revision：

```text
come_otta_baseline_20260907_v1
```

Official COME audited commit：

```text
409a19b71f62c765b1a5be62347a9455524ec176
```

Source revision：

```text
nips2026_shot_otta_uda_source_v1
```

如果 protocol 与当前临时代码冲突，以 protocol 为准。

---

## 1. 最高优先级：禁止修改其他方法代码/语义

严禁修改：

```text
shot_otta/**
ist_otta/**
nctta_otta/**
core/lbi/**
```

严禁修改既有：

```text
SHOT protocol/config semantics
IST protocol/config semantics
NCTTA protocol/config semantics
shared corrected LBI semantics
```

若必须修改 shared top-level 文件：

```text
train.py
experiment_identity.py
protocol_constants.py
```

只能做 COME-specific additive extension。

禁止改变任何已有 SHOT / IST / NCTTA branch behavior。

开始前保存：

```bash
git status --short
git diff --name-only
```

完成后必须检查：

```bash
git status --short
git diff --name-only
git diff --check
```

最终明确报告：

```text
shot_otta/** modified = false
ist_otta/** modified = false
nctta_otta/** modified = false
core/lbi/** modified = false
```

如误改其他方法文件，恢复误改后再继续。

---

## 2. 本任务只实现 8 个 sparse variants

### FC

```text
come_fc_random
come_fc_magnitude
come_fc_saliency
come_fc_lbi
```

### Conv

```text
come_conv_out_random
come_conv_out_magnitude
come_conv_out_saliency
come_conv_out_lbi
```

Matched dense references已经完成：

```text
come_fc_module_dense
come_conv_module_dense
```

本任务不要重复重写 dense trainer。

禁止新增：

```text
come_native_norm
filter_connection
其他 selector
其他 grouping mode
```

---

## 3. COME objective 必须复用 frozen baseline implementation

所有 sparse / LBI variants 必须使用与：

```text
come_otta_baseline_20260907_v1
```

完全一致的 COME objective。

Frozen：

```text
p = 2
tau = 1
opinion_eps = 1e-7
K = dataset class count
Office K = 31
VisDA K = 12
```

Constrained logits：

```python
norm = torch.norm(logits, p=2, dim=-1, keepdim=True)
constrained = logits / norm * norm.detach() * 1.0
```

然后：

```text
evidence = exp(constrained)
S = sum(evidence) + K
belief = evidence / S
uncertainty = K / S
opinion = concat(belief, uncertainty)
loss = mean entropy(opinion + 1e-7)
```

禁止：

```text
额外 epsilon/clamp 到 norm
remove norm.detach
pseudo-label
teacher
memory
EMA
feature geometry
confidence filter
source-logit anchor
batch-start fixed logits
```

若 loss/gradient 非有限，fail loudly。

---

## 4. Common controlled substrate

全部继承 baseline：

```text
same source F/B/C
same target stream
same sample order
same outer-batch partition
seed = 2026
one target pass
drop_last = false
same singleton rule
same PU / FO
same metrics
netC frozen
```

Office：

```text
R50
BS64
workers=4
base LR=.01
```

VisDA：

```text
R101
BS256
workers=4
base LR=.001
```

Controlled host optimizer：

```text
SGD
momentum=.9
weight_decay=.001
nesterov=true
netF lr multiplier=.1
netB lr multiplier=1
outer polynomial schedule
```

Random/Magnitude/Saliency 使用 host optimizer。

LBI 不使用 host persistent optimizer step。

---

## 5. FC scalar candidate universe

只允许：

```text
netB.bottleneck.weight
netB.bottleneck.bias
```

必须得到：

```text
candidate tensors = 2
candidate scalars = 524544
```

所有 BN：

```text
parameters frozen
buffers frozen
eval behavior
```

`netC` frozen。

Formal ratios：

```text
rho = 0.0005 / 0.001 / 0.002
```

Strict integer budgets：

```text
K = floor(rho * 524544)

0.0005 -> 262
0.001  -> 524
0.002  -> 1049
```

Budget 必须 global across weight+bias。

禁止：

```text
per-tensor quota
per-layer quota
ceil
budget slack
```

---

## 6. Conv out-channel candidate universe

只允许：

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

必须得到：

```text
candidate tensors = 9
candidate scalars = 12845056
grouping = out_channel
groups = 9216
```

所有 BN frozen/eval。

`netC` frozen。

Ratios：

```text
rho_G = 0.0005 / 0.001 / 0.002
```

Strict group budgets：

```text
K_G = floor(rho_G * 9216)

0.0005 -> 4
0.001  -> 9
0.002  -> 18
```

禁止：

```text
filter_connection
per-layer group quota
minimum-one-per-layer repair
```

每个 run 必须同时记录：

```text
selected_group_count
realized_group_ratio
selected_scalar_count
realized_scalar_ratio
```

group ratio 不得冒充 scalar sparsity。

---

## 7. Random semantics

### FC Random

每个 budget 必须真实执行 3 条 independent child trajectories：

```text
mask seeds:
202600
202601
202602
```

每 child：

```text
fresh source F/B/C
fresh host optimizer state
same formal target stream seed=2026
uniform global exact-K scalar mask
mask fixed for entire target stream
```

### Conv Random

相同规则，但从 9216 out-channel group pool uniform sample exact K_G groups。

### Top-level Random artifact

必须聚合：

```text
3 child executions
PU mean/std
FO mean/std
child paths
child mask seeds
child supports
```

禁止：

```text
metadata num_random_masks=3
但实际只跑 1 条 trajectory
```

---

## 8. Magnitude semantics

### FC

仅在 source checkpoint：

$$
s_j = |\theta_j^{src}|.
$$

Global top-K。

Mask：

```text
computed once from source
fixed entire target stream
```

禁止根据 adapted model 重算。

### Conv

Source checkpoint：

$$
s_g = \|W_g^{src}\|_2.
$$

Global top-K_G。

Mask整个 stream固定。

---

## 9. Saliency semantics

每个 valid outer batch：

```text
current persistent pre-adaptation model theta_t
before any update
```

### FC

$$
g_j^{(t)}
=
\nabla_{\theta_j}
L_t^{COME}(\theta_t),
$$

$$
s_j^{(t)}
=
|\theta_j^{(t)}g_j^{(t)}|.
$$

Global exact top-K。

### Conv

$$
s_g^{(t)}
=
\left\|
W_g^{(t)}
\odot
\nabla_{W_g}
L_t^{COME}(\theta_t)
\right\|_2.
$$

Global exact top-K_G。

### Timing

Support：

```text
selected exactly once per valid outer batch
fixed during current batch update
recomputed only next outer batch
```

如果实现复用 support-selection gradient作为同 batch masked SGD gradient，必须 contract 证明：

```text
model unchanged
RNG unchanged
BN state unchanged
objective state unchanged
```

因此数学等价。

否则重新 forward/backward。

---

## 10. Non-LBI masked host optimizer

Random / Magnitude / Saliency：

```text
exactly one host optimizer step per valid outer batch
outer scheduler exactly once per valid outer batch
no EMA
no omega
```

Sparse protection必须同时保证：

```text
off-mask gradients zero/masked
off-mask parameter values exact restored
off-mask optimizer momentum/state cleared
noncandidate parameters unchanged
BN exact frozen
netC exact frozen
```

持续被选中的 coordinates/groups 可以保留合法 host momentum state。

禁止 hidden off-mask update。

---

## 11. COME-LBI Stage-1 semantics

每 valid outer batch：

```text
persistent base = theta_t

theta_delta^0 = 0
Z^0 = 0
Gamma^0 = 0
```

Local states不跨 batch。

对 Stage-1 candidate k：

$$
\theta^{(k)}
=
\theta_t+\theta_\Delta^k.
$$

必须重新执行完整 current-state COME forward：

```text
netF -> netB -> netC
current logits
current L2 norm
current constrained logits
current evidence
current strength
current belief
current uncertainty
current opinion entropy
```

然后：

$$
g^k
=
\nabla_{\theta_\Delta}
L_t^{COME}(\theta_t+\theta_\Delta^k).
$$

禁止：

```text
batch-start logits cache
fixed opinion across Stage1
source logits
reuse previous candidate gradient
```

---

## 12. Corrected LBI dynamics

Coupling：

$$
c^k
=
\frac{\theta_\Delta^k-\Gamma^k}
{\nu_{LBI}}.
$$

Update：

$$
\theta_\Delta^{k+1}
=
\theta_\Delta^k
-
\alpha\kappa(g^k+c^k),
$$

$$
Z^{k+1}
=
Z^k+\alpha c^k.
$$

**最关键：**

```text
Z^{k+1} 必须使用 old-state theta_delta^k / Gamma^k
```

禁止使用：

```text
theta_delta^{k+1}
```

参与当前 Z update。

---

## 13. FC scalar prox

$$
\Gamma_j^{k+1}
=
\kappa
\operatorname{sign}(Z_j^{k+1})
\left(
|Z_j^{k+1}|-1
\right)_+.
$$

Support threshold：

```text
lbi.support_threshold = 1e-4
```

Support：

$$
M_j
=
1[|\Gamma_j|\ge10^{-4}].
$$

注意命名：

```text
come.tau = 1
lbi.support_threshold = 1e-4
```

禁止 artifact/config 中使用无法区分含义的裸：

```text
tau
```

---

## 14. Conv group prox

对 out-channel group：

$$
\Gamma_g^{k+1}
=
\kappa
\left(
1-\frac{1}{\|Z_g^{k+1}\|_2}
\right)_+
Z_g^{k+1}.
$$

zero norm必须做数值安全处理，但不能改变 support semantics。

Support：

$$
M_g
=
1[\|\Gamma_g\|_2\ge10^{-4}].
$$

Group mask必须正确 broadcast 到 scalar coordinates。

---

## 15. Strict budget / rollback

Stage-1每次更新后统计 support。

```text
FC:   K_star = K
Conv: K_star = K_G
```

严格状态机：

```text
n < K_star:
    current state feasible
    save as latest feasible
    continue

n == K_star:
    accept current state
    stop Stage1

n > K_star:
    reject current overshoot state
    rollback latest feasible state
    stop Stage1
```

禁止：

```text
budget slack
K tolerance
ceil
LBI top-K trim
post-hoc repair
per-layer quota
minimum-one repair
```

合法：

```text
final support < budget
```

---

## 16. Stage-1 cap

Frozen：

```text
stage1_max_steps = 3000
```

只是 safety cap。

不得：

```text
自动增大
根据 smoke 修改
把 cap-hit 当正常成功
```

Artifact必须记录：

```text
stage1_steps
cap_hit
overshoot
rollback
final support
```

---

## 17. Stage-2 masked refinement

Final Stage-1 state：

```text
theta_delta*
Gamma*
M*
```

Stage2 init：

$$
\theta_{2}^{0}
=
\theta_t
+
M^*\odot\theta_\Delta^*.
$$

必须保证：

```text
off-mask Stage1 delta 不得泄漏
```

Conv group mask broadcast 到 scalar。

### Stage2 objective

在 Stage2 current model上重新完整计算 COME objective：

```text
current logits
constrained logits
opinion
loss
```

不能复用 Stage1最后一次 gradient。

### Exactly one optimizer step

```text
stage2_steps = 1
```

Stage2 optimizer：

```text
SGD
momentum=.9
weight_decay=.001
nesterov=true
schedule=none
stage2_lr = configurable LBI tunable
```

每 valid outer batch：

```text
fresh local Stage2 optimizer
```

Stage2 optimizer state不跨 outer batches。

---

## 18. Stage-2 off-mask protection

Stage2必须：

```text
mask off-support gradients
restore/freeze off-mask parameter values
prevent optimizer momentum/weight-decay hidden update off-mask
noncandidate parameters unchanged
all controlled BN exact frozen
netC exact frozen
```

Contract必须检查 actual before/after state，而不是只看 gradients。

---

## 19. LBI persistent writeback

COME-LBI禁止 host persistent optimizer step。

Stage2 refined state：

$$
\widetilde\theta_t.
$$

唯一 persistent writeback：

$$
\theta_{t+1}
=
(1-\omega)\theta_t
+
\omega\widetilde\theta_t.
$$

每 valid batch：

```text
host persistent optimizer step = 0
Stage2 local optimizer step = 1
omega writeback = 1
```

禁止：

```text
host optimizer -> omega
omega -> host optimizer
double writeback
EMA
```

Off-mask coordinates在 omega writeback后仍必须 exact unchanged。

---

## 20. LBI tunables / constants

本任务只实现配置接口，不做 tuning。

LBI tunables：

```text
lbi.alpha
lbi.kappa
lbi.nu
lbi.omega
lbi.stage2_lr
```

Frozen constants：

```text
support threshold = 1e-4
stage1 max steps = 3000
stage2 steps = 1
Stage2 SGD
momentum=.9
wd=.001
Nesterov=true
Stage2 scheduler=none
strict rollback=true
top-K repair=false
```

COME host objective没有 tuning dimension：

```text
come.p=2
come.tau=1
come.class_count=dataset K
come.opinion_eps=1e-7
```

不要实现任何 COME host search。

---

## 21. Sparse scientific identity / provenance

必须记录：

```text
parent COME baseline protocol
COME-LBI protocol revision
COME baseline implementation revision
COME sparse/LBI implementation revision
official COME commit

source revision
source paths/SHA256
dataset
transfer
backbone
seed
BS
target stream identity/hash
primary metric

candidate track
candidate parameter names
candidate scalar count
grouping mode
group count

variant
requested ratio
integer K/K_G

Random:
  child index
  mask seed
  child execution path

selector definition

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

persistent_writeback
host_optimizer_persistent_step
```

LBI：

```text
persistent_writeback = lbi_omega_only
host_optimizer_persistent_step = false
```

Random/Magnitude/Saliency：

```text
persistent_writeback = masked_host_optimizer
lbi_omega = null
```

---

## 22. Formal launch restrictions

本任务不跑 formal。

未来 formal 前必须要求：

```text
formal_protocol = true
debug_max_outer_batches = None
sparse/LBI implementation revision frozen
search protocol frozen
corresponding tuned LBI tuple frozen
```

当前所有 implementation smoke：

```text
debug_only = true
formal = false
```

禁止把 smoke artifacts 标成 formal。

---

## 23. Contracts

新增/完善 canonical COME sparse/LBI contract。

至少逐项验证：

1. same COME baseline objective；
2. same source/stream/singleton/PU/FO；
3. FC candidate tensors=2；
4. FC candidate scalars=524544；
5. FC budgets=262/524/1049；
6. Conv candidate tensors=9；
7. Conv candidate scalars=12845056；
8. Conv out-channel groups=9216；
9. Conv budgets=4/9/18；
10. Conv realized scalar support recorded；
11. filter_connection inaccessible；
12. Random FC true 3 child executions；
13. Random Conv true 3 child executions；
14. Random children fresh source states；
15. Random children fresh optimizer states；
16. Random children same target stream；
17. Magnitude source-static once；
18. Saliency once/valid outer batch；
19. Saliency current-state COME objective；
20. Saliency exact global budget；
21. non-LBI each valid batch one masked host step；
22. non-LBI each valid batch one scheduler update；
23. non-LBI no omega；
24. off-mask host gradients protected；
25. off-mask host momentum/state protected；
26. FC/Conv BN exact frozen；
27. netC exact frozen；
28. LBI local restart each valid outer batch；
29. Stage1 every candidate recomputes current COME objective；
30. no batch-start logits/opinion cache；
31. objective_call_count == stage1_steps_completed + 1 Stage2 call；
32. corrected old-state Z update；
33. strict integer rollback；
34. no top-K repair；
35. Stage1 support never exceeds final strict budget；
36. Stage2 init only masked Stage1 delta；
37. Stage2 exactly one optimizer step；
38. Stage2 fresh optimizer per outer batch；
39. Stage2 current-state COME recomputation；
40. off-mask Stage2 values exact；
41. LBI host persistent optimizer step=0；
42. omega writeback exactly once / processed batch；
43. no EMA；
44. singleton before selector/objective/LBI/optimizer/PU；
45. PU read-only；
46. FO read-only；
47. labels absent from adaptation；
48. identity separates `come.tau` and `lbi.support_threshold`；
49. scientific identity separates LBI tuple/budget/selector；
50. COME baseline regressions PASS；
51. SHOT regressions PASS；
52. IST dense+sparse regressions PASS；
53. NCTTA dense+sparse regressions PASS；
54. `shot_otta/**`, `ist_otta/**`, `nctta_otta/**`, `core/lbi/**` untouched。

Contracts必须检查：

```text
counts
state transitions
artifacts
trajectory multiplicity
```

不能只检查 tensor shape / finite loss。

---

## 24. Independent review

Contracts PASS 后人工逐项 review：

```text
variant dispatch
candidate pool construction
FC global scalar indexing
Conv group construction
Random child multiplicity
Random source reset
Random optimizer reset
Random stream identity
Magnitude source snapshot
Saliency timing
Saliency gradient reuse correctness
masked host optimizer
momentum protection

COME current-state closure
candidate binding
class-count K
come.tau vs LBI support threshold

LBI local restart
old-state Z
prox
support threshold
strict rollback
Stage1 cap
Stage2 masked-delta init
Stage2 objective recompute
Stage2 exact step count
off-mask restore
omega writeback
no host persistent step

BN
netC
singleton
PU
FO
artifact aggregation
identity fallback
debug/formal markers
shared-file diff
```

不要只信 tests PASS。

发现 correctness/provenance bug：

```text
直接修
补 contract
重跑 targeted regression
```

但禁止修改 frozen scientific semantics。

---

## 25. Reviewed short smoke

只有 implementation + contracts + independent review完成后才运行。

固定：

```text
Office-31
D -> A
seed = 2026
BS = 64
rho = 0.001
debug_max_outer_batches = 5
formal = false
```

跑全部 8 个 sparse variants：

```text
come_fc_random
come_fc_magnitude
come_fc_saliency
come_fc_lbi

come_conv_out_random
come_conv_out_magnitude
come_conv_out_saliency
come_conv_out_lbi
```

Random必须是真实：

```text
3 child trajectories each track
```

本轮不跑 dense references，已有 baseline smoke可作为 matched reference。

---

## 26. Smoke 只验证 correctness

不要根据 smoke accuracy改参数。

至少检查：

### All variants

```text
processed valid batches = 5
finite loss/gradient/update
scope exact
BN exact frozen
netC exact
PU/FO read-only
debug_only=true
formal=false
```

### Random

```text
3 real children
fresh source each
fresh optimizer each
same target sample stream
exact budget
```

### Magnitude

```text
source-fixed mask
exact budget
```

### Saliency

```text
support once per batch
exact budget each batch
```

### LBI

```text
support <= K/K_G every batch
stage1 steps recorded
cap hit recorded
rollback recorded
objective call count exact
Stage2 = 1
omega = 1
host persistent step = 0
off-mask exact
```

Smoke accuracy/support utilization：

```text
只报告
不调参
```

---

## 27. Implementation revision freeze

通过：

```text
implementation
contracts
independent review
5-batch smoke
```

后才冻结新的 sparse/LBI implementation revision，例如：

```text
come_otta_sparse_lbi_20260907_v1
```

如现有 naming convention要求其他格式，则遵循现有规范。

然后只更新：

```text
260817_iclr2027-refined/protocol/come-otta/OTTA_COME_LBI_PROTOCOL_20260907_v1.md
```

中的：

```text
COME sparse/LBI implementation revision
status / freeze boundary
```

不得修改 scientific semantics，也不得因此 bump protocol revision。

---

## 28. Regression / hygiene

至少运行：

```text
COME baseline contract
COME sparse/LBI contract
SHOT relevant regressions
IST dense+sparse regressions
NCTTA dense+sparse regressions
shared LBI regressions
git diff --check
```

若全仓 tests 成本可接受，跑全仓。

清理：

```text
*.orig
*.rej
temporary patch files
temporary scripts
```

禁止删除用户现有：

```text
experiment_logs
runs
runs_smoke
protocols
artifacts
```

---

## 29. Stop condition

完成以下后立即停止：

```text
8 sparse variants implemented
contracts PASS
regressions PASS
independent review PASS
Office D->A rho=.001 5-batch reviewed smoke complete
implementation revision frozen
```

### 明确禁止继续

```text
LBI hyperparameter search
COME host search
Office full formal
VisDA formal
其他 Office transfers smoke
rho=.0005 smoke
rho=.002 smoke
search grid expansion
formal launch
paper table generation
```

不要“顺手”进入 tuning。

---

## 30. 最终输出

### 1. Changed files

逐项列出。

### 2. Implementation revision

给出最终 frozen sparse/LBI revision。

### 3. Eight variants

列：

```text
FC Random / Magnitude / Saliency / LBI
Conv Random / Magnitude / Saliency / LBI
```

确认 candidate counts / budgets / grouping。

### 4. Selector semantics

确认：

```text
Random 3 real trajectories
Magnitude source-fixed
Saliency current-state once/batch
```

### 5. COME-LBI semantics

确认：

```text
current-state objective each Stage1 candidate
old-state Z
strict rollback
Stage2 masked-delta init
Stage2 exactly one
omega exactly one
host persistent step zero
off-mask exact
```

### 6. Contracts/regressions

逐项总结。

### 7. Independent review

报告发现/修复的问题。

### 8. Reviewed smoke table

至少：

```text
variant
support trajectory
PU
FO
stage1 steps if LBI
cap hit if LBI
rollback if LBI
objective call counts
scope verdict
```

Random top-level另报：

```text
3-child PU mean/std
3-child FO mean/std
```

### 9. Other-method isolation

明确：

```text
shot_otta/** modified = false
ist_otta/** modified = false
nctta_otta/** modified = false
core/lbi/** modified = false
```

### 10. Unresolved items

只列阻塞下一阶段 LBI search的问题。

### 11. Verdict

只能：

```text
READY_FOR_COME_LBI_SEARCH
```

或：

```text
NOT_READY_FOR_COME_LBI_SEARCH
```

若 READY，立即停止，不开始 search。
