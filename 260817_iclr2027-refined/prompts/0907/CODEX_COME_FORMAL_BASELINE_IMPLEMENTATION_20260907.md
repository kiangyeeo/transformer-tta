# CODEX — COME Formal Baseline Implementation

目标：把已经通过 `COME_FEASIBILITY_GO` 的临时 feasibility implementation，严格按照正式 COME baseline protocol 升级为 **formal controlled dense baseline implementation**。

当前只完成 COME dense baseline。  
**不要实现 Random / Magnitude / Saliency / LBI。**

---

## 0. 必读规范

先完整读取：

```text
260817_iclr2027-refined/protocol/come-otta/OTTA_COME_BASELINE_PROTOCOL_20260907_v1.md
```

同时只读参考：

```text
260817_iclr2027-refined/protocol/come-otta/OTTA_COME_LBI_PROTOCOL_20260907_v1.md
```

第二份只用于保证未来接口兼容；**本任务禁止实现其中 sparse/LBI 部分**。

官方 COME source-of-truth：

```text
repo: BlueWhaleLab/COME
audited commit:
409a19b71f62c765b1a5be62347a9455524ec176
```

当前 feasibility 代码可作为工程起点：

```text
260817_iclr2027-refined/come_otta/**
260817_iclr2027-refined/configs/come_fast_feasibility_20260907_v1.yaml
260817_iclr2027-refined/tests/come_fast_feasibility_contract_test.py
```

但 feasibility gate 不是 formal protocol；有冲突时，以 baseline protocol 为准。

---

## 1. 最高优先级：禁止修改其他方法科学语义

严禁修改：

```text
shot_otta/**
ist_otta/**
nctta_otta/**
core/lbi/**
```

严禁修改既有 SHOT / IST / NCTTA protocol 或 config semantics。

如果必须修改 shared top-level 文件，例如：

```text
train.py
experiment_identity.py
protocol_constants.py
```

只能做 **COME-specific additive extension**。

不得改变已有：

```text
SHOT dispatch
IST dispatch
NCTTA dispatch
shared source loading
shared stream semantics
shared LBI engine
```

任务开始前记录：

```bash
git status --short
git diff --name-only
```

任务结束后必须重新检查：

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

如果发现误改，先恢复误改再继续。

---

## 2. 正式 dense variants 只有三个

只实现/正式化：

```text
come_full_dense
come_fc_module_dense
come_conv_module_dense
```

不要增加：

```text
come_native_norm
come_random
come_magnitude
come_saliency
come_lbi
任何 sparse variant
```

---

## 3. COME objective 必须严格保持 audited semantics

Formal objective：

```text
p = 2
tau = 1
opinion_eps = 1e-7
K = dataset class count
```

数据集：

```text
Office-31: K = 31
VisDA-C:   K = 12
```

必须 assert：

```text
logits.shape[-1] == K
```

禁止 Office / VisDA COME 路径出现 ImageNet-specific：

```text
K = 1000
+1000
uncertainty numerator = 1000
```

Constrained logits 必须保持官方 detach：

```python
norm = torch.norm(logits, p=2, dim=-1, keepdim=True)
constrained = logits / norm * norm.detach() * 1.0
```

以 audited official implementation 的精确写法为准。

禁止：

```text
额外 epsilon/clamp 到 norm
删除 norm.detach()
detach entire logits/objective
pseudo-label
feature geometry
teacher
memory
EMA
confidence filter
extra regularizer
```

Evidence / strength / belief / uncertainty / opinion entropy 严格按 protocol。

若出现 NaN/Inf，fail loudly。

---

## 4. Benchmark / source / stream 全部继承 common controlled substrate

严格保持：

```text
source revision = nips2026_shot_otta_uda_source_v1
same source F/B/C as SHOT
same netF -> netB -> netC
netC frozen

Office:
  backbone = ResNet-50
  BS = 64
  workers = 4
  K = 31

VisDA-C:
  backbone = ResNet-101
  BS = 256
  workers = 4
  K = 12

formal seed = 2026
one target pass
drop_last = false
same target sample order
same outer-batch partition
same source/eval transforms
same singleton policy
same PU / FO
```

Singleton：

```text
actual outer batch size == 1
-> skip before COME objective / scheduler / optimizer / BN update / PU
```

FO 仍覆盖完整 target set。

---

## 5. 三个 dense update scopes

### 5.1 `come_full_dense`

Trainable：

```text
all netF
all netB
```

Frozen：

```text
netC
```

BN semantics 必须与 current SHOT full-dense controlled branch完全一致。

### 5.2 `come_fc_module_dense`

只允许：

```text
netB.bottleneck.weight
netB.bottleneck.bias
```

必须得到：

```text
2 tensors
524544 scalars
```

其余全部 frozen。

所有 BN：

```text
parameters frozen
running buffers frozen
module eval behavior
```

`netC` frozen。

### 5.3 `come_conv_module_dense`

只允许 formal layer4 9 Conv weights：

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
9 tensors
12845056 scalars
```

其余全部 frozen。

所有 BN frozen/eval。

`netC` frozen。

---

## 6. Optimizer / scheduler 不允许重新设计

Formal COME dense 使用 common controlled substrate。

Office：

```text
base LR = 0.01
netF multiplier = 0.1
netB multiplier = 1.0
```

VisDA：

```text
base LR = 0.001
netF multiplier = 0.1
netB multiplier = 1.0
```

SGD：

```text
momentum = 0.9
weight_decay = 0.001
nesterov = true
```

Outer polynomial schedule：

```text
lr_t = lr_0 * (1 + 10 * t / T)^(-0.75)
```

其中：

```text
t = processed outer-stream timeline
T = full formal target-loader timeline
```

即使 debug 只跑 5 batches，也不能把 `T` 改成 5。

每 valid outer batch：

```text
scheduler update count = 1
COME objective count = 1
optimizer step count = 1
```

不要因为 feasibility smoke 好看而改 LR / scheduler / optimizer。

---

## 7. PU / FO 必须只读

PU：

```text
same current batch
post-update
read-only
```

不得修改：

```text
parameters
BN buffers
optimizer state
scheduler state
method state
```

`come_full_dense` 如果沿用 train-compatible PU forward，必须复用当前 SHOT 的 BN-state preservation semantics。

FC/Conv BN始终 frozen。

FO：

```text
final persistent model
netF.eval()
netB.eval()
netC.eval()
full target
read-only
```

Metrics：

```text
Office primary = overall sample accuracy
VisDA primary  = fixed-12-class mAcc
```

VisDA同时保留 secondary：

```text
overall accuracy
per-class accuracy
worst class
class-wise std
```

---

## 8. Formal config / provenance / identity

把当前 feasibility-only config 通道正式化。

需要支持：

```text
Office all 6 transfers
VisDA Synthetic/Train -> Real/Validation
formal_protocol = true
debug_max_outer_batches = None
```

Formal run 如设置：

```text
debug_max_outer_batches != None
```

必须 fail closed。

Formal baseline：

```text
save_model = false
no partial resume
```

Scientific identity 至少包含：

```text
protocol_revision = OTTA_COME_BASELINE_PROTOCOL_20260907_v1
implementation_revision
official_come_commit
source_checkpoint_revision
source checkpoint resolved paths/SHA256

method = COME
variant
dataset
transfer
backbone
seed
outer batch size
target stream identity/hash
singleton policy

candidate scope
candidate scalar count
BN semantics

come.p = 2
come.tau = 1
come.class_count
come.opinion_eps = 1e-7
come.norm_detach_semantics

optimizer
base LR
netF/netB multipliers
momentum
weight decay
Nesterov
scheduler
primary metric
```

不要把 gate 的：

```text
debug_only
gate_5b
Office D->A only
```

等临时语义带进 formal identity。

---

## 9. Implementation revision

完成 implementation + contract + independent review + formal-baseline 5-batch smoke 后，冻结一个新的 COME baseline implementation revision，例如：

```text
come_otta_baseline_20260907_v1
```

如果仓库现有 revision 命名规范要求其他格式，遵循现有规范，但必须：

```text
唯一
明确
进入 protocol
进入 identity
进入 artifact
```

不要提前冻结 revision；通过验收后再写入 protocol：

```text
260817_iclr2027-refined/protocol/come-otta/OTTA_COME_BASELINE_PROTOCOL_20260907_v1.md
```

只更新 implementation revision/status，不改变任何 scientific semantics。

---

## 10. Contract suite

把 feasibility contract 升成 formal COME baseline contract。

至少验证：

1. official COME commit identity；
2. independent direct-reference loss；
3. independent direct-reference gradient；
4. norm detach exact position；
5. p=2；
6. tau=1；
7. opinion_eps=1e-7 official position；
8. Office K=31；
9. VisDA K=12；
10. logits dimension == K；
11. no Office/VisDA hard-coded 1000；
12. same source F/B/C as SHOT；
13. deterministic pre-adaptation source logits match；
14. netC byte-identical；
15. full-dense scope exact；
16. FC 2 tensors / 524544 scalars；
17. Conv 9 tensors / 12845056 scalars；
18. FC/Conv BN parameters and buffers byte-identical；
19. objective calls exactly once per valid batch；
20. scheduler exactly once per valid batch；
21. optimizer exactly once per valid batch；
22. singleton early skip；
23. PU read-only；
24. FO read-only；
25. labels absent from adaptation objective；
26. Office metric correct；
27. VisDA fixed-12-class mAcc correct；
28. formal debug-limit rejection；
29. formal no-resume semantics；
30. scientific identity/provenance complete；
31. existing SHOT regressions PASS；
32. existing IST regressions PASS；
33. existing NCTTA regressions PASS；
34. other-method source files untouched。

Contract 必须检查实际 state transition，而不只是：

```text
requires_grad
tensor shapes
finite loss
```

---

## 11. Independent manual review

Tests PASS 后必须人工 review：

```text
official COME objective mapping
norm detach
K resolution
VisDA K=12
variant dispatch
candidate scope
BN train/eval behavior
optimizer parameter groups
scheduler timing
singleton
PU
FO
label boundary
formal config
artifact/provenance
experiment identity
shared-file diff
```

特别检查：

```text
feasibility-only Office D->A / 5-batch restriction 是否已从 formal path 清除
```

以及：

```text
没有修改 SHOT / IST / NCTTA / core LBI behavior
```

不要只因为 tests PASS 就跳过人工 review。

---

## 12. Reviewed short smoke

完成 implementation + contracts + review 后，只跑：

```text
Office D->A
seed = 2026
BS = 64
debug_max_outer_batches = 5
formal = false
```

三个：

```text
come_full_dense
come_fc_module_dense
come_conv_module_dense
```

要求：

```text
processed valid batches = 5
objective calls = 5
scheduler updates = 5
optimizer steps = 5
loss/grad/update finite
scope exact
netC exact
FC/Conv BN exact
PU/FO read-only
```

保留 detached diagnostics：

```text
predicted_class_count
dominant_class_ratio
softmax entropy
COME uncertainty
relative update norm
```

Smoke accuracy只作为 correctness/stability diagnostic：

```text
禁止根据 smoke accuracy 调参
禁止改 COME objective
禁止改 LR
```

---

## 13. Regression / hygiene

至少运行：

```text
COME formal baseline contract
SHOT smoke/contracts
IST dense/sparse relevant regressions
NCTTA dense/sparse relevant regressions
git diff --check
```

若全仓 tests 成本合理，则跑全仓。

不要修改 unrelated lint/style 代码。

清理：

```text
*.orig
*.rej
临时 patch files
临时 debug scripts
```

不要删除用户已有 artifacts / logs / protocols。

---

## 14. Stop condition

完成以下全部后停止：

```text
formal COME baseline implementation
formal configs
formal scientific identity/provenance
formal baseline contract
regressions
independent review
3 x Office D->A 5-batch reviewed smoke
implementation revision freeze
```

然后停止。

### 本任务明确禁止

```text
COME Random
COME Magnitude
COME Saliency
COME LBI
COME Group-LBI
sparse.py
lbi.py
sparse_trainer.py
LBI search
VisDA formal runs
Office full formal runs
hyperparameter search
native COME
```

不要“顺手继续”实现 child protocol。

---

## 15. 最终输出格式

按下面顺序汇报：

### 1. Changed files

逐文件列出。

### 2. Official COME mapping

确认：

```text
p=2
tau=1
norm detach
evidence
strength
belief
uncertainty
opinion entropy
K=dataset class count
Office K=31
VisDA K=12
```

### 3. Formal dense variants

确认：

```text
come_full_dense
come_fc_module_dense
come_conv_module_dense
```

及真实 trainable tensor/scalar counts。

### 4. Formal benchmark support

确认：

```text
Office six transfers
VisDA one transfer
formal seed / BS / metrics
singleton
PU / FO
```

### 5. Implementation revision

给出最终冻结 revision。

### 6. Contracts/regressions

逐项总结 PASS/FAIL。

### 7. Independent review

说明实际检查了什么，以及是否修复了任何问题。

### 8. Reviewed 5-batch smoke

表格：

```text
variant
PU
FO
predicted_class_count trajectory
dominant_class_ratio trajectory
softmax entropy trajectory
COME uncertainty trajectory
relative update norm trajectory
scope verdict
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

只列真正阻塞后续 sparse/LBI implementation 的问题。

### 11. Verdict

只能二选一：

```text
READY_FOR_COME_SPARSE_LBI_IMPLEMENTATION
```

或：

```text
NOT_READY_FOR_COME_SPARSE_LBI_IMPLEMENTATION
```

如果前者，立即停止，不继续实现 sparse/LBI。
