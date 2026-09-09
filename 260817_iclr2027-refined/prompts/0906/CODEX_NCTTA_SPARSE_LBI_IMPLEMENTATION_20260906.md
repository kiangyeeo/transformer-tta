# CODEX — NCTTA P2/P3 Sparse + LBI Implementation

在当前 `260817_iclr2027-refined/` 上实现完整 NCTTA FC/Conv sparse family。当前 NCTTA dense/native baseline 已完成并通过 contract/review；本任务不做 parameter search，不跑 formal。

## 1. 开始前必须读取

按顺序：

1. `protocol/nctta-otta/OTTA_NCTTA_LBI_PROTOCOL_20260906_v1.md`
2. `protocol/nctta-otta/OTTA_NCTTA_BASELINE_PROTOCOL_20260906_v2.md`
3. 当前 `nctta_otta/objective.py`、`trainer.py`、`config.py`
4. 当前 `ist_otta/sparse.py`、`ist_otta/lbi.py`、`ist_otta/trainer.py`，只作为已经验证过的工程结构模板
5. 当前 `core/lbi/engine.py`、`groups.py`、`diagnostics.py`
6. frozen refined FC/Conv protocols

禁止从旧 NCTTA/IST-LBI 语义凭记忆实现。

## 2. Hard constraints

- 不修改 `shot_otta/**`。
- 不改变已有 SHOT、IST、NCTTA dense/native scientific semantics。
- 不修改 `nctta_otta/objective.py` 的 audited loss semantics，除非发现明确 correctness bug；若发现必须停止并报告。
- `nctta_native_norm` 不进入 sparse family。
- controlled FC/Conv 所有 BN frozen，`netC` frozen。
- 不做 tuning/formal。

## 3. 实现的 8 个 variants

FC：

```text
nctta_fc_random
nctta_fc_magnitude
nctta_fc_saliency
nctta_fc_lbi
```

Conv：

```text
nctta_conv_out_random
nctta_conv_out_magnitude
nctta_conv_out_saliency
nctta_conv_out_lbi
```

建议新增：

```text
nctta_otta/sparse.py
nctta_otta/lbi.py
```

并最小扩展：

```text
nctta_otta/config.py
nctta_otta/trainer.py
train.py
protocol_constants.py
experiment_identity.py
configs/*
tests/*
```

尽量复用当前 objective-agnostic shared utilities，不复制一套 LBI engine。

## 4. Candidate / budgets

FC：

```text
candidate = netB.bottleneck.weight + bias
N = 524544
rho = 0.0005 / 0.001 / 0.002
K = 262 / 524 / 1049
global scalar counting
```

Conv：

```text
candidate = full netF.layer4 nine Conv weights
grouping = out_channel only
scalar count = 12845056
group count = 9216
rho_G = 0.0005 / 0.001 / 0.002
K_G = 4 / 9 / 18
```

`filter_connection` 禁止。

## 5. NCTTA current-state objective：实现重点

NCTTA 和 IST 最大区别：

> 不允许在 outer batch 开头冻结 NCTTA target。

每个 Saliency gradient、每个 LBI Stage-1 candidate iteration、Stage-2 step 都必须从当前 model state重新：

```text
netF -> netB -> netC
post-netB feature
effective frozen netC.fc.weight
top-k
q_dist
q_prob
hybrid q
NC loss
entropy
entropy filter
detached entropy/FCA weights
final NCTTA loss
```

LBI Stage-1：

$$
g^k=
\nabla_{\Theta_\Delta}
L_{\rm NCTTA}(B_t;\Theta_t+\Theta_\Delta^k).
$$

禁止：

```text
source feature anchor
batch-start fixed feature
fixed top-k
fixed q
fixed selected samples
extra detach
```

zero selected / NaN / Inf 必须 fail loudly。

## 6. `nu` 命名必须 fail-safe

NCTTA objective 和 LBI 都有 `nu`：

```text
nctta.nu
lbi.nu
```

要求：

- config namespace分离；
- CLI/override若支持，命名不得歧义；
- artifact/identity分别记录；
- tests 显式验证改 `nctta.nu` 不会改 `lbi.nu`，反之亦然；
- 禁止 provenance 中裸写无法判断归属的 `nu`。

## 7. Random / Magnitude / Saliency

### Random

每个 budget必须真实运行3个 fresh child trajectories：

```text
202600
202601
202602
```

每个 child：

```text
fresh source model
fresh host optimizer
same formal target stream
only mask differs
exact K / K_G
mask fixed whole stream
```

Top-level artifact聚合 child mean/std。必须验证 execution multiplicity=3，不只 metadata=3。

### Magnitude

FC：

$$
s_j=|\theta_j^{src}|.
$$

Conv：

$$
s_g=\|W_g^{src}\|_2.
$$

global exact top-K/K_G，source时计算一次，整条 stream固定。

### Saliency

每个 valid outer batch只选一次。

FC：

$$
s_j=|\theta_j\nabla_{\theta_j}L_{\rm NCTTA}|.
$$

Conv：

$$
s_g=
\|W_g\odot\nabla_{W_g}L_{\rm NCTTA}\|_2.
$$

support当前 outer batch固定，下一 batch重算。

非-LBI sparse使用 parent controlled NCTTA host optimizer/scheduler，每 valid outer batch exactly one masked optimizer step。Off-mask value + momentum/state必须保护。

## 8. LBI

直接调用 current corrected refined engine。

必须保持：

```text
local restart once / valid outer batch
corrected old-state Z
tau=1e-4
strict integer rollback
no top-K repair
stage1_max_steps=3000
masked-delta Stage2 init
stage2_steps=1
fresh local Stage2 SGD
momentum=.9
wd=.001
Nesterov=true
no Stage2 LR schedule
off-mask exact preservation
omega-only persistent writeback
```

Stage-2 objective必须在 Stage-2 current state重新完整计算 NCTTA objective。

NCTTA-LBI 不执行 host optimizer persistent step；每 processed batch exactly one omega writeback。

## 9. Provenance / revision

新增并冻结新的 sparse implementation revision，例如：

```text
nctta_otta_sparse_lbi_20260906_v1
```

但只有在实现、contracts、review完成后才能正式写入 frozen revision。

Protocol：

```text
OTTA_NCTTA_LBI_PROTOCOL_20260906_v1
```

artifact至少完整记录 protocol要求的 NCTTA namespace、LBI namespace、track、budget、selector、Random child、support、writeback mode。

Formal launch继续 blocked，因为 dataset-specific NCTTA objective tuple 与 LBI search/tuned tuples 尚未冻结。

## 10. Contracts

至少新增一个 NCTTA sparse/LBI contract，覆盖 protocol 第30节全部项目，特别检查：

```text
Random真的执行3 children
each child fresh source/optimizer
NCTTA objective every LBI candidate recomputed
top-k/q/filter trajectory可观测
no frozen target cache
nctta.nu != lbi.nu namespace
FC/Conv exact candidate counts/budgets
Saliency once/batch
LBI once/batch
corrected old-state Z
strict rollback
Stage2 exactly 1
Stage2 current-state objective recompute
off-mask exact
BN exact frozen
netC exact frozen
non-LBI host step exactly 1
LBI host persistent step = 0
omega exactly 1
singleton skip
PU/FO read-only
identity/provenance
```

现有：

```text
SHOT tests
IST dense/sparse tests
NCTTA dense/native tests
```

全部 regression PASS。

## 11. Independent review

Tests PASS 后人工逐项看：

```text
dispatch
candidate pool
selector
Random child loop
fresh reset
magnitude snapshot
saliency timing
NCTTA closure
candidate-state model binding
objective recomputation
nctta.nu/lbi.nu
LBI restart
rollback
Stage2 init/step
writeback
BN/netC/off-mask
singleton
PU/FO
artifact aggregate
identity fallback
```

发现问题直接修并升 implementation revision，再重跑 contracts。

## 12. Short smoke

Contract + independent review后：

```text
Office D->A
seed=2026
BS64
5 valid outer batches
rho=0.001 debug budget
all 8 sparse variants
Random真实3 children
```

LBI 使用当前项目已有的 debug/test tuple；如果 IST smoke已有通用 debug tuple，优先复用作为 plumbing test。必须标：

```text
debug_only = true
formal = false
```

Smoke不根据 accuracy/utilization调参数，只验：

```text
GPU/data path
real artifacts
objective calls/recomputation
support
strict budget/rollback
Stage2 count
omega count
Random child count
scope preservation
PU/FO
```

## 13. Stop

完成以下后停止：

```text
8 variants implemented
contracts PASS
all regressions PASS
independent review PASS
D->A 5-batch smoke PASS
```

不要：

```text
parameter search
dataset-specific NCTTA objective tuning
formal Office
formal VisDA
改 source feature
加 stabilization loss
改 Stage2 steps
改 LBI dynamics
```

最终输出：

1. changed files；
2. implementation revision；
3. 8 variant semantics；
4. NCTTA current-state objective如何保证每次 candidate重算；
5. `nctta.nu` / `lbi.nu`隔离；
6. contracts/regressions；
7. independent review findings；
8. smoke artifacts/results；
9. unresolved items；
10. verdict：

```text
READY_FOR_UNIFIED_NCTTA_IST_SEARCH
```

或

```text
NOT_READY_FOR_UNIFIED_NCTTA_IST_SEARCH
```
