# OTTA_CONV_LBI_PROTOCOL_20260826_v1

**Status:** Frozen protocol  
**Scope:** SHOT-OTTA + ResNet layer4 convolutional controlled adaptation + Group-Lasso Sparse Delta Learning / LBI  
**Target submission:** ICLR, late September 2026  
**Parent evaluation protocol:** `OTTA_FC_LBI_PROTOCOL_20260817_v1`  
**Scientific implementation revision:** `iclr2027_refined_conv_20260826_v1`  
**Protocol date:** 2026-08-26

> This document defines the new **Conv-layer OTTA/LBI track**.  
> Benchmark/evaluation semantics that are independent of the adaptation scope are inherited from the frozen FC protocol.  
> Conv-specific semantics are defined here and must not silently modify the frozen FC track.

---

## 1. Scope and design principle

This protocol applies to:

```text
OTTA method       : SHOT-OTTA
adaptation scope  : ResNet layer4 convolutional weights
sparse method     : Group-Lasso Sparse Delta Learning / LBI
datasets          : Office-31, VisDA-C
formal seed       : 2026
```

The Conv track is designed as the structured counterpart of the current FC track:

```text
FC-LBI   : element-wise sparse adaptation
Conv-LBI : group-wise structured sparse adaptation
```

Both tracks share the same outer OTTA pipeline:

```text
batch-local restart
-> sparse structure discovery
-> thresholded support
-> strict budget control / rollback
-> masked-delta Stage-2 refinement
-> persistent omega writeback
```

The Conv track studies **structured sparse adaptation support**, not physical network pruning. A group that is not selected is not adapted; its pretrained source weight is not removed. No inference-time FLOP reduction claim is allowed unless a separate physical graph-restructuring/export experiment is performed.

---

## 2. Inherited benchmark / evaluation invariants

Unless explicitly overridden below, the following are inherited unchanged from the FC protocol.

### 2.1 Office-31

| Setting | Value |
|---|---|
| Dataset | Office-31 |
| Backbone | ResNet-50 |
| Classes | 31 |
| Batch size | 64 |
| DataLoader workers | 4 |
| Target passes | 1 |
| `drop_last` | false |
| Formal seed | 2026 |
| Primary PU / FO metric | sample-level overall accuracy |
| Dataset aggregation | equal mean over 6 directed transfers |

Formal transfers:

```text
A -> D
A -> W
D -> A
D -> W
W -> A
W -> D
```

### 2.2 VisDA-C

| Setting | Value |
|---|---|
| Dataset | VisDA-C |
| Backbone | ResNet-101 |
| Classes | 12 |
| Batch size | 256 |
| DataLoader workers | 4 |
| Target passes | 1 |
| `drop_last` | false |
| Formal seed | 2026 |
| Primary PU / FO metric | fixed-12-class mean per-class accuracy (mAcc) |

Formal transfer:

```text
Synthetic / Train -> Real / Validation
```

Also retain:

```text
overall accuracy
12-class accuracy
worst-class accuracy / class name
class-wise standard deviation
```

### 2.3 Target stream and transforms

For a fixed:

```text
(dataset, transfer, seed=2026)
```

all variants must observe the same target stream order and augmentation RNG.

Online transform:

```text
Resize(256)
RandomCrop(224)
RandomHorizontalFlip()
ImageNet normalization
```

Final evaluation transform:

```text
Resize(256)
CenterCrop(224)
ImageNet normalization
```

### 2.4 Numerical precision

Use the repository's current non-AMP path consistently for all formal comparisons. Do not enable mixed precision for only a subset of methods.

---

## 3. Source checkpoints and SHOT objective

All formal Conv variants must start from the same frozen SHOT source checkpoints used by the FC track.

Source checkpoint revision:

```text
nips2026_shot_otta_uda_source_v1
```

Every formal run must record exact resolved paths for source F/B/C checkpoints.

SHOT objective is unchanged:

$$
\mathcal{L}_{\mathrm{SHOT}}
=
0.3\mathcal{L}_{\mathrm{pseudo}}
+
\mathcal{L}_{\mathrm{ent}}
+
\mathcal{L}_{\mathrm{div}}.
$$

Pseudo-labels are causal and current-batch only. Target labels are forbidden for adaptation and hyperparameter selection unless a future validation protocol is explicitly defined before tuning.

---

## 4. Native SHOT reference versus controlled Conv family

Two comparison roles must remain distinct.

### 4.1 Native SHOT reference

```text
full_dense
```

Native SHOT preserves its method-native update scope and BN behavior:

```text
netF adapts
netB adapts
netC frozen
native BN behavior preserved during adaptation
```

This is a native-method performance reference, not a matched-support Conv selector baseline.

### 4.2 Controlled Conv family

The controlled Conv family is:

```text
conv_module_dense
conv_out_random
conv_out_magnitude
conv_out_saliency
conv_out_lbi
conv_filter_random
conv_filter_magnitude
conv_filter_saliency
conv_filter_lbi
```

For every controlled Conv variant:

```text
only the declared layer4 Conv candidate weights may change persistently
all other netF parameters are frozen
netB is frozen
netC is frozen
all BN affine parameters are frozen
all BN running statistics are frozen
BN modules use frozen / evaluation behavior
```

---

## 5. Conv candidate scope

The first formal Conv track uses only the 9 convolutional layers in ResNet `layer4`:

```text
netF.layer4.0.conv1
netF.layer4.0.conv2
netF.layer4.0.conv3
netF.layer4.1.conv1
netF.layer4.1.conv2
netF.layer4.1.conv3
netF.layer4.2.conv1
netF.layer4.2.conv2
netF.layer4.2.conv3
```

For both ResNet-50 and ResNet-101, these layer4 convolutional shapes are:

| Layer | $C_{in}$ | $C_{out}$ | Kernel |
|---|---:|---:|---:|
| layer4.0.conv1 | 1024 | 512 | 1x1 |
| layer4.0.conv2 | 512 | 512 | 3x3 |
| layer4.0.conv3 | 512 | 2048 | 1x1 |
| layer4.1.conv1 | 2048 | 512 | 1x1 |
| layer4.1.conv2 | 512 | 512 | 3x3 |
| layer4.1.conv3 | 512 | 2048 | 1x1 |
| layer4.2.conv1 | 2048 | 512 | 1x1 |
| layer4.2.conv2 | 512 | 512 | 3x3 |
| layer4.2.conv3 | 512 | 2048 | 1x1 |

Total candidate Conv scalar weights:

$$
N_{\mathrm{conv}} = 12,845,056.
$$

No Conv bias is included because the standard ResNet convolutional layers use bias-free convolution in this candidate scope.

Layer1/2/3 and stem conv are out of scope for protocol v0.

---

## 6. Two formal Conv grouping modes

Let a candidate convolutional tensor be:

$$
W^{(l)}\in
\mathbb{R}^{C_{out}\times C_{in}\times K_h\times K_w}.
$$

Both grouping modes use the same full 4D batch-local LBI states:

$$
\Theta_\Delta^{(l)},
Z^{(l)},
\Gamma^{(l)}
\in
\mathbb{R}^{C_{out}\times C_{in}\times K_h\times K_w}.
$$

Only the group partition and group proximal operator semantics differ from element-wise FC-LBI.

### 6.1 Out-channel grouping

Each output-channel slice is one group:

$$
G_{l,o}
=
W^{(l)}[o,:,:,:].
$$

The global group pool is:

$$
\mathcal{G}_{out}
=
\bigcup_l\{G_{l,o}\}.
$$

Total number of out-channel groups:

$$
|\mathcal{G}_{out}|=9,216.
$$

### 6.2 Filter-connection grouping

Each input-output kernel connection is one group:

$$
G_{l,o,i}
=
W^{(l)}[o,i,:,:].
$$

The global group pool is:

$$
\mathcal{G}_{filter}
=
\bigcup_l\{G_{l,o,i}\}.
$$

Total number of filter-connection groups:

$$
|\mathcal{G}_{filter}|=6,553,600.
$$

### 6.3 No cross-layer shared gate

The current formal Conv definition does **not** use a shared 1D channel gate across adjacent convolutional layers.

For example, an out-channel group in `conv1` is not forcibly tied to the corresponding input channel of `conv2`.

The structured unit is the declared group inside the current candidate weight tensor.

---

## 7. Group-Lasso penalty

The main Conv-LBI method uses the unweighted Group-Lasso penalty:

$$
J(\Gamma)
=
\sum_{g\in\mathcal G}
\|\Gamma_g\|_2.
$$

This is the formal main method for both grouping modes.

A group-size-normalized / weighted variant, e.g.

$$
J_w(\Gamma)
=
\sum_g w_g\|\Gamma_g\|_2,
$$

is **not** part of the main protocol. If evaluated, it must be declared as a separate ablation before running formal experiments.

---

## 8. Restart Group-LBI algorithmic semantics

The Conv track inherits all corrected LBI semantics from the refined FC implementation.

For each online batch $B_t$, let the persistent Conv candidate state be $\Theta_t$.

### 8.1 Batch-local restart

At every online batch:

$$
\Theta_\Delta^0=0,
\qquad
Z^0=0,
\qquad
\Gamma^0=0.
$$

The persistent model $\Theta_t$ is carried across batches; the local LBI variables are not.

### 8.2 Stage 1 task gradient

At step $k$:

$$
g^k
=
\nabla_{\Theta_\Delta}
\mathcal L_{\mathrm{SHOT}}
\left(
B_t;
\Theta_t+\Theta_\Delta^k
\right).
$$

### 8.3 Correct old-state coupling

Define:

$$
c^k
=
\frac{\Theta_\Delta^k-\Gamma^k}{\nu}.
$$

Then update:

$$
\Theta_\Delta^{k+1}
=
\Theta_\Delta^k
-
\alpha\kappa
\left(
g^k+c^k
\right),
$$

$$
Z^{k+1}
=
Z^k+
\alpha c^k.
$$

The $Z$ update must use the old-state coupling based on:

$$
(\Theta_\Delta^k,\Gamma^k),
$$

not the newly updated $\Theta_\Delta^{k+1}$.

### 8.4 Group proximal operator

For every group $g\in\mathcal G$:

$$
\Gamma_g^{k+1}
=
\kappa
\left(
1-
\frac{1}{\|Z_g^{k+1}\|_2}
\right)_+
Z_g^{k+1}.
$$

Implementation must handle the zero-norm case numerically safely without changing the mathematical support semantics.

---

## 9. Thresholded group support

The support threshold is inherited from the refined FC track:

$$
\tau=10^{-4}.
$$

A group is selected iff:

$$
M_g^k
=
\mathbf 1
\left[
\|\Gamma_g^k\|_2\ge\tau
\right].
$$

The same thresholded support definition must be used for:

```text
Stage-1 group support counting
realized group density
strict budget checking
final selected group mask
masked-delta Stage-2 initialization
Stage-2 trainable support
```

Do not use:

```text
Gamma != 0
Theta_delta != 0
gradient != 0
```

as the formal support definition.

---

## 10. Sparse budget semantics

### 10.1 Budget unit

The main Conv budget is defined on **group count**, because Group Lasso selects groups rather than individual scalar weights.

For a grouping mode with global group pool $\mathcal G$:

$$
K_G
=
\left\lfloor
\rho_G |\mathcal G|
\right\rfloor.
$$

The exact formal $\rho_G$ grid is:

```text
TBD_AFTER_CORRECTNESS_REFACTOR_AND_BUDGET_PILOT
```

### 10.2 Global budget

Budgeting is global over all 9 candidate Conv layers.

Forbidden:

```text
per-layer K
minimum-one-group-per-layer
independent budget per tensor
hidden budget slack
ceil-based K
```

### 10.3 Strict upper bound and rollback

At Stage-1 step $k$, let:

$$
n_k
=
\sum_{g\in\mathcal G}M_g^k.
$$

Then:

```text
n_k < K_G : current state is feasible; save it and continue
n_k = K_G : accept current state and stop Stage 1
n_k > K_G : reject current state; rollback to the latest feasible state and stop
```

No Group-LBI top-$K$ trimming is allowed after overshoot.

Therefore the final support may satisfy:

$$
|M^\star|<K_G.
$$

This is scientifically valid.

### 10.4 Cross-grouping fairness diagnostics

Because Conv groups have unequal scalar sizes, group-count sparsity does not imply equal scalar-count sparsity.

Every run must additionally record:

$$
N_{\mathrm{selected\ scalar}}
=
\sum_{g:M_g=1}|g|,
$$

$$
\rho_{\mathrm{scalar}}
=
\frac{N_{\mathrm{selected\ scalar}}}{N_{\mathrm{conv}}}.
$$

Also record:

```text
selected_group_count
realized_group_ratio
selected_scalar_count
realized_scalar_ratio
```

The paper must not describe out-channel and filter-connection experiments as equal-cost solely because they share the same nominal group ratio $\rho_G$.

Exact cross-grouping budget-matching policy is TBD and must be fixed before formal tuning.

---

## 11. Stage-2 masked refinement

Let the final Stage-1 state be:

$$
(\Theta_\Delta^\star,\Gamma^\star,M^\star).
$$

### 11.1 Masked-delta initialization

Stage 2 must initialize from:

$$
\Theta_{t,2}^{0}
=
\Theta_t
+
M^\star\odot\Theta_\Delta^\star.
$$

The group-level mask is broadcast to the corresponding 4D Conv coordinates.

Do not densely apply off-mask Stage-1 delta values.

### 11.2 Strict mask-out freezing

During Stage 2:

```text
only selected Conv coordinates may change
off-mask gradients are masked
off-mask parameter values are restored / frozen after optimizer steps
batch-local sparse optimizer state must not leak updates to off-mask coordinates
```

### 11.3 Stage-2 optimizer

Inherit the FC refined defaults:

| Setting | Value |
|---|---:|
| Optimizer | SGD |
| Momentum | 0.9 |
| Weight decay | 0.001 |
| Nesterov | true |
| Stage-2 steps | 1 |
| LR schedule | none; fixed Stage-2 LR |

`stage2_lr` is a tunable LBI parameter and must not use a batch-local SHOT decay schedule.

---

## 12. Persistent accumulation

Let $\widetilde\Theta_t$ be the refined Stage-2 state.

Persistent writeback is:

$$
\Theta_{t+1}
=
(1-\omega)\Theta_t
+
\omega\widetilde\Theta_t.
$$

The next online batch starts from $\Theta_{t+1}$, while:

$$
\Theta_\Delta,
Z,
\Gamma
$$

are reset to zero.

Therefore the structured support is rediscovered independently for every online batch.

---

## 13. Controlled Conv baseline selector definitions

All sparse Conv baselines must operate on the same global group pool as the corresponding Conv-LBI grouping mode and use exact-$K_G$ selection when $K_G>0$.

### 13.1 `conv_module_dense`

All:

$$
N_{\mathrm{conv}}=12,845,056
$$

candidate Conv scalars are trainable.

This is the matched-module dense Conv reference.

### 13.2 Random group selection

For each grouping mode and sparse budget, use three independent deterministic masks inside the same formal seed condition.

Each mask uniformly samples exactly $K_G$ groups from the global group pool.

Report:

```text
accuracy = 3-mask mean
formal comparable runtime = mean single-mask runtime
operational runtime = 3-mask total cost, stored separately
```

### 13.3 Magnitude group selection

For each group $g$ at the source checkpoint:

$$
s_g
=
\|W_g\|_2.
$$

Select the global top-$K_G$ groups.

The mask remains fixed for the target stream.

### 13.4 Saliency group selection

At each online adaptation step:

$$
s_g
=
\|W_g\odot \nabla_{W_g}\mathcal L_{\mathrm{SHOT}}\|_2.
$$

Select the global top-$K_G$ groups.

The mask may change over time.

### 13.5 Selector matching rule

Within each grouping mode and nominal budget, comparisons among:

```text
Random
Magnitude
Saliency
LBI
```

must use the same global group candidate set and the same integer upper budget $K_G$.

---

## 14. PU / FO evaluation

Evaluation semantics are inherited unchanged from the FC protocol.

### 14.1 PU

For current batch $B_t$:

```text
receive B_t
-> adapt using unlabeled B_t
-> obtain post-update model/state
-> evaluate PU on B_t
-> evaluation must not modify persistent state
-> continue to B_{t+1}
```

For controlled Conv variants, all BN state is frozen, so the PU pass is read-only by construction.

PU is report-only and does not participate in hyperparameter selection.

### 14.2 FO

After the target stream:

```text
stop adaptation
-> freeze final model/state
-> switch to evaluation behavior
-> evaluate the full target evaluation set
```

FO is read-only. No adaptation is allowed during FO.

---

## 15. Formal seed and Random mask seeds

Formal experiment seed:

```text
2026
```

There are no formal 2020/2021/2022 seeds in this protocol.

Random group selection uses three deterministic child mask identities:

```text
random_mask_index = 0
random_mask_index = 1
random_mask_index = 2
```

These are not separate formal experiment seeds.

---

## 16. Scientific validity and support utilization

### 16.1 Scientific validity

A formal Conv-LBI run is scientifically valid only if:

```text
run completes
no NaN / Inf / runtime error
selected_group_count <= K_G for every online batch
no budget violation
no Stage-1 max-step failure
all required group-support records are complete
```

Exact-$K_G$ support is not required.

### 16.2 Stage-1 cap

Initial protocol value:

```text
stage1_max_steps = 3000
```

This is a safety cap, not a tuning dimension.

If the Conv pilot shows systematic incompatibility with this cap, amend the protocol before formal tuning; do not silently increase it during search.

### 16.3 Utilization diagnostics

For every batch:

$$
u_t
=
\frac{n_t}{K_G}.
$$

Record:

```text
min group utilization
mean group utilization
p05 group utilization
number / fraction of batches below 95%
Stage-1 mean / max steps
```

Unlike the FC track, a hard 90% utilization gate is **not frozen yet**, because coarse group counts may create unavoidable rollback quantization when $K_G$ is small.

The tuning utilization rule is:

```text
TBD_AFTER_BUDGET_GRID_IS_FIXED
```

Scientific validity remains independent of this tuning diagnostic.

---

## 17. LBI fixed constants and tunable parameters

### 17.1 Initial shared constants

| Setting | Value |
|---|---:|
| support threshold $\tau$ | $1.0\times10^{-4}$ |
| Stage-1 max steps | 3000 |
| Stage-2 steps | 1 |
| Stage-2 optimizer | SGD |
| Stage-2 momentum | 0.9 |
| Stage-2 weight decay | 0.001 |
| Stage-2 Nesterov | true |
| Stage-2 LR schedule | none |
| controlled Conv BN | frozen |
| batch-local LBI restart | every batch |
| strict rollback | enabled |
| masked-delta initialization | enabled |
| off-mask value freezing | enabled |
| Group-Lasso main penalty | unweighted |

### 17.2 Tunable parameters

Only:

```text
alpha
kappa
nu
omega
stage2_lr
```

are formal Conv-LBI tuning dimensions.

The tuning granularity is expected to be:

$$
(\mathrm{dataset},\mathrm{SHOT-OTTA},\mathrm{group\ mode},\rho_G).
$$

For Office-31, one tuned tuple for a given `(group_mode, rho_G)` must be shared across all six transfers.

Forbidden:

```text
per-transfer tuning
per-seed tuning
per-batch tuning
per-class tuning
post-hoc boundary chasing
```

The exact search grid will be defined only after code correctness and budget pilot are complete.

---

## 18. Runtime and GPU-memory protocol

Efficiency measurement inherits the FC online-batch protocol.

Per processed online batch record:

```text
batch_index
batch_size
adapt_runtime_sec
pu_runtime_sec
online_runtime_sec
lbi_stage1_runtime_sec
lbi_stage2_runtime_sec
peak_gpu_memory_allocated_bytes / mb
peak_gpu_memory_reserved_bytes / mb
```

Formal compute timing excludes:

```text
DataLoader wait
CPU preprocessing
host-to-device transfer
checkpoint I/O
JSON / summary serialization
```

For CUDA, synchronize before and after timed compute regions.

Primary run-level efficiency summaries:

```text
online batch runtime mean / std / median / p95
total online compute runtime
Stage-1 runtime share
Stage-2 runtime share
GPU peak allocated max MB
GPU peak reserved max MB
```

The paper must distinguish:

```text
structured update sparsity
GPU memory footprint
wall-clock compute
```

and must not claim wall-clock efficiency unless supported by measurements.

---

## 19. Checkpoint / resume policy

### 19.1 Baselines

Controlled Conv baselines do not save adapted-model or stream-resume checkpoints:

```text
save_model = false
stream_checkpoint = false
partial_resume = false
```

Interrupted baseline experiments are rerun.

### 19.2 Conv-LBI

Only Conv-LBI may support exact partial stream resume at a completed online batch boundary.

Batch-local variables:

```text
Theta_delta
Gamma
Z
```

must not survive across batches because they are reset by design.

Resume is engineering resilience and must not change scientific identity.

Formal efficiency comparisons should use fresh, non-resumed runs.

---

## 20. Provenance and experiment identity

Every formal run must record at least:

```text
conv_protocol_revision
scientific_implementation_revision
source_checkpoint_revision
dataset / transfer
backbone
formal seed
batch size
variant
group mode
candidate layer list
group count
candidate scalar count
requested rho_G / K_G
realized group ratio
realized scalar ratio
SHOT config
complete LBI tuple
support threshold
Stage-1 cap
Stage-2 settings
```

The scientific implementation revision must change when Conv correctness semantics change.

Efficiency-only metadata must remain separable from the scientific experiment identity.

---

## 21. Formal tuning and evaluation order

The Conv study must follow this order:

```text
P0. freeze this protocol except explicit TBD fields
P1. refactor implementation
P2. correctness / contract tests
P3. baseline sanity runs
P4. budget pilot and freeze rho_G grid
P5. predeclare Conv-LBI search grid / ranking rule
P6. run tuning
P7. freeze tuples
P8. independent final formal rerun
P9. no retuning after final formal results
```

PU is report-only throughout.

Validity precedes accuracy during tuning.

---

## 22. Required correctness / contract tests before experiments

The refactored implementation must include automated tests for at least:

1. **Group partition correctness**
   - out-channel groups cover every candidate Conv scalar exactly once;
   - filter-connection groups cover every candidate Conv scalar exactly once.

2. **Global group counts**
   - out-channel: `9216`;
   - filter-connection: `6553600`.

3. **Old-state update**
   - $Z^{k+1}$ uses $(\Theta_\Delta^k,\Gamma^k)$ and not $\Theta_\Delta^{k+1}$.

4. **Group prox**
   - zero norm -> zero group;
   - norm below/equal prox threshold -> zero group;
   - active group -> correct block soft-threshold scaling.

5. **Support threshold**
   - formal support uses $\|\Gamma_g\|_2\ge\tau$ with $\tau=10^{-4}$.

6. **Strict global budget**
   - no per-layer minimum-one behavior;
   - no budget slack;
   - selected count never exceeds $K_G$.

7. **Rollback**
   - overshoot state is discarded;
   - latest feasible state is restored exactly.

8. **Masked-delta initialization**
   - nonzero off-mask $\Theta_\Delta$ must not change Stage-2 initial model.

9. **Stage-2 strict freezing**
   - weight decay / Nesterov / optimizer state must not move off-mask Conv values.

10. **Persistent writeback**
    - only the refined persistent state survives to the next batch;
    - local LBI states restart from zero.

11. **Global baseline exact-K**
    - Random/Magnitude/Saliency operate over the same global group pool;
    - selected group count equals $K_G$ when $K_G>0$.

12. **BN / PU read-only contract**
    - controlled Conv BN state is unchanged by adaptation and PU evaluation.

No tuning should begin until these tests pass.

---

## 23. Explicit TBD fields before formal tuning

The following are intentionally **not frozen yet**:

### TBD-1. Formal Conv group-ratio grid

```text
rho_G values: TBD
```

### TBD-2. Cross-grouping comparison policy

Need to decide whether the main out-channel vs filter-connection comparison is shown by:

```text
same nominal group ratio
matched realized scalar ratio
both views
```

Current recommendation: report both group ratio and realized scalar ratio; do not claim equal adaptation cost from nominal group ratio alone.

### TBD-3. Tuning utilization gate

The FC hard90 gate is not automatically inherited because small group budgets may create coarse rollback quantization.

### TBD-4. Conv hyperparameter search grid

The search grid for:

```text
alpha / kappa / nu / omega / stage2_lr
```

will be predeclared after budget pilot, before tuning.

---

## 24. Paper-level terminology

Use:

```text
FC element-wise sparse adaptation
Conv group-structured sparse adaptation
out-channel grouping
filter-connection grouping
global group budget
realized group ratio
realized scalar update ratio
```

Avoid unsupported wording such as:

```text
physical channel pruning
inference-time acceleration
FLOP reduction
network compression
```

unless a separate physical restructuring experiment is added.

The central methodological relation is:

> FC-LBI and Conv-LBI share the same restart Sparse Delta Learning framework. FC-LBI uses singleton groups and an element-wise $L_1$ proximal operator, whereas Conv-LBI uses structured convolutional groups and a Group-Lasso proximal operator.

---

## 25. Immediate next action

After this draft protocol is accepted, the next task is **P1 Conv code refactor**, not formal experiments.

Implementation should reuse the current refined FC semantics and abstract the sparsity unit as a group partition rather than maintaining a separate legacy Conv-LBI code path.

The first refactor deliverable should contain:

```text
unified/group-aware LBI engine
out-channel group adapter
filter-connection group adapter
global group selector utilities
Conv controlled-family BN freeze
contract tests listed in Section 22
no tuning runs
```
