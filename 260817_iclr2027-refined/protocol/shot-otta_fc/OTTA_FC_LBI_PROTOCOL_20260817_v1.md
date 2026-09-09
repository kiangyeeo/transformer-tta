# OTTA_FC_LBI_PROTOCOL_20260817_v1

**Status:** Frozen protocol specification  
**Scope:** SHOT-OTTA + bottleneck FC-layer controlled adaptation + Sparse Delta Learning / LBI  
**Scientific implementation revision:** `iclr2027_refined_20260817_v1`  
**Efficiency protocol revision:** `otta_fc_batch_efficiency_20260817_v1`  
**Protocol date:** 2026-08-17

> This document is the normative protocol for the current **FC-layer** OTTA/LBI study.
> It intentionally separates benchmark/evaluation invariants, SHOT-OTTA-specific settings,
> and FC/LBI-specific settings so that future OTTA objectives can replace the SHOT-specific
> layer without redefining the benchmark, and future convolution-layer work can define a
> separate protocol without changing this frozen FC protocol.

---

## 1. Scope and freeze boundary

This protocol applies only to the current study:

```text
OTTA method      : SHOT-OTTA
adaptation scope : bottleneck FC layer
sparse method    : LBI / Sparse Delta Learning
datasets         : Office-31, VisDA-C
```

The protocol has three conceptual layers.

### Layer A — Benchmark / evaluation invariants

These settings should remain unchanged when SHOT-OTTA is later replaced by another OTTA method:

- dataset and transfer definition;
- backbone and source checkpoint;
- batch size;
- target stream construction;
- formal run seed;
- transforms;
- PU / FO evaluation semantics;
- metrics;
- FC candidate definition for the FC track;
- sparse budget definition;
- checkpoint policy;
- runtime / GPU-memory measurement protocol;
- result aggregation and provenance.

### Layer B — SHOT-OTTA-specific settings

These settings are specific to the current SHOT-OTTA instantiation:

- adaptation objective;
- pseudo-label rule;
- `cls_par`;
- `ent_par`;
- confidence threshold;
- optimizer;
- learning rate;
- SHOT learning-rate schedule;
- native SHOT trainable scope and BN behavior.

When another OTTA method is introduced, Layer B is replaced by that method's own frozen method-specific protocol.

### Layer C — FC/LBI controlled-comparison settings

These define the current FC-layer sparse adaptation study:

- FC candidate layer;
- global scalar budget;
- Random / Magnitude / Saliency selectors;
- LBI support semantics;
- LBI fixed constants;
- LBI tuning granularity.

Future **convolution-layer** experiments are out of scope for this file. They must use a new proposal/protocol and must not silently modify this frozen FC protocol.

---

## 2. Protocol identifiers and provenance

The formal protocol name is:

```text
OTTA_FC_LBI_PROTOCOL_20260817_v1
```

The frozen scientific implementation revision is:

```text
iclr2027_refined_20260817_v1
```

The efficiency measurement revision is:

```text
otta_fc_batch_efficiency_20260817_v1
```

The source-checkpoint revision must be recorded as:

```text
nips2026_shot_otta_uda_source_v1
```

Recommended metadata field:

```text
source_checkpoint_revision
```

`source_checkpoint_revision` is part of scientific provenance and must participate in the scientific experiment identity/hash.

The efficiency revision is measurement metadata and must **not** change the scientific experiment hash.

The protocol revision itself must be written to plans/manifests/results for traceability. Scientific identity is determined by the actual scientific configuration fields plus implementation/source revisions; the efficiency-only revision remains separate.

---

## 3. Datasets, transfers, backbones, and batches

### 3.1 Office-31

| Setting | Frozen value |
|---|---|
| Dataset | Office-31 |
| Backbone | ResNet-50 |
| Backbone feature dimension | 2048 |
| Bottleneck | `Linear(2048, 256)` |
| Bottleneck normalization | BatchNorm |
| Classifier | weight-normalized classifier |
| Number of classes | 31 |
| OTTA batch size | **64** |
| DataLoader workers | **4** |
| Target passes | **1** |
| `drop_last` | **false** |
| Formal run seed | **2026** |

Formal transfers:

```text
A -> D
A -> W
D -> A
D -> W
W -> A
W -> D
```

### 3.2 VisDA-C

| Setting | Frozen value |
|---|---|
| Dataset | VisDA-C |
| Backbone | ResNet-101 |
| Backbone feature dimension | 2048 |
| Bottleneck | `Linear(2048, 256)` |
| Bottleneck normalization | BatchNorm |
| Classifier | weight-normalized classifier |
| Number of classes | 12 |
| OTTA batch size | **256** |
| DataLoader workers | **4** |
| Target passes | **1** |
| `drop_last` | **false** |
| Formal run seed | **2026** |

Formal transfer:

```text
Synthetic / Train -> Real / Validation
```

### 3.3 Precision

Formal comparisons must use the same numerical precision across variants.

For this protocol, use the repository's current non-AMP path consistently. Do not enable mixed precision for only a subset of methods.

---

## 4. Source checkpoints

All formal variants must start from the same frozen SHOT source checkpoints.

The checkpoint repository is:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/nips2026/SHOT-OTTA/ckpt/source/uda
```

Because the current resolver constructs paths in the form:

```text
root / da / dataset / source
```

the configuration-level root should be the parent:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/nips2026/SHOT-OTTA/ckpt/source
```

with:

```text
da = uda
```

This avoids an invalid duplicated path such as `.../source/uda/uda/...`.

### 4.1 Office checkpoint mapping

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

Thus:

```text
A->D, A->W use source=A
D->A, D->W use source=D
W->A, W->D use source=W
```

### 4.2 VisDA checkpoint mapping

Use the existing SHOT source checkpoint mapping under:

```text
uda/VISDA-C/T/source_F.pt
uda/VISDA-C/T/source_B.pt
uda/VISDA-C/T/source_C.pt
```

### 4.3 Checkpoint provenance

Every formal run must record the resolved paths of:

```text
source_F
source_B
source_C
```

If inexpensive to implement, record SHA256 hashes as additional provenance. The minimum frozen requirement is:

```text
source_checkpoint_revision
+ exact resolved checkpoint paths
```

Changing `source_checkpoint_revision` must change the scientific experiment identity/hash.

---

## 5. Target stream and data transforms

For a fixed:

```text
(dataset, transfer, seed=2026)
```

all variants must observe the same target stream order.

The same formal seed controls:

- target permutation/order;
- augmentation RNG;
- ordinary stochastic computation used by the formal run.

The formal experiment seed is **not** the same concept as Random-baseline mask randomness.

### 5.1 Online adaptation transform

Freeze the current SHOT online transform:

```text
Resize(256)
RandomCrop(224)
RandomHorizontalFlip()
ImageNet normalization
```

### 5.2 Final evaluation transform

Freeze:

```text
Resize(256)
CenterCrop(224)
ImageNet normalization
```

### 5.3 One-pass online evaluation

Each target sample is processed in one online stream pass only.

No replay pass over the target stream is allowed unless a future method explicitly defines memory/replay as part of the method, in which case that behavior belongs to the future method-specific protocol rather than this SHOT protocol.

---

## 6. Formal seed and Random mask seeds

The single formal experimental seed is:

```text
2026
```

There are no formal seeds `2020/2021/2022` in this protocol.

Random selection is evaluated with **three independent masks inside the same formal run condition**.

Define three deterministic mask-seed identities:

```text
random_mask_index = 0
random_mask_index = 1
random_mask_index = 2
```

The implementation may deterministically derive the low-level RNG seed from:

```text
formal_seed=2026
+ random_mask_index
```

but the derived seeds must be:

- deterministic;
- distinct;
- recorded in artifacts;
- unchanged across reruns of this protocol.

The three Random masks do not count as three formal experiment seeds.

---

## 7. SHOT-OTTA objective

All SHOT-based variants in this protocol use the same causal current-batch adaptation objective.

Frozen loss components:

```yaml
components:
  - ent
  - div
  - pseudo

cls_par: 0.3
ent_par: 1.0
threshold: 0.0
```

The effective objective is:

$$
L_{\mathrm{SHOT}}
=
0.3 L_{\mathrm{pseudo}}
+
L_{\mathrm{ent}}
+
L_{\mathrm{div}}.
$$

The current implementation uses `ent_par` for the entropy/diversity terms; there is no separately tuned `div_par` in this protocol.

### 7.1 Causal current-batch pseudo-label

For the current batch:

$$
\hat y_i
=
\arg\max_c p_\theta(c\mid x_i).
$$

The pseudo-label term is based only on the current online batch.

The confidence threshold is frozen to:

$$
\delta = 0.
$$

Therefore all current-batch samples are eligible for the pseudo-label cross-entropy term under the current implementation.

Do not replace this with full-target noncausal prototype clustering inside this protocol.

---

## 8. SHOT optimizer and learning-rate settings

### 8.1 Office-31

```yaml
optimizer: sgd
lr: 0.01
lr_decay1: 0.1
lr_decay2: 1.0
momentum: 0.9
weight_decay: 0.001
nesterov: true
lr_gamma: 10.0
lr_power: 0.75
```

### 8.2 VisDA-C

Use the same settings except:

```yaml
lr: 0.001
```

### 8.3 Native SHOT learning-rate schedule

For native SHOT:

- `netF` starts with the base LR multiplied by `lr_decay1`;
- `netB` starts with the base LR multiplied by `lr_decay2`.

The global stream scheduler is frozen as:

$$
\eta_t
=
\eta_0
\left(
1
+
10\frac{t}{T}
\right)^{-0.75},
$$

where:

$$
T
=
\text{number of online target batches}.
$$

### 8.4 Controlled FC baselines

`module_dense`, `module_random`, `module_magnitude`, and `module_saliency` use the same SHOT objective and the same dataset-level base optimizer/schedule configuration.

Their controlled difference is the FC trainable/update support, not a separately tuned SHOT objective or base LR.

LBI Stage-2 has its own fixed Stage-2 LR parameter as defined later and does not use a local SHOT-style decay schedule.

---

## 9. Native SHOT versus controlled FC family

Two comparison roles must remain distinct.

### 9.1 Native SHOT reference

The native SHOT reference is:

```text
full_dense
```

It preserves the current method-native SHOT update semantics:

- `netF` adapts;
- `netB` adapts;
- `netC` is frozen;
- F/B BatchNorm running statistics follow native adaptation behavior.

This is a native-method performance reference.

It is **not** a matched-support selector baseline.

### 9.2 Controlled FC family

The controlled FC family is:

```text
module_dense
module_random
module_magnitude
module_saliency
module_lbi
```

For every controlled FC variant:

- backbone parameters are frozen;
- classifier parameters are frozen;
- all BN affine parameters are frozen;
- all BN running statistics are frozen;
- BN modules use frozen/evaluation behavior;
- the only persistent trainable/changeable model parameters are the bottleneck FC candidate scalars defined below.

The selector-level controlled comparison is primarily among these controlled FC variants.

---

## 10. FC candidate definition

The candidate parameter set is exactly:

```text
netB.bottleneck.weight
netB.bottleneck.bias
```

Do not include:

- bottleneck BN affine parameters;
- backbone parameters;
- classifier parameters;
- BN running buffers.

The bottleneck weight is:

$$
W \in \mathbb{R}^{256\times 2048}.
$$

The bottleneck bias is:

$$
b \in \mathbb{R}^{256}.
$$

Therefore:

$$
N_{\mathrm{FC}}
=
256\times 2048 + 256
=
524544.
$$

The same FC candidate size is used for Office-31 and VisDA-C.

All FC sparse selectors operate on one deterministic **global scalar pool** containing weight and bias.

---

## 11. Sparse budget

Formal sparse ratios are:

$$
\rho
\in
\{0.0005,\ 0.001,\ 0.002\}.
$$

The strict global integer budget is:

$$
K
=
\left\lfloor
\rho N_{\mathrm{FC}}
\right\rfloor.
$$

With:

$$
N_{\mathrm{FC}} = 524544,
$$

the formal budgets are:

| Ratio | Integer budget |
|---:|---:|
| `0.0005` | `262` |
| `0.001` | `524` |
| `0.002` | `1049` |

Forbidden budget semantics:

- no `ceil`;
- no per-tensor budget;
- no independent weight/bias budget;
- no minimum-one-per-tensor rule;
- no hidden budget slack.

`module_random`, `module_magnitude`, and `module_saliency` select exactly `K` scalars when `K > 0`.

`module_lbi` obeys the same strict upper bound but may end with fewer than `K` selected scalars because of threshold support and strict rollback.

---

## 12. FC baseline selector definitions

### 12.1 `module_dense`

All:

$$
N_{\mathrm{FC}}=524544
$$

candidate scalars are trainable.

It is the dense FC controlled reference and is run once per transfer rather than once per sparse budget.

### 12.2 `module_random`

For each formal sparse budget, use three independent deterministic Random masks.

Each mask:

- starts from the same source checkpoint;
- receives the same target stream;
- uses the same formal seed and augmentation protocol;
- differs only in the Random FC support.

The selected support is a global uniform sample of exactly `K` FC scalars.

### 12.3 `module_magnitude`

At the source checkpoint before target adaptation:

$$
s_j = |\theta_j|.
$$

Select the global top-`K` candidate scalars.

The mask remains fixed for the whole target stream.

### 12.4 `module_saliency`

At each online adaptation step, compute:

$$
s_j
=
|\theta_j g_j|.
$$

Select the global top-`K` candidate scalars.

The mask may change over time.

Frozen sparse-optimizer behavior:

- mask-out parameter values are restored/frozen;
- mask-out SGD momentum is cleared;
- continuously selected coordinates preserve their momentum state.

---

## 13. LBI algorithmic semantics

The LBI scientific semantics are frozen by:

```text
IMPLEMENTATION_REVISION = iclr2027_refined_20260817_v1
```

Do not change them while running this protocol.

### 13.1 Per-batch local restart

At every online batch, reset the batch-local LBI state:

```text
theta_delta
gamma
z
```

The persistent adapted model state is carried across batches; the local Split-LBI variables are not.

### 13.2 Correct old-state update

The frozen implementation uses the corrected old-state Eq. (5) semantics:

$$
\theta_\Delta^{k+1}
=
\theta_\Delta^k
-
\alpha\kappa
\nabla_{\theta_\Delta}
L(\theta_\Delta^k,\gamma^k),
$$

$$
z^{k+1}
=
z^k
-
\alpha
\nabla_\gamma
L(\theta_\Delta^k,\gamma^k),
$$

$$
\gamma^{k+1}
=
\kappa\operatorname{prox}_J(z^{k+1}).
$$

The `z` update must use the old-state coupling based on:

$$
(\theta_\Delta^k,\gamma^k),
$$

not the newly updated delta.

### 13.3 Support definition

Support is defined by the frozen threshold:

$$
\tau
=
10^{-4}.
$$

A scalar is selected if:

$$
M_j
=
\mathbf{1}
\left[
|\gamma_j|
\ge
\tau
\right].
$$

The same support definition is used for:

- Stage-1 support counting;
- strict budget checking;
- final selected mask;
- masked-delta Stage-2 initialization;
- Stage-2 trainable support.

The threshold is a protocol constant, not a dataset/budget tuning variable.

### 13.4 Strict integer budget and rollback

The maximum selected support is:

$$
K
=
\left\lfloor
\rho N_{\mathrm{FC}}
\right\rfloor.
$$

If Stage 1 overshoots the strict budget, rollback to the last valid state.

No top-`K` trimming is applied to LBI gamma support.

Therefore the final LBI selected count may satisfy:

$$
|M| < K.
$$

This is valid behavior.

### 13.5 Masked-delta Stage-2 initialization

Stage 2 initializes from the persistent base model plus only the selected Stage-1 delta:

$$
\theta_{\mathrm{init}}
=
\theta_{\mathrm{base}}
+
M\odot\theta_\Delta.
$$

Do not densely apply off-mask delta values.

### 13.6 Stage-2 mask-out freezing

During Stage 2:

- only selected FC coordinates may change;
- off-mask values are restored/frozen after optimizer steps;
- sparse optimizer mask-out state is reset as implemented;
- the Stage-2 LR is fixed and does not use a local SHOT LR decay schedule.

### 13.7 Persistent accumulation

The persistent update uses the frozen `omega` accumulation semantics of the refined implementation.

Do not redefine this operation inside this protocol.

---

## 14. LBI fixed constants and tunable parameters

### 14.1 Global frozen LBI constants

| Setting | Frozen value |
|---|---:|
| `support_threshold` | `1.0e-4` |
| `stage1_max_steps` | `3000` |
| `stage2_steps` | `1` |
| Stage-2 optimizer | SGD |
| Stage-2 momentum | `0.9` |
| Stage-2 weight decay | `0.001` |
| Stage-2 Nesterov | `true` |
| Stage-2 LR schedule | none; fixed LR |
| FC BN | frozen |
| batch-local LBI restart | every batch |
| strict rollback | enabled |
| masked-delta initialization | enabled |
| mask-out value freezing | enabled |
| diagnostic delta tolerance | `1e-12` |

`stage1_max_steps=3000` is a safety cap, not a performance tuning dimension.

If a candidate LBI configuration repeatedly reaches this cap, treat that configuration as unsuitable rather than silently increasing the cap.

`stage2_steps=1` is also frozen and must not be tuned per budget.

### 14.2 Tunable LBI parameters

Only the following LBI values are tuned for the formal study:

```text
alpha
kappa
nu
omega
stage2_lr
```

The tuning granularity is:

$$
(\mathrm{dataset},\ \mathrm{OTTA\ method},\ \rho).
$$

For the current SHOT protocol, this gives six final configurations:

| Dataset | OTTA method | Budget | Final LBI config |
|---|---|---:|---|
| Office-31 | SHOT-OTTA | `0.0005` | **TBD** |
| Office-31 | SHOT-OTTA | `0.001` | **TBD** |
| Office-31 | SHOT-OTTA | `0.002` | **TBD** |
| VisDA-C | SHOT-OTTA | `0.0005` | **TBD** |
| VisDA-C | SHOT-OTTA | `0.001` | **TBD** |
| VisDA-C | SHOT-OTTA | `0.002` | **TBD** |

For Office-31, one tuned configuration for a given budget must be shared across all six transfers.

Forbidden tuning granularity:

- per transfer;
- per seed;
- per target batch.

A formal LBI run must not silently fall back to an old/default LBI configuration if the required frozen tuned tuple is unresolved.

---

## 15. Target-label policy

Target labels must never be used by online adaptation.

They may be used only for:

- PU metric computation;
- FO metric computation;
- final offline reporting/analysis.

Any future hyperparameter-selection procedure that uses labeled validation information must be documented separately before formal tuning and applied consistently. It must not become transfer-specific hidden tuning inside this protocol.

---

## 16. PU evaluation

PU is defined as **post-update evaluation on the same current batch**.

For batch:

$$
B_t,
$$

the sequence is:

```text
receive B_t
-> adapt using unlabeled B_t
-> obtain post-update model/state
-> evaluate PU on B_t
-> ensure evaluation writes no persistent state
-> continue to B_{t+1}
```

The frozen principle is:

> **Evaluation may read persistent state but must not change persistent state.**

### 16.1 Native SHOT PU

Native SHOT adaptation is allowed to update its native BN state during adaptation.

The extra PU measurement forward must not introduce an additional BN update.

Therefore the existing snapshot/restore behavior for BN buffers around PU measurement must remain enabled.

### 16.2 Controlled FC PU

For controlled FC variants, BN is frozen throughout, so PU uses the already-frozen BN state.

### 16.3 Future method states

If a future OTTA method introduces mutable state such as:

- EMA teacher;
- prototype memory;
- confidence statistics;
- queues;
- memory banks;

the same read-only PU principle applies. That future method-specific implementation must prevent the evaluation-only PU pass from adding an extra state update.

---

## 17. FO evaluation

After the target stream is fully processed:

```text
stop adaptation
-> freeze final method/model state
-> switch evaluation modules to inference/eval behavior
-> evaluate the full target evaluation set
```

FO is read-only.

No additional test-time adaptation is allowed during FO.

---

## 18. Evaluation metrics

### 18.1 Office-31

The primary PU and FO metric is sample-level overall accuracy:

$$
\mathrm{Acc}
=
\frac{
\#\mathrm{correct\ samples}
}{
\#\mathrm{samples}
}.
$$

Per-class metrics may be stored as diagnostics but are not the primary Office score.

The dataset-level Office summary gives equal weight to the six transfers:

$$
\mathrm{OfficeAvg}
=
\frac{1}{6}
\sum_{d=1}^{6}
\mathrm{Acc}_d.
$$

### 18.2 VisDA-C

The primary PU and FO metric is fixed-12-class mean per-class accuracy:

$$
\mathrm{mAcc}
=
\frac{1}{12}
\sum_{c=1}^{12}
\mathrm{Acc}_c.
$$

Also retain as diagnostics:

- overall accuracy;
- all 12 class accuracies;
- worst-class accuracy;
- class-accuracy standard deviation.

---

## 19. Checkpoint and resume policy

### 19.1 Baselines

The following variants do **not** save adapted-model or stream-resume checkpoints:

```text
source_only
full_dense
module_dense
module_random
module_magnitude
module_saliency
```

Formal policy:

```text
save_model = false
stream_checkpoint = false
partial_resume = false
```

If interrupted, rerun the baseline experiment.

### 19.2 LBI

Only:

```text
module_lbi
```

supports partial stream resume.

Resume granularity is the **completed online batch boundary**.

After a completed batch, the implementation may save the persistent state required to resume from the exact next online batch.

Batch-local LBI variables:

```text
theta_delta
gamma
z
```

do not need to survive across batches because they are reset by design.

The final adapted target model is not saved as a formal model checkpoint:

```text
save_model = false
```

The LBI stream checkpoint exists only for engineering resilience and exact resume.

The policy applies to LBI on both Office-31 and VisDA-C.

---

## 20. Per-batch runtime and GPU-memory protocol

Efficiency is measured at the **online batch** level.

The formal efficiency measurement revision is:

```text
otta_fc_batch_efficiency_20260817_v1
```

### 20.1 Batch runtime fields

For every processed online batch, record:

```text
batch_index
batch_size

adapt_runtime_sec
pu_runtime_sec
online_runtime_sec
```

with:

$$
T_t^{\mathrm{online}}
=
T_t^{\mathrm{adapt}}
+
T_t^{\mathrm{PU}}.
$$

### 20.2 Runtime boundary

Before timing starts:

- the DataLoader has yielded the batch;
- input/label transfer to the target device has completed.

The formal batch compute timer excludes:

- DataLoader wait;
- CPU preprocessing;
- host-to-device transfer;
- checkpoint load/save;
- JSON/JSONL serialization;
- summary/finalize I/O.

For CUDA execution, synchronize before and after timed compute regions so asynchronous kernels do not invalidate timing.

### 20.3 LBI sub-timers

For LBI, additionally record:

```text
lbi_stage1_runtime_sec
lbi_stage2_runtime_sec
```

These are diagnostics inside the broader adaptation runtime.

The full adaptation timer may contain small additional overhead such as mask construction or persistent-update bookkeeping, so Stage-1 + Stage-2 need not be exactly identical to `adapt_runtime_sec`.

### 20.4 Per-batch GPU memory

After device transfer and immediately before the online compute measurement region, reset CUDA peak-memory statistics for the current batch.

After adaptation + PU, record:

```text
peak_gpu_memory_allocated_bytes
peak_gpu_memory_reserved_bytes
peak_gpu_memory_allocated_mb
peak_gpu_memory_reserved_mb
```

The memory interval therefore corresponds to the same online compute region:

$$
\mathrm{adaptation} + \mathrm{PU}.
$$

Do not use high-frequency NVML / `nvidia-smi` utilization polling in protocol v1.

---

## 21. Run-level efficiency aggregation

All formal run-level efficiency statistics must be derived from the per-batch raw records.

### 21.1 Runtime

For online batch runtimes:

$$
T_1,\ldots,T_T,
$$

record:

```text
online_batch_runtime_mean_sec
online_batch_runtime_std_sec
online_batch_runtime_median_sec
online_batch_runtime_p95_sec
online_compute_runtime_sec
```

where:

$$
\mathrm{online\_compute\_runtime\_sec}
=
\sum_{t=1}^{T} T_t.
$$

Also retain:

```text
adapt_batch_runtime_mean_sec
adapt_batch_runtime_std_sec
adapt_runtime_total_sec

pu_batch_runtime_mean_sec
pu_batch_runtime_std_sec
pu_runtime_total_sec
```

The primary efficiency views are:

- mean online batch runtime;
- total online compute runtime.

### 21.2 GPU memory

From per-batch peaks, record:

```text
gpu_peak_allocated_mean_mb
gpu_peak_allocated_max_mb
gpu_peak_reserved_mean_mb
gpu_peak_reserved_max_mb
```

The primary GPU-memory metric is:

```text
gpu_peak_allocated_max_mb
```

because it represents the maximum observed allocation requirement over the stream.

Reserved memory is retained as an allocator diagnostic.

### 21.3 FO and operational runtime

Keep separate:

```text
fo_eval_runtime_sec
wall_runtime_sec
```

`fo_eval_runtime_sec` measures final FO evaluation only.

`wall_runtime_sec` is operational wall-clock time and may include setup, DataLoader delay, checkpoint I/O, artifact writes, FO, and finalization.

Do not use wall time as the primary algorithm-compute metric.

---

## 22. Random efficiency aggregation

Each of the three Random masks retains its own complete per-batch efficiency history and child-level aggregate.

For mask:

$$
m,
$$

let:

$$
\bar T_m
$$

denote its mean online batch runtime.

The formal Random per-batch runtime is:

$$
\bar T_{\mathrm{Random}}
=
\frac{1}{3}
\sum_{m=1}^{3}
\bar T_m.
$$

Also report the standard deviation across mask-level means.

For full-stream online compute:

```text
online_compute_runtime_sec
```

at the Random top level represents the **mean single-mask stream compute cost**, not the sum of three masks.

Separately retain:

```text
random_total_online_compute_runtime_sec
```

for the actual operational cost of executing all three masks.

FO runtime follows the same distinction:

```text
fo_eval_runtime_sec
    = mean(child fo_eval_runtime_sec)

fo_eval_runtime_mask_std_sec
    = std(child fo_eval_runtime_sec)

random_total_fo_eval_runtime_sec
    = sum(child fo_eval_runtime_sec)
```

Random top-level GPU peak is the maximum over all child masks and all their batches.

---

## 23. LBI resume and efficiency history

LBI resume must restore the already completed per-batch efficiency history.

For a resumed stream:

```text
completed batch records
-> restore
-> resume from exact next batch
-> append new batch records
```

Requirements:

- no duplicated batch index;
- no missing completed batch;
- final aggregate uses the complete batch history;
- checkpoint I/O remains outside batch compute timers.

Record:

```text
runtime_resume_used
runtime_segment_count
```

Per-batch compute measurements remain usable after exact resume because the timed regions exclude checkpoint I/O and the complete raw history is restored.

`wall_runtime_sec` remains an operational diagnostic and should not be interpreted as a continuous uninterrupted-run compute measure after resume.

---

## 24. Formal efficiency execution conditions

Formal runtime/GPU comparisons must use the same hardware and execution environment.

For this project, formal efficiency comparisons should use the available NVIDIA RTX 3090 GPUs under:

```text
1 experiment process / GPU
```

The following must match across compared variants:

- GPU model;
- precision;
- PyTorch version;
- CUDA version;
- batch size;
- DataLoader worker count;
- input transform;
- target stream.

Record at least:

```text
gpu_name
gpu_device_index
gpu_total_memory_bytes
torch_version
cuda_version
runtime_comparable
```

`runtime_comparable=true` is allowed only when the formal execution condition is explicitly requested by the launcher/config.

If multiple experiment processes share one GPU, accuracy results remain valid but:

```text
runtime_comparable = false
```

for formal efficiency reporting.

---

## 25. Formal experiment matrix

The formal variants are:

```text
source_only
full_dense
module_dense
module_random
module_magnitude
module_saliency
module_lbi
```

### 25.1 Non-budgeted variants

Run once per transfer:

```text
source_only
full_dense
module_dense
```

Do not duplicate these across sparse budgets.

### 25.2 Budgeted sparse variants

For each:

$$
\rho\in\{0.0005,\ 0.001,\ 0.002\},
$$

run:

```text
module_random
module_magnitude
module_saliency
module_lbi
```

`module_random` internally evaluates three masks.

### 25.3 Office-31

For each of six transfers:

```text
3 non-budgeted identities
+ 4 sparse variants x 3 budgets
= 15 formal experiment identities
```

Total Office formal identities:

$$
6\times 15 = 90.
$$

### 25.4 VisDA-C

For the single formal transfer:

```text
15 formal experiment identities
```

### 25.5 Combined formal matrix

The complete SHOT-FC protocol contains:

$$
90 + 15 = 105
$$

top-level formal experiment identities.

The three Random masks are child executions of each Random formal identity, not additional formal seeds.

Formal LBI launch is blocked until the corresponding six dataset/budget SHOT-LBI tuned configurations are resolved.

---

## 26. Scientific identity requirements

Scientific identity must distinguish any scientifically different run.

At minimum, scientific config/hash must reflect the relevant values among:

- `implementation_revision`;
- `source_checkpoint_revision`;
- dataset;
- transfer;
- source domain;
- target domain;
- architecture/backbone;
- formal seed;
- batch size;
- variant;
- SHOT loss configuration;
- SHOT optimizer configuration;
- sparse budget ratio and integer `K` where applicable;
- Random mask protocol where applicable;
- full LBI tuned configuration where applicable.

Changing:

```text
source_checkpoint_revision
```

must change scientific identity.

Changing only:

```text
EFFICIENCY_PROTOCOL_REVISION
```

must not change scientific identity.

Resume/checkpoint engineering options must not create a new scientific experiment if the resumed execution is scientifically identical.

---

## 27. Artifact and metadata requirements

Formal provenance should be available through the existing artifact pipeline.

Relevant metadata should be propagated where applicable to:

```text
plan entry / plan.json
effective config / config.yaml
manifest.json
metrics.jsonl
summary.json
results.json
stream checkpoint metadata
launcher/status CSV and JSON
aggregate/finalize outputs
```

The per-batch raw efficiency records belong in `metrics.jsonl` or an equivalent structured batch-level artifact.

Do not put all raw batch records into status CSV files; status/aggregate files should contain run-level summaries.

---

## 28. Formal configuration cleanup

After this protocol is implemented, old SHOT-OTTA configs that encode superseded formal settings must not remain as runnable fallbacks.

In particular, remove or replace configurations that encode:

- formal seed 2020/2021/2022;
- old batch/protocol settings;
- old `save_model=true` formal policy;
- old checkpoint/resume policy;
- obsolete source-checkpoint protocol.

The codebase must expose one clear formal entry point for this protocol.

Historical code outside:

```text
260817_iclr2027-refined/
```

remains read-only.

---

## 29. What is frozen and what is still TBD

### Frozen now

- datasets and transfers;
- R50/R101 backbones;
- FC bottleneck architecture;
- Office BS64 / VisDA BS256;
- workers=4;
- seed=2026;
- one-pass stream;
- transforms;
- SHOT loss composition;
- `cls_par=0.3`;
- `ent_par=1.0`;
- threshold=0;
- dataset-specific SHOT base LR;
- SHOT optimizer/scheduler;
- source checkpoint family;
- source checkpoint revision;
- native-vs-controlled BN semantics;
- FC candidate;
- global sparse budgets;
- Random/Magnitude/Saliency definitions;
- LBI support threshold;
- LBI Stage-1 cap;
- LBI Stage-2 step count;
- LBI refined algorithm semantics;
- PU/FO semantics and metrics;
- baseline checkpoint policy;
- LBI batch-boundary resume;
- per-batch runtime/GPU-memory measurement;
- efficiency aggregation.

### Still TBD

Exactly six SHOT-LBI tuned parameter tuples:

```text
(alpha, kappa, nu, omega, stage2_lr)
```

for:

```text
Office x rho=0.0005
Office x rho=0.001
Office x rho=0.002

VisDA x rho=0.0005
VisDA x rho=0.001
VisDA x rho=0.002
```

No other protocol variable should be reopened during this tuning stage unless a concrete implementation bug is found.

---

## 30. Future extension contract

This document must remain a frozen record of the **SHOT-OTTA + FC-layer** study.

### 30.1 Replacing SHOT-OTTA with another OTTA method

Future OTTA methods should reuse Layer A benchmark/evaluation invariants and the FC controlled-comparison framework where meaningful.

They must define a new method-specific section/protocol for:

- objective;
- native trainable scope;
- native state/BN semantics;
- optimizer/LR;
- pseudo-label/confidence/memory behavior.

Do not silently overwrite the SHOT-specific settings in this file.

### 30.2 Convolution-layer experiments

Future convolution-layer Sparse Delta Learning experiments require a separate proposal/protocol defining:

- candidate convolution layers;
- scalar/group budget semantics;
- layer aggregation;
- BN relationship;
- dense/sparse baselines;
- LBI support definition for convolutional parameters.

Do not extend the current FC candidate definition to convolution layers inside this v1 file.

### 30.3 Protocol versioning

If a scientifically meaningful frozen rule must later change, create a new protocol revision rather than silently editing the meaning of this v1.

---

## 31. One-page frozen summary

| Category | Frozen rule |
|---|---|
| Protocol | `OTTA_FC_LBI_PROTOCOL_20260817_v1` |
| Implementation | `iclr2027_refined_20260817_v1` |
| Efficiency | `otta_fc_batch_efficiency_20260817_v1` |
| Source revision | `nips2026_shot_otta_uda_source_v1` |
| Formal seed | `2026` |
| Random masks | 3 deterministic independent masks |
| Office backbone | ResNet-50 |
| VisDA backbone | ResNet-101 |
| Bottleneck | Linear 2048→256 + BN |
| Office batch size | 64 |
| VisDA batch size | 256 |
| Workers | 4 |
| Target pass | 1 |
| SHOT `cls_par` | 0.3 |
| SHOT `ent_par` | 1.0 |
| SHOT threshold | 0.0 |
| Office SHOT LR | 0.01 |
| VisDA SHOT LR | 0.001 |
| SGD momentum | 0.9 |
| Weight decay | 0.001 |
| Nesterov | true |
| LR gamma/power | 10 / 0.75 |
| FC candidate | bottleneck weight + bias |
| Candidate scalars | 524544 |
| Sparse ratios | 0.0005 / 0.001 / 0.002 |
| Integer budgets | 262 / 524 / 1049 |
| Controlled FC BN | frozen |
| Native SHOT BN | method-native adaptive behavior |
| Random | global exact-K, 3 masks |
| Magnitude | source static global top-K |
| Saliency | dynamic global top-K of `|theta * grad|` |
| LBI tau | `1e-4` |
| LBI Stage-1 cap | 3000 |
| LBI Stage-2 steps | 1 |
| LBI tuned vars | alpha/kappa/nu/omega/stage2_lr |
| LBI tuning granularity | dataset × method × budget |
| PU | same-batch post-update, read-only |
| FO | final frozen read-only evaluation |
| Office metric | overall accuracy |
| VisDA metric | fixed-12-class mean per-class accuracy |
| Baseline checkpoint | none |
| LBI resume | completed-batch boundary |
| Efficiency unit | online batch |
| Runtime | adapt + PU |
| GPU memory | per-batch peak allocated/reserved |
| Primary GPU metric | max batch peak allocated memory |
| Formal runtime execution | 1 experiment / RTX 3090 |
| Old formal configs | remove after replacement |
| FC protocol status | frozen except six LBI tuned tuples |

---

**End of protocol.**
