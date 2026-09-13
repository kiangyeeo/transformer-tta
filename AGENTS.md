# AGENTS.md — Transformer SHOT-OTTA + Group Split-LBI Working Protocol

> **Status:** current working specification for the Transformer branch  
> **Last updated:** 2026-08-24
> **Scope:** DeiT-Small + SHOT-OTTA + structural-group sparse adaptation on Office-31 and VisDA-C  
> **Purpose:** repository-level instructions for Codex/agents and developers.  
> This file supersedes older Transformer notes when they conflict with the rules below.  
> The latest FC protocol under `protocol/shot-otta_fc/` remains the reference for **protocol semantics**, but FC numerical hyperparameters must not be copied blindly to Transformer.

---

## 1. Project goal

The Transformer track studies whether **Group Split-LBI** can discover a small, target-specific set of structurally meaningful Transformer coordinates that should remain plastic during online test-time adaptation.

The base representation is

\[
W^{\mathrm{TTA}} = W_{\mathrm{base}} + \Delta W,
\]

where:

- `W_base` is the persistent model entering the current online batch;
- `DeltaW` is the batch-local functional delta used by Split-LBI;
- `Gamma` is the sparse structural proxy;
- `Z` is the Split-LBI dual variable.

The Transformer extension must preserve the same scientific comparison logic as the refined FC study:

1. one frozen source model per source domain;
2. one frozen target-stream protocol;
3. one frozen SHOT objective;
4. one controlled Transformer candidate space;
5. Random / Magnitude / Saliency / Group Split-LBI compared at matched structural budget;
6. all evaluation metrics and PU/FO semantics fixed before formal runs.

The Transformer-specific novelty is the **structural group definition**, not a new TTA loss or a new optimizer family.

---

## 2. Sources of truth and precedence

Use the following precedence.

1. **Explicit user instruction in the current task**
2. **This `AGENTS.md` for the current Transformer track**
3. Latest refined FC protocol:
   `protocol/shot-otta_fc/OTTA_FC_LBI_PROTOCOL_20260817_v1.md`
4. Current implementation and tests

If old code, old proposal text, and this file disagree, do **not** silently preserve the old behavior. Mark the old behavior as legacy/superseded and implement the current rule explicitly.

Important refined-FC ideas that are inherited as **semantics**:

- post-update same-batch PU;
- read-only PU/FO evaluation;
- sample-level Office accuracy;
- fixed-class VisDA macro accuracy;
- strict integer budget;
- no hidden budget slack;
- Random uses three deterministic child masks;
- corrected old-state Split-LBI update;
- strict rollback;
- masked-delta Stage-2 initialization;
- off-mask value/state freezing;
- fixed Stage-2 LR;
- batch-local `Delta/Gamma/Z` restart;
- persistent `omega` accumulation.

Do **not** inherit FC-specific numerical optimizer values just because they appear in the FC protocol.

---

## 3. Current Transformer backbone

Primary backbone:

```text
deit_small_patch16_224.fb_in1k
```

Use the **non-distilled DeiT-Small/16-224** variant.

Model facts:

```text
number of blocks = 12
hidden dimension d = 384
MLP hidden dimension = 1536
```

The Transformer TTA source model is **not** the ImageNet checkpoint. The ImageNet checkpoint is only the initialization used by source-domain training.

Formal OTTA experiments must load the corresponding source-trained `.pth` checkpoint.

---

## 4. Source checkpoints

Current user-confirmed source checkpoints:

```text
Office-31 Amazon:
  /home/nas3/biod/wangkangyi/checkpoints/source_models/office31/amazon.pth

Office-31 DSLR:
  /home/nas3/biod/wangkangyi/checkpoints/source_models/office31/dslr.pth

Office-31 Webcam:
  /home/nas3/biod/wangkangyi/checkpoints/source_models/office31/webcam.pth

VisDA-C synthetic train:
  /home/nas3/biod/wangkangyi/checkpoints/source_models/visda-c/train.pth
```

Transfer mapping:

```text
A -> D, A -> W : amazon.pth
D -> A, D -> W : dslr.pth
W -> A, W -> D : webcam.pth
VisDA train -> validation : train.pth
```

Never use `*.last.pth` as `W0` unless an explicit recovery check proves that it is identical to the selected best checkpoint.

Formal runs should record:

```text
checkpoint absolute path
checkpoint SHA-256
best source epoch
source validation metric
source training seed
source training config
Git commit / dirty state
Python / CUDA / PyTorch / torchvision / timm versions
```

---

## 5. Server layout

All server-side assets must remain under:

```text
/home/nas3/biod/wangkangyi/
```

Confirmed important paths:

```text
repository:
  /home/nas3/biod/wangkangyi/transformer-tta/

datasets:
  /home/nas3/biod/wangkangyi/datasets/

image lists:
  /home/nas3/biod/wangkangyi/datasets/image_lists/

DeiT ImageNet pretrained files:
  /home/nas3/biod/wangkangyi/checkpoints/deit_small_patch16_224.fb_in1k/

source checkpoints:
  /home/nas3/biod/wangkangyi/checkpoints/source_models/

conda environment:
  /home/nas3/biod/wangkangyi/envs/lbi/

HF cache:
  /home/nas3/biod/wangkangyi/hf-cache/

pip cache:
  /home/nas3/biod/wangkangyi/pip-cache/

conda package cache:
  /home/nas3/biod/wangkangyi/conda-pkgs/

conda home:
  /home/nas3/biod/wangkangyi/conda-home/

temporary files:
  /home/nas3/biod/wangkangyi/tmp/

TTDA source-only results:
  /home/nas3/biod/wangkangyi/results/transformer_ttda_source_only/

OTTA source-only results:
  /home/nas3/biod/wangkangyi/results/transformer_otta_source_only/

current dense OTTA result root:
  /home/nas3/biod/wangkangyi/results/transformer_otta_full_dense/

current magnitude result root:
  /home/nas3/biod/wangkangyi/results/transformer_otta_group_magnitude/

current saliency result root:
  /home/nas3/biod/wangkangyi/results/transformer_otta_group_saliency/
```

Do not write large caches, checkpoints, datasets, or temporary artifacts to `$HOME`, `/tmp`, or the system disk.

Recommended server environment variables:

```bash
export HF_HOME=/home/nas3/biod/wangkangyi/hf-cache
export TORCH_HOME=/home/nas3/biod/wangkangyi/hf-cache/torch
export PIP_CACHE_DIR=/home/nas3/biod/wangkangyi/pip-cache
export CONDA_PKGS_DIRS=/home/nas3/biod/wangkangyi/conda-pkgs
export TMPDIR=/home/nas3/biod/wangkangyi/tmp
```

Do not hard-code an HF mirror endpoint unless it is explicitly confirmed in the server environment. The cache path above is confirmed; the mirror URL is not part of the current frozen scientific protocol.

TTA code must construct DeiT without implicitly downloading weights, then load the local source checkpoint.

---

## 6. Dataset / stream protocol

Datasets:

```text
Office-31:
  A -> D
  A -> W
  D -> A
  D -> W
  W -> A
  W -> D

VisDA-C:
  train -> validation
```

All methods under one formal `(dataset, transfer, seed)` condition must observe the same target-stream ordering and augmentation RNG.

One-pass OTTA only:

```text
receive each target sample once in the online stream
no replay
no full-target noncausal pseudo-label clustering
```

Keep `drop_last = false`.

The tail batch must be retained. For Office-31 transfers whose target is Amazon,
the 2,817-sample stream's singleton tail is merged into the preceding online
batch, producing a final online batch of 65 samples. Other Office transfers and
VisDA-C retain ordinary `drop_last=false` batching.

Target labels must never enter adaptation.

They may be used only for:

```text
PU metric computation
FO metric computation
offline reporting / analysis
```

---

## 7. SHOT objective

All Transformer SHOT-based baselines and Group Split-LBI use the same current-batch causal SHOT objective:

\[
L_{\mathrm{SHOT}}
=
0.3L_{\mathrm{pseudo}}
+
L_{\mathrm{ent}}
+
L_{\mathrm{div}}.
\]

Current-batch pseudo-label:

\[
\hat y_i=\arg\max_c p_\theta(c\mid x_i).
\]

Use:

```text
cls_par = 0.3
ent_par = 1.0
pseudo-label confidence threshold = 0.0
```

Do not replace this with full-target prototype clustering in the current Transformer protocol.

---

## 8. PU and FO semantics

### 8.1 PU

PU means **post-update accuracy on the same current batch**.

For online batch `B_t`:

```text
receive B_t
-> forward/backward using unlabeled B_t
-> optimizer/adaptation update
-> obtain post-update persistent state
-> perform a separate read-only PU forward on B_t
-> continue to B_{t+1}
```

Do **not** compute PU from the pre-update logits used to construct the loss.

PU evaluation must not change any persistent model or method state.

For DeiT this is simpler than ResNet SHOT because the controlled Transformer path uses LayerNorm rather than BatchNorm running statistics, but the general read-only rule still applies.

If future methods add mutable states such as:

```text
EMA teacher
prototype memory
queue
confidence statistics
memory bank
```

the extra PU forward must not update those states.

### 8.2 FO

After the online stream ends:

```text
stop adaptation
freeze final model/method state
switch to inference/eval behavior
run a separate complete target evaluation pass
```

FO is fully read-only.

No adaptation is allowed during the FO pass.

---

## 9. Accuracy aggregation

This section is mandatory. Do not use ambiguous `mean(batch_acc)` logic.

### 9.1 Office-31

For each transfer, primary PU/FO accuracy is **sample-level overall accuracy**:

\[
\mathrm{Acc}
=
\frac{\#\mathrm{correct\ samples}}
{\#\mathrm{samples}}.
\]

Implementation:

```python
correct += (pred == label).sum().item()
total += label.numel()
acc = correct / total
```

Forbidden as the formal metric:

```python
np.mean(batch_accuracies)
```

because the final batch may be smaller.

The six Office transfers are then given equal weight:

\[
\mathrm{OfficeAvg}
=
\frac{1}{6}
\sum_{d=1}^{6}\mathrm{Acc}_d.
\]

Do not pool all samples from all six transfers into one global accuracy.

### 9.2 VisDA-C

Primary PU/FO metric is fixed-12-class mean per-class accuracy:

\[
\mathrm{mAcc}
=
\frac{1}{12}
\sum_{c=1}^{12}
\mathrm{Acc}_c.
\]

Also retain:

```text
overall sample accuracy
12 individual class accuracies
worst-class accuracy
class-accuracy standard deviation
```

Formal VisDA comparison must use the macro 12-class score, not plain overall sample accuracy.

---

## 10. Transformer candidate space

The controlled Transformer candidate space is fixed to the **last 3 blocks**:

```text
blocks 9, 10, 11
```

Only the following weight tensors are eligible:

```text
attn.qkv.weight
attn.proj.weight
mlp.fc1.weight
mlp.fc2.weight
```

Frozen for the controlled candidate family:

```text
all bias parameters
all LayerNorm parameters
class token
position embedding
patch embedding
final norm
classifier/head
all parameters in blocks 0..8
```

Candidate scalar count:

```text
5,308,416
```

The candidate space must be identical for:

```text
candidate_dense
group_random
group_magnitude
group_saliency
group_lbi
```

---

## 11. Structural groups

For `d = 384`, define paired architecture-aware groups.

### 11.1 Q-K group

For block `l`, coordinate `p`:

\[
G_{l,p}^{QK}
=
\{
W_{Q,l}[p,:],
W_{K,l}[p,:]
\}.
\]

### 11.2 V-O group

\[
G_{l,p}^{VO}
=
\{
W_{V,l}[p,:],
W_{O,l}[:,p]
\}.
\]

### 11.3 FFN group

\[
G_{l,p}^{FFN}
=
\{
W_{1,l}[p,:],
W_{2,l}[:,p]
\}.
\]

Per block:

```text
QK groups  = 384
VO groups  = 384
FFN groups = 1536
total      = 2304
```

Across three candidate blocks:

```text
total structural groups = 6912
```

Each paired group contains:

```text
768 scalar weights
```

Therefore, for this exact DeiT-S grouping:

\[
\rho_{\mathrm{struct}}
=
\rho_{\mathrm{local\ scalar}}
\]

up to integer floor effects.

The timm `qkv.weight` remains physically fused with shape `[3d, d]`. Q/K/V are logical slices for group construction; do not rewrite the model into three separate Linear modules.

---

## 12. Current formal sparse budgets

Following the completed Transformer pilot and the explicit 2026-08-24 protocol
revision, the current Transformer formal budgets are:

\[
\rho_{\mathrm{struct}}
\in
\{0.0005,\ 0.001,\ 0.002\}.
\]

Use the strict integer rule:

\[
K
=
\left\lfloor
\rho_{\mathrm{struct}} \times 6912
\right\rfloor.
\]

Thus:

| rho | K groups | active candidate scalars |
|---:|---:|---:|
| 0.0005 | 3 | 2,304 |
| 0.0010 | 6 | 4,608 |
| 0.0020 | 13 | 9,984 |

No `ceil`.

No hidden slack.

No per-block quota.

No separate QK / VO / FFN quota.

No minimum-one-group-per-block rule.

The group pool is global across all 6912 candidate groups.

If a later pilot demonstrates that this range is unusable, update the protocol **before** formal runs; do not post-hoc cherry-pick budgets after seeing the full result matrix.

---

## 13. Dense baselines

### 13.1 Source-only

No parameter updates.

The source-only model must remain byte/parameter identical before and after the online stream.

PU and FO are still separate inference passes, and their predictions should agree under deterministic evaluation conditions.

### 13.2 Full-dense

Role: **native/unrestricted SHOT reference**, not a matched-support sparse selector.

Update:

```text
all DeiT parameters except classifier/head
```

The head remains frozen.

Full-dense is allowed to update parameters outside the controlled candidate space.

### 13.3 Candidate-dense

Role: dense controlled reference for structural selectors.

Update exactly all candidate weights:

```text
blocks 9,10,11:
  attn.qkv.weight
  attn.proj.weight
  mlp.fc1.weight
  mlp.fc2.weight
```

Everything else is frozen.

Candidate-dense is run once per transfer, not once per sparse budget.

---

## 14. Shared optimizer settings for controlled Transformer baselines

The following methods must share the same SHOT optimizer configuration:

```text
candidate_dense
group_random
group_magnitude
group_saliency
```

Current frozen Transformer optimizer:

```yaml
optimizer: AdamW
lr: 1.0e-5
betas: [0.9, 0.999]
eps: 1.0e-8
weight_decay: 0.01
steps_per_online_batch: 1
model_mode: eval
```

Do not tune a separate base LR for Random, Magnitude, or Saliency.

Their scientific difference must be the update support, not the optimizer.

Full-dense should use the same Transformer SHOT optimizer family unless a separately documented native-method reason requires otherwise.

FC uses SGD because that is the frozen FC/SHOT implementation. Do **not** copy FC SGD numerical values into the Transformer controlled baselines merely for superficial consistency.

---

## 15. Random baseline

Random selects structural groups, not scalar weights.

For each formal budget use **three deterministic independent masks** inside the same formal experiment condition:

```text
random_mask_index = 0
random_mask_index = 1
random_mask_index = 2
```

The three child executions share:

```text
source checkpoint
formal seed
target stream
augmentation protocol
SHOT loss
optimizer
budget
candidate space
```

They differ only in the selected Random structural support.

Each child mask selects exactly `K` groups uniformly from the global 6912-group pool.

The three masks are **not** three formal experiment seeds.

Record deterministic low-level RNG seeds in artifacts.

Formal Random accuracy:

\[
\mathrm{Acc}_{\mathrm{Random}}
=
\frac{1}{3}
\sum_{m=0}^{2}
\mathrm{Acc}_m.
\]

Also record mask standard deviation as a diagnostic.

Do not require masks at different budgets to be nested unless a later protocol explicitly freezes that rule.

---

## 16. Magnitude baseline

Old implementation behavior based on `sum(|W|)` is superseded for the new formal Transformer rerun.

At `W0`, before target adaptation, compute one score per structural group:

\[
s_g^{\mathrm{mag}}
=
\|W_g\|_2.
\]

Select the global top-`K` groups.

The Magnitude mask is static for the complete online target stream.

Do not recompute Magnitude after adaptation begins.

Because all current paired groups contain the same 768 scalar weights, no additional group-size normalization is needed for the current DeiT-S protocol.

---

## 17. Saliency baseline

Old pure `gradient_l2_norm` behavior is superseded for the new formal Transformer rerun.

Use first-order weight-gradient saliency:

\[
S
=
|W \odot \nabla_W L_{\mathrm{SHOT}}|.
\]

For structural group `g`, aggregate using L2 norm:

\[
s_g^{\mathrm{sal}}
=
\|W_g \odot \nabla_{W_g}L_{\mathrm{SHOT}}\|_2.
\]

At every online adaptation step:

```text
compute SHOT loss
backward
compute all group saliency scores
select global top-K groups
apply/update only selected groups
```

The mask is dynamic and may change from batch to batch.

Previously selected coordinates may retain their historical accumulated model update; the current-step mask controls only the current update.

---

## 18. Strict sparse optimizer semantics

For every sparse method:

```text
group_random
group_magnitude
group_saliency
group_lbi Stage 2
```

gradient masking alone is insufficient when using AdamW.

After a sparse optimizer step, ensure off-mask coordinates do not change due to:

```text
decoupled weight decay
Adam first moment
Adam second moment
optimizer bookkeeping
```

Required semantics:

1. snapshot off-mask parameter values before the optimizer step;
2. apply the masked update;
3. restore off-mask parameter values exactly;
4. reset/clear off-mask Adam state as required by the implementation;
5. preserve state for coordinates that remain continuously selected.

At minimum, off-mask handling must cover:

```text
exp_avg
exp_avg_sq
```

Do not claim strict sparse adaptation if off-mask parameters can drift because of AdamW state or weight decay.

Add tests that verify off-mask values are bitwise/numerically unchanged after sparse steps.

---

## 19. Group Split-LBI — corrected refined semantics

The old Transformer proposal contains outdated Split-LBI details. Use the refined semantics below.

For current batch `B_t`, let the persistent model entering the batch be:

\[
\theta_{\mathrm{base}}^t.
\]

### 19.1 Batch-local restart

At the beginning of every online batch:

\[
\theta_\Delta^0=0,\qquad
\gamma^0=0,\qquad
z^0=0.
\]

These three variables do **not** persist across batches.

Only the adapted persistent model state is carried to the next batch.

### 19.2 Stage-1 objective

\[
L(\theta_\Delta,\gamma)
=
L_{\mathrm{SHOT}}
\left(
B_t;\theta_{\mathrm{base}}+\theta_\Delta
\right)
+
\frac{1}{2\nu}
\|\theta_\Delta-\gamma\|_2^2.
\]

For structural Group Split-LBI, the sparsity/prox is group-wise, but the functional coupling follows the same refined Split-LBI logic.

### 19.3 Correct old-state update

The corrected update is:

\[
\theta_\Delta^{k+1}
=
\theta_\Delta^k
-
\alpha\kappa
\nabla_{\theta_\Delta}
L(\theta_\Delta^k,\gamma^k),
\]

\[
z^{k+1}
=
z^k
-
\alpha
\nabla_\gamma
L(\theta_\Delta^k,\gamma^k),
\]

\[
\gamma^{k+1}
=
\kappa\operatorname{prox}_{J}(z^{k+1}).
\]

Critical rule:

> The `z` update must use the **old state**
> \((\theta_\Delta^k,\gamma^k)\),
> not the newly updated \(\theta_\Delta^{k+1}\).

Do not implement the sequential bug where `theta_delta` is updated first and then reused inside the same-step `z` coupling gradient.

---

## 20. Group prox and support

For group `g`, use a group-sparsity prox rather than scalar element-wise support.

A standard group soft-threshold form is:

\[
\gamma_g
=
\kappa
\left(
1-\frac{\lambda}{\|z_g\|_2}
\right)_+
z_g.
\]

The exact implementation must preserve paired groups:

```text
Q row + K row
V row + O column
fc1 row + fc2 column
```

Support is defined at the group level.

The old FC scalar threshold `|gamma_j| >= 1e-4` cannot be transplanted blindly to a 768-scalar group without checking scale.

For the Transformer formal implementation, define and freeze a group support criterion before formal LBI runs, for example a normalized group norm:

\[
M_g
=
\mathbf 1
\left[
\frac{\|\gamma_g\|_2}{\sqrt{|g|}}
\ge
\tau_g
\right].
\]

Because all groups have equal size in the current DeiT-S protocol, equivalent unnormalized group-norm thresholding is possible, but the chosen form and `tau_g` must be frozen and logged.

Do not silently tune `tau_g` per transfer or per batch.

---

## 21. Strict Group-LBI budget and rollback

The maximum active group count is the same `K` used by the sparse baselines.

Random / Magnitude / Saliency:

```text
exactly K active groups
```

Group Split-LBI:

```text
active groups <= K
```

If Stage 1 transitions from a valid state to an overshoot state:

```text
last valid count <= K
new count > K
```

rollback to the last valid Stage-1 state.

Do not:

```text
accept hidden tolerance/slack
top-K trim gamma after overshoot
allow K+1 groups
```

Report:

```text
requested K
realized selected group count
utilization = realized_count / K
Stage-1 steps
rollback occurrence
```

---

## 22. Stage 2 — masked-delta initialization

The old dense-delta initialization is superseded.

Correct Stage-2 initialization:

\[
\theta_{\mathrm{init}}
=
\theta_{\mathrm{base}}
+
M\odot\theta_\Delta.
\]

Off-mask Stage-1 delta must not enter the persistent model.

During Stage 2:

```text
only selected groups may change
off-mask values must remain equal to theta_base
off-mask optimizer state must not create drift
```

Current design keeps:

```text
stage2_steps = 1
```

Stage-2 LR is fixed during the batch.

Do not apply a batch-local SHOT LR decay inside Stage 2.

---

## 23. Persistent omega accumulation

After Stage 2 obtains the refined batch model:

\[
\theta_{\mathrm{refined}}^t,
\]

the next persistent model uses:

\[
\theta_{\mathrm{base}}^{t+1}
=
(1-\omega)\theta_{\mathrm{base}}^t
+
\omega\theta_{\mathrm{refined}}^t.
\]

Then the next batch starts with fresh:

```text
theta_delta = 0
gamma = 0
z = 0
```

`omega` changes future model state and therefore changes future gradients and support discovery.

It is not a cosmetic output interpolation.

---

## 24. LBI hyperparameters

Do not copy the final FC winner tuple directly to Transformer.

The following must be retuned for the Transformer Group-LBI implementation:

```text
alpha
kappa
nu
omega
stage2_lr
```

Also validate/freeze Transformer-specific:

```text
group support threshold tau_g
stage1_max_steps
```

Recommended search structure:

### Phase A — Stage-1 dynamics

Start with an FC-inspired anchor grid for:

```text
alpha
kappa
nu
```

but treat those numbers only as pilot starting points.

Reject configurations that show:

```text
NaN / Inf
budget violation
persistent max-step failures
support never grows
support instantly explodes
very poor utilization
```

The 3000-step value is a batch-local Stage-1 cap, not an outer-stream stop.
If any online batch reaches exactly 3000 Stage-1 steps, retain the cap-hit
diagnostic, finish that batch normally, and continue through every later batch
and the complete FO pass. A cap hit alone does not invalidate the LBI run.

### Phase B — persistent/refinement interaction

For shortlisted Stage-1 settings, jointly sweep:

```text
omega × stage2_lr
```

because these two parameters interact.

For Office-31, one selected tuple per budget must be shared across all six transfers.

Forbidden tuning:

```text
per transfer
per formal seed
per target batch
```

---

## 25. Baseline matrix and parameter unification

Formal controlled comparison:

| Method | Candidate scope | Support | Mask type | Optimizer |
|---|---|---|---|---|
| Source-only | none | none | none | none |
| Full-dense | all except head | unrestricted | dense | Transformer AdamW |
| Candidate-dense | last-3 QKV/O/MLP weights | all 6912 groups | dense | shared AdamW |
| Random | same candidate | exact K groups | static | shared AdamW |
| Magnitude | same candidate | exact K groups | static | shared AdamW |
| Saliency | same candidate | exact K groups | dynamic | shared AdamW |
| Group Split-LBI | same candidate | <= K groups | discovered per batch | LBI Stage-1 + fixed Stage-2 optimizer |

Random / Magnitude / Saliency must not get separate tuned base learning rates.

Do not compare Group-LBI using paired structural groups against scalar Random/Magnitude/Saliency and claim selector superiority. All matched sparse selectors must operate on the same structural group pool.

---

## 26. Formal run seed and Random child masks

Use one formal experimental seed per formal condition.

Current refined FC protocol uses:

```text
formal seed = 2026
```

Unless a separate Transformer protocol explicitly changes it, use the same formal seed for the Transformer rerun.

Random masks are child executions, not formal seeds.

Example:

```text
formal_seed = 2026
random_mask_index = 0
random_mask_index = 1
random_mask_index = 2
```

Derived RNG seeds must be:

```text
deterministic
distinct
recorded
stable across reruns
```

---

## 27. Results to report

For every formal run retain:

```text
PU primary metric
FO primary metric
dataset-specific diagnostics
requested budget rho
requested K
realized K
runtime
peak GPU allocated memory
source checkpoint hash
Git commit
scientific config hash
```

For Office:

```text
per-transfer PU
per-transfer FO
six-transfer PU average
six-transfer FO average
```

For VisDA:

```text
PU 12-class mAcc
FO 12-class mAcc
12 class-wise accuracies
overall accuracy diagnostic
```

For Random:

```text
3-mask mean accuracy
3-mask std
mean single-mask comparable runtime
three-mask operational total runtime
```

For Group-LBI additionally:

```text
Stage-1 steps per batch
realized utilization
rollback count
Stage-1 runtime
Stage-2 runtime
selected QK / VO / FFN group counts
historical active-group union
final source-relative parameter delta support
```

---

## 28. Runtime / GPU-memory protocol

Efficiency should be measured per online batch.

Timer boundary:

```text
DataLoader has yielded
H2D transfer has completed
-> start timer
adaptation
PU forward
-> stop timer
```

Exclude:

```text
DataLoader wait
CPU preprocessing
H2D
checkpoint I/O
JSON serialization
summary I/O
```

For CUDA, synchronize before and after timed regions.

Record:

```text
adapt_runtime_sec
pu_runtime_sec
online_runtime_sec
peak_gpu_memory_allocated_mb
peak_gpu_memory_reserved_mb
```

Primary memory metric:

```text
max peak allocated memory over all online batches
```

Do not compare runtime numbers across different hardware as if they were directly equivalent.

---

## 29. Superseded old Transformer implementation details

The new rerun must not silently reuse these old behaviors:

```text
budget = ceil(rho * 6912)
```

Replace with:

```text
budget = floor(rho * 6912)
```

Old magnitude:

```text
group score = sum(abs(W))
```

Replace with:

\[
\|W_g\|_2.
\]

Old saliency:

```text
group score = gradient L2 norm only
```

Replace with:

\[
\|W_g \odot \nabla W_g\|_2.
\]

Old proposal Stage-1 equations that do not preserve corrected old-state semantics are superseded.

Old Stage-2 dense-delta initialization is superseded by masked-delta initialization.

Any sparse path that only masks gradients but allows AdamW off-mask drift is superseded.

The previous Transformer proposal `0.005 / 0.01 / 0.02` with
`34 / 69 / 138` groups is superseded by the explicit 2026-08-24 revision to
`0.0005 / 0.001 / 0.002` with `3 / 6 / 13` groups.

---

## 30. Testing requirements before formal rerun

At minimum add/maintain tests for:

1. exactly 6912 structural groups;
2. exactly 768 scalar weights per group;
3. no structural group overlap;
4. exact coverage of candidate qkv/proj/fc1/fc2 slices;
5. QK paired masking;
6. VO paired masking;
7. FFN paired masking;
8. exact floor budget `3 / 6 / 13`;
9. Random exact-K selection;
10. deterministic three-mask reproduction;
11. Magnitude top-K by group L2;
12. Saliency top-K by `|W*g|` group L2;
13. AdamW off-mask parameter invariance;
14. Adam state off-mask handling;
15. Group prox numerical correctness;
16. old-state `z` update correctness;
17. strict rollback correctness;
18. masked-delta Stage-2 initialization;
19. post-update PU uses a separate forward;
20. PU evaluation does not mutate persistent state;
21. FO is an independent full-target pass;
22. Office correct/total aggregation;
23. VisDA fixed-12-class macro aggregation;
24. source checkpoint local-only loading;
25. experiment identity changes when scientific fields change.

---

## 31. Development workflow for agents

Before editing:

```bash
cd /home/nas3/biod/wangkangyi/transformer-tta
git status --short --branch
```

Read:

```text
AGENTS.md
protocol/shot-otta_fc/OTTA_FC_LBI_PROTOCOL_20260817_v1.md
```

Then inspect the current implementation and tests.

Rules:

- Do not overwrite unrelated user changes.
- Do not silently change a frozen scientific semantic.
- If algorithm semantics change, update code + tests + config + documentation together.
- Do not commit datasets, checkpoints, caches, logs, or large run artifacts.
- Do not auto-download pretrained models on the server.
- Run CPU/synthetic tests first.
- Then run one single-GPU short-stream pilot.
- Only after the pilot passes should the full 8-GPU matrix be launched.
- A formal result is invalid if its config, checkpoint, metric definition, candidate space, group definition, budget rule, or formal seed differs from the frozen condition without an explicit protocol revision.

---

## 32. Current execution checklist

Before launching the new Transformer baseline matrix, verify all boxes:

```text
[ ] source checkpoints resolved and hashed
[ ] target stream shuffle/order bug fixed
[ ] PU is post-update and read-only
[ ] FO is frozen full-target evaluation
[ ] Office metric = sample-level overall accuracy
[ ] Office final average = equal-weight mean of 6 transfers
[ ] VisDA metric = fixed-12-class mean per-class accuracy
[ ] candidate = blocks 9-11 qkv/proj/fc1/fc2 weights only
[ ] bias frozen
[ ] LayerNorm frozen
[ ] head frozen for controlled family
[ ] 6912 paired groups validated
[ ] budget uses floor
[ ] formal budgets = 0.0005 / 0.001 / 0.002
[ ] K = 3 / 6 / 13
[ ] Candidate-dense / Random / Magnitude / Saliency share AdamW config
[ ] Random uses three deterministic child masks
[ ] Magnitude uses W0 group L2
[ ] Saliency uses group L2 of |W * grad|
[ ] sparse AdamW cannot move off-mask coordinates
[ ] Group-LBI uses corrected old-state z update
[ ] Group-LBI uses strict rollback
[ ] Group-LBI Stage-2 init is masked delta
[ ] Stage-2 uses fixed LR and one refinement step
[ ] theta_delta / gamma / z reset every online batch
[ ] omega updates persistent model only after Stage 2
[ ] LBI alpha/kappa/nu/omega/stage2_lr are retuned for Transformer
[ ] runtime and GPU memory use the same measurement boundary
```

---

## 33. One-sentence scientific principle

> **All controlled Transformer methods must share the same source checkpoint, target stream, SHOT objective, candidate space, structural groups, metric definitions, optimizer family, and budget semantics; the selector is the primary controlled difference.**
