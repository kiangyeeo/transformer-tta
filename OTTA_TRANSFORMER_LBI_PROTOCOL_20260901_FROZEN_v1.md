# OTTA_TRANSFORMER_LBI_PROTOCOL_20260901_FROZEN_v1

**Status:** Frozen overall evaluation protocol / v1  
**Scope:** DeiT-S + SHOT-OTTA + structural Group Split-LBI controlled adaptation  
**Datasets:** Office-31, VisDA-C  
**Target submission:** ICLR, late September 2026  
**Semantic parent:** `transformer-tta/protocol/shot-otta_fc/OTTA_FC_LBI_PROTOCOL_20260817_v1.md`  
**Transformer implementation revision:** `transformer_group_lbi_otta_20260824_v2` at Git revision `8d878bceec9ae41c3b58c155db323eea9417c271`  
**Protocol date:** 2026-09-01

> This document defines the **overall Transformer OTTA evaluation track**.  
> The Transformer method preserves the refined FC-LBI algorithmic semantics and replaces scalar support with architecture-aware QK / VO / FFN group support.  
> Tuning, the formal comparison matrix, and the independent post-tuning rerun of the 21 frozen-winner Group-LBI conditions are complete. This document freezes the protocol actually used for final Transformer evaluation; no post-formal retuning or semantic change is permitted under this protocol identity.

---

## 1. Scope and design principle

This protocol applies to:

```text
OTTA method       : SHOT-OTTA
backbone          : non-distilled DeiT-S/16-224
adaptation scope  : controlled last-3-block Transformer weights
sparse method     : structural Group Split-LBI
datasets          : Office-31, VisDA-C
formal seed       : 2026
```

The controlled comparison is:

```text
candidate_dense
group_random
group_magnitude
group_saliency
group_lbi
```

All five use the same candidate tensors. Random, Magnitude, Saliency, and Group-LBI additionally share the same structural group pool and strict integer budget.

The study concerns **sparse parameter adaptation**, not physical pruning. An unselected group remains at its persistent pretrained/source value; it is not removed from the graph. No FLOP reduction, inference acceleration, or model-compression claim is permitted without a separate export/restructuring experiment.

---

## 2. Inherited benchmark / evaluation invariants

### 2.1 Office-31

| Setting | Value |
|---|---|
| Dataset | Office-31 |
| Backbone | DeiT-S/16-224 |
| Classes | 31 |
| Online / FO batch size | 64 / 64 |
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

For target Amazon (2,817 samples), the final 1-sample tail is merged into the preceding 64-sample batch, yielding a final online batch of 65. Other transfers retain ordinary `drop_last=false` batching.

### 2.2 VisDA-C

| Setting | Value |
|---|---|
| Dataset | VisDA-C |
| Backbone | DeiT-S/16-224 |
| Classes | 12 |
| Online / FO batch size | 256 / 256 |
| DataLoader workers | 4 |
| Target passes | 1 |
| `drop_last` | false |
| Formal seed | 2026 |
| Primary PU / FO metric | fixed-12-class mean per-class accuracy (mAcc) |

Formal transfer:

```text
Synthetic / Train -> Real / Validation
```

For both PU and FO, retain all 12 per-class accuracies, fixed-12-class macro mean accuracy, and overall sample accuracy. Also retain worst-class accuracy/class name and class-wise standard deviation.

### 2.3 Target stream and transforms

For fixed `(dataset, transfer, seed=2026)`, every variant observes the same fixed random target permutation and augmentation RNG.

Online transform:

```text
Resize((256,256), bilinear)
RandomCrop(224)
RandomHorizontalFlip(0.5)
ImageNet normalization
```

FO transform:

```text
Resize((256,256), bilinear)
CenterCrop(224)
ImageNet normalization
```

The current source-training pipeline uses bicubic interpolation, whereas TTA uses bilinear. This mismatch must be disclosed and frozen as part of the current protocol; it must not be changed for only a subset of methods.

### 2.4 Numerical precision

Source training uses AMP. All formal TTA variants use the repository's non-AMP path consistently.

### 2.5 Result serialization and display precision

Computation, serialization, and display are separate. Accuracy and all ranking inputs are computed from the unrounded floating-point values. JSON and CSV artifacts retain Python's raw serialized float representation without protocol-level rounding or truncation; a value such as `63.72026979055733` is stored as the computed float representation. Tuning and ranking read these raw numeric fields and must never compare preformatted strings or rounded table values.

Human-readable Markdown and paper tables may format values to four decimal places. Display formatting must not modify the underlying artifact, be written back as the authoritative numeric value, or affect eligibility, shortlist, ranking, or winner selection.

---

## 3. Source model provenance and training

### 3.1 Implementation and ImageNet initialization

The authoritative model is:

```text
timm.create_model("deit_small_patch16_224.fb_in1k", pretrained=False, ...)
timm version: 1.0.28
architecture: non-distilled DeiT-S/16-224
blocks: 12
hidden dimension: 384
MLP dimension: 1536
```

This uses the timm implementation, not the Facebook DeiT repository implementation. The `.fb_in1k` suffix denotes timm's Facebook ImageNet-1K weight configuration/source label.

ImageNet initialization is strict-loaded locally from:

```text
checkpoints/deit_small_patch16_224.fb_in1k/model.safetensors
SHA-256: 1e747b4a8d0df2cfbd3c450e8c97685d867448ab0c2ddbfb34b6885f5cb23e5b
```

Implicit network download is forbidden. After loading the 1,000-class model, `reset_classifier()` creates `Linear(384,31)` for Office-31 or `Linear(384,12)` for VisDA-C. New head weights use truncated normal with std 0.02; bias is zero.

### 3.2 Source-domain split and recipe

| Setting | Office-31 | VisDA-C synthetic source |
|---|---:|---:|
| Train/validation split | per-class stratified 90/10 | per-class stratified 90/10 |
| Split / train seed | 2026 / 2026 | 2026 / 2026 |
| Train/val sizes | A 2534/283; D 448/50; W 715/80 | 137158/15239 |
| Epochs | 100 | 10 |
| Global batch size | 64 | 64 |
| Optimizer | AdamW | AdamW |
| Betas / epsilon | (0.9, 0.999) / 1e-8 | same |
| Backbone LR | 6.25e-5 | 6.25e-5 |
| Head LR | 6.25e-4 | 6.25e-4 |
| Weight decay | 0.05 | 0.05 |
| No-decay set | bias / norm / token | bias / norm / token |
| Scheduler | per-step cosine | per-step cosine |
| Warmup / min LR | 5 epochs / 1e-6 | 1 epoch / 1e-6 |
| Loss | CE + label smoothing 0.1 | same |
| Fine-tune scope | full model | full model |
| AMP | on | on |
| Drop / drop-path | 0 / 0 | 0 / 0 |
| EMA / gradient clipping | off / none | off / none |

Source train uses bicubic resize, random crop, and random horizontal flip; source validation uses bicubic resize and center crop. The best checkpoint is selected by source validation accuracy.

### 3.3 Formal source checkpoints

```text
A -> D, A -> W : checkpoints/source_models/office31/amazon.pth
D -> A, D -> W : checkpoints/source_models/office31/dslr.pth
W -> A, W -> D : checkpoints/source_models/office31/webcam.pth
VisDA train -> validation : checkpoints/source_models/visda-c/train.pth
```

The formal TTA state `W0` is the selected source-trained best `.pth`, not the raw ImageNet checkpoint and not `*.last.pth`. Every formal run records absolute path, SHA-256, best epoch, source validation metric, training config, Git state, and software/CUDA environment.

Verified best-checkpoint source-validation results are:

| Source domain | Selected epoch (one-based) | Selection metric | Best validation result |
|---|---:|---|---:|
| Amazon | 8 | overall accuracy | 88.6926% |
| DSLR | 5 | overall accuracy | 98.0000% |
| Webcam | 6 | overall accuracy | 100.0000% |
| VisDA synthetic | 9 | fixed-class macro accuracy | 99.9952% |

The corresponding VisDA overall validation accuracy is 99.9934%. These are held-out source-validation results from the checkpoint manifests, not target-domain OTTA results.

---

## 4. SHOT objective and causal constraint

All SHOT variants use:

$$
\mathcal L_{\mathrm{SHOT}}
=
0.3\mathcal L_{\mathrm{pseudo}}
+\mathcal L_{\mathrm{ent}}
+\mathcal L_{\mathrm{div}}.
$$

Current-batch pseudo-labels are:

$$
\hat y_i=\arg\max_c p_\theta(c\mid x_i).
$$

Use `cls_par=0.3`, `ent_par=1.0`, and confidence threshold `0.0`. Pseudo-labels are causal and current-batch only. Full-target clustering, replay, and target labels in adaptation are forbidden.

---

## 5. Native references versus controlled Transformer family

### 5.1 Source-only

No parameter updates. The source model must remain parameter-identical before and after the stream.

### 5.2 Full-dense

`full_dense` is the native/unrestricted SHOT reference. It updates all DeiT parameters except classifier/head, including attention/MLP weights and biases, LayerNorm, patch embedding, class token, position embedding, and final norm. The classifier weight and bias remain frozen. The model remains in evaluation mode.

It is not a matched-support selector baseline.

### 5.3 Controlled candidate family

The controlled family is:

```text
candidate_dense
group_random
group_magnitude
group_saliency
group_lbi
```

Only declared candidate weights may change persistently. All biases, LayerNorm parameters, class token, position embedding, patch embedding, final norm, classifier/head, and blocks 0--8 are frozen. The model remains in evaluation mode.

---

## 6. Transformer candidate scope

Candidate blocks:

```text
blocks 9, 10, 11
```

Candidate tensors in every block:

```text
attn.qkv.weight
attn.proj.weight
mlp.fc1.weight
mlp.fc2.weight
```

Totals:

| Quantity | Value |
|---|---:|
| Candidate blocks | 3 |
| Candidate tensors | 12 |
| Candidate scalar weights | 5,308,416 |
| Structural groups | 6,912 |
| Scalars per group | 768 |

The fused timm `qkv.weight` with shape `[3d,d]` remains physically fused; Q/K/V are logical slices used only for group construction.

---

## 7. Structural group definitions

Let `d=384` and the MLP hidden dimension be 1536.

### 7.1 QK group

For block $l$ and coordinate $p$:

$$
G_{l,p}^{QK}=\{W_{Q,l}[p,:],W_{K,l}[p,:]\}.
$$

### 7.2 VO group

$$
G_{l,p}^{VO}=\{W_{V,l}[p,:],W_{O,l}[:,p]\}.
$$

### 7.3 FFN group

$$
G_{l,p}^{FFN}=\{W_{1,l}[p,:],W_{2,l}[:,p]\}.
$$

Per block:

```text
QK  = 384 groups
VO  = 384 groups
FFN = 1536 groups
total = 2304 groups
```

Across three blocks, `N_group=6912`. Every group contains exactly 768 unique scalar weights, and groups partition the controlled candidate coordinates exactly once. Therefore structural group ratio and controlled-candidate scalar ratio coincide up to integer flooring.

---

## 8. Group-Lasso penalty and normalized support

The main method uses unweighted Group Lasso:

$$
J(\Gamma)=\sum_{g\in\mathcal G}\|\Gamma_g\|_2.
$$

Because every Transformer group has equal size, support scoring is normalized as:

$$
s_g=\frac{\|\Gamma_g\|_2}{\sqrt{|g|}},\qquad |g|=768.
$$

A group is active iff:

$$
M_g=\mathbf 1[s_g\ge\tau_g],\qquad \tau_g=10^{-4}.
$$

The same thresholded definition is used for Stage-1 counting, budget checks, rollback, final masks, utilization, masked-delta initialization, and Stage-2 trainability. `Gamma != 0`, delta nonzero, or gradient nonzero are not formal support definitions.

---

## 9. Restart Group Split-LBI semantics

For online batch $B_t$, let the persistent candidate state entering the batch be $\Theta_t$.

### 9.1 Batch-local restart

$$
\Theta_\Delta^0=0,\qquad Z^0=0,\qquad\Gamma^0=0.
$$

The persistent model carries across batches; `Theta_delta`, `Z`, and `Gamma` do not.

### 9.2 Stage-1 task gradient

$$
g^k=\nabla_{\Theta_\Delta}\mathcal L_{\mathrm{SHOT}}(B_t;\Theta_t+\Theta_\Delta^k).
$$

### 9.3 Correct old-state coupling

$$
c^k=\frac{\Theta_\Delta^k-\Gamma^k}{\nu},
$$

$$
\Theta_\Delta^{k+1}=\Theta_\Delta^k-\alpha\kappa(g^k+c^k),
$$

$$
Z^{k+1}=Z^k+\alpha c^k.
$$

The `Z` update must use old-state `(Theta_delta^k, Gamma^k)`, not the updated delta.

### 9.4 Group proximal operator

$$
\Gamma_g^{k+1}
=\kappa\left(1-\frac{\lambda_{prox}}{\|Z_g^{k+1}\|_2}\right)_+Z_g^{k+1},
$$

with `prox_lambda=1.0`. Zero norms must be handled safely without changing support semantics.

---

## 10. Sparse budget and Stage-1 validity

### 10.1 Formal budgets

The current formal Transformer budget grid is:

| rho | K = floor(rho x 6912) | Active candidate scalars at exact K |
|---:|---:|---:|
| 0.0005 | 3 | 2,304 |
| 0.0010 | 6 | 4,608 |
| 0.0020 | 13 | 9,984 |

The budget is global across all blocks and group types. No `ceil`, hidden slack, per-block/type quota, or minimum-one rule is allowed.

### 10.2 Strict rollback

At Stage-1 step $k$, let $S_k=\sum_g M_g^k$:

```text
S_k < K : save current feasible state and continue
S_k = K : accept current state and stop with exact budget reached
S_k > K : reject overshoot state, rollback to latest feasible state, and stop
```

Overshoot is never repaired by top-K trimming or filling.

### 10.3 Formal validity decision

| Situation | Required action | Formal status |
|---|---|---|
| Final `S_t=K` | accept | valid |
| `S_t<K` before cap | continue | undecided |
| Step jumps from `<K` to `>K` | rollback to latest feasible state | budget-feasible; valid if no cap failure |
| Cap reached while `S_t<K` | mark scientific-invalid and terminate the current run | **Stage-1 failure / invalid** |
| Any `S_t>K` after rollback | abort | budget violation / invalid |

Exact-K is not required after a legitimate overshoot rollback. A hit at the fixed 3,000-step cap is scientific-invalid: after the current `run_batch()` returns, the runner records the invalid reason and termination batch, discards the condition from selection/formal reporting, performs no PU, and processes no subsequent target batches. In revision `8d878bc`, Stage 2 is internal to `run_batch()` and is therefore computed once for the invalidating batch before runner-level invalidation; that computation has no reportable scientific result and is discarded. The protocol does not claim that the invalidating batch avoids Stage-2 compute.

Initial safety cap:

```text
stage1_max_steps = 3000
```

It is fixed, not a tuning dimension.

---

## 11. Stage-2 masked refinement

Given final feasible `(Theta_delta*, Gamma*, M*)`, initialize:

$$
\Theta_{t,2}^{0}=\Theta_t+M^*\odot\Theta_\Delta^*.
$$

Off-mask Stage-1 delta values are not applied.

Stage-2 rules:

```text
only selected coordinates may change
off-mask gradients are masked
off-mask parameter values are restored after the optimizer step
off-mask exp_avg and exp_avg_sq are cleared and asserted zero
```

| Setting | Value |
|---|---:|
| Optimizer | AdamW |
| Betas | (0.9, 0.999) |
| Epsilon | 1e-8 |
| Weight decay | 0.01 |
| Steps | 1 |
| LR schedule | none |
| LR | fixed, selected per dataset/budget profile |
| AMP | off |

---

## 12. Persistent accumulation

Let $\widetilde\Theta_t$ be the refined Stage-2 state:

$$
\Theta_{t+1}=(1-\omega)\Theta_t+\omega\widetilde\Theta_t.
$$

The next batch begins from $\Theta_{t+1}$, while all local LBI states restart from zero. Structural support is rediscovered independently for every online batch.

---

## 13. Controlled baseline definitions

### 13.1 Candidate-dense

Updates all 5,308,416 candidate scalar weights in the 12 declared tensors and nothing else.

### 13.2 Random

Uniformly selects exactly `K` groups from the canonical global group pool. Use three deterministic child masks:

```text
mask seeds = 202600, 202601, 202602
cross-budget policy = nested prefixes of the same permutation
mask refresh = once before adaptation
```

Report accuracy as the three-mask mean with population standard deviation. Comparable efficiency is mean single-mask cost; total three-mask operational cost is stored separately. These masks are not three formal experiment seeds.

### 13.3 Magnitude

Formal selector name: **paired-group L2 norm**.

For each Transformer structural paired group $g$ (QK, VO, or FFN), score:

$$
s_g=\|W_g\|_2.
$$

Scores are computed once at source `W0` in float64 on CPU with canonical group-ID tie break. Select global exact top-K; the mask stays fixed throughout the target stream. This is not scalar magnitude, parameter-wise magnitude, or a layer norm.

### 13.4 Saliency

Formal selector name: **paired-group L2 norm of (weight * gradient)**.

For every online batch, after backward and before update, score each complete QK, VO, or FFN paired group:

$$
s_g=\|(W\odot\nabla_W\mathcal L_{\mathrm{SHOT}})_g\|_2.
$$

Select global exact top-K with canonical tie break. The mask is refreshed every online batch using the current pre-update weights and current-batch SHOT gradient. This is not a gradient norm or parameter-wise saliency.

The immutable v2 machine metadata string `paired_group_l2_norm_of_abs_weight_times_gradient` is a legacy serialization alias for the same score: taking elementwise absolute value before squaring does not change the L2 norm. New documentation and display labels use the canonical names above; existing formal artifacts and their hashes are not rewritten.

### 13.5 Selector matching

Within each budget, Random, Magnitude, Saliency, and Group-LBI use identical candidate blocks/tensors, structural groups, integer K, Stage-2 optimizer family, target stream, seed, and non-AMP precision. Group-LBI has an additional Stage-1 discovery cost that must be reported separately.

---

## 14. PU / FO evaluation

### 14.1 PU

```text
receive B_t
-> adapt using unlabeled B_t
-> write persistent post-update state
-> separately evaluate post-update accuracy on B_t
-> guarantee evaluation is read-only
-> continue to B_{t+1}
```

PU is not computed from pre-update loss logits. PU is report-only and never participates in hyperparameter selection.

### 14.2 FO

After the online stream:

```text
stop adaptation
-> freeze final model and method state
-> run a separate full target evaluation using CenterCrop
```

FO is read-only; no update, optimizer step, or method-state mutation is permitted.

---

## 15. Accuracy aggregation

Office transfer accuracy is sample-level `correct/total`, never the unweighted mean of batch accuracies. Office overall score is the equal mean of six transfer accuracies, not a pooled-sample score.

VisDA primary accuracy is:

$$
\mathrm{mAcc}=\frac{1}{12}\sum_{c=1}^{12}\mathrm{Acc}_c.
$$

Overall sample accuracy and class diagnostics are secondary retained metrics.

---

## 16. Scientific validity and utilization diagnostics

A formal Group-LBI condition is scientifically valid only if:

```text
all online batches complete
no NaN / Inf / runtime exception
selected_group_count <= K for every batch
budget_violation_rate = 0
failure_rate = 0
max-step-underfill count = 0
support and timing records are complete
```

For every batch, record:

$$
u_t=\frac{S_t}{K}.
$$

Define:

$$
\mathrm{util95\_rate}
=
\frac{\#\{t:u_t\ge 0.95\}}
{\#\{\text{processed online batches}\}}.
$$

For one transfer, this is computed over that transfer's processed online batches. For an Office candidate, the ranking implementation pools the six transfers at the batch-count level:

$$
\mathrm{Office\ util95\_rate}
=
\frac{\sum_{d=1}^{6}\#\{t\in d:u_t\ge0.95\}}
{\sum_{d=1}^{6}\#\{t\in d\}}.
$$

It is not the equal mean of six transfer-level rates. This matches `_pooled_tuning_diagnostics()` and the tuning summarizer, which sum `utilization_ge_95_count` and divide by the total number of Office online batches.

Required summaries:

```text
selected groups min / mean / max
utilization min / mean / p05
count and fraction below 90% and 95%
exact-K reach rate
underfill failure rate
max-step hit count/rate
Stage-1 steps mean / max
rollback count/rate
```

Low utilization caused by legitimate rollback is reported separately from max-step failure. There is no hard 90% or 95% utilization eligibility gate. `util95` is a post-validity ranking diagnostic/tie-breaker only.

---

## 17. LBI fixed constants and tuning dimensions

### 17.1 Fixed constants

| Setting | Value |
|---|---:|
| `prox_lambda` | 1.0 |
| `tau_g` | 1e-4 |
| Stage-1 max steps | 3000 |
| Stage-2 optimizer | AdamW |
| Stage-2 betas / eps | (0.9,0.999) / 1e-8 |
| Stage-2 weight decay | 0.01 |
| Stage-2 steps | 1 |
| Stage-2 scheduler | none |
| batch-local restart | every online batch |
| strict rollback | enabled |
| top-K trim/fill | disabled |
| AMP | off |

`prox_lambda` is technically CLI/profile-overridable and appeared in historical searches, but the current formal protocol freezes it at 1.0.

### 17.2 Tuning dimensions and declared grid

Stage-1 candidates:

| ID | alpha | kappa | nu |
|---|---:|---:|---:|
| A1 | 0.10 | 1.0 | 0.50 |
| A2 | 0.15 | 1.0 | 0.50 |
| A3 | 0.20 | 1.0 | 0.50 |
| A4 | 0.10 | 1.5 | 0.50 |
| A5 | 0.10 | 2.0 | 0.50 |
| A6 | 0.10 | 1.0 | 0.25 |
| A7 | 0.10 | 1.0 | 1.00 |
| A8 | 0.15 | 1.0 | 1.00 |

The formal Stage-1 search contains A1--A8 only. F1--F4 were historical proposed fallbacks and were not executed or considered in the formal selection process.

Stage-2 grid:

```text
omega     in {0.05, 0.10, 0.20, 0.30}
stage2_lr in {0.005, 0.010, 0.020}
```

Only `alpha`, `kappa`, `nu`, `omega`, and `stage2_lr` are formal tuning dimensions.

---

## 18. Formal tuning and selection procedure

For each `(dataset, budget)`:

```text
Stage-1 A1--A8 search
-> scientific-valid filtering
-> dataset-specific frozen ranking
-> shortlist the top 3 valid anchors
-> omega x stage2_lr search for each shortlisted anchor
-> repeat full scientific-validity checking for every Stage-2 configuration
-> apply the dataset-specific frozen ranking rule
-> freeze one tuple
-> independently rerun the frozen-winner Group-LBI conditions from source W0
```

If fewer than three Stage-1 anchors are scientific-valid for a dataset/budget, all and only those valid anchors enter Stage 2; no invalid candidate is added to fill the shortlist. Under the normal top-3 case, Stage 2 evaluates:

$$
3\ \text{anchors}\times4\ \omega\text{ values}\times3\ \text{LR values}
=36
$$

configurations per dataset/budget. Every configuration is revalidated because `omega` and `stage2_lr` change persistent online dynamics; Stage-1 anchor validity is not inherited as Stage-2 validity.

Selection requirements precede accuracy:

```text
failure_rate = 0
budget_violation_rate = 0
max-step-underfill rate = 0
```

PU is report-only. Utilization is not an eligibility gate.

Office-31 candidates are ranked lexicographically after validity by:

```text
1. mean FO margin
2. worst-transfer FO margin
3. mean FO
4. util95
5. Stage-1 maximum steps (lower is better)
6. Stage-1 mean steps (lower is better)
```

For dataset/budget/transfer $d$, define the frozen sparse reference:

$$
\mathrm{best\_sparse\_FO}_d
=
\max\left(
\mathrm{Random\_FO}_d,
\mathrm{Magnitude\_FO}_d,
\mathrm{Saliency\_FO}_d
\right),
$$

where `Random_FO` is the three deterministic child-mask mean. Then:

$$
\mathrm{FO\_margin}_d
=
\mathrm{LBI\_FO}_d-\mathrm{best\_sparse\_FO}_d.
$$

For Office-31:

$$
\mathrm{mean\_FO\_margin}=\frac{1}{6}\sum_{d=1}^{6}\mathrm{FO\_margin}_d,
$$

$$
\mathrm{worst\_transfer\_FO\_margin}=\min_d\mathrm{FO\_margin}_d,
$$

$$
\mathrm{mean\_FO}=\frac{1}{6}\sum_{d=1}^{6}\mathrm{LBI\_FO}_d.
$$

VisDA-C has one formal transfer, so `FO_margin = LBI_FO - best_sparse_FO`. Within one VisDA budget, the sparse reference is fixed across LBI candidates; therefore FO-margin descending and FO descending yield identical ordering. VisDA-C candidates are ranked lexicographically after validity by:

```text
1. FO margin descending (equivalently FO descending within the same budget)
2. util95
3. Stage-1 maximum steps (lower is better)
4. Stage-1 mean steps (lower is better)
```

For Office-31, one tuple per budget is shared across all six transfers; per-transfer tuning is forbidden. VisDA-C is tuned independently and does not inherit Office tuples. Per-seed, per-batch, per-class, or post-hoc boundary tuning is forbidden.

The frozen winners are:

| Dataset | rho | Anchor | alpha | kappa | nu | omega | Stage-2 LR |
|---|---:|---|---:|---:|---:|---:|---:|
| Office-31 | .0005 | A7 | .10 | 1.0 | 1.00 | .10 | .005 |
| Office-31 | .001 | A1 | .10 | 1.0 | .50 | .05 | .005 |
| Office-31 | .002 | A4 | .10 | 1.5 | .50 | .05 | .010 |
| VisDA-C | .0005 | A3 | .20 | 1.0 | .50 | .20 | .005 |
| VisDA-C | .001 | A5 | .10 | 2.0 | .50 | .20 | .005 |
| VisDA-C | .002 | A4 | .10 | 1.5 | .50 | .20 | .005 |

Office winners are shared by all six transfers at the corresponding budget. After tuple freeze, Group-LBI alone received an independent fresh rerun: Office contributes `3 budgets x 6 transfers = 18` conditions and VisDA-C contributes `3 budgets x 1 transfer = 3`, for 21 frozen-winner Group-LBI formal runs. Every one starts fresh from the corresponding source `W0`.

The broader formal comparison matrix combines these 21 Group-LBI runs with the frozen source-only, full-dense, candidate-dense, Random (three deterministic child masks), Magnitude, and Saliency artifacts. This protocol does not claim that every baseline was rerun after the Group-LBI winner freeze.

---

## 19. Formal Stage-1 diagnostics

Every frozen-winner formal run retains the per-batch and aggregate Stage-1 diagnostics required by Section 16, separately for each Office transfer and for VisDA-C. These include K, selected-group min/mean/max, utilization min/mean/p05, counts below 90% and 95%, exact-K reach rate, cap hits, Stage-1 step mean/max, and rollback count/rate.

Earlier Office pilot tables produced before the target-Amazon singleton-merge correction are **historical pre-fix pilots and scientifically superseded**. They are excluded from this frozen protocol and must not be cited as current formal diagnostics. Only artifacts from the 21 frozen-winner Group-LBI runs, corrected stream construction, and final validity implementation are reportable as formal Group-LBI results.

---

## 20. Runtime and GPU-memory protocol

Per online batch record:

```text
batch index / batch size
adapt runtime
PU runtime
online runtime
Stage-1 runtime
Stage-2 runtime
Stage-1 steps / stop reason
rollback indicator
peak allocated / reserved GPU memory
```

CUDA synchronization is required around timed compute regions. Exclude DataLoader wait, CPU preprocessing, host-to-device transfer, checkpoint I/O, and result serialization.

Primary summaries are mean/std/median/p95 online runtime, total online compute, Stage-1/Stage-2 runtime shares, and maximum allocated/reserved memory. Report GPU model, batch size, AMP status, software stack, and whether a run was resumed.

Formal measurements use NVIDIA GeForce RTX 3090 GPUs. The retained formal artifacts, rather than pre-fix pilot measurements, are authoritative for peak memory, Stage-1/Stage-2 runtime, online seconds per batch, steps, and cap-hit reporting. Stage 1 is reported separately because it dominates Group-LBI adaptation cost.

The budget grid is frozen at `.0005/.001/.002`; no larger budget is added after formal-result inspection.

---

## 21. Checkpoint / resume policy

Source-only and controlled baselines do not save adapted-model or partial-stream checkpoints; interrupted runs are restarted.

Group-LBI may checkpoint only at completed online-batch boundaries. Batch-local `Theta_delta`, `Gamma`, and `Z` must never persist across a boundary. Checkpoint I/O is outside efficiency timing, and no final adapted-model checkpoint is retained.

Resume is engineering resilience, not a different scientific identity. Formal efficiency comparisons should use fresh, non-resumed runs.

---

## 22. Provenance and experiment identity

Every formal run records at least:

```text
protocol and implementation revision
Git commit and dirty state
source checkpoint path / hash / manifest
dataset / transfer / formal seed
model and timm version
target stream and preprocessing identity
variant and update scope
candidate blocks / tensors / scalar count
group definitions / group count / group size
requested rho / integer K / realized support
SHOT configuration
complete LBI tuple and fixed constants
Stage-1 cap / stop-reason distribution
Stage-2 optimizer settings
PU / FO aggregation definition
GPU / CUDA / PyTorch / torchvision environment
artifact_path
log_path
```

When produced by the runner/aggregator, `summary_path` and `config_path` / `effective_config_path` are retained as well. The formal freeze artifact index records existing summary, metrics, manifest, effective-config, and log paths without inventing absent files. Every aggregate row must be traceable to the corresponding raw run artifact and execution log.

Any change to support, rollback, Stage-2 masking, persistent update, PU/FO, or validity semantics requires a new scientific implementation revision.

Revision audit: Git commit `8d878bceec9ae41c3b58c155db323eea9417c271` contains the final target-Amazon singleton-tail merge, strict-budget rollback, and runner-level 3,000-step scientific invalidation/early termination. The artifact-emitted implementation label remains `transformer_group_lbi_otta_20260824_v2`, so this protocol retains that label rather than inventing a v3 identity after execution; the full Git hash is authoritative. As documented in Section 10.3, invalidation occurs after `engine.run_batch()` returns, so this revision does not avoid Stage-2 computation inside the invalidating batch.

---

## 23. Formal evaluation matrix and order

Required methods:

```text
source_only
full_dense
candidate_dense
group_random x 3 child masks
group_magnitude
group_saliency
group_lbi
```

Sparse variants run at `rho={.0005,.001,.002}` on all six Office transfers and VisDA train->validation.

The completed execution order was:

```text
P0. freeze method, benchmark, budget, and search semantics
P1. enforce max-step scientific invalidation and singleton-merge stream behavior
P2. pass correctness and contract tests
P3. validate source checkpoints and baseline sanity
P4. complete the predeclared A1--A8 tuning search
P5. apply dataset-specific ranking and freeze the six winning tuples
P6. independently rerun the 21 frozen-winner Group-LBI conditions from source W0
P7. aggregate PU, FO, support, runtime, and memory
P8. assemble the formal comparison matrix from frozen baseline artifacts plus the 21 fresh Group-LBI runs
P9. freeze this v1 protocol; no further retuning or semantic change
```

---

## 24. Required correctness / contract tests

Before formal tuning and evaluation, automated tests verified:

1. structural groups cover each of the 5,308,416 candidate scalars exactly once;
2. global counts are 6,912 groups and 768 unique scalars per group;
3. logical Q/K/V slices match fused timm qkv layout;
4. `Z` uses old-state delta/gamma coupling;
5. group prox handles zero, threshold, and active norms correctly;
6. support uses normalized `||Gamma_g||_2/sqrt(768) >= 1e-4`;
7. integer budgets are `3/6/13`, global, with no slack;
8. overshoot restores the latest feasible state exactly and never top-K trims;
9. max-step underfill is rejected as Stage-1 failure;
10. masked-delta initialization excludes all off-mask delta values;
11. AdamW weight decay and moment state cannot move off-mask values;
12. local LBI states reset every batch and only omega-updated persistent state survives;
13. Random/Magnitude/Saliency select exact K from the identical global pool;
14. PU is post-update and read-only; FO is fully read-only;
15. Office and VisDA accuracy aggregations match Section 15;
16. identical stream ordering and augmentation RNG hold across variants.

These contracts are part of the frozen scientific implementation. A future violation invalidates the affected run and requires a new protocol/implementation revision.

---

## 25. Freeze control and paper terminology

### 25.1 Freeze control

The A1--A8 search space, no-hard-utilization-gate policy, six winning tuples, validity behavior, corrected target stream, formal comparison matrix, and 21-run post-tuning Group-LBI rerun are frozen under this v1 identity. Formal execution and aggregation are complete.

Any subsequent change to budgets, search space, ranking, source checkpoint, preprocessing, stream construction, support, rollback, validity, optimizer, PU/FO, or metric aggregation constitutes a new protocol revision and requires a new formal matrix. Existing v1 results must not be silently overwritten or retuned.

### 25.2 Terminology

Use:

```text
Transformer structural-group sparse adaptation
QK / VO / FFN paired groups
global structural budget
realized group/scalar update ratio
batch-local restart Group Split-LBI
persistent omega accumulation
```

Avoid:

```text
physical pruning
FLOP reduction
inference acceleration
network compression
formal result (for provisional-profile or pilot runs)
```

The central method statement is:

> Refined FC-LBI and Transformer Group Split-LBI share the same restart Sparse Delta Learning semantics. The Transformer extension changes only the sparsity unit: singleton scalar support is replaced by equal-size architecture-aware QK, VO, and FFN paired-group support.

The formal comparison matrix and the independent 21-condition frozen-winner Group-LBI rerun are complete, and this protocol is frozen. No further retuning, winner replacement, or protocol change is permitted for the reported v1 results.
