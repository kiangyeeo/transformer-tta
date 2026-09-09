# NCTTA-OTTA P0 Final Refactor Contract

**Status:** Frozen for baseline implementation  
**Date:** 2026-09-06  
**Project:** `260817_iclr2027-refined`  
**Scope:** NCTTA mechanism on the current controlled SHOT-OTTA substrate  
**Next:** freeze baseline protocol, then implement dense NCTTA baselines only.

## 1. One-line decision

The ICLR mainline will **not** reproduce the entire official TTAB/NCTTA pipeline. We retain the official NCTTA adaptation objective, but instantiate it on exactly the same source model, F/B/C architecture, online stream, optimizer/scheduler substrate, update scopes, BN controls, PU/FO evaluation, and provenance discipline already used by SHOT-OTTA and IST-OTTA.

Reason: the scientific question is whether Sparse-LBI works across different **adaptation objectives**. If source, architecture, stream, optimizer, update scope, and evaluation also change, the objective-agnostic claim becomes confounded.

## 2. Source of truth

Conflicts are resolved in this order:

1. `260817_iclr2027-refined/`
   - benchmark / source checkpoints
   - `netF -> netB -> netC`
   - target stream / seed / batch size / singleton behavior
   - PU / FO / metrics
   - FC / Conv candidate scopes
   - controlled BN semantics
   - optimizer / LR schedule substrate
   - runtime / artifact / provenance
   - corrected LBI semantics

2. Official NCTTA: `https://github.com/Cevaaa/NCTTA`
   - `ttab/model_adaptation/nctta.py`
   - `ttab/configs/algorithms.py`
   - NCTTA-specific feature-classifier objective, filtering, and weighting

3. New `nctta_otta/` code
   - integration only; it must not redefine either source of truth.

`shot_otta/**` is read-only.

## 3. What the official NCTTA method actually contributes

For a current batch, official NCTTA uses the classifier-input feature and frozen classifier weights to diagnose and reduce feature-classifier misalignment.

Let feature be $h_i\in\mathbb R^D$, classifier weights $W=[w_1,\dots,w_C]^T$, and logits $z_i$.

Official `compute_nc_loss`:

1. L2-normalizes features and classifier rows.
2. Uses current predicted probabilities to choose top-$k$ candidate classes.
3. Builds a distance-based soft target $q_{\rm dist}$ from normalized feature-classifier distances.
4. Builds a probability-based soft target $q_{\rm prob}$ from current model confidence.
5. Mixes the two targets:
$$
q=(1-\alpha)q_{\rm dist}+\alpha q_{\rm prob}.
$$
6. The actual NCTTA call uses InfoNCE-style NC loss, cosine geometry, `tau_align=1.0`, and `reduce="none"`.
7. It adds per-sample entropy:
$$
L_i^{\rm base}=H_i+sL_{{\rm NC},i}.
$$
8. It keeps samples satisfying:
$$
H_i<\gamma_{\rm ent}.
$$
9. It reweights each selected sample by:
$$
\lambda_i=
r_{\rm ent}\exp[-(H_i-m_{\rm ent})]
+\frac{\nu}{1+\eta d_i},
$$
where $d_i$ is the normalized feature distance to the currently predicted classifier weight.
10. The batch loss is the mean of selected weighted samples.

These semantics are method-specific and must be preserved.

## 4. Native NCTTA update scope vs our controlled study

Official NCTTA configures the model in train mode, freezes the full model, then enables normalization-layer affine parameters; its source comment explicitly says only BN is used in the paper. For BatchNorm2d it also disables tracked running statistics.

Therefore:

> **Native NCTTA is primarily normalization-only TTA.**

Our mainline deliberately changes the trainable parameter scope while keeping the NCTTA objective:

- `nctta_full_dense`: same `netF + netB` scope as SHOT full-dense
- `nctta_fc_module_dense`: only the same bottleneck FC weight/bias as SHOT FC module-dense
- `nctta_conv_module_dense`: only the same `netF.layer4` 9 Conv weights as SHOT Conv module-dense

Reason: the controlled comparison fixes the candidate universe and changes the adaptation objective. This is an intentional re-instantiation of the NCTTA objective, **not** an exact reproduction of the official NCTTA training scope.

An optional native norm-only NCTTA reference may be added later as a report-only diagnostic, but it must not block the matched FC/Conv mainline.

## 5. Common substrate: frozen

The following stay identical to current SHOT/IST formal semantics:

```text
source revision = nips2026_shot_otta_uda_source_v1
same source F/B/C checkpoints
same netF -> netB -> netC architecture
Office-31: ResNet-50, outer BS=64
VisDA-C: ResNet-101, outer BS=256
formal seed=2026
one target pass
same outer sample order / batch boundaries
same singleton handling as SHOT
same source transforms / evaluation reference view semantics
same PU / FO definitions
same Office / VisDA primary metrics
netC frozen
same optimizer / LR schedule substrate
same FC candidate
same Conv layer4 candidate
controlled FC/Conv BN frozen
same runtime / provenance discipline
```

No separate NCTTA source training is allowed.

## 6. Feature / classifier geometry: frozen and critical

Current F/B/C model:

$$
f_i=\mathrm{netF}(x_i)\in\mathbb R^{2048},
$$

$$
h_i=\mathrm{netB}(f_i)\in\mathbb R^{256},
$$

$$
z_i=\mathrm{netC}(h_i).
$$

The NCTTA feature must be:

$$
\boxed{h_i=\mathrm{netB}(\mathrm{netF}(x_i))}
$$

i.e. the **post-netB 256-d classifier-input feature**.

Do not copy the official generic ResNet `avgpool` forward hook into our model. `netF.avgpool` is 2048-d and is not the direct input of `netC`.

Classifier reference:

$$
\boxed{W_C=\mathrm{netC.fc.weight.detach()}}
$$

using the effective forward weight of the weight-normalized classifier, not an internal reparameterization tensor such as `weight_v`.

Required invariant:

```text
post-netB feature dim == effective classifier weight dim == 256
```

## 7. What we do NOT import from TTAB/NCTTA infrastructure

The main controlled branch does not import:

- TTAB dataset/scenario framework
- TTAB model-selection machinery
- oracle checkpoint selection using target labels
- episodic reset
- multi-step per-batch search
- Fisher regularization
- stochastic parameter restoration
- TTAB benchmark-specific source models

Reason: these are separate infrastructure/state variables and are not required to define the NCTTA objective we want to test.

Our scientific adaptation unit remains one incoming outer target batch.

## 8. NCTTA state machine on our substrate

NCTTA has no IST-style multi-view set, PLCA, memory bank, teacher, or host EMA.

For an incoming batch $B_t$:

```text
theta_t
-> forward current batch
-> post-netB feature h_t + logits z_t
-> recompute top-k / q_dist / q_prob / hybrid target
-> recompute entropy filter + sample weights
-> NCTTA loss
-> one declared optimizer update
-> theta_{t+1}
-> PU read-only
```

The model/optimizer trajectory is persistent. There is no additional method-specific persistent memory.

This simplicity is the main reason NCTTA should be integrated faster than IST.

## 9. Important rule for future NCTTA-LBI

Unlike IST, NCTTA does **not** first construct a fixed batch-local hard/soft target that should be frozen during adaptation.

The NCTTA objective depends on the current candidate model through:
- logits / predicted probabilities
- top-$k$ classes
- feature-classifier distances
- hybrid target
- entropy filter
- sample reweighting

Therefore, when future LBI Stage-1 evaluates candidate state $\theta_t+\Delta^k$:

$$
g^k=
\nabla_\Delta
L_{\rm NCTTA}(B_t;\theta_t+\Delta^k),
$$

the complete NCTTA objective above must be recomputed at that candidate state.

Do not compute NCTTA targets once at the start of the outer batch and reuse them across all LBI iterations unless a later protocol explicitly proves and freezes such a change.

## 10. Optimizer / scheduler

Use the same controlled SHOT/IST substrate, not the official NCTTA optimizer recipe as a separate experimental variable.

```text
Office base LR = 0.01
VisDA base LR = 0.001
netF LR multiplier = 0.1
netB LR multiplier = 1.0
SGD momentum = 0.9
weight decay = 0.001
Nesterov = true
```

Global one-pass schedule:

$$
\eta_t=\eta_0\left(1+10\frac{t}{T}\right)^{-0.75}.
$$

Scheduler time is outer-batch time.

## 11. NCTTA-specific hyperparameters

Official repository defaults currently expose:

```text
thre_ent        = 0.4 * log(1000)
margin_ent      = 0.4 * log(1000)
reweight_ent    = 1.0
nu              = 5.0
eta             = 1.0
scale           = 5.0
top_k           = 10
mix_prob_weight = 0.3
```

The official NCTTA call additionally fixes:

```text
NC type = infonce
metric = cos
tau_align = 1.0
margin = 0.2
```

P0 freezes the **meaning and code path** of these parameters, but does not yet declare the `log(1000)`-based defaults as the final Office-31 / VisDA-C formal values. That threshold is clearly expressed in a 1000-class form in the repository.

Rules:
- P1 must expose these fields explicitly and implement them faithfully.
- A short correctness smoke may use recorded official defaults.
- No parameter search is allowed during P1.
- Dataset-specific formal NCTTA objective values must be frozen before formal baseline/search and must never be changed after seeing LBI results.

## 12. Empty entropy-filter case

Official code computes a mean after entropy filtering, so an empty selected set can yield an invalid loss.

For P1:
- detect and record selected count explicitly;
- do not silently replace the official objective with “use all samples”;
- do not silently turn the loss into zero;
- correctness smoke should fail loudly with diagnostic metadata if the objective becomes non-finite.

A scientifically justified formal handling, if needed, must be written into the later baseline protocol before formal runs.

## 13. P1 dense variants only

P1 implements:

```text
nctta_full_dense
nctta_fc_module_dense
nctta_conv_module_dense
```

No Random, Magnitude, Saliency, or LBI in P1.

Recommended module layout:

```text
nctta_otta/
├── __init__.py
├── config.py
├── objective.py
└── trainer.py
```

NCTTA does not need IST-style `data.py`, `memory.py`, `plca.py`, or `ema.py`.

## 14. P1 contract requirements

Correctness first; accuracy must not choose the implementation.

1. NCTTA branch loads exactly the same source F/B/C as SHOT.
2. Deterministic initialization logits match the SHOT branch before adaptation.
3. Stream sample identities / batch boundaries / singleton behavior match current formal SHOT.
4. No target label enters adaptation.
5. No future-target information is accessed.
6. Alignment feature is post-netB 256-d.
7. Classifier reference is effective frozen `netC.fc.weight`.
8. `netC` remains byte-level unchanged.
9. NCTTA objective terms match the official code path.
10. Loss is finite for valid selected batches and gradients are nonzero on declared trainable parameters.
11. Full-dense changes only its declared SHOT-matched scope/state.
12. FC module-dense changes only bottleneck FC weight/bias; BN parameters and buffers remain unchanged.
13. Conv module-dense changes only layer4 9 Conv weights; BN parameters and buffers remain unchanged.
14. PU is completely read-only.
15. FO is completely read-only.
16. Scientific identity / protocol / implementation revision are exact and fail-safe.
17. Existing SHOT and IST regression tests still pass.

Contract tests should check state transitions and call/update counts, not only shapes.

## 15. Development order after this contract

```text
P0 final contract + baseline protocol
-> P1 dense implementation
-> dense contract tests
-> independent code review
-> Office D->A 5-valid-batch smoke
-> freeze implementation revision
-> write/freeze NCTTA sparse/LBI protocol
-> implement FC + Conv sparse families
-> sparse contract + independent review
-> short smoke
-> STOP
-> unified IST + NCTTA parameter-search design
-> tuning
-> final formal experiments
```

Smoke is for correctness only. Do not change algorithms or parameters because of smoke accuracy.

## 16. Freeze boundary

From P1 onward, absent a reproducible correctness bug, do not change:

```text
same SHOT source F/B/C
same R50/R101 model substrate
same seed-2026 stream / batch sizes / one pass / singleton semantics
same PU/FO and metrics
same optimizer/scheduler substrate
same full / FC / Conv matched scopes
netC frozen
controlled FC/Conv BN frozen

NCTTA feature = post-netB 256-d
classifier reference = effective frozen netC.fc.weight
official top-k + q_dist + q_prob + hybrid target semantics
official entropy + NC loss semantics
official entropy filtering + reweighting semantics
NC call = InfoNCE / cosine / tau_align=1.0
no TTAB oracle selection
no Fisher
no stochastic restore

P1 dense only
future sparse/LBI uses corrected refined LBI only
Conv sparse mainline uses out_channel grouping only
filter_connection remains closed
```

Any semantic change requires a new protocol/implementation revision and an explicit reason.
