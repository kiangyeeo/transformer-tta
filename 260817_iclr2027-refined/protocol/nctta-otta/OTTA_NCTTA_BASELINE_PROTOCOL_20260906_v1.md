# OTTA NCTTA Baseline Protocol

**Protocol revision:** `OTTA_NCTTA_BASELINE_PROTOCOL_20260906_v1`  
**Implementation revision:** `nctta_otta_p1_dense_baseline_20260906_v2`  
**Status:** Frozen for controlled dense-baseline correctness; formal dataset-specific NCTTA objective hyperparameters remain unresolved.

## 1. Purpose

This protocol defines NCTTA dense baselines on the same controlled OTTA substrate used by the refined SHOT/IST experiments. The scientific variable is the **NCTTA adaptation objective**; source checkpoints, F/B/C architecture, online stream, optimizer/scheduler substrate, trainable candidate scopes, controlled BN semantics, PU/FO evaluation, and provenance remain matched to the common substrate.

The official NCTTA repository is the source of truth for NCTTA-specific objective semantics. This controlled branch is not an exact reproduction of the official norm-only update scope.

## 2. Common substrate

Frozen invariants:

```text
source revision = nips2026_shot_otta_uda_source_v1
same source F/B/C checkpoints as SHOT/IST
same netF -> netB -> netC architecture
Office-31: ResNet-50, BS=64
VisDA-C: ResNet-101, BS=256
seed=2026
one target pass
same fixed outer stream / batch boundaries / singleton skip
same PU / FO / metrics
netC frozen
same SHOT optimizer / global outer-batch LR schedule
same FC candidate and Conv layer4 candidate
controlled FC/Conv BN frozen
same artifact / provenance discipline
```

`shot_otta/**` remains read-only.

## 3. NCTTA objective

For each current outer batch, explicitly compute

$$
f_i=\mathrm{netF}(x_i),\qquad
h_i=\mathrm{netB}(f_i),\qquad
z_i=\mathrm{netC}(h_i).
$$

The NCTTA alignment feature is the **current post-netB 256-d classifier-input feature** $h_i$. The classifier reference is the effective frozen weight

$$
W_C=\mathrm{netC.fc.weight.detach()}.
$$

Do not use the 2048-d `netF.avgpool` feature in the F/B/C model, and do not replace the current feature with a frozen source feature in the main NCTTA branch.

The objective faithfully follows official `Cevaaa/NCTTA` code:

1. row-normalize feature and classifier weights;
2. obtain current probabilities from current logits;
3. choose top-$k$ candidate classes from current predicted probabilities;
4. build standardized-distance target $q_{dist}$;
5. build probability target $q_{prob}$;
6. mix them using `mix_prob_weight`;
7. compute InfoNCE-style NC loss with cosine geometry and `tau_align=1.0`;
8. compute per-sample softmax entropy;
9. apply strict entropy filter `entropy < thre_ent`;
10. compute detached entropy and predicted-class FCA-distance coefficients;
11. optimize the selected-sample weighted mean of `entropy + scale * NC`.

No target label or future target data enters adaptation.

## 4. Deliberate difference from native NCTTA

Official NCTTA freezes the model and primarily updates normalization affine parameters (the official implementation enables BatchNorm2d/LayerNorm/GroupNorm and comments that only BN is used in the paper). Our controlled dense variants deliberately use SHOT-matched update scopes to isolate objective effects:

| Variant | Persistent trainable scope | BN semantics | netC |
|---|---|---|---|
| `nctta_full_dense` | all `netF + netB` parameters | same as SHOT full-dense | frozen |
| `nctta_fc_module_dense` | `netB.bottleneck.weight/bias` | all BN frozen | frozen |
| `nctta_conv_module_dense` | full `netF.layer4` 9 Conv weights | all BN frozen | frozen |

Therefore these are **controlled NCTTA-objective instantiations**, not exact native NCTTA reproductions. A native norm-only NCTTA reference is intentionally left for the next baseline revision.

## 5. Optimizer and update timing

Use the common SHOT/IST optimizer substrate:

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

One valid incoming outer batch performs exactly one NCTTA objective evaluation, one scheduler update, and one optimizer step. Historical singleton batches of actual size 1 are skipped before adaptation/PU to match SHOT trajectory semantics.

PU and FO are read-only.

## 6. P1 NCTTA method parameters

For correctness smoke only, record the official repository defaults:

```text
thre_ent        = 0.4 * log(1000)
margin_ent      = 0.4 * log(1000)
reweight_ent    = 1.0
nu              = 5.0
eta             = 1.0
scale           = 5.0
top_k           = 10
mix_prob_weight = 0.3
NC type         = infonce
metric          = cos
tau_align       = 1.0
margin          = 0.2
```

These values are **not yet formal Office/VisDA hyperparameters**. `formal_protocol=true` must fail until dataset-specific NCTTA objective hyperparameters are frozen. Smoke accuracy must not be used to silently change them.

If entropy filtering selects zero samples or the objective/gradient becomes non-finite, fail loudly with diagnostics; do not invent a fallback.

## 7. State and provenance contracts

Required checks:

1. exact same source F/B/C and initialization logits as SHOT;
2. same stream, batch boundaries, singleton semantics;
3. current post-netB feature dimension equals effective classifier dimension (=256);
4. `netC` byte-identical throughout;
5. FC dense changes exactly the two bottleneck Linear tensors and no BN buffers;
6. Conv dense changes exactly the nine layer4 Conv weights and no BN buffers;
7. full dense changes only SHOT-declared `netF + netB` state;
8. PU and FO are read-only;
9. objective numerics/gradients match an independent direct implementation of official NCTTA code;
10. protocol/implementation/source revisions are stored in scientific identity and artifacts;
11. the protocol revision must correspond to an actual protocol file;
12. existing SHOT/IST regressions remain passing.

## 8. Freeze boundary and next step

This revision freezes P1 dense correctness only. It does **not** freeze:

- native norm-only NCTTA reference implementation;
- Office/VisDA formal NCTTA objective hyperparameters;
- Random/Magnitude/Saliency;
- LBI parameter grids;
- sparse/LBI formal launch.

Next: add the native norm-only reference, then write/freeze the NCTTA sparse/LBI protocol. Do not alter the current-feature NCTTA objective to use source/frozen features as a stability fix; that would define a different method.
