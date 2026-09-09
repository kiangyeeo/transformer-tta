# OTTA NCTTA Baseline Protocol

**Protocol revision:** `OTTA_NCTTA_BASELINE_PROTOCOL_20260906_v2`  
**Implementation revision:** `nctta_otta_baseline_20260906_v3`  
**Parent:** `OTTA_NCTTA_BASELINE_PROTOCOL_20260906_v1`  
**Status:** Frozen for controlled dense correctness and the native-scope reference; formal dataset-specific NCTTA hyperparameters remain unresolved.

## 1. Purpose and source of truth

This revision preserves the three controlled P1 variants and adds exactly one official-scope reference: `nctta_native_norm`. Priority is: current refined SHOT substrate for F/B/C, stream, seed, batch boundaries, PU/FO and provenance; official `Cevaaa/NCTTA` for native update scope/BN/optimizer; the audited local `nctta_otta/objective.py` for the objective port.

Audited upstream commit: `b4d442472a36af6b3f4d6e75f5138ca97d7d8eec`.

No official checkpoint, source retraining, backbone replacement, classifier replacement, source-feature teacher, or new stabilization loss is permitted.

## 2. Common substrate

```text
source revision = nips2026_shot_otta_uda_source_v1
same source_F / source_B / source_C
same netF -> netB -> netC
Office-31 = ResNet-50, batch size 64
VisDA-C = ResNet-101, batch size 256
seed = 2026
one fixed-order target pass
same outer batch boundaries
actual singleton outer batch is skipped
same PU and FO data protocols
netC frozen
```

## 3. Variant roles

| Variant | Role | Trainable scope |
|---|---|---|
| `nctta_native_norm` | official/native update-scope reference | official normalization selector |
| `nctta_full_dense` | NCTTA objective + controlled full scope | all `netF + netB` parameters |
| `nctta_fc_module_dense` | NCTTA objective + controlled FC scope | `netB.bottleneck.weight/bias` |
| `nctta_conv_module_dense` | NCTTA objective + controlled Conv scope | nine `netF.layer4.*.conv*.weight` tensors |

No `native_full`, `native_fc`, or `native_conv` variants exist. The native reference is excluded from FC/Conv matched sparse comparisons.

## 4. Native selector and BN semantics

The official `_initialize_model` first calls `model.train()` and freezes the complete model. It then enables affine parameters for concrete `BatchNorm2d`, `LayerNorm`, and `GroupNorm` modules. Only BN is used in the paper. For `BatchNorm2d` it sets:

```text
requires_grad = true for affine weight/bias
track_running_stats = false
running_mean = None
running_var = None
```

The current SHOT F/B/C architecture contains ResNet `BatchNorm2d` modules in `netF`, one `BatchNorm1d` in `netB`, and no LayerNorm/GroupNorm. Therefore the actual native scope is exactly:

```text
netF BatchNorm2d weight/bias: trainable, current-batch statistics
netB BatchNorm1d weight/bias and running buffers: frozen/source behavior
all Conv: frozen
netB bottleneck Linear: frozen
netC: frozen
```

`netF` is in train mode during adaptation. `netB` and `netC` are in eval mode so the project-specific `netB.BatchNorm1d` is not silently added to official adaptation. The artifact stores the full trainable-parameter manifest with name, module type, shape, tensor count, and scalar count.

## 5. Objective

All four variants call the same audited `nctta_otta/objective.py`; native does not reimplement a loss. The current feature and classifier reference remain:

$$h=\mathrm{netB}(\mathrm{netF}(x)),\qquad W_C=\mathrm{netC.fc.weight.detach()}.$$

The port preserves top-k probability candidates, standardized cosine-distance `q_dist`, probability `q_prob`, hybrid target, InfoNCE NC loss, `tau_align=1.0`, entropy, strict entropy filtering, detached entropy/FCA-distance reweighting, and the selected-sample weighted mean. Target labels, future samples, and frozen source features never enter adaptation.

Correctness-smoke objective defaults remain unchanged from v1: `thre_ent=margin_ent=0.4*log(1000)`, `reweight_ent=1`, `nu=5`, `eta=1`, `scale=5`, `top_k=10`, and `mix_prob_weight=0.3`.

## 6. Optimizer provenance

Official NCTTA obtains its optimizer from TTAB `define_optimizer`. With the official NCTTA/default configuration used by this reference:

```text
optimizer = SGD
lr = 0.001
momentum = 0.9
dampening = 0
weight_decay = 0
nesterov = true
scheduler = none
```

`nctta_native_norm` uses exactly this optimizer and a single uniform parameter group. It performs no SHOT polynomial scheduler step. The three controlled variants retain v1 common SHOT optimizer/scheduler semantics unchanged (Office base LR 0.01, VisDA base LR 0.001, netF multiplier 0.1, netB multiplier 1, momentum 0.9, weight decay 0.001, Nesterov, polynomial global schedule).

Every valid outer batch performs exactly one objective call and one optimizer step.

## 7. PU and FO

PU occurs after adaptation on the same current batch/reference view, under `torch.no_grad()`, with no optimizer step and no parameter or persistent-state write. Native `BatchNorm2d` continues to use current-batch statistics. Any BN buffers are snapshotted/restored defensively.

FO freezes all parameters and performs no optimizer/state writeback. Modules are switched to eval mode, but PyTorch `BatchNorm2d` still uses evaluation-batch statistics when `running_mean` and `running_var` are `None`. The fixed FO order and batch protocol are retained, and artifacts must record:

```text
native_norm_fo_uses_batch_statistics = true
```

The source-running-stat behavior must not be restored for native `BatchNorm2d`.

## 8. Contracts and provenance

Required passing checks are:

1. shared SHOT source loader and deterministic pre-configuration logits;
2. identical target order, batch boundaries, singleton policy, and no label/future leakage;
3. exact reuse of the audited objective and 256-d post-netB feature;
4. byte-identical `netC`;
5. native selector matches concrete official module types and excludes `BatchNorm1d`;
6. all Conv, bottleneck Linear, and classifier tensors remain unchanged for native;
7. native BN has no running mean/variance and PU/FO do not write parameter, optimizer, or persistent state;
8. before/after outer-batch snapshots report exact changed tensor names and scalar count;
9. exactly one adaptation optimizer step per valid outer batch and no native scheduler step;
10. protocol, implementation, source revision, optimizer provenance, and artifact identity agree;
11. all three controlled NCTTA variants and existing SHOT/IST contracts regress without semantic changes.

## 9. Freeze boundary

This revision authorizes only correctness contracts and the five-valid-batch Office D→A native smoke. It does not authorize formal runs, accuracy-driven hyperparameter changes, Random/Magnitude/Saliency/LBI, Conv grouping, parameter search, source anchoring, or changes under `shot_otta/**` or to IST scientific semantics.

Dataset-specific formal NCTTA hyperparameters and later sparse-search protocol remain unresolved. Smoke accuracy is diagnostic only.
