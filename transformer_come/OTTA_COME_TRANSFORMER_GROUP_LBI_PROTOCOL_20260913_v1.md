# COME-Transformer Group-LBI implementation protocol — 2026-09-13 v1

Status: implementation-frozen; formal hyperparameter profiles remain blocked
until the separately declared search freezes six dataset-by-budget tuples.

This document adds `group_lbi` to the COME-Transformer implementation. It
inherits source checkpoints, the one-pass stream, preprocessing, PU/FO,
DeiT candidate tensors, 6,912 paired QK/VO/FFN groups, and floor budgets
3/6/13 from `OTTA_COME_TRANSFORMER_LBI_PROTOCOL_20260909_v1.md`. The stable
COME objective is inherited from the 2026-09-11 v2 baseline protocol. No SHOT
loss component enters this path.

## Frozen algorithm

Every online batch starts with local `Delta=Gamma=Z=0`. Stage 1 evaluates a
fresh stable log-domain COME current-logit objective at every candidate state:

```text
c_k       = (Delta_k - Gamma_k) / nu
Delta_k+1 = Delta_k - alpha*kappa*(grad_COME_k + c_k)
Z_k+1     = Z_k + alpha*c_k
Gamma_k+1 = kappa*group_soft_threshold(Z_k+1, prox_lambda=1)
```

Support is `||Gamma_g||_2 / sqrt(768) >= 1e-4`. Exact K is accepted, a count
below K continues, and overshoot restores the complete last-feasible
Delta/Gamma/Z/mask state. There is no top-K trim/fill or budget slack. Any
batch that executes 3,000 Stage-1 steps records a cap-hit diagnostic, retains
its last feasible support, and continues through Stage 2, same-batch PU, every
later online batch, and FO. The cap is local and does not invalidate the run.

Stage 2 initializes `base + mask*Delta`, creates one fresh local AdamW,
evaluates one fresh COME objective, and performs one strict masked step.
AdamW uses betas `(0.9, 0.999)`, eps `1e-8`, weight decay `0.01`, one step,
no scheduler, and no AMP. Only `base + omega*(refined-base)` persists.
Off-mask and off-scope values remain exact; there is no persistent host
optimizer, scheduler, teacher, or native EMA.

PU is a separate read-only post-writeback forward on the same online tensor;
FO is an independent read-only center-crop pass. Labels are used only for
metrics. A singleton online batch is a stream protocol failure. Only complete,
successful outer-batch boundaries are resumable.

## Identity and formal gate

```text
protocol_revision       = come_transformer_group_lbi_otta_20260913_v1
implementation_revision = come_transformer_sparse_lbi_20260913_v1
```

Checked-in alpha/kappa/nu/omega/Stage-2-LR values are provisional search
anchors. Normal resolution fails closed. `--allow-provisional-lbi` explicitly
opts into non-formal implementation/search/smoke execution; its resolved
profile records `formal_eligible=false`. Formal runs require `status: frozen`
for the selected dataset and budget.

Artifacts record the COME identity, LBI tuple, requested/realized support,
support hashes/distributions, Stage-1/2/omega call counts, runtime and memory,
collapse diagnostics, checkpoint/stream hashes, and final source-relative
support.
