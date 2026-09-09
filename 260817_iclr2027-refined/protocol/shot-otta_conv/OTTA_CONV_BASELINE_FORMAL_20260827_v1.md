# OTTA_CONV_BASELINE_FORMAL_20260827_v1

## 1. Purpose

This document defines the frozen formal baseline evaluation protocol for Conv structured sparse adaptation under OTTA.

This protocol is built on:

- OTTA_CONV_LBI_PROTOCOL_20260826_v1
- iclr2027_refined_conv_20260826_v1

It only defines the formal baseline evaluation matrix and budget selection.
It does not modify the Conv-LBI algorithm.

---

## 2. Benchmark Scope

Datasets:

- Office-31
- VisDA-C

Protocol:

- SHOT-OTTA
- seed=2026
- PU/FO evaluation inherited from frozen FC/Conv OTTA protocol
- BN/evaluation behavior inherited from frozen implementation

---

## 3. Formal Conv Variants

Dense:

- conv_module_dense


Structured sparse baselines:

Out-channel grouping:

- conv_out_random
- conv_out_magnitude
- conv_out_saliency


Filter-connection grouping:

- conv_filter_random
- conv_filter_magnitude
- conv_filter_saliency


No Conv-LBI is included in this protocol.

Conv-LBI has separate tuning/final evaluation protocol.

---

## 4. Formal Budget

The formal budget grid is:

$$
\rho_G\in\{0.0005,0.001,0.002\}
$$

The budget is defined over global Conv group count:

$$
K_G=\lfloor\rho_G|\mathcal G|\rfloor
$$

where:

- out-channel:
  $$
  |\mathcal G|=9216
  $$

- filter-connection:
  $$
  |\mathcal G|=6553600
  $$


Corresponding budgets:

| Grouping | 0.0005 | 0.001 | 0.002 |
|---|---:|---:|---:|
| out-channel | 4 | 9 | 18 |
| filter-connection | 3276 | 6553 | 13107 |


The pilot-only budget:

$$
\rho_G=0.005
$$

is not included in the formal main evaluation.
It may be reported as sensitivity analysis if needed.

---

## 5. Baseline Selection Semantics

All sparse baselines use global group selection.

Random:

- global uniform exact-K group sampling
- seed=2026
- 3 deterministic child masks


Magnitude:

$$
s_g=\|W_g\|_2
$$

global top-K.


Saliency:

$$
s_g=\|W_g\odot\nabla W_g\|_2
$$

global dynamic top-K.

No:

- per-layer budget;
- minimum-one-per-layer;
- tolerance budget.

---

## 6. Evaluation Scope

Office-31:

All six transfers.

VisDA-C:

Full target stream evaluation.

Metrics:

- Office: overall accuracy
- VisDA-C: mAcc / overall / class-wise metrics

Efficiency:

Report:

- runtime
- peak allocated GPU memory
- peak reserved GPU memory

---

## 7. Reproducibility

All formal runs:

- seed=2026
- frozen Conv implementation revision:
  iclr2027_refined_conv_20260826_v1

No budget changes are allowed after formal evaluation starts.

---

## 8. Relation to Conv-LBI

This document freezes baseline comparison settings only.

Conv-LBI evaluation will use the same frozen benchmark and budget grid,
with additional LBI-specific tuning protocol.