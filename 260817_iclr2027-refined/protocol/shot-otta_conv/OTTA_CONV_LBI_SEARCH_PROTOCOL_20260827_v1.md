# OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v1

**Status:** Freeze before the first Conv-LBI search launch  
**Scope:** SHOT-OTTA + Conv Group-LBI search / selection / final formal evaluation  
**Scientific implementation revision:** `iclr2027_refined_conv_20260826_v1`  
**Protocol date:** 2026-08-27

This document defines **how Conv-LBI hyperparameters are searched and frozen**.
It does not change the Conv-LBI algorithm.

Normative upstream files:

```text
protocol/shot-otta_conv/OTTA_CONV_LBI_PROTOCOL_20260826_v1.md
protocol/shot-otta_conv/OTTA_CONV_BASELINE_FORMAL_20260827_v1.md
configs/otta_conv_lbi_protocol_20260826_v1.yaml
```

If this search protocol conflicts with the frozen Conv method protocol, the
frozen Conv method protocol wins.

---

## 1. Search target

The formal Conv budgets remain:

$$
\rho_G \in \{0.0005,\ 0.001,\ 0.002\}.
$$

The two structural groupings are tuned separately:

```text
out_channel
filter_connection
```

Final Conv-LBI tuning granularity is:

$$
(\text{dataset},\ \text{group mode},\ \rho_G).
$$

Therefore the study freezes exactly **12 final Conv-LBI tuples**:

```text
Office × out_channel        × 3 budgets
Office × filter_connection  × 3 budgets

VisDA  × out_channel        × 3 budgets
VisDA  × filter_connection  × 3 budgets
```

Each final tuple contains:

```text
alpha
kappa
nu
omega
stage2_lr
```

`kappa` is frozen to:

```text
kappa = 1
```

throughout this v1 search. It is not a tuning axis.

---

## 2. Frozen method semantics

The search must not change:

```text
Conv candidate layers
group partition
group prox
support_threshold = 1e-4
strict integer global group budget
strict rollback
no top-K trimming for LBI
masked-delta Stage-2 initialization
Stage-2 off-mask freezing/restoration
Stage-2 steps = 1
batch-local LBI restart
persistent omega writeback semantics
PU / FO semantics
BN frozen semantics
seed = 2026
```

The Stage-1 safety cap remains:

```text
stage1_max_steps = 3000
```

It is a safety limit, not a tuning variable.

If a candidate repeatedly reaches the cap, the candidate is unsuitable.
Do not increase the cap to rescue it.

LBI is allowed to finish with:

$$
|M_G| < K_G
$$

because strict rollback does not top-K trim the support.

Therefore **exact K is not required for LBI validity**.

---

## 3. Formal group budgets

### 3.1 Out-channel

$$
|\mathcal G_{\mathrm{out}}| = 9216
$$

| $\rho_G$ | $K_G$ |
|---:|---:|
| .0005 | 4 |
| .001 | 9 |
| .002 | 18 |

### 3.2 Filter-connection

$$
|\mathcal G_{\mathrm{filter}}| = 6,553,600
$$

| $\rho_G$ | $K_G$ |
|---:|---:|
| .0005 | 3,276 |
| .001 | 6,553 |
| .002 | 13,107 |

The pilot-only `.005` budget is not reopened during Conv-LBI tuning.

---

## 4. Scientific validity

A Conv-LBI run is **scientific-valid** only if all are true:

```text
completed
finite / error-free
selected_group_count <= K_G on every online batch
zero Stage-1 max-step failures
no support / mask / budget contract violation
```

PU is always report-only.

PU must not be used to choose hyperparameters.

---

## 5. Conv-specific support-utilization gate

Do not copy the FC `min utilization >= .90` rule literally.

For out-channel, the formal budgets are only 4 / 9 / 18 groups, so a
minimum-based 90% rule becomes excessively discrete.

Define per-batch group utilization:

$$
u_t =
\frac{\text{selected\_group\_count}_t}{K_G}.
$$

A candidate is **utilization-eligible** when:

### out_channel

```text
mean(u_t) >= 0.90
p05(u_t)  >= 0.75
```

### filter_connection

```text
mean(u_t) >= 0.90
p05(u_t)  >= 0.90
```

These are tuning gates only.

They do not redefine the LBI algorithm and do not require exact K.

Always report:

```text
min utilization
mean utilization
p05 utilization
fraction of batches below 0.95
selected scalar count
realized scalar ratio
```

---

## 6. General phase discipline

Every search phase has two explicit modes.

### MODE=PREPARE

PREPARE may:

```text
verify references
create matrix / plan
create launcher
perform static checks
write PREPARE.md
```

PREPARE must not:

```text
launch real training
probe GPUs
change the grid after seeing results
```

### MODE=FINALIZE

FINALIZE may:

```text
aggregate existing completed runs
audit scientific validity
audit utilization
rank eligible candidates
write FINALIZE.md
```

FINALIZE must not:

```text
launch
retry
rerun
silently add candidates
silently expand a boundary
```

If a phase has no eligible candidate and this protocol does not explicitly
permit another phase, stop and create a new protocol revision.

---

# PART A — OFFICE-31

## 7. R0 — D→A Stage-1 reachability pilot

### 7.1 Purpose

R0 is an engineering / dynamics diagnostic.

It answers:

> Can the current refined Group-LBI implementation reliably grow a useful
> support under each group mode and formal budget?

R0 is **not an accuracy-selection phase**.

Dataset / transfer:

```text
Office-31 D→A
source = DSLR
target = Amazon
seed = 2026
```

Run both:

```text
out_channel
filter_connection
```

and all three formal budgets.

### 7.2 Fixed R0 adaptation-strength settings

To reduce persistent drift while Stage-1 dynamics are diagnosed:

```text
kappa      = 1
omega      = 0.00625
stage2_lr  = 0.005
```

These are diagnostic R0 settings, not final tuned values.

### 7.3 Exact R0 Stage-1 anchor grid

For every `(group_mode, budget)`, run exactly these five anchors:

| Anchor | alpha | nu |
|---|---:|---:|
| A0 | .10 | .50 |
| A1 | .05 | .50 |
| A2 | .20 | .50 |
| A3 | .10 | .25 |
| A4 | .10 | 1.00 |

Total R0 scientific conditions:

$$
2 \times 3 \times 5 = 30.
$$

Do not expand this grid inside R0.

### 7.4 R0 ranking

Accuracy must not participate.

For each `(group_mode, budget)`:

1. scientific-valid;
2. utilization-eligible;
3. higher `p05 utilization`;
4. higher `mean utilization`;
5. lower Stage-1 max steps;
6. lower Stage-1 mean steps;
7. lower Stage-1 runtime.

Freeze:

```text
primary Stage-1 anchor
backup Stage-1 anchor
```

for the next phase.

If fewer than two are eligible, retain every eligible candidate.
If zero are eligible, stop for that `(group_mode, budget)`.

Do not rescue it by increasing `stage1_max_steps`.

---

## 8. R1 — Office six-transfer Stage-1 validation

For each `(group_mode, budget)`, run the R0 primary and backup anchors over all:

```text
A→D
A→W
D→A
D→W
W→A
W→D
```

Keep fixed:

```text
kappa     = 1
omega     = 0.00625
stage2_lr = 0.005
```

### R1 eligibility

The Stage-1 anchor must be scientific-valid on all six transfers and satisfy
the utilization gate on all six transfers.

Rank eligible anchors by:

1. worst-transfer `p05 utilization` descending;
2. six-transfer mean utilization descending;
3. worst Stage-1 max steps ascending;
4. six-transfer mean Stage-1 steps ascending;
5. mean Stage-1 runtime ascending.

Accuracy does not select the Stage-1 anchor.

Freeze exactly one Office Stage-1 anchor per:

```text
group_mode × budget
```

giving six Office Stage-1 anchors.

If neither R0 anchor is eligible across all six transfers, stop for that
setting. Do not invent another anchor inside R1.

---

## 9. R2 — Office omega × Stage-2 LR joint search

R2 tunes adaptation strength after Stage-1 dynamics are frozen.

For every frozen Office Stage-1 anchor, the initial joint grid is:

```text
omega     ∈ {.00625, .025, .10}
stage2_lr ∈ {.0025, .005, .010}
```

This is a:

$$
3 \times 3
$$

joint grid.

Run every candidate on all six Office transfers.

### 9.1 R2 eligibility

A candidate must:

```text
be scientific-valid on all six transfers
be utilization-eligible on all six transfers
```

### 9.2 R2 selection

Office primary tuning metric:

$$
\text{FO six-transfer macro accuracy}.
$$

Rank eligible candidates by:

1. FO six-transfer macro descending;
2. worst-transfer FO descending;
3. Stage-1 max steps ascending;
4. Stage-1 mean steps ascending;
5. online runtime ascending.

PU is report-only.

### 9.3 One-time predeclared upper-bound expansion

Only if the R2 winner lies on an **upper boundary**:

```text
omega = .10
or
stage2_lr = .010
```

one final local expansion is allowed:

```text
omega upper extension     = .30
stage2_lr upper extension = .020
```

Only add the missing boundary-neighbor rows needed around the current winner.

Do not reopen the full Cartesian grid.

After that one expansion FINALIZE, stop even if the winner remains on a
boundary.

No lower-bound expansion is permitted in Office v1.

---

## 10. Office tuple freeze

After R2 and any permitted one-time expansion, freeze exactly six tuples:

```text
out_channel       × {.0005,.001,.002}
filter_connection × {.0005,.001,.002}
```

Write:

```text
selected_configs/OFFICE_CONV_LBI_FINAL_TUPLES.json
selected_configs/OFFICE_CONV_LBI_FINAL_TUPLES.md
```

After freeze:

```text
no retuning
no new omega/LR rows
no new Stage-1 anchors
```

---

# PART B — VISDA-C

## 11. V0 — VisDA Stage-1 anchor transfer diagnostic

VisDA is tuned separately from Office.

For each `(group_mode, budget)`, start from the corresponding frozen Office
Stage-1 anchor:

```text
(alpha_office, nu_office)
```

Construct at most five predeclared VisDA Stage-1 candidates:

```text
(alpha_office,       nu_office)
(0.5*alpha_office,   nu_office)
(2.0*alpha_office,   nu_office)
(alpha_office,       0.5*nu_office)
(alpha_office,       2.0*nu_office)
```

Deduplicate identical rows.

Use:

```text
kappa     = 1
omega     = 0.003125
stage2_lr = 0.005
```

for this Stage-1 diagnostic.

Run the full VisDA target stream.

### V0 eligibility and ranking

Accuracy does not select Stage-1 dynamics.

Rank by:

1. scientific-valid;
2. utilization-eligible;
3. p05 utilization descending;
4. mean utilization descending;
5. Stage-1 max steps ascending;
6. Stage-1 mean steps ascending;
7. Stage-1 runtime ascending.

Freeze one VisDA Stage-1 anchor per `(group_mode, budget)`.

If no candidate is eligible, stop and revise the protocol.
Do not silently widen the Stage-1 grid.

---

## 12. V1 — VisDA omega regime diagnostic

For every frozen VisDA Stage-1 anchor:

```text
stage2_lr = .005
```

and test:

```text
omega ∈ {.0015625, .003125, .00625, .0125, .025}
```

Eligibility:

```text
scientific-valid
utilization-eligible
```

Among eligible omega values, keep the best **two** by:

1. FO mAcc descending;
2. FO worst-class accuracy descending;
3. FO class-wise std ascending;
4. FO overall accuracy descending;
5. Stage-1 max steps ascending;
6. Stage-1 mean steps ascending.

PU is report-only.

If only one omega is eligible, keep one.
If none are eligible, stop.

---

## 13. V2 — VisDA omega × Stage-2 LR final joint search

For each `(group_mode, budget)`:

```text
omega = the one or two V1 survivors
stage2_lr ∈ {.0025, .005, .010}
```

Reuse an identical V1 row rather than rerunning it.

Eligibility remains:

```text
scientific-valid
utilization-eligible
```

Final VisDA ranking:

1. FO mAcc descending;
2. FO worst-class accuracy descending;
3. FO class-wise std ascending;
4. FO overall accuracy descending;
5. Stage-1 max steps ascending;
6. Stage-1 mean steps ascending.

PU is report-only.

No boundary expansion is allowed after V2.

Freeze exactly six VisDA tuples.

Write:

```text
selected_configs/VISDA_CONV_LBI_FINAL_TUPLES.json
selected_configs/VISDA_CONV_LBI_FINAL_TUPLES.md
```

---

# PART C — FINAL FORMAL EVALUATION

## 14. Fresh formal Conv-LBI rerun

Tuning artifacts are not reused as final formal results.

After all 12 tuples are frozen, create a separate formal root and fresh-run:

### Office

$$
2\ \text{group modes}
\times
3\ \text{budgets}
\times
6\ \text{transfers}
=
36
$$

formal Conv-LBI conditions.

### VisDA

$$
2
\times
3
\times
1
=
6
$$

formal Conv-LBI conditions.

Total:

$$
42
$$

formal Conv-LBI conditions.

Formal final runs must not modify the frozen tuple after results are seen.

---

## 15. Formal execution conditions

For final efficiency comparison:

```text
1 experiment process / GPU
runtime_comparable = true
runtime_resume_used = false for a fresh uninterrupted run when possible
GPU = NVIDIA GeForce RTX 4090
PyTorch = 2.4.1
CUDA = 12.4
```

If an exact batch-boundary resume is operationally necessary, preserve the
frozen resume semantics and complete per-batch efficiency history, but record
the resume metadata explicitly.

Tuning / diagnostic runtime is not a substitute for final formal runtime.

---

## 16. Required reporting

Every LBI summary must record at least:

```text
dataset
transfer
seed
group_mode
rho_G
K_G

alpha
kappa
nu
omega
stage2_lr

selected_group_count
group utilization min / mean / p05
fraction below 0.95
selected_scalar_count
realized_scalar_ratio

Stage-1 mean / max steps
Stage-1 max-step failures
Stage-1 runtime
Stage-2 runtime
online runtime

PU
FO

peak allocated GPU memory
peak reserved GPU memory

protocol revision
implementation revision
source checkpoint revision
experiment key
experiment config SHA256
```

VisDA must additionally report:

```text
PU mAcc / overall
FO mAcc / overall
PU 12-class accuracy
FO 12-class accuracy
worst-class
class-wise std
```

---

## 17. Baseline comparison after final formal runs

For each group mode and budget, compare Conv-LBI against the already frozen:

```text
Random
Magnitude
Saliency
conv_module_dense
```

using the same formal budget.

Primary claims must distinguish:

```text
nominal group ratio
realized scalar ratio
```

Do not describe inactive Conv groups as physically pruned inference weights.
This study evaluates **group-structured sparse adaptation**, not inference
FLOP pruning.

---

## 18. Search closure rules

After a tuple is frozen:

```text
do not retune from final-formal accuracy
do not add a new budget
do not reopen .005
do not increase Stage-1 max steps
do not tune support_threshold
do not tune kappa in v1
do not merge out/filter into one shared tuple
```

Any scientifically meaningful change requires:

```text
OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2
```

rather than silently editing v1.

---

## 19. Planned phase order

```text
Conv formal baselines          DONE / FROZEN

R0  Office D→A reachability
R1  Office six-transfer Stage-1 validation
R2  Office omega × Stage-2 LR
    -> freeze 6 Office tuples

V0  VisDA Stage-1 diagnostic
V1  VisDA omega regime
V2  VisDA omega × Stage-2 LR
    -> freeze 6 VisDA tuples

Fresh formal Conv-LBI rerun
    -> Office 36
    -> VisDA 6

GLOBAL Conv-LBI FINALIZE
```

---

## 20. Immediate next action

The next executable phase is:

```text
R0 — Office-31 D→A Stage-1 reachability pilot
```

with exactly:

```text
2 group modes
× 3 budgets
× 5 Stage-1 anchors
= 30 scientific conditions
```

No other Conv-LBI search should be launched before R0 FINALIZE.

---

**End of protocol.**
