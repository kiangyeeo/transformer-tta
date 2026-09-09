# OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2

**Status:** Frozen amendment after R0 FINALIZE  
**Scope:** Conv-LBI search continuation after `filter_connection` R0 blockage  
**Inherits:** `OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v1`  
**Scientific implementation revision:** `iclr2027_refined_conv_20260826_v1`

This v2 does **not** change the Conv-LBI algorithm. It only authorizes one
predeclared filter-connection Stage-1 rescue phase and allows the already
eligible out-channel branch to continue in parallel.

## 1. R0 result incorporated into v2

R0 Office D→A completed 30/30 conditions.

Frozen out-channel R0 choices:

| Budget | Primary | Backup |
|---:|---|---|
| .0005 | A1: `alpha=.05, nu=.50` | A0: `alpha=.10, nu=.50` |
| .001 | A4: `alpha=.10, nu=1.00` | A1: `alpha=.05, nu=.50` |
| .002 | A4: `alpha=.10, nu=1.00` | A1: `alpha=.05, nu=.50` |

All three filter-connection cells were blocked: 0/15 filter rows were
scientific-valid because Stage-1 hit the 3000-step cap on almost every batch.

The strongest prior filter direction was the more aggressive Stage-1 region
(e.g. `alpha=.20, nu=.50`), which sometimes approached the target group count
but still failed reachability. Therefore v2 expands only the Stage-1 dynamics
for the blocked filter branch.

## 2. Frozen rules unchanged

Keep:

```text
formal budgets = {.0005,.001,.002}
kappa = 1
omega = .00625 during Stage-1 diagnostics/validation
stage2_lr = .005 during Stage-1 diagnostics/validation
support_threshold = 1e-4
stage1_max_steps = 3000
stage2_steps = 1
strict group budget + rollback
no LBI top-K trim
PU/FO report-only for Stage-1 phases
```

Do not increase the Stage-1 cap and do not tune `kappa`.

## 3. Branch O — out-channel R1 continues

Use the frozen R0 primary/backup anchors above.

Validate them over all six Office transfers. The existing D→A R0 artifacts are
scientifically identical and must be **reused**, not rerun.

Therefore new R1-out executions are:

$$
5\ \text{new transfers}\times 3\ \text{budgets}\times 2\ \text{anchors}=30.
$$

New transfers:

```text
A→D
A→W
D→W
W→A
W→D
```

R1-out eligibility and ranking remain exactly as in v1:

1. scientific-valid on all six transfers;
2. utilization-eligible on all six transfers;
3. worst-transfer p05 utilization descending;
4. six-transfer mean utilization descending;
5. worst Stage-1 max steps ascending;
6. six-transfer mean Stage-1 steps ascending;
7. mean Stage-1 runtime ascending.

Accuracy does not select the Stage-1 anchor.

Freeze one out-channel Stage-1 anchor per budget.

## 4. Branch F — filter-connection R0F rescue

R0F uses only Office D→A and all three formal budgets.

For every budget run exactly these eight anchors:

| Anchor | alpha | nu |
|---|---:|---:|
| F0 | .25 | .50 |
| F1 | .30 | .50 |
| F2 | .40 | .50 |
| F3 | .60 | .50 |
| F4 | .20 | .25 |
| F5 | .30 | .25 |
| F6 | .40 | .25 |
| F7 | .30 | .125 |

Total:

$$
3\ \text{budgets}\times8\ \text{anchors}=24
$$

scientific conditions.

No additional filter anchor may be added after seeing R0F results.

### R0F scientific validity

A row is scientific-valid only if:

```text
completed
finite / error-free
selected_group_count <= K_G on every online batch
zero Stage-1 max-step failures
no support / mask / budget contract violation
```

### R0F filter utilization gate

For each batch:

$$
u_t=\frac{\text{selected_group_count}_t}{K_G}.
$$

Eligible iff:

```text
mean(u_t) >= .90
p05(u_t)  >= .90
```

### R0F ranking

For each budget:

1. scientific-valid;
2. utilization-eligible;
3. p05 utilization descending;
4. mean utilization descending;
5. Stage-1 max steps ascending;
6. Stage-1 mean steps ascending;
7. Stage-1 runtime ascending.

Freeze primary + backup.

PU/FO do not participate.

If a budget still has zero eligible rows, mark it `BLOCKED_V2` and stop the
filter branch. Do not increase the cap, change penalty semantics, or add a
third rescue grid inside v2.

## 5. Parallel execution authorization

Branch O (R1-out) and Branch F (R0F-filter) are independent and may run
concurrently.

The recommended 32-GPU allocation is:

```text
8 machines × 4 GPUs

Nodes 0-5: filter R0F, 24 GPUs, one condition/GPU
Nodes 6-7: out-channel R1, 8 GPUs, queued 30 new conditions
```

No GPU oversubscription:

```text
1 experiment process / GPU
```

These are tuning/diagnostic runs; runtime is diagnostic, not final-formal
efficiency.

## 6. Next phase after this burst

If filter R0F succeeds:

```text
out branch    -> R2 omega × stage2_lr
filter branch -> R1 six-transfer Stage-1 validation
```

These two branches should again run in parallel.

If any filter budget remains `BLOCKED_V2`, stop that filter budget and create
a new scientific protocol only after diagnosing whether the issue is
hyperparameter reachability or the unweighted group formulation itself.

**End of v2 amendment.**
