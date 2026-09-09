# Codex Prompt — R0 Office D→A Conv-LBI Reachability (PREPARE / FINALIZE)

Project:
`/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined`

Conda:
`SHOT_TTA`

Normative files:

```text
protocol/shot-otta_conv/OTTA_CONV_LBI_PROTOCOL_20260826_v1.md
protocol/shot-otta_conv/OTTA_CONV_BASELINE_FORMAL_20260827_v1.md
protocol/shot-otta_conv/OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v1.md
configs/otta_conv_lbi_protocol_20260826_v1.yaml
```

Use current refined code as the only implementation source of truth.

This phase is:

```text
R0 — Office-31 D→A Conv-LBI Stage-1 reachability
```

Search root:

```text
experiment_logs/office_conv_lbi_r0_reachability_seed2026_20260827
```

## Hard boundaries

- Do NOT modify algorithm code, frozen protocols, baselines, FC code/results, checkpoints, or correctness tests.
- Do NOT run Conv baselines.
- Do NOT run VisDA.
- Do NOT tune `kappa`, `omega`, `stage2_lr`, support threshold, Stage-1 cap, or budget.
- Formal seed = 2026.
- Office D→A only. Read the repository's formal Office mapping and verify the source/target identifiers; do not guess.
- Formal budgets only: `.0005/.001/.002`.
- Group modes only: `out_channel` and `filter_connection`.
- `kappa=1`.
- `omega=.00625`.
- `stage2_lr=.005`.
- `stage1_max_steps=3000`.
- `support_threshold=1e-4`.
- Stage-2 steps = 1.
- PU / FO are report-only in R0 and must NOT affect Stage-1 anchor selection.
- R0 must not expand the grid after results are seen.

## Exact R0 Stage-1 anchors

For every `(group_mode, budget)` run exactly:

| Anchor | alpha | nu |
|---|---:|---:|
| A0 | .10 | .50 |
| A1 | .05 | .50 |
| A2 | .20 | .50 |
| A3 | .10 | .25 |
| A4 | .10 | 1.00 |

Total:

```text
2 group modes × 3 budgets × 5 anchors = 30 scientific conditions
```

No more, no fewer.

Formal group budgets must resolve to:

```text
out_channel:
.0005 -> K_G=4
.001  -> K_G=9
.002  -> K_G=18

filter_connection:
.0005 -> K_G=3276
.001  -> K_G=6553
.002  -> K_G=13107
```

## GPU / launcher policy

Use exactly one 4-GPU machine:

```text
GPUs=0,1,2,3
max-workers=4
workers-per-gpu=1
```

One experiment process per GPU.

R0 is a tuning / diagnostic phase, so do NOT label its runtime as final formal efficiency.
Still record Stage-1 / Stage-2 / online runtime and memory for diagnostics.

Use one foreground, non-interactive launcher with two sequential waves:

```text
Wave 0: all 15 out_channel conditions
Wave 1: all 15 filter_connection conditions
```

Wave 1 starts only after Wave 0 finishes.

Within each wave use the current multi-GPU runner with:

```text
--gpus 0,1,2,3
--max-workers 4
--workers-per-gpu 1
--resume
--resume-partial-runs
```

All Python / runner commands inside `.sh` must use:

```bash
conda run --no-capture-output -n SHOT_TTA ...
```

Do not ask the user to `conda activate`.

Launcher path:

```text
tools/run_office_conv_lbi_r0_da_reachability_noninteractive.sh
```

---

# MODE=PREPARE

1. Verify all three normative protocol files and the frozen Conv implementation revision.
2. Verify Office D→A formal source/target identifiers from the repository.
3. Create matrix + full plan with exactly 30 scientific conditions.
4. Create explicit Wave 0 / Wave 1 plan files:
   - Wave 0 = 15 out-channel rows.
   - Wave 1 = 15 filter-connection rows.
5. Propagate the search protocol revision as provenance without changing the repository's canonical scientific identity semantics.
6. Verify every row has the exact:
   - dataset / transfer / seed;
   - group mode;
   - budget and K_G;
   - anchor alpha / nu;
   - `kappa=1`;
   - `omega=.00625`;
   - `stage2_lr=.005`;
   - Stage-1 cap / support threshold / Stage-2 steps inherited from frozen protocol.
7. Verify:
   - total conditions = 30;
   - unique experiment keys = 30;
   - unique experiment-config SHA = 30;
   - unique output roots = 30;
   - no `.005`;
   - no baseline variants;
   - no VisDA;
   - no non-R0 LBI rows.
8. Create:
   `tools/run_office_conv_lbi_r0_da_reachability_noninteractive.sh`
9. Launcher must fail-fast before training if plan validation fails.
10. Static checks only:
    - plan uniqueness;
    - wave counts 15 / 15;
    - `bash -n`;
    - `git diff --check`.
11. Do NOT launch training and do NOT probe GPUs.
12. Write:

```text
experiment_logs/office_conv_lbi_r0_reachability_seed2026_20260827/phase_records/R0/PREPARE.md
```

Record exact plan count, waves, launcher, revisions, and static-check results.

13. Print only:

```bash
cd /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined
bash tools/run_office_conv_lbi_r0_da_reachability_noninteractive.sh
```

Then stop.

---

# MODE=FINALIZE

Aggregate only existing R0 artifacts.

Do NOT launch, retry, rerun, or add rows.

## Scientific validity

A row is scientific-valid only if:

```text
completed
finite / error-free
selected_group_count <= K_G on every online batch
zero Stage-1 max-step failures
no support / mask / budget contract violation
```

Exact `K_G` is NOT required for LBI.

Define per-batch utilization:

```text
u_t = selected_group_count_t / K_G
```

## Utilization eligibility

For `out_channel`:

```text
mean(u_t) >= .90
p05(u_t)  >= .75
```

For `filter_connection`:

```text
mean(u_t) >= .90
p05(u_t)  >= .90
```

For every row report at least:

```text
group_mode
budget
K_G
anchor
alpha
nu
scientific-valid
selected group min / mean / max
utilization min / mean / p05
fraction batches below .95
selected scalar count
realized scalar ratio
Stage-1 mean / max steps
Stage-1 max-step failures
Stage-1 runtime
Stage-2 runtime
online runtime
peak allocated / reserved GPU memory
PU
FO
```

PU / FO are report-only. Do not rank by accuracy.

## R0 ranking

For each of the six `(group_mode, budget)` cells:

1. scientific-valid;
2. utilization-eligible;
3. p05 utilization descending;
4. mean utilization descending;
5. Stage-1 max steps ascending;
6. Stage-1 mean steps ascending;
7. Stage-1 runtime ascending.

Freeze:

```text
primary Stage-1 anchor
backup Stage-1 anchor
```

for R1.

If fewer than two are eligible, retain every eligible row.
If zero are eligible, mark that cell `BLOCKED` and stop for that cell.

Do not increase `stage1_max_steps`.
Do not expand alpha / nu.
Do not use PU / FO to rescue or reorder candidates.

Write:

```text
experiment_logs/office_conv_lbi_r0_reachability_seed2026_20260827/phase_records/R0/FINALIZE.md
```

and:

```text
experiment_logs/office_conv_lbi_r0_reachability_seed2026_20260827/selected_configs/R0_STAGE1_PRIMARY_BACKUP.json
```

The JSON must contain the primary / backup anchor for all six cells, or an explicit `BLOCKED` status.

At the end print only:

```text
R0_COMPLETE: YES/NO
ELIGIBLE_CELLS: X/6
BLOCKED_CELLS: ...
NEXT_PHASE: R1 or STOP
```

Then stop.
