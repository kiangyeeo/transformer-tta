# Codex Prompt

Project:
`/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined`

Conda:
`SHOT_TTA`

Normative files:
```text
protocol/shot-otta_conv/OTTA_CONV_LBI_PROTOCOL_20260826_v1.md
protocol/shot-otta_conv/OTTA_CONV_BASELINE_FORMAL_20260827_v1.md
protocol/shot-otta_conv/OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v1.md
protocol/shot-otta_conv/OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2.md
```

Implementation source of truth:
```text
iclr2027_refined_conv_20260826_v1
```

Global fixed scientific settings:
```text
seed = 2026
kappa = 1
omega = 0.00625
stage2_lr = 0.005
stage1_max_steps = 3000
support_threshold = 1e-4
stage2_steps = 1
strict group budget + rollback
no LBI top-K trimming
controlled Conv BN frozen
```

Global execution rules:
- Use only GPUs `0,1,2,3` on this machine.
- One experiment process per GPU:
  `max-workers=4`, `workers-per-gpu=1`.
- All Python / runner commands inside `.sh` must use:
  `conda run --no-capture-output -n SHOT_TTA ...`
- Do not require `conda activate`.
- Do not modify algorithm code, frozen protocol files, FC code/results, source checkpoints, baselines, or correctness tests.
- Do not run VisDA or Conv baselines.
- PU/FO may be recorded but must not change Stage-1 anchor selection.
- Tuning/diagnostic runtime is not final-formal efficiency; do not label it `runtime_comparable=true`.
- Reuse the current repository runner / plan conventions. Do not invent a second execution framework.
- `MODE=PREPARE` must not launch training and must not probe GPUs.
- `MODE=FINALIZE` must not launch, retry, rerun, or change the grid.


# Assignment

Phase:
```text
R1 — Office-31 out-channel six-transfer Stage-1 validation
```

Important:
- D→A was already executed in R0.
- D→A must not be rerun.
- Only the transfers in this prompt are new runs.

Frozen R0 anchors:
```text
rho_G=.0005, K_G=4:
  primary A1 alpha=.05 nu=.50
  backup  A0 alpha=.10 nu=.50

rho_G=.001, K_G=9:
  primary A4 alpha=.10 nu=1.00
  backup  A1 alpha=.05 nu=.50

rho_G=.002, K_G=18:
  primary A4 alpha=.10 nu=1.00
  backup  A1 alpha=.05 nu=.50
```

Run exactly these 12 new rows:

| Row | Transfer | source | target | rho_G | K_G | Anchor | alpha | nu |
|---:|---|---:|---:|---:|---:|---|---:|---:|
| 1 | W→A | 2 | 0 | .0005 | 4 | A1 | 0.05 | 0.5 |
| 2 | W→A | 2 | 0 | .0005 | 4 | A0 | 0.1 | 0.5 |
| 3 | W→A | 2 | 0 | .001 | 9 | A4 | 0.1 | 1 |
| 4 | W→A | 2 | 0 | .001 | 9 | A1 | 0.05 | 0.5 |
| 5 | W→A | 2 | 0 | .002 | 18 | A4 | 0.1 | 1 |
| 6 | W→A | 2 | 0 | .002 | 18 | A1 | 0.05 | 0.5 |
| 7 | W→D | 2 | 1 | .0005 | 4 | A1 | 0.05 | 0.5 |
| 8 | W→D | 2 | 1 | .0005 | 4 | A0 | 0.1 | 0.5 |
| 9 | W→D | 2 | 1 | .001 | 9 | A4 | 0.1 | 1 |
| 10 | W→D | 2 | 1 | .001 | 9 | A1 | 0.05 | 0.5 |
| 11 | W→D | 2 | 1 | .002 | 18 | A4 | 0.1 | 1 |
| 12 | W→D | 2 | 1 | .002 | 18 | A1 | 0.05 | 0.5 |

Exact method:
```text
variant = conv_out_lbi
group_mode = out_channel
```

Branch root:
```text
experiment_logs/office_conv_lbi_r1_out_stage1_validation_seed2026_20260827
```

Node artifacts:
```text
plan:
experiment_logs/conv_lbi_32gpu_burst_20260827/plans/node7_out_r1.jsonl

launcher:
tools/run_conv_lbi_32gpu_node7_out_r1.sh

prepare record:
experiment_logs/conv_lbi_32gpu_burst_20260827/node_records/node7_out_r1_PREPARE.md

finalize record:
experiment_logs/conv_lbi_32gpu_burst_20260827/node_records/node7_out_r1_FINALIZE.md
```

# MODE=PREPARE

1. Verify all normative files and implementation revision.
2. Verify every Office source/target mapping from repository code/config; do not guess.
3. Build exactly 12 scientific rows matching the table.
4. Verify every row is `conv_out_lbi` / `out_channel`.
5. Verify each listed transfer has exactly 6 rows:
   `3 budgets × primary/backup`.
6. Verify exact budget mapping:
   ```text
   .0005 -> K_G=4
   .001  -> K_G=9
   .002  -> K_G=18
   ```
7. Verify exact anchor mapping shown above.
8. Verify fixed:
   `kappa=1`, `omega=.00625`, `stage2_lr=.005`,
   `stage1_max_steps=3000`, `support_threshold=1e-4`,
   `stage2_steps=1`.
9. Verify:
   - exactly 12 rows;
   - 12 unique experiment keys;
   - 12 unique scientific SHA values;
   - 12 unique output roots;
   - no D→A;
   - no filter row;
   - no baseline;
   - no VisDA;
   - no `.005`.
10. Create:
   ```text
   experiment_logs/conv_lbi_32gpu_burst_20260827/plans/node7_out_r1.jsonl
   tools/run_conv_lbi_32gpu_node7_out_r1.sh
   ```
11. Launcher:
   ```text
   GPUs=0,1,2,3
   max-workers=4
   workers-per-gpu=1
   ```
   Queue the remaining rows locally until all rows finish.
12. Static checks only:
   plan validator/equivalent, uniqueness checks, `bash -n`, `git diff --check`.
13. Do not launch training and do not probe GPUs.
14. Write:
   ```text
   experiment_logs/conv_lbi_32gpu_burst_20260827/node_records/node7_out_r1_PREPARE.md
   ```
15. Print only:
   ```bash
   cd /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined
   bash tools/run_conv_lbi_32gpu_node7_out_r1.sh
   ```
16. Stop.

# MODE=FINALIZE

Aggregate existing artifacts only.

Do not launch, retry, rerun, or change anchors.

1. Audit exactly 12 planned rows.
2. Verify:
   completion, summary existence, plan/hash match, output-root uniqueness,
   finite metrics, budget safety, max-step failures, mask/support diagnostics.
3. For every row report:
   ```text
   transfer
   budget / K_G
   anchor / alpha / nu
   scientific_valid
   selected_group_count min/mean/max
   utilization min/mean/p05
   fraction below .95
   selected_scalar_count
   realized_scalar_ratio
   Stage-1 mean/max steps
   Stage-1 max-step failures
   Stage-1 runtime
   Stage-2 runtime
   online runtime
   peak allocated/reserved GPU memory
   PU
   FO
   ```
4. Write:
   ```text
   experiment_logs/conv_lbi_32gpu_burst_20260827/node_records/node7_out_r1_FINALIZE.md
   ```

### Global out-channel R1 aggregation responsibility

Node 7 is the **global R1 FINALIZE owner** for out-channel.

After auditing Node 7, also read:
```text
experiment_logs/conv_lbi_32gpu_burst_20260827/node_records/node6_out_r1_FINALIZE.md
experiment_logs/office_conv_lbi_r0_reachability_seed2026_20260827/phase_records/R0/FINALIZE.md
experiment_logs/office_conv_lbi_r0_reachability_seed2026_20260827/selected_configs/R0_STAGE1_PRIMARY_BACKUP.json
```

The complete six-transfer evidence is:
```text
existing R0 D→A rows
+ Node6 A→D/A→W/D→W rows
+ Node7 W→A/W→D rows
```

If Node6 is not complete yet:
- do not launch/retry;
- write `WAITING_FOR_NODE_6: YES`;
- stop without freezing R1.

For each budget, compare exactly the same two R0 anchors across all six transfers.

Scientific-valid:
- every transfer completed and finite;
- selected_group_count <= K_G on every batch;
- zero Stage-1 max-step failures;
- no support/mask/budget violation.

Out-channel utilization-eligible on every transfer:
```text
mean(u_t) >= .90
p05(u_t)  >= .75
```

Rank the two candidates per budget exactly by:
1. scientific-valid on all six transfers;
2. utilization-eligible on all six transfers;
3. worst-transfer p05 utilization descending;
4. six-transfer mean utilization descending;
5. worst Stage-1 max steps ascending;
6. six-transfer mean Stage-1 steps ascending;
7. mean Stage-1 runtime ascending.

Accuracy does not participate.

Freeze exactly one Stage-1 anchor per budget and write:
```text
experiment_logs/office_conv_lbi_r1_out_stage1_validation_seed2026_20260827/
  selected_configs/OFFICE_OUT_STAGE1_FROZEN.json
  phase_records/R1/FINALIZE.md
```

If neither candidate survives for a budget, mark that budget BLOCKED and do not invent another anchor.


End with:
```text
NODE7_FINALIZE_COMPLETE: YES/NO
PLANNED: 12
COMPLETED: X
FAILED: X
MISSING: X
```
