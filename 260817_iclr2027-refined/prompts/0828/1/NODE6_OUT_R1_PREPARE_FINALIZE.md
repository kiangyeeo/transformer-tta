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

Run exactly these 18 new rows:

| Row | Transfer | source | target | rho_G | K_G | Anchor | alpha | nu |
|---:|---|---:|---:|---:|---:|---|---:|---:|
| 1 | A→D | 0 | 1 | .0005 | 4 | A1 | 0.05 | 0.5 |
| 2 | A→D | 0 | 1 | .0005 | 4 | A0 | 0.1 | 0.5 |
| 3 | A→D | 0 | 1 | .001 | 9 | A4 | 0.1 | 1 |
| 4 | A→D | 0 | 1 | .001 | 9 | A1 | 0.05 | 0.5 |
| 5 | A→D | 0 | 1 | .002 | 18 | A4 | 0.1 | 1 |
| 6 | A→D | 0 | 1 | .002 | 18 | A1 | 0.05 | 0.5 |
| 7 | A→W | 0 | 2 | .0005 | 4 | A1 | 0.05 | 0.5 |
| 8 | A→W | 0 | 2 | .0005 | 4 | A0 | 0.1 | 0.5 |
| 9 | A→W | 0 | 2 | .001 | 9 | A4 | 0.1 | 1 |
| 10 | A→W | 0 | 2 | .001 | 9 | A1 | 0.05 | 0.5 |
| 11 | A→W | 0 | 2 | .002 | 18 | A4 | 0.1 | 1 |
| 12 | A→W | 0 | 2 | .002 | 18 | A1 | 0.05 | 0.5 |
| 13 | D→W | 1 | 2 | .0005 | 4 | A1 | 0.05 | 0.5 |
| 14 | D→W | 1 | 2 | .0005 | 4 | A0 | 0.1 | 0.5 |
| 15 | D→W | 1 | 2 | .001 | 9 | A4 | 0.1 | 1 |
| 16 | D→W | 1 | 2 | .001 | 9 | A1 | 0.05 | 0.5 |
| 17 | D→W | 1 | 2 | .002 | 18 | A4 | 0.1 | 1 |
| 18 | D→W | 1 | 2 | .002 | 18 | A1 | 0.05 | 0.5 |

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
experiment_logs/conv_lbi_32gpu_burst_20260827/plans/node6_out_r1.jsonl

launcher:
tools/run_conv_lbi_32gpu_node6_out_r1.sh

prepare record:
experiment_logs/conv_lbi_32gpu_burst_20260827/node_records/node6_out_r1_PREPARE.md

finalize record:
experiment_logs/conv_lbi_32gpu_burst_20260827/node_records/node6_out_r1_FINALIZE.md
```

# MODE=PREPARE

1. Verify all normative files and implementation revision.
2. Verify every Office source/target mapping from repository code/config; do not guess.
3. Build exactly 18 scientific rows matching the table.
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
   - exactly 18 rows;
   - 18 unique experiment keys;
   - 18 unique scientific SHA values;
   - 18 unique output roots;
   - no D→A;
   - no filter row;
   - no baseline;
   - no VisDA;
   - no `.005`.
10. Create:
   ```text
   experiment_logs/conv_lbi_32gpu_burst_20260827/plans/node6_out_r1.jsonl
   tools/run_conv_lbi_32gpu_node6_out_r1.sh
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
   experiment_logs/conv_lbi_32gpu_burst_20260827/node_records/node6_out_r1_PREPARE.md
   ```
15. Print only:
   ```bash
   cd /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined
   bash tools/run_conv_lbi_32gpu_node6_out_r1.sh
   ```
16. Stop.

# MODE=FINALIZE

Aggregate existing artifacts only.

Do not launch, retry, rerun, or change anchors.

1. Audit exactly 18 planned rows.
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
   experiment_logs/conv_lbi_32gpu_burst_20260827/node_records/node6_out_r1_FINALIZE.md
   ```

Node 6 is not the global R1 selector.
Do not freeze cross-transfer winners here.
Node 7 will aggregate Node 6 + Node 7 + existing D→A R0 evidence.


End with:
```text
NODE6_FINALIZE_COMPLETE: YES/NO
PLANNED: 18
COMPLETED: X
FAILED: X
MISSING: X
```
