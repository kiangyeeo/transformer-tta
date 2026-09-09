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
R0F — Office-31 D→A filter-connection Stage-1 rescue
```

Exact transfer:
```text
D→A = source 1 (DSLR) -> target 0 (Amazon)
```

Exact method:
```text
variant = conv_filter_lbi
group_mode = filter_connection
rho_G = .001
K_G = 6553
```

Run exactly these 4 anchors:

| Row | Anchor | alpha | nu | rho_G | K_G |
|---:|---|---:|---:|---:|---:|
| 1 | F0 | 0.25 | 0.5 | .001 | 6553 |
| 2 | F1 | 0.3 | 0.5 | .001 | 6553 |
| 3 | F2 | 0.4 | 0.5 | .001 | 6553 |
| 4 | F3 | 0.6 | 0.5 | .001 | 6553 |

Branch root:
```text
experiment_logs/office_conv_lbi_r0f_filter_reachability_seed2026_20260827
```

Node artifacts:
```text
plan:
experiment_logs/conv_lbi_32gpu_burst_20260827/plans/node2_filter_r0f.jsonl

launcher:
tools/run_conv_lbi_32gpu_node2_filter_r0f.sh

prepare record:
experiment_logs/conv_lbi_32gpu_burst_20260827/node_records/node2_filter_r0f_PREPARE.md

finalize record:
experiment_logs/conv_lbi_32gpu_burst_20260827/node_records/node2_filter_r0f_FINALIZE.md
```

# MODE=PREPARE

1. Verify all normative protocol files exist and the implementation revision is exactly `iclr2027_refined_conv_20260826_v1`.
2. Verify Office D→A mapping from the repository itself: `source=1,target=0`.
3. Build exactly 4 scientific rows, matching the table above.
4. Verify every row is:
   - `conv_filter_lbi`;
   - `filter_connection`;
   - `rho_G=.001`;
   - `K_G=6553`;
   - seed 2026;
   - the exact `(alpha,nu)` shown above;
   - `kappa=1`, `omega=.00625`, `stage2_lr=.005`,
     `stage1_max_steps=3000`, `support_threshold=1e-4`,
     `stage2_steps=1`.
5. Verify:
   - 4 rows exactly;
   - 4 unique experiment keys;
   - 4 unique scientific SHA values;
   - 4 unique output roots;
   - no `.005`;
   - no out-channel;
   - no baseline;
   - no VisDA.
6. Create:
   ```text
   experiment_logs/conv_lbi_32gpu_burst_20260827/plans/node2_filter_r0f.jsonl
   tools/run_conv_lbi_32gpu_node2_filter_r0f.sh
   ```
7. Launcher must use:
   ```text
   GPUs=0,1,2,3
   max-workers=4
   workers-per-gpu=1
   ```
8. Use the repository's canonical LBI runner and exact completed-batch resume semantics.
9. Run static checks only:
   - plan validator / equivalent;
   - uniqueness checks;
   - `bash -n tools/run_conv_lbi_32gpu_node2_filter_r0f.sh`;
   - `git diff --check`.
10. Do not launch training and do not probe GPUs.
11. Write:
   ```text
   experiment_logs/conv_lbi_32gpu_burst_20260827/node_records/node2_filter_r0f_PREPARE.md
   ```
   with the exact plan rows, keys/SHA/output roots and all static-check results.
12. Print only:
   ```bash
   cd /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined
   bash tools/run_conv_lbi_32gpu_node2_filter_r0f.sh
   ```
13. Stop.

# MODE=FINALIZE

Aggregate only existing artifacts for this node.

Do not launch, retry, rerun, or change hyperparameters.

1. Audit exactly the 4 planned rows.
2. Verify:
   - completed/missing/failed count;
   - summary exists;
   - scientific SHA matches plan;
   - no duplicate output;
   - finite metrics;
   - selected_group_count never exceeds `K_G=6553`;
   - Stage-1 max-step failure count;
   - support/mask/budget diagnostics.
3. For each row report:
   ```text
   anchor
   alpha
   nu
   completed
   scientific_valid
   selected_group_count min/mean/max
   utilization min/mean/p05
   fraction batches below .95
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
   experiment_logs/conv_lbi_32gpu_burst_20260827/node_records/node2_filter_r0f_FINALIZE.md
   ```

This node is **not** the budget-level selector. Do not choose primary/backup here.
Node 3 owns the combined `.001` budget selection after both halves finish.


End with:
```text
NODE2_FINALIZE_COMPLETE: YES/NO
PLANNED: 4
COMPLETED: X
FAILED: X
MISSING: X
```
