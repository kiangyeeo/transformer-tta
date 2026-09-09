# Codex Prompt — Office Conv-LBI Out-channel R2

Project:
`/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined`

Conda environment:
`SHOT_TTA`

Normative files that MUST already exist on the server:
```text
protocol/shot-otta_conv/OTTA_CONV_LBI_PROTOCOL_20260826_v1.md
protocol/shot-otta_conv/OTTA_CONV_BASELINE_FORMAL_20260827_v1.md
protocol/shot-otta_conv/OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v1.md
protocol/shot-otta_conv/OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2.md
```

Required frozen R1 result:
```text
experiment_logs/office_conv_lbi_r1_out_stage1_validation_seed2026_20260827/
  selected_configs/OFFICE_OUT_STAGE1_FROZEN.json
  phase_records/R1/FINALIZE.md
```

Scientific implementation:
```text
iclr2027_refined_conv_20260826_v1
```

## Non-negotiable scientific invariants

This phase changes ONLY:
```text
omega
stage2_lr
```

Everything else must remain frozen.

Exact common settings:
```text
dataset = office
method = shot
variant = conv_out_lbi
group_mode = out_channel
seed = 2026

kappa = 1
stage1_max_steps = 3000
support_threshold = 1e-4
stage2_steps = 1
budget_tolerance = 1e-4          # compatibility metadata only; strict K remains authoritative
delta_nonzero_tolerance = 1e-12

candidate_scope = netF.layer4_conv
controlled Conv BN = frozen
strict integer global group budget = enabled
strict rollback = enabled
LBI top-K trimming = forbidden
batch-local LBI state restart = enabled
masked-delta Stage-2 initialization = enabled
off-mask value freezing/restoration = enabled
save_model = false
runtime_comparable = false
```

Do not change SHOT loss, Office batch size/workers, source checkpoints, optimizer semantics,
candidate layers, group semantics, transforms, PU/FO semantics, or any other base-config field.

Exact frozen Stage-1 anchor by budget, taken from `OFFICE_OUT_STAGE1_FROZEN.json`:
```text
rho_G=0.0005, K_G=4:
  anchor=A1, alpha=0.05, nu=0.50

rho_G=0.001, K_G=9:
  anchor=A1, alpha=0.05, nu=0.50

rho_G=0.002, K_G=18:
  anchor=A4, alpha=0.10, nu=1.00
```

Office domain mapping MUST be verified from the repository before plan creation:
```text
amazon=0
dslr=1
webcam=2
```

Exact six transfers:
```text
AD = 0→1
AW = 0→2
DA = 1→0
DW = 1→2
WA = 2→0
WD = 2→1
```

Execution:
```text
GPUs = 0,1,2,3
max-workers = 4
workers-per-gpu = 1
1 experiment process / GPU
```

Every Python/runner call in generated shell scripts must use:
```bash
conda run --no-capture-output -n SHOT_TTA ...
```

Do not require `conda activate`.

`MODE=PREPARE`:
- MUST NOT launch training.
- MUST NOT probe/query/reserve GPUs.

`MODE=FINALIZE`:
- MUST NOT launch, retry, resume, rerun, or alter the grid.
- Node FINALIZE is an audit only; it does NOT select the global R2 winner.


# Node 5 assignment

This node owns exactly ONE R2 `(omega, stage2_lr)` cell:

```text
omega = 0.1
stage2_lr = 0.0025
cell_id = omega_0p1__lr_0p0025
```

The already-completed central cell:
```text
omega=.00625, stage2_lr=.005
```
is NOT assigned to any R2 node and MUST NOT be rerun.

This node runs the assigned cell over:
```text
3 budgets × 6 transfers = 18 scientific conditions
```

Exact rows:

| Row | Transfer | source | target | rho_G | K_G | Stage-1 anchor | alpha | nu | omega | stage2_lr |
|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | AD | 0 | 1 | 0.0005 | 4 | A1 | 0.05 | 0.5 | 0.1 | 0.0025 |
| 2 | AW | 0 | 2 | 0.0005 | 4 | A1 | 0.05 | 0.5 | 0.1 | 0.0025 |
| 3 | DA | 1 | 0 | 0.0005 | 4 | A1 | 0.05 | 0.5 | 0.1 | 0.0025 |
| 4 | DW | 1 | 2 | 0.0005 | 4 | A1 | 0.05 | 0.5 | 0.1 | 0.0025 |
| 5 | WA | 2 | 0 | 0.0005 | 4 | A1 | 0.05 | 0.5 | 0.1 | 0.0025 |
| 6 | WD | 2 | 1 | 0.0005 | 4 | A1 | 0.05 | 0.5 | 0.1 | 0.0025 |
| 7 | AD | 0 | 1 | 0.001 | 9 | A1 | 0.05 | 0.5 | 0.1 | 0.0025 |
| 8 | AW | 0 | 2 | 0.001 | 9 | A1 | 0.05 | 0.5 | 0.1 | 0.0025 |
| 9 | DA | 1 | 0 | 0.001 | 9 | A1 | 0.05 | 0.5 | 0.1 | 0.0025 |
| 10 | DW | 1 | 2 | 0.001 | 9 | A1 | 0.05 | 0.5 | 0.1 | 0.0025 |
| 11 | WA | 2 | 0 | 0.001 | 9 | A1 | 0.05 | 0.5 | 0.1 | 0.0025 |
| 12 | WD | 2 | 1 | 0.001 | 9 | A1 | 0.05 | 0.5 | 0.1 | 0.0025 |
| 13 | AD | 0 | 1 | 0.002 | 18 | A4 | 0.1 | 1 | 0.1 | 0.0025 |
| 14 | AW | 0 | 2 | 0.002 | 18 | A4 | 0.1 | 1 | 0.1 | 0.0025 |
| 15 | DA | 1 | 0 | 0.002 | 18 | A4 | 0.1 | 1 | 0.1 | 0.0025 |
| 16 | DW | 1 | 2 | 0.002 | 18 | A4 | 0.1 | 1 | 0.1 | 0.0025 |
| 17 | WA | 2 | 0 | 0.002 | 18 | A4 | 0.1 | 1 | 0.1 | 0.0025 |
| 18 | WD | 2 | 1 | 0.002 | 18 | A4 | 0.1 | 1 | 0.1 | 0.0025 |

## Exact artifact paths for this node

Branch output root passed via `--output-root`:
```text
experiment_logs/office_conv_lbi_r2_out_joint_sweep_seed2026_20260828/runs/omega_0p1__lr_0p0025
```

Important: the cell-specific directory above is mandatory because the repository's canonical
suffix contains dataset/transfer/group/budget but does NOT itself encode `omega` and
`stage2_lr`. Do not use a shared R2 output root for different cells.

Plan:
```text
experiment_logs/conv_lbi_r2_out_32gpu_20260828/plans/node5_omega_0p1__lr_0p0025.jsonl
```

Optional generated per-row configs, if the existing plan builder uses them:
```text
experiment_logs/conv_lbi_r2_out_32gpu_20260828/plans/node5_omega_0p1__lr_0p0025_configs/
```

Launcher:
```text
tools/run_conv_lbi_r2_out_node5.sh
```

PREPARE record:
```text
experiment_logs/conv_lbi_r2_out_32gpu_20260828/node_records/node5_PREPARE.md
```

FINALIZE record:
```text
experiment_logs/conv_lbi_r2_out_32gpu_20260828/node_records/node5_FINALIZE.md
```

# MODE=PREPARE

1. Read and verify all four normative protocol files.
2. Read and verify `OFFICE_OUT_STAGE1_FROZEN.json`.
   Abort if its three primary anchors are not exactly:
   ```text
   .0005 -> A1 alpha=.05 nu=.50
   .001  -> A1 alpha=.05 nu=.50
   .002  -> A4 alpha=.10 nu=1.00
   ```
3. Verify the repository Office mapping `[amazon,dslr,webcam] = [0,1,2]`.
4. Inspect the existing R1 Node6/Node7 preparation/launcher pattern and reuse the same
   canonical scientific-config/hash construction and `run_experiments_multi_gpu.py`.
   Do not copy R1 scientific rows.
5. Build exactly 18 rows from the table above.
6. Every row must have:
   ```text
   variant=conv_out_lbi
   group_mode=out_channel
   seed=2026
   omega=0.1
   stage2_lr=0.0025
   ```
   plus the budget-specific frozen `alpha/nu/K_G`.
7. Use the exact cell-specific output root:
   ```text
   experiment_logs/office_conv_lbi_r2_out_joint_sweep_seed2026_20260828/runs/omega_0p1__lr_0p0025
   ```
8. Use the repository's canonical scientific identity/hash builder.
   `omega` and `stage2_lr` MUST participate in the scientific config/hash.
   The human anchor label may remain metadata; do not invent a new identity rule.
9. Verify before writing PREPARE:
   - exactly 18 rows;
   - exactly 18 unique experiment keys;
   - exactly 18 unique scientific SHA256 values;
   - exactly 18 unique expected output roots;
   - exactly six transfers;
   - exactly six rows per budget;
   - correct source/target mapping;
   - `.0005 -> 4`, `.001 -> 9`, `.002 -> 18`;
   - exact frozen alpha/nu per budget;
   - all rows have exactly `omega=0.1`, `stage2_lr=0.0025`;
   - no central `.00625/.005` row;
   - no filter_connection;
   - no baseline;
   - no VisDA;
   - no `.005` budget.
10. Create the JSONL plan and launcher at the exact paths above.
11. Launcher must:
   - fail fast if plan/PREPARE record is missing;
   - use a node-specific lock;
   - CPU-validate the exact on-disk plan before launch;
   - use only GPUs `0,1,2,3`;
   - use `--max-workers 4 --workers-per-gpu 1`;
   - use only the frozen exact-batch-boundary `--resume --resume-partial-runs` behavior;
   - preserve command history and launcher logs;
   - use `conda run --no-capture-output -n SHOT_TTA`.
12. Run only static/CPU checks:
   ```text
   plan validation
   key/SHA/output-root uniqueness
   bash -n tools/run_conv_lbi_r2_out_node5.sh
   git diff --check
   ```
13. Do NOT launch training and do NOT probe GPUs.
14. Write `experiment_logs/conv_lbi_r2_out_32gpu_20260828/node_records/node5_PREPARE.md` with:
   - protocol and frozen-R1 SHA/provenance;
   - exact assigned cell;
   - 18-row count;
   - key/SHA/output-root uniqueness;
   - exact plan SHA256;
   - exact launcher path;
   - all static-check results.
15. Print only:
```bash
cd /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined
bash tools/run_conv_lbi_r2_out_node5.sh
```
16. Stop.

# MODE=FINALIZE

Audit only the existing 18 artifacts from this node.

Do not launch/retry/resume/rerun.

For every row:
1. Match completed artifact to plan by scientific SHA256 and experiment key.
2. Verify no duplicate completed artifact.
3. Verify summary/metrics are finite and error-free.
4. Verify per-batch:
   ```text
   selected_group_count <= K_G
   ```
5. Verify:
   ```text
   Stage-1 max-step failures
   support/mask/budget contract diagnostics
   controlled Conv BN frozen diagnostics
   ```
6. Compute/report:
   ```text
   selected_group_count min/mean/max
   group utilization min/mean/p05
   fraction of batches below .95
   selected_scalar_count / realized_scalar_ratio
   Stage-1 mean/max steps
   Stage-1 max-step failure count
   Stage-1 runtime
   Stage-2 runtime
   online runtime
   peak allocated/reserved GPU memory
   PU
   FO
   ```
7. Scientific-valid iff:
   ```text
   completed
   finite/error-free
   selected_group_count <= K_G on every batch
   zero Stage-1 max-step failures
   no support/mask/budget contract violation
   ```
8. Out-channel utilization-eligible for a transfer iff:
   ```text
   mean utilization >= .90
   p05 utilization >= .75
   ```
9. Do NOT rank by FO on this node. Global selection happens only after all 8 nodes and
   the reused central cell are audited.
10. Write `experiment_logs/conv_lbi_r2_out_32gpu_20260828/node_records/node5_FINALIZE.md`.
11. End with exactly:
```text
NODE5_R2_FINALIZE_COMPLETE: YES/NO
CELL: omega_0p1__lr_0p0025
PLANNED: 18
COMPLETED: X
FAILED: X
MISSING: X
SCIENTIFIC_VALID: X/18
UTILIZATION_ELIGIBLE: X/18
```
12. Stop.
