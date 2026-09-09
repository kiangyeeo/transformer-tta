# Codex Prompt — Conv Out-Channel Group-Size Bias Sanity Check
## Artifact-only / read-only analysis
## No training, no GPU, no scientific-code changes

Project root:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined
```

Conda:

```text
SHOT_TTA
```

Goal:

> Check whether the current Conv `out_channel` LBI support is disproportionately dominated by the largest channel groups, especially the 3x3 `conv2` groups, due to heterogeneous group sizes.

This is a reviewer-risk sanity check only.

Do not change the method.
Do not tune anything.
Do not run/retry/resume any experiment.
Do not probe GPUs.
Do not modify source/config/protocol/test files.

---

# 1. Scientific context

The frozen Conv candidate scope contains exactly these 9 tensors:

```text
netF.layer4.0.conv1.weight
netF.layer4.0.conv2.weight
netF.layer4.0.conv3.weight
netF.layer4.1.conv1.weight
netF.layer4.1.conv2.weight
netF.layer4.1.conv3.weight
netF.layer4.2.conv1.weight
netF.layer4.2.conv2.weight
netF.layer4.2.conv3.weight
```

For `out_channel`, one group is one output channel across the entire:

```text
Cin x Kh x Kw
```

slice.

For Office ResNet-50 the expected group geometry is:

| tensor | kernel | Cout / group count | scalars per group |
|---|---:|---:|---:|
| layer4.0.conv1 | 1x1 | 512 | 1024 |
| layer4.0.conv2 | 3x3 | 512 | 4608 |
| layer4.0.conv3 | 1x1 | 2048 | 512 |
| layer4.1.conv1 | 1x1 | 512 | 2048 |
| layer4.1.conv2 | 3x3 | 512 | 4608 |
| layer4.1.conv3 | 1x1 | 2048 | 512 |
| layer4.2.conv1 | 1x1 | 512 | 2048 |
| layer4.2.conv2 | 3x3 | 512 | 4608 |
| layer4.2.conv3 | 1x1 | 2048 | 512 |

Expected totals:

```text
out-channel groups = 9216
candidate scalars   = 12845056
```

Verify these from the actual implementation/model metadata if possible.

Do not silently correct the table. If the repository/artifacts disagree, report the exact discrepancy.

---

# 2. Scope of analysis

Use the already completed **Office Conv Out-LBI final/frozen runs** only.

Target all three formal budgets:

```text
rho_G = .0005   K=4
rho_G = .001    K=9
rho_G = .002    K=18
```

and all six Office transfers:

```text
A->D
A->W
D->A
D->W
W->A
W->D
```

Use the final frozen Out-LBI tuples already selected by the Office R2 pipeline.

Locate them from existing canonical artifacts such as:

```text
OFFICE_OUT_LBI_FINAL_TUPLES.json
OFFICE_OUT_LBI_FINAL_TUPLES.md
OFFICE_OUT_R2_FINAL_COMPARISON.*
phase_records/R2_BOUNDARY_EXPANSION/FINALIZE.md
```

Do not assume a run-root path if the canonical finalizer points elsewhere.

Resolve the exact final run for every:

```text
budget x transfer
```

from existing provenance.

No reruns.

---

# 3. First question: are selected-group identities recoverable?

Before doing any statistics, inspect the existing final-run artifacts:

```text
summary.json
metrics.jsonl
results.json
manifest.json
config.yaml
checkpoint metadata
diagnostic artifacts
any saved masks/support files
```

Also inspect the current artifact-writing code read-only if needed.

Determine whether existing artifacts preserve, for each online batch, enough information to recover:

```text
which out-channel groups were selected
and which candidate tensor/layer each selected group belongs to
```

Important:

- A global `selected_group_count` is NOT enough.
- A global realized scalar count is NOT enough.
- Final model parameter differences are NOT a valid substitute for the per-batch mask, because persistent writeback accumulates changes across batches.
- Do not infer exact group identity from aggregate counts.

If identities are not recoverable from existing artifacts, skip Sections 4-7 and go directly to Section 8.

---

# 4. If identities ARE recoverable: per-layer support statistics

For every final run and every online batch, compute for each of the 9 tensors:

```text
candidate_group_count
group_scalar_size
selected_group_count
selection_rate = selected_group_count / candidate_group_count
selected_scalar_count = selected_group_count * group_scalar_size
fraction_of_selected_groups
fraction_of_selected_scalars
```

Then aggregate over:

```text
per run
per budget
per transfer
all six transfers
```

Report mean / median / min / max where meaningful.

Do not collapse only to a single final batch.

The support is batch-local/dynamic, so the main analysis should describe selection over the full online stream.

---

# 5. Conv-type aggregation

Also aggregate the 9 tensors into:

```text
conv1
conv2
conv3
```

and:

```text
1x1
3x3
```

For each budget report:

```text
candidate group share
selected group share
candidate scalar share
selected scalar share
mean per-group selection rate
```

The key reviewer-risk quantity is whether the three 3x3 `conv2` tensors are selected much more often than expected from their candidate-group share.

---

# 6. Group-size bias diagnostics

Use the 9 tensor-level points with:

```text
group_scalar_size
selection_rate
```

to report simple descriptive evidence.

At minimum:

```text
selection rate by group size:
  512
  1024
  2048
  4608
```

and compare:

```text
3x3 size=4608
vs
1x1 sizes=512/1024/2048
```

Do not overclaim causality because layer identity, block position, and group size are confounded.

If there are enough batch-level observations, additionally report:

```text
Spearman correlation:
log(group_scalar_size) vs selection_rate
```

globally and by budget.

Treat this only as a descriptive sanity metric, not causal proof.

---

# 7. Reviewer-risk classification

Classify the result as exactly one of:

```text
OUT_GROUP_SIZE_RISK = LOW
OUT_GROUP_SIZE_RISK = MODERATE
OUT_GROUP_SIZE_RISK = HIGH
```

Use these guidelines:

### LOW

```text
3x3 conv2 is not strongly dominant;
selection appears across conv1/conv2/conv3;
no evidence that large groups monopolize support.
```

### MODERATE

```text
3x3 groups are clearly enriched,
but meaningful support still appears in smaller 1x1 groups
and no single group-size class monopolizes selection.
```

### HIGH

```text
support is overwhelmingly concentrated in the largest 3x3 groups
across budgets/transfers,
or smaller groups are nearly never selected.
```

Do not invent hard numerical thresholds unless the observed distribution makes an obvious separation.

---

# 8. If identities are NOT recoverable

If the existing formal artifacts do not preserve per-layer/per-group support identity, do NOT modify code and do NOT rerun training.

Write a precise audit explaining:

```text
what fields are available
what fields are missing
why final model diff cannot reconstruct dynamic masks
whether any exact per-layer statistic can still be recovered
```

Then conclude:

```text
EXISTING_ARTIFACTS_SUFFICIENT = NO
OUT_GROUP_SIZE_RISK = NOT_MEASURABLE_FROM_EXISTING_ARTIFACTS
```

and propose the minimum future diagnostic logging needed:

```text
per online batch:
  tensor name
  selected out-channel indices OR at minimum selected count per tensor
```

Do not implement that logging in this task.

---

# 9. Allowed writes

Create only:

```text
experiment_logs/conv_out_group_size_sanity_20260829/
```

Inside it, write:

```text
AUDIT.md
```

If existing artifacts are sufficient, also write:

```text
PER_LAYER_SELECTION.csv
PER_CONVTYPE_SELECTION.csv
GROUP_SIZE_SELECTION.csv
```

No other writes are allowed.

Do not create a permanent `tools/` script.

If temporary analysis code is needed, use a shell/Python here-document or `/tmp`, and do not modify the repository source tree.

---

# 10. Safety

Before and after analysis, record:

```bash
git status --short
git rev-parse HEAD
```

Do not require the tree to be clean because other ongoing experiment artifacts may exist.

But verify that this task itself changed only:

```text
experiment_logs/conv_out_group_size_sanity_20260829/**
```

Do not:

```text
git add
git commit
git push
git reset
git clean
git stash
git checkout
```

Do not launch Python training entrypoints.

Do not call `nvidia-smi`.

---

# 11. Final output

`AUDIT.md` must end with:

```text
FINAL_RUNS_RESOLVED: X/18
EXISTING_ARTIFACTS_SUFFICIENT: YES/NO
PER_BATCH_GROUP_IDENTITY_RECOVERABLE: YES/NO
SOURCE_CODE_CHANGED: NO
TRAINING_RERUN: NO
OUT_GROUP_SIZE_RISK: LOW/MODERATE/HIGH/NOT_MEASURABLE_FROM_EXISTING_ARTIFACTS
NEXT_ACTION: NONE / ADD_DIAGNOSTIC_LOGGING_FOR_FUTURE_RUN_ONLY
```

If artifacts are sufficient, include a concise reviewer-facing interpretation:

```text
Does heterogeneous group size appear to dominate the current Out-LBI support?
What evidence supports the answer?
```

Then STOP.
