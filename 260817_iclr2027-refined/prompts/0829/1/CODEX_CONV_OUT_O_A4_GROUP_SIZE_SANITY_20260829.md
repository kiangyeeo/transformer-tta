# Codex Prompt — O_A4 Out-Channel Group-Size Bias Sanity Check
## Artifact-only analysis of existing D0 trace
## No code changes, no training, no GPU

Worktree root:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI-filter-d0
```

Project root:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI-filter-d0/260817_iclr2027-refined
```

Expected branch:

```text
diag/filter-d0-20260828
```

Main production tree:

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI
```

The main tree is READ-ONLY and must not be modified.

---

# 1. Goal

Use the **already completed O_A4 D0 trace** to answer one reviewer-risk question:

> Does Conv `out_channel` Group-LBI disproportionately select the largest channel groups, especially the 3x3 `conv2` groups, because group sizes differ across ResNet-50 layer4 convolutions?

This is a sanity check only.

Do NOT:

```text
train
rerun
retry
resume
probe GPUs
change algorithms
change logging
modify source/config/protocol/tests
```

Use existing artifacts only.

---

# 2. Exact existing run to inspect

Under:

```text
experiment_logs/conv_filter_lbi_d0_mechanism_20260828/
```

locate the completed run with exact ID:

```text
O_A4
```

Verify its metadata is:

```text
dataset = office
transfer = D->A
variant/grouping = out_channel
rho_G = .002
K_G = 18
alpha = .10
nu = 1.00
kappa = 1
omega = .00625
stage2_lr = .005
stage1_max_steps = 3000
support_threshold = 1e-4
seed = 2026
first 6 real online batches
```

If the run cannot be uniquely resolved or any of these scientific fields disagree, STOP and report the mismatch.

Do not substitute another run.

---

# 3. Expected out-channel geometry

The nine frozen Conv candidate tensors are:

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

For `out_channel`, one group is one complete output-channel slice:

```text
Cin x Kh x Kw
```

Expected geometry:

| tensor | kernel | candidate groups | scalars/group |
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
candidate groups = 9216
candidate scalars = 12845056
```

Candidate group shares by conv type:

```text
conv1: 1536 / 9216 = 16.6667%
conv2: 1536 / 9216 = 16.6667%
conv3: 6144 / 9216 = 66.6667%
```

Candidate group shares by group scalar size:

```text
512 : 6144 / 9216 = 66.6667%
1024:  512 / 9216 =  5.5556%
2048: 1024 / 9216 = 11.1111%
4608: 1536 / 9216 = 16.6667%
```

Verify the geometry against the existing trace / current frozen candidate specification.

If the trace reports a different candidate geometry, STOP and report it.

---

# 4. First recoverability check

Inspect O_A4 trace artifacts only.

Determine whether the trace records, at each terminal/accepted Stage-1 state for each of the 6 online batches, enough information to recover at least:

```text
tensor name
candidate group count for tensor
selected/support group count for tensor
group scalar size
```

Do not require channel indices for this sanity check.

If per-tensor selected/support counts are absent, do not infer them from global counts.

If absent, write the audit with:

```text
O_A4_TRACE_SUFFICIENT = NO
OUT_GROUP_SIZE_RISK = NOT_MEASURABLE_FROM_O_A4_TRACE
```

and STOP.

---

# 5. Which Stage-1 state to use

For each of the six online batches, use the **final committed feasible Stage-1 support** that actually proceeds to Stage-2.

Do not use:

```text
an overshooting candidate that was rolled back
an intermediate snapshot as if it were final
candidate support before rollback
```

If O_A4 has no rollback, confirm that explicitly.

The sum of the nine per-tensor selected counts for each batch must equal the final global committed selected-group count.

Expected O_A4 final global support from the prior diagnostic is:

```text
18 groups per batch
```

Verify this from the actual artifacts rather than assuming it.

If layer totals do not sum to the committed global count, STOP and report `TRACE_LAYER_SUM_MISMATCH`.

---

# 6. Required per-batch table

For each of the six batches and each of the nine tensors, compute:

```text
online_batch_index
tensor_name
conv_type = conv1/conv2/conv3
kernel = 1x1/3x3
group_scalar_size
candidate_group_count
selected_group_count
selection_rate = selected_group_count / candidate_group_count
selected_group_share_within_batch = selected_group_count / global_selected_group_count
selected_scalar_count = selected_group_count * group_scalar_size
selected_scalar_share_within_batch
```

Write:

```text
experiment_logs/conv_out_o_a4_group_size_sanity_20260829/PER_LAYER_PER_BATCH.csv
```

---

# 7. Aggregate by tensor / conv type / group size

Across all six batches, aggregate selected **group occurrences**.

Important:

```text
A group selected in two different online batches counts as two selection occurrences.
```

This is a dynamic-support sanity check, not a unique-channel-count analysis.

## Per tensor

Report:

```text
total selected occurrences
mean selected count per batch
mean selection rate
share of all selected group occurrences
share of all selected scalar occurrences
```

## By conv type

Aggregate:

```text
conv1
conv2
conv3
```

For each report:

```text
candidate group share
selected group occurrence share
enrichment = selected group share / candidate group share
mean selection rate
selected scalar occurrence share
```

## By group size

Aggregate:

```text
512
1024
2048
4608
```

with the same statistics.

Write:

```text
PER_LAYER_AGGREGATE.csv
PER_CONVTYPE_AGGREGATE.csv
PER_GROUP_SIZE_AGGREGATE.csv
```

under:

```text
experiment_logs/conv_out_o_a4_group_size_sanity_20260829/
```

---

# 8. Key question: does 3x3 conv2 dominate?

The main descriptive statistic is:

```text
conv2 candidate group share = 16.6667%
```

Compare it to:

```text
conv2 selected-group occurrence share across the 6 O_A4 batches
```

Also report:

```text
conv2 enrichment ratio
= selected share / 16.6667%
```

Interpret descriptively:

```text
~1x:
  no obvious enrichment

clearly >1x:
  enrichment exists

very large enrichment with conv1/conv3 rarely selected:
  possible group-size domination
```

Do not invent a causal claim from one D->A diagnostic.

The purpose is reviewer-risk triage.

---

# 9. Additional descriptive checks

If sufficient data exist, compute:

```text
Spearman correlation across the 9 tensor-level observations:
log(group_scalar_size) vs mean selection_rate
```

Because there are only 9 tensor points and layer identity is confounded with group size, label it:

```text
DESCRIPTIVE_ONLY
```

Also report whether each conv type is selected at least once across the six batches:

```text
conv1 selected? YES/NO
conv2 selected? YES/NO
conv3 selected? YES/NO
```

and whether each of the four group-size classes is selected at least once.

---

# 10. Risk classification

End with exactly one:

```text
OUT_GROUP_SIZE_RISK = LOW
OUT_GROUP_SIZE_RISK = MODERATE
OUT_GROUP_SIZE_RISK = HIGH
OUT_GROUP_SIZE_RISK = NOT_MEASURABLE_FROM_O_A4_TRACE
```

Use this qualitative rule:

### LOW

```text
conv2 does not strongly dominate;
conv1/conv2/conv3 all receive meaningful support;
largest groups do not monopolize selection.
```

### MODERATE

```text
conv2 / size-4608 groups are clearly enriched,
but smaller 1x1 groups still receive meaningful support.
```

### HIGH

```text
selection is overwhelmingly concentrated in conv2 / size-4608 groups
across most or all six batches,
while smaller groups are rarely or never selected.
```

Do not use accuracy.

Do not claim this proves or disproves group-size causality.

---

# 11. Reviewer-facing interpretation

In `AUDIT.md`, answer in plain language:

1. Does O_A4 show obvious domination by the largest 3x3 groups?
2. How large is conv2's candidate share versus selected share?
3. Are smaller 1x1 groups still selected?
4. Is this evidence strong enough to require changing the current Out-LBI method before ICLR?
5. What is the limitation of using one D->A diagnostic run?

Keep the recommendation conservative.

---

# 12. Allowed writes

Create only:

```text
experiment_logs/conv_out_o_a4_group_size_sanity_20260829/**
```

Required:

```text
AUDIT.md
```

If trace is sufficient, also:

```text
PER_LAYER_PER_BATCH.csv
PER_LAYER_AGGREGATE.csv
PER_CONVTYPE_AGGREGATE.csv
PER_GROUP_SIZE_AGGREGATE.csv
```

Do not create/modify permanent scripts.

Temporary analysis code may be placed only in `/tmp`.

---

# 13. Safety

Before and after:

```bash
cd /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI-filter-d0
git branch --show-current
git rev-parse HEAD
git status --short
```

Do not require a clean worktree; existing D0/lambda/prompt artifacts are expected.

But verify this task itself writes only:

```text
260817_iclr2027-refined/experiment_logs/conv_out_o_a4_group_size_sanity_20260829/**
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

No GPU probing.

No training command.

---

# 14. Final footer

`AUDIT.md` must end with:

```text
O_A4_RESOLVED: YES/NO
O_A4_TRACE_SUFFICIENT: YES/NO
BATCHES_ANALYZED: X/6
LAYER_SUM_CHECK: PASS/FAIL/NOT_AVAILABLE
SOURCE_CODE_CHANGED: NO
TRAINING_RERUN: NO
MAIN_TREE_WRITTEN: NO
OUT_GROUP_SIZE_RISK: LOW/MODERATE/HIGH/NOT_MEASURABLE_FROM_O_A4_TRACE
NEXT_ACTION: NONE / HUMAN_REVIEW / ADD_MINIMAL_LOGGING_IF_STILL_NEEDED
```

Then STOP.
