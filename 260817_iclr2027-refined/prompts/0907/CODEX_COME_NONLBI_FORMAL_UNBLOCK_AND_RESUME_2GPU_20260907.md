# CODEX — Fix COME Non-LBI Sparse Formal Gate, Then Resume Office Baselines

## Goal

Fix the confirmed protocol/implementation mismatch that currently blocks all formal COME sparse variants.

Correct semantics:

```text
Random / Magnitude / Saliency:
formal execution = ALLOWED

COME-LBI:
formal execution = BLOCKED
until LBI search protocol + tuned tuples are frozen
```

After the fix passes contracts/review/dry-run, immediately resume the previously requested **COME Office all baselines formal run on 2 GPUs**.

Do not run LBI.

---

## 1. Root cause

Current code incorrectly does:

```python
if config.get("formal_protocol") is not False:
    raise ValueError("formal COME sparse/LBI launch remains blocked")
```

This over-blocks non-LBI sparse baselines.

Current protocol section 27 already says only:

```text
LBI tuned tuple未冻结前 formal LBI blocked.
```

Therefore implementation gating is too broad.

Also current sparse trainer hardcodes:

```text
formal = false
formal_protocol = false
```

for child/final/Random top-level artifacts, which is wrong for formal non-LBI sparse runs.

---

## 2. Required code semantics

### `come_otta/sparse_config.py`

Allow:

```text
come_fc_random
come_fc_magnitude
come_fc_saliency
come_conv_out_random
come_conv_out_magnitude
come_conv_out_saliency
```

when:

```text
formal_protocol = true
```

Continue fail-closed for:

```text
come_fc_lbi
come_conv_out_lbi
```

when formal is requested.

Expected error for LBI:

```text
formal COME-LBI launch remains blocked until search protocol and tuned tuples are frozen
```

`formal_protocol` must remain boolean.

---

## 3. Formal artifact markers

In `come_otta/sparse_trainer.py`:

For non-LBI sparse formal runs:

```text
formal = true
formal_protocol = true
debug_only = false
debug_smoke = false
debug_max_outer_batches = null
```

For reviewed smoke/debug:

```text
formal = false
formal_protocol = false
```

Random top-level summary must be formal only if all 3 children are formal.

Do not change sparse update semantics.

---

## 4. Keep LBI blocked

Even after config resolution, retain a second execution-layer fail-closed guard:

```text
if formal_run and variant in LBI_VARIANTS:
    fail
```

This is defense in depth.

Do not enable:

```text
come_fc_lbi
come_conv_out_lbi
```

for formal execution.

---

## 5. Implementation revision

Because execution/provenance code changes, bump:

```text
come_otta_sparse_lbi_20260907_v1
```

to:

```text
come_otta_sparse_lbi_20260907_v2
```

Update consistently in:

```text
protocol_constants.py
COME sparse debug configs
COME LBI debug config
OTTA_COME_LBI_PROTOCOL_20260907_v1.md
```

Do not bump the scientific protocol revision.

Scientific sparse/LBI semantics are unchanged.

---

## 6. Add dedicated formal non-LBI sparse config

Add:

```text
configs/come_otta_sparse_formal_20260907_v2.yaml
```

Base identity may be:

```text
variant = come_fc_magnitude
requested_budget = 0.001
formal_protocol = true
debug_max_outer_batches = null
save_model = false
```

Output root must be formal, not `runs_smoke`.

Use CLI/config overrides for transfer/variant/budget/GPU.

The config must not contain an `lbi` tuple.

---

## 7. Protocol status correction

In:

```text
protocol/come-otta/OTTA_COME_LBI_PROTOCOL_20260907_v1.md
```

Status must say:

```text
non-LBI sparse baselines (Random/Magnitude/Saliency) allow formal launch;
COME-LBI remains blocked until search protocol and tuned tuples are frozen.
```

Keep section 27 semantics:

```text
Random/Magnitude/Saliency formal allowed.
LBI formal blocked before tuned tuple freeze.
```

No scientific-semantic changes.

---

## 8. Contracts

Update/add contracts proving all of the following:

1. formal FC Magnitude resolves;
2. formal FC Random resolves;
3. formal FC Saliency resolves;
4. formal Conv Random resolves;
5. formal Conv Magnitude resolves;
6. formal Conv Saliency resolves;
7. formal FC LBI fails closed;
8. formal Conv LBI fails closed;
9. formal sparse rejects debug_max_outer_batches;
10. formal sparse has save_model=false;
11. child summary formal markers are correct;
12. Random top-level formal markers are correct;
13. Random still executes 3 real children;
14. sparse scientific identity uses implementation revision v2;
15. SHOT / IST / NCTTA / core LBI regressions remain unchanged.

Run:

```text
come_sparse_lbi_contract_test.py
come_formal_baseline_contract_test.py
protocol_alignment_smoke_test.py
implementation_revision_smoke_test.py
random_multimask_summary_smoke_test.py
relevant SHOT/IST/NCTTA/shared-LBI regressions
git diff --check
```

If full local server test environment is available, run all applicable tests.

---

## 9. Independent review before GPU launch

Review:

```text
formal non-LBI resolver
LBI fail-closed guard
formal/debug markers
Random child aggregation
implementation revision
protocol consistency
output root
source checkpoint identity
stream identity
```

Confirm no change to:

```text
COME objective
candidate scope
budgets
selector semantics
optimizer
BN
netC
PU/FO
```

---

## 10. Required dry-runs

Before GPU execution:

### Must succeed

Resolve/dry-run:

```text
come_fc_magnitude
formal_protocol=true
rho=0.001
```

and:

```text
come_conv_out_saliency
formal_protocol=true
rho=0.001
```

No debug limit.

### Must fail

Resolve/dry-run:

```text
come_fc_lbi
formal_protocol=true
```

Expected fail-closed.

No GPU run is needed for this LBI negative test.

---

## 11. Then resume Office formal baselines

If all above PASS, continue directly with the prior task:

```text
CODEX_COME_OFFICE_ALL_BASELINES_2GPU_20260907.md
```

Run only:

```text
Dense:
come_full_dense
come_fc_module_dense
come_conv_module_dense

FC:
come_fc_random
come_fc_magnitude
come_fc_saliency

Conv:
come_conv_out_random
come_conv_out_magnitude
come_conv_out_saliency
```

Office transfers:

```text
A->D
A->W
D->A
D->W
W->A
W->D
```

Budgets:

```text
0.0005
0.001
0.002
```

Random seeds:

```text
202600
202601
202602
```

Hardware:

```text
2 GPUs
max 2 concurrent jobs / GPU if memory-safe
```

Do not run COME-LBI.

---

## 12. Stop conditions

If the gate fix passes, do not stop after the fix: proceed into the Office baseline matrix.

Stop only when either:

```text
A) a new correctness blocker is found
```

or:

```text
B) all COME Office baselines are complete and summarized
```

No LBI search/formal in this task.

---

## Final report

Report:

```text
1. changed files
2. implementation revision
3. formal non-LBI gate PASS
4. formal LBI remains blocked PASS
5. contract/regression results
6. dry-run results
7. GPU scheduler settings
8. Office completion count
9. summary artifact paths
10. other-method isolation
```

Final verdict:

```text
COME_OFFICE_BASELINES_COMPLETE
```

or:

```text
COME_OFFICE_BASELINES_INCOMPLETE
```
