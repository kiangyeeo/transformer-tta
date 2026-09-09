# CODEX — SAFE: Stabilize COME Objective and Re-run ALL Office + VisDA Baselines on GPU 0–3

## 0. ABSOLUTE SAFETY RULES — READ FIRST

The local Codex configuration is intentionally permissive:

```toml
approval_policy = "never"
sandbox_mode = "danger-full-access"
```

Therefore **you have no interactive approval safety net**.

The rules below have higher priority than convenience, cleanup, or speed.

### 0.1 Never delete, move, rename, overwrite, revert, or clean existing user data

DO NOT run destructive commands such as:

```bash
rm
rm -r
rm -rf
rmdir
find ... -delete
git clean
git reset
git reset --hard
git checkout -- .
git checkout -- <path>
git restore .
git restore <path>
git revert
git stash
git stash -u
mv <existing-user-file-or-dir> ...
truncate
```

Also DO NOT:

```text
delete existing runs
delete existing experiment_logs
delete old COME artifacts
delete failed VisDA artifacts
delete prompts
delete protocols
delete checkpoints
delete datasets
delete untracked files
delete .orig/.rej files unless they were created by THIS task and their exact path is known
recursively chmod/chown
modify shell startup files
modify conda/system configuration
kill unrelated processes
use killall/pkill broad patterns
```

If cleanup seems useful:

```text
DO NOT CLEAN IT.
Leave it and report it.
```

### 0.2 Existing user modifications must be preserved

At task start run only read-only Git inspection:

```bash
git status --short
git diff --name-only
git diff
```

Save the initial status/diff to a NEW task log file.

If an allowlisted file already has user modifications before this task:

```text
do not revert them
do not replace the whole file
apply the smallest possible patch on top
```

If the existing change conflicts with this task and cannot be safely preserved:

```text
STOP and report the exact conflict.
```

Never use Git commands to restore a "clean" version.

### 0.3 Edit allowlist

Only the following existing scientific/provenance files may be modified, and only when required for this task:

```text
260817_iclr2027-refined/come_otta/objective.py

260817_iclr2027-refined/protocol_constants.py
260817_iclr2027-refined/experiment_identity.py
    # only if revision/provenance plumbing actually requires it

260817_iclr2027-refined/configs/come_baseline_protocol_20260907_v1.yaml
260817_iclr2027-refined/configs/come_otta_sparse_formal_20260907_v2.yaml
260817_iclr2027-refined/configs/come_fast_feasibility_20260907_v1.yaml
260817_iclr2027-refined/configs/come_otta_sparse_debug_20260907_v1.yaml
260817_iclr2027-refined/configs/come_otta_sparse_lbi_debug_20260907_v1.yaml
    # only revision/provenance fields; scientific hyperparameters unchanged

260817_iclr2027-refined/protocol/come-otta/OTTA_COME_BASELINE_PROTOCOL_20260907_v1.md
260817_iclr2027-refined/protocol/come-otta/OTTA_COME_LBI_PROTOCOL_20260907_v1.md
    # only implementation revision/status/numerical-stability provenance;
    # do not change scientific semantics
```

Tests may be:

```text
new files under tests/ with COME-specific names
or minimal edits to existing COME-specific tests only
```

You may create NEW:

```text
launcher scripts
queue manifests
scheduler logs
summary scripts
new formal run directories
new experiment_logs directories
new task logs
```

### 0.4 Absolute read-only paths

The following must not be modified:

```text
shot_otta/**
ist_otta/**
nctta_otta/**
core/lbi/**
datasets/**
checkpoints/**
existing historical runs/**
existing historical experiment_logs/**
```

Also do not modify unrelated top-level source files.

If you believe a change outside the allowlist is required:

```text
STOP.
Report the exact file and reason.
Do not modify it.
```

### 0.5 No blind file replacement

For existing files:

```text
prefer minimal patch edits
do not cat > existing_file
do not overwrite entire existing files with generated replacements
```

For NEW files only, `cat > new_file` is allowed.

### 0.6 Process safety

Only terminate PIDs that were launched by THIS task and whose PID is recorded by this task's launcher.

Never terminate unrelated user/system jobs.

---

# 1. Goal

A confirmed float32 numerical overflow exists in the current direct COME opinion computation on VisDA controlled variants:

```text
exp(constrained_logits) -> inf
belief = inf / inf -> NaN
COME entropy -> NaN
```

This is not OOM, not a source-checkpoint problem, and not a GPU-count problem.

This task must:

1. make the COME opinion computation numerically stable using a mathematically equivalent log-domain formulation;
2. bump dense and sparse implementation revisions;
3. run one very short synthetic direct-vs-stable loss/gradient equivalence contract;
4. immediately re-run **ALL COME Office + VisDA adaptation baselines** from source checkpoints using the same stable implementation revision;
5. summarize only the new stable-revision results.

Do NOT run model smoke tests.

Do NOT run COME-LBI.

Do NOT perform hyperparameter search.

---

# 2. Current old revisions

Current pre-stability revisions are:

```text
COME baseline implementation:
come_otta_baseline_20260907_v1

COME sparse/LBI implementation:
come_otta_sparse_lbi_20260907_v2
```

Old results are historical evidence only.

Do not delete or overwrite them.

Do not count them toward the new stable-revision completion matrix.

---

# 3. Scientific semantics MUST remain unchanged

Frozen COME semantics:

```text
p = 2
tau = 1
Office K = 31
VisDA K = 12
opinion_eps = 1e-7
```

Constrained logits must remain exactly:

```python
norm = torch.norm(logits, p=2, dim=-1, keepdim=True)
constrained = logits / norm * norm.detach() * 1.0
```

Keep:

```text
no norm epsilon
no norm clamp
detach only on multiply-back norm
```

Do not change:

```text
learning rate
batch size
optimizer
scheduler
source checkpoints
target stream
seed
BN behavior
candidate scope
selector semantics
sparse budget
Random seeds
```

Forbidden numerical "fixes":

```text
logit clamp
loss clipping
gradient clipping
NaN-to-zero
skip problematic batch
float64 workaround
confidence filtering
extra regularization
change tau
change K
change epsilon
change LR
change BS
```

---

# 4. Required log-domain stable computation

Original mathematical objective:

$$
e_k=\exp(\tilde z_k)
$$

$$
S=\sum_k e_k+K
$$

$$
b_k=e_k/S
$$

$$
u=K/S
$$

Do not explicitly materialize the unstable raw `exp(constrained)` path for the loss.

Compute:

$$
\log S
=
\operatorname{logsumexp}
(\tilde z_1,\ldots,\tilde z_K,\log K)
$$

Then:

$$
b_k=\exp(\tilde z_k-\log S)
$$

$$
u=\exp(\log K-\log S)
$$

A valid implementation pattern is:

```python
log_k = constrained.new_full(
    (constrained.shape[0], 1),
    math.log(float(class_count)),
)

log_strength = torch.logsumexp(
    torch.cat([constrained, log_k], dim=1),
    dim=1,
    keepdim=True,
)

belief = torch.exp(constrained - log_strength)
uncertainty = torch.exp(log_k - log_strength)
opinion = torch.cat([belief, uncertainty], dim=1)

safe_opinion = opinion + 1e-7
entropy_per_sample = -(
    safe_opinion * torch.log(safe_opinion)
).sum(dim=1)
loss = entropy_per_sample.mean()
```

Equivalent `logaddexp(logsumexp(...), logK)` is acceptable.

Critical:

```text
log_k constant w.r.t. logits
constrained is NOT detached
original norm.detach semantics remain intact
```

Do not change the mathematical loss.

---

# 5. Stable diagnostics

Do not reintroduce overflow just for diagnostics.

If current diagnostics need uncertainty/opinion entropy:

```text
compute them directly from stable belief/uncertainty
```

If current code exposes evidence/strength diagnostics, represent them only in a finite/stable way that does not enter the loss gradient.

Do not call raw `torch.exp(constrained)` merely to populate logs.

---

# 6. Required synthetic equivalence contract — this is NOT a smoke run

Before formal GPU execution, run only a quick synthetic unit/contract test.

No dataset loading.
No checkpoint loading.
No target-stream batches.

Cover:

```text
float32
K=12
K=31
multiple seeds
moderate logits where old direct formula is finite
```

Compare old direct formula vs new stable formula.

Check:

```text
loss close
belief close
uncertainty close
opinion entropy close
dL/dlogits close
```

Use meaningful float32 tolerances, approximately:

```text
loss: 1e-6 to 1e-5 scale
gradient: ~1e-5 scale
```

Also add a large-finite-logit regression where:

```text
old direct implementation becomes nonfinite
new stable implementation remains finite
new gradient remains finite
```

This contract must PASS.

Also explicitly test that the constrained-logit detach behavior is unchanged.

After this contract and basic compile/revision sanity pass:

```text
START FORMAL GPU RUNS IMMEDIATELY.
```

No 1-batch or 5-batch smoke.

---

# 7. Implementation revision bump

Because floating-point evaluation changes, bump both revisions.

Use:

```text
COME baseline implementation:
come_otta_baseline_20260908_v2

COME sparse/LBI implementation:
come_otta_sparse_lbi_20260908_v3
```

If the repository requires an equivalent naming format, keep the same semantic version distinction.

Update only relevant revision/provenance fields.

Do NOT bump scientific protocol revision.

Add a concise protocol/status note:

```text
same frozen COME mathematical objective;
numerically stable log-domain evaluation replaces direct exponential evaluation
```

---

# 8. LBI remains blocked

Do not run:

```text
come_fc_lbi
come_conv_out_lbi
```

Their formal gate must remain fail-closed until LBI search/tuned tuples are frozen.

This task runs only:

```text
dense
Random
Magnitude
Saliency
```

---

# 9. Source checkpoints remain exactly the SHOT source checkpoints

Keep:

```text
source checkpoint revision:
nips2026_shot_otta_uda_source_v1
```

COME must load the same source F/B/C as SHOT.

Keep source path and SHA256 provenance checks.

Do not retrain or substitute source checkpoints.

---

# 10. GPU constraint — ONLY GPU 0,1,2,3

Available physical GPUs:

```text
GPU 0
GPU 1
GPU 2
GPU 3
```

**Use only these four GPUs.**

Do not use any other GPU index.

Set launcher assignments explicitly.

Do not rely on accidental visibility of other devices.

---

# 11. GPU utilization / kill constraint

Resource policy:

> If GPU utilization remains too low (approximately below 30%) over a 3-hour window, the allocation may be killed.

Therefore keep all four GPUs continuously fed while pending work exists.

Default scheduler:

```text
GPU0: max 2 concurrent jobs
GPU1: max 2 concurrent jobs
GPU2: max 2 concurrent jobs
GPU3: max 2 concurrent jobs
```

Global maximum:

```text
8 concurrent formal jobs
```

Start the formal queue directly after the tiny equivalence contract.

Do not wait for user confirmation.

Whenever a job finishes:

```text
immediately refill that GPU from pending queue
```

If pending jobs exist and a GPU has only one active job with low utilization:

```text
schedule a second compatible formal job on that GPU
```

Never change scientific settings to raise utilization.

---

# 12. OOM policy

If a specific identity OOMs under two-way same-GPU concurrency:

1. record the identity;
2. immediately keep other GPUs working;
3. retry that exact identity once with that GPU temporarily limited to one job;
4. do NOT modify batch size or any scientific setting;
5. after it finishes, return that GPU to max 2 jobs.

Do not globally reduce concurrency unless repeated OOM evidence requires it.

---

# 13. Task ordering to prevent GPU idle time

Use long/short mixed scheduling.

Distribute early across GPU 0–3:

```text
VisDA jobs
D->A
W->A
Random identities
```

Interleave with shorter Office target D/W jobs.

Do not leave:

```text
all VisDA
all Random
all long target-A jobs
```

to the end.

As long as pending work exists, prioritize continuous utilization.

---

# 14. GPU monitoring

Create a NEW scheduler log.

Every ~5 minutes record at least:

```text
timestamp
GPU id
utilization.gpu
memory.used
active job count
pending job count
```

Use read-only `nvidia-smi`.

Do not create a process that interferes with training.

---

# 15. Old adaptation results MUST NOT be reused as stable-revision results

All COME adaptation results generated under:

```text
come_otta_baseline_20260907_v1
come_otta_sparse_lbi_20260907_v2
```

must be treated as historical/pre-stability.

Do not delete them.

Do not overwrite them.

Do not skip new stable jobs because an old-revision directory exists.

All new formal adaptation identities must execute from source under the new stable revision.

---

# 16. Source-only exception

Source-only does not evaluate the COME adaptation objective.

Therefore do not re-run source-only.

Reference the existing shared SHOT source-only artifact only when source SHA/revision matches.

Source-only is not part of the 147 COME adaptation identities.

---

# 17. Complete stable formal matrix

## Office-31

Transfers:

```text
A->D
A->W
D->A
D->W
W->A
W->D
```

Dense:

```text
come_full_dense
come_fc_module_dense
come_conv_module_dense
```

FC sparse non-LBI:

```text
come_fc_random
come_fc_magnitude
come_fc_saliency
```

Conv sparse non-LBI:

```text
come_conv_out_random
come_conv_out_magnitude
come_conv_out_saliency
```

Budgets:

```text
rho=0.0005
rho=0.001
rho=0.002
```

Office frozen settings remain unchanged.

---

## VisDA-C

Transfer:

```text
Train/Synthetic -> Validation/Real
```

Dense:

```text
come_full_dense
come_fc_module_dense
come_conv_module_dense
```

FC sparse non-LBI:

```text
come_fc_random
come_fc_magnitude
come_fc_saliency
```

Conv sparse non-LBI:

```text
come_conv_out_random
come_conv_out_magnitude
come_conv_out_saliency
```

Budgets:

```text
rho=0.0005
rho=0.001
rho=0.002
```

Frozen VisDA:

```text
ResNet-101
BS=256
workers=4
K=12
seed=2026
base lr=.001
primary=fixed-12-class mAcc
```

---

# 18. Random semantics unchanged

Each Random top-level identity must execute exactly 3 real child trajectories:

```text
202600
202601
202602
```

Each child:

```text
fresh source F/B/C
fresh optimizer state
same target stream
same scientific settings
only mask seed differs
```

Do not reuse old-revision Random child artifacts.

---

# 19. Total stable-revision work

Top-level:

```text
Office = 126
VisDA = 21
Total = 147
```

Actual executions:

```text
Dense = 21
Magnitude + Saliency = 84
Random child executions = 126
Total = 231
```

Completion target:

```text
147 / 147 top-level
231 / 231 actual executions
```

Only new stable-revision runs count.

---

# 20. Formal settings

Every adaptation run:

```text
formal_protocol = true
debug_max_outer_batches = null
save_model = false
seed = 2026
```

No partial-stream formal results.

If a process is interrupted:

```text
rerun the same identity from source
```

Do not resume a partial model state.

---

# 21. New output root only

Do not overwrite old runs.

Create a new stable-revision output root, e.g.:

```text
260817_iclr2027-refined/runs/come_formal_baselines_stable_20260908/
```

Create new summary/log root, e.g.:

```text
260817_iclr2027-refined/experiment_logs/come_office_visda_all_baselines_stable_20260908/
```

All launcher scripts/logs created by this task must live under a new task-specific directory.

Do not edit historical artifact files in place.

---

# 22. Failure handling

## Numerical/correctness failure after stable fix

If any new stable run still produces:

```text
NaN/Inf
source SHA mismatch
scope violation
identity mismatch
formal marker mismatch
```

record the exact identity and diagnostics.

Do not change settings.

Keep unrelated healthy jobs running so GPUs stay utilized.

If the same stable correctness failure appears in >=2 distinct identities of the same family:

```text
stop launching NEW jobs for that affected family
continue unrelated healthy families
report the blocker
```

## Infrastructure failure

For transient process/I/O/OOM failure:

```text
retry same identity once
```

No scientific-setting changes.

---

# 23. No result-driven tuning

Do not react to accuracy.

Forbidden:

```text
change LR after seeing result
change budget
change selector
change BN mode
rerun because accuracy is low
pick favorable Random child
skip unfavorable result
```

This is a frozen formal matrix.

---

# 24. Artifact completeness audit AFTER runs

Do not use this as a pre-run smoke.

After GPU jobs finish, validate:

```text
21/21 dense
63/63 FC sparse top-level
63/63 Conv sparse top-level
147/147 total top-level
231/231 actual executions
```

Every Random identity must have:

```text
3/3 children
```

Every new adaptation artifact must carry the new stable revision.

---

# 25. Office summary

Create stable-revision Office tables containing:

```text
shared source-only reference

come_full_dense
come_fc_module_dense
come_conv_module_dense

FC random/magnitude/saliency x 3 budgets
Conv random/magnitude/saliency x 3 budgets
```

Report:

```text
PU
FO
```

Random:

```text
three child values
mean
std
```

Office aggregate:

```text
first aggregate Random children within each transfer
then equal-weight average the 6 transfers
```

---

# 26. VisDA summary

Include all stable variants.

Primary:

```text
PU fixed-12-class mAcc
FO fixed-12-class mAcc
```

Also report:

```text
overall accuracy
12 classwise accuracies
worst-class accuracy
classwise std
```

Random:

```text
three child values
mean/std
```

---

# 27. Final summary artifacts

Create new files such as:

```text
COME_OFFICE_BASELINES_STABLE_SUMMARY.md
COME_OFFICE_BASELINES_STABLE_SUMMARY.csv

COME_VISDA_BASELINES_STABLE_SUMMARY.md
COME_VISDA_BASELINES_STABLE_SUMMARY.csv

COME_ALL_BASELINES_STABLE_MANIFEST.jsonl
COME_ALL_BASELINES_STABLE_COMPLETENESS.json

gpu_scheduler.log
failed_or_retried_runs.jsonl
```

Do not overwrite old summary files.

---

# 28. Final provenance audit

Every new COME adaptation result must use:

```text
come_otta_baseline_20260908_v2
```

or, for sparse:

```text
come_otta_sparse_lbi_20260908_v3
```

No final new COME adaptation row may point to:

```text
come_otta_baseline_20260907_v1
come_otta_sparse_lbi_20260907_v2
```

Source-only is the only allowed shared old/non-COME-adaptation reference.

---

# 29. Final Git / filesystem safety audit

At the end run read-only:

```bash
git status --short
git diff --name-only
git diff --check
```

Do NOT clean anything afterward.

Report every modified existing file.

Confirm:

```text
shot_otta/** modified = false
ist_otta/** modified = false
nctta_otta/** modified = false
core/lbi/** modified = false
datasets/checkpoints untouched
historical runs untouched
historical experiment_logs untouched
```

Also report:

```text
unexpected existing files deleted = 0
unexpected existing files moved = 0
unexpected existing files overwritten = 0
```

---

# 30. STOP condition

Stop after:

```text
stable objective implemented
synthetic equivalence contract PASS
revisions bumped
Office stable baselines complete
VisDA stable baselines complete
147/147 top-level complete
231/231 actual executions complete
stable summaries generated
```

Do NOT continue into:

```text
COME LBI search
COME LBI formal
IST search
paper table editing
new method work
cleanup
```

---

# 31. Final report

## A. Safety audit

Report:

```text
pre-existing user changes preserved
files deleted = 0
historical artifacts overwritten = 0
out-of-allowlist scientific files modified = 0
```

## B. Numerical fix

Report old/new formulas and confirm:

```text
mathematical objective unchanged
detach unchanged
p/tau/K/eps unchanged
```

## C. Equivalence contract

Report:

```text
K=12 PASS
K=31 PASS
loss max difference
gradient max difference
large-logit overflow regression PASS
```

## D. Revisions

Report exact dense/sparse stable revisions.

## E. Changed files

List every existing file modified and every new task file created.

## F. Other-method isolation

Confirm SHOT / IST / NCTTA / core LBI untouched.

## G. Scheduler

Report:

```text
GPU 0,1,2,3 only
max concurrent jobs
per-GPU concurrency
utilization log path
OOM/retries
```

## H. Completion

```text
147/147 top-level
231/231 actual executions
```

## I. Office results

Full stable summary + equal-transfer aggregate.

## J. VisDA results

Full stable summary.

## K. Random audit

Confirm all expected identities have 3/3 real children.

## L. Provenance

Confirm all new COME adaptation artifacts use the stable revisions.

## M. Failures

List only real failures/retries/quarantines.

## N. Verdict

Only:

```text
COME_STABLE_OFFICE_VISDA_BASELINES_COMPLETE
```

or:

```text
COME_STABLE_OFFICE_VISDA_BASELINES_INCOMPLETE
```

If COMPLETE, stop immediately.
