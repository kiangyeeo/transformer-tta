# CODEX — NCTTA P1 Dense Baseline Implementation

Implement P1 for NCTTA in `260817_iclr2027-refined/`.

## Authority

Read and obey first:

1. `protocol/nctta-otta/NCTTA_OTTA_P0_REFACTOR_CONTRACT_20260906_v2.md`
2. current SHOT refined protocol/constants
3. current IST integration as the structural template
4. official NCTTA source of truth:
   - https://github.com/Cevaaa/NCTTA
   - `ttab/model_adaptation/nctta.py`
   - `ttab/configs/algorithms.py`

Do not reimplement NCTTA from memory.

## Hard constraints

- Do not modify `shot_otta/**`.
- Do not change existing SHOT/IST scientific semantics.
- Same source F/B/C checkpoints as SHOT.
- Same Office R50 / VisDA R101.
- Same seed=2026 stream, BS64/256, one pass, singleton behavior, PU/FO, metrics.
- `netC` frozen.
- Same SHOT optimizer/scheduler substrate.
- P1 is dense baseline only: no Random/Magnitude/Saliency/LBI and no tuning.

## Implement

Create a clean independent module, preferably:

```text
nctta_otta/
├── __init__.py
├── config.py
├── objective.py
└── trainer.py
```

Add only the objective-agnostic shared dispatch/provenance extensions needed in:
- `train.py`
- `protocol_constants.py`
- `experiment_identity.py`
- config files/tests

### Variants

Implement exactly:

```text
nctta_full_dense
nctta_fc_module_dense
nctta_conv_module_dense
```

Scopes must exactly match the corresponding SHOT/IST controlled dense families.

### NCTTA objective

Faithfully port the official code path:

- normalize classifier weights/features
- top-k from current predicted probabilities
- distance soft target
- probability soft target
- hybrid target
- InfoNCE NC loss
- cosine metric
- `tau_align=1.0`
- entropy loss
- entropy filter
- entropy-based coefficient
- predicted-class FCA-distance coefficient
- weighted selected-sample mean

Do not import the TTAB runtime.

Use explicit forward:

```python
f = net_f(inputs)
h = net_b(f)
logits = net_c(h)
```

NCTTA feature is `h` (post-netB 256-d), not `netF.avgpool`.

Classifier reference is the effective frozen `netC.fc.weight.detach()` after normal forward semantics; do not substitute weight-normalization internals.

Expose official method parameters explicitly:

```text
thre_ent
margin_ent
reweight_ent
nu
eta
scale
top_k
mix_prob_weight
```

For P1 correctness smoke, record the official repository defaults. Do not tune them.

Do not add Fisher regularization, stochastic restore, oracle model selection, episodic reset, or TTAB multi-step selection.

If entropy filtering selects zero samples or loss becomes non-finite, fail loudly with diagnostics; do not invent a fallback.

## Tests

Add a dedicated NCTTA P1 contract test. At minimum verify:

1. same source checkpoint and initialization logits as SHOT;
2. same stream/batch/singleton semantics;
3. no target-label/future leakage;
4. feature dim == classifier dim == 256;
5. effective classifier weight is detached and `netC` stays byte-identical;
6. official NCTTA loss path is numerically checked against a direct/reference implementation on synthetic tensors;
7. finite loss and nonzero gradients on declared scope;
8. FC module-dense changes only bottleneck FC weight/bias, including BN buffers unchanged;
9. Conv module-dense changes only layer4 9 Conv weights, including BN buffers unchanged;
10. PU/FO are read-only;
11. scientific identity/protocol revision are correct;
12. existing SHOT and IST tests/regressions pass.

Test state transitions/update counts, not only shapes.

## Independent self-review

After tests pass, manually inspect:
- method dispatch
- trainable-scope construction
- BN modes/buffers
- classifier-weight extraction
- outer loop/singleton skip
- optimizer/scheduler time
- PU/FO
- artifact metadata
- experiment-identity fallback

Do not trust only “tests PASS”.

## Smoke

Only after contract + self-review:

```text
Office D->A
seed=2026
BS=64
5 valid outer batches
```

Run the three dense variants only.

Smoke validates real GPU/data path, scope preservation, finite objective, counters, and artifacts. Do not use smoke accuracy to change parameters.

## Stop condition

When P1 is complete, stop. Do not implement sparse/LBI and do not run formal experiments.

Output:
1. changed files;
2. exact official NCTTA code semantics ported;
3. deliberate deviations from native NCTTA and why;
4. contract/regression results;
5. independent review findings;
6. short-smoke artifacts/results;
7. unresolved items;
8. final verdict: `READY_FOR_NCTTA_SPARSE_PROTOCOL` or `NOT_READY`.
