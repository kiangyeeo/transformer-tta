# COME-OTTA on the SHOT-Transformer substrate

Implements `OTTA_COME_TRANSFORMER_LBI_PROTOCOL_20260909_v1` sections 1-11 and
14-21 for the non-LBI variants:

```text
SHOT-Transformer substrate + COME current-logit objective
```

Everything except the host objective is **imported** from `transformer/`:
source W0 loading and manifest/SHA-256 verification, the target stream
(including the Amazon `43x64 + 65` tail), the 12-tensor candidate universe
(5,308,416 scalars), the 6912 QK/VO/FFN paired groups, the floor budgets
3/6/13, the strict masked AdamW step, and the PU/FO metric machinery.
`tests/transformer_come_substrate_identity_test.py` asserts that those are the
same code objects and that every config block matches the SHOT YAML.

## Layout

One flat package, one config, one runner - the same organisation as
`transformer_ist/`. The variant is a runtime argument, not a subpackage.

| Module | Holds |
|---|---|
| `config.py` | frozen-field validation, budgets, per-variant revisions/scopes/selection policies, transfer resolution and experiment identity |
| `config.yaml` | the blocks shared by all five variants |
| `objective.py` | the single backbone-agnostic COME implementation |
| `common.py` | the objective step, the singleton guard, the RNG/state audits, collapse diagnostics |
| `model.py` | the five update scopes, all built on the SHOT loaders |
| `data.py`, `groups.py`, `optimizer.py` | verbatim re-exports of the SHOT stream, group partition/scorers and masked AdamW |
| `runner.py` | the dense stream, the sparse stream, the Random three-child parent, and the `run_transfer` dispatcher |
| `matrix.py`, `aggregate.py`, `finalize.py` | one-process-per-GPU scheduler, SHOT aggregation semantics, Random recovery |
| `cli.py` | `transfer` / `matrix` / `finalize` |

## Implemented

| Variant | Support | `--budget` |
|---|---|---|
| `full_dense` | non-head unrestricted reference | rejected |
| `candidate_dense` | all 12 candidate tensors | rejected |
| `group_random` | 3 real children, seeds 202600/1/2 | required |
| `group_magnitude` | source-W0 paired L2, static | required |
| `group_saliency` | current-batch \|W*grad(L_COME)\| | required |

## Not implemented on purpose

`group_lbi` is absent. The COME-LBI search protocol and the six
`(alpha, kappa, nu, omega, stage2_lr)` tuples are not frozen (protocol
sections 17 and 21), so formal COME-LBI must fail closed; there is no code
path that could launch it with a provisional tuple. `--variant group_lbi` is
rejected by the CLI, by `require_supported_variant` and by
`variant_config_view`, with no dense fallback.

## Config

`config.yaml` carries only what all five variants share: the model, the target
data and stream, the optimizer, the `come:` block and the runtime policy. The
per-variant blocks - protocol revision, `adaptation` update scope, `selection`
policy and artifact root - are derived in `config.py` by
`variant_config_view(raw, variant)`, which rebuilds exactly the config each
variant used to own as its own YAML. Deriving them means a variant cannot
acquire a different substrate through a YAML edit, and the substrate identity
test still compares every derived block against the matching SHOT YAML.

Resolution is unchanged: for all 77 formal conditions (and the 63 Random
children) the resolved config, its `scientific_config_sha256` and its
`experiment_key` are byte-identical to the pre-flattening per-variant packages.

## Objective

`transformer_come/objective.py` is the single, backbone-agnostic COME
implementation (`p=2`, `come.tau=1`, `norm.detach()`, no norm epsilon/clamp,
stable log-domain opinion, `entropy_epsilon=1e-7` with no renormalization).
It is bit-exact against the audited FC reference
`260817_iclr2027-refined/come_otta/objective.py`, with one deliberate
strengthening: non-finite logits, a zero logit norm or a non-finite loss raise
instead of being repaired (protocol section 6).

Every variant reaches it through `transformer_come.common.come_objective_step`,
which takes only the model, the current batch and the class count - no label,
pseudo-label, teacher, memory or source anchor.

## Per valid outer batch

```text
objective_call_count = 1
backward_calls       = 1
optimizer_step_count = 1     (masked, for the sparse variants)
scheduler_step_count = 0     (constant LR, no scheduler exists)
omega_writeback_count = 0
native_ema_commit_count = 0
```

An actual `BS=1` online batch is rejected before the objective, the selector,
the optimizer and PU, with `invalid_reason: stream_protocol_mismatch`.

## CLI

Validate one condition without creating artifacts:

```bash
python -m transformer_come transfer --variant candidate_dense \
  --dataset office31 --source dslr --target amazon \
  --device cuda --dry-run
```

One sparse condition:

```bash
python -m transformer_come transfer --variant group_saliency --budget 0.001 \
  --dataset office31 --source dslr --target amazon --device cuda
```

`matrix` takes `--variants` (`all` or a comma-separated list) and schedules one
process per GPU, one condition at a time per GPU. Each variant gets its own
timestamped run root under `results/transformer_come_otta_<variant>/`;
`--output-root` overrides that root and therefore requires a single variant.
`finalize --run-root` re-aggregates a Random run from completed children.

## Tests

```bash
PY=/home/nas3/biod/wangkangyi/envs/lbi/bin/python
for t in objective substrate_identity source_identity full_dense \
         candidate_dense sparse_optimizer group_magnitude group_random \
         group_saliency stream_contract; do
  $PY tests/transformer_come_${t}_test.py || break
done
```

All are CPU-only; none touches a GPU and none runs an experiment. The runner
tests are synthetic but use a fixture carrying the real candidate tensor
shapes, so the real group partition, exact-K masks, strict masked AdamW and
saliency scoring are exercised end to end.
`transformer_come_source_identity_test.py` is the exception that reads the
actual source checkpoints (CPU forward only) and skips cleanly if they are
absent.

The expensive Saliency state-hash audit is off by default. Turn it on for
correctness runs:

```bash
COME_SELECTION_STATE_AUDIT_BATCHES=5 $PY -m transformer_come transfer \
  --variant group_saliency --budget 0.001 ...
```

## Launching

One script per variant under `transformer_come/scripts/`, each using the matrix
scheduler: one process per GPU, one condition at a time per GPU, artifacts
under `results/transformer_come_otta_<variant>/`.

```bash
./transformer_come/scripts/run_full_dense.sh        # 7 conditions
./transformer_come/scripts/run_candidate_dense.sh   # 7 conditions
./transformer_come/scripts/run_group_magnitude.sh   # 3 rho x 7 = 21 conditions
./transformer_come/scripts/run_group_saliency.sh    # 3 rho x 7 = 21 conditions
./transformer_come/scripts/run_group_random.sh      # 21 conditions x 3 children = 63 runs
```

Each script takes environment overrides (defaults shown):

| variable | default | meaning |
|---|---|---|
| `COME_<VARIANT>_GPUS` | `0,1,2,3,4,5,6,7` | physical GPU ids, one process each |
| `COME_<VARIANT>_DATASETS` | `all` | `all`, `office31` or `visda-c` |
| `COME_<VARIANT>_RHOS` | `all` | sparse only: `all` = 0.0005,0.001,0.002 |
| `COME_<VARIANT>_OUTPUT_ROOT` | `results/transformer_come_otta_<variant>` | artifact root |
| `COME_DRY_RUN` | `0` | `1` validates assets/identities, touches no GPU |
| `COME_ALLOW_BUSY_GPUS` | `0` | `1` overrides the one-process-per-GPU guard |

`<VARIANT>` is `FULL_DENSE`, `CANDIDATE_DENSE`, `GROUP_RANDOM`,
`GROUP_MAGNITUDE` or `GROUP_SALIENCY`.

The scripts refuse to start if any target GPU already has a compute process,
because sharing a GPU makes the measured runtime formally non-comparable
(protocol section 18). Restrict the `_GPUS` list to idle GPUs, or override
with `COME_ALLOW_BUSY_GPUS=1` and record `runtime_comparable=false`.

Sizing, from the measured SHOT baselines on the same RTX 3090s: an Office
condition is ~30 s wall; the VisDA condition is ~4 min and peaks at ~15 GB
allocated (BS 256), so VisDA needs an effectively idle card. Random runs its
three children sequentially inside one process, so its VisDA condition is
about three times a single stream.

## Status

**P0-P2 implementation and CPU contracts complete; P4 GPU real-data smoke
pending.** This is not a full acceptance of COME-Transformer.

Verified so far (CPU): the objective contracts (C02-C08), substrate identity
against SHOT (Contract A/B), the dense and sparse per-batch counts, scope and
off-mask/off-scope exactness, Random's three real children, Magnitude's static
W0 support, Saliency's single objective and gradient reuse with its RNG/state
audits, the singleton guard, target-label isolation, and - on the actual
source W0 - bit-exact SHOT/COME initial logits plus bad-source rejection (the
CPU half of C01).

Still required before anything downstream, per protocol section 22.2 P4:

- [ ] C01 on real target batches through the real loader (actual W0, GPU).
- [ ] Single-GPU 5-valid-batch smoke on Office D->A and on VisDA-C: counts,
      finiteness, scope, state transitions and numerics.
- [ ] Amazon tail test on the real stream: actual `43x64 + 65`, PU/FO sample
      counts, no singleton batch.
- [ ] CUDA-specific contracts: CUDA RNG audit around Saliency selection, real
      CUDA AdamW off-mask exactness, non-AMP FP32.
- [ ] Efficiency runs with `COME_SELECTION_STATE_AUDIT_BATCHES` unset (no hash
      instrumentation), after the audited correctness smoke has passed.

P3 (LBI), P5 (search / formal runs) and every matrix run remain unstarted.
