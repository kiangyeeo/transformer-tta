# NCTTA P1 Dense Implementation Audit

**Date:** 2026-09-06  
**Audited implementation:** `nctta_otta_p1_dense_baseline_20260906_v1`  
**Patched implementation:** `nctta_otta_p1_dense_baseline_20260906_v2`  
**Baseline protocol:** `OTTA_NCTTA_BASELINE_PROTOCOL_20260906_v1`

## Verdict

The NCTTA objective port and the three controlled dense update scopes are scientifically consistent with the frozen P0 contract and with the official `Cevaaa/NCTTA` implementation. No objective-port or trainable-scope correctness bug was found.

One provenance defect was found: artifacts identified a protocol revision that did not correspond to an actual protocol file. This audit adds the concrete baseline protocol, updates the protocol revision, bumps the implementation revision, and adds a protocol-file existence contract.

## Official NCTTA objective audit

PASS:

- current feature and classifier rows are L2-normalized;
- current logits are softmaxed before top-k selection;
- top-k classes are chosen by current predicted probability, matching official code;
- top-k FCA distances are standardized per sample;
- `q_dist = softmax(-standardized_distance)`;
- `q_prob = softmax(log(topk_probability + eps))`;
- hybrid target mixes `q_dist` and `q_prob` using `mix_prob_weight`;
- official active NC branch is InfoNCE-style with cosine similarity and `tau_align=1.0`;
- entropy is per-sample softmax entropy;
- filter is strict `entropy < thre_ent`;
- entropy and predicted-class FCA-distance coefficients use detached values;
- total selected-sample loss matches the official weighted mean;
- current NCTTA objective is recomputed from current model outputs.

The port intentionally does not import TTAB runtime/model selection/Fisher/stochastic restore.

## Feature / classifier geometry audit

PASS.

Current project classifier consumes the post-netB 256-dimensional feature, so the NCTTA feature is correctly defined as:

$$
h=\mathrm{netB}(\mathrm{netF}(x)).
$$

The classifier reference is the effective frozen `netC.fc.weight.detach()` after normal forward semantics. Using the 2048-d `netF.avgpool` output would be dimensionally and semantically wrong for this F/B/C architecture.

Using a frozen **source** feature instead of the current feature would also not be faithful to official NCTTA: official NCTTA captures the feature from the current model forward on every adaptation step and constructs its alignment objective from that current feature/current prediction.

## Dense update-scope audit

### `nctta_full_dense`

PASS. All named parameters in `netF + netB` are trainable; `netC` is frozen. `netF`/`netB` are in train behavior and BN state follows the existing SHOT full-dense semantics.

### `nctta_fc_module_dense`

PASS. Exactly these two full tensors are trainable:

```text
netB.bottleneck.weight
netB.bottleneck.bias
```

Total candidate scalars: `524,544`. All BN modules are eval/frozen. This is **not** BN-only adaptation.

### `nctta_conv_module_dense`

PASS. Exactly the nine full `netF.layer4.*.conv*.weight` tensors are trainable. Total candidate scalars: `12,845,056`. All BN modules are eval/frozen. This is **not** BN-only adaptation.

## Interpretation of the 5-batch D->A smoke collapse

The reported FC/Conv FO values near 3.5% are not explained by an accidental BN-only scope. The implementation really updates the full declared FC/Conv tensors.

The smoke pattern is consistent with aggressive self-reinforcing entropy collapse under this deliberate non-native parameterization:

- FC selected counts: `39, 63, 64, 64, 64`;
- Conv selected counts: `39, 62, 64, 64, 64`.

After the first update almost every sample passes the low-entropy filter, while final accuracy approaches the 31-class chance level. That is a strong collapse signature.

This is plausible because native NCTTA was designed around normalization-parameter adaptation, while the controlled FC/Conv branches give the objective much larger feature-space freedom. In addition, the current P1 uses the SHOT base LR/substrate and the official ImageNet-oriented objective defaults only for correctness smoke. Neither was tuned for Office FC/Conv NCTTA.

Do **not** fix this by replacing the current feature with the source feature; that would define a different source-anchored method. The correct next diagnostic is to add the faithful native norm-only reference and later resolve NCTTA-specific optimization/objective hyperparameters before formal runs.

## Provenance fix

Fixed:

- added `protocol/nctta-otta/OTTA_NCTTA_BASELINE_PROTOCOL_20260906_v1.md`;
- protocol revision is now `OTTA_NCTTA_BASELINE_PROTOCOL_20260906_v1`;
- implementation revision is now `nctta_otta_p1_dense_baseline_20260906_v2`;
- P1 contract now asserts that the referenced protocol file exists.

No NCTTA objective/trainer scientific logic was changed by this patch.

## Regression status

PASS:

- NCTTA P1 contract;
- IST P1 contract;
- IST sparse/LBI contract;
- SHOT protocol alignment;
- implementation revision identity;
- all other runnable CPU tests in the provided archive.

One existing test, `multi_gpu_launcher_path_consistency_test.py`, cannot run from the uploaded archive because the archive explicitly excludes `experiment_logs/` and the test reads historical plan files from that directory. This is an archive-content limitation, not a code failure.
