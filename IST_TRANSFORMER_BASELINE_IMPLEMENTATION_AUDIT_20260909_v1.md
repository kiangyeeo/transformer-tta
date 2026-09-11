# IST Transformer non-LBI implementation audit

Implementation revisions are ist_transformer_baseline_20260909_v1 for dense
and ist_transformer_sparse_lbi_20260909_v1 for sparse controls.

Implemented variants are full_dense, candidate_dense, group_random,
group_magnitude, and group_saliency. Group-LBI, LBI tuple handling, parameter
search, and formal experiment execution are intentionally absent.

## Protocol alignment

- Uses the existing local source-trained DeiT loader and source manifest/SHA
  checks. No pretrained download, last checkpoint, new bottleneck, or alternate
  classifier path is introduced.
- Uses the classifier-input 384-D pre-logits representation and validates its
  logits against model(x) on a synthetic startup input.
- Keeps the model in eval mode and FP32. Full dense freezes only the head;
  controlled variants expose exactly the existing 12 candidate tensors.
- Builds one cached reference and eight independently augmented adaptation
  views from one raw decode. PLCA and memory commit occur once per processed
  outer batch with view-level hard/soft targets.
- Native IST traverses the complete 8B task once using fixed scientific inner
  batches. Micro-chunks accumulate a sample-weighted gradient and do not add
  optimizer steps.
- Non-LBI writeback is exactly one EMA. Sparse paths additionally restore
  off-mask coordinates exactly after EMA and clear off-mask Adam moments.
- Random runs three fresh child trajectories; Magnitude is source-static;
  Saliency scores the complete fixed task once per outer batch.
- PU occurs after EMA on the cached reference view. PU and FO are guarded by
  model/optimizer/memory/EMA/RNG fingerprints.
- Amazon uses 43 batches of 64 followed by one batch of 65. An unexpected
  singleton raises before view construction or any method state transition.

## Remaining gate

Formal matrices remain unlaunched. Group-LBI, its six dataset-by-budget tuples,
and its search protocol remain deliberately unavailable; neither a stub result
nor a dense fallback exists.

## Verification evidence (2026-09-10)

The repository base commit recorded during verification was
`d97860a8350a09543e9be36de5b959c29691107d`; the worktree was intentionally
dirty because this implementation was not committed by the agent.

The following CPU/synthetic suites passed in the `lbi` environment:

- `tests/transformer_ist_contract_test.py`, including raw decode/view/RNG,
  Amazon `43x64+65`, strict config/hash/LBI rejection, CE+KL, causal FIFO
  memory, EMA, and numerical agreement with the refined FC PLCA reference.
- `tests/transformer_ist_runner_test.py`, including the native 8/9/8 step
  contracts, micro-chunk accumulation, PU/FO read-only state hashes, three
  real Random child orchestration, and target-label permutation producing the
  same adapted model/memory/RNG trajectory.
- All six existing Transformer source/full/candidate/Random/Magnitude/Saliency
  contract scripts. No existing `transformer/**` or FC source was modified.

The matrix dry-run resolved 66 requested top-level Office conditions for the
five in-scope variants (54 Random child executions) and created no experiment
artifacts. CLI/config resolution rejected `group_lbi`.

## GPU smoke evidence (RTX 3090, diagnostic only)

Artifacts are under:

`/home/nas3/biod/wangkangyi/results/transformer_ist_smoke/20260910_v1/`

All runs used fresh local source checkpoints, `formal_protocol=false`, and
`result_validity=diagnostic_incomplete`. Accuracy was not used for selection or
tuning.

| Smoke | Outer batches | Views | Optimizer steps | Selector calls | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|
| Office D->A full_dense | 5 | 2,560 | 40 | 0 | 4,463.7 |
| Office D->A candidate_dense | 5 | 2,560 | 40 | 0 | 1,511.2 |
| Office D->A group_magnitude, rho=.001 | 5 | 2,560 | 40 | static W0 once | 1,538.5 |
| Office D->A group_saliency, rho=.001 | 5 | 2,560 | 40 | 5 | 1,552.5 |
| Office D->A group_random, rho=.001 | 5 x 3 children | 2,560/child | 40/child | static once/child | 1,538.5 max |
| VisDA train->validation saliency, rho=.001 | 1 | 2,048 | 8 | 1 | 2,548.2 |

For every single trajectory, pre-inference, PLCA, memory commit, EMA, and PU
counts equaled the number of processed outer batches. Adapter/model logits had
zero observed maximum absolute error; the feature point was `[N,384]`.
Candidate/sparse scope contained exactly 12 tensors and 5,308,416 scalars.
Every rho=.001 sparse mask contained exactly 6 groups. Magnitude recorded CPU
float64 source-W0 scoring. Random used seeds 202600/202601/202602 and all three
children had identical raw/view/inner-order hashes while starting from fresh
state. PU/FO state hashes and all off-scope exact checks passed.

Office used source checkpoint
`/home/nas3/biod/wangkangyi/checkpoints/source_models/office31/dslr.pth`
with SHA-256
`8bb6d57e04d653e14b05955eb40a3a1f600634281a824ca3a57356a78de9566e`.
VisDA used
`/home/nas3/biod/wangkangyi/checkpoints/source_models/visda-c/train.pth`
with SHA-256
`c68f9b0ed7bff7a98ef6e9413015d4df3f633e9d684802adaf0caa5d1fd05bfd`.

One initial Random parent aggregation failed after all three children completed
because child summaries omitted their mask index/seed. The runner was fixed,
its CPU orchestration contract was rerun, and a new fresh-W0 three-child smoke
completed successfully at `office_group_random_rho001_5b_retry/`. The failed
diagnostic directory was retained rather than being repurposed as a result.

DataLoader worker shutdown emitted non-fatal NFS temporary-directory cleanup
warnings in this server environment. All smoke parent processes exited
successfully except the explicitly retained pre-fix Random parent, and the
completed JSON manifests/summaries report valid state audits. This is an
operational warning, not evidence of a scientific-state mutation.
