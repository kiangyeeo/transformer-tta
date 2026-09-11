# IST-Transformer non-LBI baselines

This package implements the non-LBI portion of the 2026-09-09 IST Transformer
protocol for the source-trained, non-distilled DeiT-S model.

Supported variants are full_dense, candidate_dense, group_random,
group_magnitude, and group_saliency. Group-LBI is deliberately unavailable; no
default LBI tuple or dense fallback is provided.

## Architecture and state mapping

- Source loading, manifest/SHA verification, and local-only model construction
  are reused from transformer.source_only.
- The PLCA feature is exactly the 384-dimensional input to the existing direct
  classifier: forward_head(forward_features(x), pre_logits=True).
- Candidate tensors, paired QK/VO/FFN groups, magnitude/saliency ranking, and
  strict masked AdamW are imported from the existing Transformer substrate.
- The model remains in eval mode during both adaptation and evaluation.
- Each processed raw outer batch executes 1+8 views, pre-inference once,
  past-only PLCA once, memory commit once, a fixed CE+KL task, full native
  traversal, EMA once, and read-only PU.

The target label is retained by the runner only for PU/FO accounting. It is not
accepted by the view materializer, fixed task, PLCA, memory, selector, or
optimizer helpers.

## CLI

Validate one condition without creating artifacts:

    python -m transformer_ist transfer --variant candidate_dense \
      --dataset office31 --source dslr --target amazon \
      --device cuda --dry-run

Run a non-formal five-batch smoke:

    python -m transformer_ist transfer --variant group_saliency --budget 0.001 \
      --dataset office31 --source dslr --target amazon \
      --device cuda --debug-max-outer-batches 5 \
      --output-dir /home/nas3/biod/wangkangyi/results/transformer_ist_smoke/example

The matrix command exists for a later explicitly authorized formal launch. This
implementation does not invoke it automatically.

## Verification

CPU contracts:

    python tests/transformer_ist_contract_test.py
    python tests/transformer_ist_runner_test.py

The pre-existing Transformer source/dense/selector contracts must remain
unchanged. Non-LBI runs never save adapted models or stream-resume checkpoints.
