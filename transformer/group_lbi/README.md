# DeiT-S structural Group Split-LBI SHOT-OTTA

This package implements the Transformer main experiment defined by `AGENTS.md`.
It runs Group Split-LBI on the six Office-31 transfers and VisDA-C
`train -> validation`, always from the four frozen local source checkpoints.

## Scientific behavior

- seed 2026 fixed random target order and augmentation stream;
- one pass, `drop_last=false`, including size-one tail batches;
- causal current-batch SHOT objective `0.3*pseudo + ent + div`;
- only qkv/proj/fc1/fc2 weights in blocks 9, 10, and 11 are candidates;
- 6912 paired QK/VO/FFN groups, each containing 768 unique scalars;
- floor budgets `0.005/0.010/0.020 -> K=34/69/138`;
- normalized group support `||gamma_g||_2/sqrt(768) >= tau_g`;
- corrected old-state Split-LBI update and strict last-feasible rollback;
- realized support is allowed to remain below K and is never filled or trimmed;
- masked-delta Stage-2 initialization, one fixed-LR strict masked AdamW step;
- persistent omega update after Stage 2, then separate post-update PU;
- independent frozen CenterCrop FO pass.

The profiles in `config.yaml` are deliberately marked `provisional_default`.
They make the complete pipeline runnable, but must be retuned and frozen before
the results are treated as formal paper results. Office profiles are keyed only
by dataset and budget, so one tuple is necessarily shared by all six transfers.

## Validation and runs

Run the CPU/synthetic protocol test (it does not open Office/VisDA images):

```bash
/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  tests/transformer_group_lbi_test.py
```

Validate all source assets, hashes, profiles, budgets, and 21 experiment
identities without loading images or running adaptation:

```bash
/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.group_lbi matrix \
  --datasets all --budgets all \
  --devices 0,1,2,3,4,5,6,7 --dry-run
```

Run one condition with online/Stage-1 and FO tqdm progress:

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.group_lbi transfer \
  --dataset office31 --source amazon --target dslr \
  --budget 0.005 --device cuda
```

Temporarily override a selected dataset/budget profile for tuning:

```bash
/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.group_lbi matrix \
  --datasets office31 --budgets 0.005 --devices 0,1,2,3,4,5 \
  --lbi-alpha 0.2 --lbi-kappa 1.0 --lbi-nu 0.5 \
  --lbi-omega 0.1 --lbi-prox-lambda 1.0 --lbi-tau-g 1e-4 \
  --lbi-stage1-max-steps 3000 --lbi-stage2-lr 1e-5
```

Run the complete 21-condition matrix with one process per GPU:

```bash
GROUP_LBI_GPUS=0,1,2,3,4,5,6,7 \
  bash transformer/group_lbi/run_all.sh
```

Resume one interrupted condition from its last completed batch:

```bash
/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.group_lbi transfer \
  --resume-run-dir /absolute/path/to/interrupted/condition
```

Resume a matrix, skipping completed conditions and restoring conditions that
still contain a stream checkpoint:

```bash
/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.group_lbi matrix \
  --resume-run-root /absolute/path/to/interrupted/matrix \
  --devices 0,1,2,3,4,5,6,7
```

Checkpointing is enabled by default. Use `--no-stream-checkpoint` only for a
short disposable pilot. Checkpoint I/O is outside adaptation/PU timing, and the
engineering checkpoint is removed after successful completion; no final
adapted-model checkpoint is retained.

## Results

The default root is:

```text
/home/nas3/biod/wangkangyi/results/transformer_otta_group_lbi/
```

Each condition contains its effective config, manifest, raw batch JSONL and
summary. A completed matrix adds `aggregate.json`, `results.csv`,
`tuning_diagnostics.csv`, `visda_per_class.csv`, logs, and matrix status. Office PU/FO uses sample-level
correct/total and the Office summary is the equal-weight mean of six transfers.
VisDA PU/FO uses fixed-12-class macro accuracy and retains every class result,
overall accuracy, worst-class accuracy, and class standard deviation.

`summary.json -> selection` and `tuning_diagnostics.csv` retain the tuning
health metrics over online batches: mean/min utilization, utilization >=90%
and >=95% rates, exact 3000-step and configured-step-cap hit rates, Stage-1
mean/max steps, rollback rate, average selected groups, budget-violation rate,
and failure rate. For Office, `aggregate.json` and the CSV include an
`OFFICE_POOLED` row computed by pooling all online batches from the six
transfers; accuracy remains the required equal-transfer mean and is never
sample- or batch-pooled.

A completed condition must have `budget_violation_rate=0` and
`failure_rate=0`. NaN, Inf, or an online-batch exception aborts the condition
and writes an `online_batch_failure` JSONL record plus failure diagnostics in
the failed `summary.json`; the implementation never skips a failed batch and
continues adaptation.

For a matrix that was already running before these diagnostics were added,
rebuild the aggregate after it completes. The command reads the retained
per-batch `metrics.jsonl`, backfills missing diagnostic fields, and does not run
adaptation again:

```bash
/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.group_lbi summarize \
  --run-root /absolute/path/to/completed/matrix \
  --datasets office31 --budgets all
```
