# DeiT-S full-dense SHOT-OTTA

This package runs the protocol-aligned full-dense Transformer baseline over
the six Office-31 transfers and VisDA-C `train -> validation`.

Frozen behavior:

- formal seed `2026` controls target order and augmentation RNG;
- online inputs use bilinear
  `Resize(256x256) -> RandomCrop(224) -> RandomHorizontalFlip -> ToTensor ->
  ImageNet Normalize`;
- FO inputs use bilinear
  `Resize(256x256) -> CenterCrop(224) -> ToTensor -> ImageNet Normalize`;
- Office uses batch size `64`; VisDA-C uses `256`; FO deliberately uses the
  same batch size rather than the FC ResNet runner's `3x` evaluation batch;
- each online batch receives one AdamW update, followed by a separate
  read-only same-batch PU forward;
- the classifier/head is frozen and every other DeiT parameter is trainable;
- FO freezes the final model and evaluates the complete target set with no
  adaptation;
- Office accuracy is sample-level correct/total and its six-transfer summary
  is an equal-weight transfer mean;
- VisDA accuracy is fixed-12-class macro accuracy, with every class retained;
- no adapted-model or stream-resume checkpoint is written.

## Environment

Run from the repository root:

```bash
export HF_HOME=/home/nas3/biod/wangkangyi/hf-cache
export TORCH_HOME=/home/nas3/biod/wangkangyi/hf-cache/torch
export PIP_CACHE_DIR=/home/nas3/biod/wangkangyi/pip-cache
export CONDA_PKGS_DIRS=/home/nas3/biod/wangkangyi/conda-pkgs
export TMPDIR=/home/nas3/biod/wangkangyi/tmp
```

## Validate without running the experiment

Validate all seven tasks, local lists, manifests, and actual checkpoint hashes:

```bash
/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.full_dense matrix \
  --datasets all --devices 0 --dry-run
```

Validate one transfer and print its complete resolved configuration:

```bash
/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.full_dense transfer \
  --dataset office31 --source amazon --target dslr \
  --device cuda --dry-run
```

## Formal commands

Run all seven transfers sequentially on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.full_dense matrix \
  --datasets all --devices 0
```

When using the matrix command, `--devices` contains physical GPU ids and the
launcher sets `CUDA_VISIBLE_DEVICES` separately for every child. For example,
run up to seven transfers concurrently:

```bash
/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.full_dense matrix \
  --datasets all --devices 0,1,2,3,4,5,6
```

Or use the wrapper (override `FULL_DENSE_GPUS` as needed):

```bash
FULL_DENSE_GPUS=0,1,2,3,4,5,6 bash transformer/full_dense/run_all.sh
```

Run one transfer with visible per-batch tqdm bars:

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.full_dense transfer \
  --dataset visda-c --source train --target validation --device cuda
```

The matrix launcher shows transfer-level progress. Each child's batch-level
tqdm output is stored in `logs/*.log` and can be followed with `tail -F`.

Successful all-dataset output includes:

```text
full_dense_seed2026_<UTC>/
  matrix.json
  aggregate.json
  results.csv
  visda_per_class.csv
  logs/*.log
  results/<dataset>/<source-target>/
    effective_config.yaml
    manifest.json
    metrics.jsonl
    summary.json
```
