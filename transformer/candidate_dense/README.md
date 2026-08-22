# DeiT-S candidate-dense SHOT-OTTA

This package runs the controlled candidate-dense Transformer baseline from
`AGENTS.md`. It evaluates the four source checkpoints over all six Office-31
transfers and VisDA-C `train -> validation`.

Only these 12 fused/native DeiT weight tensors are trainable:

```text
blocks.{9,10,11}.attn.qkv.weight
blocks.{9,10,11}.attn.proj.weight
blocks.{9,10,11}.mlp.fc1.weight
blocks.{9,10,11}.mlp.fc2.weight
```

The loader rejects a model unless these tensors have the expected DeiT-S
shapes and contain exactly `5,308,416` scalars. All biases, LayerNorms,
tokens, embeddings, blocks 0-8, final norm, and classifier/head stay frozen.

Protocol behavior:

- formal seed `2026` fixes target order and augmentation RNG;
- one target-stream pass, no replay, `drop_last=false`;
- Office batch size `64`, VisDA-C batch size `256`;
- one fixed AdamW update per online batch with LR `1e-5`;
- causal current-batch SHOT objective `0.3*pseudo + ent + div`;
- a separate post-update, same-batch, read-only PU forward;
- an independent frozen full-target FO pass;
- Office uses sample-level correct/total and a six-transfer equal mean;
- VisDA uses fixed-12-class macro accuracy and reports every class;
- no adapted model or stream-resume checkpoint is written.

## Environment

Run commands from the repository root:

```bash
export HF_HOME=/home/nas3/biod/wangkangyi/hf-cache
export TORCH_HOME=/home/nas3/biod/wangkangyi/hf-cache/torch
export PIP_CACHE_DIR=/home/nas3/biod/wangkangyi/pip-cache
export CONDA_PKGS_DIRS=/home/nas3/biod/wangkangyi/conda-pkgs
export TMPDIR=/home/nas3/biod/wangkangyi/tmp/transformer_candidate_dense
mkdir -p "$TMPDIR"
```

## Validate without running the experiment

This validates all seven tasks, list files, manifests, and actual checkpoint
hashes. It does not construct a model or create a result directory.

```bash
/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.candidate_dense matrix \
  --datasets all --devices 0 --dry-run
```

Validate one transfer and print its resolved scientific configuration:

```bash
/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.candidate_dense transfer \
  --dataset office31 --source amazon --target dslr \
  --device cuda --dry-run
```

## Formal commands

Run all seven transfers sequentially on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.candidate_dense matrix \
  --datasets all --devices 0
```

Run one process on each of seven GPUs:

```bash
CANDIDATE_DENSE_GPUS=0,1,2,3,4,5,6 \
  bash transformer/candidate_dense/run_all.sh
```

Run only Office-31 or only VisDA-C:

```bash
/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.candidate_dense matrix \
  --datasets office31 --devices 0,1,2,3,4,5

/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.candidate_dense matrix \
  --datasets visda-c --devices 0
```

Run a single transfer with visible online and FO tqdm bars:

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.candidate_dense transfer \
  --dataset visda-c --source train --target validation \
  --device cuda
```

The matrix process shows transfer-level progress. Child batch-level tqdm
output is stored under `logs/*.log` and can be followed with `tail -F`.

## Results

The default root is:

```text
/home/nas3/biod/wangkangyi/results/transformer_otta_candidate_dense/
```

A successful all-dataset run creates:

```text
candidate_dense_seed2026_<UTC>/
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

`aggregate.json` contains the six Office transfer PU/FO scores, their
equal-transfer mean, and the VisDA fixed-12 macro PU/FO scores.
`visda_per_class.csv` contains PU and FO accuracy for every VisDA class.

