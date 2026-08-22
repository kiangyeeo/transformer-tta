# DeiT-S structural-group Magnitude SHOT-OTTA

This package implements the controlled `group_magnitude` baseline frozen by
`AGENTS.md`. It uses the Amazon, DSLR, Webcam, and VisDA synthetic-train source
checkpoints for all six Office-31 transfers and VisDA-C `train -> validation`.

## Frozen scientific behavior

- formal target-stream and augmentation seed: `2026`;
- one pass, no replay, `drop_last=false`;
- Office batch size 64; VisDA-C batch size 256;
- causal current-batch SHOT loss: `0.3*pseudo + ent + div`;
- AdamW: LR `1e-5`, betas `(0.9,0.999)`, eps `1e-8`, weight decay `0.01`;
- model remains in eval mode and the classifier/head stays frozen;
- candidate weights are only qkv/proj/fc1/fc2 in blocks 9, 10, and 11;
- 6912 global paired structural groups, each with exactly 768 scalars;
- budgets `0.005/0.01/0.02` use floor and select exactly `34/69/138` groups;
- group scores are paired-group L2 norms computed once from source W0 on CPU
  in float64; stable ties are resolved by ascending canonical group id;
- the top-K structural mask is static for the complete target stream;
- off-mask values and AdamW `exp_avg/exp_avg_sq` remain frozen/zero;
- PU is a separate post-update same-batch read-only forward;
- FO is an independent frozen full-target CenterCrop pass.

Target labels are used only for PU/FO metrics. No adapted model or stream
checkpoint is saved.

## Environment

Run from the repository root:

```bash
cd /home/nas3/biod/wangkangyi/transformer-tta

export HF_HOME=/home/nas3/biod/wangkangyi/hf-cache
export TORCH_HOME=/home/nas3/biod/wangkangyi/hf-cache/torch
export PIP_CACHE_DIR=/home/nas3/biod/wangkangyi/pip-cache
export CONDA_PKGS_DIRS=/home/nas3/biod/wangkangyi/conda-pkgs
export TMPDIR=/home/nas3/biod/wangkangyi/tmp/transformer_group_magnitude
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$TMPDIR"
```

## Validate without running an experiment

Run the CPU/synthetic protocol test:

```bash
/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  tests/transformer_group_magnitude_test.py
```

Validate all 21 conditions, list files, class mappings, checkpoint manifests,
actual checkpoint SHA-256 values, budgets, and scientific identities. Dry-run
does not construct a model or create a result directory:

```bash
/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.group_magnitude matrix \
  --datasets all --budgets all \
  --devices 0,1,2,3,4,5,6,7 --dry-run
```

Validate one condition and print its resolved config:

```bash
/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.group_magnitude transfer \
  --dataset office31 --source amazon --target dslr \
  --budget 0.005 --device cuda --dry-run
```

## Formal commands

Run one complete condition with visible online and FO tqdm progress bars:

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.group_magnitude transfer \
  --dataset office31 --source amazon --target dslr \
  --budget 0.005 --device cuda
```

Run all 21 conditions on eight GPUs, one experiment process per GPU:

```bash
GROUP_MAGNITUDE_GPUS=0,1,2,3,4,5,6,7 \
  bash transformer/group_magnitude/run_all.sh
```

Run only Office-31, only VisDA-C, or a budget subset:

```bash
/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.group_magnitude matrix \
  --datasets office31 --budgets all \
  --devices 0,1,2,3,4,5,6,7

/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.group_magnitude matrix \
  --datasets visda-c --budgets all --devices 0,1,2

/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.group_magnitude matrix \
  --datasets all --budgets 0.005,0.02 \
  --devices 0,1,2,3,4,5,6,7
```

The matrix process displays condition-level tqdm progress. Each subprocess's
online and FO batch bars are stored in `logs/*.log`; use `tail -F` on one log
to follow it.

## Results

Default root:

```text
/home/nas3/biod/wangkangyi/results/transformer_otta_group_magnitude/
```

A successful complete matrix creates:

```text
group_magnitude_seed2026_<UTC>/
  matrix.json
  aggregate.json
  results.csv
  visda_per_class.csv
  logs/*.log
  results/rho-<budget>/<dataset>/<source-target>/
    effective_config.yaml
    manifest.json
    mask.json
    metrics.jsonl
    summary.json
```

`mask.json` records the source checkpoint hash, scoring definition and dtype,
score-vector hash, top-K boundary, selected group ids/scores, and block/kind
counts. `aggregate.json` contains every transfer and the six-transfer Office
equal mean. VisDA's primary `PU-Acc`/`FO-Acc` are fixed-12-class macro scores;
`visda_per_class.csv` contains PU and FO accuracy for all 12 classes at every
budget.
