# DeiT-S structural-group Random SHOT-OTTA

This package implements the controlled `group_random` baseline frozen by
`AGENTS.md`. It uses the four local source checkpoints for all six Office-31
transfers and VisDA-C `train -> validation`.

## Frozen scientific behavior

- formal stream/augmentation seed: `2026`;
- one pass, no replay, `drop_last=false`;
- Office batch size 64; VisDA-C batch size 256;
- causal current-batch SHOT loss: `0.3*pseudo + ent + div`;
- AdamW: LR `1e-5`, betas `(0.9,0.999)`, eps `1e-8`, weight decay `0.01`;
- model remains in eval mode and the head stays frozen;
- candidate weights are only qkv/proj/fc1/fc2 in blocks 9, 10, and 11;
- 6912 global paired structural groups, each with 768 scalars;
- budgets `0.005/0.01/0.02` use floor and therefore select exactly
  `34/69/138` groups;
- three child masks use seeds `202600/202601/202602`;
- each seed uses one fixed permutation, so its support is nested across budgets;
- off-mask values and AdamW `exp_avg/exp_avg_sq` remain exactly zero/frozen;
- PU is a separate post-update same-batch read-only forward;
- FO is an independent frozen full-target pass.

Target labels are used only for PU/FO reporting. No adapted model or stream
checkpoint is saved.

## Validate without running an experiment

The dry-run checks every image list, class mapping, checkpoint manifest, actual
checkpoint SHA-256, transfer, budget, and scientific identity. It does not
construct a model or create a result directory.

```bash
cd /home/nas3/biod/wangkangyi/transformer-tta

/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.group_random matrix \
  --datasets all --budgets all \
  --devices 0,1,2,3,4,5,6,7 --dry-run
```

Validate one condition:

```bash
/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.group_random transfer \
  --dataset office31 --source amazon --target dslr \
  --budget 0.005 --device cuda --dry-run
```

## Recommended execution sequence

First run the CPU/synthetic test, then the dry-run above. Per `AGENTS.md`, run a
single-GPU short-stream pilot before launching the complete matrix. The package
does not silently substitute a short stream for a formal condition, so a pilot
must be prepared as a separately declared non-formal configuration.

Run one complete formal condition with visible online and FO tqdm bars:

```bash
cd /home/nas3/biod/wangkangyi/transformer-tta

export HF_HOME=/home/nas3/biod/wangkangyi/hf-cache
export TORCH_HOME=/home/nas3/biod/wangkangyi/hf-cache/torch
export PIP_CACHE_DIR=/home/nas3/biod/wangkangyi/pip-cache
export CONDA_PKGS_DIRS=/home/nas3/biod/wangkangyi/conda-pkgs
export TMPDIR=/home/nas3/biod/wangkangyi/tmp/transformer_group_random
mkdir -p "$TMPDIR"

CUDA_VISIBLE_DEVICES=0 \
/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.group_random transfer \
  --dataset office31 --source amazon --target dslr \
  --budget 0.005 --device cuda
```

Run all 21 transfer/budget conditions (63 child masks), with one process per
GPU and eight GPUs available to the scheduler:

```bash
cd /home/nas3/biod/wangkangyi/transformer-tta

GROUP_RANDOM_GPUS=0,1,2,3,4,5,6,7 \
  bash transformer/group_random/run_all.sh
```

Run only Office-31 or VisDA-C, or a budget subset:

```bash
/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.group_random matrix \
  --datasets office31 --budgets 0.005,0.01,0.02 \
  --devices 0,1,2,3,4,5,6,7

/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.group_random matrix \
  --datasets visda-c --budgets 0.005 --devices 0
```

The matrix process displays condition-level tqdm progress. Batch-level child
bars are stored in `logs/*.log`; use `tail -F <log>` to follow one condition.

## Results

Default result root:

```text
/home/nas3/biod/wangkangyi/results/transformer_otta_group_random/
```

A complete run creates:

```text
group_random_seed2026_<UTC>/
  matrix.json
  aggregate.json
  results.csv
  mask_results.csv
  visda_per_class.csv
  logs/*.log
  results/rho-<budget>/<dataset>/<source-target>/
    effective_config.yaml
    manifest.json
    summary.json
    mask_{00,01,02}/
      effective_config.yaml
      manifest.json
      mask.json
      metrics.jsonl
      summary.json
```

Office rows report three-mask mean/std for every transfer; the Office summary
is the equal-weight mean of the six transfer-level means. VisDA uses fixed
12-class macro accuracy and `visda_per_class.csv` contains PU/FO mean and mask
standard deviation for every class and budget.
