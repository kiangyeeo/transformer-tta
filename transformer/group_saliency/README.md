# DeiT-S dynamic structural-group Saliency SHOT-OTTA

This package implements the controlled `group_saliency` baseline specified by
`AGENTS.md`. It runs the six Office-31 transfers and VisDA-C
`train -> validation` from the four frozen local source checkpoints.

## Frozen behavior

- formal stream and augmentation seed `2026`, one pass, `drop_last=false`;
- Office batch size 64 and VisDA-C batch size 256;
- causal current-batch SHOT loss `0.3*pseudo + ent + div`;
- AdamW with LR `1e-5`, betas `(0.9,0.999)`, eps `1e-8`, weight decay `0.01`;
- only qkv/proj/fc1/fc2 weights in blocks 9, 10, and 11 are candidates;
- 6912 global paired QK/VO/FFN groups of 768 scalars each;
- budgets `0.0005/0.001/0.002` select exactly `3/6/13` groups by floor;
- after every backward, score each paired group by the paired-group L2 norm of
  `(weight * gradient)` and
  refresh a global top-K mask before the single AdamW step;
- current off-mask values are restored and their AdamW `exp_avg/exp_avg_sq`
  states are cleared; historical parameter updates are not reverted;
- PU is a separate post-update same-batch read-only forward;
- FO is an independent frozen full-target CenterCrop pass.

Target labels are used only for PU and FO metrics. No adapted model or stream
checkpoint is saved.

## Commands

From the repository root, validate the implementation with the CPU/synthetic
test:

```bash
/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  tests/transformer_group_saliency_test.py
```

Validate the config, source checkpoints, hashes, lists, class mappings, budgets,
and all 21 scientific identities without creating a run directory:

```bash
/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.group_saliency matrix \
  --datasets all --budgets all \
  --devices 0,1,2,3,4,5,6,7 --dry-run
```

Run one condition with visible online and FO tqdm bars:

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.group_saliency transfer \
  --dataset office31 --source amazon --target dslr \
  --budget 0.0005 --device cuda
```

Run all 21 conditions on eight GPUs, one process per GPU:

```bash
GROUP_SALIENCY_GPUS=0,1,2,3,4,5,6,7 \
  bash transformer/group_saliency/run_all.sh
```

Subset examples:

```bash
/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.group_saliency matrix \
  --datasets office31 --budgets all --devices 0,1,2,3,4,5

/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.group_saliency matrix \
  --datasets visda-c --budgets 0.001 --devices 0
```

The default matrix is `0.0005,0.001,0.002`. Custom structural rho values are
also accepted and use `floor(rho * 6912)` groups:

```bash
/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.group_saliency matrix \
  --datasets all --rho 0.005 --devices 0,1
```

## Results

The default root is:

```text
/home/nas3/biod/wangkangyi/results/transformer_otta_group_saliency/
```

A completed matrix contains `aggregate.json`, `results.csv`,
`visda_per_class.csv`, condition logs, and per-condition configs, manifests,
`metrics.jsonl`, and `summary.json`. Online records include the complete dynamic
group ids, mask hash, selected scores, block/kind counts, loss, PU, runtime, and
GPU memory. Office aggregation is the equal-weight mean of six sample-level
transfer accuracies. VisDA primary PU/FO values are fixed-12-class macro
accuracies, with all class accuracies retained separately.
