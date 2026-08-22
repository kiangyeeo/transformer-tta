# DeiT-S source-only OTTA baseline

This folder contains the complete no-adaptation baseline required by
`AGENTS.md`. It evaluates the four source checkpoints over the six Office-31
transfers and VisDA-C `train -> validation`.

Protocol behavior:

- formal seed is exactly `2026`;
- the online target stream is one fixed seed-2026 random permutation;
- online inputs use FC-aligned bilinear
  `Resize(256x256) -> RandomCrop(224) -> RandomHorizontalFlip`;
- each online batch receives a separate read-only PU forward after zero update;
- FO is a second, independent complete pass with bilinear `Resize(256x256)`
  and center crop;
- PU and FO both apply `ToTensor` and ImageNet normalization;
- online/PU and FO use the same dataset-specific batch size: Office `64`,
  VisDA-C `256` (the FC runner's FO-only `3x` batch is intentionally not used);
- `drop_last=false`, including a size-one tail batch;
- Office PU/FO is accumulated as total correct / total samples;
- the Office summary is the equal-weight mean of six transfer accuracies;
- VisDA PU/FO is fixed-12-class macro accuracy, with overall, worst-class,
  class standard deviation, and all 12 class accuracies retained;
- source checkpoints are constructed with `pretrained=False` and loaded only
  from the local versioned `.pth` files; `*.last.pth` is rejected;
- the model is fully frozen and its state hash is checked before the stream,
  after PU, and after FO.

PU and FO need not be numerically identical even though the model is unchanged:
PU uses the seeded online random augmentation, while FO uses center crop. Both
passes are deterministic on a fixed environment.

## Run commands

From the repository root, validate one condition and all of its local assets
without running inference:

```bash
/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.source_only transfer \
  --dataset office31 --source amazon --target dslr \
  --device cuda --dry-run
```

Run the complete seven-transfer matrix on seven dedicated GPUs (one process
per GPU):

```bash
bash transformer/source_only/run_all.sh
```

Select a different set of GPU ids without editing the script:

```bash
SOURCE_ONLY_GPUS=1,2,3,4,5,6,7 \
  bash transformer/source_only/run_all.sh
```

Run all transfers sequentially on one GPU:

```bash
/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.source_only matrix \
  --datasets all --devices 0
```

Run one transfer directly with visible tqdm PU/FO bars:

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.source_only transfer \
  --dataset visda-c --source train --target validation --device cuda
```

The matrix launcher shows a seven-transfer progress bar. Child PU/FO tqdm bars
are retained in `logs/*.log`, so they can be followed with `tail -F`. A
successful run produces:

```text
source_only_seed2026_<UTC>/
  matrix.json
  aggregate.json
  results.csv
  logs/*.log
  results/<dataset>/<source-target>/
    effective_config.yaml
    manifest.json
    metrics.jsonl
    summary.json
```

`aggregate.json` is the main report. Transfer `summary.json` files contain the
full per-class PU/FO arrays and name-indexed mappings.
