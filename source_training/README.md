# DeiT-S source-model training

This directory trains the complete non-distilled
`deit_small_patch16_224.fb_in1k` source model. It constructs timm with
`pretrained=False`, strictly loads the local ImageNet `model.safetensors`, and
uses source labels only. VisDA-C target-validation labels are never loaded.

## Recommended: use all eight GPUs

The launcher runs four jobs concurrently. Amazon, DSLR, Webcam, and VisDA
receive GPU pairs `0,1`, `2,3`, `4,5`, and `6,7`, respectively. Each job uses
DataParallel with the configured global batch size 64, so the learning rates
do not change.

```bash
cd /home/nas3/biod/wangkangyi/transformer-tta
bash source_training/run_all_sources_8gpu.sh
```

Per-job logs are written under:

```text
/home/nas3/biod/wangkangyi/results/transformer_source_training_logs/
```

The original sequential launcher remains available:

```bash
bash source_training/run_all_sources.sh 0
```

## Current recipe

- deterministic stratified 90/10 source split, seed 2026;
- full-model AdamW fine-tuning with label smoothing and cosine warmup;
- Office-31: 100 epochs; VisDA-C: 10 epochs; global batch size 64;
- AMP on CUDA and tqdm progress for training and validation;
- deterministic CuBLAS plus math-SDP attention;
- in-process data loading (`workers=0`) to avoid the server's non-fatal NFS
  multiprocessing cleanup tracebacks;
- automatic exact resume from `<domain>.last.pth`.

Best checkpoints are written to the paths required by `AGENTS.md`:

```text
/home/nas3/biod/wangkangyi/checkpoints/source_models/office31/amazon.pth
/home/nas3/biod/wangkangyi/checkpoints/source_models/office31/dslr.pth
/home/nas3/biod/wangkangyi/checkpoints/source_models/office31/webcam.pth
/home/nas3/biod/wangkangyi/checkpoints/source_models/visda-c/train.pth
```

The `.last.pth` file is only a resumable training state and must not be used as
the formal OTTA `W0`.
