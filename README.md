# ICLR 2027 — Code-Only Package

This package contains the implementation, runnable configurations, experiment definitions, utility scripts, and tests for the SHOT-OTTA / Split-LBI project.

Intentionally excluded: all experimental outputs and checkpoints, result tables, run logs, launcher logs, historical plans, status files, Codex prompts, Python caches, and Git metadata.

## Layout

- `train.py`, `shot_otta/`, `core/lbi/`, and `visda_otta/`: target-domain online adaptation and Split-LBI implementation.
- `source_training/`: SHOT source-domain trainer used to create the compatible `source_F.pt`, `source_B.pt`, and `source_C.pt` checkpoints.
- `configs/` and `experiments/`: runnable configuration and experiment-matrix files.
- `tools/` and `scripts/`: launch, summarization, and maintenance utilities.
- `tests/`: smoke and engineering tests.

## Setup

Create a Python environment and install the dependencies in `requirements.txt`. The runtime also requires separately supplied datasets and source-model checkpoints; they are not distributed in this code-only package.

Set their locations with `--data-root` and `--source-checkpoint-root`, or edit the corresponding entries in `configs/`.

## Quick configuration check

From the repository root:

```bash
python train.py --config configs/shot_otta.yaml --variant full_dense --dry-run
```

The dry run only parses the effective configuration. A real run additionally needs the external datasets and source checkpoints.

## Train source models

`source_training/image_source.py` trains a source model and writes checkpoints
that the OTTA code loads. For example, to train the Office source model for
Amazon (`--s 0`):

```bash
cd source_training
python image_source.py \
  --data-root /path/to/dataset-root \
  --dset office --s 0 --t 1 --da uda \
  --output /path/to/source-checkpoints \
  --net resnet50 --classifier bn --layer wn
```

This produces `/path/to/source-checkpoints/uda/office/A/source_F.pt`,
`source_B.pt`, and `source_C.pt`. Set the target-adaptation config's
`source_checkpoint.root` to `/path/to/source-checkpoints`.
