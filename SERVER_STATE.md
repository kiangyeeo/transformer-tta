# Server State

Last updated: 2026-08-17 (Asia/Shanghai)

This file records dynamic server-side facts that are not represented by files
in the Git working tree. It complements `catalog.md`, which defines the path
layout, and `AGENTS.md`, which defines project and experiment rules.

## Verification Levels

- **User-confirmed:** the user has observed or completed the server-side state.
- **Artifact-verified:** metadata or hashes have been copied into the repository
  or independently inspected by Codex.

The current source-model status is **user-confirmed**, not artifact-verified.
No checkpoint binary is stored in this repository.

## Source Checkpoints

All four DeiT-S source-training runs are reported complete at the output paths
frozen in the source-training configurations:

| Dataset | Source domain | Classes | Best checkpoint path | Status |
| --- | --- | ---: | --- | --- |
| Office-31 | Amazon | 31 | `/home/nas3/biod/wangkangyi/checkpoints/source_models/office31/amazon.pth` | User-confirmed complete |
| Office-31 | DSLR | 31 | `/home/nas3/biod/wangkangyi/checkpoints/source_models/office31/dslr.pth` | User-confirmed complete |
| Office-31 | Webcam | 31 | `/home/nas3/biod/wangkangyi/checkpoints/source_models/office31/webcam.pth` | User-confirmed complete |
| VisDA-C | synthetic `train` | 12 | `/home/nas3/biod/wangkangyi/checkpoints/source_models/visda-c/train.pth` | User-confirmed complete |

These `.pth` files are the selected best `W0` checkpoints. Files such as
`*.last.pth` are resume states and must not be used as `W0` for TTA unless an
explicit recovery procedure proves they match the selected best checkpoint.

## Required Before Formal TTA Runs

The Transformer TTA loader must load the corresponding `.pth` file through
`source_training.deit_model.load_deit_source_checkpoint`; it must not reload the
ImageNet initialization as the task source model. Before a formal experiment
matrix is launched, record from each adjacent manifest:

- checkpoint SHA-256;
- best epoch and source-validation metric;
- source-training seed and effective configuration;
- Python, CUDA, PyTorch, torchvision, timm, and safetensors versions;
- training Git commit and dirty state.

Until those fields are recorded, documentation must say "user-confirmed" and
must not claim byte-level checkpoint reproducibility or locally verified model
quality.

The DeiT TTDA and OTTA source-only evaluators now enforce this boundary at
runtime: they require every adjacent source manifest, bind experiment identity
to the manifest checkpoint SHA-256, and ask the versioned checkpoint loader to
hash and verify the actual `.pth` before inference. These local implementations
have only been tested with synthetic checkpoints; the four real server
artifacts remain user-confirmed until the server-side preflight reads them.

## TTDA Source-only Output

The user confirmed the formal result root for the seven no-adaptation DeiT
TTDA controls as:

`/home/nas3/biod/wangkangyi/results/transformer_ttda_source_only/`

No real evaluation run has been launched by Codex. The implementation provides
a fixed six-direction Office-31 plus one-direction VisDA-C plan, a dry-run
preflight, one-GPU A-to-D pilot command, seven-slot launcher command, and a
strict summary tool.

## OTTA Source-only Output

The new DeiT source-only OTTA control uses the separate default result root:

`/home/nas3/biod/wangkangyi/results/transformer_otta_source_only/`

It runs the same six Office-31 directions and VisDA-C train-to-validation task
as an ordered target stream. It records every incoming batch, retains a
size-one tail batch, and reports macro-per-class `PU-Acc` and `FO-Acc` plus
overall and per-class metrics. FO is a separate full-target prediction pass
after the stream, matching the final-offline protocol. Zero adaptation means
the final model is still `W0`, so the implementation requires the independent
PU/FO predictions and the before/after model hashes to match exactly. No real
OTTA evaluation has been launched by Codex. Trainable DeiT OTTA remains
unimplemented.

## Server Boundary

All server assets remain under `/home/nas3/biod/wangkangyi/`. In particular:

- repository: `/home/nas3/biod/wangkangyi/transformer-tta/`
- datasets: `/home/nas3/biod/wangkangyi/datasets/`
- source checkpoints: `/home/nas3/biod/wangkangyi/checkpoints/source_models/`
- TTDA source-only results: `/home/nas3/biod/wangkangyi/results/transformer_ttda_source_only/`
- OTTA source-only results: `/home/nas3/biod/wangkangyi/results/transformer_otta_source_only/`
- environment: `/home/nas3/biod/wangkangyi/envs/lbi/`
- caches and temporary files: the dedicated paths in `catalog.md`

Local development is limited to code, configuration, documentation, and static
inspection. Package installation, real checkpoint loading, and GPU execution
belong to the server environment.
