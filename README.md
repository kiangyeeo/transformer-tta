# Transformer Split-LBI Test-Time Adaptation

This repository studies sparse update-support discovery for test-time adaptation (TTA) with Split-LBI. The legacy paper applies a SHOT objective and element-wise sparse updates to the 256-dimensional bottleneck fully connected layer of ResNet models. The legacy model factory also supports VGG. The current development goal is architecture-aware Transformer adaptation with a non-distilled, ImageNet-1K-pretrained DeiT-Small/16-224 backbone.

## Read First

- [`AGENTS.md`](AGENTS.md): authoritative project context, current implementation status, engineering boundaries, server constraints, experiment protocol, and unresolved decisions. Codex reads this file automatically from the repository root.
- [`TRANSFORMER_GROUP_SPLIT_LBI_PROPOSAL_DOLLAR_MATH.md`](TRANSFORMER_GROUP_SPLIT_LBI_PROPOSAL_DOLLAR_MATH.md): the proposed Transformer QK, VO, and FFN grouping design.
- [`catalog.md`](catalog.md): server-side paths for the repository, datasets, checkpoints, environments, and caches.
- [`26445_Test_Time_Adaptation_via (1).pdf`](26445_Test_Time_Adaptation_via%20%281%29.pdf): the legacy manuscript covering bottleneck-FC sparse adaptation.

## Current Status

| Component | Status |
| --- | --- |
| ResNet/VGG + SHOT OTTA | Implemented |
| FC scalar Dense/Random/Magnitude/Saliency/Split-LBI | Implemented |
| Dense ResNet convolution updates | Indirectly included in `full_dense` |
| Conv filter/channel Group Split-LBI | Not found in this repository |
| DeiT-S backbone and source training | Not implemented |
| Transformer paired QK/VO/FFN groups | Not implemented |
| TTDA | Not implemented |

This repository is therefore a reusable FC-OTTA engineering foundation, not an already completed FC + Conv + Transformer framework.

Two method-level decisions remain unresolved:

1. The current LBI Stage 2 starts from the current model plus a dense local delta, so the final local update is not guaranteed to be zero outside the discovered mask. The proposal's final equation suggests a strict masked delta, but that differs from the existing engine. In OTTA, even a strict K-group update per batch can accumulate into a source-relative support larger than K over the full stream.
2. Target labels must never enter adaptation. However, the existing analysis scripts rank candidates by target FO accuracy. Before formal experiments, the project must choose an independent validation rule, report every budget, or explicitly disclose oracle-style selection.

## Repository Layout

```text
train.py                 Legacy single-run SHOT-OTTA entry point
shot_otta/               Data, models, losses, trainer, and artifacts
core/lbi/                Existing element-wise Split-LBI implementation
source_training/         SHOT-style ResNet/VGG source trainer
visda_otta/              VisDA-C metrics and helper logic
configs/                 Single-run configurations
experiments/             Experiment matrices
tools/                   Planner, launchers, status checks, and summaries
scripts/                 Maintenance utilities
tests/                   Synthetic smoke and engineering tests
```

The reusable pieces include the SHOT objective, image-list data pipeline, artifact system, experiment planning and launch infrastructure, summaries, and the broad LBI stage lifecycle. Transformer work still requires a DeiT model adapter, source trainer, structural group registry, group proximal operator, matched group selectors, Transformer-aware experiment identity, and TTDA support.

## Installation

Create an isolated Python environment and install the declared dependencies:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` was audited against all Python files in this repository. It includes:

- every third-party package imported by the current legacy code, tools, and tests;
- `timm` and `safetensors`, which are required by the already selected local DeiT-S checkpoint stack.

Versions are intentionally not pinned yet because the server's Python, CUDA, PyTorch, and torchvision compatibility set has not been frozen. Once the server environment is validated, record those versions and create a reproducible lock or environment snapshot.

## Quick Checks

Parse the legacy configuration without accessing datasets or checkpoints:

```bash
python train.py --config configs/shot_otta.yaml --variant full_dense --dry-run
```

Run the current smoke suite on CPU. This is recommended because `tests/lbi_smoke_test.py` has a fake fixture that remains on CPU when CUDA is visible:

```bash
CUDA_VISIBLE_DEVICES=-1 bash -c 'for f in tests/*_test.py; do python "$f" || exit 1; done'
```

## Data and Source Models

The repository does not contain Office-31, VisDA-C, image-list files, or source checkpoints. The legacy loader expects SHOT-style `office/*_list.txt` and `VISDA-C/*_list.txt` files, while the server stores raw images under the `office31/` and `visda-c/` directories documented in `catalog.md`. Image lists must be generated and validated before training or adaptation.

The catalogued `deit_small_patch16_224.fb_in1k/model.safetensors` file is only an ImageNet-1K initialization. It is not an Office-31 or VisDA-C source model. The project still needs:

- three 31-class Office-31 source models trained on Amazon, DSLR, and Webcam;
- one 12-class VisDA-C source model trained on the synthetic `train` domain.

Every selector and budget compared under the same protocol must use the same hashed source model, `W0`.

`source_training/image_source.py` is derived from the SHOT source trainer and writes legacy `source_F.pt`, `source_B.pt`, and `source_C.pt` checkpoints. It currently supports only ResNet and VGG. A DeiT source trainer and a versioned Transformer checkpoint schema are first-stage implementation tasks.

## Server Constraints

The experiment server has eight RTX 3090 GPUs with 24 GB of memory each, but its primary disk is full. Code, data, checkpoints, environments, caches, temporary files, and experiment outputs must remain under:

```text
/home/nas3/biod/wangkangyi/
```

See `catalog.md` for the exact known paths. Model construction must not trigger implicit downloads. Set `HF_HOME`, `TORCH_HOME`, `PIP_CACHE_DIR`, `CONDA_PKGS_DIRS`, and `TMPDIR` to locations under the approved root. The formal output directory has not yet been defined and must be confirmed before large experiment launches.

## Dependency Policy

Every new third-party Python import must be added to `requirements.txt` in the same change. Do not rely on ad hoc packages installed only in the server environment. Transitive dependencies are resolved by pip and should be pinned through the future validated environment lock rather than manually copied into the direct dependency list.
