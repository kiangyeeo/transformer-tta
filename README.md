# Transformer Split-LBI Test-Time Adaptation

This repository studies sparse update-support discovery for test-time adaptation (TTA) with Split-LBI. The legacy paper applies a SHOT objective and element-wise sparse updates to the 256-dimensional bottleneck fully connected layer of ResNet models. The legacy model factory also supports VGG. The current development goal is architecture-aware Transformer adaptation with a non-distilled, ImageNet-1K-pretrained DeiT-Small/16-224 backbone.

## Read First

- [`AGENTS.md`](AGENTS.md): authoritative project context, current implementation status, engineering boundaries, server constraints, experiment protocol, and unresolved decisions. Codex reads this file automatically from the repository root.
- [`TRANSFORMER_GROUP_SPLIT_LBI_PROPOSAL_DOLLAR_MATH.md`](TRANSFORMER_GROUP_SPLIT_LBI_PROPOSAL_DOLLAR_MATH.md): the proposed Transformer QK, VO, and FFN grouping design.
- [`SOURCE_TRAINING_DEIT_PROTOCOL.md`](SOURCE_TRAINING_DEIT_PROTOCOL.md): the frozen DeiT source-model architecture, parameters, validation isolation, checkpoint schema, and server commands.
- [`SERVER_STATE.md`](SERVER_STATE.md): the latest recorded server-side datasets, source checkpoints, verification boundary, and next-step prerequisites.
- [`catalog.md`](catalog.md): server-side paths for the repository, datasets, checkpoints, environments, and caches.
- [`26445_Test_Time_Adaptation_via (1).pdf`](26445_Test_Time_Adaptation_via%20%281%29.pdf): the legacy manuscript covering bottleneck-FC sparse adaptation.

## Current Status

| Component | Status |
| --- | --- |
| ResNet/VGG + SHOT OTTA | Implemented |
| FC scalar Dense/Random/Magnitude/Saliency/Split-LBI | Implemented |
| Dense ResNet convolution updates | Indirectly included in `full_dense` |
| Conv filter/channel Group Split-LBI | Not found in this repository |
| DeiT-S source training | Implemented; all four server-side source checkpoints are reported complete |
| DeiT-S TTDA source-only integration | Implemented and CPU-tested with synthetic checkpoints |
| DeiT-S OTTA source-only streaming integration | Implemented and CPU-tested with synthetic checkpoints |
| DeiT-S trainable TTDA / OTTA adaptation | Not implemented |
| Transformer paired QK/VO/FFN groups | Not implemented |
| TTDA | Source-only control only; adaptation is not implemented |

This repository now also contains a strict DeiT TTDA source-only control, but it is not an already completed FC + Conv + Transformer adaptation framework.

Two method-level decisions remain unresolved:

1. The current LBI Stage 2 starts from the current model plus a dense local delta, so the final local update is not guaranteed to be zero outside the discovered mask. The proposal's final equation suggests a strict masked delta, but that differs from the existing engine. In OTTA, even a strict K-group update per batch can accumulate into a source-relative support larger than K over the full stream.
2. Target labels must never enter adaptation. However, the existing analysis scripts rank candidates by target FO accuracy. Before formal experiments, the project must choose an independent validation rule, report every budget, or explicitly disclose oracle-style selection.

## Repository Layout

```text
train.py                 Legacy single-run SHOT-OTTA entry point
evaluate_deit_ttda.py    DeiT TTDA source-only evaluation entry point
evaluate_deit_otta.py    DeiT OTTA source-only streaming entry point
shot_otta/               Data, models, losses, trainer, and artifacts
core/lbi/                Existing element-wise Split-LBI implementation
source_training/         Legacy SHOT plus local-only DeiT source training
visda_otta/              VisDA-C metrics and helper logic
configs/                 Single-run configurations
experiments/             Experiment matrices
tools/                   Planner, launchers, status checks, and summaries
scripts/                 Maintenance utilities
tests/                   Synthetic smoke and engineering tests
```

The reusable pieces include the SHOT objective, image-list data pipeline, artifact system, experiment planning and launch infrastructure, summaries, and the broad LBI stage lifecycle. The strict local DeiT adapter, Transformer-aware identities, and zero-adaptation TTDA/OTTA evaluators are implemented. Transformer structural groups, group proximal operators, matched selectors, trainable OTTA, and trainable TTDA remain future work.

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

The repository does not contain Office-31, VisDA-C, image-list files, or source checkpoint binaries. Those assets remain on the experiment server under the paths documented in `catalog.md`. The legacy loader expects SHOT-style `office/*_list.txt` and `VISDA-C/*_list.txt` files; the newer DeiT source configurations use the server-side `image_lists/office31` and `image_lists/visda-c` paths.

The catalogued `deit_small_patch16_224.fb_in1k/model.safetensors` file is only the ImageNet-1K initialization. The four task-specific DeiT-S source models have now been trained and are reported present at:

- `/home/nas3/biod/wangkangyi/checkpoints/source_models/office31/amazon.pth`
- `/home/nas3/biod/wangkangyi/checkpoints/source_models/office31/dslr.pth`
- `/home/nas3/biod/wangkangyi/checkpoints/source_models/office31/webcam.pth`
- `/home/nas3/biod/wangkangyi/checkpoints/source_models/visda-c/train.pth`

This completion status is based on the user's server-side confirmation on 2026-08-17. The checkpoint files and adjacent manifests are not copied into this local repository, so their SHA-256 values, best epochs, validation metrics, and exact software versions are not locally audited yet. See [`SERVER_STATE.md`](SERVER_STATE.md) for the persistent verification boundary.

Every selector and budget compared under the same protocol must use the same hashed source model, `W0`.

`source_training/image_source.py` remains the legacy SHOT ResNet/VGG trainer and writes `source_F.pt`, `source_B.pt`, and `source_C.pt`. Transformer source models use [`train_source_deit.py`](train_source_deit.py), a direct `Linear(384, 31/12)` classifier, and the catalogued single-file `.pth` schema. The model is fully fine-tuned on source data; head-only training is deliberately rejected as a different linear-probe experiment.

The frozen parameters, rationale, data-isolation rule, checkpoint schema, and server commands are documented in [`SOURCE_TRAINING_DEIT_PROTOCOL.md`](SOURCE_TRAINING_DEIT_PROTOCOL.md). The two dataset configurations are:

- [`configs/source_deit_office31.yaml`](configs/source_deit_office31.yaml), reused with `--source-domain amazon|dslr|webcam`;
- [`configs/source_deit_visda.yaml`](configs/source_deit_visda.yaml), fixed to the synthetic `train` source.

Build deterministic absolute-path image lists before training:

```bash
python tools/build_image_lists.py --dataset office31 \
  --dataset-root /home/nas3/biod/wangkangyi/datasets/office31 \
  --output-root /home/nas3/biod/wangkangyi/datasets/image_lists/office31

python tools/build_image_lists.py --dataset visda-c \
  --dataset-root /home/nas3/biod/wangkangyi/datasets/visda-c \
  --output-root /home/nas3/biod/wangkangyi/datasets/image_lists/visda-c
```

Then validate the four source runs before launching them:

```bash
python train_source_deit.py --config configs/source_deit_office31.yaml --source-domain amazon --dry-run
python train_source_deit.py --config configs/source_deit_office31.yaml --source-domain dslr --dry-run
python train_source_deit.py --config configs/source_deit_office31.yaml --source-domain webcam --dry-run
python train_source_deit.py --config configs/source_deit_visda.yaml --dry-run
```

## DeiT TTDA Source-only Baseline

This control evaluates the unchanged source checkpoint on the full target
dataset. It creates no optimizer, computes no adaptation loss, performs no
backward pass, and uses one full-target prediction pass. Fixed-class macro
accuracy is reported as `Acc`; overall and per-class accuracies are also
recorded. The evaluator verifies that the complete model state hash is
unchanged. PU/FO remain specific to the legacy streaming OTTA protocol and are
not duplicated into this TTDA control.

The final report contains the seven transfer-level rows plus the Office-31
six-direction average. VisDA-C additionally expands all 12 canonical classes
into `visda_classwise.csv`, `visda_classwise_wide.csv`, and a classwise section
of `report.md`; its reported `Acc` is the arithmetic mean of those 12 class
accuracies, while `overall-Acc` is sample-weighted.

Prepare the environment and generate the fixed seven-task plan:

```bash
export PROJECT_ROOT=/home/nas3/biod/wangkangyi/transformer-tta
export RESULT_ROOT=/home/nas3/biod/wangkangyi/results/transformer_ttda_source_only
export PYTHON_BIN=/home/nas3/biod/wangkangyi/envs/lbi/bin/python
export HF_HOME=/home/nas3/biod/wangkangyi/hf-cache
export TORCH_HOME=/home/nas3/biod/wangkangyi/hf-cache/torch
export PIP_CACHE_DIR=/home/nas3/biod/wangkangyi/pip-cache
export CONDA_PKGS_DIRS=/home/nas3/biod/wangkangyi/conda-pkgs
export TMPDIR=/home/nas3/biod/wangkangyi/tmp
export PLAN_PATH="$RESULT_ROOT/plans/deit_ttda_source_only.json"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" tools/plan_deit_ttda_source_only.py \
  --config configs/deit_ttda_source_only.yaml \
  --matrix experiments/deit_ttda_source_only.yaml \
  --output "$PLAN_PATH"

"$PYTHON_BIN" tools/run_experiments_multi_gpu.py "$PLAN_PATH" \
  --runs-root "$RESULT_ROOT/runs" \
  --logs-root "$RESULT_ROOT/launcher_logs/full_dry_run" \
  --workdir "$PROJECT_ROOT" \
  --gpus 0,1,2,3,4,5,6 \
  --max-workers 7 \
  --workers-per-gpu 1 \
  --dry-run
```

Run the required single-GPU A-to-D pilot, then the status-aware full matrix.
The full launcher skips the already completed pilot automatically:

```bash
"$PYTHON_BIN" tools/run_experiments.py "$PLAN_PATH" \
  --runs-root "$RESULT_ROOT/runs" \
  --logs-root "$RESULT_ROOT/launcher_logs/pilot_AD" \
  --workdir "$PROJECT_ROOT" \
  --dataset office31 --source 0 --target 1 --max-experiments 1

"$PYTHON_BIN" tools/run_experiments_multi_gpu.py "$PLAN_PATH" \
  --runs-root "$RESULT_ROOT/runs" \
  --logs-root "$RESULT_ROOT/launcher_logs/full" \
  --workdir "$PROJECT_ROOT" \
  --gpus 0,1,2,3,4,5,6 \
  --max-workers 7 \
  --workers-per-gpu 1

"$PYTHON_BIN" tools/summarize_deit_ttda_source_only.py \
  --plan "$PLAN_PATH" \
  --runs-root "$RESULT_ROOT/runs" \
  --output-dir "$RESULT_ROOT/summary"
```

These commands are provided for the user to execute on the server; Codex has
not launched the real evaluations.

## DeiT OTTA Source-only Baseline

This control loads the same four versioned DeiT `.pth` source checkpoints and
traverses each target domain as one ordered batch stream. Every batch produces
an `online_batch` artifact, state is carried to the next batch, and a final
size-one batch is retained. No optimizer, loss, backward call, or parameter
update is allowed. `PU-Acc` summarizes predictions made during the stream;
`FO-Acc` uses the final-model evaluation semantics. Because this control keeps
`W_T == W_0`, FO reuses the exact stream predictions and PU/FO must be bitwise
consistent. A trainable OTTA method will intentionally remove that equality.

Generate and dry-run the seven-task DeiT OTTA plan:

```bash
export PROJECT_ROOT=/home/nas3/biod/wangkangyi/transformer-tta
export RESULT_ROOT=/home/nas3/biod/wangkangyi/results/transformer_otta_source_only
export PYTHON_BIN=/home/nas3/biod/wangkangyi/envs/lbi/bin/python
export HF_HOME=/home/nas3/biod/wangkangyi/hf-cache
export TORCH_HOME=/home/nas3/biod/wangkangyi/hf-cache/torch
export PIP_CACHE_DIR=/home/nas3/biod/wangkangyi/pip-cache
export CONDA_PKGS_DIRS=/home/nas3/biod/wangkangyi/conda-pkgs
export TMPDIR=/home/nas3/biod/wangkangyi/tmp
export PLAN_PATH="$RESULT_ROOT/plans/deit_otta_source_only.json"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" tools/plan_deit_otta_source_only.py \
  --config configs/deit_otta_source_only.yaml \
  --matrix experiments/deit_otta_source_only.yaml \
  --output "$PLAN_PATH"

"$PYTHON_BIN" tools/run_experiments_multi_gpu.py "$PLAN_PATH" \
  --runs-root "$RESULT_ROOT/runs" \
  --logs-root "$RESULT_ROOT/launcher_logs/full_dry_run" \
  --workdir "$PROJECT_ROOT" \
  --gpus 0,1,2,3,4,5,6 \
  --max-workers 7 \
  --workers-per-gpu 1 \
  --dry-run
```

After a one-GPU A-to-D pilot, run the remaining status-aware tasks and produce
the PU/FO report, including both scores for all 12 VisDA-C classes:

```bash
"$PYTHON_BIN" tools/run_experiments.py "$PLAN_PATH" \
  --runs-root "$RESULT_ROOT/runs" \
  --logs-root "$RESULT_ROOT/launcher_logs/pilot_AD" \
  --workdir "$PROJECT_ROOT" \
  --dataset office31 --source 0 --target 1 --max-experiments 1

"$PYTHON_BIN" tools/run_experiments_multi_gpu.py "$PLAN_PATH" \
  --runs-root "$RESULT_ROOT/runs" \
  --logs-root "$RESULT_ROOT/launcher_logs/full" \
  --workdir "$PROJECT_ROOT" \
  --gpus 0,1,2,3,4,5,6 \
  --max-workers 7 \
  --workers-per-gpu 1

"$PYTHON_BIN" tools/summarize_deit_otta_source_only.py \
  --plan "$PLAN_PATH" \
  --runs-root "$RESULT_ROOT/runs" \
  --output-dir "$RESULT_ROOT/summary"
```

Codex has not launched these server evaluations.

## Server Constraints

The experiment server has eight RTX 3090 GPUs with 24 GB of memory each, but its primary disk is full. Code, data, checkpoints, environments, caches, temporary files, and experiment outputs must remain under:

```text
/home/nas3/biod/wangkangyi/
```

See `catalog.md` for the exact known paths. Model construction must not trigger implicit downloads. Set `HF_HOME`, `TORCH_HOME`, `PIP_CACHE_DIR`, `CONDA_PKGS_DIRS`, and `TMPDIR` to locations under the approved root. The TTDA source-only result root is frozen; the OTTA source-only config uses its own sibling result root. Output locations for later trainable Transformer adaptation matrices remain undecided.

## Dependency Policy

Every new third-party Python import must be added to `requirements.txt` in the same change. Do not rely on ad hoc packages installed only in the server environment. Transitive dependencies are resolved by pip and should be pinned through the future validated environment lock rather than manually copied into the direct dependency list.
