# DeiT-S Source-Training Protocol

This document freezes source-training protocol v1 for the Transformer branch.
It is a methodological contract, not a claim that the resulting checkpoints
have already been trained or validated on the server.

## Scope and isolation

The protocol produces four source models from the same local ImageNet-1K
initialization:

| Dataset | Source domain | Classes | Output |
| --- | --- | ---: | --- |
| Office-31 | Amazon | 31 | `source_models/office31/amazon.pth` |
| Office-31 | DSLR | 31 | `source_models/office31/dslr.pth` |
| Office-31 | Webcam | 31 | `source_models/office31/webcam.pth` |
| VisDA-C | synthetic `train` | 12 | `source_models/visda-c/train.pth` |

Only a source-domain image list is accepted by the trainer. There is no target
list option in the source-training configuration. Target labels therefore
cannot enter optimization, early stopping, or checkpoint selection.

The Office and VisDA configurations are respectively
[`configs/source_deit_office31.yaml`](configs/source_deit_office31.yaml) and
[`configs/source_deit_visda.yaml`](configs/source_deit_visda.yaml). Changing a
scientific parameter changes the recorded effective-config hash.

## Model contract

Both datasets use the exact non-distilled timm model
`deit_small_patch16_224.fb_in1k`. Construction always passes
`pretrained=False`; the 1000-class ImageNet state is loaded strictly from the
catalogued local `model.safetensors`. Only after that strict load succeeds is
the pretrained classifier replaced:

```text
Office-31: CLS(384) -> Linear(384, 31)
VisDA-C:   CLS(384) -> Linear(384, 12)
```

There is no 256-dimensional bottleneck, BatchNorm bottleneck, or
weight-normalized classifier in the Transformer source model. Those belong to
the legacy SHOT ResNet path. The new linear head is initialized with truncated
normal standard deviation 0.02 and zero bias, matching the ViT/DeiT head scale.

Source training is full-model fine-tuning. A head-only run would be linear
probing and is rejected by configuration validation because it changes the
meaning of the source model used for TTA. During later Transformer TTA, the
classifier is frozen and structural candidates are drawn only from the agreed
last three Transformer blocks.

## Frozen parameters and rationale

The evidence labels below distinguish direct reference settings from local
design decisions:

- **Reference**: inherited from official SHOT or DeiT practice.
- **Derived**: numerically derived from a reference setting.
- **Decision**: a conservative project choice that must be disclosed and can
  only be changed by versioning this protocol.

| Parameter | Office-31 | VisDA-C | Rationale |
| --- | ---: | ---: | --- |
| Epochs | 100 | 10 | **Reference.** These are the official SHOT source-training horizons for Office and VisDA respectively. They do not come from the old OTTA attachment logs. |
| Global batch size | 64 | 64 | **Reference.** SHOT uses 64. On the planned one-process-per-3090 execution this is also the per-GPU batch size. Batch 64 is a fixed comparability setting, not a claim of optimality. |
| Optimizer | AdamW | AdamW | **Reference/decision.** AdamW follows DeiT/ViT fine-tuning rather than mechanically transferring ResNet SGD. |
| Backbone LR | `6.25e-5` | `6.25e-5` | **Derived.** DeiT's `5e-4` reference LR is linearly scaled by `64/512`. |
| Head LR | `6.25e-4` | `6.25e-4` | **Derived.** The newly initialized classifier uses the SHOT-style 10x learning-rate ratio relative to the pretrained backbone. |
| Adam betas / epsilon | `(0.9, 0.999)` / `1e-8` | same | **Reference.** Standard AdamW numerical settings. |
| Weight decay | `0.05` | `0.05` | **Reference.** DeiT's AdamW regularization. Biases, all one-dimensional norm parameters, class token, and position embedding are placed in explicit zero-decay groups. |
| Scheduler | cosine | cosine | **Reference.** DeiT-style decay, applied per optimizer step. |
| Warmup | 5 epochs | 1 epoch | **Derived/decision.** Five epochs follows the DeiT reference. One epoch is the smallest explicit warmup for the ten-epoch VisDA schedule. |
| Minimum LR | `1e-6` | `1e-6` | **Decision.** A nonzero floor below the backbone LR avoids an abrupt terminal zero while still providing substantial decay. |
| Loss | cross-entropy | cross-entropy | Source labels supervise source training; target labels remain inaccessible. |
| Label smoothing | `0.1` | `0.1` | **Reference.** Shared by SHOT source training and DeiT training. |
| Input | 224x224 | 224x224 | **Reference.** Fixed by the selected DeiT-S/16-224 checkpoint. |
| Resize / crop | square resize 256, random crop 224 for training; center crop 224 for validation | same | **Reference/decision.** Preserves the existing SHOT crop geometry while switching interpolation to the DeiT-compatible bicubic mode. |
| Horizontal flip | probability `0.5` | `0.5` | **Reference.** The existing source recipe's minimal spatial augmentation. |
| Normalization | ImageNet mean/std | same | **Reference.** Required by the ImageNet-pretrained initialization. |
| Mixup, CutMix, RandAugment, random erasing | disabled | disabled | **Decision.** Version 1 isolates backbone adaptation from a second strong-augmentation study. These must not be silently enabled. |
| Dropout / DropPath | `0.0 / 0.0` | `0.0 / 0.0` | **Decision.** Keeps source v1 and later TTA deterministic and avoids adding a dataset-dependent regularization axis. |
| EMA | disabled | disabled | **Decision.** Ensures a single unambiguous W0 rather than ordinary-vs-EMA checkpoint ambiguity. |
| Gradient clipping | disabled | disabled | **Reference/decision.** Not part of the frozen base recipe; non-finite gradients are a failed pilot, not permission for an unrecorded fix. |
| AMP | enabled on CUDA | enabled on CUDA | Engineering choice for RTX 3090 throughput and memory. The effective value and software versions are recorded. |
| Training seed | 2020 | 2020 | Protocol convention shared with the legacy experiments. It is not selected for better accuracy. |

Primary references:

- Official SHOT training commands: <https://github.com/tim-learn/SHOT#training>
- Official SHOT source trainer: <https://github.com/tim-learn/SHOT/blob/master/object/image_source.py>
- Official DeiT training implementation: <https://github.com/facebookresearch/deit/blob/main/main.py>
- Selected timm model card: <https://huggingface.co/timm/deit_small_patch16_224.fb_in1k>

## Source validation and checkpoint selection

Each source list is split once with seed 2020, independently within every
class. The requested split is 90% training and 10% validation. Each class with
at least two images contributes at least one validation image and retains at
least one training image, so the aggregate percentage may differ slightly from
10% on small Office domains. The exact indices, per-class counts, and hashes
are written to `<domain>.split.json`. The generated `class_to_idx.json` path
and SHA-256 are also embedded in W0 so class-index semantics cannot drift
between source training and target evaluation.

- Office-31 selects the best epoch by source-validation overall top-1 accuracy
  for SHOT comparability and also records macro-per-class accuracy.
- VisDA-C selects the best epoch by source-validation 12-class macro accuracy,
  matching the dataset's principal evaluation metric.
- Ties keep the earlier checkpoint because only a strict improvement replaces
  W0.
- The trainer does not retrain on the full source list after model selection;
  doing so would create a new model without an independently selected stopping
  point.

## Checkpoint and resume contract

The final `.pth` is a versioned single-file checkpoint containing:

- one complete DeiT `state_dict` including the direct classifier head;
- architecture, dataset, source domain, number of classes, and head schema;
- local ImageNet checkpoint path and SHA-256;
- effective source-training configuration and its SHA-256;
- split indices and hashes;
- best epoch and source-validation metrics;
- optimizer parameter-group names and counts;
- Git commit/dirty state and Python/PyTorch/torchvision/timm/safetensors versions.

The adjacent manifest records the final checkpoint's own SHA-256. The
`<domain>.last.pth` file additionally stores optimizer, AMP scaler, and RNG
state and is atomically replaced after every epoch. It is for exact recovery
from server disconnections and is not a TTA W0.

Every selector, budget, and TTA stream seed for a source domain must load the
same final `.pth` and verify the manifest hash.

## Server preparation and commands

Generate absolute-path image lists after the datasets have been extracted:

```bash
cd /home/nas3/biod/wangkangyi/transformer-tta

python tools/build_image_lists.py \
  --dataset office31 \
  --dataset-root /home/nas3/biod/wangkangyi/datasets/office31 \
  --output-root /home/nas3/biod/wangkangyi/datasets/image_lists/office31

python tools/build_image_lists.py \
  --dataset visda-c \
  --dataset-root /home/nas3/biod/wangkangyi/datasets/visda-c \
  --output-root /home/nas3/biod/wangkangyi/datasets/image_lists/visda-c
```

Resolve all four runs without loading images or timm:

```bash
python train_source_deit.py --config configs/source_deit_office31.yaml --source-domain amazon --dry-run
python train_source_deit.py --config configs/source_deit_office31.yaml --source-domain dslr --dry-run
python train_source_deit.py --config configs/source_deit_office31.yaml --source-domain webcam --dry-run
python train_source_deit.py --config configs/source_deit_visda.yaml --dry-run
```

Train one source model per GPU/process. For example:

```bash
CUDA_VISIBLE_DEVICES=0 python train_source_deit.py --config configs/source_deit_office31.yaml --source-domain amazon
CUDA_VISIBLE_DEVICES=1 python train_source_deit.py --config configs/source_deit_office31.yaml --source-domain dslr
CUDA_VISIBLE_DEVICES=2 python train_source_deit.py --config configs/source_deit_office31.yaml --source-domain webcam
CUDA_VISIBLE_DEVICES=3 python train_source_deit.py --config configs/source_deit_visda.yaml
```

Before launching, set all environment/cache/temp variables to the locations in
`catalog.md`. The model code never uses `pretrained=True` and must not download
weights implicitly.
