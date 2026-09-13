# COME VisDA-C collapse-mitigation pilot

This directory is deliberately separate from the frozen `transformer_come`
implementation and its formal artifacts.  It tests whether reduced update
strength mitigates the observed VisDA-C full-dense class collapse.

All eight runs reuse the exact source checkpoint, target permutation,
augmentation seed, batch boundaries, PU/FO semantics, stable log-domain COME
objective and collapse diagnostics.  They are **non-formal pilots** and write
only below:

```text
/home/nas3/biod/wangkangyi/results/transformer_come_mitigation/
```

The frozen matrix uses one process per GPU:

| ID | Scope | Optimizer | LR | tau | Global grad clip |
|---|---|---|---:|---:|---:|
| A | all except head | AdamW | 3e-6 | 1.0 | none |
| B | all except head | AdamW | 1e-6 | 1.0 | none |
| C | all except head | AdamW | 3e-6 | 0.5 | none |
| D | all except head | AdamW | 1e-6 | 0.5 | none |
| E | LayerNorm affine | SGD(momentum=.9) | 1e-3 | 1.0 | none |
| F | all except head | AdamW | 3e-6 | 0.25 | none |
| G | all except head | AdamW | 3e-6 | 1.0 | 1.0 |
| H | all except head | AdamW | 3e-6 | 0.5 | 1.0 |

E is *official-style*, not an official reproduction: it follows the published
ViT update scope and optimizer, but retains this repository's DeiT-S/VisDA-C
stream and batch size so the comparison remains controlled.

Run:

```bash
./transformer_come_mitigation/run_visda_8gpu.sh
```

Each child produces the usual `effective_config.yaml`, `manifest.json`,
`metrics.jsonl`, and `summary.json`.  The parent writes `plan.json`,
`matrix.json`, `aggregate.json`, `results.csv`, and one log per GPU.

## Tau refinement

The follow-up suite keeps the scope strictly full dense and evaluates the four
requested points: `tau={.8,.9}` crossed with `lr={1e-6,3e-6}`.  It uses four
GPUs concurrently and writes a new timestamped run root:

```bash
./transformer_come_mitigation/run_tau_refinement_4gpu.sh
```

## Learning-rate refinement at tau=1

This suite keeps `tau=1.0` and the full-dense update scope, and evaluates only
the three previously unrun learning rates `1e-7`, `3e-7`, and `5e-7`.  The
existing `1e-6` result is intentionally not recomputed and is used only as a
post-run comparison point.

```bash
./transformer_come_mitigation/run_lr_refinement_3gpu.sh
```

## Mechanism controls

This two-run suite isolates two proposed collapse amplifiers at `tau=1.0` and
`lr=1e-6`:

1. candidate dense: update exactly the established 12 candidate tensors while
   retaining persistent AdamW state;
2. full dense with persistent parameters, but zero `exp_avg` and `exp_avg_sq`
   immediately before every outer-batch optimizer step.  Adam's scalar step
   counter remains persistent, so this intervention changes only the two
   requested moment buffers.

```bash
./transformer_come_mitigation/run_mechanism_controls_2gpu.sh
```
