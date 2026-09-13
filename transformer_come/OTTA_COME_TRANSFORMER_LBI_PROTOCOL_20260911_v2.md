# COME-Transformer OTTA protocol — 2026-09-11 v2

Status: current frozen protocol for the five implemented non-LBI COME variants.
This revision supersedes `OTTA_COME_TRANSFORMER_LBI_PROTOCOL_20260909_v1` for
all new formal runs.

## Scope

This revision covers:

- `full_dense`
- `candidate_dense`
- `group_random`
- `group_magnitude`
- `group_saliency`

`group_lbi` remains outside the implemented COME protocol.

## Revision from v1

The only numerical protocol change is the host AdamW learning rate:

```text
v1: lr = 1e-5
v2: lr = 1e-7
```

The v2 value applies identically to all five variants.  AdamW remains
`betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=0.01`, with one optimizer step
per valid online batch, persistent optimizer state, and no scheduler.  Formal
v2 does not reset Adam moments between batches.

The revision is motivated by the seed-2026 VisDA-C collapse study: full-dense
performance improved monotonically as LR decreased, and `lr=1e-7` reduced the
dominant-class drift most strongly among the tested full-dense learning rates.

## Unchanged frozen semantics

All other scientific fields retain the v1 semantics:

- DeiT-Small source checkpoints and their SHA-256 verification;
- fixed seed-2026 one-pass target stream and PU/FO definitions;
- Office-31 batch size 64 and VisDA-C batch size 256;
- COME entropy-of-opinion objective with `p=2`, `tau=1`, detached logit norm,
  stable log-domain opinion, and entropy epsilon `1e-7`;
- full-dense and 12-tensor candidate-dense update scopes;
- the 6,912 structural groups and floor budgets `3/6/13` for
  `rho=0.0005/0.001/0.002`;
- Random seeds, Magnitude ranking, Saliency gradient reuse, strict masked
  AdamW, target-label isolation, and deterministic runtime policy.

COME continues to reuse the SHOT source loader, stream, parameter scopes,
group construction, mask machinery and optimizer implementation.  The COME
objective and the v2 host LR are the two intentional scientific differences
from the matched SHOT configuration.

## Identity and compatibility

Each variant has a distinct `come_transformer_*_20260911_v2` protocol
revision.  Baseline and sparse implementation revisions are also bumped to
`20260911_v2`.  Therefore every v2 resolved configuration receives a new
scientific hash and experiment key.

Artifacts produced with v1 `lr=1e-5` are historical controls.  They must not
be relabeled, overwritten, or aggregated with v2 results.

## Launch

The five variant matrices may be run sequentially with:

```bash
./transformer_come/scripts/run_all.sh
```

The launch scripts default to physical GPUs `0,1,2`, with one experiment
process per GPU.  The existing per-variant scripts remain valid for selective
execution, and their `COME_<VARIANT>_GPUS` variables can override this
operational default without changing the scientific protocol.
