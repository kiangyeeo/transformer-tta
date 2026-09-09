# IST-OTTA P1 Baseline Implementation Audit — v4

**Status:** P1 correctness closed / ready for baseline protocol freeze  
**Date:** 2026-09-06  
**Implementation revision:** `ist_otta_p1_baseline_20260906_v4`  
**Parent contract:** `IST_OTTA_P0_REFACTOR_CONTRACT_20260902_v1`

## 1. P1 conclusion

The three dense IST baselines are implemented on the common refined OTTA substrate and have passed the P1 contract suite and short Office D->A smoke validation:

```text
ist_full_dense
ist_fc_module_dense
ist_conv_module_dense
```

The smoke processed 5 valid outer batches for each variant. The reported final memory size was 2560, exactly matching `5 x 64 x extend(8)`. Hard CE, soft KL and total loss remained finite. Scope checks confirmed `netF+netB` for full dense, bottleneck weight/bias only for FC module dense, and the 9 formal layer4 Conv weights only for Conv module dense.

No formal Office/VisDA baseline experiment has been started.

## 2. Final P1 semantics

IST retains its method-specific core:

- raw-image `extend=8` multi-view self-training;
- robust PLCA (`repeat=1`, `K=50`, `gamma=3`, `mode=l2`);
- causal memory bank (`max_len=10000`);
- corrected hard CE + pre-correction soft KL;
- `iters=1`;
- one outer-batch parameter moving average with `m=0.9`.

The controlled study reuses the current refined SHOT substrate for source F/B/C, source checkpoints, fixed seed-2026 outer stream, Office/VisDA batch sizes, optimizer/scheduler substrate, PU/FO semantics, candidate scopes, controlled BN behavior, metrics and provenance.

Historical SHOT singleton compatibility is explicit: an actual outer batch of size 1 is skipped before augmentation/adaptation/memory/EMA/PU; FO remains full-target read-only evaluation.

## 3. v4 review fixes

Two robustness fixes were added after the first smoke. Neither changes any already exercised Office/VisDA formal trajectory.

### 3.1 Logical IST state batch index

Memory and EMA now use the number of already processed valid outer batches as their state-bearing commit index, while artifacts retain the raw DataLoader `outer_batch_index` separately. This avoids a commit-count gap if a skipped singleton were ever followed by another valid batch.

For the current formal streams this is behavior-preserving: all valid batches before a possible tail singleton have identical raw and logical indices.

### 3.2 IST-only smoke option

`--debug-max-outer-batches` is now rejected for SHOT configs. It remains an IST-only non-formal smoke option and is still rejected by IST formal runs. This prevents a debug flag from being accepted by a SHOT non-formal config while the SHOT trainer does not implement that limit.

## 4. Validation after v4 fixes

```text
python -m py_compile ...                                      PASS
python tests/ist_otta_p1_contract_test.py                    PASS
three IST dense variants --dry-run                           PASS
SHOT + --debug-max-outer-batches                             rejected as intended
formal IST + --debug-max-outer-batches                       rejected as intended
```

Contract output:

```text
IST-OTTA P1 contracts passed
```

A strengthened singleton/state-index contract checks that a hypothetical sequence `[64, 1, 64]` maps state-bearing IST commits to contiguous logical indices `[0, 1]`.

## 5. Freeze recommendation

P1 baseline implementation is ready for protocol freeze. No more baseline semantic changes should be made unless a reproducible correctness bug is found.

The next phase should not run formal experiments yet. It should first freeze the IST baseline protocol, then implement the matched IST FC sparse family and IST Conv out-channel family using the current corrected LBI engine. LBI-specific restart granularity and IST-EMA/LBI-writeback composition remain intentionally outside P1 and must be resolved in the dedicated IST-LBI protocol before implementation.
