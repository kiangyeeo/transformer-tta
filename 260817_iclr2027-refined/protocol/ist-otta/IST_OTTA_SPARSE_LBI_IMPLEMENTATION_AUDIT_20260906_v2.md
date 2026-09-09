# IST-OTTA Sparse/LBI Implementation Audit 2026-09-06 v2

**Implementation revision:** `ist_otta_sparse_lbi_20260906_v2`  
**Protocol:** `OTTA_IST_LBI_PROTOCOL_20260906_v1`  
**Status:** Correctness-reviewed implementation; formal LBI remains blocked until search protocol and tuned tuples are frozen.

## Review fixes

1. Fixed IST Random execution: `ist_fc_random` and `ist_conv_out_random` now run exactly three independent child masks per top-level condition, each from a fresh source model and the same formal target/augmentation stream.
2. Child mask seeds are deterministic and recorded: `selection_seed * 100 + mask_index`; formal Random requires `selection_seed == formal seed`.
3. Parent Random summary reports 3-mask mean/std PU/FO and separates mean single-mask efficiency from total three-mask operational cost.
4. Strengthened scientific-identity fallback so direct IST sparse identity construction uses the sparse/LBI implementation revision rather than the dense baseline revision.
5. No file under `shot_otta/` was modified by this v2 review. Shared `core/lbi/engine.py` remains the objective-agnostic gradient-accumulation extension introduced by v1; its default SHOT path is unchanged.

## Scientific semantics unchanged

The following remain exactly as frozen in `OTTA_IST_LBI_PROTOCOL_20260906_v1`:

- PLCA once and memory commit once per processed outer batch;
- fixed hard/soft targets for the full batch-local sparse/LBI task;
- full-outer-batch objective for saliency and LBI;
- one LBI support discovery per outer batch;
- corrected old-state Z update;
- strict integer budget and rollback, no top-K repair;
- Stage-2 masked-delta initialization and exactly one masked SGD step;
- native IST EMA for non-LBI sparse baselines;
- native IST EMA disabled for LBI, with omega as the sole writeback;
- FC scalar and Conv out-channel-only mainline;
- singleton skip and read-only PU/FO.
