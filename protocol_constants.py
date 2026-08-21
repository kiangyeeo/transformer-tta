"""Single source of truth for the frozen formal protocol identifiers."""

PROTOCOL_REVISION = "OTTA_FC_LBI_PROTOCOL_20260817_v1"
IMPLEMENTATION_REVISION = "iclr2027_refined_20260817_v1"
EFFICIENCY_PROTOCOL_REVISION = "otta_fc_batch_efficiency_20260817_v1"
SOURCE_CHECKPOINT_REVISION = "nips2026_shot_otta_uda_source_v1"

FORMAL_SEED = 2026
FC_CANDIDATE_PARAM_COUNT = 524544
FORMAL_BUDGETS = (0.0005, 0.001, 0.002)
FORMAL_INTEGER_BUDGETS = (262, 524, 1049)

LBI_FROZEN_CONSTANTS = {
    "support_threshold": 1.0e-4,
    "stage1_max_steps": 3000,
    "stage2_steps": 1,
    "delta_nonzero_tolerance": 1.0e-12,
    # Compatibility metadata only; strict integer K and rollback remain
    # defined by the LBI engine and are not relaxed by this value.
    "budget_tolerance": 1.0e-4,
}
