"""Single source of truth for the frozen formal protocol identifiers."""

PROTOCOL_REVISION = "OTTA_FC_LBI_PROTOCOL_20260817_v1"
IMPLEMENTATION_REVISION = "iclr2027_refined_20260817_v1"
CONV_PROTOCOL_REVISION = "OTTA_CONV_LBI_PROTOCOL_20260826_v1"
CONV_IMPLEMENTATION_REVISION = "iclr2027_refined_conv_20260826_v1"
IST_PROTOCOL_REVISION = "OTTA_IST_BASELINE_PROTOCOL_20260906_v1"
IST_IMPLEMENTATION_REVISION = "ist_otta_p1_baseline_20260906_v4"
IST_LBI_PROTOCOL_REVISION = "OTTA_IST_LBI_PROTOCOL_20260906_v1"
IST_LBI_IMPLEMENTATION_REVISION = "ist_otta_sparse_lbi_20260906_v2"
NCTTA_PROTOCOL_REVISION = "OTTA_NCTTA_BASELINE_PROTOCOL_20260906_v2"
NCTTA_IMPLEMENTATION_REVISION = "nctta_otta_baseline_20260906_v3"
NCTTA_LBI_PROTOCOL_REVISION = "OTTA_NCTTA_LBI_PROTOCOL_20260906_v1"
NCTTA_LBI_IMPLEMENTATION_REVISION = "nctta_otta_sparse_lbi_20260907_v2"
COME_PROTOCOL_REVISION = "OTTA_COME_BASELINE_PROTOCOL_20260907_v1"
COME_IMPLEMENTATION_REVISION = "come_otta_baseline_20260908_v2"
COME_LBI_PROTOCOL_REVISION = "OTTA_COME_LBI_PROTOCOL_20260907_v1"
COME_LBI_IMPLEMENTATION_REVISION = "come_otta_sparse_lbi_20260908_v3"
EFFICIENCY_PROTOCOL_REVISION = "otta_fc_batch_efficiency_20260817_v1"
SOURCE_CHECKPOINT_REVISION = "nips2026_shot_otta_uda_source_v1"

FORMAL_SEED = 2026
FC_CANDIDATE_PARAM_COUNT = 524544
FORMAL_BUDGETS = (0.0005, 0.001, 0.002)
FORMAL_INTEGER_BUDGETS = (262, 524, 1049)
CONV_CANDIDATE_PARAM_COUNT = 12845056
CONV_GROUP_COUNTS = {
    "out_channel": 9216,
    "filter_connection": 6553600,
}

LBI_FROZEN_CONSTANTS = {
    "support_threshold": 1.0e-4,
    "stage1_max_steps": 3000,
    "stage2_steps": 1,
    "delta_nonzero_tolerance": 1.0e-12,
    # Compatibility metadata only; strict integer K and rollback remain
    # defined by the LBI engine and are not relaxed by this value.
    "budget_tolerance": 1.0e-4,
}

IST_LBI_FROZEN_CONSTANTS = {
    "support_threshold": 1.0e-4,
    "stage1_max_steps": 3000,
    "stage2_steps": 1,
    "delta_nonzero_tolerance": 1.0e-12,
    "budget_tolerance": 0.0,
}

NCTTA_LBI_FROZEN_CONSTANTS = {
    "support_threshold": 1.0e-4,
    "stage1_max_steps": 3000,
    "stage2_steps": 1,
    "delta_nonzero_tolerance": 1.0e-12,
    "budget_tolerance": 0.0,
}

COME_LBI_FROZEN_CONSTANTS = {
    "support_threshold": 1.0e-4,
    "stage1_max_steps": 3000,
    "stage2_steps": 1,
    "delta_nonzero_tolerance": 1.0e-12,
    "budget_tolerance": 0.0,
}
