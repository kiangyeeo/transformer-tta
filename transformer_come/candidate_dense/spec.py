"""CLI/matrix spec for the COME candidate-dense variant."""

from transformer_come.cli_common import VariantSpec

from .config import IMPLEMENTATION_REVISION, PROTOCOL_REVISION


SPEC = VariantSpec(
    variant="candidate_dense",
    module="transformer_come.candidate_dense",
    description="Protocol-aligned DeiT-S candidate-dense COME-OTTA baseline",
    run_prefix="come_candidate_dense_seed2026",
    protocol_revision=PROTOCOL_REVISION,
    implementation_revision=IMPLEMENTATION_REVISION,
)
