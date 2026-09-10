"""CLI/matrix spec for the COME full-dense variant."""

from transformer_come.cli_common import VariantSpec

from .config import IMPLEMENTATION_REVISION, PROTOCOL_REVISION


SPEC = VariantSpec(
    variant="full_dense",
    module="transformer_come.full_dense",
    description="Protocol-aligned DeiT-S full-dense COME-OTTA baseline",
    run_prefix="come_full_dense_seed2026",
    protocol_revision=PROTOCOL_REVISION,
    implementation_revision=IMPLEMENTATION_REVISION,
)
