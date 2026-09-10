"""CLI/matrix spec for the COME Group-Magnitude variant."""

from transformer_come.cli_common import VariantSpec

from .config import IMPLEMENTATION_REVISION, PROTOCOL_REVISION


SPEC = VariantSpec(
    variant="group_magnitude",
    module="transformer_come.group_magnitude",
    description="Protocol-aligned DeiT-S structural-group Magnitude COME-OTTA",
    run_prefix="come_group_magnitude_seed2026",
    protocol_revision=PROTOCOL_REVISION,
    implementation_revision=IMPLEMENTATION_REVISION,
    sparse=True,
)
