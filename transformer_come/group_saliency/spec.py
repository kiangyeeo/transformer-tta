"""CLI/matrix spec for the COME Group-Saliency variant."""

from transformer_come.cli_common import VariantSpec

from .config import IMPLEMENTATION_REVISION, PROTOCOL_REVISION


SPEC = VariantSpec(
    variant="group_saliency",
    module="transformer_come.group_saliency",
    description="Protocol-aligned DeiT-S dynamic structural-group Saliency COME-OTTA",
    run_prefix="come_group_saliency_seed2026",
    protocol_revision=PROTOCOL_REVISION,
    implementation_revision=IMPLEMENTATION_REVISION,
    sparse=True,
)
