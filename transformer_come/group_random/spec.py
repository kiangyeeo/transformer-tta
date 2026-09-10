"""CLI/matrix spec for the COME Group-Random variant."""

from transformer_come.cli_common import VariantSpec

from .config import IMPLEMENTATION_REVISION, NUM_RANDOM_MASKS, PROTOCOL_REVISION


SPEC = VariantSpec(
    variant="group_random",
    module="transformer_come.group_random",
    description="Protocol-aligned DeiT-S structural-group Random COME-OTTA",
    run_prefix="come_group_random_seed2026",
    protocol_revision=PROTOCOL_REVISION,
    implementation_revision=IMPLEMENTATION_REVISION,
    sparse=True,
    children_per_condition=NUM_RANDOM_MASKS,
    with_finalize=True,
)
