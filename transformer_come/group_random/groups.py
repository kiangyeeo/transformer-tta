"""Paired groups and deterministic Random masks, shared with SHOT.

Same canonical 6912-group permutation per seed, so a COME Random child uses
exactly the same support as the matched SHOT child, including the
cross-budget nested-prefix property.
"""

from transformer.group_random.groups import (
    BLOCKS,
    GROUP_KINDS,
    HIDDEN_DIM,
    MLP_DIM,
    STRUCTURAL_GROUPS,
    StructuralGroup,
    build_masks,
    hash_masked_values,
    mask_record,
    mask_sha256,
    selected_group_ids,
    structural_groups,
)

__all__ = [
    "BLOCKS",
    "GROUP_KINDS",
    "HIDDEN_DIM",
    "MLP_DIM",
    "STRUCTURAL_GROUPS",
    "StructuralGroup",
    "build_masks",
    "hash_masked_values",
    "mask_record",
    "mask_sha256",
    "selected_group_ids",
    "structural_groups",
]
