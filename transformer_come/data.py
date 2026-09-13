"""Target stream and FO loader, shared verbatim with SHOT-Transformer.

All COME variants consume the identical one-pass target stream, including
the Amazon ``43x64 + 65`` tail, so the loaders are imported rather than
re-implemented.
"""

from transformer.source_only.data import (
    FixedOrderSampler,
    MergeSingletonTailBatchSampler,
    TargetImageList,
    build_target_loaders,
    build_transforms,
    fixed_random_order,
    read_records,
)

__all__ = [
    "FixedOrderSampler",
    "MergeSingletonTailBatchSampler",
    "TargetImageList",
    "build_target_loaders",
    "build_transforms",
    "fixed_random_order",
    "read_records",
]
