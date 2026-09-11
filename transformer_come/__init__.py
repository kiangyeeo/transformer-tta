"""COME-OTTA on the frozen SHOT-Transformer substrate.

Only the host adaptation objective differs from ``transformer``: source W0,
target stream, candidate universe, QK/VO/FFN grouping, integer budget, strict
masked AdamW and PU/FO semantics are imported from the SHOT packages rather
than reimplemented here.
"""

from .config import DENSE_VARIANTS, SPARSE_VARIANTS, SUPPORTED_VARIANTS

__all__ = ["DENSE_VARIANTS", "SPARSE_VARIANTS", "SUPPORTED_VARIANTS"]
