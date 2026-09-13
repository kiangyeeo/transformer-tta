"""COME-OTTA on the frozen SHOT-Transformer substrate.

The host adaptation objective and protocol-v2 LR differ from ``transformer``.
Source W0, target stream, candidate universe, QK/VO/FFN grouping, integer
budget, corrected Group Split-LBI engine, strict masked AdamW implementation,
and PU/FO semantics are imported from the SHOT packages rather than
reimplemented here.
"""

from .config import DENSE_VARIANTS, SPARSE_VARIANTS, SUPPORTED_VARIANTS

__all__ = ["DENSE_VARIANTS", "SPARSE_VARIANTS", "SUPPORTED_VARIANTS"]
