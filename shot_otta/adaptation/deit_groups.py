"""DeiT-S structural group definitions for TTA support selection.

Implements the proposal's paired Q-K, V-O, and FFN groups over the last-3
candidate blocks.  A group is the unit of support discovery; the matched
selectors (Random / Magnitude / Saliency / Group Split-LBI) must all share
this exact definition.

For timm's fused ``attn.qkv.weight`` of shape ``[3d, d]``:

* Q rows ``[0:d]``, K rows ``[d:2d]``, V rows ``[2d:3d]``;
* Q-K group ``p``  = Q row ``p``        + K row ``p``;
* V-O group ``p``  = V row ``p``        + ``attn.proj`` column ``p``;
* FFN group ``p``  = ``mlp.fc1`` row ``p`` + ``mlp.fc2`` column ``p``.
"""

import math

import torch


D = 384
D_FFN = 1536
DEFAULT_CANDIDATE_BLOCKS = (9, 10, 11)

GROUPS_PER_BLOCK = D + D + D_FFN  # 2304
SCALARS_PER_GROUP = 2 * D  # 768
TOTAL_GROUP_COUNT = GROUPS_PER_BLOCK * len(DEFAULT_CANDIDATE_BLOCKS)  # 6912

CANDIDATE_WEIGHT_COMPONENTS = (
    "attn.qkv.weight",
    "attn.proj.weight",
    "mlp.fc1.weight",
    "mlp.fc2.weight",
)


def candidate_group_order(candidate_blocks=DEFAULT_CANDIDATE_BLOCKS):
    """Canonical ordered group descriptors ``(block, kind, index)``.

    Order per block: Q-K groups first (``d``), then V-O groups (``d``), then
    FFN groups (``d_ffn``).  The position in this list is the global group
    index used by the selectors and by :func:`build_group_mask_dict`.
    """
    groups = []
    for block in candidate_blocks:
        for kind, count in (("qk", D), ("vo", D), ("ffn", D_FFN)):
            for index in range(count):
                groups.append((int(block), kind, int(index)))
    return groups


def total_group_count(candidate_blocks=DEFAULT_CANDIDATE_BLOCKS):
    """Return the number of structural groups in the candidate scope."""
    return len(candidate_group_order(candidate_blocks))


def active_group_count(budget, total_groups=TOTAL_GROUP_COUNT):
    """Return ``ceil(budget * total_groups)`` active groups."""
    if isinstance(budget, bool) or not isinstance(budget, (int, float)):
        raise ValueError("budget must be a number")
    if not 0.0 < float(budget) <= 1.0:
        raise ValueError("budget must be in (0, 1]")
    return int(math.ceil(float(budget) * int(total_groups)))


def select_random_groups(total_groups, num_active, seed):
    """Deterministically choose ``num_active`` distinct group indices."""
    if isinstance(num_active, bool) or int(num_active) != num_active:
        raise ValueError("num_active must be an integer")
    num_active = int(num_active)
    if not 0 <= num_active <= int(total_groups):
        raise ValueError("num_active must be in [0, total_groups]")
    if num_active == 0:
        return []
    if num_active == int(total_groups):
        return list(range(int(total_groups)))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    permutation = torch.randperm(int(total_groups), generator=generator)
    return permutation[:num_active].tolist()


def _new_mask_dict(candidate_blocks=DEFAULT_CANDIDATE_BLOCKS):
    masks = {}
    for block in candidate_blocks:
        block = int(block)
        masks[f"blocks.{block}.attn.qkv.weight"] = torch.zeros(
            (3 * D, D), dtype=torch.bool
        )
        masks[f"blocks.{block}.attn.proj.weight"] = torch.zeros(
            (D, D), dtype=torch.bool
        )
        masks[f"blocks.{block}.mlp.fc1.weight"] = torch.zeros(
            (D_FFN, D), dtype=torch.bool
        )
        masks[f"blocks.{block}.mlp.fc2.weight"] = torch.zeros(
            (D, D_FFN), dtype=torch.bool
        )
    return masks


def build_group_mask_dict(selected, *, candidate_blocks=DEFAULT_CANDIDATE_BLOCKS):
    """Build per-parameter boolean masks for the selected global group indices.

    The returned dict maps the exact DeiT candidate parameter name
    (e.g. ``blocks.9.attn.qkv.weight``) to a boolean mask of the same shape
    as that weight tensor.  A Q-K / V-O / FFN group masks both of its paired
    row/column slices.
    """
    selected = {int(index) for index in selected}
    masks = _new_mask_dict(candidate_blocks)
    group_index = 0
    for block in candidate_blocks:
        block = int(block)
        qkv = masks[f"blocks.{block}.attn.qkv.weight"]
        proj = masks[f"blocks.{block}.attn.proj.weight"]
        fc1 = masks[f"blocks.{block}.mlp.fc1.weight"]
        fc2 = masks[f"blocks.{block}.mlp.fc2.weight"]
        # Q-K groups: Q row p + K row p.
        for p in range(D):
            if group_index in selected:
                qkv[p, :] = True
                qkv[D + p, :] = True
            group_index += 1
        # V-O groups: V row p + output-projection column p.
        for p in range(D):
            if group_index in selected:
                qkv[2 * D + p, :] = True
                proj[:, p] = True
            group_index += 1
        # FFN groups: fc1 row p + fc2 column p.
        for p in range(D_FFN):
            if group_index in selected:
                fc1[p, :] = True
                fc2[:, p] = True
            group_index += 1
    return masks


def count_active_scalars(mask_dict):
    """Return the number of scalar weights covered by a mask dict."""
    return sum(int(mask.count_nonzero().item()) for mask in mask_dict.values())
