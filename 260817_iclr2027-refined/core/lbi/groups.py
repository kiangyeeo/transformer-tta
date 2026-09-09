"""Vectorized group operations for controlled layer4 Conv adaptation."""

import torch

from .diagnostics import max_support_count


CONV_GROUP_MODES = ("out_channel", "filter_connection")


def _validate_group_tensor(value, name="Conv candidate"):
    if value.ndim != 4:
        raise ValueError(f"{name} must be a 4D Conv weight tensor")


def validate_group_mode(group_mode):
    if group_mode not in CONV_GROUP_MODES:
        raise ValueError(
            "group_mode must be one of: " + ", ".join(CONV_GROUP_MODES)
        )


def local_group_count(value, group_mode):
    """Return a tensor's group count from shape only, with no norm compute."""
    validate_group_mode(group_mode)
    _validate_group_tensor(value)
    if group_mode == "out_channel":
        return int(value.shape[0])
    return int(value.shape[0] * value.shape[1])


def group_view(value, group_mode):
    """View a Conv tensor as ``[groups, group_scalars]``."""
    validate_group_mode(group_mode)
    _validate_group_tensor(value)
    if group_mode == "out_channel":
        return value.reshape(value.shape[0], -1)
    return value.reshape(value.shape[0] * value.shape[1], -1)


def group_scores(value, group_mode):
    """Return one L2 norm per declared Conv group."""
    return torch.linalg.vector_norm(group_view(value, group_mode), dim=1)


def group_count(named_parameters, group_mode):
    """Count the global group pool using tensor shapes only."""
    return sum(
        local_group_count(parameter, group_mode)
        for _, parameter in named_parameters
    )


def _broadcast_group_mask(group_mask, parameter, group_mode):
    _validate_group_tensor(parameter)
    if group_mode == "out_channel":
        return group_mask.reshape(parameter.shape[0], 1, 1, 1).expand_as(
            parameter
        )
    return group_mask.reshape(
        parameter.shape[0], parameter.shape[1], 1, 1
    ).expand_as(parameter)


def group_support_masks(gamma, group_mode, support_threshold):
    """Build parameter masks from the sole thresholded group support."""
    masks = {}
    for name, value in gamma.items():
        selected_groups = group_scores(value.detach(), group_mode).ge(
            support_threshold
        )
        masks[name] = _broadcast_group_mask(
            selected_groups, value, group_mode
        )
    return masks


def selected_group_count(masks, named_parameters, group_mode):
    """Count selected groups from broadcast parameter masks."""
    count = 0
    for name, parameter in named_parameters:
        mask = masks[name].to(device=parameter.device, dtype=torch.bool)
        if group_mode == "out_channel":
            selected = mask[:, 0, 0, 0]
        else:
            selected = mask[:, :, 0, 0]
        count += int(selected.count_nonzero().item())
    return count


def selected_scalar_count(masks):
    return sum(int(mask.count_nonzero().item()) for mask in masks.values())


def group_lasso_prox(z_value, kappa, group_mode):
    """Unweighted Group-Lasso prox with an exact zero-norm branch."""
    view = group_view(z_value, group_mode)
    norms = torch.linalg.vector_norm(view, dim=1)
    scale = torch.zeros_like(norms)
    nonzero = norms.gt(0)
    # No epsilon is used: it would change the formal support semantics.
    scale[nonzero] = torch.clamp(
        1.0 - norms[nonzero].reciprocal(), min=0.0
    )
    return kappa * (view * scale.unsqueeze(1)).reshape_as(z_value)


def _group_offsets(named_parameters, group_mode):
    offsets = {}
    total = 0
    for name, parameter in named_parameters:
        if name in offsets:
            raise ValueError(f"duplicate candidate parameter name: {name}")
        local_count = local_group_count(parameter, group_mode)
        offsets[name] = (total, total + local_count)
        total += local_count
    return offsets, total


def masks_from_global_group_indices(
    named_parameters, group_mode, selected_indices
):
    """Broadcast exact global group selections back to Conv tensors."""
    named_parameters = list(named_parameters)
    offsets, total = _group_offsets(named_parameters, group_mode)
    selected = torch.zeros(total, dtype=torch.bool, device="cpu")
    if selected_indices.numel():
        indices = selected_indices.to(device="cpu", dtype=torch.long)
        if int(indices.min().item()) < 0 or int(indices.max().item()) >= total:
            raise ValueError("selected group index is outside the global pool")
        selected[indices] = True
    masks = {}
    for name, parameter in named_parameters:
        start, end = offsets[name]
        local = selected[start:end].to(device=parameter.device)
        masks[name] = _broadcast_group_mask(local, parameter, group_mode)
    return masks


def global_group_score_mask(
    named_parameters, score_by_name, group_mode, requested_budget
):
    """Select global top-K Conv groups with deterministic score ordering."""
    named_parameters = list(named_parameters)
    _, total = _group_offsets(named_parameters, group_mode)
    max_count = max_support_count(requested_budget, total)
    if max_count == 0:
        selected_indices = torch.empty(0, dtype=torch.long)
    else:
        # Keep score ranking on the parameter device.  This matters for the
        # 6.55M-group filter-connection saliency baseline, where copying all
        # scores to CPU and stable-sorting them every online batch is
        # needlessly expensive.  Ties at the K-th score are resolved by the
        # smallest deterministic global group indices, matching stable global
        # ordering semantics without a full argsort.
        global_scores = torch.cat(
            [
                group_scores(score_by_name[name].detach(), group_mode)
                .reshape(-1)
                for name, _ in named_parameters
            ]
        )
        if max_count >= total:
            selected_indices = torch.arange(
                total, device=global_scores.device, dtype=torch.long
            )
        else:
            top_values = torch.topk(
                global_scores, max_count, largest=True, sorted=False
            ).values
            cutoff = top_values.min()
            strict_indices = torch.nonzero(
                global_scores > cutoff, as_tuple=False
            ).reshape(-1)
            remaining = max_count - int(strict_indices.numel())
            tie_indices = torch.nonzero(
                global_scores == cutoff, as_tuple=False
            ).reshape(-1)[:remaining]
            selected_indices = torch.cat(
                (strict_indices, tie_indices), dim=0
            )
            if int(selected_indices.numel()) != max_count:
                raise RuntimeError(
                    "global Conv group top-K selection produced the wrong "
                    "number of groups"
                )
    return masks_from_global_group_indices(
        named_parameters, group_mode, selected_indices
    )


def build_random_group_masks(
    named_parameters, group_mode, requested_budget, seed
):
    """Uniformly sample exactly K groups from the global Conv pool."""
    named_parameters = list(named_parameters)
    _, total = _group_offsets(named_parameters, group_mode)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    max_count = max_support_count(requested_budget, total)
    if max_count == 0:
        selected_indices = torch.empty(0, dtype=torch.long)
    else:
        selected_indices = torch.randperm(
            total, generator=generator, device="cpu"
        )[:max_count]
    return masks_from_global_group_indices(
        named_parameters, group_mode, selected_indices
    )
