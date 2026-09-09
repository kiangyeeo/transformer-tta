"""COME sparse support selection and exact masked host SGD."""

import math

import torch

from core.lbi.diagnostics import max_support_count
from core.lbi.groups import (
    build_random_group_masks,
    global_group_score_mask,
    group_count,
    selected_group_count,
    selected_scalar_count,
)


FC_VARIANTS = {
    "come_fc_module_dense",
    "come_fc_random",
    "come_fc_magnitude",
    "come_fc_saliency",
    "come_fc_lbi",
}
CONV_VARIANTS = {
    "come_conv_module_dense",
    "come_conv_out_random",
    "come_conv_out_magnitude",
    "come_conv_out_saliency",
    "come_conv_out_lbi",
}
RANDOM_VARIANTS = {"come_fc_random", "come_conv_out_random"}
MAGNITUDE_VARIANTS = {"come_fc_magnitude", "come_conv_out_magnitude"}
SALIENCY_VARIANTS = {"come_fc_saliency", "come_conv_out_saliency"}
LBI_VARIANTS = {"come_fc_lbi", "come_conv_out_lbi"}
SPARSE_VARIANTS = (
    RANDOM_VARIANTS | MAGNITUDE_VARIANTS | SALIENCY_VARIANTS | LBI_VARIANTS
)


def variant_track(variant):
    if variant in FC_VARIANTS:
        return "fc_scalar"
    if variant in CONV_VARIANTS:
        return "conv_out_channel"
    if variant == "come_full_dense":
        return "full"
    raise ValueError("unsupported COME variant: " + str(variant))


def variant_selection(variant):
    if variant.endswith("_module_dense") or variant == "come_full_dense":
        return "dense"
    for selection in ("random", "magnitude", "saliency", "lbi"):
        if variant.endswith("_" + selection):
            return selection
    raise ValueError("unsupported COME variant: " + str(variant))


def _global_scalar_mask(named_parameters, scores, requested_budget):
    """Return deterministic global exact-floor-K masks across all tensors."""
    named_parameters = list(named_parameters)
    total = sum(parameter.numel() for _, parameter in named_parameters)
    selected_count = max_support_count(requested_budget, total)
    flat_scores = torch.cat(
        [scores[name].detach().abs().reshape(-1) for name, _ in named_parameters]
    )
    selected = torch.zeros(total, dtype=torch.bool, device=flat_scores.device)
    if selected_count:
        top_values = torch.topk(
            flat_scores, selected_count, largest=True, sorted=False
        ).values
        cutoff = top_values.min()
        strict = torch.nonzero(flat_scores > cutoff, as_tuple=False).reshape(-1)
        remaining = selected_count - int(strict.numel())
        ties = torch.nonzero(flat_scores == cutoff, as_tuple=False).reshape(-1)[
            :remaining
        ]
        chosen = torch.cat((strict, ties))
        if int(chosen.numel()) != selected_count:
            raise RuntimeError("COME global scalar top-K produced the wrong budget")
        selected[chosen] = True
    masks = {}
    offset = 0
    for name, parameter in named_parameters:
        size = parameter.numel()
        masks[name] = selected[offset : offset + size].reshape_as(parameter)
        offset += size
    return masks


def build_static_masks(variant, named_parameters, requested_budget, seed=None):
    """Build one stream-fixed Random or source-checkpoint Magnitude mask."""
    named_parameters = list(named_parameters)
    if variant == "come_fc_random":
        if seed is None:
            raise ValueError("COME FC Random requires a child mask seed")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        total = sum(parameter.numel() for _, parameter in named_parameters)
        count = max_support_count(requested_budget, total)
        chosen = torch.randperm(total, generator=generator)[:count]
        flat = torch.zeros(total)
        flat[chosen] = 1.0
        scores = {}
        offset = 0
        for name, parameter in named_parameters:
            size = parameter.numel()
            scores[name] = (
                flat[offset : offset + size].to(parameter.device).reshape_as(parameter)
            )
            offset += size
        return _global_scalar_mask(named_parameters, scores, requested_budget)
    if variant == "come_fc_magnitude":
        return _global_scalar_mask(
            named_parameters,
            {name: parameter.detach() for name, parameter in named_parameters},
            requested_budget,
        )
    if variant == "come_conv_out_random":
        if seed is None:
            raise ValueError("COME Conv Random requires a child mask seed")
        return build_random_group_masks(
            named_parameters, "out_channel", requested_budget, int(seed)
        )
    if variant == "come_conv_out_magnitude":
        return global_group_score_mask(
            named_parameters,
            {name: parameter.detach() for name, parameter in named_parameters},
            "out_channel",
            requested_budget,
        )
    raise ValueError("not a static COME sparse variant: " + variant)


def build_saliency_masks(variant, named_parameters, requested_budget):
    """Select current-state |theta*COME-gradient| support exactly once."""
    named_parameters = list(named_parameters)
    scores = {}
    for name, parameter in named_parameters:
        if parameter.grad is None:
            raise RuntimeError("missing COME saliency gradient for " + name)
        if not bool(torch.isfinite(parameter.grad).all().item()):
            raise RuntimeError("non-finite COME saliency gradient for " + name)
        scores[name] = parameter.detach() * parameter.grad.detach()
    if variant == "come_fc_saliency":
        return _global_scalar_mask(named_parameters, scores, requested_budget)
    if variant == "come_conv_out_saliency":
        return global_group_score_mask(
            named_parameters, scores, "out_channel", requested_budget
        )
    raise ValueError("not a COME saliency variant: " + variant)


def mask_statistics(masks, named_parameters, track, total_model_param_count):
    named_parameters = list(named_parameters)
    candidate_count = sum(parameter.numel() for _, parameter in named_parameters)
    scalar_count = selected_scalar_count(masks)
    stats = {
        "selected_param_count": scalar_count,
        "selected_scalar_count": scalar_count,
        "realized_scalar_ratio": float(scalar_count / candidate_count),
        "selected_over_scope_ratio": float(scalar_count / candidate_count),
        "selected_over_model_ratio": float(scalar_count / total_model_param_count),
    }
    if track == "conv_out_channel":
        groups = group_count(named_parameters, "out_channel")
        selected_groups = selected_group_count(masks, named_parameters, "out_channel")
        stats.update(
            {
                "total_group_count": groups,
                "selected_group_count": selected_groups,
                "realized_group_ratio": float(selected_groups / groups),
            }
        )
    return stats


def masked_optimizer_step(optimizer, named_parameters, masks):
    """Perform one host SGD step with exact off-mask value/state protection."""
    named_parameters = list(named_parameters)
    before = {name: parameter.detach().clone() for name, parameter in named_parameters}
    for name, parameter in named_parameters:
        mask = masks[name].to(parameter.device, torch.bool)
        if parameter.grad is not None:
            parameter.grad.mul_(mask.to(parameter.grad.dtype))
        momentum = optimizer.state.get(parameter, {}).get("momentum_buffer")
        if momentum is not None:
            momentum.mul_(mask.to(momentum.device, momentum.dtype))
    optimizer.step()
    with torch.no_grad():
        for name, parameter in named_parameters:
            mask = masks[name].to(parameter.device, torch.bool)
            parameter.copy_(torch.where(mask, parameter, before[name]))
            momentum = optimizer.state.get(parameter, {}).get("momentum_buffer")
            if momentum is not None:
                momentum.mul_(mask.to(momentum.device, momentum.dtype))
                if not torch.equal(momentum[~mask], torch.zeros_like(momentum[~mask])):
                    raise RuntimeError("COME off-mask momentum was not cleared")
            if not torch.equal(parameter.detach()[~mask], before[name][~mask]):
                raise RuntimeError("COME off-mask parameter changed")


def validate_ratio(requested_budget, allowed):
    if requested_budget is None or isinstance(requested_budget, bool):
        raise ValueError("COME sparse requested_budget must be explicit")
    value = float(requested_budget)
    if not math.isfinite(value) or not any(
        abs(value - item) <= 1.0e-12 for item in allowed
    ):
        raise ValueError(
            "COME sparse requested_budget must be one of "
            + ", ".join(str(item) for item in allowed)
        )
