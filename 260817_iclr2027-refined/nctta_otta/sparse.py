"""Sparse support selection and exact masked SGD for controlled NCTTA."""

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
    "nctta_fc_module_dense",
    "nctta_fc_random",
    "nctta_fc_magnitude",
    "nctta_fc_saliency",
    "nctta_fc_lbi",
}
CONV_VARIANTS = {
    "nctta_conv_module_dense",
    "nctta_conv_out_random",
    "nctta_conv_out_magnitude",
    "nctta_conv_out_saliency",
    "nctta_conv_out_lbi",
}
RANDOM_VARIANTS = {"nctta_fc_random", "nctta_conv_out_random"}
MAGNITUDE_VARIANTS = {"nctta_fc_magnitude", "nctta_conv_out_magnitude"}
SALIENCY_VARIANTS = {"nctta_fc_saliency", "nctta_conv_out_saliency"}
LBI_VARIANTS = {"nctta_fc_lbi", "nctta_conv_out_lbi"}
SPARSE_VARIANTS = (
    RANDOM_VARIANTS | MAGNITUDE_VARIANTS | SALIENCY_VARIANTS | LBI_VARIANTS
)


def variant_track(variant):
    if variant in FC_VARIANTS:
        return "fc_scalar"
    if variant in CONV_VARIANTS:
        return "conv_out_channel"
    if variant == "nctta_full_dense":
        return "full"
    if variant == "nctta_native_norm":
        return "native_norm"
    raise ValueError("unsupported NCTTA variant: " + str(variant))


def variant_selection(variant):
    if variant.endswith("_module_dense") or variant == "nctta_full_dense":
        return "dense"
    if variant == "nctta_native_norm":
        return "native"
    for selection in ("random", "magnitude", "saliency", "lbi"):
        if variant.endswith("_" + selection):
            return selection
    raise ValueError("unsupported NCTTA variant: " + str(variant))


def _global_scalar_mask(named_parameters, scores, requested_budget):
    named_parameters = list(named_parameters)
    total = sum(parameter.numel() for _, parameter in named_parameters)
    selected_count = max_support_count(requested_budget, total)
    flat_scores = torch.cat(
        [scores[name].detach().abs().reshape(-1) for name, _ in named_parameters]
    )
    selected = torch.zeros(total, dtype=torch.bool, device=flat_scores.device)
    if selected_count:
        values = torch.topk(
            flat_scores, selected_count, largest=True, sorted=False
        ).values
        cutoff = values.min()
        strict = torch.nonzero(flat_scores > cutoff, as_tuple=False).reshape(-1)
        remaining = selected_count - int(strict.numel())
        ties = torch.nonzero(flat_scores == cutoff, as_tuple=False).reshape(-1)[
            :remaining
        ]
        selected[torch.cat((strict, ties))] = True
    masks = {}
    offset = 0
    for name, parameter in named_parameters:
        count = parameter.numel()
        masks[name] = selected[offset : offset + count].reshape_as(parameter)
        offset += count
    return masks


def build_static_masks(variant, named_parameters, requested_budget, seed=None):
    """Build a stream-fixed Random or source-magnitude mask."""
    named_parameters = list(named_parameters)
    if variant == "nctta_fc_random":
        if seed is None:
            raise ValueError("NCTTA FC Random requires a child mask seed")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        total = sum(parameter.numel() for _, parameter in named_parameters)
        count = max_support_count(requested_budget, total)
        chosen = torch.randperm(total, generator=generator)[:count]
        score = torch.zeros(total)
        score[chosen] = 1.0
        scores = {}
        offset = 0
        for name, parameter in named_parameters:
            size = parameter.numel()
            scores[name] = (
                score[offset : offset + size].to(parameter.device).reshape_as(parameter)
            )
            offset += size
        return _global_scalar_mask(named_parameters, scores, requested_budget)
    if variant == "nctta_fc_magnitude":
        return _global_scalar_mask(
            named_parameters,
            {name: parameter.detach() for name, parameter in named_parameters},
            requested_budget,
        )
    if variant == "nctta_conv_out_random":
        if seed is None:
            raise ValueError("NCTTA Conv Random requires a child mask seed")
        return build_random_group_masks(
            named_parameters, "out_channel", requested_budget, int(seed)
        )
    if variant == "nctta_conv_out_magnitude":
        return global_group_score_mask(
            named_parameters,
            {name: parameter.detach() for name, parameter in named_parameters},
            "out_channel",
            requested_budget,
        )
    raise ValueError("not a static NCTTA sparse variant: " + variant)


def build_saliency_masks(variant, named_parameters, requested_budget):
    """Select current-state |theta*gradient| support exactly once."""
    named_parameters = list(named_parameters)
    scores = {}
    for name, parameter in named_parameters:
        if parameter.grad is None:
            raise RuntimeError("missing NCTTA saliency gradient for " + name)
        if not bool(torch.isfinite(parameter.grad).all().item()):
            raise RuntimeError("non-finite NCTTA saliency gradient for " + name)
        scores[name] = parameter.detach() * parameter.grad.detach()
    if variant == "nctta_fc_saliency":
        return _global_scalar_mask(named_parameters, scores, requested_budget)
    if variant == "nctta_conv_out_saliency":
        return global_group_score_mask(
            named_parameters, scores, "out_channel", requested_budget
        )
    raise ValueError("not an NCTTA saliency variant: " + variant)


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
        if parameter.grad is not None:
            parameter.grad.mul_(
                masks[name].to(parameter.grad.device, parameter.grad.dtype)
            )
        buffer = optimizer.state.get(parameter, {}).get("momentum_buffer")
        if buffer is not None:
            buffer.mul_(masks[name].to(buffer.device, buffer.dtype))
    optimizer.step()
    with torch.no_grad():
        for name, parameter in named_parameters:
            mask = masks[name].to(parameter.device, torch.bool)
            parameter.copy_(torch.where(mask, parameter, before[name]))
            buffer = optimizer.state.get(parameter, {}).get("momentum_buffer")
            if buffer is not None:
                buffer.mul_(mask.to(buffer.device, buffer.dtype))


def validate_ratio(requested_budget, allowed):
    if requested_budget is None or isinstance(requested_budget, bool):
        raise ValueError("NCTTA sparse requested_budget must be explicit")
    value = float(requested_budget)
    if not math.isfinite(value) or not any(
        abs(value - item) <= 1.0e-12 for item in allowed
    ):
        raise ValueError(
            "NCTTA sparse requested_budget must be one of "
            + ", ".join(str(x) for x in allowed)
        )
    return value
