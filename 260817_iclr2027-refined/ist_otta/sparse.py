"""Sparse support selection and exact masked SGD for IST families."""

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
    "ist_fc_module_dense",
    "ist_fc_random",
    "ist_fc_magnitude",
    "ist_fc_saliency",
    "ist_fc_lbi",
}
CONV_VARIANTS = {
    "ist_conv_module_dense",
    "ist_conv_out_random",
    "ist_conv_out_magnitude",
    "ist_conv_out_saliency",
    "ist_conv_out_lbi",
}
RANDOM_VARIANTS = {"ist_fc_random", "ist_conv_out_random"}
MAGNITUDE_VARIANTS = {
    "ist_fc_magnitude", "ist_conv_out_magnitude"
}
SALIENCY_VARIANTS = {"ist_fc_saliency", "ist_conv_out_saliency"}
LBI_VARIANTS = {"ist_fc_lbi", "ist_conv_out_lbi"}
SPARSE_VARIANTS = (
    RANDOM_VARIANTS | MAGNITUDE_VARIANTS | SALIENCY_VARIANTS | LBI_VARIANTS
)
NON_LBI_VARIANTS = (FC_VARIANTS | CONV_VARIANTS) - LBI_VARIANTS


def variant_track(variant):
    if variant in FC_VARIANTS:
        return "fc_scalar"
    if variant in CONV_VARIANTS:
        return "conv_out_channel"
    if variant == "ist_full_dense":
        return "full"
    raise ValueError("unsupported IST variant: " + str(variant))


def variant_selection(variant):
    if variant.endswith("_module_dense") or variant == "ist_full_dense":
        return "dense"
    for selection in ("random", "magnitude", "saliency", "lbi"):
        if variant.endswith("_" + selection):
            return selection
    raise ValueError("unsupported IST variant: " + str(variant))


def _global_scalar_mask(named_parameters, scores, requested_budget):
    named_parameters = list(named_parameters)
    total = sum(parameter.numel() for _, parameter in named_parameters)
    selected_count = max_support_count(requested_budget, total)
    flat_scores = torch.cat(
        [scores[name].detach().abs().reshape(-1) for name, _ in named_parameters]
    )
    selected = torch.zeros(total, dtype=torch.bool, device=flat_scores.device)
    if selected_count:
        if selected_count == total:
            indices = torch.arange(total, device=flat_scores.device)
        else:
            values = torch.topk(
                flat_scores, selected_count, largest=True, sorted=False
            ).values
            cutoff = values.min()
            strict = torch.nonzero(
                flat_scores > cutoff, as_tuple=False
            ).reshape(-1)
            remaining = selected_count - int(strict.numel())
            ties = torch.nonzero(
                flat_scores == cutoff, as_tuple=False
            ).reshape(-1)[:remaining]
            indices = torch.cat((strict, ties))
        selected[indices] = True
    masks = {}
    offset = 0
    for name, parameter in named_parameters:
        count = parameter.numel()
        masks[name] = selected[offset:offset + count].reshape_as(parameter)
        offset += count
    return masks


def build_static_masks(variant, named_parameters, requested_budget, seed=None):
    named_parameters = list(named_parameters)
    if variant == "ist_fc_random":
        if seed is None:
            raise ValueError("FC Random requires a child mask seed")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        total = sum(parameter.numel() for _, parameter in named_parameters)
        selected_count = max_support_count(requested_budget, total)
        indices = torch.randperm(total, generator=generator)[:selected_count]
        score = torch.zeros(total)
        if selected_count:
            score[indices] = 1.0
        score_by_name = {}
        offset = 0
        for name, parameter in named_parameters:
            count = parameter.numel()
            score_by_name[name] = score[
                offset:offset + count
            ].to(parameter.device).reshape_as(parameter)
            offset += count
        return _global_scalar_mask(
            named_parameters, score_by_name, requested_budget
        )
    if variant == "ist_fc_magnitude":
        return _global_scalar_mask(
            named_parameters,
            {name: parameter.detach() for name, parameter in named_parameters},
            requested_budget,
        )
    if variant == "ist_conv_out_random":
        if seed is None:
            raise ValueError("Conv Random requires a child mask seed")
        return build_random_group_masks(
            named_parameters, "out_channel", requested_budget, int(seed)
        )
    if variant == "ist_conv_out_magnitude":
        return global_group_score_mask(
            named_parameters,
            {name: parameter.detach() for name, parameter in named_parameters},
            "out_channel",
            requested_budget,
        )
    return {}


def build_saliency_masks(variant, named_parameters, requested_budget):
    named_parameters = list(named_parameters)
    scores = {}
    for name, parameter in named_parameters:
        if parameter.grad is None:
            raise RuntimeError("missing saliency gradient for " + name)
        scores[name] = parameter.detach() * parameter.grad.detach()
    if variant == "ist_fc_saliency":
        return _global_scalar_mask(
            named_parameters, scores, requested_budget
        )
    if variant == "ist_conv_out_saliency":
        return global_group_score_mask(
            named_parameters, scores, "out_channel", requested_budget
        )
    raise ValueError("not an IST saliency variant: " + variant)


def mask_statistics(masks, named_parameters, track, total_model_param_count):
    named_parameters = list(named_parameters)
    candidate_count = sum(parameter.numel() for _, parameter in named_parameters)
    scalar_count = selected_scalar_count(masks)
    stats = {
        "selected_param_count": scalar_count,
        "selected_scalar_count": scalar_count,
        "realized_scalar_ratio": float(scalar_count / candidate_count),
        "selected_over_scope_ratio": float(scalar_count / candidate_count),
        "selected_over_model_ratio": float(
            scalar_count / total_model_param_count
        ),
    }
    if track == "conv_out_channel":
        groups = group_count(named_parameters, "out_channel")
        selected_groups = selected_group_count(
            masks, named_parameters, "out_channel"
        )
        stats.update(
            {
                "total_group_count": groups,
                "selected_group_count": selected_groups,
                "realized_group_ratio": float(selected_groups / groups),
            }
        )
    return stats


def masked_optimizer_step(optimizer, named_parameters, masks):
    """Perform one SGD step while preserving all off-mask state exactly."""

    named_parameters = list(named_parameters)
    before = {
        name: parameter.detach().clone()
        for name, parameter in named_parameters
    }
    for name, parameter in named_parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(
                masks[name].to(parameter.grad.device, parameter.grad.dtype)
            )
        state = optimizer.state.get(parameter, {})
        buffer = state.get("momentum_buffer")
        if buffer is not None:
            buffer.mul_(masks[name].to(buffer.device, buffer.dtype))
    optimizer.step()
    with torch.no_grad():
        for name, parameter in named_parameters:
            mask = masks[name].to(parameter.device, torch.bool)
            parameter.copy_(torch.where(mask, parameter, before[name]))
            state = optimizer.state.get(parameter, {})
            buffer = state.get("momentum_buffer")
            if buffer is not None:
                buffer.mul_(mask.to(buffer.device, buffer.dtype))


def validate_ratio(requested_budget, allowed):
    if requested_budget is None or isinstance(requested_budget, bool):
        raise ValueError("IST sparse requested_budget must be explicit")
    value = float(requested_budget)
    if not math.isfinite(value) or not any(
        abs(value - item) <= 1.0e-12 for item in allowed
    ):
        raise ValueError(
            "IST sparse requested_budget must be one of "
            + ", ".join(str(item) for item in allowed)
        )
    return value


def restore_offmask_values(named_parameters, masks, base_values):
    """Undo floating-point EMA drift outside the selected support."""

    with torch.no_grad():
        for name, parameter in named_parameters:
            mask = masks[name].to(parameter.device, torch.bool)
            parameter.copy_(
                torch.where(mask, parameter, base_values[name])
            )
