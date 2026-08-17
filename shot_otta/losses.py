"""Shared SHOT adaptation loss retained from nips2026 implementations."""

import torch
import torch.nn as nn


def entropy(probabilities):
    epsilon = 1.0e-5
    values = -probabilities * torch.log(probabilities + epsilon)
    return torch.sum(values, dim=1)


def _shot_objective(outputs, loss_config):
    """Compute the one-batch SHOT objective on raw outputs/logits."""
    probabilities = nn.Softmax(dim=1)(outputs)
    enabled_losses = set(loss_config["components"])

    total_loss = torch.tensor(0.0, device=outputs.device)
    classification_loss = torch.tensor(0.0, device=outputs.device)
    entropy_loss = torch.tensor(0.0, device=outputs.device)
    diversity_loss = torch.tensor(0.0, device=outputs.device)
    pseudo_ratio = None
    max_probability_mean = None

    if "pseudo" in enabled_losses and loss_config["cls_par"] > 0:
        max_probabilities, pseudo_labels = probabilities.max(dim=1)
        threshold_mask = max_probabilities.ge(
            loss_config["threshold"]
        ).float()
        per_sample_ce = nn.CrossEntropyLoss(reduction="none")(
            outputs, pseudo_labels
        )
        classification_loss = (per_sample_ce * threshold_mask).mean()
        total_loss = (
            total_loss + classification_loss * loss_config["cls_par"]
        )
        pseudo_ratio = float(threshold_mask.mean().item())
        max_probability_mean = float(max_probabilities.mean().item())

    if "ent" in enabled_losses:
        entropy_loss = torch.mean(entropy(probabilities))
        total_loss = total_loss + entropy_loss * loss_config["ent_par"]

    if "div" in enabled_losses:
        mean_probability = probabilities.mean(dim=0)
        diversity_loss = torch.sum(
            mean_probability * torch.log(mean_probability + 1.0e-5)
        )
        total_loss = total_loss + diversity_loss * loss_config["ent_par"]

    return total_loss, {
        "loss_cls": float(classification_loss.item()),
        "loss_ent": float(entropy_loss.item()),
        "loss_div": float(diversity_loss.item()),
        "pseudo_ratio": pseudo_ratio,
        "max_prob_mean": max_probability_mean,
    }


def shot_adaptation_loss(inputs, net_f, net_b, net_c, loss_config):
    """Compute the one-batch SHOT objective used by dense and LBI variants."""
    features = net_b(net_f(inputs))
    outputs = net_c(features)
    return _shot_objective(outputs, loss_config)


def deit_shot_adaptation_loss(logits, loss_config):
    """Compute the one-batch SHOT objective directly on DeiT logits.

    DeiT has no separate bottleneck, so its logits are already the
    ``net_c(features)`` term of the legacy path.
    """
    return _shot_objective(logits, loss_config)
