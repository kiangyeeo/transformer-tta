"""Causal current-batch SHOT loss for the DeiT full-dense baseline."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def shot_loss(logits: torch.Tensor, config: dict) -> tuple[torch.Tensor, dict]:
    probabilities = torch.softmax(logits, dim=1)
    max_probabilities, pseudo_labels = probabilities.max(dim=1)
    threshold_mask = max_probabilities.ge(float(config["threshold"]))
    per_sample_ce = F.cross_entropy(logits, pseudo_labels, reduction="none")
    classification = (per_sample_ce * threshold_mask.to(logits.dtype)).mean()
    entropy = -(probabilities * torch.log(probabilities + 1.0e-5)).sum(dim=1).mean()
    mean_probability = probabilities.mean(dim=0)
    diversity = (mean_probability * torch.log(mean_probability + 1.0e-5)).sum()
    total = (
        float(config["cls_par"]) * classification
        + float(config["ent_par"]) * entropy
        + float(config["ent_par"]) * diversity
    )
    return total, {
        "loss_cls": float(classification.detach().item()),
        "loss_ent": float(entropy.detach().item()),
        "loss_div": float(diversity.detach().item()),
        "pseudo_ratio": float(threshold_mask.to(torch.float32).mean().item()),
        "max_prob_mean": float(max_probabilities.detach().mean().item()),
    }
