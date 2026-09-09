#!/usr/bin/env python3
"""Synthetic contracts for the stable log-domain COME opinion objective."""

import json
import os.path as osp
import sys

import torch


PROJECT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

from come_otta.objective import constrain_logits, subjective_opinion  # noqa: E402


def direct_opinion(logits, class_count):
    constrained = constrain_logits(logits)
    evidence = torch.exp(constrained)
    strength = evidence.sum(dim=1, keepdim=True) + class_count
    belief = evidence / strength
    uncertainty = logits.new_full((logits.shape[0], 1), float(class_count)) / strength
    opinion = torch.cat((belief, uncertainty), dim=1)
    entropy = -((opinion + 1.0e-7) * torch.log(opinion + 1.0e-7)).sum(dim=1)
    return belief, uncertainty, entropy


def main():
    maxima = {
        "loss_abs_diff": 0.0,
        "belief_abs_diff": 0.0,
        "uncertainty_abs_diff": 0.0,
        "entropy_abs_diff": 0.0,
        "gradient_abs_diff": 0.0,
    }
    for class_count in (12, 31):
        for seed in (7, 79, 2026):
            generator = torch.Generator().manual_seed(seed)
            actual_logits = (
                torch.randn(11, class_count, generator=generator, dtype=torch.float32)
                * 3.0
            ).requires_grad_(True)
            direct_logits = actual_logits.detach().clone().requires_grad_(True)
            stable = subjective_opinion(actual_logits, class_count)
            direct_belief, direct_uncertainty, direct_entropy = direct_opinion(
                direct_logits, class_count
            )
            stable_loss = stable["entropy"].mean()
            direct_loss = direct_entropy.mean()
            stable_gradient = torch.autograd.grad(stable_loss, actual_logits)[0]
            direct_gradient = torch.autograd.grad(direct_loss, direct_logits)[0]
            comparisons = {
                "loss_abs_diff": (stable_loss - direct_loss).abs().item(),
                "belief_abs_diff": (stable["belief"] - direct_belief).abs().max().item(),
                "uncertainty_abs_diff": (
                    stable["uncertainty"] - direct_uncertainty
                ).abs().max().item(),
                "entropy_abs_diff": (
                    stable["entropy"] - direct_entropy
                ).abs().max().item(),
                "gradient_abs_diff": (
                    stable_gradient - direct_gradient
                ).abs().max().item(),
            }
            for name, value in comparisons.items():
                maxima[name] = max(maxima[name], value)
            assert comparisons["loss_abs_diff"] <= 1.0e-5
            assert comparisons["belief_abs_diff"] <= 1.0e-5
            assert comparisons["uncertainty_abs_diff"] <= 1.0e-5
            assert comparisons["entropy_abs_diff"] <= 1.0e-5
            assert comparisons["gradient_abs_diff"] <= 1.0e-5

    probe = torch.randn(5, 9, dtype=torch.float32, requires_grad=True)
    reference_probe = probe.detach().clone().requires_grad_(True)
    constrained = constrain_logits(probe)
    norm = torch.norm(reference_probe, p=2, dim=-1, keepdim=True)
    reference = reference_probe / norm * norm.detach()
    assert torch.equal(constrained, reference)
    constrained_gradient = torch.autograd.grad(constrained.square().sum(), probe)[0]
    reference_gradient = torch.autograd.grad(reference.square().sum(), reference_probe)[0]
    assert torch.equal(constrained_gradient, reference_gradient)
    no_detach_probe = probe.detach().clone().requires_grad_(True)
    no_detach_norm = no_detach_probe.norm(p=2, dim=-1, keepdim=True)
    no_detach = no_detach_probe / no_detach_norm * no_detach_norm
    no_detach_gradient = torch.autograd.grad(
        no_detach.square().sum(), no_detach_probe
    )[0]
    assert not torch.allclose(constrained_gradient, no_detach_gradient)

    large_logits = torch.full((4, 12), -100.0, dtype=torch.float32)
    large_logits[:, 0] = 100.0
    direct_large = large_logits.clone().requires_grad_(True)
    stable_large = large_logits.clone().requires_grad_(True)
    _, _, direct_entropy = direct_opinion(direct_large, 12)
    stable = subjective_opinion(stable_large, 12)
    stable_loss = stable["entropy"].mean()
    stable_gradient = torch.autograd.grad(stable_loss, stable_large)[0]
    assert not torch.isfinite(direct_entropy).all()
    assert torch.isfinite(stable_loss)
    assert torch.isfinite(stable["belief"]).all()
    assert torch.isfinite(stable["uncertainty"]).all()
    assert torch.isfinite(stable_gradient).all()

    print(json.dumps({
        "status": "PASS",
        "dtype": "float32",
        "class_counts": [12, 31],
        "seeds": [7, 79, 2026],
        "max_differences": maxima,
        "large_logit_overflow_regression": "PASS",
        "constrained_logit_detach_contract": "PASS",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
