#!/usr/bin/env python3
"""CPU contracts for the stable log-domain COME objective (C02-C08).

Covers: dataset class count, equivalence with the direct formula, the
large-logit overflow regression, the stop-gradient location, the absence of a
norm epsilon/clamp, fail-loudly behavior and the no-renormalization rule.  The
port is additionally checked value-by-value against the audited FC reference
``260817_iclr2027-refined/come_otta/objective.py``.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformer_come.objective import (  # noqa: E402
    OFFICIAL_COME_COMMIT,
    OFFICIAL_ENTROPY_EPSILON,
    OFFICIAL_P,
    OFFICIAL_TAU,
    come_loss,
    constrain_logits,
    subjective_opinion,
)

FC_OBJECTIVE_PATH = (
    PROJECT_ROOT / "260817_iclr2027-refined" / "come_otta" / "objective.py"
)


def _load_fc_reference():
    spec = importlib.util.spec_from_file_location(
        "fc_come_objective_reference", FC_OBJECTIVE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def direct_opinion(logits, class_count):
    """The original direct-exp formula, kept only as a test reference."""
    constrained = constrain_logits(logits)
    evidence = torch.exp(constrained)
    strength = evidence.sum(dim=1, keepdim=True) + class_count
    belief = evidence / strength
    uncertainty = logits.new_full((logits.shape[0], 1), float(class_count)) / strength
    opinion = torch.cat((belief, uncertainty), dim=1)
    entropy = -((opinion + 1.0e-7) * torch.log(opinion + 1.0e-7)).sum(dim=1)
    return belief, uncertainty, entropy


def check_frozen_constants() -> None:
    assert OFFICIAL_P == 2.0
    assert OFFICIAL_TAU == 1.0
    assert OFFICIAL_ENTROPY_EPSILON == 1.0e-7
    assert OFFICIAL_COME_COMMIT == "409a19b71f62c765b1a5be62347a9455524ec176"
    reference = _load_fc_reference()
    assert reference.OFFICIAL_COME_COMMIT == OFFICIAL_COME_COMMIT
    assert reference.OFFICIAL_P == OFFICIAL_P
    assert reference.OFFICIAL_TAU == OFFICIAL_TAU
    assert reference.OFFICIAL_ENTROPY_EPSILON == OFFICIAL_ENTROPY_EPSILON


def check_class_count_contract() -> dict:
    """C02: C is the adaptation classifier output dimension, never 1000."""
    rejected = 0
    logits = torch.randn(4, 31)
    for class_count in (1000, 12, 30):
        try:
            subjective_opinion(logits, class_count)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError(f"class_count={class_count} was accepted for C=31")
    for bad in (31.5, True, "31"):
        try:
            subjective_opinion(logits, bad)
        except (ValueError, TypeError):
            rejected += 1
        else:
            raise AssertionError(f"class_count={bad!r} was accepted")
    try:
        subjective_opinion(torch.randn(2, 3, 31), 31)
    except ValueError:
        rejected += 1
    else:
        raise AssertionError("Three-dimensional logits were accepted")
    for class_count in (12, 31):
        values = subjective_opinion(torch.randn(5, class_count), class_count)
        assert values["opinion"].shape == (5, class_count + 1)
        assert come_loss(
            torch.randn(5, class_count), class_count
        ).diagnostics["class_count_c"] == class_count
    return {"rejected_class_count_cases": rejected}


def check_moderate_equivalence() -> dict:
    """C03: stable log-domain matches the direct formula, gradients included."""
    maxima = {
        "loss_abs_diff": 0.0,
        "belief_abs_diff": 0.0,
        "uncertainty_abs_diff": 0.0,
        "entropy_abs_diff": 0.0,
        "gradient_abs_diff": 0.0,
    }
    reference = _load_fc_reference()
    for class_count in (12, 31):
        for seed in (7, 79, 2026):
            generator = torch.Generator().manual_seed(seed)
            actual = (
                torch.randn(11, class_count, generator=generator, dtype=torch.float32)
                * 3.0
            ).requires_grad_(True)
            direct = actual.detach().clone().requires_grad_(True)
            fc = actual.detach().clone().requires_grad_(True)
            stable = subjective_opinion(actual, class_count)
            direct_belief, direct_uncertainty, direct_entropy = direct_opinion(
                direct, class_count
            )
            fc_values = reference.subjective_opinion(fc, class_count)

            stable_loss = stable["entropy"].mean()
            direct_loss = direct_entropy.mean()
            fc_loss = fc_values["entropy"].mean()
            stable_gradient = torch.autograd.grad(stable_loss, actual)[0]
            direct_gradient = torch.autograd.grad(direct_loss, direct)[0]
            fc_gradient = torch.autograd.grad(fc_loss, fc)[0]

            # Bit-exact against the audited FC implementation.
            assert torch.equal(stable["constrained_logits"], fc_values["constrained_logits"])
            assert torch.equal(stable["belief"], fc_values["belief"])
            assert torch.equal(stable["uncertainty"], fc_values["uncertainty"])
            assert torch.equal(stable["entropy"], fc_values["entropy"])
            assert torch.equal(stable_gradient, fc_gradient)

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
                assert value <= 1.0e-5, (name, value)
    return {"max_differences": maxima}


def check_large_logit_regression() -> None:
    """C04: direct exp overflows on VisDA-scale logits; the stable path does not."""
    large = torch.full((4, 12), -100.0, dtype=torch.float32)
    large[:, 0] = 100.0
    direct = large.clone().requires_grad_(True)
    stable_input = large.clone().requires_grad_(True)
    _, _, direct_entropy = direct_opinion(direct, 12)
    stable = subjective_opinion(stable_input, 12)
    stable_loss = stable["entropy"].mean()
    stable_gradient = torch.autograd.grad(stable_loss, stable_input)[0]
    assert not torch.isfinite(direct_entropy).all()
    assert torch.isfinite(stable_loss)
    assert torch.isfinite(stable["belief"]).all()
    assert torch.isfinite(stable["uncertainty"]).all()
    assert torch.isfinite(stable_gradient).all()
    # The loss, not only the forward masses, must stay usable.
    result = come_loss(large.clone().requires_grad_(True), 12)
    assert math.isfinite(result.diagnostics["loss"])


def check_detach_contract() -> None:
    """C05: tau=1 keeps the stop-gradient; the Jacobian is not the identity."""
    probe = torch.randn(5, 9, dtype=torch.float32, requires_grad=True)
    reference_probe = probe.detach().clone().requires_grad_(True)
    constrained = constrain_logits(probe)
    norm = torch.norm(reference_probe, p=2, dim=-1, keepdim=True)
    reference = reference_probe / norm * norm.detach()
    assert torch.equal(constrained, reference)
    constrained_gradient = torch.autograd.grad(constrained.square().sum(), probe)[0]
    reference_gradient = torch.autograd.grad(
        reference.square().sum(), reference_probe
    )[0]
    assert torch.equal(constrained_gradient, reference_gradient)

    identity_probe = probe.detach().clone().requires_grad_(True)
    identity_gradient = torch.autograd.grad(
        identity_probe.square().sum(), identity_probe
    )[0]
    assert not torch.allclose(constrained_gradient, identity_gradient)

    no_detach_probe = probe.detach().clone().requires_grad_(True)
    no_detach_norm = no_detach_probe.norm(p=2, dim=-1, keepdim=True)
    no_detach = no_detach_probe / no_detach_norm * no_detach_norm
    no_detach_gradient = torch.autograd.grad(
        no_detach.square().sum(), no_detach_probe
    )[0]
    assert not torch.allclose(constrained_gradient, no_detach_gradient)

    # Forward values match the raw logits at tau=1, but the loss gradient does
    # not equal the gradient of the same loss on unconstrained logits.
    logits = torch.randn(6, 12, requires_grad=True)
    shortcut = logits.detach().clone().requires_grad_(True)
    assert torch.allclose(constrain_logits(logits), logits, atol=1e-6)
    constrained_loss_gradient = torch.autograd.grad(
        come_loss(logits, 12).loss, logits
    )[0]
    log_c = shortcut.new_full((shortcut.shape[0], 1), math.log(12.0))
    log_strength = torch.logsumexp(torch.cat((shortcut, log_c), dim=1), dim=1, keepdim=True)
    opinion = torch.cat(
        (torch.exp(shortcut - log_strength), torch.exp(log_c - log_strength)), dim=1
    )
    entropy_input = opinion + 1.0e-7
    shortcut_gradient = torch.autograd.grad(
        -(entropy_input * torch.log(entropy_input)).sum(dim=1).mean(), shortcut
    )[0]
    assert not torch.allclose(constrained_loss_gradient, shortcut_gradient)


def check_no_norm_repair() -> dict:
    """C06: no epsilon/clamp; zero-norm and non-finite input fail loudly."""
    # Inspect the parsed operations, not the text: the docstrings and the
    # error messages mention these repairs precisely because the code must
    # never perform them.
    import ast

    module = ast.parse(
        (PROJECT_ROOT / "transformer_come" / "objective.py").read_text(encoding="utf-8")
    )
    forbidden_calls = {"clamp", "clamp_min", "clamp_max", "nan_to_num", "normalize"}
    checked = 0
    for node in module.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in {"constrain_logits", "subjective_opinion", "come_loss"}:
            continue
        detach_calls = 0
        for item in ast.walk(node):
            if isinstance(item, ast.Attribute):
                assert item.attr not in forbidden_calls, (node.name, item.attr)
                detach_calls += int(item.attr == "detach")
            if isinstance(item, ast.Name):
                assert item.id not in forbidden_calls, (node.name, item.id)
            if isinstance(item, ast.Constant) and isinstance(item.value, float):
                assert item.value != 1.0e-6, node.name
        if node.name == "constrain_logits":
            assert detach_calls == 1
        checked += 1
    assert checked == 3

    failures = 0
    zero = torch.zeros(3, 12, requires_grad=True)
    try:
        come_loss(zero, 12)
    except RuntimeError:
        failures += 1
    else:
        raise AssertionError("A zero logit norm did not fail loudly")
    for bad in (float("nan"), float("inf")):
        logits = torch.randn(3, 12)
        logits[0, 0] = bad
        try:
            come_loss(logits, 12)
        except RuntimeError:
            failures += 1
        else:
            raise AssertionError(f"Non-finite logits ({bad}) did not fail loudly")
    return {"fail_loudly_cases": failures}


def check_no_renormalization() -> dict:
    """C07: the epsilon shift is not renormalized and is used in both places."""
    class_count = 31
    logits = torch.randn(7, class_count) * 2.0
    values = subjective_opinion(logits, class_count)
    opinion_sum = values["opinion"].sum(dim=1)
    shifted_sum = values["entropy_input"].sum(dim=1)
    expected_shift = (class_count + 1) * 1.0e-7
    assert torch.allclose(opinion_sum, torch.ones_like(opinion_sum), atol=1e-6)
    assert torch.allclose(
        shifted_sum, torch.ones_like(shifted_sum) + expected_shift, atol=1e-6
    )
    assert not torch.allclose(
        shifted_sum, torch.ones_like(shifted_sum), rtol=0.0, atol=1.0e-9
    )
    manual = -(
        (values["opinion"] + 1.0e-7) * torch.log(values["opinion"] + 1.0e-7)
    ).sum(dim=1)
    assert torch.equal(values["entropy"], manual)
    renormalized = values["entropy_input"] / shifted_sum.unsqueeze(1)
    renormalized_entropy = -(renormalized * torch.log(renormalized)).sum(dim=1)
    renormalized_difference = (
        (values["entropy"] - renormalized_entropy).abs().max().item()
    )
    assert renormalized_difference > 0.0
    return {
        "opinion_row_sum_with_epsilon": float(shifted_sum[0].item()),
        "entropy_shift_from_renormalizing": renormalized_difference,
    }


def check_objective_inputs() -> None:
    """C08: the objective sees current logits and C only."""
    import inspect

    parameters = list(inspect.signature(come_loss).parameters)
    assert parameters[:2] == ["logits", "class_count"]
    assert not any(
        name in parameters for name in ("labels", "targets", "pseudo_labels", "teacher")
    )
    logits = torch.randn(8, 12, requires_grad=True)
    first = come_loss(logits, 12)
    second = come_loss(logits, 12)
    assert first.diagnostics["loss"] == second.diagnostics["loss"]
    assert set(first.diagnostics) == {
        "loss",
        "come_opinion_entropy",
        "come_mean_uncertainty_mass",
        "come_mean_log_strength",
        "class_count_c",
        "come_p",
        "come_tau",
        "come_entropy_epsilon",
        "finite_loss",
    }


def main() -> int:
    check_frozen_constants()
    class_report = check_class_count_contract()
    equivalence = check_moderate_equivalence()
    check_large_logit_regression()
    check_detach_contract()
    fail_loudly = check_no_norm_repair()
    renormalization = check_no_renormalization()
    check_objective_inputs()
    print(
        json.dumps(
            {
                "status": "PASS",
                "dtype": "float32",
                "class_counts": [12, 31],
                "seeds": [7, 79, 2026],
                **class_report,
                **equivalence,
                **fail_loudly,
                **renormalization,
                "fc_reference_bit_exact": True,
                "large_logit_overflow_regression": "PASS",
                "constrained_logit_detach_contract": "PASS",
            },
            sort_keys=True,
        )
    )
    print("Transformer COME objective tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
