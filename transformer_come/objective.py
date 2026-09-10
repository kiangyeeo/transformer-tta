"""Stable log-domain COME entropy-of-opinion objective (backbone-agnostic).

Ported from the audited stable FC implementation
``260817_iclr2027-refined/come_otta/objective.py`` so that the COME
mathematics exists exactly once per track and stays comparable across the
ResNet/FC and Transformer implementations.  Nothing here is
architecture-specific: the caller supplies current logits and the dataset
class count.

Frozen semantics (protocol sections 5-6, handoff pitfalls 1-6):

* ``p=2``, ``come.tau=1``, norm over the last (class) dimension, ``keepdim``.
* ``norm.detach()`` stays: at ``tau=1`` the forward value matches the raw
  logits but the Jacobian is ``I - z z^T / r^2``, which is *not* identity.
* No norm epsilon, no norm clamp, no ``normalize(..., eps=...)``, no
  ``nan_to_num``.
* The subjective opinion is built in the log domain; the direct
  ``exp(constrained)`` form overflows on VisDA float32 and must never come
  back.
* ``entropy_epsilon = 1e-7`` is added to the opinion and the result is *not*
  renormalized; the same shifted value is used as the multiplier and inside
  the log.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


OFFICIAL_COME_COMMIT = "409a19b71f62c765b1a5be62347a9455524ec176"
OFFICIAL_P = 2.0
OFFICIAL_TAU = 1.0
OFFICIAL_ENTROPY_EPSILON = 1.0e-7


@dataclass(frozen=True)
class COMEResult:
    """COME loss and detached scalar diagnostics for one objective call."""

    loss: torch.Tensor
    diagnostics: dict


def constrain_logits(logits, p=OFFICIAL_P, tau=OFFICIAL_TAU):
    """Apply the official logit constraint with the official detach location.

    The audited official implementation has no norm clamp/epsilon.  Keeping
    that behavior is part of the formal objective: non-finite values fail
    loudly in :func:`come_loss` instead of silently changing the objective.
    """

    norm = torch.norm(logits, p=p, dim=-1, keepdim=True)
    return logits / norm * norm.detach() * tau


def subjective_opinion(
    logits,
    class_count,
    p=OFFICIAL_P,
    tau=OFFICIAL_TAU,
    entropy_epsilon=OFFICIAL_ENTROPY_EPSILON,
):
    """Construct COME belief/uncertainty masses from constrained logits."""

    if logits.ndim != 2:
        raise ValueError("COME logits must have shape [batch, classes]")
    if isinstance(class_count, bool) or int(class_count) != class_count:
        raise ValueError("COME class_count must be an integer")
    class_count = int(class_count)
    if class_count < 2 or logits.shape[1] != class_count:
        raise ValueError(
            "COME class_count must equal the logits class dimension; "
            f"got C={class_count}, logits.shape={tuple(logits.shape)}"
        )
    constrained = constrain_logits(logits, p=p, tau=tau)
    log_c = constrained.new_full(
        (constrained.shape[0], 1), math.log(float(class_count))
    )
    log_strength = torch.logsumexp(
        torch.cat((constrained, log_c), dim=1), dim=1, keepdim=True
    )
    belief = torch.exp(constrained - log_strength)
    uncertainty = torch.exp(log_c - log_strength)
    opinion = torch.cat((belief, uncertainty), dim=1)
    # No renormalization after the epsilon shift: the shifted value is both
    # the multiplier and the log argument, and the row sum stays
    # 1 + (C + 1) * entropy_epsilon by construction.
    entropy_input = opinion + entropy_epsilon
    entropy = -(entropy_input * torch.log(entropy_input)).sum(dim=1)
    return {
        "constrained_logits": constrained,
        "log_evidence": constrained,
        "log_strength": log_strength,
        "belief": belief,
        "uncertainty": uncertainty,
        "opinion": opinion,
        "entropy_input": entropy_input,
        "entropy": entropy,
    }


def come_loss(
    logits,
    class_count,
    p=OFFICIAL_P,
    tau=OFFICIAL_TAU,
    entropy_epsilon=OFFICIAL_ENTROPY_EPSILON,
):
    """Return the mean COME entropy of opinion plus read-only diagnostics.

    Strengthening over the FC reference, required by protocol section 6: a
    non-finite input logit, a zero logit norm or a non-finite loss raises
    instead of being repaired.  Failing loudly is the contract; adding an
    epsilon/clamp or calling ``nan_to_num`` would change the formal
    objective, and skipping the batch would silently shorten the stream.
    """

    if not torch.isfinite(logits).all():
        raise RuntimeError(
            "COME received non-finite logits; the objective fails loudly "
            "instead of clamping or repairing them"
        )
    values = subjective_opinion(
        logits,
        class_count,
        p=p,
        tau=tau,
        entropy_epsilon=entropy_epsilon,
    )
    loss = values["entropy"].mean()
    if not torch.isfinite(loss.detach()).all():
        raise RuntimeError(
            "COME objective produced a non-finite loss (zero logit norm or "
            "unrepresentable norm); no epsilon/clamp/nan_to_num is applied"
        )
    with torch.no_grad():
        diagnostics = {
            "loss": float(loss.detach().item()),
            "come_opinion_entropy": float(values["entropy"].detach().mean().item()),
            "come_mean_uncertainty_mass": float(
                values["uncertainty"].detach().mean().item()
            ),
            "come_mean_log_strength": float(
                values["log_strength"].detach().mean().item()
            ),
            "class_count_c": int(class_count),
            "come_p": float(p),
            "come_tau": float(tau),
            "come_entropy_epsilon": float(entropy_epsilon),
            "finite_loss": True,
        }
    return COMEResult(loss=loss, diagnostics=diagnostics)


__all__ = [
    "COMEResult",
    "OFFICIAL_COME_COMMIT",
    "OFFICIAL_ENTROPY_EPSILON",
    "OFFICIAL_P",
    "OFFICIAL_TAU",
    "come_loss",
    "constrain_logits",
    "subjective_opinion",
]
