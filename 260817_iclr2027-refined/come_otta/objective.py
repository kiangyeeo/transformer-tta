"""Official COME entropy-of-opinion objective with dataset-derived class count."""

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
    """Apply Eq. 9 with the official detach location and norm dimension.

    The audited official implementation has no norm clamp/epsilon.  Keeping
    that behavior is part of the formal baseline: non-finite values fail
    loudly in the trainer instead of silently changing the objective.
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
            f"got K={class_count}, logits.shape={tuple(logits.shape)}"
        )
    constrained = constrain_logits(logits, p=p, tau=tau)
    log_k = constrained.new_full(
        (constrained.shape[0], 1), math.log(float(class_count))
    )
    log_strength = torch.logsumexp(
        torch.cat((constrained, log_k), dim=1), dim=1, keepdim=True
    )
    belief = torch.exp(constrained - log_strength)
    uncertainty = torch.exp(log_k - log_strength)
    opinion = torch.cat((belief, uncertainty), dim=1)
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


def come_loss(logits, class_count, p=OFFICIAL_P, tau=OFFICIAL_TAU):
    """Return mean official COME entropy of opinion and read-only diagnostics."""

    values = subjective_opinion(logits, class_count, p=p, tau=tau)
    loss = values["entropy"].mean()
    with torch.no_grad():
        diagnostics = {
            "loss": float(loss.detach().item()),
            "come_opinion_entropy": float(
                values["entropy"].detach().mean().item()
            ),
            "come_mean_uncertainty_mass": float(
                values["uncertainty"].detach().mean().item()
            ),
            "class_count_k": int(class_count),
            "p": float(p),
            "tau": float(tau),
            "finite_loss": bool(torch.isfinite(loss.detach()).item()),
        }
    return COMEResult(loss=loss, diagnostics=diagnostics)
