"""Formal controlled dense COME baseline on the common OTTA substrate."""

from .objective import come_loss, constrain_logits, subjective_opinion

__all__ = ["come_loss", "constrain_logits", "subjective_opinion"]
