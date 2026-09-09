"""One-shot outer-batch parameter/state moving average for IST."""

import torch


class OuterBatchEMA:
    """Anchor independent clones, then apply EMA exactly once per batch."""

    def __init__(self, momentum):
        self.momentum = float(momentum)
        if not 0.0 <= self.momentum <= 1.0:
            raise ValueError("EMA momentum must be in [0, 1]")
        self._anchor = None
        self._batch_index = None
        self.commit_count = 0

    @property
    def anchor(self):
        return self._anchor

    def state_dict(self):
        if self._anchor is not None or self._batch_index is not None:
            raise RuntimeError("EMA may be checkpointed only between batches")
        return {
            "momentum": self.momentum,
            "commit_count": self.commit_count,
        }

    def load_state_dict(self, state):
        if float(state["momentum"]) != self.momentum:
            raise RuntimeError("EMA checkpoint momentum mismatch")
        self._anchor = None
        self._batch_index = None
        self.commit_count = int(state["commit_count"])
        if self.commit_count < 0:
            raise RuntimeError("EMA checkpoint has invalid commit_count")

    def begin_batch(self, batch_index, named_models):
        if self._anchor is not None:
            raise RuntimeError("previous EMA anchor has not been committed")
        if int(batch_index) != self.commit_count:
            raise RuntimeError("EMA batches must be committed in order")
        self._batch_index = int(batch_index)
        self._anchor = {
            model_name: {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
            for model_name, model in named_models
        }

    def commit(self, batch_index, named_models):
        if self._anchor is None or int(batch_index) != self._batch_index:
            raise RuntimeError("EMA commit does not match its batch anchor")
        momentum = self.momentum
        with torch.no_grad():
            for model_name, model in named_models:
                raw_state = model.state_dict()
                old_state = self._anchor[model_name]
                trainable_parameter_names = {
                    name
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                }
                averaged = {}
                for name, raw_value in raw_state.items():
                    if (
                        name in trainable_parameter_names
                        and torch.is_floating_point(raw_value)
                    ):
                        old_value = old_state[name].to(raw_value.device)
                        averaged[name] = (
                            momentum * old_value
                            + (1.0 - momentum) * raw_value.detach()
                        )
                    else:
                        # Frozen parameters and all buffers must retain their
                        # raw state.  Applying arithmetic to an unchanged
                        # frozen tensor would create an out-of-scope update.
                        averaged[name] = raw_value.detach().clone()
                model.load_state_dict(averaged, strict=True)
        self._anchor = None
        self._batch_index = None
        self.commit_count += 1
