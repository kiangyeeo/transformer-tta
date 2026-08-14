"""Deterministic VisDA-C target-order metadata."""

import torch


def build_target_order(seed, dataset_size):
    """Return a fixed permutation independent of global RNG consumption."""
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.randperm(dataset_size, generator=generator).tolist(), {
        "sampler": "fixed_random_permutation",
        "seed": int(seed),
        "adapter": "visda_fixed_seed_order",
    }
