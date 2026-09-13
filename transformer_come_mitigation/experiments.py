"""Experiment suites for the isolated VisDA-C COME mitigation pilot."""

from __future__ import annotations


PROTOCOL_REVISION = "come_transformer_mitigation_pilot_20260911_v1"
IMPLEMENTATION_REVISION = "come_transformer_mitigation_pilot_impl_20260911_v1"

# A-D are the previously proposed low-LR/tau grid. E is an official-style
# LayerNorm-affine/SGD control (while retaining this repository's VisDA stream).
# F-H use the remaining GPUs to cover tau=.25 and global gradient clipping.
EXPERIMENTS = (
    {
        "id": "A_full_lr3e-6_tau1",
        "scope": "full_dense",
        "optimizer": "adamw",
        "lr": 3.0e-6,
        "tau": 1.0,
        "gradient_clip_norm": None,
    },
    {
        "id": "B_full_lr1e-6_tau1",
        "scope": "full_dense",
        "optimizer": "adamw",
        "lr": 1.0e-6,
        "tau": 1.0,
        "gradient_clip_norm": None,
    },
    {
        "id": "C_full_lr3e-6_tau0.5",
        "scope": "full_dense",
        "optimizer": "adamw",
        "lr": 3.0e-6,
        "tau": 0.5,
        "gradient_clip_norm": None,
    },
    {
        "id": "D_full_lr1e-6_tau0.5",
        "scope": "full_dense",
        "optimizer": "adamw",
        "lr": 1.0e-6,
        "tau": 0.5,
        "gradient_clip_norm": None,
    },
    {
        "id": "E_norm_sgd_lr1e-3_tau1",
        "scope": "layernorm_affine",
        "optimizer": "sgd",
        "lr": 1.0e-3,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "tau": 1.0,
        "gradient_clip_norm": None,
    },
    {
        "id": "F_full_lr3e-6_tau0.25",
        "scope": "full_dense",
        "optimizer": "adamw",
        "lr": 3.0e-6,
        "tau": 0.25,
        "gradient_clip_norm": None,
    },
    {
        "id": "G_full_lr3e-6_tau1_clip1",
        "scope": "full_dense",
        "optimizer": "adamw",
        "lr": 3.0e-6,
        "tau": 1.0,
        "gradient_clip_norm": 1.0,
    },
    {
        "id": "H_full_lr3e-6_tau0.5_clip1",
        "scope": "full_dense",
        "optimizer": "adamw",
        "lr": 3.0e-6,
        "tau": 0.5,
        "gradient_clip_norm": 1.0,
    },
)

TAU_REFINEMENT_EXPERIMENTS = (
    {
        "id": "T09_full_lr1e-6_tau0.9",
        "scope": "full_dense",
        "optimizer": "adamw",
        "lr": 1.0e-6,
        "tau": 0.9,
        "gradient_clip_norm": None,
    },
    {
        "id": "T09_full_lr3e-6_tau0.9",
        "scope": "full_dense",
        "optimizer": "adamw",
        "lr": 3.0e-6,
        "tau": 0.9,
        "gradient_clip_norm": None,
    },
    {
        "id": "T08_full_lr1e-6_tau0.8",
        "scope": "full_dense",
        "optimizer": "adamw",
        "lr": 1.0e-6,
        "tau": 0.8,
        "gradient_clip_norm": None,
    },
    {
        "id": "T08_full_lr3e-6_tau0.8",
        "scope": "full_dense",
        "optimizer": "adamw",
        "lr": 3.0e-6,
        "tau": 0.8,
        "gradient_clip_norm": None,
    },
)

LR_REFINEMENT_EXPERIMENTS = (
    {
        "id": "LR1e-7_full_tau1",
        "scope": "full_dense",
        "optimizer": "adamw",
        "lr": 1.0e-7,
        "tau": 1.0,
        "gradient_clip_norm": None,
    },
    {
        "id": "LR3e-7_full_tau1",
        "scope": "full_dense",
        "optimizer": "adamw",
        "lr": 3.0e-7,
        "tau": 1.0,
        "gradient_clip_norm": None,
    },
    {
        "id": "LR5e-7_full_tau1",
        "scope": "full_dense",
        "optimizer": "adamw",
        "lr": 5.0e-7,
        "tau": 1.0,
        "gradient_clip_norm": None,
    },
)

MECHANISM_CONTROL_EXPERIMENTS = (
    {
        "id": "MC_candidate_lr1e-6_tau1",
        "scope": "candidate_dense",
        "optimizer": "adamw",
        "lr": 1.0e-6,
        "tau": 1.0,
        "gradient_clip_norm": None,
        "reset_adam_moments_each_batch": False,
    },
    {
        "id": "MC_full_lr1e-6_tau1_reset-moments",
        "scope": "full_dense",
        "optimizer": "adamw",
        "lr": 1.0e-6,
        "tau": 1.0,
        "gradient_clip_norm": None,
        "reset_adam_moments_each_batch": True,
    },
)

EXPERIMENT_SUITES = {
    "initial-8gpu": EXPERIMENTS,
    "tau-refinement": TAU_REFINEMENT_EXPERIMENTS,
    "lr-refinement": LR_REFINEMENT_EXPERIMENTS,
    "mechanism-controls": MECHANISM_CONTROL_EXPERIMENTS,
}


def experiment_by_id(experiment_id: str) -> dict:
    matches = [
        item
        for experiments in EXPERIMENT_SUITES.values()
        for item in experiments
        if item["id"] == experiment_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Unknown mitigation experiment: {experiment_id}")
    return dict(matches[0])


__all__ = [
    "EXPERIMENTS",
    "EXPERIMENT_SUITES",
    "IMPLEMENTATION_REVISION",
    "LR_REFINEMENT_EXPERIMENTS",
    "MECHANISM_CONTROL_EXPERIMENTS",
    "PROTOCOL_REVISION",
    "TAU_REFINEMENT_EXPERIMENTS",
    "experiment_by_id",
]
