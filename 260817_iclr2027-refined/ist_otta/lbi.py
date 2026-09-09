"""IST-specific orchestration around the shared corrected LBI engine."""


import math

import torch

from core.lbi import SplitLBIEngine


class ISTLBIExecutor:
    """One support discovery and one omega writeback per valid outer batch."""

    def __init__(self, total_model_param_count):
        self.engine = SplitLBIEngine(total_model_param_count)
        self.support_discovery_count = 0
        self.omega_writeback_count = 0
        self.native_ema_commit_count = 0

    def run_outer_batch(
        self,
        candidate_parameters,
        gradient_accumulator,
        runtime_config,
        timing=None,
    ):
        candidate_parameters = list(candidate_parameters)

        def forbidden_non_accumulated_closure():
            raise RuntimeError(
                "IST-LBI must use the full-objective gradient accumulator"
            )

        result = self.engine.run_step(
            candidate_parameters,
            forbidden_non_accumulated_closure,
            runtime_config,
            timing=timing,
            gradient_accumulator=gradient_accumulator,
        )
        if result.statistics["stage2_steps_completed"] != 1:
            raise RuntimeError("IST-LBI Stage 2 must complete exactly one step")
        # The shared FC engine preserves legacy SHOT arithmetic, whose dense
        # interpolation can round base==refined off-mask coordinates by 1 ULP.
        # IST's stricter contract restores those coordinates bit-exactly.
        with torch.no_grad():
            for name, parameter in candidate_parameters:
                mask = result.state.mask[name].to(parameter.device, torch.bool)
                parameter.copy_(
                    torch.where(mask, parameter, result.base_parameters[name])
                )
                result.applied_parameters[name] = parameter.detach().clone()
        tolerance = float(runtime_config["delta_nonzero_tolerance"])
        nonzero_count = 0
        l1 = 0.0
        squared_l2 = 0.0
        candidate_count = 0
        for name, parameter in candidate_parameters:
            difference = parameter.detach() - result.base_parameters[name]
            candidate_count += parameter.numel()
            nonzero_count += int(
                difference.abs().gt(tolerance).count_nonzero().item()
            )
            l1 += float(difference.abs().sum().item())
            squared_l2 += float((difference * difference).sum().item())
        result.statistics.update(
            {
                "applied_update_nonzero_count": nonzero_count,
                "applied_update_over_scope_ratio": float(
                    nonzero_count / candidate_count
                ),
                "applied_update_over_model_ratio": float(
                    nonzero_count / self.engine.total_model_param_count
                ),
                "applied_update_l1": l1,
                "applied_update_l2": math.sqrt(squared_l2),
                "off_mask_exact_preservation": True,
            }
        )
        self.support_discovery_count += 1
        self.omega_writeback_count += 1
        max_support = int(
            result.statistics["stage1_max_support_count"]
        )
        selected_support = int(
            result.statistics["stage1_support_count"]
        )
        result.statistics.update(
            {
                "stage1_cap_hit": (
                    result.statistics["stage1_stop_reason"]
                    == "max_steps_reached"
                ),
                "support_utilization": (
                    float(selected_support / max_support)
                    if max_support > 0 else 0.0
                ),
                "overshoot_rollback_status": (
                    "rollback"
                    if result.statistics["stage1_rollback_used"]
                    else "not_used"
                ),
                "lbi_support_discovery_count": self.support_discovery_count,
                "lbi_omega_writeback_count": self.omega_writeback_count,
                "native_ist_ema_commit_count": 0,
                "native_ist_ema": False,
                "persistent_writeback": "lbi_omega_only",
            }
        )
        return result
