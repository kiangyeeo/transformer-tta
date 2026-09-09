"""NCTTA current-state orchestration around the shared corrected LBI engine."""

import math

import torch

from core.lbi import SplitLBIEngine
from .objective import NCTTAObjectiveError


class NCTTALBIExecutor:
    """One local restart and omega-only writeback per valid outer batch."""

    def __init__(self, total_model_param_count):
        self.engine = SplitLBIEngine(total_model_param_count)
        self.support_discovery_count = 0
        self.omega_writeback_count = 0
        self.host_optimizer_persistent_step_count = 0

    def run_outer_batch(
        self, candidate_parameters, current_state_closure, runtime_config, timing=None
    ):
        candidate_parameters = list(candidate_parameters)
        objective_evaluation_count = 0

        def current_state_gradient(named_parameters):
            nonlocal objective_evaluation_count
            for _, parameter in named_parameters:
                parameter.grad = None
            output = current_state_closure()
            objective_evaluation_count += 1
            loss = output[0] if isinstance(output, tuple) else output
            objective_diagnostics = output[1] if isinstance(output, tuple) else {}
            loss.backward()
            for name, parameter in named_parameters:
                failure = {
                    "candidate_parameter": name,
                    "objective_call_index": objective_evaluation_count,
                    "objective_diagnostics": objective_diagnostics,
                }
                if parameter.grad is None:
                    raise NCTTAObjectiveError(
                        "NCTTA-LBI missing candidate gradient", failure
                    )
                if not bool(torch.isfinite(parameter.grad).all().item()):
                    raise NCTTAObjectiveError(
                        "NCTTA-LBI non-finite candidate gradient", failure
                    )
            return output

        def forbidden_cached_closure():
            raise RuntimeError(
                "NCTTA-LBI requires current-state objective recomputation"
            )

        result = self.engine.run_step(
            candidate_parameters,
            forbidden_cached_closure,
            runtime_config,
            timing=timing,
            gradient_accumulator=current_state_gradient,
        )
        if result.statistics["stage2_steps_completed"] != 1:
            raise RuntimeError("NCTTA-LBI Stage 2 must complete exactly one step")
        # The scalar engine's dense arithmetic can round base==refined by one
        # ULP. Restore off-support values so NCTTA's scope contract is exact.
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
            nonzero_count += int(difference.abs().gt(tolerance).count_nonzero().item())
            l1 += float(difference.abs().sum().item())
            squared_l2 += float((difference * difference).sum().item())
        self.support_discovery_count += 1
        self.omega_writeback_count += 1
        max_support = int(result.statistics["stage1_max_support_count"])
        selected = int(result.statistics["stage1_support_count"])
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
                "stage1_cap_hit": result.statistics["stage1_stop_reason"]
                == "max_steps_reached",
                "support_utilization": float(selected / max_support)
                if max_support
                else 0.0,
                "overshoot_rollback_status": "rollback"
                if result.statistics["stage1_rollback_used"]
                else "not_used",
                "lbi_support_discovery_count": self.support_discovery_count,
                "lbi_omega_writeback_count": self.omega_writeback_count,
                "host_optimizer_persistent_step_count": 0,
                "persistent_writeback": "lbi_omega_only",
                "objective_recomputation": "every_stage1_candidate_and_stage2_current_state",
            }
        )
        if runtime_config.get("group_mode") == "out_channel":
            scalar_count = int(result.statistics["support_param_count"])
            result.statistics.update(
                {
                    "selected_param_count": scalar_count,
                    "selected_group_count": selected,
                    "realized_group_ratio": float(
                        result.statistics["stage1_support_ratio"]
                    ),
                    "selected_scalar_count": scalar_count,
                    "realized_scalar_ratio": float(
                        result.statistics["support_over_scope_ratio"]
                    ),
                }
            )
        else:
            result.statistics.update(
                {
                    "selected_param_count": selected,
                    "selected_scalar_count": selected,
                    "realized_scalar_ratio": float(
                        result.statistics["stage1_support_ratio"]
                    ),
                }
            )
        return result
