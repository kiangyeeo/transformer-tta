"""COME current-state orchestration around the frozen corrected LBI engine."""

import math

import torch

from core.lbi import SplitLBIEngine

from .trainer import COMEObjectiveError


class COMELBIExecutor:
    """Run one local restart and one omega-only writeback per valid batch."""

    def __init__(self, total_model_param_count):
        self.engine = SplitLBIEngine(total_model_param_count)
        self.local_restart_count = 0
        self.support_discovery_count = 0
        self.omega_writeback_count = 0
        self.stage2_optimizer_instance_count = 0
        self.host_optimizer_persistent_step_count = 0

    def run_outer_batch(
        self, candidate_parameters, current_state_closure, runtime_config, timing=None
    ):
        candidate_parameters = list(candidate_parameters)
        self.local_restart_count += 1
        objective_evaluation_count = 0

        def current_state_gradient(named_parameters):
            nonlocal objective_evaluation_count
            for _, parameter in named_parameters:
                parameter.grad = None
            output = current_state_closure()
            objective_evaluation_count += 1
            loss = output[0] if isinstance(output, tuple) else output
            objective_diagnostics = output[1] if isinstance(output, tuple) else {}
            if not bool(torch.isfinite(loss.detach()).item()):
                raise COMEObjectiveError(
                    "COME-LBI objective is non-finite",
                    {
                        "objective_call_index": objective_evaluation_count,
                        "objective_diagnostics": objective_diagnostics,
                    },
                )
            loss.backward()
            for name, parameter in named_parameters:
                failure = {
                    "candidate_parameter": name,
                    "objective_call_index": objective_evaluation_count,
                    "objective_diagnostics": objective_diagnostics,
                }
                if parameter.grad is None:
                    raise COMEObjectiveError(
                        "COME-LBI missing candidate gradient", failure
                    )
                if not bool(torch.isfinite(parameter.grad).all().item()):
                    raise COMEObjectiveError(
                        "COME-LBI non-finite candidate gradient", failure
                    )
            return output

        def forbidden_cached_closure():
            raise RuntimeError(
                "COME-LBI requires current-state objective recomputation"
            )

        result = self.engine.run_step(
            candidate_parameters,
            forbidden_cached_closure,
            runtime_config,
            timing=timing,
            gradient_accumulator=current_state_gradient,
        )
        statistics = result.statistics
        if statistics["stage2_steps_completed"] != 1:
            raise RuntimeError("COME-LBI Stage 2 must complete exactly one step")
        if objective_evaluation_count != statistics["stage1_steps_completed"] + 1:
            raise RuntimeError("COME-LBI objective call count violates Stage1+Stage2")
        if statistics["stage1_support_count"] > statistics["stage1_max_support_count"]:
            raise RuntimeError("COME-LBI support exceeds the strict integer budget")

        # Dense interpolation may round base==refined by one ULP on FC.  Make
        # the frozen off-support writeback contract exact for both tracks.
        with torch.no_grad():
            for name, parameter in candidate_parameters:
                mask = result.state.mask[name].to(parameter.device, torch.bool)
                parameter.copy_(
                    torch.where(mask, parameter, result.base_parameters[name])
                )
                result.applied_parameters[name] = parameter.detach().clone()
                if not torch.equal(
                    parameter.detach()[~mask], result.base_parameters[name][~mask]
                ):
                    raise RuntimeError("COME-LBI off-mask writeback changed values")
                if not bool(torch.isfinite(parameter.detach()).all().item()):
                    raise COMEObjectiveError(
                        "COME-LBI persistent parameter update is non-finite",
                        {"candidate_parameter": name},
                    )

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
        self.stage2_optimizer_instance_count += 1
        selected = int(statistics["stage1_support_count"])
        maximum = int(statistics["stage1_max_support_count"])
        statistics.update(
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
                "stage1_cap_hit": statistics["stage1_stop_reason"]
                == "max_steps_reached",
                "support_utilization": float(selected / maximum) if maximum else 0.0,
                "overshoot": statistics["stage1_stop_reason"]
                == "strict_budget_rollback",
                "rollback": bool(statistics["stage1_rollback_used"]),
                "overshoot_rollback_status": "rollback"
                if statistics["stage1_rollback_used"]
                else "not_used",
                "lbi_local_restart_count": self.local_restart_count,
                "lbi_support_discovery_count": self.support_discovery_count,
                "lbi_omega_writeback_count": self.omega_writeback_count,
                "stage2_optimizer_instance_count": self.stage2_optimizer_instance_count,
                "host_optimizer_persistent_step_count": 0,
                "persistent_writeback": "lbi_omega_only",
                "objective_call_count": objective_evaluation_count,
                "objective_recomputation": (
                    "every_stage1_candidate_and_stage2_current_state"
                ),
                "stage2_initialization": "base_plus_masked_stage1_delta",
                "stage2_masked_delta_init": True,
                "stage2_objective_recomputed": True,
            }
        )
        if runtime_config.get("group_mode") == "out_channel":
            scalar_count = int(statistics["support_param_count"])
            statistics.update(
                {
                    "selected_param_count": scalar_count,
                    "selected_group_count": selected,
                    "realized_group_ratio": float(statistics["stage1_support_ratio"]),
                    "selected_scalar_count": scalar_count,
                    "realized_scalar_ratio": float(
                        statistics["support_over_scope_ratio"]
                    ),
                }
            )
        else:
            statistics.update(
                {
                    "selected_param_count": selected,
                    "selected_scalar_count": selected,
                    "realized_scalar_ratio": float(statistics["stage1_support_ratio"]),
                }
            )
        return result
