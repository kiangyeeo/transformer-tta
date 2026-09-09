#!/usr/bin/env python3
"""Validate an experiment matrix and generate a non-executing plan."""

import argparse
import copy
import json
import os
import os.path as osp
import shlex
import sys
from itertools import permutations

import yaml


PROJECT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
WORKSPACE_ROOT = osp.dirname(PROJECT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from experiment_identity import (  # noqa: E402
    IMPLEMENTATION_REVISION,
    SUPPORTED_VARIANTS,
    build_experiment_identity,
)
from protocol_constants import (  # noqa: E402
    EFFICIENCY_PROTOCOL_REVISION,
    FORMAL_SEED,
    PROTOCOL_REVISION,
    SOURCE_CHECKPOINT_REVISION,
)
from shot_otta.config import (  # noqa: E402
    DATASETS,
    load_yaml,
    resolve_effective_config,
    validate_lbi_config,
)
from shot_otta.artifacts import experiment_output_root  # noqa: E402


SPARSE_VARIANTS = {
    "module_random",
    "module_magnitude",
    "module_saliency",
    "module_lbi",
}
DENSE_OR_SOURCE_VARIANTS = {
    "source_only",
    "full_dense",
    "module_dense",
}
COMMON_TRAINING_KEYS = {
    "batch_size",
    "workers",
    "gpu_id",
    "save_model",
}
LBI_CONFIG_FIELDS = (
    "alpha",
    "kappa",
    "nu",
    "omega",
    "stage1_max_steps",
    "budget_tolerance",
    "stage2_lr",
    "stage2_steps",
    "delta_nonzero_tolerance",
    "support_threshold",
)

ACTIVE_VISDA_BATCH_PATH = osp.join(
    PROJECT_DIR,
    "experiment_logs",
    "CURRENT_VISDA_BATCH_SIZE.txt",
)


def _active_visda_batch_size():
    try:
        with open(ACTIVE_VISDA_BATCH_PATH, "r", encoding="utf-8") as file_obj:
            return int(file_obj.read().strip())
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(
            "VISDA-C planning requires a valid frozen batch size at "
            f"{ACTIVE_VISDA_BATCH_PATH}"
        ) from error


def _validate_visda_batch_size(matrix):
    if "VISDA-C" not in matrix.get("datasets", {}):
        return
    expected = 256
    actual = matrix.get("common_training", {}).get("batch_size", expected)
    if int(actual) != expected:
        raise ValueError(
            "VISDA-C matrix batch_size does not match the active frozen "
            f"batch size: matrix={actual}, frozen={expected}"
        )


def _require_unique(values, label):
    seen = set()
    duplicates = []
    for value in values:
        comparable = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        )
        if comparable in seen:
            duplicates.append(value)
        seen.add(comparable)
    if duplicates:
        raise ValueError(f"Duplicate {label}: {duplicates}")


def _expand_transfers(dataset, transfer_spec):
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset in matrix: {dataset}")
    domain_count = len(DATASETS[dataset]["domains"])
    if transfer_spec == "all_ordered_pairs":
        return [
            [source, target]
            for source, target in permutations(range(domain_count), 2)
        ]
    if not isinstance(transfer_spec, list) or not transfer_spec:
        raise ValueError(
            f"{dataset}.transfers must be a non-empty list or "
            "all_ordered_pairs"
        )
    transfers = []
    for transfer in transfer_spec:
        if not isinstance(transfer, list) or len(transfer) != 2:
            raise ValueError(
                f"Invalid transfer for {dataset}: {transfer}"
            )
        source, target = transfer
        if not isinstance(source, int) or not isinstance(target, int):
            raise ValueError(
                f"Transfer indices must be integers: {transfer}"
            )
        if source == target:
            raise ValueError(
                f"source == target is forbidden: {dataset} {transfer}"
            )
        if not 0 <= source < domain_count:
            raise ValueError(f"Invalid source for {dataset}: {source}")
        if not 0 <= target < domain_count:
            raise ValueError(f"Invalid target for {dataset}: {target}")
        transfers.append([source, target])
    _require_unique(transfers, f"transfer in {dataset}")
    return transfers


def _normalize_lbi_trials(spec):
    has_lbi = "lbi" in spec
    has_lbi_trials = "lbi_trials" in spec
    if has_lbi and has_lbi_trials:
        raise ValueError(
            "module_lbi must define exactly one of lbi or lbi_trials; "
            "both were provided"
        )
    if not has_lbi and not has_lbi_trials:
        raise ValueError(
            "module_lbi must define exactly one of lbi or lbi_trials"
        )

    if has_lbi:
        normalized = copy.deepcopy(spec["lbi"])
        validate_lbi_config(normalized)
        spec["lbi"] = normalized
        return [normalized]

    trials = spec["lbi_trials"]
    if not isinstance(trials, list) or not trials:
        raise ValueError(
            "module_lbi lbi_trials must be a non-empty list"
        )
    normalized_trials = []
    seen = {}
    for index, trial in enumerate(trials):
        if not isinstance(trial, dict):
            raise ValueError(
                f"LBI trial at index {index} must be a mapping"
            )
        normalized = copy.deepcopy(trial)
        try:
            validate_lbi_config(normalized)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid LBI trial at index {index}: {error}"
            ) from error
        comparable = tuple(
            normalized[field] for field in LBI_CONFIG_FIELDS
        )
        if comparable in seen:
            raise ValueError(
                "Duplicate LBI trial configurations at indices "
                f"{seen[comparable]} and {index}"
            )
        seen[comparable] = index
        normalized_trials.append(normalized)
    spec["lbi_trials"] = normalized_trials
    return normalized_trials


def _normalize_lbi_budget_trials(spec):
    """Validate explicitly paired requested-budget / LBI trial entries."""
    trials = spec["lbi_budget_trials"]
    if not isinstance(trials, list) or not trials:
        raise ValueError(
            "module_lbi lbi_budget_trials must be a non-empty list"
        )
    normalized_trials = []
    seen = set()
    for index, trial in enumerate(trials):
        if not isinstance(trial, dict):
            raise ValueError(
                f"LBI budget trial at index {index} must be a mapping"
            )
        if "budget" not in trial:
            raise ValueError(
                f"LBI budget trial at index {index} must define budget"
            )
        budget = float(trial["budget"])
        if not 0.0 <= budget <= 1.0:
            raise ValueError(
                f"LBI budget trial at index {index} has invalid budget: "
                f"{budget}"
            )
        lbi = copy.deepcopy(trial)
        del lbi["budget"]
        try:
            validate_lbi_config(lbi)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid LBI budget trial at index {index}: {error}"
            ) from error
        comparable = (budget,) + tuple(lbi[field] for field in LBI_CONFIG_FIELDS)
        if comparable in seen:
            raise ValueError(
                "Duplicate LBI budget trial configuration at index "
                f"{index}"
            )
        seen.add(comparable)
        normalized_trials.append((budget, lbi))
    spec["_normalized_lbi_budget_trials"] = normalized_trials


def _validate_variant_specs(variants):
    if not isinstance(variants, dict) or not variants:
        raise ValueError("variants must be a non-empty mapping")
    unsupported = sorted(set(variants) - SUPPORTED_VARIANTS)
    if unsupported:
        raise ValueError(f"Unsupported variants: {unsupported}")
    for variant, spec in variants.items():
        if spec is None:
            spec = {}
            variants[variant] = spec
        if not isinstance(spec, dict):
            raise ValueError(f"{variant} config must be a mapping")
        if variant != "module_lbi":
            forbidden_lbi_blocks = [
                key for key in ("lbi", "lbi_trials") if key in spec
            ]
            if forbidden_lbi_blocks:
                raise ValueError(
                    f"{variant} must not define "
                    f"{', '.join(forbidden_lbi_blocks)}"
                )
        if variant in DENSE_OR_SOURCE_VARIANTS:
            if "budgets" in spec:
                raise ValueError(
                    f"{variant} must not define a budget sweep"
                )
            if "selection_seed_mode" in spec:
                raise ValueError(
                    f"{variant} must not define selection_seed_mode"
                )
            continue
        if variant == "module_lbi" and spec.get("lbi_tuned") == "unresolved":
            conflicting = [
                key for key in ("lbi", "lbi_trials", "lbi_budget_trials")
                if key in spec
            ]
            if conflicting:
                raise ValueError(
                    "unresolved module_lbi cannot define "
                    f"{', '.join(conflicting)}"
                )
            budgets = spec.get("budgets")
            if not isinstance(budgets, list) or not budgets:
                raise ValueError(
                    "unresolved module_lbi requires a non-empty budgets list"
                )
            spec["budgets"] = [float(value) for value in budgets]
            _require_unique(spec["budgets"], "budget in module_lbi")
            spec["_unresolved_lbi"] = True
            continue
        paired_lbi_trials = (
            variant == "module_lbi" and "lbi_budget_trials" in spec
        )
        if paired_lbi_trials:
            conflicting = [
                key for key in ("budgets", "lbi", "lbi_trials")
                if key in spec
            ]
            if conflicting:
                raise ValueError(
                    "module_lbi lbi_budget_trials must not be combined "
                    f"with {', '.join(conflicting)}"
                )
            _normalize_lbi_budget_trials(spec)
        else:
            budgets = spec.get("budgets")
            if not isinstance(budgets, list) or not budgets:
                raise ValueError(
                    f"{variant} requires a non-empty budgets list"
                )
            normalized_budgets = [float(value) for value in budgets]
            for budget in normalized_budgets:
                if not 0.0 <= budget <= 1.0:
                    raise ValueError(
                        f"{variant} budget must be in [0, 1]: {budget}"
                    )
            _require_unique(normalized_budgets, f"budget in {variant}")
            spec["budgets"] = normalized_budgets
        if variant == "module_random":
            mode = spec.get("selection_seed_mode")
            if mode != "same_as_run_seed":
                raise ValueError(
                    "module_random selection_seed_mode must be "
                    "same_as_run_seed"
                )
            num_random_masks = spec.get("num_random_masks", 3)
            if (
                isinstance(num_random_masks, bool)
                or int(num_random_masks) != num_random_masks
                or int(num_random_masks) <= 0
            ):
                raise ValueError(
                    "module_random num_random_masks must be a positive integer"
                )
            spec["num_random_masks"] = int(num_random_masks)
        elif "selection_seed_mode" in spec:
            raise ValueError(
                f"{variant} must not define selection_seed_mode"
            )
        if variant == "module_lbi" and not paired_lbi_trials:
            spec["_normalized_lbi_trials"] = (
                _normalize_lbi_trials(spec)
            )


def validate_matrix(matrix):
    if matrix.get("method") != "shot":
        raise ValueError("Matrix currently supports method=shot only")
    if matrix.get("task") != "otta":
        raise ValueError("Matrix currently supports task=otta only")
    if matrix.get("formal_protocol"):
        if matrix.get("protocol_revision") != PROTOCOL_REVISION:
            raise ValueError("formal matrix protocol_revision is not current")
        if matrix.get("seeds") != [FORMAL_SEED]:
            raise ValueError(f"formal matrix requires seeds=[{FORMAL_SEED}]")
    seeds = matrix.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a non-empty list")
    if not all(isinstance(seed, int) for seed in seeds):
        raise ValueError("Every seed must be an integer")
    _require_unique(seeds, "seed")
    datasets = matrix.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("datasets must be a non-empty mapping")
    normalized_transfers = {}
    for dataset, spec in datasets.items():
        if not isinstance(spec, dict) or "transfers" not in spec:
            raise ValueError(f"{dataset} must define transfers")
        normalized_transfers[dataset] = _expand_transfers(
            dataset,
            spec["transfers"],
        )
    _validate_visda_batch_size(matrix)
    variants = matrix.get("variants")
    _validate_variant_specs(variants)
    common = matrix.get("common_training", {})
    if not isinstance(common, dict):
        raise ValueError("common_training must be a mapping")
    unsupported_common = sorted(set(common) - COMMON_TRAINING_KEYS)
    if unsupported_common:
        raise ValueError(
            f"Unsupported common_training keys: {unsupported_common}"
        )
    if not isinstance(matrix.get("output_root"), str):
        raise ValueError("output_root must be a string")
    return normalized_transfers


def _variant_points(variant, spec, seed):
    if variant == "source_only":
        return [(None, None, None, None, None)]
    if variant in {"full_dense", "module_dense"}:
        return [(1.0, None, None, None, None)]
    if variant == "module_random":
        return [
            (
                budget,
                seed,
                spec["selection_seed_mode"],
                None,
                None,
            )
            for budget in spec["budgets"]
        ]
    if variant == "module_lbi":
        if spec.get("_unresolved_lbi"):
            return [
                (budget, None, None, None, None)
                for budget in spec["budgets"]
            ]
        if "_normalized_lbi_budget_trials" in spec:
            return [
                (budget, None, None, lbi, trial_index)
                for trial_index, (budget, lbi) in enumerate(
                    spec["_normalized_lbi_budget_trials"]
                )
            ]
        return [
            (budget, None, None, lbi, trial_index)
            for budget in spec["budgets"]
            for trial_index, lbi in enumerate(
                spec["_normalized_lbi_trials"]
            )
        ]
    return [
        (budget, None, None, None, None)
        for budget in spec["budgets"]
    ]


def _apply_plan_config(
    base_config,
    matrix,
    dataset,
    source,
    target,
    seed,
    variant,
    requested_budget,
    selection_seed,
    num_random_masks,
    lbi,
):
    effective = copy.deepcopy(base_config)
    effective["method"] = matrix["method"]
    effective["task"] = matrix["task"]
    effective["seed"] = seed
    effective["variant"] = variant
    effective["requested_budget"] = requested_budget
    effective["selection_seed"] = selection_seed
    effective["allow_unresolved_lbi"] = lbi is None
    if variant == "module_random":
        effective["num_random_masks"] = num_random_masks
    if variant == "module_lbi":
        effective["lbi"] = copy.deepcopy(lbi)
    effective["data"]["dataset"] = dataset
    effective["data"]["source"] = source
    effective["data"]["target"] = target
    common = matrix.get("common_training", {})
    if "batch_size" in common:
        effective["data"]["batch_size"] = int(common["batch_size"])
    if "workers" in common:
        effective["data"]["workers"] = int(common["workers"])
    if "gpu_id" in common:
        effective["device"]["gpu_id"] = str(common["gpu_id"])
    if "save_model" in common:
        effective["output"]["save_model"] = bool(common["save_model"])
    effective["output"]["root"] = matrix["output_root"]
    return resolve_effective_config(effective, WORKSPACE_ROOT)


def _effective_overrides(
    matrix,
    dataset,
    source,
    target,
    seed,
    variant,
    requested_budget,
    selection_seed,
    num_random_masks,
    lbi,
):
    overrides = {
        "dataset": dataset,
        "source": source,
        "target": target,
        "seed": seed,
        "variant": variant,
        "requested_budget": requested_budget,
        "selection_seed": selection_seed,
        "num_random_masks": num_random_masks,
        "output_root": matrix["output_root"],
        **copy.deepcopy(matrix.get("common_training", {})),
    }
    if variant == "module_lbi":
        overrides["lbi"] = copy.deepcopy(lbi)
    return overrides


def _command_arguments(
    base_config_path,
    matrix,
    overrides,
    identity,
):
    train_path = osp.join(PROJECT_DIR, "train.py")
    arguments = [
        "python",
        train_path,
        "--config",
        osp.abspath(base_config_path),
        "--dataset",
        overrides["dataset"],
        "--source",
        str(overrides["source"]),
        "--target",
        str(overrides["target"]),
        "--seed",
        str(overrides["seed"]),
        "--variant",
        overrides["variant"],
        "--output-root",
        matrix["output_root"],
    ]
    common = matrix.get("common_training", {})
    if "batch_size" in common:
        arguments.extend(["--batch-size", str(common["batch_size"])])
    if "workers" in common:
        arguments.extend(["--workers", str(common["workers"])])
    if "gpu_id" in common:
        arguments.extend(["--gpu-id", str(common["gpu_id"])])
    if common.get("save_model") is True:
        arguments.append("--save-model")
    elif common.get("save_model") is False:
        arguments.append("--no-save-model")
    if overrides["variant"] in SPARSE_VARIANTS:
        arguments.extend(
            [
                "--requested-budget",
                str(overrides["requested_budget"]),
            ]
        )
    if overrides["variant"] == "module_lbi":
        lbi = overrides["lbi"]
        if lbi is not None:
            arguments.extend(
                [
                "--lbi-alpha",
                str(lbi["alpha"]),
                "--lbi-kappa",
                str(lbi["kappa"]),
                "--lbi-nu",
                str(lbi["nu"]),
                "--lbi-omega",
                str(lbi["omega"]),
                "--lbi-stage1-max-steps",
                str(lbi["stage1_max_steps"]),
                "--lbi-budget-tolerance",
                str(lbi["budget_tolerance"]),
                "--lbi-stage2-lr",
                str(lbi["stage2_lr"]),
                "--lbi-stage2-steps",
                str(lbi["stage2_steps"]),
                "--lbi-delta-nonzero-tolerance",
                str(lbi["delta_nonzero_tolerance"]),
                "--lbi-support-threshold",
                str(lbi["support_threshold"]),
                ]
            )
    if overrides["variant"] == "module_random":
        arguments.extend(
            [
                "--selection-seed", str(overrides["selection_seed"]),
                "--num-random-masks", str(overrides["num_random_masks"]),
            ]
        )
    arguments.extend(
        [
            "--experiment-key",
            identity["experiment_key"],
            "--experiment-config-sha256",
            identity["experiment_config_sha256"],
        ]
    )
    return arguments


def _validate_unique_identities(entries):
    keys = [entry["experiment_key"] for entry in entries]
    hashes = [
        entry["experiment_config_sha256"] for entry in entries
    ]
    _require_unique(keys, "experiment_key")
    _require_unique(hashes, "experiment_config_sha256")


def build_plan(matrix, base_config, base_config_path):
    transfers_by_dataset = validate_matrix(matrix)
    if base_config.get("method") != matrix["method"]:
        raise ValueError("Base config method does not match matrix")
    if base_config.get("task") != matrix["task"]:
        raise ValueError("Base config task does not match matrix")
    entries = []
    for dataset, transfers in transfers_by_dataset.items():
        for source, target in transfers:
            for seed in matrix["seeds"]:
                for variant, spec in matrix["variants"].items():
                    for (
                        requested_budget,
                        selection_seed,
                        selection_seed_mode,
                        lbi,
                        lbi_trial_index,
                    ) in _variant_points(variant, spec, seed):
                        num_random_masks = (
                            spec["num_random_masks"]
                            if variant == "module_random"
                            else None
                        )
                        effective = _apply_plan_config(
                            base_config,
                            matrix,
                            dataset,
                            source,
                            target,
                            seed,
                            variant,
                            requested_budget,
                            selection_seed,
                            num_random_masks,
                            lbi,
                        )
                        identity = build_experiment_identity(effective)
                        overrides = _effective_overrides(
                            matrix,
                            dataset,
                            source,
                            target,
                            seed,
                            variant,
                            requested_budget,
                            selection_seed,
                            num_random_masks,
                            lbi,
                        )
                        command = _command_arguments(
                            base_config_path,
                            matrix,
                            overrides,
                            identity,
                        )
                        lbi_resolved = (
                            variant != "module_lbi" or effective.get("lbi") is not None
                        )
                        lbi_config = effective.get("lbi") or {}
                        entries.append(
                            {
                                "implementation_revision": (
                                    IMPLEMENTATION_REVISION
                                ),
                                "protocol_revision": matrix.get(
                                    "protocol_revision", PROTOCOL_REVISION
                                ),
                                "efficiency_protocol_revision": (
                                    EFFICIENCY_PROTOCOL_REVISION
                                ),
                                "source_checkpoint_revision": (
                                    SOURCE_CHECKPOINT_REVISION
                                ),
                                "experiment_key": identity[
                                    "experiment_key"
                                ],
                                "experiment_config_sha256": identity[
                                    "experiment_config_sha256"
                                ],
                                "scientific_config": identity[
                                    "scientific_config"
                                ],
                                "method": matrix["method"],
                                "task": matrix["task"],
                                "dataset": dataset,
                                "source": source,
                                "target": target,
                                "seed": seed,
                                "variant": variant,
                                "requested_budget": requested_budget,
                                "selection_seed": selection_seed,
                                "selection_seed_mode": (
                                    selection_seed_mode
                                ),
                                "num_random_masks": num_random_masks,
                                "lbi_trial_index": lbi_trial_index,
                                "lbi_resolved": lbi_resolved,
                                "unresolved_lbi_reason": (
                                    "six dataset×budget SHOT tuned tuples are TBD"
                                    if not lbi_resolved
                                    else None
                                ),
                                "alpha": (
                                    lbi_config.get("alpha")
                                    if lbi_resolved and variant == "module_lbi"
                                    else None
                                ),
                                "kappa": (
                                    lbi_config.get("kappa")
                                    if lbi_resolved and variant == "module_lbi"
                                    else None
                                ),
                                "nu": (
                                    lbi_config.get("nu")
                                    if lbi_resolved and variant == "module_lbi"
                                    else None
                                ),
                                "omega": (
                                    lbi_config.get("omega")
                                    if lbi_resolved and variant == "module_lbi"
                                    else None
                                ),
                                "stage1_max_steps": (
                                    lbi_config.get("stage1_max_steps")
                                    if lbi_resolved and variant == "module_lbi"
                                    else None
                                ),
                                "budget_tolerance": (
                                    lbi_config.get("budget_tolerance")
                                    if lbi_resolved and variant == "module_lbi"
                                    else None
                                ),
                                "stage2_lr": (
                                    lbi_config.get("stage2_lr")
                                    if lbi_resolved and variant == "module_lbi"
                                    else None
                                ),
                                "stage2_steps_requested": (
                                    lbi_config.get("stage2_steps")
                                    if lbi_resolved and variant == "module_lbi"
                                    else None
                                ),
                                "delta_nonzero_tolerance": (
                                    lbi_config.get("delta_nonzero_tolerance")
                                    if lbi_resolved and variant == "module_lbi"
                                    else None
                                ),
                                "effective_overrides": overrides,
                                "command_args": command,
                                "expected_output_root": experiment_output_root(
                                    effective
                                ),
                            }
                        )
    _validate_unique_identities(entries)
    plan = {
        "plan_schema_version": 1,
        "protocol_revision": matrix.get("protocol_revision", PROTOCOL_REVISION),
        "implementation_revision": IMPLEMENTATION_REVISION,
        "efficiency_protocol_revision": EFFICIENCY_PROTOCOL_REVISION,
        "source_checkpoint_revision": SOURCE_CHECKPOINT_REVISION,
        "method": matrix["method"],
        "task": matrix["task"],
        "matrix_path": None,
        "base_config_path": osp.abspath(base_config_path),
        "experiment_count": len(entries),
        "experiments": entries,
    }
    if "VISDA-C" in matrix["datasets"]:
        plan["batch_size"] = 256
    plan["formal_matrix_summary"] = {
        "dataset_counts": {
            dataset: sum(1 for entry in entries if entry["dataset"] == dataset)
            for dataset in transfers_by_dataset
        },
        "variant_counts": {
            variant: sum(1 for entry in entries if entry["variant"] == variant)
            for variant in matrix["variants"]
        },
        "unresolved_lbi_count": sum(
            1 for entry in entries
            if entry["variant"] == "module_lbi" and not entry["lbi_resolved"]
        ),
    }
    return plan


def _dump_json(path, payload):
    os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)


def write_plan_outputs(plan, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    _dump_json(osp.join(output_dir, "plan.json"), plan)
    with open(
        osp.join(output_dir, "plan.jsonl"),
        "w",
        encoding="utf-8",
    ) as file_obj:
        for experiment in plan["experiments"]:
            file_obj.write(
                json.dumps(experiment, ensure_ascii=False) + "\n"
            )
    with open(
        osp.join(output_dir, "commands.sh"),
        "w",
        encoding="utf-8",
    ) as file_obj:
        file_obj.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        for experiment in plan["experiments"]:
            file_obj.write(
                shlex.join(experiment["command_args"]) + "\n"
            )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix")
    parser.add_argument("base_config")
    parser.add_argument("--output-dir", default="iclr2027/plans/pilot")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print plan count without writing files",
    )
    return parser


def main():
    args = build_parser().parse_args()
    matrix = load_yaml(args.matrix)
    base_config = load_yaml(args.base_config)
    plan = build_plan(matrix, base_config, args.base_config)
    plan["matrix_path"] = osp.abspath(args.matrix)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "valid": True,
                    "experiment_count": plan["experiment_count"],
                    **plan.get("formal_matrix_summary", {}),
                },
                ensure_ascii=False,
            )
        )
        return 0
    write_plan_outputs(plan, args.output_dir)
    print(
        f"Generated {plan['experiment_count']} experiments in "
        f"{osp.abspath(args.output_dir)}; commands were not executed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
