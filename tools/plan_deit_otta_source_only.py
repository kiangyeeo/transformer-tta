#!/usr/bin/env python3
"""Build the fixed seven-task DeiT OTTA source-only execution plan."""

import argparse
import copy
import json
import os
import os.path as osp
import sys
from itertools import permutations


PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shot_otta.ttda.config import (  # noqa: E402
    DATASET_SPECS,
    experiment_output_root,
    load_yaml,
    resolve_config,
)


def _expand_transfers(dataset, transfers):
    domain_count = len(DATASET_SPECS[dataset]["domains"])
    if transfers == "all_ordered_pairs":
        return [list(pair) for pair in permutations(range(domain_count), 2)]
    if not isinstance(transfers, list) or not transfers:
        raise ValueError(f"{dataset}.transfers must be a non-empty list")
    normalized = []
    for transfer in transfers:
        if (
            not isinstance(transfer, list)
            or len(transfer) != 2
            or not all(isinstance(value, int) for value in transfer)
        ):
            raise ValueError(f"Invalid transfer for {dataset}: {transfer}")
        source, target = transfer
        if source == target:
            raise ValueError(f"Self-transfer is forbidden: {dataset} {transfer}")
        if not 0 <= source < domain_count or not 0 <= target < domain_count:
            raise ValueError(f"Transfer index is out of range: {dataset} {transfer}")
        normalized.append([source, target])
    if len({tuple(value) for value in normalized}) != len(normalized):
        raise ValueError(f"Duplicate transfer in {dataset}")
    return normalized


def validate_matrix(matrix):
    fixed = {
        "schema_version": 1,
        "method": "no_tta",
        "task": "otta",
        "variant": "source_only",
    }
    for field, expected in fixed.items():
        if matrix.get(field) != expected:
            raise ValueError(f"matrix {field} must be {expected}")
    seeds = matrix.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("matrix seeds must be a non-empty list")
    if not all(isinstance(seed, int) for seed in seeds):
        raise ValueError("Every matrix seed must be an integer")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate matrix seed")
    datasets = matrix.get("datasets")
    if set(datasets or {}) != {"office31", "visda-c"}:
        raise ValueError("Matrix must contain exactly office31 and visda-c")
    transfers = {
        dataset: _expand_transfers(dataset, spec.get("transfers"))
        for dataset, spec in datasets.items()
    }
    expected_office = [
        [0, 1],
        [0, 2],
        [1, 0],
        [1, 2],
        [2, 0],
        [2, 1],
    ]
    if transfers["office31"] != expected_office:
        raise ValueError("Office-31 matrix must contain all six directions")
    if transfers["visda-c"] != [[0, 1]]:
        raise ValueError("VisDA-C matrix must be train -> validation")
    return transfers


def _command_args(config_path, effective):
    data = effective["data"]
    return [
        sys.executable,
        osp.join(PROJECT_ROOT, "evaluate_deit_otta.py"),
        "--config",
        osp.abspath(config_path),
        "--dataset",
        data["dataset"],
        "--source",
        str(data["source"]),
        "--target",
        str(data["target"]),
        "--seed",
        str(effective["seed"]),
        "--batch-size",
        str(effective["evaluation"]["batch_size"]),
        "--workers",
        str(effective["evaluation"]["workers"]),
        "--gpu-id",
        effective["device"]["gpu_id"],
        "--output-root",
        effective["output"]["root"],
        "--experiment-key",
        effective["experiment_key"],
        "--experiment-config-sha256",
        effective["experiment_config_sha256"],
    ]


def build_plan(matrix, base_config, config_path, matrix_path):
    transfers = validate_matrix(matrix)
    for field in ("method", "task", "variant"):
        if base_config.get(field) != matrix[field]:
            raise ValueError(f"Base config {field} does not match matrix")
    entries = []
    for dataset in ("office31", "visda-c"):
        for source, target in transfers[dataset]:
            for seed in matrix["seeds"]:
                raw = copy.deepcopy(base_config)
                raw["data"]["dataset"] = dataset
                raw["data"]["source"] = source
                raw["data"]["target"] = target
                raw["seed"] = seed
                effective = resolve_config(
                    raw, PROJECT_ROOT, expected_task="otta"
                )
                entries.append(
                    {
                        "experiment_key": effective["experiment_key"],
                        "experiment_config_sha256": effective[
                            "experiment_config_sha256"
                        ],
                        "scientific_config": effective["scientific_config"],
                        "method": "no_tta",
                        "task": "otta",
                        "dataset": dataset,
                        "source": source,
                        "target": target,
                        "source_name": effective["data"]["source_name"],
                        "target_name": effective["data"]["target_name"],
                        "seed": seed,
                        "variant": "source_only",
                        "requested_budget": None,
                        "selection_seed": None,
                        "source_checkpoint_path": effective[
                            "source_checkpoint"
                        ]["path"],
                        "source_checkpoint_sha256": effective[
                            "source_checkpoint"
                        ]["sha256"],
                        "command_args": _command_args(config_path, effective),
                        "expected_output_root": experiment_output_root(effective),
                    }
                )
    keys = [entry["experiment_key"] for entry in entries]
    hashes = [entry["experiment_config_sha256"] for entry in entries]
    if len(set(keys)) != len(keys) or len(set(hashes)) != len(hashes):
        raise RuntimeError("OTTA source-only plan identities are not unique")
    if len(entries) != 7:
        raise RuntimeError(f"Expected exactly 7 experiments, got {len(entries)}")
    return {
        "plan_schema_version": 1,
        "method": "no_tta",
        "task": "otta",
        "variant": "source_only",
        "base_config_path": osp.abspath(config_path),
        "matrix_path": osp.abspath(matrix_path),
        "experiment_count": len(entries),
        "experiments": entries,
    }


def write_plan(path, plan):
    path = osp.abspath(path)
    os.makedirs(osp.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as file_obj:
        json.dump(plan, file_obj, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan = build_plan(
        load_yaml(args.matrix),
        load_yaml(args.config),
        args.config,
        args.matrix,
    )
    write_plan(args.output, plan)
    print(
        json.dumps(
            {
                "plan": osp.abspath(args.output),
                "experiment_count": plan["experiment_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
