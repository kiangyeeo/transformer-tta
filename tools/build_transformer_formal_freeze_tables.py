#!/usr/bin/env python3
"""Build raw-float SHOT-OTTA Transformer freeze artifacts from frozen runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean


RESULTS = Path("/home/nas3/biod/wangkangyi/results")
FINAL_RUN = RESULTS / "transformer_otta_final_result/final_seed2026_20260831T045930.789463160Z"
OUTPUT = FINAL_RUN / "formal_freeze"

SOURCES = {
    "source_model": {
        "office31": RESULTS / "transformer_otta_source_only/source_only_seed2026_20260827T121449.321668Z",
        "visda-c": RESULTS / "transformer_otta_source_only/source_only_seed2026_20260822T032636.681837Z",
    },
    "full_dense": {
        "office31": RESULTS / "transformer_otta_full_dense/full_dense_seed2026_20260827T122349.461814Z",
        "visda-c": RESULTS / "transformer_otta_full_dense/full_dense_seed2026_20260822T032304.521700Z",
    },
    "candidate_dense": {
        "office31": RESULTS / "transformer_otta_candidate_dense/candidate_dense_seed2026_20260827T122647.475235Z",
        "visda-c": RESULTS / "transformer_otta_candidate_dense/candidate_dense_seed2026_20260822T040314.256355Z",
    },
    "random": {
        "office31": RESULTS / "transformer_otta_group_random/group_random_seed2026_20260827T122832.263075Z",
        "visda-c": RESULTS / "transformer_otta_group_random/group_random_seed2026_20260824T104618.470355Z",
    },
    "magnitude": {
        "office31": RESULTS / "transformer_otta_group_magnitude/group_magnitude_seed2026_20260827T123530.687445Z",
        "visda-c": RESULTS / "transformer_otta_group_magnitude/group_magnitude_seed2026_20260824T110106.190390Z",
    },
    "saliency": {
        "office31": RESULTS / "transformer_otta_group_saliency/group_saliency_seed2026_20260827T124035.289092Z",
        "visda-c": RESULTS / "transformer_otta_group_saliency/group_saliency_seed2026_20260824T110829.006812Z",
    },
    "lbi": {"office31": FINAL_RUN, "visda-c": FINAL_RUN},
}

METHOD_ORDER = [
    "source_model", "full_dense", "candidate_dense", "random",
    "magnitude", "saliency", "lbi",
]
BUDGETS = [0.0005, 0.001, 0.002]
OFFICE_TRANSFERS = [
    "amazon->dslr", "amazon->webcam", "dslr->amazon",
    "dslr->webcam", "webcam->amazon", "webcam->dslr",
]
OFFICE_SHORT = {
    "amazon->dslr": "A_to_D", "amazon->webcam": "A_to_W",
    "dslr->amazon": "D_to_A", "dslr->webcam": "D_to_W",
    "webcam->amazon": "W_to_A", "webcam->dslr": "W_to_D",
}
VISDA_CLASSES = [
    "aeroplane", "bicycle", "bus", "car", "horse", "knife",
    "motorcycle", "person", "plant", "skateboard", "train", "truck",
]


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def find_log(root: Path, dataset: str, transfer: str, budget: float | None) -> Path:
    source, target = transfer.split("->")
    if root == FINAL_RUN:
        budget_text = str(budget)
        matches = list((root / "logs").glob(
            f"{dataset}_b{budget_text}_*_o*_lr*_{source}-{target}.log"
        ))
    else:
        prefix = f"rho-{budget}_" if budget is not None else ""
        matches = list((root / "logs").glob(f"{prefix}{dataset}_{source}-{target}.log"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one log for {dataset} {transfer} {budget} in {root}, got {matches}")
    return matches[0]


def artifact_paths(summary_path: Path, log_path: Path, source_root: Path) -> dict:
    run_dir = summary_path.parent
    paths = {
        "summary": summary_path,
        "metrics": run_dir / "metrics.jsonl",
        "manifest": run_dir / "manifest.json",
        "effective_config": run_dir / "effective_config.yaml",
        "log": log_path,
    }
    result = {}
    for name, path in paths.items():
        result[f"{name}_path"] = str(path) if path.exists() else None
        result[f"{name}_relative_to_source_root"] = (
            relative_or_absolute(path, source_root) if path.exists() else None
        )
    return result


def selected_parameters(summary: dict) -> dict:
    selection = summary.get("selection", {})
    lbi = summary.get("lbi", {})
    optimization = summary.get("optimization", summary.get("stage2_optimization", {}))
    return {
        "formal_seed": summary.get("formal_seed"),
        "requested_budget": summary.get("requested_budget", selection.get("requested_budget")),
        "requested_group_count": summary.get("requested_group_count", selection.get("requested_group_count")),
        "realized_group_count": summary.get("realized_group_count", selection.get("realized_group_count_mean")),
        "alpha": lbi.get("alpha"),
        "kappa": lbi.get("kappa"),
        "nu": lbi.get("nu"),
        "omega": lbi.get("omega"),
        "prox_lambda": lbi.get("prox_lambda"),
        "tau_g": lbi.get("tau_g"),
        "stage1_max_steps": lbi.get("stage1_max_steps"),
        "stage2_lr": lbi.get("stage2_lr"),
        "optimizer": optimization.get("optimizer"),
        "lr": optimization.get("lr"),
        "weight_decay": optimization.get("weight_decay"),
        "selection_method": summary.get("selection_method", selection.get("type")),
    }


def validity(summary: dict) -> dict:
    selection = summary.get("selection", {})
    failure_rate = selection.get("failure_rate", summary.get("failure_rate"))
    violation_rate = selection.get("budget_violation_rate", summary.get("budget_violation_rate"))
    max_steps = selection.get("max_steps_reached_count", summary.get("max_steps_reached_count"))
    hit_3000 = selection.get(
        "stage1_3000_step_hit_count", summary.get("stage1_3000_step_hit_count")
    )
    checks = [summary.get("status") == "completed"]
    for value in (failure_rate, violation_rate, max_steps, hit_3000):
        if value is not None:
            checks.append(value == 0)
    if summary.get("valid_lbi_run") is not None:
        checks.append(summary["valid_lbi_run"] is True)
    if summary.get("result_validity") is not None:
        checks.append(summary["result_validity"] != "invalid")
    requested = summary.get("requested_group_count", selection.get("requested_group_count"))
    realized_max = selection.get("realized_group_count_max", summary.get("realized_group_count"))
    strict_budget_ok = None if requested is None or realized_max is None else realized_max <= requested
    if strict_budget_ok is not None:
        checks.append(strict_budget_ok)
    return {
        "scientific_valid": all(checks),
        "status": summary.get("status"),
        "valid_lbi_run": summary.get("valid_lbi_run"),
        "result_validity": summary.get("result_validity"),
        "invalid_reason": summary.get("invalid_reason"),
        "failure_rate": failure_rate,
        "budget_violation_rate": violation_rate,
        "max_steps_reached_count": max_steps,
        "stage1_3000_step_hit_count": hit_3000,
        "strict_budget_ok": strict_budget_ok,
        "requested_group_count": requested,
        "realized_group_count_max": realized_max,
    }


def metrics(summary: dict) -> dict:
    names = summary.get("class-names", [])
    pu_by_name = summary.get("PU-Acc-per-class-by-name")
    fo_by_name = summary.get("FO-Acc-per-class-by-name")
    if pu_by_name is None and summary.get("PU-Acc-per-class") is not None:
        pu_by_name = dict(zip(names, summary["PU-Acc-per-class"]))
    if fo_by_name is None and summary.get("FO-Acc-per-class") is not None:
        fo_by_name = dict(zip(names, summary["FO-Acc-per-class"]))
    return {
        "primary_metric": summary.get("primary_metric"),
        "PU_Acc": summary.get("PU-Acc"),
        "FO_Acc": summary.get("FO-Acc"),
        "PU_macro_mean": summary.get("PU-mean-class-Acc", summary.get("PU-Acc")),
        "FO_macro_mean": summary.get("FO-mean-class-Acc", summary.get("FO-Acc")),
        "PU_overall_accuracy": summary.get("PU-overall-Acc", summary.get("PU-Acc")),
        "FO_overall_accuracy": summary.get("FO-overall-Acc", summary.get("FO-Acc")),
        "PU_per_class_accuracy": pu_by_name,
        "FO_per_class_accuracy": fo_by_name,
    }


def steps_runtime_memory(summary: dict) -> tuple[dict, dict, dict]:
    selection = summary.get("selection", {})
    steps = {key: value for key, value in selection.items() if key.startswith("stage1_")}
    runtime = {
        key: value for key, value in summary.items()
        if "runtime" in key and isinstance(value, (int, float))
    }
    memory = {
        key: value for key, value in summary.items()
        if ("memory" in key or key.startswith("gpu_peak_")) and isinstance(value, (int, float))
    }
    return steps, runtime, memory


def summary_paths(method: str, dataset: str, root: Path) -> list[Path]:
    if method == "lbi":
        return sorted((root / "runs" / dataset).glob("**/summary.json"))
    aggregate = load(root / "aggregate.json")
    wanted = [x for x in aggregate["transfers"] if x.get("dataset") == dataset]
    paths = []
    for item in wanted:
        path = Path(item["output_dir"]) / "summary.json"
        if not path.exists():
            raise FileNotFoundError(path)
        paths.append(path)
    return paths


def build_records() -> list[dict]:
    records = []
    for method in METHOD_ORDER:
        for dataset in ("office31", "visda-c"):
            root = SOURCES[method][dataset]
            for summary_path in summary_paths(method, dataset, root):
                summary = load(summary_path)
                budget = summary.get("requested_budget", summary.get("selection", {}).get("requested_budget"))
                log_path = find_log(root, dataset, summary["transfer"], budget)
                steps, runtime, memory = steps_runtime_memory(summary)
                record = {
                    "method": method,
                    "variant": summary.get("variant"),
                    "dataset": dataset,
                    "source": summary.get("source"),
                    "target": summary.get("target"),
                    "transfer": summary.get("transfer"),
                    "protocol_revision": summary.get("protocol_revision"),
                    "experiment_key": summary.get("experiment_key"),
                    "scientific_config_sha256": summary.get("scientific_config_sha256"),
                    "source_root": str(root),
                    "parameters": selected_parameters(summary),
                    "validity": validity(summary),
                    "accuracy": metrics(summary),
                    "stage1_steps": steps,
                    "runtime": runtime,
                    "memory": memory,
                    "artifacts": artifact_paths(summary_path, log_path, root),
                }
                if method == "random":
                    record["random_child_runs"] = []
                    for mask in summary.get("masks", []):
                        child_path = Path(mask["summary_path"])
                        child = load(child_path)
                        child_steps, child_runtime, child_memory = steps_runtime_memory(child)
                        record["random_child_runs"].append({
                            "random_mask_index": child.get("random_mask_index"),
                            "mask_seed": child.get("mask_seed"),
                            "validity": validity(child),
                            "accuracy": metrics(child),
                            "stage1_steps": child_steps,
                            "runtime": child_runtime,
                            "memory": child_memory,
                            "artifacts": artifact_paths(child_path, log_path, root),
                        })
                records.append(record)
    return records


def key_budget(record: dict) -> float | None:
    return record["parameters"]["requested_budget"]


def expected_rows(method: str) -> list[float | None]:
    return [None] if method in {"source_model", "full_dense", "candidate_dense"} else BUDGETS


def write_office(records: list[dict], path: Path) -> None:
    fields = ["method", "budget"]
    for phase in ("PU", "FO"):
        fields.extend(f"{phase}_{OFFICE_SHORT[t]}" for t in OFFICE_TRANSFERS)
        fields.append(f"mean_{phase}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in METHOD_ORDER:
            for budget in expected_rows(method):
                selected = [
                    r for r in records if r["method"] == method and r["dataset"] == "office31"
                    and key_budget(r) == budget
                ]
                if len(selected) != 6:
                    raise RuntimeError(f"Office expected 6 records for {method} {budget}, got {len(selected)}")
                by_transfer = {r["transfer"]: r for r in selected}
                row = {
                    "method": method,
                    "budget": 1.0 if method == "candidate_dense" else budget,
                }
                for phase in ("PU", "FO"):
                    values = []
                    for transfer in OFFICE_TRANSFERS:
                        value = by_transfer[transfer]["accuracy"][f"{phase}_Acc"]
                        row[f"{phase}_{OFFICE_SHORT[transfer]}"] = value
                        values.append(value)
                    row[f"mean_{phase}"] = fmean(values)
                writer.writerow(row)


def write_visda(records: list[dict], path: Path) -> None:
    fields = ["method", "budget"]
    for phase in ("PU", "FO"):
        fields.extend([
            f"{phase}_Acc_primary", f"{phase}_macro_mean", f"{phase}_overall_accuracy",
        ])
        fields.extend(f"{phase}_{name}" for name in VISDA_CLASSES)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in METHOD_ORDER:
            for budget in expected_rows(method):
                selected = [
                    r for r in records if r["method"] == method and r["dataset"] == "visda-c"
                    and key_budget(r) == budget
                ]
                if len(selected) != 1:
                    raise RuntimeError(f"VisDA expected 1 record for {method} {budget}, got {len(selected)}")
                accuracy = selected[0]["accuracy"]
                row = {
                    "method": method,
                    "budget": 1.0 if method == "candidate_dense" else budget,
                }
                for phase in ("PU", "FO"):
                    row[f"{phase}_Acc_primary"] = accuracy[f"{phase}_Acc"]
                    row[f"{phase}_macro_mean"] = accuracy[f"{phase}_macro_mean"]
                    row[f"{phase}_overall_accuracy"] = accuracy[f"{phase}_overall_accuracy"]
                    per_class = accuracy[f"{phase}_per_class_accuracy"]
                    if list(per_class) != VISDA_CLASSES:
                        raise RuntimeError(f"Unexpected VisDA class order for {method} {budget}: {list(per_class)}")
                    for name in VISDA_CLASSES:
                        row[f"{phase}_{name}"] = per_class[name]
                writer.writerow(row)


def main() -> None:
    records = build_records()
    if len(records) != 105:
        raise RuntimeError(f"Expected 105 reported condition records, got {len(records)}")
    if not all(record["validity"]["scientific_valid"] for record in records):
        bad = [r["experiment_key"] for r in records if not r["validity"]["scientific_valid"]]
        raise RuntimeError(f"Invalid reported records: {bad}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "description": "SHOT-OTTA Transformer frozen result records; all floats are stored without display rounding or truncation.",
        "float_policy": "raw JSON numeric floats copied from summaries or computed with unrounded Python float arithmetic",
        "method_order": METHOD_ORDER,
        "source_roots_by_method_and_dataset": {
            method: {dataset: str(path) for dataset, path in roots.items()}
            for method, roots in SOURCES.items()
        },
        "reported_condition_count": len(records),
        "random_physical_child_run_count": sum(len(r.get("random_child_runs", [])) for r in records),
        "records": records,
    }
    with (OUTPUT / "all_runs.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    write_office(records, OUTPUT / "office31_results.csv")
    write_visda(records, OUTPUT / "visda_c_results.csv")
    print(f"Wrote 3 files to {OUTPUT}")
    print(f"Reported conditions: {len(records)}")
    print(f"Random physical child runs: {payload['random_physical_child_run_count']}")


if __name__ == "__main__":
    main()
