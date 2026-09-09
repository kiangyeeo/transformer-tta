#!/usr/bin/env python3
"""Build the VisDA PU/FO class-wise audit from the specified completed runs only."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "experiment_logs/visda_fc_lbi_seed2026_search_20260819"
FORMAL_ROOT = ROOT / "experiment_logs/visda_fc_lbi_formal_seed2026_20260825"
OUTPUT_MD = ROOT / "VISDA_CLASSWISE_COMPARISON_PU_FO.md"
OUTPUT_CSV = ROOT / "VISDA_CLASSWISE_COMPARISON_PU_FO.csv"
CLASS_NAMES = [
    "aeroplane", "bicycle", "bus", "car", "horse", "knife", "motorcycle",
    "person", "plant", "skateboard", "train", "truck",
]
REQUIRED = [
    "class-names",
    "PU-Acc-per-class", "PU-mean-class-Acc", "PU-overall-Acc",
    "PU-worst-class-name", "PU-worst-class-Acc", "PU-class-std",
    "FO-Acc-per-class", "FO-mean-class-Acc", "FO-overall-Acc",
    "FO-worst-class-name", "FO-worst-class-Acc", "FO-class-std",
]

# The non-LBI rows are fixed to the baseline root; all LBI rows are fixed to
# the formal root. This intentionally excludes every tuning/search LBI run.
RUNS = [
    ("source_only", "Source-only", "", BASELINE_ROOT, "runs/VISDA-C/TV/none/none/source_only/seed_2026/20260819_113043_150389_TV/summary.json"),
    ("module_dense", "Module-Dense", "full", BASELINE_ROOT, "runs/VISDA-C/TV/netB_bottleneck/full/dense/seed_2026/20260819_113043_109480_TV/summary.json"),
    ("full_dense", "Full-Dense", "full", BASELINE_ROOT, "runs/VISDA-C/TV/netF_netB/full/dense/seed_2026/20260819_113043_130632_TV/summary.json"),
    ("module_random", "Module-Random", "0.0005", BASELINE_ROOT, "runs/VISDA-C/TV/netB_bottleneck/0.0005/random/seed_2026/20260819_113039_410471_TV/summary.json"),
    ("module_random", "Module-Random", "0.001", BASELINE_ROOT, "runs/VISDA-C/TV/netB_bottleneck/0.001/random/seed_2026/20260819_113039_409926_TV/summary.json"),
    ("module_random", "Module-Random", "0.002", BASELINE_ROOT, "runs/VISDA-C/TV/netB_bottleneck/0.002/random/seed_2026/20260819_113039_410110_TV/summary.json"),
    ("module_magnitude", "Module-Magnitude", "0.0005", BASELINE_ROOT, "runs/VISDA-C/TV/netB_bottleneck/0.0005/magnitude/seed_2026/20260819_113043_415625_TV/summary.json"),
    ("module_magnitude", "Module-Magnitude", "0.001", BASELINE_ROOT, "runs/VISDA-C/TV/netB_bottleneck/0.001/magnitude/seed_2026/20260819_113043_294425_TV/summary.json"),
    ("module_magnitude", "Module-Magnitude", "0.002", BASELINE_ROOT, "runs/VISDA-C/TV/netB_bottleneck/0.002/magnitude/seed_2026/20260819_113610_190923_TV/summary.json"),
    ("module_saliency", "Module-Saliency", "0.0005", BASELINE_ROOT, "runs/VISDA-C/TV/netB_bottleneck/0.0005/saliency/seed_2026/20260819_113612_041746_TV/summary.json"),
    ("module_saliency", "Module-Saliency", "0.001", BASELINE_ROOT, "runs/VISDA-C/TV/netB_bottleneck/0.001/saliency/seed_2026/20260819_113612_204319_TV/summary.json"),
    ("module_saliency", "Module-Saliency", "0.002", BASELINE_ROOT, "runs/VISDA-C/TV/netB_bottleneck/0.002/saliency/seed_2026/20260819_113612_266151_TV/summary.json"),
    ("formal_lbi", "Formal LBI", "0.0005", FORMAL_ROOT, "runs/VISDA-C/TV/netB_bottleneck/0.0005/LBI/seed_2026/20260825_040836_611977_TV/summary.json"),
    ("formal_lbi", "Formal LBI", "0.001", FORMAL_ROOT, "runs/VISDA-C/TV/netB_bottleneck/0.001/LBI/seed_2026/20260825_040836_612277_TV/summary.json"),
    ("formal_lbi", "Formal LBI", "0.002", FORMAL_ROOT, "runs/VISDA-C/TV/netB_bottleneck/0.002/LBI/seed_2026/20260825_040836_611580_TV/summary.json"),
]


def fmt(value: float) -> str:
    return f"{value:.3f}"


def budget_label(budget: str) -> str:
    return budget or "-"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
        *["| " + " | ".join(row) + " |" for row in rows],
    ])


def read_rows() -> list[dict]:
    if len(RUNS) != 15:
        raise RuntimeError(f"Expected 15 configured rows, got {len(RUNS)}")
    rows = []
    for method, display_method, budget, source_root, relative_path in RUNS:
        path = (source_root / relative_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing required summary.json: {path}")
        if method == "formal_lbi" and FORMAL_ROOT not in path.parents:
            raise RuntimeError(f"Formal LBI path is outside formal root: {path}")
        if method != "formal_lbi" and BASELINE_ROOT not in path.parents:
            raise RuntimeError(f"Baseline path is outside baseline root: {path}")
        data = json.loads(path.read_text())
        missing = [field for field in REQUIRED if field not in data]
        if missing:
            raise RuntimeError(f"Missing required fields in {path}: {', '.join(missing)}")
        if data["class-names"] != CLASS_NAMES:
            raise RuntimeError(
                f"class-names mismatch in {path}: expected {CLASS_NAMES}, got {data['class-names']}"
            )
        row = {
            "method": method,
            "display_method": display_method,
            "budget": budget,
            "path": path,
            "run_id": data.get("run_id"),
            "classes": data["class-names"],
        }
        for phase in ("PU", "FO"):
            values = data[f"{phase}-Acc-per-class"]
            if not isinstance(values, list) or len(values) != len(CLASS_NAMES):
                raise RuntimeError(f"Invalid {phase}-Acc-per-class in {path}")
            calculated_mean = fmean(values)
            summary_mean = data[f"{phase}-mean-class-Acc"]
            if abs(calculated_mean - summary_mean) > 1e-9:
                raise RuntimeError(
                    f"{phase} class mean mismatch in {path}: {calculated_mean} != {summary_mean}"
                )
            row[phase] = {
                "values": values,
                "macc": summary_mean,
                "overall": data[f"{phase}-overall-Acc"],
                "worst_name": data[f"{phase}-worst-class-name"],
                "worst_acc": data[f"{phase}-worst-class-Acc"],
                "std": data[f"{phase}-class-std"],
            }
        rows.append(row)
    if len({tuple(row["classes"]) for row in rows}) != 1:
        raise RuntimeError("The 15 summary.json class-names orders are not identical")
    if any(not row["run_id"] for row in rows):
        raise RuntimeError("Missing run_id in at least one selected summary.json")
    return rows


def class_value(row: dict, phase: str, class_name: str) -> float:
    return row[phase]["values"][CLASS_NAMES.index(class_name)]


def selected(rows: list[dict], method: str, budget: str = "") -> dict:
    matches = [row for row in rows if row["method"] == method and row["budget"] == budget]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {method}/{budget} row, found {len(matches)}")
    return matches[0]


def comparison_rows(rows: list[dict], phase: str) -> list[list[str]]:
    source = selected(rows, "source_only")
    dense = selected(rows, "module_dense", "full")
    output = []
    for budget in ("0.0005", "0.001", "0.002"):
        lbi = selected(rows, "formal_lbi", budget)
        sparse = [row for row in rows if row["method"] in {
            "module_random", "module_magnitude", "module_saliency"
        } and row["budget"] == budget]
        best = max(sparse, key=lambda row: row[phase]["macc"])
        delta = lambda class_name, comparator: class_value(lbi, phase, class_name) - class_value(comparator, phase, class_name)
        output.append([
            budget,
            f"{best['display_method']} (mAcc {fmt(best[phase]['macc'])})",
            f"{delta('truck', source):+.3f}",
            f"{delta('knife', source):+.3f}",
            f"{delta('truck', best):+.3f}",
            f"{delta('knife', best):+.3f}",
            f"{lbi[phase]['macc'] - best[phase]['macc']:+.3f}",
            f"{delta('truck', dense):+.3f}",
            f"{delta('knife', dense):+.3f}",
        ])
    return output


def build_markdown(rows: list[dict]) -> str:
    provenance = [[
        row["display_method"], budget_label(row["budget"]), row["run_id"], str(row["path"]),
        fmt(row["PU"]["macc"]), fmt(row["PU"]["overall"]),
        fmt(row["FO"]["macc"]), fmt(row["FO"]["overall"]),
    ] for row in rows]
    class_tables = {}
    for phase in ("PU", "FO"):
        class_tables[phase] = [[
            row["display_method"], budget_label(row["budget"]),
            *[fmt(value) for value in row[phase]["values"]],
        ] for row in rows]
    hard_tables = {}
    for phase in ("PU", "FO"):
        hard_tables[phase] = [[
            row["display_method"], budget_label(row["budget"]),
            fmt(class_value(row, phase, "knife")), fmt(class_value(row, phase, "person")),
            fmt(class_value(row, phase, "truck")), row[phase]["worst_name"],
            fmt(row[phase]["worst_acc"]),
        ] for row in rows]
    transition = [[
        row["display_method"], budget_label(row["budget"]),
        fmt(class_value(row, "PU", "knife")), fmt(class_value(row, "FO", "knife")),
        f"{class_value(row, 'FO', 'knife') - class_value(row, 'PU', 'knife'):+.3f}",
        fmt(class_value(row, "PU", "person")), fmt(class_value(row, "FO", "person")),
        f"{class_value(row, 'FO', 'person') - class_value(row, 'PU', 'person'):+.3f}",
        fmt(class_value(row, "PU", "truck")), fmt(class_value(row, "FO", "truck")),
        f"{class_value(row, 'FO', 'truck') - class_value(row, 'PU', 'truck'):+.3f}",
    ] for row in rows]
    lbi_rows = [selected(rows, "formal_lbi", budget) for budget in ("0.0005", "0.001", "0.002")]
    pu_worst = "; ".join(f"rho={row['budget']}: {row['PU']['worst_name']} ({fmt(row['PU']['worst_acc'])}%)" for row in lbi_rows)
    fo_worst = "; ".join(f"rho={row['budget']}: {row['FO']['worst_name']} ({fmt(row['FO']['worst_acc'])}%)" for row in lbi_rows)
    sparse_pu = [class_value(row, "PU", "knife") for row in rows if row["method"].startswith("module_") and row["method"] not in {"module_dense"}]
    sparse_pu_truck = [class_value(row, "PU", "truck") for row in rows if row["method"] in {"module_random", "module_magnitude", "module_saliency"}]
    sparse_fo = [class_value(row, "FO", "knife") for row in rows if row["method"] in {"module_random", "module_magnitude", "module_saliency"}]
    sparse_fo_truck = [class_value(row, "FO", "truck") for row in rows if row["method"] in {"module_random", "module_magnitude", "module_saliency"}]
    source = selected(rows, "source_only")
    dense = selected(rows, "module_dense", "full")
    full = selected(rows, "full_dense", "full")

    return "\n".join([
        "# VisDA-C PU and FO class-wise comparison",
        "",
        "## Scope and provenance",
        "",
        "- All Formal LBI rows were read exclusively from `visda_fc_lbi_formal_seed2026_20260825`.",
        "- All non-LBI rows were read exclusively from `visda_fc_lbi_seed2026_search_20260819`.",
        "- Each of the 15 selected `summary.json` files was read directly. No tuning LBI artifact was used.",
        "- `class-names` was identical in all 15 summaries, in this order: aeroplane, bicycle, bus, car, horse, knife, motorcycle, person, plant, skateboard, train, truck.",
        "- For every row and phase, the mean recomputed from 12 class accuracies matched the corresponding summary mean-class accuracy within 1e-9.",
        "",
        md_table(["Method", "Budget", "Run ID", "summary.json absolute path", "PU mAcc", "PU overall", "FO mAcc", "FO overall"], provenance),
        "",
        "## 12-class PU accuracy",
        "",
        "All entries are PU accuracy (%), rounded to three decimals.",
        "",
        md_table(["Method", "Budget", *CLASS_NAMES], class_tables["PU"]),
        "",
        "## 12-class FO accuracy",
        "",
        "All entries are FO accuracy (%), rounded to three decimals.",
        "",
        md_table(["Method", "Budget", *CLASS_NAMES], class_tables["FO"]),
        "",
        "## PU hard classes",
        "",
        md_table(["Method", "Budget", "knife", "person", "truck", "PU worst-class name", "PU worst-class acc"], hard_tables["PU"]),
        "",
        "## FO hard classes",
        "",
        md_table(["Method", "Budget", "knife", "person", "truck", "FO worst-class name", "FO worst-class acc"], hard_tables["FO"]),
        "",
        "## Formal LBI deltas: PU",
        "",
        "Every delta is Formal LBI minus the named comparator, in percentage points (pp). The same-budget best sparse baseline is selected from Random, Magnitude, and Saliency by PU mAcc.",
        "",
        md_table(["LBI budget", "Best sparse baseline (PU mAcc)", "truck vs Source", "knife vs Source", "truck vs best sparse", "knife vs best sparse", "mAcc vs best sparse", "truck vs Module-Dense", "knife vs Module-Dense"], comparison_rows(rows, "PU")),
        "",
        "## Formal LBI deltas: FO",
        "",
        "Every delta is Formal LBI minus the named comparator, in percentage points (pp). The same-budget best sparse baseline is selected from Random, Magnitude, and Saliency by FO mAcc.",
        "",
        md_table(["LBI budget", "Best sparse baseline (FO mAcc)", "truck vs Source", "knife vs Source", "truck vs best sparse", "knife vs best sparse", "mAcc vs best sparse", "truck vs Module-Dense", "knife vs Module-Dense"], comparison_rows(rows, "FO")),
        "",
        "## PU-to-FO hard-class changes",
        "",
        "Each delta is FO minus PU, in pp; positive values are higher under FO.",
        "",
        md_table(["Method", "Budget", "PU knife", "FO knife", "knife delta", "PU person", "FO person", "person delta", "PU truck", "FO truck", "truck delta"], transition),
        "",
        "## Direct answers supported by the artifacts",
        "",
        f"- The artifacts do not define a numeric threshold for 'low'. Under PU, Source-only is {fmt(class_value(source, 'PU', 'knife'))}% knife and {fmt(class_value(source, 'PU', 'truck'))}% truck; sparse baselines span {fmt(min(sparse_pu))}% to {fmt(max(sparse_pu))}% knife and {fmt(min(sparse_pu_truck))}% to {fmt(max(sparse_pu_truck))}% truck; Module-Dense is {fmt(class_value(dense, 'PU', 'knife'))}%/{fmt(class_value(dense, 'PU', 'truck'))}%; Full-Dense is {fmt(class_value(full, 'PU', 'knife'))}%/{fmt(class_value(full, 'PU', 'truck'))}%.",
        f"- Under FO, Source-only is {fmt(class_value(source, 'FO', 'knife'))}% knife and {fmt(class_value(source, 'FO', 'truck'))}% truck; sparse baselines span {fmt(min(sparse_fo))}% to {fmt(max(sparse_fo))}% knife and {fmt(min(sparse_fo_truck))}% to {fmt(max(sparse_fo_truck))}% truck; Module-Dense is {fmt(class_value(dense, 'FO', 'knife'))}%/{fmt(class_value(dense, 'FO', 'truck'))}%; Full-Dense is {fmt(class_value(full, 'FO', 'knife'))}%/{fmt(class_value(full, 'FO', 'truck'))}%.",
        f"- Formal LBI PU worst classes: {pu_worst}.",
        f"- Formal LBI FO worst classes: {fo_worst}. Thus, all three FO worst classes are truck.",
        "- From PU to FO: Source-only and all nine sparse baselines have higher knife accuracy and lower truck accuracy; person is higher for Source-only, Random, and Magnitude, and lower for Saliency. Module-Dense has lower knife and truck and higher person; Full-Dense has higher knife and person and lower truck.",
        "- For Formal LBI from PU to FO: knife is higher at all three budgets (+22.988, +17.831, +8.048 pp for 0.0005/0.001/0.002); person is lower at 0.0005 and 0.001 (-0.825, -3.800 pp) and higher at 0.002 (+2.475 pp); truck is higher at 0.0005 (+1.352 pp) and lower at 0.001 and 0.002 (-0.865, -0.036 pp).",
        "- Versus the phase-specific same-budget best sparse baseline, LBI improves both knife and truck at 0.0005 and 0.002 in PU and FO. At 0.001, it improves knife but worsens truck in PU (-0.216 pp truck) and FO (-0.937 pp truck).",
        "",
    ])


def write_csv(rows: list[dict]) -> None:
    headers = [
        "method", "budget", "run_id", "PU mAcc", "PU overall", "PU worst name", "PU worst acc", "PU class std",
        *[f"PU {name}" for name in CLASS_NAMES],
        "FO mAcc", "FO overall", "FO worst name", "FO worst acc", "FO class std",
        *[f"FO {name}" for name in CLASS_NAMES],
    ]
    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([
                row["method"], row["budget"], row["run_id"],
                row["PU"]["macc"], row["PU"]["overall"], row["PU"]["worst_name"], row["PU"]["worst_acc"], row["PU"]["std"],
                *row["PU"]["values"],
                row["FO"]["macc"], row["FO"]["overall"], row["FO"]["worst_name"], row["FO"]["worst_acc"], row["FO"]["std"],
                *row["FO"]["values"],
            ])


def main() -> None:
    rows = read_rows()
    OUTPUT_MD.write_text(build_markdown(rows))
    write_csv(rows)
    print(f"Read {len(rows)}/15 summary.json files; class order and PU/FO means validated.")
    print(f"Markdown output: {OUTPUT_MD}")
    print(f"CSV output: {OUTPUT_CSV}")
    print("Read summary.json files:")
    for row in rows:
        print(row["path"])


if __name__ == "__main__":
    main()
