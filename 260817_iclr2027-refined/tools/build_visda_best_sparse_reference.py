#!/usr/bin/env python3
"""Build the authoritative VisDA best-sparse reference from parent summaries."""

import argparse
import csv
import json
from pathlib import Path


VARIANTS = {"module_random", "module_magnitude", "module_saliency"}


def _number(value):
    return float(value) if value is not None else None


def _sort_key(row):
    # PU deliberately does not appear in this selection key.
    return (
        -_number(row["FO-Acc"]),
        -_number(row["FO-worst-class-Acc"]),
        _number(row["FO-class-std"]),
        -_number(row["FO-overall-Acc"]),
        row["variant"],
    )


def _write_csv(path, rows):
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build(runs_root, output_dir):
    rows = []
    for path in sorted(Path(runs_root).rglob("summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if (
            summary.get("status") == "completed"
            and summary.get("dataset") == "VISDA-C"
            and summary.get("variant") in VARIANTS
        ):
            rows.append({**summary, "source_summary_path": str(path.resolve())})
    budgets = sorted({float(row["requested_budget"]) for row in rows})
    selected = []
    for budget in budgets:
        candidates = [row for row in rows if float(row["requested_budget"]) == budget]
        if len(candidates) != 3 or {row["variant"] for row in candidates} != VARIANTS:
            raise ValueError(f"budget {budget} lacks exactly three sparse parents")
        winner = sorted(candidates, key=_sort_key)[0]
        selected.append(
            {
                **winner,
                "best_sparse_method": winner["variant"],
                "selection_reason": "FO-Acc desc; FO-worst-class-Acc desc; FO-class-std asc; FO-overall-Acc desc",
            }
        )
    classwise = []
    for row in selected:
        for metric in ("PU", "FO"):
            for class_id, (name, value) in enumerate(zip(row["class-names"], row[f"{metric}-Acc-per-class"])):
                classwise.append({"requested_budget": row["requested_budget"], "best_sparse_method": row["best_sparse_method"], "source_summary_path": row["source_summary_path"], "metric": metric, "class_id": class_id, "class_name": name, "accuracy": value})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "all_sparse_methods.csv", rows)
    _write_csv(output_dir / "best_sparse_per_budget.csv", selected)
    _write_csv(output_dir / "best_sparse_classwise_long.csv", classwise)
    lines = ["# Best sparse reference", "", "Selection uses FO only: FO-Acc descending, FO worst-class accuracy descending, FO class std ascending, then FO overall accuracy descending. PU is recorded as evidence and is not used for selection.", "", "| budget | method | FO-Acc | FO worst | FO std | FO overall | source summary |", "|---:|---|---:|---:|---:|---:|---|"]
    for row in selected:
        lines.append(f"| {row['requested_budget']} | {row['best_sparse_method']} | {row['FO-Acc']:.4f} | {row['FO-worst-class-Acc']:.4f} | {row['FO-class-std']:.4f} | {row['FO-overall-Acc']:.4f} | `{row['source_summary_path']}` |")
    (output_dir / "REFERENCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows, selected, classwise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows, selected, classwise = build(args.runs_root, args.output_dir)
    print(json.dumps({"all_sparse_methods": len(rows), "best_per_budget": len(selected), "classwise_rows": len(classwise)}))


if __name__ == "__main__":
    main()
