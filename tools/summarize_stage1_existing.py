#!/usr/bin/env python3
"""Aggregate existing Transformer Group-LBI Stage-1 artifacts without running experiments."""

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path("/home/nas3/biod/wangkangyi")
STAGE1 = ROOT / "results/transformer_otta_group_lbi_tuning/stage1"
OUT = ROOT / "transformer-tta"
BASELINES = {
    "Random": ROOT / "results/transformer_otta_group_random/group_random_seed2026_20260824T104618.470355Z/results.csv",
    "Magnitude": ROOT / "results/transformer_otta_group_magnitude/group_magnitude_seed2026_20260824T110106.190390Z/results.csv",
    "Saliency": ROOT / "results/transformer_otta_group_saliency/group_saliency_seed2026_20260824T110829.006812Z/results.csv",
}
BUDGETS = [0.0005, 0.001, 0.002]
TRANSFERS = ["A->D", "A->W", "D->A", "D->W", "W->A", "W->D"]
TRANSFER_SHORT = {
    "amazon->dslr": "A->D", "amazon->webcam": "A->W", "dslr->amazon": "D->A",
    "dslr->webcam": "D->W", "webcam->amazon": "W->A", "webcam->dslr": "W->D",
    "train->validation": "train->validation",
}


def f(x, n=4):
    if x is None:
        return "NA"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)
    return f"{x:.{n}f}"


def read_json(path):
    with path.open() as fh:
        return json.load(fh)


def read_online(path):
    rows = []
    with path.open() as fh:
        for line in fh:
            event = json.loads(line)
            if event.get("event") == "online_batch":
                rows.append(event)
    return rows


def baseline_table():
    values = defaultdict(dict)
    sources = {}
    for method, path in BASELINES.items():
        sources[method] = str(path)
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                key = (row["dataset"], float(row["requested_budget"]), TRANSFER_SHORT[row["transfer"]])
                values[key][method] = float(row["FO-Acc"])
    out = {}
    for key, methods in values.items():
        if set(methods) != set(BASELINES):
            raise RuntimeError(f"Incomplete sparse baseline set for {key}: {methods}")
        out[key] = {**methods, "best_sparse_FO": max(methods.values())}
    return out, sources


def failure_locations(summary, online):
    locations = []
    for event in online:
        flags = []
        if event.get("stage1_steps_completed") == 3000 or event.get("stage1_stop_reason") == "max_steps_reached":
            flags.append("3000_hit")
        if event.get("failure") or event.get("status") == "failed":
            flags.append("failure")
        realized = event.get("realized_group_count")
        requested = event.get("requested_group_count", summary["requested_group_count"])
        if event.get("budget_violation") or (realized is not None and realized > requested):
            flags.append("budget_violation")
        for flag in flags:
            locations.append({"type": flag, "transfer": TRANSFER_SHORT[summary["transfer"]],
                              "batch_index": event.get("batch_index"), "batch_size": event.get("batch_size")})
    return locations


def candidate(summary_path, sparse):
    s = read_json(summary_path)
    online = read_online(summary_path.with_name("metrics.jsonl"))
    if not online:
        raise RuntimeError(f"No online batches: {summary_path}")
    nonsingle = [x for x in online if x["batch_size"] >= 2]
    single = [x for x in online if x["batch_size"] == 1]
    hit = lambda x: x.get("stage1_steps_completed") == 3000 or x.get("stage1_stop_reason") == "max_steps_reached"
    all_hits = sum(hit(x) for x in online)
    non_hits = sum(hit(x) for x in nonsingle)
    single_hits = sum(hit(x) for x in single)
    sel = s["selection"]
    transfer = TRANSFER_SHORT[s["transfer"]]
    budget = float(s["requested_budget"])
    base = sparse[(s["dataset"], budget, transfer)]
    locations = failure_locations(s, online)
    failure_rate = float(sel["failure_rate"])
    violation_rate = float(sel["budget_violation_rate"])
    return {
        "dataset": s["dataset"], "budget": budget,
        "anchor": summary_path.parents[5].name, "transfer": transfer,
        "alpha": s["lbi"]["alpha"], "kappa": s["lbi"]["kappa"], "nu": s["lbi"]["nu"],
        "PU": s["PU-Acc"], "FO": s["FO-Acc"], **base, "FO_margin": s["FO-Acc"] - base["best_sparse_FO"],
        "failure_rate": failure_rate, "budget_violation_rate": violation_rate,
        "3000_hit_count_all_batches": all_hits, "3000_hit_rate_all_batches": all_hits / len(online),
        "3000_hit_count_nonsingleton": non_hits,
        "3000_hit_rate_nonsingleton": non_hits / len(nonsingle) if nonsingle else 0.0,
        "3000_hit_count_singleton": single_hits, "singleton_batch_count": len(single),
        "mean_utilization": sel["utilization_mean"], "min_utilization": sel["utilization_min"],
        "util95_rate": sel["utilization_ge_95_rate"], "mean_realized_K": sel["realized_group_count_mean"],
        "rollback_rate": sel["rollback_rate"], "stage1_mean_steps": sel["stage1_steps_mean"],
        "stage1_max_steps": sel["stage1_steps_max"], "batch_count": len(online),
        "nonsingleton_batch_count": len(nonsingle), "strict_valid_all_batches": failure_rate == 0 and violation_rate == 0 and all_hits == 0,
        "valid_excluding_singleton": failure_rate == 0 and violation_rate == 0 and non_hits == 0,
        "issue_locations": locations, "artifact": str(summary_path),
    }


def aggregate_office(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["budget"], row["anchor"])].append(row)
    out = []
    for (budget, anchor), group in sorted(grouped.items()):
        if {r["transfer"] for r in group} != set(TRANSFERS):
            raise RuntimeError(f"Incomplete Office transfers: {budget} {anchor}")
        total_batches = sum(r["batch_count"] for r in group)
        util95_count = sum(r["util95_rate"] * r["batch_count"] for r in group)
        a = {
            "budget": budget, "anchor": anchor, "alpha": group[0]["alpha"], "kappa": group[0]["kappa"], "nu": group[0]["nu"],
            "valid_transfer_count_all_batches": sum(r["strict_valid_all_batches"] for r in group),
            "valid_transfer_count_nonsingleton": sum(r["valid_excluding_singleton"] for r in group),
            "mean_FO_margin": mean(r["FO_margin"] for r in group), "worst_transfer_FO_margin": min(r["FO_margin"] for r in group),
            "mean_FO": mean(r["FO"] for r in group), "util95_rate": util95_count / total_batches,
            "mean_utilization": sum(r["mean_utilization"] * r["batch_count"] for r in group) / total_batches,
            "min_utilization": min(r["min_utilization"] for r in group),
            "stage1_max_steps": max(r["stage1_max_steps"] for r in group),
            "stage1_mean_steps": sum(r["stage1_mean_steps"] * r["batch_count"] for r in group) / total_batches,
            "total_3000_hits_all_batches": sum(r["3000_hit_count_all_batches"] for r in group),
            "total_3000_hits_nonsingleton": sum(r["3000_hit_count_nonsingleton"] for r in group),
            "total_3000_hits_singleton": sum(r["3000_hit_count_singleton"] for r in group),
            "strict_valid_all_batches": all(r["strict_valid_all_batches"] for r in group),
            "valid_excluding_singleton": all(r["valid_excluding_singleton"] for r in group),
            "hit_transfers_all": [r["transfer"] for r in group if r["3000_hit_count_all_batches"]],
            "hit_transfers_nonsingleton": [r["transfer"] for r in group if r["3000_hit_count_nonsingleton"]],
        }
        a["singleton_only_strict_invalid"] = (not a["strict_valid_all_batches"] and a["valid_excluding_singleton"]
                                               and a["total_3000_hits_singleton"] > 0)
        out.append(a)
    return out


def rank(rows, validity, metric_fields):
    eligible = [r for r in rows if r[validity]]
    def key(r):
        return tuple((-r[name] if direction == "desc" else r[name]) for name, direction in metric_fields)
    ranked = sorted(eligible, key=key)
    for i, row in enumerate(ranked, 1):
        row = row.copy(); row["rank"] = i
        yield row


def office_per_transfer_md(rows):
    lines = ["# Transformer Group-LBI Stage-1 — Office-31 per-transfer results", "",
             "> Existing artifacts only; no experiment was rerun. PU is diagnostic only and is excluded from ranking.", ""]
    headers = ["anchor", "transfer", "α", "κ", "ν", "PU", "FO", "best sparse FO", "FO margin", "fail", "budget viol.",
               "3000 all (n/r)", "3000 non-single (n/r)", "3000 singleton", "singleton batches", "util mean/min/≥95", "K mean", "rollback", "steps mean/max", "issues"]
    for budget in BUDGETS:
        lines += [f"## Budget {budget:g}", "", "| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
        for r in sorted((x for x in rows if x["budget"] == budget), key=lambda x: (x["anchor"], TRANSFERS.index(x["transfer"]))):
            issues = ", ".join(f"{x['type']}@{x['transfer']}/b{x['batch_index']}/n{x['batch_size']}" for x in r["issue_locations"]) or "—"
            vals = [r["anchor"], r["transfer"], f(r["alpha"]), f(r["kappa"]), f(r["nu"]), f(r["PU"]), f(r["FO"]),
                    f(r["best_sparse_FO"]), f(r["FO_margin"]), f(r["failure_rate"]), f(r["budget_violation_rate"]),
                    f"{r['3000_hit_count_all_batches']}/{f(r['3000_hit_rate_all_batches'])}",
                    f"{r['3000_hit_count_nonsingleton']}/{f(r['3000_hit_rate_nonsingleton'])}", str(r["3000_hit_count_singleton"]),
                    str(r["singleton_batch_count"]), f"{f(r['mean_utilization'])}/{f(r['min_utilization'])}/{f(r['util95_rate'])}",
                    f(r["mean_realized_K"]), f(r["rollback_rate"]), f"{f(r['stage1_mean_steps'])}/{r['stage1_max_steps']}", issues]
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")
    return "\n".join(lines)


def office_ranked_md(aggs):
    metrics = [("mean_FO_margin", "desc"), ("worst_transfer_FO_margin", "desc"), ("mean_FO", "desc"),
               ("util95_rate", "desc"), ("stage1_max_steps", "asc"), ("stage1_mean_steps", "asc")]
    lines = ["# Transformer Group-LBI Stage-1 — Office-31 rankings", "",
             "> Ranking is mechanical and follows the supplied lexicographic rules. Hard90 and PU are not gates or tie-breakers.", ""]
    hdr = ["rank", "anchor", "(α,κ,ν)", "valid strict", "valid non-single", "mean margin", "worst margin", "mean FO", "util≥95", "util mean/min", "steps max/mean", "hits all/non/single", "hit transfers", "classification"]
    ranks_out = {}
    for budget in BUDGETS:
        subset = [x for x in aggs if x["budget"] == budget]
        lines += [f"## Budget {budget:g}", ""]
        for title, validity in [("Ranking A: all-batch strict validity", "strict_valid_all_batches"),
                                ("Ranking B: validity excluding singleton batches", "valid_excluding_singleton")]:
            ranked = list(rank(subset, validity, metrics)); ranks_out[(budget, validity)] = ranked
            lines += [f"### {title}", "", f"Eligible: {len(ranked)}/8", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
            for r in ranked:
                if r["singleton_only_strict_invalid"]: cls = "strict-invalid only from singleton hit"
                elif r["total_3000_hits_nonsingleton"]: cls = "3000 hit on batch_size≥2"
                else: cls = "eligible under this validity rule"
                vals = [str(r["rank"]), r["anchor"], f"({f(r['alpha'])},{f(r['kappa'])},{f(r['nu'])})",
                        f"{r['valid_transfer_count_all_batches']}/6", f"{r['valid_transfer_count_nonsingleton']}/6",
                        f(r["mean_FO_margin"]), f(r["worst_transfer_FO_margin"]), f(r["mean_FO"]), f(r["util95_rate"]),
                        f"{f(r['mean_utilization'])}/{f(r['min_utilization'])}", f"{r['stage1_max_steps']}/{f(r['stage1_mean_steps'])}",
                        f"{r['total_3000_hits_all_batches']}/{r['total_3000_hits_nonsingleton']}/{r['total_3000_hits_singleton']}",
                        ", ".join(r["hit_transfers_all"]) or "—", cls]
                lines.append("| " + " | ".join(vals) + " |")
            lines.append("")
        invalid = [r for r in subset if not r["valid_excluding_singleton"]]
        if invalid:
            lines += ["### Ineligible configurations with nonsingleton 3000 hits", ""]
            for r in invalid:
                lines.append(f"- {r['anchor']}: {r['total_3000_hits_nonsingleton']} hit(s), transfers: {', '.join(r['hit_transfers_nonsingleton']) or 'none'}")
            lines.append("")
    return "\n".join(lines), ranks_out


def visda_md(rows):
    metrics = [("FO_margin", "desc"), ("FO", "desc"), ("util95_rate", "desc"), ("stage1_max_steps", "asc"), ("stage1_mean_steps", "asc")]
    lines = ["# Transformer Group-LBI Stage-1 — VisDA-C rankings", "",
             "> Existing artifacts only. PU is diagnostic and excluded from selection.", ""]
    ranked_out = {}
    hdr = ["rank", "anchor", "α", "κ", "ν", "PU", "FO", "best sparse FO", "margin", "fail", "budget viol.", "3000 all (n/r)", "3000 non-single (n/r)", "3000 singleton", "singleton batches", "util mean/min/≥95", "K mean", "rollback", "steps mean/max", "issues"]
    for budget in BUDGETS:
        subset = [x for x in rows if x["budget"] == budget]
        ranked = list(rank(subset, "strict_valid_all_batches", metrics)); ranked_out[budget] = ranked
        lines += [f"## Budget {budget:g}", "", f"Valid candidates: {len(ranked)}/8", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        for r in ranked:
            issues = ", ".join(f"{x['type']}@b{x['batch_index']}/n{x['batch_size']}" for x in r["issue_locations"]) or "—"
            vals = [str(r["rank"]), r["anchor"], f(r["alpha"]), f(r["kappa"]), f(r["nu"]), f(r["PU"]), f(r["FO"]),
                    f(r["best_sparse_FO"]), f(r["FO_margin"]), f(r["failure_rate"]), f(r["budget_violation_rate"]),
                    f"{r['3000_hit_count_all_batches']}/{f(r['3000_hit_rate_all_batches'])}",
                    f"{r['3000_hit_count_nonsingleton']}/{f(r['3000_hit_rate_nonsingleton'])}", str(r["3000_hit_count_singleton"]),
                    str(r["singleton_batch_count"]), f"{f(r['mean_utilization'])}/{f(r['min_utilization'])}/{f(r['util95_rate'])}",
                    f(r["mean_realized_K"]), f(r["rollback_rate"]), f"{f(r['stage1_mean_steps'])}/{r['stage1_max_steps']}", issues]
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")
    return "\n".join(lines), ranked_out


def main():
    sparse, baseline_sources = baseline_table()
    rows = []
    for dataset_dir in [STAGE1 / "office31", STAGE1 / "visda-c"]:
        for path in sorted(dataset_dir.glob("rho-*/A*/group_lbi_*/results/rho-*/*/*/summary.json")):
            rows.append(candidate(path, sparse))
    office = [x for x in rows if x["dataset"] == "office31"]
    visda = [x for x in rows if x["dataset"] == "visda-c"]
    if len(office) != 144 or len(visda) != 24:
        raise RuntimeError(f"Unexpected completed matrix size: office={len(office)}, visda={len(visda)}")
    aggs = aggregate_office(office)
    office_ranked, office_ranks = office_ranked_md(aggs)
    visda_ranked, visda_ranks = visda_md(visda)
    summary = {
        "office": {str(b): {"strict_valid_candidates": sum(x["strict_valid_all_batches"] for x in aggs if x["budget"] == b),
                             "nonsingleton_valid_candidates": sum(x["valid_excluding_singleton"] for x in aggs if x["budget"] == b)} for b in BUDGETS},
        "visda": {str(b): {"valid_candidates": sum(x["strict_valid_all_batches"] for x in visda if x["budget"] == b)} for b in BUDGETS},
    }
    office_ranked += "\n## Brief validity summary\n\n" + "\n".join(
        f"- Office budget {b:g}: strict-valid candidates = {summary['office'][str(b)]['strict_valid_candidates']}; nonsingleton-valid candidates = {summary['office'][str(b)]['nonsingleton_valid_candidates']}"
        for b in BUDGETS) + "\n"
    visda_ranked += "\n## Brief validity summary\n\n" + "\n".join(
        f"- VisDA budget {b:g}: valid candidates = {summary['visda'][str(b)]['valid_candidates']}" for b in BUDGETS) + "\n"
    (OUT / "stage1_office_per_transfer.md").write_text(office_per_transfer_md(office) + "\n")
    (OUT / "stage1_office_ranked.md").write_text(office_ranked)
    (OUT / "stage1_visda_ranked.md").write_text(visda_ranked)
    payload = {"schema_version": 1, "scope": "existing Stage-1 artifacts only", "baseline_sources": baseline_sources,
               "office_per_transfer": office, "office_aggregate": aggs,
               "office_rankings": {f"{b}:{v}": rows for (b, v), rows in office_ranks.items()},
               "visda_candidates": visda, "visda_rankings": {str(k): v for k, v in visda_ranks.items()}, "summary": summary}
    (OUT / "stage1_tuning_full.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
