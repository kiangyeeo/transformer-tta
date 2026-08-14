#!/usr/bin/env python3
"""Rank budget-specific LBI candidates without hiding incomplete or invalid runs."""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path

TRANSFERS = {(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)}
TUPLE_FIELDS = ("alpha", "kappa", "nu", "omega", "stage1_max_steps", "budget_tolerance", "stage2_lr", "stage2_steps_requested")
def load_json(path):
    with Path(path).open(encoding="utf-8") as handle: return json.load(handle)
def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer=csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)
def number(value): return None if value in (None, "") else float(value)
def tuple_from(record): return tuple(record.get(field) for field in TUPLE_FIELDS)
def tuple_label(values): return "|".join(f"{field}={value}" for field,value in zip(TUPLE_FIELDS,values))
def load_baselines(path):
    result={}
    with Path(path).open(newline="",encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key=(int(row["source"]),int(row["target"]),int(row["seed"]),float(row["requested_budget"]))
            if key in result: raise ValueError(f"duplicate baseline identity: {key}")
            result[key]=float(row["best_sparse_FO"])
    return result
def scan_summaries(runs_root):
    summaries={}
    for path in sorted(Path(runs_root).rglob("summary.json")):
        try: summary=load_json(path)
        except (OSError,json.JSONDecodeError) as exc: raise ValueError(f"invalid summary {path}: {exc}") from exc
        key=summary.get("experiment_key")
        if not key: raise ValueError(f"summary missing experiment_key: {path}")
        if key in summaries: raise ValueError(f"duplicate run identity: {key}: {summaries[key][0]} and {path}")
        summaries[key]=(path,summary)
    return summaries
def analyze(plan_path,runs_root,baseline_path,budget,seed,phase,output_dir,top_k):
    plan=load_json(plan_path); experiments=plan.get("experiments")
    if not isinstance(experiments,list): raise ValueError("plan must contain an experiments list")
    selected=[e for e in experiments if e.get("variant")=="module_lbi" and float(e.get("requested_budget"))==budget and int(e.get("seed"))==seed]
    if not selected: raise ValueError("no matching LBI experiments in plan")
    identities=set()
    for exp in selected:
        key=exp.get("experiment_key")
        if not key or key in identities: raise ValueError(f"duplicate or missing planned run identity: {key}")
        identities.add(key)
    summaries=scan_summaries(runs_root); baselines=load_baselines(baseline_path); grouped=defaultdict(list); per_run=[]
    for exp in selected:
        values=tuple_from(exp); label=tuple_label(values); source,target=int(exp["source"]),int(exp["target"]); baseline_key=(source,target,seed,budget)
        if baseline_key not in baselines: raise ValueError(f"missing same-budget baseline: {baseline_key}")
        summary_path,summary=summaries.get(exp["experiment_key"],(None,None)); completed=bool(summary and summary.get("status")=="completed"); valid=bool(completed and summary.get("valid_lbi_run") is True); fo=number(summary.get("FO-Acc")) if summary else None; margin=fo-baselines[baseline_key] if fo is not None else None
        row={"phase":phase,"candidate_tuple":label,"experiment_key":exp["experiment_key"],"source":source,"target":target,"seed":seed,"requested_budget":budget,"summary_path":str(summary_path) if summary_path else "","completed":completed,"valid_lbi_run":valid,"FO_Acc":fo,"best_sparse_FO":baselines[baseline_key],"FO_margin":margin,"budget_reached_all":bool(summary and summary.get("budget_reached_all_steps")),"max_steps_hit_count":int(summary.get("max_steps_hit_count",0)) if summary else 0,"stage1_steps":number(summary.get("stage1_steps_mean")) if summary else None,"runtime":number(summary.get("runtime")) if summary else None}
        per_run.append(row); grouped[values].append(row)
    rankings=[]
    for values,rows in grouped.items():
        transfers={(r["source"],r["target"]) for r in rows}
        if len(rows)!=len(transfers): raise ValueError(f"duplicate transfer identity for candidate {tuple_label(values)}")
        if transfers != TRANSFERS or len(rows) != 6: raise ValueError(f"candidate lacks required six directed transfers: {tuple_label(values)}")
        complete=True; numeric=lambda field,subset=rows:[r[field] for r in subset if r[field] is not None]; valid_rows=[r for r in rows if r["valid_lbi_run"]]; margins=numeric("FO_margin"); valid_margins=numeric("FO_margin",valid_rows); fos=numeric("FO_Acc"); valid_fos=numeric("FO_Acc",valid_rows); steps=numeric("stage1_steps"); runtimes=numeric("runtime")
        ranking={"phase":phase,"candidate_tuple":tuple_label(values),**dict(zip(TUPLE_FIELDS,values)),"complete_six_transfer_configuration":complete,"planned_run_count":len(rows),"completed_run_count":sum(r["completed"] for r in rows),"valid_lbi_run_count":len(valid_rows),"valid_lbi_run_ratio":len(valid_rows)/len(rows),"budget_reached_all_count":sum(r["budget_reached_all"] for r in rows),"max_steps_hit_count_sum":sum(r["max_steps_hit_count"] for r in rows),"mean_FO_all":sum(fos)/len(fos) if fos else None,"mean_FO_valid_only":sum(valid_fos)/len(valid_fos) if valid_fos else None,"mean_FO_margin_all":sum(margins)/len(margins) if margins else None,"mean_FO_margin_valid_only":sum(valid_margins)/len(valid_margins) if valid_margins else None,"worst_transfer_FO_margin":min(margins) if margins else None,"best_transfer_FO_margin":max(margins) if margins else None,"stage1_steps_min":min(steps) if steps else None,"stage1_steps_mean":sum(steps)/len(steps) if steps else None,"stage1_steps_max":max(steps) if steps else None,"runtime_mean":sum(runtimes)/len(runtimes) if runtimes else None,"runtime_sum":sum(runtimes) if runtimes else None}
        rankings.append(ranking)
    low=lambda value:float("-inf") if value is None else value; high=lambda value:float("inf") if value is None else value
    rankings.sort(key=lambda r:(-r["valid_lbi_run_count"],-low(r["mean_FO_margin_all"]),-low(r["worst_transfer_FO_margin"]),-low(r["mean_FO_all"]),high(r["stage1_steps_max"]),high(r["stage1_steps_mean"]),high(r["runtime_sum"]),r["candidate_tuple"]))
    for index,row in enumerate(rankings,1): row["rank"]=index
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    per_fields=["phase","candidate_tuple","experiment_key","source","target","seed","requested_budget","summary_path","completed","valid_lbi_run","FO_Acc","best_sparse_FO","FO_margin","budget_reached_all","max_steps_hit_count","stage1_steps","runtime"]
    rank_fields=["rank","phase","candidate_tuple",*TUPLE_FIELDS,"complete_six_transfer_configuration","planned_run_count","completed_run_count","valid_lbi_run_count","valid_lbi_run_ratio","budget_reached_all_count","max_steps_hit_count_sum","mean_FO_all","mean_FO_valid_only","mean_FO_margin_all","mean_FO_margin_valid_only","worst_transfer_FO_margin","best_transfer_FO_margin","stage1_steps_min","stage1_steps_mean","stage1_steps_max","runtime_mean","runtime_sum"]
    write_csv(output_dir/"per_run.csv",per_run,per_fields); write_csv(output_dir/"candidate_ranking.csv",rankings,rank_fields); (output_dir/"candidate_ranking.json").write_text(json.dumps(rankings,indent=2)+"\n",encoding="utf-8"); (output_dir/"candidate_ranking.md").write_text("# Candidate ranking\n\n"+"\n".join(f"{r['rank']}. `{r['candidate_tuple']}` — valid {r['valid_lbi_run_count']}/{r['planned_run_count']}, margin {r['mean_FO_margin_all']}" for r in rankings)+"\n",encoding="utf-8")
    chosen=[r for r in rankings if r["complete_six_transfer_configuration"]][:top_k]; selection={"phase":phase,"budget":budget,"seed":seed,"ranking_rule":["valid_lbi_run_count desc","mean_FO_margin_all desc","worst_transfer_FO_margin desc","mean_FO_all desc","stage1_steps_max asc","stage1_steps_mean asc","runtime_sum asc"],"top_k":top_k,"complete_configuration_count":sum(r["complete_six_transfer_configuration"] for r in rankings),"selected":chosen}; (output_dir/"selection.json").write_text(json.dumps(selection,indent=2)+"\n",encoding="utf-8"); return rankings
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--plan",required=True); parser.add_argument("--runs-root",required=True); parser.add_argument("--baseline-reference",required=True); parser.add_argument("--budget",type=float,required=True); parser.add_argument("--seed",type=int,required=True); parser.add_argument("--phase",required=True); parser.add_argument("--output-dir",required=True); parser.add_argument("--top-k",type=int,required=True); args=parser.parse_args(); analyze(args.plan,args.runs_root,args.baseline_reference,args.budget,args.seed,args.phase,args.output_dir,args.top_k)
if __name__=="__main__": main()
