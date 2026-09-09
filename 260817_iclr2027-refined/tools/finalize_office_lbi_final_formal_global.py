#!/usr/bin/env python3
"""Artifact-only global FINALIZE for the 18 Office final-formal LBI runs."""
import csv
import datetime as dt
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path: sys.path.insert(0, str(PROJECT))
from tools.check_experiment_status import scan_run_summaries

ROOT = PROJECT / "experiment_logs/shot_otta_office_lbi_final_formal_20260818"
BASE = PROJECT / "experiment_logs/shot_otta_office_seed2026_stage1_20260818"
SELECTED = PROJECT / "experiment_logs/shot_otta_office_lbi_joint_sweep_20260818/selected_configs/OFFICE_FINAL_LBI_TUPLES.json"
MACHINES = (("machine1", 0.0005), ("machine2", 0.001), ("machine3", 0.002))
TRANSFERS = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
NAMES = {(0,1):"A->D",(0,2):"A->W",(1,0):"D->A",(1,2):"D->W",(2,0):"W->A",(2,1):"W->D"}

def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True); Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
def write_csv(path, rows):
    fields=list(rows[0])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w",newline="",encoding="utf-8") as fh:
        writer=csv.DictWriter(fh,fieldnames=fields,extrasaction="ignore"); writer.writeheader()
        for r in rows: writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def num(v, label):
    if v is None or isinstance(v,bool): raise ValueError(f"missing {label}")
    v=float(v)
    if not math.isfinite(v): raise ValueError(f"nonfinite {label}")
    return v
def avg(values): return sum(values)/len(values)
def same(a,b): return math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-15)

def summary_index(root):
    records, invalid=scan_run_summaries(str(root)); index=defaultdict(list)
    for r in records: index[r['experiment_key']].append(r)
    return index,invalid

def completed_for_plan(plan_path,index,invalid):
    plan=read(plan_path); rows=[]; problems=[]
    for entry in plan.get('experiments',[]):
        exact=[r for r in index.get(entry['experiment_key'],[]) if r['experiment_config_sha256']==entry['experiment_config_sha256']]
        mismatch=[r for r in index.get(entry['experiment_key'],[]) if r['experiment_config_sha256']!=entry['experiment_config_sha256']]
        done=[r for r in exact if r['status']=='completed']
        if mismatch: problems.append((entry['experiment_key'],'hash mismatch'))
        elif len(exact)>1: problems.append((entry['experiment_key'],'duplicate'))
        elif len(done)!=1: problems.append((entry['experiment_key'],f"completed={len(done)}"))
        else: rows.append((entry,done[0]))
    if invalid: problems.extend((x.get('experiment_key'), 'invalid summary') for x in invalid)
    if problems: raise RuntimeError(json.dumps({'plan':str(plan_path),'problems':problems},indent=2))
    return plan,rows

def lbi_rows():
    index,invalid=summary_index(ROOT/'runs'); output=[]; tuples={}
    for machine,budget in MACHINES:
        marker=ROOT/f'phase_records/{machine}/FINALIZE_READY.json'
        if not marker.exists(): raise RuntimeError(f'missing marker: {marker}')
        plan_path=ROOT/f'plans/{machine}/plan.json'; plan,rows=completed_for_plan(plan_path,index,invalid)
        if len(rows)!=6 or {(e['source'],e['target']) for e,_ in rows}!=set(TRANSFERS): raise RuntimeError(f'not six transfers: {machine}')
        for entry,record in rows:
            s=record['summary']; frozen={'requested_budget':budget,'alpha':entry['alpha'],'kappa':entry['kappa'],'nu':entry['nu'],'omega':entry['omega'],'stage2_lr':entry['stage2_lr'],'stage1_max_steps':3000,'stage2_steps_requested':1,'seed':2026}
            for key,value in frozen.items():
                if key not in s or not same(s[key],value): raise RuntimeError(f'drift {machine} {entry["experiment_key"]}: {key}')
            if s.get('variant')!='module_lbi' or s.get('runtime_comparable') is not True: raise RuntimeError(f'invalid formal run: {entry["experiment_key"]}')
            output.append({'machine':machine,'budget':budget,'source':entry['source'],'target':entry['target'],'transfer':NAMES[(entry['source'],entry['target'])], 'experiment_key':entry['experiment_key'],'experiment_config_sha256':entry['experiment_config_sha256'],'PU':num(s.get('PU-Acc'),'PU'),'FO':num(s.get('FO-Acc'),'FO'),'summary_path':record['summary_path'],'gpu_name':s.get('gpu_name'),'torch_version':s.get('torch_version'),'cuda_version':s.get('cuda_version'),'runtime_comparable':s.get('runtime_comparable')})
        tuples[budget]={'alpha':rows[0][0]['alpha'],'kappa':rows[0][0]['kappa'],'nu':rows[0][0]['nu'],'omega':rows[0][0]['omega'],'stage2_lr':rows[0][0]['stage2_lr']}
    if len(output)!=18 or len({x['experiment_key'] for x in output})!=18: raise RuntimeError('18-run identity check failed')
    return output,tuples

def baseline_rows():
    index,invalid=summary_index(BASE/'runs'); result=[]
    plans=[BASE/'plans/baseline_machine1/plan.json',BASE/'plans/baseline_machine2/plan.json',BASE/'plans/machine3/baseline/plan.json']
    for plan in plans:
        _,rows=completed_for_plan(plan,index,invalid)
        for e,r in rows:
            s=r['summary']; result.append({'source':e['source'],'target':e['target'],'variant':e['variant'],'budget':e.get('requested_budget'),'PU':num(s.get('PU-Acc'),'baseline PU'),'FO':num(s.get('FO-Acc'),'baseline FO'),'gpu_name':s.get('gpu_name'),'torch_version':s.get('torch_version'),'cuda_version':s.get('cuda_version'),'runtime_comparable':s.get('runtime_comparable')})
    if len(result)!=72: raise RuntimeError(f'baseline count={len(result)}')
    return result

def normalize_summary(machine):
    a=read(ROOT/f'reports/{machine}/final_budget_summary.json')
    return a.get('summary',a.get('efficiency',a)), a.get('hardware_comparability',a.get('hardware',{}))

def main():
    selected=read(SELECTED)
    if selected.get('office_lbi_hyperparameter_search_closed') is not True: raise RuntimeError('Office LBI search not closed')
    lbi,tuples=lbi_rows(); baseline=baseline_rows()
    for budget,t in tuples.items():
        expected=selected['budgets'][format(budget,'.4g')]
        for key,value in t.items():
            if not same(value,expected[key]): raise RuntimeError(f'selected tuple drift budget={budget}: {key}')
    grouped=defaultdict(dict)
    for r in baseline: grouped[(r['source'],r['target'],r['budget'])][r['variant']]=r
    base_summary=[]
    for label,variant,budget in [('Source-only','source_only',None),('Full dense','full_dense',1.0),('Module dense','module_dense',1.0)]:
        rows=[r for r in baseline if r['variant']==variant and r['budget']==budget]
        if len(rows)!=6: raise RuntimeError(f'baseline table missing {label}')
        base_summary.append({'method':label,'budget':'—','PU':avg([r['PU'] for r in rows]),'FO':avg([r['FO'] for r in rows]),'row_count':6})
    for budget in (0.0005,0.001,0.002):
        for label,variant in [('Random','module_random'),('Magnitude','module_magnitude'),('Saliency','module_saliency')]:
            rows=[r for r in baseline if r['variant']==variant and same(r['budget'],budget)]
            if len(rows)!=6: raise RuntimeError(f'baseline table missing {label}@{budget}')
            base_summary.append({'method':label,'budget':budget,'PU':avg([r['PU'] for r in rows]),'FO':avg([r['FO'] for r in rows]),'row_count':6})
    lbi_by_budget=defaultdict(list)
    for r in lbi: lbi_by_budget[r['budget']].append(r)
    analysis_budgets=[]
    for budget,rows in sorted(lbi_by_budget.items()):
        bmap={x['variant']:x for t in TRANSFERS for x in grouped[(t[0],t[1],budget)].values()}
        # per-transfer comparison uses the required sparse maximum.
        margins=[]
        for r in rows:
            m=grouped[(r['source'],r['target'],budget)]
            dense=grouped[(r['source'],r['target'],1.0)]
            sparse=max((m[x] for x in ('module_random','module_magnitude','module_saliency')),key=lambda x:x['FO'])
            r['best_sparse_variant']=sparse['variant']; r['best_sparse_FO']=sparse['FO']; r['best_sparse_PU']=sparse['PU']; r['module_dense_FO']=dense['module_dense']['FO']; r['module_dense_PU']=dense['module_dense']['PU']; r['full_dense_FO']=dense['full_dense']['FO']; r['full_dense_PU']=dense['full_dense']['PU']
            for label in ('best_sparse','module_dense','full_dense'):
                r[f'LBI_minus_{label}_FO']=r['FO']-r[f'{label}_FO']; r[f'LBI_minus_{label}_PU']=r['PU']-r[f'{label}_PU']
            margins.append(r)
        base_summary.append({'method':'LBI','budget':budget,'PU':avg([r['PU'] for r in rows]),'FO':avg([r['FO'] for r in rows]),'row_count':6})
        analysis_budgets.append({'budget':budget,'frozen_tuple':tuples[budget],'mean_PU':avg([r['PU'] for r in rows]),'mean_FO':avg([r['FO'] for r in rows]),'best_sparse_mean_PU':avg([r['best_sparse_PU'] for r in rows]),'best_sparse_mean_FO':avg([r['best_sparse_FO'] for r in rows]),'LBI_minus_best_sparse_mean_PU':avg([r['LBI_minus_best_sparse_PU'] for r in rows]),'LBI_minus_best_sparse_mean_FO':avg([r['LBI_minus_best_sparse_FO'] for r in rows]),'module_dense_mean_PU':avg([r['module_dense_PU'] for r in rows]),'module_dense_mean_FO':avg([r['module_dense_FO'] for r in rows]),'LBI_minus_module_dense_mean_PU':avg([r['LBI_minus_module_dense_PU'] for r in rows]),'LBI_minus_module_dense_mean_FO':avg([r['LBI_minus_module_dense_FO'] for r in rows]),'full_dense_mean_PU':avg([r['full_dense_PU'] for r in rows]),'full_dense_mean_FO':avg([r['full_dense_FO'] for r in rows]),'LBI_minus_full_dense_mean_PU':avg([r['LBI_minus_full_dense_PU'] for r in rows]),'LBI_minus_full_dense_mean_FO':avg([r['LBI_minus_full_dense_FO'] for r in rows])})
    efficiency=[]
    for machine,budget in MACHINES:
        s,hardware=normalize_summary(machine)
        models={r['gpu_name'] for r in lbi if r['machine']==machine}; versions={r['torch_version'] for r in lbi if r['machine']==machine}; cuda={r['cuda_version'] for r in lbi if r['machine']==machine}
        efficiency.append({'budget':budget,'machine':machine,'runtime_comparable':hardware.get('runtime_comparable',True),'gpu_models':sorted(models),'torch_versions':sorted(versions),'cuda_versions':sorted(cuda),'online_batch_runtime_mean_sec':s['online_batch_runtime_mean_sec'],'online_compute_runtime_sec':s['online_compute_runtime_sec'],'fo_eval_runtime_sec':s['fo_eval_runtime_sec'],'gpu_peak_allocated_max_mb':s['gpu_peak_allocated_max_mb'],'gpu_peak_reserved_max_mb':s['gpu_peak_reserved_max_mb']})
    baseline_hardware={(r['gpu_name'],r['torch_version'],r['cuda_version']) for r in baseline}; lbi_hardware={(r['gpu_name'],r['torch_version'],r['cuda_version']) for r in lbi}
    comparable=all(r['runtime_comparable'] is True for r in lbi) and lbi_hardware==baseline_hardware
    best=max(analysis_budgets,key=lambda x:x['mean_FO'])
    reports=ROOT/'reports/global'
    lbi.sort(key=lambda x:(x['budget'],x['source'],x['target']))
    write_csv(reports/'OFFICE_FINAL_LBI_18_RUNS.csv',lbi)
    base_summary.sort(key=lambda x: ({'Source-only':0,'Full dense':1,'Module dense':2,'Random':3,'Magnitude':4,'Saliency':5,'LBI':6}[x['method']], str(x['budget'])))
    write_csv(reports/'OFFICE_FINAL_RESULTS_TABLE.csv',base_summary)
    write_csv(reports/'OFFICE_FINAL_EFFICIENCY_TABLE.csv',efficiency)
    result_md=['# Office final results — 18 formal LBI runs','', '| method | budget | PU | FO | n |','|---|---:|---:|---:|---:|']
    result_md += [f"| {r['method']} | {r['budget']} | {r['PU']:.6f} | {r['FO']:.6f} | {r['row_count']} |" for r in base_summary]
    result_md += ['',f"Highest final formal LBI mean FO: budget {best['budget']}, FO={best['mean_FO']:.6f}.",'','Office LBI hyperparameter tuning is closed.','These 18 runs are final formal evaluation runs.','No further Office LBI tuning is recommended.']
    (reports/'OFFICE_FINAL_RESULTS_TABLE.md').write_text('\n'.join(result_md)+'\n',encoding='utf-8')
    eff_md=['# Office final LBI efficiency','',f"Strict runtime comparison allowed: `{str(comparable).lower()}`.",'','| budget | online batch sec | online compute sec | FO eval sec | peak allocated MB | comparable |','|---:|---:|---:|---:|---:|---|']
    eff_md += [f"| {r['budget']} | {r['online_batch_runtime_mean_sec']:.6f} | {r['online_compute_runtime_sec']:.6f} | {r['fo_eval_runtime_sec']:.6f} | {r['gpu_peak_allocated_max_mb']:.3f} | {str(r['runtime_comparable']).lower()} |" for r in efficiency]
    (reports/'OFFICE_FINAL_EFFICIENCY_TABLE.md').write_text('\n'.join(eff_md)+'\n',encoding='utf-8')
    analysis={'schema_version':1,'created_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'formal_lbi_run_count':len(lbi),'formal_lbi_completed':18,'runtime_comparable':comparable,'hardware_reason':'all formal LBI and seed-2026 baseline runs share GPU model/PyTorch/CUDA provenance' if comparable else 'hardware provenance differs; do not make strict runtime comparisons','frozen_tuples':tuples,'budget_analysis':analysis_budgets,'highest_final_formal_mean_FO_budget':best['budget'],'office_lbi_hyperparameter_tuning_closed':True,'these_are_final_formal_evaluation_runs':True,'no_further_office_lbi_tuning_recommended':True,'training_launched_by_finalize':False}
    write_json(reports/'OFFICE_FINAL_ANALYSIS.json',analysis)
    lines=['# Office final LBI analysis','', '18 / 18 final formal LBI runs are complete.',f"runtime_comparable = {str(comparable).lower()}.",'', '| budget | LBI PU | LBI FO | LBI − best sparse FO | LBI − Module dense FO | LBI − Full dense FO |','|---:|---:|---:|---:|---:|---:|']
    lines += [f"| {r['budget']} | {r['mean_PU']:.6f} | {r['mean_FO']:.6f} | {r['LBI_minus_best_sparse_mean_FO']:.6f} | {r['LBI_minus_module_dense_mean_FO']:.6f} | {r['LBI_minus_full_dense_mean_FO']:.6f} |" for r in analysis_budgets]
    lines += ['',f"Frozen budget with highest final formal mean FO: `{best['budget']}` ({best['mean_FO']:.6f}).",'','Office LBI hyperparameter tuning is closed.','These 18 runs are final formal evaluation runs.','No further Office LBI tuning is recommended.','', 'No training was launched by FINALIZE.']
    (reports/'OFFICE_FINAL_ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'formal_lbi_completed':18,'runtime_comparable':comparable,'highest_budget':best['budget'],'highest_mean_FO':best['mean_FO']},ensure_ascii=False))
if __name__=='__main__': main()
