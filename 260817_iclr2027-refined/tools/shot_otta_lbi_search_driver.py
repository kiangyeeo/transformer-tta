#!/usr/bin/env python3
"""Reproducible phased SHOT-OTTA Split-LBI search driver (4 GPU launcher)."""
import argparse, csv, hashlib, json, os, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[2]; ICLR=ROOT/'iclr2027'; sys.path.insert(0,str(ICLR))
from tools.check_experiment_status import check_status, load_plan

def utc(): return datetime.now(timezone.utc).isoformat()
def cmdlog(hist, args, cwd=ROOT):
    import shlex
    with hist.open('a',encoding='utf-8') as f: f.write('cd '+shlex.quote(str(cwd))+'\n'+shlex.join([str(x) for x in args])+'\n')
def run(hist,args,cwd=ROOT,stdout=None):
    cmdlog(hist,args,cwd)
    if stdout:
        stdout.parent.mkdir(parents=True,exist_ok=True)
        with stdout.open('w',encoding='utf-8') as out: subprocess.run(args,cwd=cwd,check=True,stdout=out,stderr=subprocess.STDOUT,text=True)
    else: subprocess.run(args,cwd=cwd,check=True)
def sha(p):
    h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(x if isinstance(x,str) else json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def matrix(trials,budgets,variants=None):
    v=variants or {'module_lbi':{'budgets':budgets,'lbi_trials':trials}}
    return {'method':'shot','task':'otta','datasets':{'office':{'transfers':[[0,1]]}},'seeds':[2026],'variants':v,'common_training':{'batch_size':64,'workers':4,'gpu_id':'0','save_model':False},'output_root':'iclr2027/runs'}
def load_summary(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def valid(s): return s.get('budget_reached_all_steps') is True and s.get('max_steps_hit_count',0)==0
def all_summaries():
    result=[]
    for p in (ICLR/'runs').glob('**/summary.json'):
        try: result.append((p,load_summary(p)))
        except Exception: pass
    return result
def baseline(s):
    b=[]
    for p,x in all_summaries():
        if x.get('variant') in {'module_random','module_magnitude','module_saliency'} and all(x.get(k)==s.get(k) for k in ('dataset','source','target','seed','requested_budget')) and x.get('candidate_scope')=='netB.bottleneck': b.append((p,x))
    return max(b,key=lambda z:z[1].get('FO-Acc',float('-inf'))) if b else (None,None)
def enrich(path,s):
    bp,bs=baseline(s); s=dict(s); s['summary_path']=str(path); s['valid_lbi_run']=valid(s); s['baseline_summary_path']=str(bp) if bp else None; s['best_sparse_variant']=bs.get('variant') if bs else None; s['best_sparse_exact_selected_count']=bs.get('selected_param_count') if bs else None; s['best_sparse_FO']=bs.get('FO-Acc') if bs else None; s['FO_margin']=(s['FO-Acc']-bs['FO-Acc']) if bs else None; return s
def rank(rows): return sorted(rows,key=lambda x:(x['valid_lbi_run'],x.get('FO-Acc',-1e9),x.get('FO_margin',-1e9) if x.get('FO_margin') is not None else -1e9,-x.get('stage1_steps_completed_max',1e99),-x.get('stage1_steps_completed_mean',1e99),-x.get('runtime',1e99)),reverse=True)
def trial(s): return {k:s[k] for k in ('alpha','kappa','nu','omega','stage1_max_steps','budget_tolerance','stage2_lr','delta_nonzero_tolerance') if k in s} | {'stage2_steps':s.get('stage2_steps_requested',1)}
def report(p,title,rows,note=''):
    fields=['alpha','kappa','nu','omega','stage2_lr','requested_budget','valid_lbi_run','budget_reached_all_steps','budget_hit_rate','max_steps_hit_count','FO-Acc','best_sparse_variant','best_sparse_FO','FO_margin','stage1_steps_completed_mean','stage1_steps_completed_max','runtime','online_batch_runtime_mean_sec','online_batch_runtime_std_sec','online_batch_runtime_median_sec','online_batch_runtime_p95_sec','online_compute_runtime_sec','adapt_batch_runtime_mean_sec','adapt_batch_runtime_std_sec','adapt_runtime_total_sec','pu_batch_runtime_mean_sec','pu_batch_runtime_std_sec','pu_runtime_total_sec','gpu_peak_allocated_mean_mb','gpu_peak_allocated_max_mb','gpu_peak_reserved_mean_mb','gpu_peak_reserved_max_mb','fo_eval_runtime_sec','wall_runtime_sec','peak_gpu_memory_allocated_mb','peak_gpu_memory_reserved_mb','gpu_name','runtime_comparable','summary_path']
    lines=[f'# {title}','',note,'','|'+ '|'.join(fields)+'|','|'+ '|'.join(['---']*len(fields))+'|']
    for r in rows: lines.append('|'+ '|'.join(str(r.get(f,'')) for f in fields)+'|')
    write(p,'\n'.join(lines)+'\n')
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--search-id',required=True); a=ap.parse_args(); sid=a.search_id; D=ICLR/'experiment_logs'/sid
 for x in ['matrices','plans','launcher_logs','status','results','round_reports','final']: (D/x).mkdir(parents=True,exist_ok=True)
 hist=D/'02_COMMAND_HISTORY.sh'; hist.touch(exist_ok=True); os.chmod(hist,0o755)
 if not (D/'00_PROTOCOL.md').exists():
  write(D/'00_PROTOCOL.md','''# Protocol\n\nGoal: search valid SHOT-OTTA Split-LBI on Office Amazon→DSLR, seed 2026, batch size 64. Fixed semantics: SHOT loss, candidate scope `netB.bottleneck`, masked-delta initialization, Stage 3 accumulation, frozen BN, reset-every-online-step state, data order, source transfer, 3000 Stage-1 max steps, tolerance 0.0001 and Stage-2 steps 1. Valid LBI requires budget reached at every online step and zero max-step hits. Rank: validity, FO-Acc, FO margin against strongest matching sparse baseline, lower max/mean Stage-1 steps, runtime. One worker is assigned to each GPU 0–3; GPU id is operational only. Missing identities run once with one retry; completed identities are skipped. Round 4 adds 0.0005 and conditionally 0.004/0.005 under the stated validity, max-step, positive-margin, and FO-drop rules.\n''')
  env=[]
  for c in [['date','-u'],['date'],['hostname'],['pwd'],['whoami'],['git','rev-parse','HEAD'],['git','status','--short'],['conda','info','--envs'],['python','--version'],['which','python'],['python','-c','import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())'],['nvidia-smi']]:
   cmdlog(hist,c); q=subprocess.run(c,cwd=ROOT,text=True,capture_output=True); env.append('$ '+' '.join(c)+'\n'+q.stdout+q.stderr)
  env.append('\nSHA256\n'+'\n'.join(f'{sha(ICLR/x)}  iclr2027/{x}' for x in ['train.py','tools/plan_experiments.py','tools/run_experiments.py','tools/summarize_runs.py','core/lbi/engine.py','shot_otta/trainer.py','configs/otta_fc_lbi_protocol_20260817_v1.yaml']))
  write(D/'01_ENVIRONMENT.txt','\n'.join(env)); write(D/'04_DECISION_LOG.md','# Decision log\n'); write(D/'05_CURRENT_STATUS.md',f'Search ID: {sid}\nCurrent Round: initialization\n')
 def phase(name,mx):
  mp=D/'matrices'/f'{name}.yaml'; pp=D/'plans'/name; lp=D/'launcher_logs'/name; sp=D/'status'/name; rp=D/'results'/name
  write(mp,yaml.safe_dump(mx,sort_keys=False)); run(hist,['python','iclr2027/tools/plan_experiments.py',str(mp),'iclr2027/configs/otta_fc_lbi_protocol_20260817_v1.yaml','--dry-run'],stdout=lp/'planner_dry_run.log'); run(hist,['python','iclr2027/tools/plan_experiments.py',str(mp),'iclr2027/configs/otta_fc_lbi_protocol_20260817_v1.yaml','--output-dir',str(pp)])
  run(hist,['python','iclr2027/tools/check_experiment_status.py',str(pp/'plan.json'),'iclr2027/runs','--output-dir',str(sp/'pre')]); run(hist,['python','iclr2027/tools/run_experiments_multi_gpu.py',str(pp/'plan.json'),'--runs-root','iclr2027/runs','--logs-root',str(lp),'--gpus','0,1,2,3','--max-workers','4','--dry-run','--command-history',str(hist)],stdout=lp/'launcher_dry_run.json')
  run(hist,['python','iclr2027/tools/run_experiments_multi_gpu.py',str(pp/'plan.json'),'--runs-root','iclr2027/runs','--logs-root',str(lp),'--gpus','0,1,2,3','--max-workers','4','--resume','--summarize-after-run','--command-history',str(hist)])
  run(hist,['python','iclr2027/tools/check_experiment_status.py',str(pp/'plan.json'),'iclr2027/runs','--output-dir',str(sp/'post')]); run(hist,['python','iclr2027/tools/summarize_runs.py','iclr2027/runs','--plan',str(pp/'plan.json'),'--output-dir',str(rp)])
  plan=load_plan(pp/'plan.json'); st=check_status(plan,str(ICLR/'runs')); by={x['experiment_key']:x for x in st['experiments']}; rows=[]
  for e in plan['experiments']:
   ps=by[e['experiment_key']].get('matching_summary_paths',[])
   if ps: rows.append(enrich(ps[0],load_summary(ps[0])))
  write(D/'05_CURRENT_STATUS.md',f'Search ID: {sid}\nCurrent Round: {name}\ncompleted: {len(rows)}/{len(plan["experiments"])}\nresume: tmux attach -t shot_lbi_search_{sid}\n')
  return rows
 fixed=lambda alpha=.1,kappa=1,nu=1,omega=.1,lr=.01:{'alpha':alpha,'kappa':kappa,'nu':nu,'omega':omega,'stage1_max_steps':3000,'budget_tolerance':.0001,'stage2_lr':lr,'stage2_steps':1,'delta_nonzero_tolerance':1e-12}
 r1=phase('ROUND1_STAGE1',matrix([fixed(.15),fixed(.2),fixed(.1,1.5),fixed(.1,2),fixed(.1,1,.5)],[.002])); old=[enrich(p,x) for p,x in all_summaries() if x.get('variant')=='module_lbi' and x.get('requested_budget')==.002 and x.get('alpha')==.1 and x.get('kappa')==1 and x.get('nu')==1 and x.get('omega')==.1 and x.get('stage2_lr')==.01]
 r1all=r1+old; report(D/'round_reports'/'ROUND1_STAGE1_SEARCH.md','Round 1 Stage 1 search',rank(r1all),'Old baseline is read-only and was not rerun.')
 top=[trial(x) for x in rank(r1all) if x['valid_lbi_run']][:2]
 if not top:
  r1f=phase('ROUND1_FALLBACK',matrix([fixed(.3),fixed(.1,3),fixed(.1,1,.25),fixed(.2,2),fixed(.2,1,.5),fixed(.1,2,.5)],[.002])); report(D/'round_reports'/'ROUND1_FALLBACK.md','Round 1 fallback',rank(r1f)); top=[trial(x) for x in rank(r1f) if x['valid_lbi_run']][:2]
 if not top: raise RuntimeError('No valid Round 1 LBI configuration; stopping per protocol')
 r2=phase('ROUND2_OMEGA',matrix([{**t,'omega':o} for t in top for o in [.05,.1,.2,.3]],[.002])); report(D/'round_reports'/'ROUND2_OMEGA_SEARCH.md','Round 2 omega',rank(r2)); top=[trial(x) for x in rank(r2) if x['valid_lbi_run']][:2]
 r2b=phase('ROUND2B_STAGE2_LR',matrix([{**t,'stage2_lr':lr} for t in top for lr in [.005,.01,.02]],[.002])); report(D/'round_reports'/'ROUND2B_STAGE2_LR_SEARCH.md','Round 2B Stage 2 LR',rank(r2b)); top=[trial(x) for x in rank(r2b) if x['valid_lbi_run']][:2]
 r3=phase('ROUND3_MULTIBUDGET',matrix(top,[.001,.002,.003])); report(D/'round_reports'/'ROUND3_MULTIBUDGET.md','Round 3 multi-budget',rank(r3))
 groups={tuple((x[k] for k in ('alpha','kappa','nu','omega','stage2_lr'))):[] for x in r3}
 for x in r3: groups[tuple(x[k] for k in ('alpha','kappa','nu','omega','stage2_lr'))].append(x)
 best=max(groups.values(),key=lambda z:(sum(x['valid_lbi_run'] for x in z),sum(x['FO-Acc'] for x in z if x['valid_lbi_run'])/max(1,sum(x['valid_lbi_run'] for x in z))))
 final=trial(best[0]); r4=phase('ROUND4_SMALL_BUDGET',matrix([final],[.0005])); curve=r3+r4; at3=[x for x in r3 if x['requested_budget']==.003 and trial(x)==final]
 if at3 and at3[0]['valid_lbi_run'] and at3[0].get('stage1_steps_completed_max',9999)<=2850 and (at3[0].get('FO_margin') or -1)>0:
  x4=phase('ROUND4_BUDGET_004',matrix([final],[.004])); curve+=x4
  if x4 and x4[0]['valid_lbi_run'] and x4[0].get('stage1_steps_completed_max',9999)<=2850 and (x4[0].get('FO_margin') or -1)>0: curve+=phase('ROUND4_BUDGET_005',matrix([final],[.005]))
 report(D/'round_reports'/'ROUND4_FINAL_BUDGET_CURVE.md','Round 4 final budget curve',rank(curve))
 budgets=sorted({x['requested_budget'] for x in curve}); variants={v:{'budgets':budgets, **({'selection_seed_mode':'same_as_run_seed'} if v=='module_random' else {})} for v in ['module_random','module_magnitude','module_saliency']}; br=phase('FINAL_SPARSE_BASELINES',matrix([],[],variants)); _=br
 curve=[enrich(x['summary_path'],load_summary(x['summary_path'])) for x in curve]; report(D/'final'/'FINAL_REPORT.md','Final report',rank(curve),'This is the best hyperparameter-search result on Office Amazon→DSLR, seed 2020, batch size 64; it is not a multi-seed, multi-transfer paper result.')
 write(D/'final'/'final_results.json',{'search_id':sid,'best_setting':final,'results':curve});
 with (D/'final'/'final_results.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,sorted({k for x in curve for k in x}));w.writeheader();w.writerows(curve)
 write(D/'final'/'REPRODUCE_COMMANDS.sh',f'#!/usr/bin/env bash\nset -euo pipefail\nconda activate SHOT_TTA\npython iclr2027/tools/shot_otta_lbi_search_driver.py --search-id {sid}\n'); os.chmod(D/'final'/'REPRODUCE_COMMANDS.sh',0o755)
 arts=[]
 for p in D.rglob('*'):
  if p.is_file(): arts.append({'path':str(p.relative_to(ROOT)),'type':'file','size':p.stat().st_size,'sha256':sha(p),'created_at':utc(),'related_round':next((q for q in ['ROUND1','ROUND2','ROUND3','ROUND4','FINAL'] if q in p.name.upper()),'general')})
 write(D/'final'/'ARTIFACT_MANIFEST.json',arts); write(D/'05_CURRENT_STATUS.md',f'Search ID: {sid}\nCurrent Round: complete\ncompleted\nresume: tmux attach -t shot_lbi_search_{sid}\n')
if __name__=='__main__': main()
