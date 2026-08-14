#!/usr/bin/env python3
import csv, json, sys, tempfile
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJECT))
from tools.analyze_lbi_budget_search import TRANSFERS, analyze
def experiment(name,s,t,alpha): return {"experiment_key":f"{name}-{s}{t}","variant":"module_lbi","source":s,"target":t,"seed":2020,"requested_budget":.001,"alpha":alpha,"kappa":1,"nu":.5,"omega":.2,"stage1_max_steps":3000,"budget_tolerance":.0001,"stage2_lr":.02,"stage2_steps_requested":1}
def summary(exp,valid,fo): return {"status":"completed","experiment_key":exp["experiment_key"],"valid_lbi_run":valid,"FO-Acc":fo,"budget_reached_all_steps":valid,"max_steps_hit_count":0 if valid else 1,"stage1_steps_mean":10,"runtime":1}
def main():
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp); runs=root/'runs'; xs=[experiment('high_fo_invalid',s,t,.1) for s,t in TRANSFERS]+[experiment('valid',s,t,.2) for s,t in TRANSFERS]; (root/'plan.json').write_text(json.dumps({'experiments':xs}))
  with (root/'baseline.csv').open('w',newline='') as h:
   w=csv.DictWriter(h,fieldnames=['source','target','seed','requested_budget','best_sparse_FO']); w.writeheader(); [w.writerow({'source':s,'target':t,'seed':2020,'requested_budget':.001,'best_sparse_FO':80}) for s,t in TRANSFERS]
  for exp in xs:
   p=runs/exp['experiment_key']; p.mkdir(parents=True); p.joinpath('summary.json').write_text(json.dumps(summary(exp,exp['alpha']==.2,81 if exp['alpha']==.2 else 99)))
  rows=analyze(root/'plan.json',runs,root/'baseline.csv',.001,2020,'synthetic',root/'out',2); assert rows[0]['alpha']==.2 and rows[0]['mean_FO_margin_all']==1; assert rows[1]['alpha']==.1 and rows[1]['valid_lbi_run_count']==0; assert len(list(csv.DictReader((root/'out'/'per_run.csv').open())))==12; assert len(json.loads((root/'out'/'selection.json').read_text())['selected'])==2
  broken=xs[:-1]; (root/'broken.json').write_text(json.dumps({'experiments':broken}))
  try: analyze(root/'broken.json',runs,root/'baseline.csv',.001,2020,'broken',root/'broken_out',1); raise AssertionError('missing transfer did not fail')
  except ValueError as exc: assert 'six directed transfers' in str(exc)
  duplicate=list(xs); duplicate[-1]=dict(duplicate[-1],experiment_key=duplicate[-2]['experiment_key']); (root/'duplicate.json').write_text(json.dumps({'experiments':duplicate}))
  try: analyze(root/'duplicate.json',runs,root/'baseline.csv',.001,2020,'duplicate',root/'duplicate_out',1); raise AssertionError('duplicate identity did not fail')
  except ValueError as exc: assert 'duplicate or missing planned run identity' in str(exc)
 print('budget search analysis tests passed')
if __name__=='__main__': main()
