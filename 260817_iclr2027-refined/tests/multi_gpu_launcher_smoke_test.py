#!/usr/bin/env python3
"""CPU/fake-process coverage for the multi-GPU launcher."""
import json, sys, tempfile, threading, time
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJECT))
from tools.run_experiments_multi_gpu import execute
def exp(i):
 key=f'fake_{i}'; return {'experiment_key':key,'experiment_config_sha256':(str(i)*64)[:64],'method':'shot','task':'otta','dataset':'office','source':0,'target':1,'seed':2020,'variant':'module_lbi','requested_budget':.002,'command_args':['fake',key,(str(i)*64)[:64],'--gpu-id','0']}
def main():
 with tempfile.TemporaryDirectory() as d:
  root=Path(d); runs=root/'runs'; active={}; maximum={}; seen=[]; lock=threading.Lock()
  xs=[exp(i) for i in range(17)]; done=xs[0]; rd=runs/'done'; rd.mkdir(parents=True); (rd/'summary.json').write_text(json.dumps({'status':'completed','experiment_key':done['experiment_key'],'experiment_config_sha256':done['experiment_config_sha256']}))
  def fake(cmd, cwd, out, err):
   key,gpu=cmd[1],cmd[-1]
   with lock: active[gpu]=active.get(gpu,0)+1; maximum[gpu]=max(maximum.get(gpu,0),active[gpu]); seen.append((key,gpu))
   time.sleep(.04)
   q=runs/key; q.mkdir(parents=True); (q/'summary.json').write_text(json.dumps({'status':'completed','experiment_key':key,'experiment_config_sha256':cmd[2]})); Path(out).write_text('ok'); Path(err).write_text('')
   with lock: active[gpu]-=1
   return 0
  for x in xs: x['expected_output_root']=str(runs/x['experiment_key'])
  plan={'experiments':xs}; r=execute(plan,runs,root/'logs',['0','1','2','3','4','5','6','7'],root,process_executor=fake,workers_per_gpu=2)
  assert len(seen)==16 and set(g for _,g in seen)=={'0','1','2','3','4','5','6','7'} and all(v<=2 for v in maximum.values())
  assert {x['assigned_gpu'] for x in r['records'] if x['status']=='completed'}=={'0','1','2','3','4','5','6','7'}
  assert all('gpu_slot_index' in x for x in r['records'] if x['status']=='completed')
  assert all('actual_command' in x and 'stdout_log' in x and 'stderr_log' in x for x in r['records'] if x['status']=='completed')
  assert any(x['status']=='skipped_completed' for x in r['records'])
  execute(plan,runs,root/'logs2',['7','6','5','4','3','2','1','0'],root,process_executor=lambda *_:(_ for _ in ()).throw(AssertionError('resume reran completed')))
 print('multi GPU launcher smoke test passed')
if __name__=='__main__': main()
