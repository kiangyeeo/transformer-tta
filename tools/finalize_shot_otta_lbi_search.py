#!/usr/bin/env python3
"""Normalize final deliverables for a completed phased LBI search."""
import argparse, csv, hashlib, json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x): Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def main():
 a=argparse.ArgumentParser();a.add_argument('search_id');x=a.parse_args(); d=ROOT/'iclr2027/experiment_logs'/x.search_id; final=d/'final'; prior=read(final/'final_results.json'); setting=prior['best_setting']; rows=[]
 for row in prior['results']:
  if all(row.get(k)==v for k,v in setting.items() if k!='stage2_steps') and row.get('stage2_steps_requested')==setting['stage2_steps']: rows.append(row)
 rows.sort(key=lambda r:r['requested_budget'])
 fields=['protocol_revision','implementation_revision','source_checkpoint_revision','efficiency_protocol_revision','requested_budget','max_support_count','target_support_count','exact_budget_reached_all_steps','budget_reached_all_steps','budget_hit_count','online_steps','valid_lbi_run','max_steps_hit_count','stage1_support_count_min','stage1_support_count_max','stage1_steps_completed_min','stage1_steps_completed_mean','stage1_steps_completed_max','PU-Acc','FO-Acc','best_sparse_variant','best_sparse_exact_selected_count','best_sparse_FO','FO_margin','runtime','online_batch_runtime_mean_sec','online_batch_runtime_std_sec','online_batch_runtime_median_sec','online_batch_runtime_p95_sec','online_compute_runtime_sec','adapt_batch_runtime_mean_sec','adapt_batch_runtime_std_sec','adapt_runtime_total_sec','pu_batch_runtime_mean_sec','pu_batch_runtime_std_sec','pu_runtime_total_sec','gpu_peak_allocated_mean_mb','gpu_peak_allocated_max_mb','gpu_peak_reserved_mean_mb','gpu_peak_reserved_max_mb','fo_eval_runtime_sec','wall_runtime_sec','peak_gpu_memory_allocated_mb','peak_gpu_memory_reserved_mb','gpu_name','runtime_comparable','experiment_key','summary_path']
 lines=['# FINAL REPORT','', '这是 Office Amazon→DSLR、seed 2026、batch size 64 上用于超参数搜索的最佳结果，不是多 seed、多 transfer 的论文最终结果。','', '## A. 最终最佳 setting','', *[f'- {k}: {v}' for k,v in setting.items()],'- batch_size: 64','- seed: 2026','- transfer: Amazon→DSLR (source=0, target=1)','- candidate scope: netB.bottleneck','- BN policy: frozen','- initialization: dense','- stage3 mode: accumulation','- state lifecycle: reset_every_online_step','', '## B. 最终配置的 budget curve','', '|'+ '|'.join(fields)+'|','|'+ '|'.join(['---']*len(fields))+'|']
 for r in rows: lines.append('|'+ '|'.join(str(r.get(k,'')) for k in fields)+'|')
 lines += ['', '## C. 搜索历史','', '- Round 1、Round 2、Round 2B、Round 3 和 Round 4 的候选、诊断与筛选见 `round_reports/`。','- Round 4 在 budget 0.003 无效（2 个 max-step hit；budget hit rate 0.75）后按协议停止，未运行 0.004/0.005。','', '## D. 资源使用','', '- 启动记录（GPU、开始/结束时间、运行时长、返回码）见 `launcher_logs/*/launcher_manifest.json`。','- 本次搜索无训练失败或重试。','', '## E. 路径索引','', '- matrices/: 原始矩阵；plans/: 计划；launcher_logs/: 实际命令与 stdout/stderr；status/: 状态；results/: summaries/diagnostics；round_reports/: 决策；02_COMMAND_HISTORY.sh：命令历史。','']
 (final/'FINAL_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
 write(final/'final_results.json',{'search_id':x.search_id,'best_setting':setting,'results':rows})
 with (final/'final_results.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fields);w.writeheader();w.writerows([{k:r.get(k) for k in fields} for r in rows])
 result_by_key={row.get('experiment_key'): row for row in prior['results']}
 manifest=[]
 for plan_path in d.glob('plans/*/plan.json'):
  plan=read(plan_path); records={}
  lp=d/'launcher_logs'/plan_path.parent.name/'launcher_manifest.json'
  if lp.exists():
   for r in read(lp).get('records',[]): records.setdefault(r.get('experiment_key'),[]).append(r)
  for e in plan['experiments']:
   rec=(records.get(e['experiment_key']) or [{}])[-1]
   result=result_by_key.get(e['experiment_key'],{})
   manifest.append({'protocol_revision':e.get('protocol_revision',result.get('protocol_revision')),'implementation_revision':e.get('implementation_revision',result.get('implementation_revision')),'source_checkpoint_revision':e.get('source_checkpoint_revision',result.get('source_checkpoint_revision')),'efficiency_protocol_revision':e.get('efficiency_protocol_revision',result.get('efficiency_protocol_revision')),'round':plan_path.parent.name,'trial_index':e.get('lbi_trial_index'),'experiment_key':e['experiment_key'],'experiment_config_sha256':e['experiment_config_sha256'],'dataset':e['dataset'],'source':e['source'],'target':e['target'],'seed':e['seed'],'requested_budget':e['requested_budget'],'max_support_count':result.get('max_support_count'),'target_support_count':result.get('target_support_count',result.get('max_support_count')),'alpha':e.get('alpha'),'kappa':e.get('kappa'),'nu':e.get('nu'),'omega':e.get('omega'),'stage1_max_steps':e.get('stage1_max_steps'),'stage2_lr':e.get('stage2_lr'),'stage2_steps':e.get('stage2_steps_requested'),'batch_size':e['effective_overrides'].get('batch_size',e['effective_overrides'].get('data',{}).get('batch_size')),'assigned_gpu':rec.get('assigned_gpu'),'actual_command':rec.get('actual_command',e['command_args']),'started_at_utc':rec.get('started_at_utc'),'completed_at_utc':rec.get('completed_at_utc'),'return_code':rec.get('return_code'),'run_output_dir':rec.get('run_output_dir'),'stdout_log':rec.get('stdout_log'),'stderr_log':rec.get('stderr_log'),'summary_path':rec.get('summary_path'),'budget_diagnostics_path':str(d/'results'/plan_path.parent.name/'lbi_budget_diagnostics.csv'),'status':rec.get('status','skipped_completed')})
 write(d/'03_EXPERIMENT_MANIFEST.json',manifest)
 artifacts=[]
 for p in d.rglob('*'):
  if p.is_file() and p.name!='ARTIFACT_MANIFEST.json': artifacts.append({'path':str(p.relative_to(ROOT)),'type':'file','size':p.stat().st_size,'sha256':sha(p),'created_at':datetime.now(timezone.utc).isoformat(),'related_round':next((z for z in ('ROUND1','ROUND2','ROUND3','ROUND4','FINAL') if z in p.as_posix().upper()),'general')})
 write(final/'ARTIFACT_MANIFEST.json',artifacts)
if __name__=='__main__': main()
