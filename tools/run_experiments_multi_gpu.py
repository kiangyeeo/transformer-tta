#!/usr/bin/env python3
"""Status-aware, recoverable one-process-per-GPU plan executor."""
import argparse
import csv
import json
import shlex
import signal
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
from tools.check_experiment_status import check_status, load_plan
from tools.summarize_runs import build_summary_outputs, write_summary_outputs

_STOP = threading.Event()
_PROCESSES = set()
_PROCESS_LOCK = threading.Lock()
_ACTIVE_ATTEMPTS = {}
_ATTEMPT_LOCK = threading.Lock()

def now(): return datetime.now(timezone.utc).isoformat()
def safe(key): return ''.join(c if c.isalnum() or c in '._-' else '_' for c in key)
def replace_gpu(args, gpu):
    result = list(args)
    try: result[result.index('--gpu-id') + 1] = str(gpu)
    except (ValueError, IndexError): raise ValueError('command lacks --gpu-id VALUE')
    return result
def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)
def append_history(path, command, cwd):
    if not path: return
    with threading.Lock():
        with Path(path).open('a', encoding='utf-8') as f:
            f.write('cd ' + shlex.quote(str(cwd)) + '\n' + shlex.join(command) + '\n')
def find_stream_resume_dir(runs_root, experiment):
    """Return the furthest incomplete stream checkpoint for one identity."""
    candidates = []
    for metadata_path in Path(runs_root).rglob('stream_state.json'):
        try:
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        if metadata.get('status') != 'in_progress':
            continue
        if any(metadata.get(field) != experiment.get(field) for field in ('experiment_key', 'experiment_config_sha256')):
            continue
        state_path = metadata_path.with_name('stream_state.pt')
        if not state_path.is_file():
            continue
        candidates.append((int(metadata.get('loader_batches_processed', -1)), metadata_path.stat().st_mtime, str(metadata_path.parent.parent.resolve())))
    return max(candidates)[2] if candidates else None
def command_for_experiment(experiment, gpu, resume_run_dir=None, enable_stream_checkpoint=False):
    command = replace_gpu(experiment['command_args'], gpu)
    if enable_stream_checkpoint:
        command.append('--enable-stream-checkpoint')
    if resume_run_dir:
        command.extend(['--resume-run-dir', resume_run_dir])
    return command
def status_map(plan, runs_root):
    value = check_status(plan, str(runs_root))
    return {x['experiment_key']: x for x in value['experiments']}, value


def validate_visda_batch_size(plan):
    entries = [
        experiment
        for experiment in plan.get('experiments', [])
        if experiment.get('dataset') == 'VISDA-C'
    ]
    if not entries:
        return
    frozen_path = PROJECT_DIR / 'experiment_logs' / 'CURRENT_VISDA_BATCH_SIZE.txt'
    try:
        frozen = int(frozen_path.read_text(encoding='utf-8').strip())
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(
            f'VISDA-C launcher requires a valid frozen batch size at {frozen_path}'
        ) from error
    observed = {
        int(experiment['scientific_config']['data']['batch_size'])
        for experiment in entries
    }
    if observed != {frozen}:
        raise RuntimeError(
            'VISDA-C plan batch size does not match the active frozen batch '
            f'size: plan={sorted(observed)}, frozen={frozen}'
        )


def default_executor(command, cwd, stdout, stderr, attempt_state=None):
    with Path(stdout).open('w', encoding='utf-8') as out, Path(stderr).open('w', encoding='utf-8') as err:
        p = subprocess.Popen(command, cwd=str(cwd), stdout=out, stderr=err, text=True)
        if attempt_state is not None:
            attempt_state['pid'] = p.pid
            marker = attempt_state['directory'] / 'attempt_started.json'
            started = json.loads(marker.read_text(encoding='utf-8'))
            started['pid'] = p.pid
            write_json(marker, started)
        with _PROCESS_LOCK: _PROCESSES.add(p)
        try: return p.wait()
        finally:
            with _PROCESS_LOCK: _PROCESSES.discard(p)
def signal_handler(signum, _frame):
    _STOP.set()
    with _ATTEMPT_LOCK:
        active_attempts = list(_ACTIVE_ATTEMPTS.values())
    for attempt in active_attempts:
        write_json(attempt['directory'] / 'launcher_signal.json', {
            'signal': int(signum), 'received_at_utc': now(),
            'experiment_key': attempt['experiment_key'], 'pid': attempt.get('pid'),
        })
    with _PROCESS_LOCK:
        for p in list(_PROCESSES):
            if p.poll() is None: p.terminate()
def execute(plan, runs_root, logs_root, gpus, workdir, dry_run=False, summarize_after_run=False, command_history=None, process_executor=None, workers_per_gpu=1, max_workers=None, resume_partial_runs=False):
    """Run plan; process_executor is intentionally injectable for CPU smoke tests."""
    if workers_per_gpu < 1: raise ValueError('workers_per_gpu must be positive')
    validate_visda_batch_size(plan)
    process_executor = process_executor or default_executor
    slots=[(gpu, index) for gpu in gpus for index in range(workers_per_gpu)]; slots=slots[:min(len(slots), max_workers if max_workers is not None else len(slots))]
    logs_root = Path(logs_root); logs_root.mkdir(parents=True, exist_ok=True)
    runs_root, workdir = Path(runs_root).resolve(), Path(workdir).resolve()
    smap, status = status_map(plan, runs_root)
    blocking = [x for x in status['experiments'] if x['plan_status'] not in ('missing','completed')]
    if status['unassociated_invalid_summaries'] or blocking:
        raise RuntimeError('preflight status contains invalid, duplicate, or hash-mismatched summaries')
    todo = [e for e in plan['experiments'] if smap[e['experiment_key']]['plan_status'] == 'missing']
    if dry_run:
        report={'gpus':gpus,'workers_per_gpu':workers_per_gpu,'slot_count':len(slots),'completed':[e['experiment_key'] for e in plan['experiments'] if e not in todo], 'will_execute':[{'experiment_key':e['experiment_key'],'assigned_gpu':slots[i%len(slots)][0],'gpu_slot_index':slots[i%len(slots)][1],'resume_run_dir':find_stream_resume_dir(runs_root,e) if resume_partial_runs else None,'actual_command':command_for_experiment(e,slots[i%len(slots)][0],find_stream_resume_dir(runs_root,e) if resume_partial_runs else None,enable_stream_checkpoint=resume_partial_runs)} for i,e in enumerate(todo)]}
        print(json.dumps(report, indent=2)); return {'records':[], 'dry_run':True, 'report':report}
    records=[]; records_lock=threading.Lock(); halt=threading.Event()
    def worker(slot):
        gpu,gpu_slot_index=slots[slot]
        while not _STOP.is_set() and not halt.is_set():
            with queue_lock:
                if not queue: return
                exp=queue.pop(0)
            resume_run_dir = find_stream_resume_dir(runs_root, exp) if resume_partial_runs else None
            command=command_for_experiment(exp,gpu,resume_run_dir,enable_stream_checkpoint=resume_partial_runs); attempt=0
            while attempt < 2 and not _STOP.is_set():
                attempt += 1; started=now(); timer=time.perf_counter()
                adir=logs_root/safe(exp['experiment_key'])/(datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')+f'_gpu{gpu}_try{attempt}')
                adir.mkdir(parents=True); out,err=adir/'stdout.log',adir/'stderr.log'
                append_history(command_history, command, workdir)
                attempt_state={'directory':adir,'experiment_key':exp['experiment_key'],'pid':None}
                write_json(adir/'attempt_started.json', {'experiment_key':exp['experiment_key'],'experiment_config_sha256':exp['experiment_config_sha256'],'assigned_gpu':gpu,'gpu_slot_index':gpu_slot_index,'started_at_utc':started,'resume_run_dir':resume_run_dir,'actual_command':command,'command_display':shlex.join(command),'stdout_log':str(out.resolve()),'stderr_log':str(err.resolve())})
                with _ATTEMPT_LOCK: _ACTIVE_ATTEMPTS[str(adir)] = attempt_state
                executor_exception = None
                try:
                    if process_executor is default_executor:
                        rc=process_executor(command, workdir, out, err, attempt_state)
                    else:
                        rc=process_executor(command, workdir, out, err)
                except BaseException as error:
                    rc=-1; executor_exception={'type':type(error).__name__,'message':str(error)}
                finally:
                    with _ATTEMPT_LOCK: _ACTIVE_ATTEMPTS.pop(str(adir),None)
                post,_=status_map(plan,runs_root)
                row={'experiment_key':exp['experiment_key'],'experiment_config_sha256':exp['experiment_config_sha256'],'assigned_gpu':gpu,'gpu_slot_index':gpu_slot_index,'pid':attempt_state['pid'],'actual_command':command,'command_display':shlex.join(command),'attempt':attempt,'started_at_utc':started,'completed_at_utc':now(),'runtime_seconds':time.perf_counter()-timer,'return_code':int(rc),'termination_signal':(-int(rc) if int(rc)<0 else None),'executor_exception':executor_exception,'resume_run_dir':resume_run_dir,'stdout_log':str(out.resolve()),'stderr_log':str(err.resolve()),'run_output_dir':None,'summary_path':None,'pre_run_status':'missing','post_run_status':post[exp['experiment_key']]['plan_status']}
                paths=post[exp['experiment_key']].get('matching_summary_paths') or []
                if paths: row['summary_path']=paths[0]; row['run_output_dir']=str(Path(paths[0]).parent)
                row['status']='completed' if rc == 0 and row['post_run_status']=='completed' else ('interrupted' if _STOP.is_set() else 'failed')
                write_json(adir/'launch_record.json',row)
                with records_lock: records.append(row)
                if row['status']=='completed': break
            if row['status']!='completed': halt.set(); return
    queue=list(todo); queue_lock=threading.Lock()
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futures=[pool.submit(worker,i) for i in range(len(slots))]
        for future in as_completed(futures): future.result()
    smap,_=status_map(plan,runs_root)
    for e in plan['experiments']:
        if smap[e['experiment_key']]['plan_status']=='completed' and not any(r['experiment_key']==e['experiment_key'] for r in records):
            records.append({'experiment_key':e['experiment_key'],'experiment_config_sha256':e['experiment_config_sha256'],'status':'skipped_completed','assigned_gpu':None})
    payload={'launcher_schema_version':1,'created_at_utc':now(),'interrupted':_STOP.is_set(),'halted_after_failure':halt.is_set(),'records':records}
    write_json(logs_root/'launcher_manifest.json',payload)
    fields=sorted({k for r in records for k in r})
    with (logs_root/'launcher_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fields); w.writeheader(); w.writerows(records)
    if summarize_after_run and not halt.is_set() and not _STOP.is_set(): write_summary_outputs(build_summary_outputs(str(runs_root),plan=plan),logs_root/'summary')
    return payload
def main():
    p=argparse.ArgumentParser(); p.add_argument('plan'); p.add_argument('--runs-root',required=True); p.add_argument('--logs-root',required=True); p.add_argument('--workdir',default='.'); p.add_argument('--gpus',default='0,1,2,3'); p.add_argument('--max-workers',type=int,default=4); p.add_argument('--workers-per-gpu',type=int,default=1); p.add_argument('--dry-run',action='store_true'); p.add_argument('--resume',action='store_true'); p.add_argument('--resume-partial-runs',action='store_true',help='Resume matching in-progress stream checkpoints for missing identities.'); p.add_argument('--summarize-after-run',action='store_true'); p.add_argument('--command-history')
    a=p.parse_args(); gpus=[x.strip() for x in a.gpus.split(',') if x.strip()]
    if not gpus or a.max_workers < 1 or a.workers_per_gpu < 1: p.error('non-empty --gpus and positive --max-workers/--workers-per-gpu required')
    signal.signal(signal.SIGINT,signal_handler); signal.signal(signal.SIGTERM,signal_handler)
    slots=min(a.max_workers,len(gpus)*a.workers_per_gpu)
    try:
        result=execute(load_plan(a.plan),a.runs_root,a.logs_root,gpus,a.workdir,a.dry_run,a.summarize_after_run,a.command_history,workers_per_gpu=a.workers_per_gpu,max_workers=slots,resume_partial_runs=a.resume_partial_runs)
    except BaseException as error:
        write_json(Path(a.logs_root) / 'launcher_exception.json', {'created_at_utc':now(),'exception_type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
        raise
    if result.get('halted_after_failure') or result.get('interrupted'): raise SystemExit(1)
if __name__=='__main__': main()
