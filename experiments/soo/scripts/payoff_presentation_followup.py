"""Freeze and analyze the bounded adapter/offset presentation follow-up."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import subprocess

from analyze_explicit_payoff import digest, read, write_json
from analyze_payoff_followups import diagnostic_analysis
from review_payoff_behavior import counts, write_csv


def compare_cells(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['condition'], row['column_swap'], row['name_swap'], row['query'], row['alignment']].append(row)
    cells = [{**dict(zip(('condition','column_swap','name_swap','query','alignment'), key)), **counts(group)}
             for key, group in groups.items()]
    contrasts = []
    for column in (0, 1):
        for name in (0, 1):
            subset = [r for r in rows if r['column_swap']==column and r['name_swap']==name]
            def rate(condition, query, metric, alignment=None):
                selected = [r for r in subset if r['condition']==condition and r['query']==query and
                            (alignment is None or r['alignment']==alignment)]
                return sum(r[metric] for r in selected)/len(selected)
            contrasts.append({'column_swap': column, 'name_swap': name,
                'adapter_minus_offset_private': rate('adapter','private','correct')-rate('offset','private','correct'),
                'adapter_minus_base_opposed_recipient': rate('adapter','message','recipient_optimal','opposed')-rate('base','message','recipient_optimal','opposed')})
    return cells, contrasts


def analyze(batch_path):
    batch = json.loads(batch_path.read_text())
    out = Path(batch['analysis'])
    out.mkdir(parents=True, exist_ok=False)
    try:
        assert digest(Path(batch['source_ledger'])) == batch['source_ledger_sha256']
        for name, expected in batch['controller_sha256'].items():
            assert digest(Path(batch['controller'])/name)==expected
        summaries, rows = {}, []
        base_data = base_inputs = base_manifest = None
        for condition, entry in batch['runs'].items():
            run = Path(entry['path'])
            summaries[condition] = diagnostic_analysis(run,Path(entry['historical']),out/condition,condition)
            data = read(run/'data/goals_roles_pilot/main.jsonl')+read(run/'data/goals_roles_pilot/checks.jsonl')
            inputs = read(run/'output/inputs.jsonl')
            manifest = json.loads((run/'output/manifest.json').read_text())
            if condition=='base':
                base_data,base_inputs,base_manifest=data,inputs,manifest
            assert data==base_data and inputs==base_inputs
            for field in ('batches','model','revision','dtype','attention','chat_kwargs','batch_size','max_new_tokens','do_sample','source_sha256','versions'):
                assert manifest[field]==base_manifest[field], (condition,field)
            for e,s in zip(data,read(run/'output/scored.jsonl')):
                rows.append({'condition':condition,'column_swap':e['column_swap'],'name_swap':e['name_swap'],
                             'canonical_example_id':e['canonical_example_id'],**s})
        cells, contrasts = compare_cells(rows)
        write_csv(out/'all_trials.csv',rows)
        write_csv(out/'cells.csv',cells)
        write_csv(out/'contrasts.csv',contrasts)
        passed = all(s['sentinel_repeat_passed'] and s['regressions_passed'] and
                     s['historical_answer_changes']==s['historical_token_changes']==0 for s in summaries.values())
        result = {'protocol':batch['protocol'],'conditions':summaries,'contrasts':contrasts,
                  'all_checks_passed':passed,'interpretation':'exploratory development presentation robustness; no mechanism or generalization claim'}
        write_json(out/'analysis.json',result)
        batch['status']='completed_and_analyzed' if passed else 'discrepancy_requires_inspection'
        batch['result']=result
    except Exception as error:
        batch['status']='analysis_failed_requires_inspection'
        batch['error']=repr(error)
        raise
    finally:
        batch['updated_utc']=datetime.now(timezone.utc).isoformat()
        write_json(batch_path,batch)


def stage(source_path):
    from launch_goals_roles import ROOT,SOO,freeze
    source=json.loads(source_path.read_text())
    assert source['status']=='finished'
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root=SOO/'results/study2'
    controller=root/f'payoff-presentation2-controller-{stamp}'
    controller.mkdir()
    for name in ('payoff_presentation_followup.py','analyze_payoff_followups.py','analyze_explicit_payoff.py','review_payoff_behavior.py'):
        shutil.copy2(SOO/'scripts'/name,controller/name)
    for name in ('PAYOFF_PRESENTATION.md','PLAN.md','FINDINGS.md'):
        shutil.copy2(SOO/name,controller/name)
    picker=subprocess.check_output(['/usr/bin/python3.11',str(ROOT/'gpu_pick.py'),'--hours','0.5','--vram-gb','75','--cpus','4','--mem-gb','96','--no-requeue-ok','--fast','--cuda','13.0','--top','10'],text=True)
    (controller/'gpu_pick.txt').write_text(picker)
    historical={r['tag']:r['path'] for r in source['runs']['development']}
    runs={'base':{'path':source['diagnostic']['path'],'historical':historical['base'],'status':'reused_completed'}}
    for condition in ('adapter','offset'):
        run=freeze(condition,data_dir=Path(source['diagnostic']['path'])/'data/goals_roles_pilot',
                   diagnostic_followup='payoff_presentation2')
        runs[condition]={'path':str(run),'historical':historical[condition],'status':'staged_not_submitted','script_sha256':digest(run/'run.sh')}
    batch_path=root/f'payoff-presentation2-batch-{stamp}.json'
    batch={'protocol':'payoff_presentation2','status':'staged_not_submitted',
           'authorization':'User asked to keep investigating the held-out behavioral difference; bounded presentation follow-up specified before outputs.',
           'source_ledger':str(source_path.resolve()),'source_ledger_sha256':digest(source_path),
           'created_utc':datetime.now(timezone.utc).isoformat(),'controller':str(controller),
           'runs':runs,'analysis':str(root/f'payoff-presentation2-analysis-{stamp}'),
           'maximum_new_scored_generations':3456}
    script=controller/'analyze.sh'
    command=['/usr/bin/python3.11','-u',str(controller/'payoff_presentation_followup.py'),'--analyze',str(batch_path)]
    script.write_text(f'''#!/bin/bash --login
#SBATCH --job-name=soo-paypresentation-audit
#SBATCH --partition=dw
#SBATCH --qos=dw87
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output={SOO}/slurm-logs/%x-%j.out
set -euo pipefail
export PYTHONPATH={shlex.quote(str(Path(runs['adapter']['path'])/'src'))}
{shlex.join(command)}
''')
    subprocess.run(['bash','-n',str(script)],check=True)
    batch['controller_sha256']={p.name:digest(p) for p in controller.iterdir() if p.is_file()}
    write_json(batch_path,batch)
    print(batch_path,flush=True)
    # Persist each successful submission immediately; no automatic resubmission path.
    jobs=[]
    for condition in ('adapter','offset'):
        entry=runs[condition]
        job=subprocess.check_output(['sbatch','--parsable',str(Path(entry['path'])/'run.sh')],text=True).strip().split(';')[0]
        assert job.isdigit()
        entry.update(job_id=job,status='submitted',submitted_utc=datetime.now(timezone.utc).isoformat())
        jobs.append(job)
        write_json(Path(entry['path'])/'submission.json',entry)
        write_json(batch_path,batch)
        print(condition,job,flush=True)
    audit_job=subprocess.check_output(['sbatch','--parsable','--dependency=afterany:'+':'.join(jobs),str(script)],text=True).strip().split(';')[0]
    batch.update(status='submitted',analysis_job_id=audit_job,updated_utc=datetime.now(timezone.utc).isoformat())
    write_json(batch_path,batch)
    print('analysis',audit_job,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--stage',type=Path)
    choice.add_argument('--analyze',type=Path)
    args=parser.parse_args()
    stage(args.stage) if args.stage else analyze(args.analyze)
