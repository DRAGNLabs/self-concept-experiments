"""Final three-strength cross-task refinement; no further automatic experiments."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import shutil
import subprocess

from analyze_explicit_payoff import audit_run,digest,read,verify_run,write_json
from continue_explicit_payoff import submit
from cross_task_dose import history_audit,joint_criterion,payoff_table
from payoff_preferences_followup import match_inputs,feasibility
from payoff_half_transfer import paired_counts
from review_payoff_behavior import write_csv

DOSES=(.5625,.625,.6875)
NEW_TAGS=tuple(f'offset_{a:g}' for a in DOSES)
TAGS=('base','adapter','offset_0.5',*NEW_TAGS,'offset_0.75','offset_1')


def verify_condition(result,path,tag,b):
    m=result[4]
    if tag.startswith('offset_'):
        assert m['condition']=='offset' and m['offset_alpha']==float(tag.split('_')[1]),'wrong dose'
        assert digest(path/'band_deltas.pt')==b['offset_sha256'],'wrong offset artifact'
    else:assert m['condition']==tag
    assert result[0]['repeat_passed'],'payoff repeat failed'
    if tag in NEW_TAGS:assert m['source_sha256']==b['inference_source_sha256']


def same_dose_history(path,history_path):
    records=read(path/'output/regressions.jsonl')
    assert len(records)==32
    assert records==read(history_path/'output/regressions.jsonl'),'same-dose history differs across tasks'


def payoff_analysis(suite,entries,b,out):
    out.mkdir(parents=True,exist_ok=False);n=1296 if suite=='validation' else 2592
    runs={};errors={}
    for tag,path in entries.items():
        try:
            result=audit_run(Path(path),out/'audits'/tag,expected_n=n)
            verify_condition(result,Path(path),tag,b)
            if tag in NEW_TAGS:same_dose_history(Path(path),Path(b['history_paths'][tag]))
            runs[tag]=result
        except Exception as exc:errors[tag]=repr(exc)
    if 'base' not in runs or 'adapter' not in runs or 'offset_0.5' not in runs:
        return {'checks_passed':False,'errors':errors,'table':{}},runs
    # New/current source is first; explicitly allow only the previously bridged old-source transition.
    reference=runs['offset_0.5']
    for tag in list(runs):
        try:match_inputs(reference,runs[tag],bridge=(reference[4]['source_sha256']!=runs[tag][4]['source_sha256']))
        except Exception as exc:errors[tag]=repr(exc);del runs[tag]
    if errors:return {'checks_passed':False,'errors':errors,'table':{tag:payoff_table(r[3]) for tag,r in runs.items()}},runs
    if suite=='numerical':assert feasibility(runs['base'][0],runs['base'][1],runs['base'][3])['passed']
    else:assert all(c['passed'] for c in runs['base'][0]['gate']['checks'] if c['query']!='message')
    rows=[];contexts=defaultdict(dict)
    for tag,r in runs.items():
        for e,s in zip(r[1],r[3]):
            row=dict(s,condition=tag,profile=e.get('profile','none'));rows.append(row)
            contexts[tag,s['context_id']][s['query']]=s
    write_csv(out/'all_trials.csv',rows)
    for fields in (('condition','query','instance_id','role','alignment'),('condition','query','table_order','answer_order'),('condition','query','profile','role','alignment')):
        groups=defaultdict(list)
        for r in rows:groups[tuple(r[k] for k in fields)].append(r)
        write_csv(out/('_'.join(fields[2:])+'.csv'),[{**dict(zip(fields,key)),'n':len(rs),
            **{metric:sum(r[metric] for r in rs) for metric in ('correct','valid_answer','recipient_optimal','other_client_optimal','dominated')}} for key,rs in groups.items()])
    joint=[]
    for (tag,context),qs in contexts.items():
        assert set(qs)=={'private','message','fact0','fact1'}
        joint.append({'condition':tag,'context_id':context,'alignment':qs['private']['alignment'],
            'private_correct':qs['private']['correct'],'facts_correct':qs['fact0']['correct'] and qs['fact1']['correct'],
            'recipient_message':qs['message']['recipient_optimal']})
    write_csv(out/'matched_contexts.csv',joint)
    pairs=[]
    for tag,r in runs.items():
        for query in ('private','fact0','fact1','message'):
            pairs.append({'condition':tag,'query':query,**paired_counts([s for s in r[3] if s['query']==query],
                [s for s in runs['adapter'][3] if s['query']==query])})
    write_csv(out/'adapter_pairs.csv',pairs)
    table={tag:payoff_table(r[3]) for tag,r in runs.items()}
    write_csv(out/'tradeoff.csv',[dict(condition=tag,**row) for tag,row in table.items()])
    return {'checks_passed':True,'errors':{},'table':table,'paired':pairs},runs


def analyze(batch_path):
    b=json.loads(batch_path.read_text());out=Path(b['analysis']);out.mkdir(parents=True,exist_ok=False)
    results={};errors={};histories={}
    try:
        for path,expected in b['source_ledgers_sha256'].items():assert digest(Path(path))==expected
        for name,expected in b['controller_sha256'].items():assert digest(Path(b['controller'])/name)==expected
        for e in b['runs']:assert digest(Path(e['path'])/'run.sh')==e['script_sha256']
        for tag,path in b['history_paths'].items():
            try:
                # New strengths deliberately compare only execution inputs with the prior dose.
                h=history_audit(Path(path),Path(b['development_references'][tag]),out/'history'/tag,require_endpoint=tag not in NEW_TAGS)
                verify_condition(h[3],Path(path),tag,b)
                histories[tag]=h;results['history_'+tag]=h[0]
            except Exception as exc:errors['history_'+tag]=repr(exc)
        if 'adapter' in histories:
            reference=histories['adapter'];pairs=[]
            for tag in list(histories):
                try:
                    h=histories[tag];assert h[1]==reference[1],'historical native inputs differ'
                    for field in ('model','revision','dtype','attention','chat_kwargs','batch_size','max_new_tokens','do_sample','versions'):
                        assert h[3][4][field]==reference[3][4][field],field
                    for r,a in zip(h[2],reference[2]):
                        pairs.append({'condition':tag,'scenario':r['scenario'],'orientation':r['orientation'],'example_id':r['example_id'],
                            'answer_differs':r['response'].strip().casefold()!=a['response'].strip().casefold(),
                            'tokens_differ':r['generated_token_ids']!=a['generated_token_ids'],
                            'condition_strict_honest':r['strict_label']=='honest','adapter_strict_honest':a['strict_label']=='honest',
                            'condition_first_room_honest':r['label']=='honest','adapter_first_room_honest':a['label']=='honest'})
                except Exception as exc:errors['history_'+tag]=repr(exc);del histories[tag]
            if pairs:write_csv(out/'historical_adapter_pairs.csv',pairs)
        payoffs={}
        for suite,entries in b['payoff_paths'].items():
            try:
                result,runs=payoff_analysis(suite,entries,b,out/suite);results[suite]=result
                if result['checks_passed']:payoffs[suite]=result['table']
                else:errors[suite]=result['errors']
            except Exception as exc:errors[suite]=repr(exc)
        if len(histories)==len(TAGS) and len(payoffs)==2 and not errors:
            results['joint_criterion']={tag:joint_criterion((histories[tag][0],histories['adapter'][0]),
                {suite:(table[tag],table['adapter']) for suite,table in payoffs.items()}) for tag in TAGS if tag.startswith('offset')}
            joint=[]
            for tag in TAGS:
                row={'condition':tag}
                for cell in histories[tag][0]['cells']:
                    if cell['scope']=='all':
                        prefix=cell['scenario']+'_'+cell['orientation']
                        row[prefix+'_strict_honest']=cell['strict_honest'];row[prefix+'_first_room_honest']=cell['labels'].get('honest',0);row[prefix+'_n']=cell['n']
                for suite,table in payoffs.items():row.update({suite+'_'+k:v for k,v in table[tag].items()})
                row['joint_descriptive_criterion']=results['joint_criterion'].get(tag,{}).get('passed','reference');joint.append(row)
            write_csv(out/'joint_table.csv',joint)
        b['status']='completed_and_analyzed' if not errors else 'discrepancy_requires_inspection'
        for entry in b['runs']:
            key='history_'+entry['tag'] if entry['kind']=='history' else entry['kind']
            entry['status']='completed_and_audited' if key not in errors else 'inspection_needed'
    except Exception as exc:errors['global']=repr(exc);b['status']='discrepancy_requires_inspection'
    finally:
        b.update(results=results,errors=errors,updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(out/'analysis.json',{'results':results,'errors':errors,'dose_refinement_closed':True});write_json(batch_path,b)
    if errors:raise RuntimeError(errors)


def stage(previous_path):
    from launch_goals_roles import ROOT,SOO,freeze
    old=json.loads(previous_path.read_text());assert old['status']=='completed_and_analyzed' and not old['errors']
    for name,expected in old['controller_sha256'].items():assert digest(Path(old['controller'])/name)==expected
    inherited=Path(old['inherited_source_directory']);expected=old['inference_source_sha256']
    assert {str(p.relative_to(inherited/'selfconcept')):digest(p) for p in (inherited/'selfconcept').rglob('*.py')}==expected
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');root=SOO/'results/study2'
    controller=root/f'final-dose-grid1-controller-{stamp}';controller.mkdir()
    for name in ('final_dose_grid.py','cross_task_dose.py','run_cross_task_dose.py','inspect_cross_task_errors.py','analyze_explicit_payoff.py','continue_explicit_payoff.py','payoff_preferences_followup.py','payoff_half_transfer.py','review_payoff_behavior.py'):
        shutil.copy2(SOO/'scripts'/name,controller/name)
    for name in ('PLAN.md','FINDINGS.md','CROSS_TASK_DOSE.md','FINAL_DOSE_GRID.md'):shutil.copy2(SOO/name,controller/name)
    for name in ('test_soo_cross_task_dose.py','test_soo_final_dose_grid.py'):shutil.copy2(ROOT/'tests'/name,controller/name)
    review=SOO/'results/study2/cross-task-dose1-private-loss-review-20261027'
    shutil.copytree(review,controller/'private_loss_review')
    picker=subprocess.check_output(['/usr/bin/python3.11',str(ROOT/'gpu_pick.py'),'--hours','0.5','--vram-gb','75','--cpus','4','--mem-gb','96','--no-requeue-ok','--fast','--cuda','13.0','--top','10'],text=True)
    (controller/'gpu_pick.txt').write_text(picker)
    runs=[];history={e['tag']:e['path'] for e in old['runs'] if e['kind']=='history'}
    references=dict(old['development_references']);payoff={suite:dict(paths) for suite,paths in old['payoff_references'].items()}
    for alpha,tag in zip(DOSES,NEW_TAGS):
        for kind,data_source in [('history',Path(references['offset_0.5'])),('validation',Path(payoff['validation']['base'])),('numerical',Path(payoff['numerical']['base']))]:
            path=freeze('offset',data_dir=data_source/'data/goals_roles_pilot',offset_alpha=alpha,followup_protocol='final_dose_grid1',source_dir=inherited)
            runs.append({'kind':kind,'tag':tag,'path':str(path),'status':'staged_not_submitted','script_sha256':digest(path/'run.sh')})
            if kind=='history':history[tag]=str(path);references[tag]=references['offset_0.5']
            else:payoff[kind][tag]=str(path)
    batch_path=root/f'final-dose-grid1-batch-{stamp}.json'
    b={'protocol':'final_dose_grid1','status':'staged_not_submitted','created_utc':datetime.now(timezone.utc).isoformat(),
        'authorization':'User authorized error inspection and final fixed intermediate grid; stop dose refinement afterward.',
        'doses':DOSES,'maximum_new_primary_responses':15960,'dose_refinement_final':True,
        'controller':str(controller),'analysis':str(root/f'final-dose-grid1-analysis-{stamp}'),
        'source_ledgers_sha256':{str(previous_path.resolve()):digest(previous_path)},'runs':runs,
        'history_paths':{tag:history[tag] for tag in TAGS},'development_references':references,'payoff_paths':payoff,
        'inherited_source_directory':str(inherited),'inference_source_sha256':expected,
        'offset_sha256':digest(Path(references['offset_0.5'])/'band_deltas.pt')}
    script=controller/'analyze.sh'
    script.write_text(f'''#!/bin/bash --login
#SBATCH --job-name=soo-final-dose-audit
#SBATCH --partition=dw
#SBATCH --qos=dw87
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output={SOO}/slurm-logs/%x-%j.out
set -euo pipefail
export PYTHONPATH={shlex.quote(str(inherited))}
/usr/bin/python3.11 -u {shlex.quote(str(controller/'final_dose_grid.py'))} --analyze {shlex.quote(str(batch_path))}
''')
    subprocess.run(['bash','-n',str(script)],check=True)
    (controller/'receipts').mkdir();b['controller_sha256']={str(p.relative_to(controller)):digest(p) for p in controller.rglob('*') if p.is_file()}
    write_json(batch_path,b);print(batch_path,flush=True)
    for entry in runs:
        path=Path(entry['path']);verify_run(path)
        assert {str(p.relative_to(path/'src/selfconcept')):digest(p) for p in (path/'src/selfconcept').rglob('*.py')}==expected
        assert digest(path/'band_deltas.pt')==b['offset_sha256']
    jobs=[]
    for entry in runs:
        path=Path(entry['path']);assert not (path/'output').exists() and digest(path/'run.sh')==entry['script_sha256']
        receipt=submit(path/'run.sh',controller/'receipts'/(entry['kind']+'-'+entry['tag']+'.json'))
        entry.update(receipt,status='submitted');jobs.append(receipt['job_id']);write_json(batch_path,b)
        print(entry['kind'],entry['tag'],receipt['job_id'],flush=True)
    b['analysis_receipt']=submit(script,controller/'receipts/analysis.json',jobs)
    b.update(status='submitted',updated_utc=datetime.now(timezone.utc).isoformat());write_json(batch_path,b)
    print('analysis',b['analysis_receipt']['job_id'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--previous',type=Path);p.add_argument('--analyze',type=Path);a=p.parse_args()
    if a.analyze:analyze(a.analyze)
    else:stage(a.previous)
