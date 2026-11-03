"""Frozen numerical-preference and offset-dose workflow; no automatic expansion."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shlex
import shutil
import subprocess

from analyze_explicit_payoff import audit_run, digest, read, verify_run, write_json
from continue_explicit_payoff import submit
from review_payoff_behavior import write_csv
from selfconcept.soo.payoff_preferences import build_dataset, rule_keys

DOSES=(0.0,0.25,0.5,0.75,1.0)
SETTING_FIELDS=('model','revision','dtype','attention','chat_kwargs','batch_size','max_new_tokens','do_sample','versions','batches')


def feasibility(audit,data,scores):
    checks=[r for r in audit['gate']['checks'] if r['query'] in ('private','fact0','fact1')]
    assert len(checks)==84
    for profile in range(6):
        for query in ('private','fact0','fact1'):
            rows=[s for e,s in zip(data,scores) if e['profile']==profile and e['query']==query]
            assert len(rows)==108
            for metric,threshold in (('valid_answer',.95),('correct',.90)):
                count=sum(r[metric] for r in rows)
                checks.append({'group':f'profile{profile}','query':query,'metric':metric,'count':count,
                               'n':len(rows),'threshold':threshold,'passed':count/len(rows)>=threshold})
    return {'passed':all(r['passed'] for r in checks),'checks':checks,'scope':'private/factual only; recommendations descriptive'}


def match_inputs(first,second,bridge=False):
    # audit tuples: audit, data, native inputs, scores, manifest
    assert first[1]==second[1] and first[2]==second[2], 'prompt/data mismatch'
    for field in SETTING_FIELDS:assert first[4][field]==second[4][field],field
    a,b=first[4]['source_sha256'],second[4]['source_sha256']
    if bridge:
        # First is the new runner, second the immutable original execution.
        assert not set(b)-set(a)
        assert set(a)-set(b)=={'soo/payoff_diagnostic.py','soo/payoff_preferences.py'}
        assert {k for k in a.keys()&b.keys() if a[k]!=b[k]}=={'soo/run_goals_roles.py'}
    else:assert a==b


def endpoint_bridge(new_run,old_run):
    fields=('responses.jsonl','regressions.jsonl')
    comparisons={name:read(new_run/'output'/name)==read(old_run/'output'/name) for name in fields}
    return {'passed':all(comparisons.values()),'exact_record_identity':comparisons,
            'n_task':432,'n_historical':32}


def numerical_audit(entry,original,out):
    result=audit_run(Path(entry['path']),out,expected_n=2592)
    audit,data,inputs,scores,manifest=result
    assert data==build_dataset(original), 'not the frozen six-profile factorial'
    for e in data:assert e['rule_answers']==rule_keys(e['instance'])
    assert manifest['condition']==entry['condition'] and manifest['offset_alpha']==1.0
    return result


def dose_analysis(batch,out):
    out.mkdir(parents=True,exist_ok=False)
    old={k:audit_run(Path(v),out/'historical'/k) for k,v in batch['historical'].items()}
    trials,table,runs=[],[],{}
    for entry in batch['dose_runs']:
        alpha=entry['alpha'];label=f'offset_{alpha:g}'
        result=audit_run(Path(entry['path']),out/'audits'/label)
        assert result[4]['condition']=='offset' and result[4]['offset_alpha']==alpha
        match_inputs(result,old['offset'],bridge=True)
        # The offset artifact, rather than only its recorded alpha, must be identical.
        assert digest(Path(entry['path'])/'band_deltas.pt')==digest(Path(batch['historical']['offset'])/'band_deltas.pt')
        runs[alpha]=result
    for label,result in [('base',old['base']),('adapter',old['adapter'])]+[(f'offset_{a:g}',runs[a]) for a in DOSES]:
        scores=result[3]
        private=[r for r in scores if r['query']=='private']
        facts=[r for r in scores if r['query'].startswith('fact')]
        opposed=[r for r in scores if r['query']=='message' and r['alignment']=='opposed']
        aligned=[r for r in scores if r['query']=='message' and r['alignment']=='aligned']
        table.append({'condition':label,'private_correct':sum(r['correct'] for r in private),'private_n':len(private),
                      'facts_correct':sum(r['correct'] for r in facts),'facts_n':len(facts),
                      'opposed_recipient':sum(r['recipient_optimal'] for r in opposed),'opposed_n':len(opposed),
                      'aligned_correct':sum(r['correct'] for r in aligned),'aligned_n':len(aligned)})
        trials.extend({'condition':label,**r} for r in scores)
    base_count=table[0]['opposed_recipient'];adapter_count=table[1]['opposed_recipient']
    assert adapter_count>base_count
    for row in table:
        row['communication_effect_retention']=(row['opposed_recipient']-base_count)/(adapter_count-base_count)
        row['perfect_private_facts_above_baseline_recommendations']=(row['private_correct']==row['private_n'] and
            row['facts_correct']==row['facts_n'] and row['opposed_recipient']>base_count)
    bridges={}
    for alpha,key in ((0.0,'base'),(1.0,'offset')):
        entry=next(e for e in batch['dose_runs'] if e['alpha']==alpha)
        match_inputs(runs[alpha],old[key],bridge=True)
        bridges[str(alpha)]=endpoint_bridge(Path(entry['path']),Path(batch['historical'][key]))
    historical={}
    for entry in batch['dose_runs']:
        records=read(Path(entry['path'])/'output/regressions.jsonl')
        historical[str(entry['alpha'])]={'complete':all(r['completion_status']=='complete' for r in records),
            'labels':dict(Counter(r['scenario']+':'+r['label'] for r in records))}
    clean=(all(v['passed'] for v in bridges.values()) and all(r[0]['repeat_passed'] for r in runs.values()) and
           all(v['complete'] for v in historical.values()))
    write_csv(out/'tradeoff.csv',table);write_csv(out/'all_trials.csv',trials)
    # Matplotlib lives in the inference environment; CPU audit Python stays lightweight.
    subprocess.run([batch['plot_python'],str(Path(__file__).resolve()),'--plot',str(out/'tradeoff.csv')],check=True,timeout=300)
    result={'all_checks_passed':clean,'bridges':bridges,'historical_by_dose':historical,'tradeoff':table,
            'interpretation':'fixed-grid descriptive strength tradeoff; no matched-effectiveness or equivalence claim'}
    write_json(out/'analysis.json',result)
    return result


def plot_tradeoff(path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table=list(csv.DictReader(path.open()))
    fig,ax=plt.subplots(figsize=(7,4.5))
    for i,r in enumerate(table):
        x=100*float(r['opposed_recipient'])/float(r['opposed_n'])
        y=100*float(r['private_correct'])/float(r['private_n'])
        ax.scatter(x,y,marker='s' if i<2 else 'o')
        ax.annotate(r['condition'],(x,y),xytext=(4,-12-10*(i%2)),textcoords='offset points',fontsize=8)
    ax.set(xlabel='Opposed recipient-optimal recommendations (%)',ylabel='Correct private choices (%)',ylim=(0,105),xlim=(-2,105))
    fig.tight_layout();fig.savefig(path.with_suffix('.png'),dpi=180);plt.close(fig)


def numerical_analysis(batch,out):
    out.mkdir(parents=True,exist_ok=False)
    original=read(Path(batch['historical']['base'])/'data/goals_roles_pilot/main.jsonl')+read(Path(batch['historical']['base'])/'data/goals_roles_pilot/checks.jsonl')
    runs={e['condition']:numerical_audit(e,original,out/'audits'/e['condition']) for e in batch['numeric_runs']}
    rows=[]
    for condition,result in runs.items():
        match_inputs(result,runs['base'])
        for e,s in zip(result[1],result[3]):
            rows.append({'condition':condition,'profile':e['profile'],'lexical_instance':e['lexical_instance'],**s,
                         **{rule+'_answer':answer for rule,answer in e['rule_answers'].items()},
                         **{rule+'_match':s['valid_answer'] and s['answer']==answer for rule,answer in e['rule_answers'].items()}})
    fields=('correct','recipient_optimal','other_client_optimal','dominated','total_match','minimum_match','largest_single_match')
    def grouped(keys):
        groups=defaultdict(list)
        for r in rows:groups[tuple(r[k] for k in keys)].append(r)
        return [{**dict(zip(keys,key)),'n':len(rs),**{f:sum(r[f] for r in rs) for f in fields}} for key,rs in groups.items()]
    cells=grouped(('condition','query','profile','lexical_instance','role','alignment'))
    write_csv(out/'cells.csv',cells)
    write_csv(out/'order_cells.csv',grouped(('condition','query','table_order','answer_order')))
    write_csv(out/'all_trials.csv',rows)
    disagreements=[]
    for condition in runs:
        for query in ('private','message'):
            for left,right in (('total','minimum'),('total','largest_single'),('minimum','largest_single')):
                selected=[r for r in rows if r['condition']==condition and r['query']==query and r[left+'_answer']!=r[right+'_answer']]
                disagreements.append({'condition':condition,'query':query,'first_rule':left,'second_rule':right,
                                      'n':len(selected),'first_rule_matches':sum(r[left+'_match'] for r in selected),
                                      'second_rule_matches':sum(r[right+'_match'] for r in selected),
                                      'correct':sum(r['correct'] for r in selected)})
    write_csv(out/'rule_disagreements.csv',disagreements)
    contexts=defaultdict(dict)
    for r in rows:contexts[r['condition'],r['context_id']][r['query']]=r
    joint=[]
    for (condition,context),queries in contexts.items():
        assert set(queries)=={'private','message','fact0','fact1'}
        m=queries['message']
        joint.append({'condition':condition,'context_id':context,'profile':m['profile'],'alignment':m['alignment'],
                      'recipient_message':m['recipient_optimal'],'private_and_facts_correct':all(queries[q]['correct'] for q in ('private','fact0','fact1'))})
    write_csv(out/'matched_contexts.csv',joint)
    profile_lookup={(r['condition'],r['lexical_instance'],r['role'],r['alignment'],r['query'],r['table_order'],r['answer_order'],r['profile']):r for r in rows}
    profile_pairs=[]
    for r in rows:
        for profile in range(r['profile']+1,6):
            other=profile_lookup[r['condition'],r['lexical_instance'],r['role'],r['alignment'],r['query'],r['table_order'],r['answer_order'],profile]
            profile_pairs.append({'condition':r['condition'],'first':r['example_id'],'second':other['example_id'],
                'first_profile':r['profile'],'second_profile':profile,'query':r['query'],
                'answer_changed':r['answer']!=other['answer'],'lost_correct':r['correct'] and not other['correct'],
                'gained_correct':not r['correct'] and other['correct']})
    write_csv(out/'profile_pairs.csv',profile_pairs)
    result={'all_checks_passed':all(r[0]['repeat_passed'] and r[0]['regressions_passed'] for r in runs.values()),
            'summaries':{k:r[0]['summary'] for k,r in runs.items()},'rule_disagreements':disagreements,
            'baseline_gate':feasibility(*[runs['base'][i] for i in (0,1,3)]),
            'interpretation':'exploratory six-feasible-profile preference comparison on three lexical/scaled templates; no internal utility or independent-task claim'}
    write_json(out/'analysis.json',result)
    return result


def release_allowed(dose,gate,audit):
    return dose['all_checks_passed'] and gate['passed'] and audit['repeat_passed'] and audit['regressions_passed']


def run_stage(batch_path,stage):
    with batch_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        b=json.loads(batch_path.read_text());controller=Path(b['controller'])
        def save():
            b['updated_utc']=datetime.now(timezone.utc).isoformat()
            temp=batch_path.with_suffix('.tmp');write_json(temp,b);temp.replace(batch_path)
        def launch(entry,label):
            assert entry['status']=='staged_not_submitted'
            p=Path(entry['path']);verify_run(p)
            assert digest(p/'run.sh')==entry['script_sha256'] and not (p/'output').exists()
            receipt=submit(p/'run.sh',controller/'receipts'/(label+'.json'))
            entry.update(receipt,status='submitted');save();return receipt['job_id']
        def next_stage(name,jobs):
            receipt=submit(controller/(name+'.sh'),controller/'receipts'/(name+'.json'),jobs)
            b.setdefault('controllers',{})[name]=receipt;save()
        try:
            assert digest(Path(b['source_ledger']))==b['source_ledger_sha256']
            for n,h in b['controller_sha256'].items():assert digest(controller/n)==h,n
            if stage=='start':
                assert b['status']=='staged_not_submitted'
                # Verify every staged execution before the first submission.
                for entry in b['dose_runs']+b['numeric_runs']:
                    verify_run(Path(entry['path']))
                    assert not (Path(entry['path'])/'output').exists()
                jobs=[launch(b['numeric_runs'][0],'numeric_base')]
                jobs += [launch(e,'dose_'+str(e['alpha'])) for e in b['dose_runs']]
                b['status']='dose_and_numeric_baseline_submitted';save();next_stage('check',jobs)
            elif stage=='check':
                assert b['status']=='dose_and_numeric_baseline_submitted'
                dose=dose_analysis(b,Path(b['dose_analysis']));b['dose_result']=dose
                original=read(Path(b['historical']['base'])/'data/goals_roles_pilot/main.jsonl')+read(Path(b['historical']['base'])/'data/goals_roles_pilot/checks.jsonl')
                baseline=numerical_audit(b['numeric_runs'][0],original,Path(b['numeric_analysis'])/'baseline_audit')
                gate=feasibility(*[baseline[i] for i in (0,1,3)]);b['numeric_baseline_gate']=gate
                write_json(Path(b['numeric_analysis'])/'baseline_audit/private_gate.json',gate)
                for e in b['dose_runs']+[b['numeric_runs'][0]]:e['status']='completed'
                if not release_allowed(dose,gate,baseline[0]):
                    b['status']='checks_failed_stopped';save();return
                jobs=[launch(e,'numeric_'+e['condition']) for e in b['numeric_runs'][1:]]
                b['status']='numeric_interventions_submitted';save();next_stage('final',jobs)
            elif stage=='final':
                assert b['status']=='numeric_interventions_submitted'
                result=numerical_analysis(b,Path(b['numeric_analysis'])/'paired')
                b['numeric_result']=result
                for e in b['numeric_runs']:e['status']='completed'
                b['status']='completed_and_analyzed' if result['all_checks_passed'] else 'discrepancy_requires_inspection'
                save()
            else:raise ValueError(stage)
        except Exception as error:
            b['status']='execution_error_requires_inspection';b.setdefault('errors',{})[stage]=repr(error);save();raise


def stage_all(source_path):
    from launch_goals_roles import ROOT,SOO,freeze
    source=json.loads(source_path.read_text());assert source['status']=='finished'
    preflight=SOO/'results/study2/payoff-preferences1-token-preflight/summary.json'
    token=json.loads(preflight.read_text());assert token['status']=='passed' and token['prompts_checked']==2592
    for name,expected in token['source_sha256'].items():assert digest(ROOT/name)==expected
    validation=preflight.parent/'validation.json'
    validated=json.loads(validation.read_text());assert validated['status']=='passed'
    for name,expected in validated['sha256'].items():assert digest(ROOT/name)==expected,name
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');root=SOO/'results/study2'
    controller=root/f'payoff-preferences1-controller-{stamp}';controller.mkdir()
    for name in ('payoff_preferences_followup.py','analyze_explicit_payoff.py','continue_explicit_payoff.py','review_payoff_behavior.py'):
        shutil.copy2(SOO/'scripts'/name,controller/name)
    for name in ('PAYOFF_PREFERENCES.md','PLAN.md','FINDINGS.md'):shutil.copy2(SOO/name,controller/name)
    shutil.copy2(preflight,controller/'native_token_preflight.json')
    shutil.copy2(validation,controller/'validation.json')
    shutil.copy2(preflight.parent/'runner_alpha.diff',controller/'runner_alpha.diff')
    shutil.copy2(ROOT/'tests/test_soo_payoff_preferences.py',controller/'test_soo_payoff_preferences.py')
    picker=subprocess.check_output(['/usr/bin/python3.11',str(ROOT/'gpu_pick.py'),'--hours','0.5','--vram-gb','75','--cpus','4','--mem-gb','96','--no-requeue-ok','--fast','--cuda','13.0','--top','10'],text=True)
    (controller/'gpu_pick.txt').write_text(picker)
    historical={e['tag']:e['path'] for e in source['runs']['development'] if e['tag'] in ('base','adapter','offset')}
    def frozen(condition,data,alpha=1.0):
        p=freeze(condition,data_dir=data,offset_alpha=alpha,followup_protocol='payoff_preferences1')
        return {'condition':condition,'alpha':alpha,'path':str(p),'status':'staged_not_submitted','script_sha256':digest(p/'run.sh')}
    dose_runs=[frozen('offset',Path(historical['base'])/'data/goals_roles_pilot',a) for a in DOSES]
    numeric_runs=[frozen(c,SOO/'data/payoff_preferences1') for c in ('base','adapter','offset')]
    batch_path=root/f'payoff-preferences1-batch-{stamp}.json'
    b={'protocol':'payoff_preferences1','status':'staged_not_submitted','created_utc':datetime.now(timezone.utc).isoformat(),
       'authorization':'User requested continuation with numerical preferences and offset strength; fixed protocol and eight snapshots before outputs.',
       'source_ledger':str(source_path.resolve()),'source_ledger_sha256':digest(source_path),'controller':str(controller),
       'historical':historical,'dose_runs':dose_runs,'numeric_runs':numeric_runs,'maximum_new_scored_generations':9936,
       'plot_python':str(ROOT/'.venv-313/bin/python'),
       'dose_analysis':str(root/f'payoff-dose1-analysis-{stamp}'),'numeric_analysis':str(root/f'payoff-preferences1-analysis-{stamp}')}
    for stage in ('check','final'):
        command=['/usr/bin/python3.11','-u',str(controller/'payoff_preferences_followup.py'),'--batch',str(batch_path),'--stage',stage]
        script=controller/(stage+'.sh')
        script.write_text(f'''#!/bin/bash --login
#SBATCH --job-name=soo-paypref-{stage}
#SBATCH --partition=dw
#SBATCH --qos=dw87
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output={SOO}/slurm-logs/%x-%j.out
set -euo pipefail
export OPENSSL_CONF=/dev/null
export PYTHONPATH={shlex.quote(str(Path(numeric_runs[0]['path'])/'src'))}
{shlex.join(command)}
''')
        subprocess.run(['bash','-n',str(script)],check=True)
    (controller/'receipts').mkdir()
    b['controller_sha256']={p.name:digest(p) for p in controller.iterdir() if p.is_file()}
    write_json(batch_path,b);print(batch_path,flush=True)
    run_stage(batch_path,'start')
    print(json.dumps({'status':json.loads(batch_path.read_text())['status'],'batch':str(batch_path)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage-all',type=Path);p.add_argument('--batch',type=Path)
    p.add_argument('--plot',type=Path)
    p.add_argument('--stage',choices=('start','check','final'))
    args=p.parse_args()
    if args.plot:plot_tradeoff(args.plot)
    elif args.stage_all:stage_all(args.stage_all)
    else:
        if not args.batch or not args.stage:p.error('--batch and --stage required')
        run_stage(args.batch,args.stage)
