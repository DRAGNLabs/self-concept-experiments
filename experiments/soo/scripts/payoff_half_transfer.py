"""Run a fixed alpha-0.5 transfer on two existing payoff evaluation sets."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import shutil
import subprocess

from analyze_explicit_payoff import audit_run,digest,read,verify_run,write_json
from continue_explicit_payoff import submit
from payoff_preferences_followup import feasibility,match_inputs
from review_payoff_behavior import write_csv
from selfconcept.soo.payoff_preferences import build_dataset,rule_keys
from selfconcept.soo.run_goals_roles import source_hashes


def paired_counts(first,second):
    assert [r['example_id'] for r in first]==[r['example_id'] for r in second]
    return {'n':len(first),'answer_differences':sum(a['answer']!=b['answer'] for a,b in zip(first,second)),
            'first_correct_second_wrong':sum(a['correct'] and not b['correct'] for a,b in zip(first,second)),
            'first_wrong_second_correct':sum(not a['correct'] and b['correct'] for a,b in zip(first,second))}


def selected_dose_checks(result,path,selected):
    reference_manifest=json.loads((selected/'output/manifest.json').read_text())
    assert result[4]['source_sha256']==reference_manifest['source_sha256']
    assert result[4]['condition']=='offset' and result[4]['offset_alpha']==0.5
    assert digest(path/'band_deltas.pt')==digest(selected/'band_deltas.pt')
    records=read(path/'output/regressions.jsonl')
    historical_equal=records==read(selected/'output/regressions.jsonl')
    assert len(records)==32
    return {'passed':result[0]['repeat_passed'] and historical_equal,
            'sentinel_repeat_passed':result[0]['repeat_passed'],
            'historical_exact_record_identity':historical_equal,
            'historical_labels':dict(Counter(r['scenario']+':'+r['label'] for r in records)),
            'generic_full_strength_regression_expectation':result[0]['regressions_passed']}


def analyze_suite(name,suite,selected,original,out):
    out.mkdir(parents=True,exist_ok=False)
    n=1296 if name=='validation' else 2592
    entries={**suite['references'],'half':suite['half']['path']}
    runs={tag:audit_run(Path(path),out/'audits'/tag,expected_n=n) for tag,path in entries.items()}
    half=runs['half'];half_path=Path(entries['half'])
    checks=selected_dose_checks(half,half_path,selected)
    for tag,result in runs.items():
        match_inputs(half,result,bridge=(name=='validation' and tag!='half'))
        if tag!='half':assert result[4]['condition']==tag
    assert runs['offset'][4]['offset_alpha']==1.0
    assert digest(half_path/'band_deltas.pt')==digest(Path(entries['offset'])/'band_deltas.pt')
    for tag in ('base','adapter','offset'):
        checks['passed'] &= runs[tag][0]['repeat_passed'] and runs[tag][0]['regressions_passed']
    if name=='numerical':
        assert half[1]==build_dataset(original)
        for e in half[1]:assert e['rule_answers']==rule_keys(e['instance'])
        gate=feasibility(*[runs['base'][i] for i in (0,1,3)])
    else:
        scope=[c for c in runs['base'][0]['gate']['checks'] if c['query']!='message']
        assert len(scope)==84
        gate={'passed':all(c['passed'] for c in scope),'checks':scope}
    assert gate['passed'],'previous private/factual baseline no longer passes'
    data=half[1];rows=[]
    for tag,result in runs.items():
        for e,s in zip(data,result[3]):
            row={'condition':tag,**s}
            if name=='numerical':
                row.update(profile=e['profile'],lexical_instance=e['lexical_instance'],
                           **{k+'_match':s['valid_answer'] and s['answer']==v for k,v in e['rule_answers'].items()})
            rows.append(row)
    write_csv(out/'all_trials.csv',rows)
    metrics=['correct','valid_answer','recipient_optimal','other_client_optimal','dominated']
    if name=='numerical':metrics+=['total_match','minimum_match','largest_single_match']
    def cells(fields):
        groups=defaultdict(list)
        for r in rows:groups[tuple(r[k] for k in fields)].append(r)
        return [{**dict(zip(fields,key)),'n':len(rs),**{m:sum(r[m] for r in rs) for m in metrics}} for key,rs in groups.items()]
    write_csv(out/'cells.csv',cells(('condition','query','instance_id','role','alignment')))
    write_csv(out/'order_cells.csv',cells(('condition','query','table_order','answer_order')))
    if name=='numerical':write_csv(out/'profile_cells.csv',cells(('condition','query','profile','role','alignment')))
    paired=[]
    for reference in ('base','adapter','offset'):
        for query in ('private','message','fact0','fact1'):
            a=[r for r in half[3] if r['query']==query];b=[r for r in runs[reference][3] if r['query']==query]
            paired.append({'first':'half','second':reference,'query':query,**paired_counts(a,b)})
    write_csv(out/'paired_counts.csv',paired)
    contexts=defaultdict(dict)
    for row in rows:contexts[row['condition'],row['context_id']][row['query']]=row
    joint=[]
    for (tag,context),qs in contexts.items():
        assert set(qs)=={'private','message','fact0','fact1'}
        joint.append({'condition':tag,'context_id':context,'alignment':qs['message']['alignment'],
                      'recipient_message':qs['message']['recipient_optimal'],
                      'private_and_facts_correct':all(qs[q]['correct'] for q in ('private','fact0','fact1'))})
    write_csv(out/'matched_contexts.csv',joint)
    half_responses=read(half_path/'output/responses.jsonl')
    disagreements={}
    for tag in ('base','adapter','offset'):
        responses=read(Path(entries[tag])/'output/responses.jsonl')
        disagreements[tag]={**paired_counts(half[3],runs[tag][3]),
                            'token_differences':sum(a['generated_token_ids']!=b['generated_token_ids'] for a,b in zip(half_responses,responses))}
    result={'suite':name,'checks':checks,'baseline_gate':gate,
            'summaries':{tag:r[0]['summary'] for tag,r in runs.items()},
            'by_query_alignment':cells(('condition','query','alignment')),
            'paired':paired,'all_query_disagreements':disagreements,
            'interpretation':'fixed development-selected dose on existing evaluation sets; no new blind holdout or equivalence claim'}
    write_json(out/'analysis.json',result)
    return result


def analyze(batch_path):
    b=json.loads(batch_path.read_text());out=Path(b['analysis']);out.mkdir(parents=True,exist_ok=False)
    try:
        assert b['status']=='submitted'
        for name,expected in b['controller_sha256'].items():assert digest(Path(b['controller'])/name)==expected
        for path,expected in b['source_ledgers_sha256'].items():assert digest(Path(path))==expected
        original=read(Path(b['original_development'])/'data/goals_roles_pilot/main.jsonl')+read(Path(b['original_development'])/'data/goals_roles_pilot/checks.jsonl')
        results={};errors={}
        for name,suite in b['suites'].items():
            try:
                results[name]=analyze_suite(name,suite,Path(b['selected_dose']),original,out/name)
                suite['half']['status']='completed'
            except Exception as error:errors[name]=repr(error)
        b['results']=results;b['errors']=errors
        clean=not errors and len(results)==2 and all(r['checks']['passed'] for r in results.values())
        b['status']='completed_and_analyzed' if clean else 'discrepancy_requires_inspection'
        write_json(out/'analysis.json',{'results':results,'errors':errors,'all_checks_passed':clean})
        if errors:raise RuntimeError(errors)
    except Exception as error:
        b['status']='execution_error_requires_inspection';b['error']=repr(error);raise
    finally:
        b['updated_utc']=datetime.now(timezone.utc).isoformat();write_json(batch_path,b)


def stage(private_path,preferences_path):
    from launch_goals_roles import ROOT,SOO,freeze
    private=json.loads(private_path.read_text());prefs=json.loads(preferences_path.read_text())
    assert private['status']=='finished' and prefs['status']=='completed_and_analyzed'
    assert private['baseline_audits']['validation']['private_gate']['passed']
    assert prefs['numeric_baseline_gate']['passed'] and prefs['dose_result']['all_checks_passed'] and prefs['numeric_result']['all_checks_passed']
    selected=Path(next(e['path'] for e in prefs['dose_runs'] if e['alpha']==0.5))
    expected_source=json.loads((selected/'output/manifest.json').read_text())['source_sha256']
    assert source_hashes()==expected_source,'inference source changed since selected dose'
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');root=SOO/'results/study2'
    controller=root/f'payoff-half-transfer1-controller-{stamp}';controller.mkdir()
    for name in ('payoff_half_transfer.py','payoff_preferences_followup.py','analyze_explicit_payoff.py','continue_explicit_payoff.py','review_payoff_behavior.py'):
        shutil.copy2(SOO/'scripts'/name,controller/name)
    for name in ('PLAN.md','FINDINGS.md','PAYOFF_HALF_TRANSFER.md'):shutil.copy2(SOO/name,controller/name)
    shutil.copy2(ROOT/'tests/test_soo_payoff_half_transfer.py',controller/'test_soo_payoff_half_transfer.py')
    picker=subprocess.check_output(['/usr/bin/python3.11',str(ROOT/'gpu_pick.py'),'--hours','0.5','--vram-gb','75','--cpus','4','--mem-gb','96','--no-requeue-ok','--fast','--cuda','13.0','--top','10'],text=True)
    (controller/'gpu_pick.txt').write_text(picker)
    suites={
        'validation':{'references':{e['tag']:e['path'] for e in private['runs']['validation'] if e['tag'] in ('base','adapter','offset')}},
        'numerical':{'references':{e['condition']:e['path'] for e in prefs['numeric_runs']}}}
    for name,suite in suites.items():
        path=freeze('offset',data_dir=Path(suite['references']['base'])/'data/goals_roles_pilot',offset_alpha=0.5,followup_protocol='payoff_half_transfer1')
        suite['half']={'path':str(path),'status':'staged_not_submitted','script_sha256':digest(path/'run.sh')}
        # New frozen inference is byte-identical to the already executed half-dose runner.
        assert {str(p.relative_to(path/'src/selfconcept')):digest(p) for p in (path/'src/selfconcept').rglob('*.py')}==expected_source
    batch_path=root/f'payoff-half-transfer1-batch-{stamp}.json'
    b={'protocol':'payoff_half_transfer1','status':'staged_not_submitted','created_utc':datetime.now(timezone.utc).isoformat(),
       'authorization':'User requested proceeding with half-strength transfer to larger evaluation sets; exactly two fixed alpha-0.5 runs.',
       'source_ledgers_sha256':{str(p.resolve()):digest(p) for p in (private_path,preferences_path)},
       'selected_dose':str(selected),'original_development':prefs['historical']['base'],
       'controller':str(controller),'suites':suites,'maximum_new_scored_generations':3888,
       'analysis':str(root/f'payoff-half-transfer1-analysis-{stamp}')}
    command=['/usr/bin/python3.11','-u',str(controller/'payoff_half_transfer.py'),'--analyze',str(batch_path)]
    script=controller/'analyze.sh'
    script.write_text(f'''#!/bin/bash --login
#SBATCH --job-name=soo-half-transfer-audit
#SBATCH --partition=dw
#SBATCH --qos=dw87
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output={SOO}/slurm-logs/%x-%j.out
set -euo pipefail
export PYTHONPATH={shlex.quote(str(Path(suites['validation']['half']['path'])/'src'))}
{shlex.join(command)}
''')
    subprocess.run(['bash','-n',str(script)],check=True)
    (controller/'receipts').mkdir()
    b['controller_sha256']={p.name:digest(p) for p in controller.iterdir() if p.is_file()}
    write_json(batch_path,b);print(batch_path,flush=True)
    for suite in suites.values():verify_run(Path(suite['half']['path']))
    jobs=[]
    for name,suite in suites.items():
        entry=suite['half'];path=Path(entry['path'])
        assert digest(path/'run.sh')==entry['script_sha256'] and not (path/'output').exists()
        receipt=submit(path/'run.sh',controller/'receipts'/(name+'.json'))
        entry.update(receipt,status='submitted');jobs.append(receipt['job_id']);write_json(batch_path,b)
        print(name,receipt['job_id'],flush=True)
    b['analysis_receipt']=submit(script,controller/'receipts/analysis.json',jobs)
    b.update(status='submitted',updated_utc=datetime.now(timezone.utc).isoformat());write_json(batch_path,b)
    print('analysis',b['analysis_receipt']['job_id'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--private-batch',type=Path);p.add_argument('--preferences-batch',type=Path);p.add_argument('--analyze',type=Path)
    a=p.parse_args()
    if a.analyze:analyze(a.analyze)
    else:
        if not a.private_batch or not a.preferences_batch:p.error('both source batches are required')
        stage(a.private_batch,a.preferences_batch)
