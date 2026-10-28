"""Freeze, execute and audit the bounded repeatability/bridge follow-up."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

from extension_repeat_selection import TAGS, digest, read, select
from extension_bridge_task import build, score, gate


def write(path, obj):
    path=Path(path); tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2)+'\n'); tmp.replace(path)


def jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(r)+'\n' for r in rows))


def hashes(root):
    return {str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}


def verify(batch):
    root=Path(batch['controller'])
    for name,expected in batch['frozen_sha256'].items(): assert digest(root/name)==expected, name
    for name,expected in batch['reference_sha256'].items(): assert digest(name)==expected, name


def submit(script, receipt_path, dependencies=()):
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text()); assert receipt['script_sha256']==digest(script)
        return receipt
    command=['sbatch','--parsable']
    if dependencies: command += ['--dependency=afterany:'+':'.join(dependencies)]
    job=subprocess.check_output(command+[str(script)],text=True).strip().split(';')[0]
    receipt=dict(job_id=job,script_sha256=digest(script),dependencies=list(dependencies),submitted_utc=datetime.now(timezone.utc).isoformat())
    write(receipt_path,receipt); return receipt


def execute(batch_path, kind, tag):
    """Call the unchanged inherited runner; replace its explicit-plan callback only."""
    from selfconcept.soo import run_goals_roles as runner, goals_roles_repeatability
    batch=json.loads(batch_path.read_text()); verify(batch)
    entry=batch['runs'][kind][tag]; root=Path(batch['controller']); out=Path(entry['output'])
    assert not out.exists(), 'fresh output required'
    plan_path=root/entry['plan']; plan=json.loads(plan_path.read_text()); examples=read(root/entry['data'])
    args=runner.parser().parse_args(['--revision',batch['revision'],'--data',str(root/entry['data']),
        '--out',str(out),'--provenance',str(root/'provenance.json'),'--condition',entry['condition'],
        '--repeat-plan',str(plan_path),'--regression-data',str(root/'historical_data')]+
        (['--adapter',str(root/'adapter')] if tag=='adapter' else [])+
        (['--offsets',str(root/'band_deltas.pt'),'--offset-alpha',str(entry['alpha'])] if entry['condition']=='offset' else []))
    assert args.batch_size==8 and args.max_new_tokens==32 and not args.resume
    assert tag in TAGS and entry['alpha']==({'offset_0.5':.5,'offset_0.6875':.6875}.get(tag,1.0))
    def callback(path, output, actual_examples, encoded, generate):
        assert path==plan_path and actual_examples==examples
        assert {e['example_id']:e['input_ids'] for e in encoded}==plan['expected_inputs']
        by_id={r['example_id']:r for r in encoded}
        with (output/'planned_responses.jsonl').open('x') as handle:
            for phase in plan['phases']:
                seen=set()
                for index,ids in enumerate(phase['batches']):
                    assert not seen.intersection(ids); seen.update(ids)
                    raw=generate([by_id[eid] for eid in ids])
                    assert [r['example_id'] for r in raw]==ids
                    for r in raw:
                        runner.append_record(handle,dict(r,phase=phase['name'],batch_index=index,batch_members=ids,
                            padded_width=max(len(by_id[eid]['input_ids']) for eid in ids)))
                print(phase['name'],len(seen),'responses',flush=True)
        write(output/'execution_complete.json',dict(phases=len(plan['phases']),generations=plan['generations']))
    goals_roles_repeatability.execute_plan=callback
    out.mkdir(parents=True,exist_ok=False)
    with (out/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runner.run(args,examples)


RAW_FIELDS=('example_id','response','raw_response','generated_token_ids','prompt_tokens','completion_status','generated_tokens')


def raw_only(r):
    return {k:r[k] for k in RAW_FIELDS}


def audit_run(batch,kind,tag):
    entry=batch['runs'][kind][tag]; root=Path(batch['controller']); out=Path(entry['output'])
    plan=json.loads((root/entry['plan']).read_text()); examples=read(root/entry['data'])
    assert json.loads((out/'execution_complete.json').read_text())==dict(phases=len(plan['phases']),generations=plan['generations'])
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['source_sha256']==batch['inference_source_sha256']
    assert manifest['condition']==entry['condition'] and manifest['offset_alpha']==entry['alpha']
    assert manifest['revision']==batch['revision'] and manifest['model']=='google/gemma-4-31B-it'
    assert manifest['batch_size']==8 and manifest['max_new_tokens']==32 and manifest['do_sample'] is False
    assert manifest['dtype']=='bfloat16' and manifest['attention']=='sdpa' and manifest['chat_kwargs']=={'enable_thinking':False}
    assert manifest['versions']==batch['reference_versions'] and 'A100' in manifest['hardware']['gpu']
    for path,expected in manifest['artifact_sha256'].items(): assert digest(path)==expected
    assert manifest['example_ids']==[e['example_id'] for e in examples]
    inputs=read(out/'inputs.jsonl')
    assert inputs==read(root/entry['preflight']/'token_spans.jsonl')
    assert manifest['prompt_ids_sha256']==hashlib.sha256(json.dumps([r['input_ids'] for r in inputs]).encode()).hexdigest()
    assert {r['example_id']:r['input_ids'] for r in inputs}==plan['expected_inputs']
    assert read(out/'regressions.jsonl')==read(Path(batch['history_references'][tag])/'output/regressions.jsonl')
    if tag=='base': assert json.loads((out/'intervention.json').read_text())['inactive_hook_check']['passed']
    responses=read(out/'planned_responses.jsonl'); expected=[]
    for phase in plan['phases']:
        for i,ids in enumerate(phase['batches']):
            expected += [(phase['name'],i,ids,eid) for eid in ids]
    assert len(responses)==len(expected)==plan['generations']
    by_phase=defaultdict(dict)
    for r,(phase,index,ids,eid) in zip(responses,expected):
        assert (r['phase'],r['batch_index'],r['batch_members'],r['example_id'])==(phase,index,ids,eid)
        assert r['padded_width']==max(len(plan['expected_inputs'][i]) for i in ids)
        assert r['prompt_tokens']==len(plan['expected_inputs'][eid]) and r['generated_tokens']==len(r['generated_token_ids'])
        assert r['completion_status'] in ('complete','truncated')
        assert eid not in by_phase[phase]; by_phase[phase][eid]=r
    if kind!='repeat':
        for name in ('sentinel_before','sentinel_after'):
            assert all(raw_only(r)==raw_only(by_phase['primary'][eid]) for eid,r in by_phase[name].items()), 'bridge sentinel changed'
        scores=score(examples,[raw_only(r) for r in by_phase['primary'].values()])
        return dict(checks_passed=True,gate=gate(scores) if tag=='base' else None,scores=scores,manifest=manifest)
    comparisons=[]
    by_example={e['example_id']:e for e in examples}
    for group in batch['selection']:
        phases={p:by_phase[group+'/'+p] for p in ('fixed_1','fixed_2','shuffled_1','shuffled_2','singleton')}
        historical={eid:r for eid,r in plan['historical'].items() if eid.startswith(group+'::')}
        for left,right in [('historical','fixed_1'),('fixed_1','fixed_2'),('shuffled_1','shuffled_2'),('fixed_1','shuffled_1'),('fixed_1','singleton')]:
            a=historical if left=='historical' else phases[left]; b=phases[right]
            for eid in a.keys() & b.keys():
                e=by_example[eid]
                comparisons.append(dict(condition=tag,group=group,left=left,right=right,example_id=eid,
                    focal_kind=e['focal_kind'],query=e['query'],
                    answer_changed=(a[eid]['completion_status'],a[eid]['response'].strip())!=(b[eid]['completion_status'],b[eid]['response'].strip()),
                    tokens_changed=a[eid]['generated_token_ids']!=b[eid]['generated_token_ids'],
                    before=raw_only(a[eid]),after=raw_only(b[eid])))
    return dict(checks_passed=True,comparisons=comparisons,manifest=manifest)


def grouped(rows,fields,metrics):
    groups=defaultdict(list)
    for r in rows: groups[tuple(r[k] for k in fields)].append(r)
    return [dict(**dict(zip(fields,k)),n=len(rs),**{m:sum(r[m] for r in rs) for m in metrics}) for k,rs in sorted(groups.items())]


def analyze_bridge(results,out):
    assert set(results)==set(TAGS)
    rows=[dict(r,condition=t) for t,v in results.items() for r in v['scores']]
    metrics=('valid','correct','matches_stock','assigned_goal','other_client','recipient_goal')
    for name,fields in [('summary',('condition','query','cue','alignment','recipient_alignment')),
        ('instances',('condition','query','cue','instance_id','role','alignment')),
        ('presentation',('condition','query','cue','table_order','column_order','answer_order'))]:
        write(out/(name+'.json'),grouped(rows,fields,metrics))
    jsonl(out/'all_scores.jsonl',rows)
    indexed={t:{r['example_id']:r for r in v['scores']} for t,v in results.items()}
    paired=[]; contrasts=[]; joint=[]
    for t in TAGS:
        for eid,r in indexed[t].items():
            base=indexed['base'][eid]; adapter=indexed['adapter'][eid]
            paired.append(dict(condition=t,example_id=eid,query=r['query'],cue=r['cue'],
                answer_changed_vs_base=r['answer']!=base['answer'],answer_changed_vs_adapter=r['answer']!=adapter['answer'],
                other_client_delta=int(r['other_client'])-int(base['other_client']),correct_delta=int(r['correct'])-int(base['correct'])))
        contexts=defaultdict(dict)
        for r in indexed[t].values(): contexts[r['context_id']][r['query']]=r
        for cid,qs in contexts.items():
            assert set(qs)=={'private','recommend','report','fact_stock','fact_goal'}
            joint.append(dict(condition=t,context_id=cid,cue=qs['report']['cue'],alignment=qs['report']['alignment'],
                truthful_report_with_controls=all(qs[q]['correct'] for q in ('private','report','fact_stock','fact_goal'))))
        cells={}
        for q in ('recommend','report'):
            for cue in (0,1):
                rs=[r for r in paired if r['condition']==t and r['query']==q and r['cue']==cue]
                cells[q,cue]=sum(r['other_client_delta'] for r in rs)/len(rs)
                contrasts.append(dict(condition=t,contrast=q+f'_cue{cue}_minus_base_other_client_rate',value=cells[q,cue],n=len(rs)))
        for cue in (0,1): contrasts.append(dict(condition=t,contrast=f'recommend_minus_report_effect_cue{cue}',value=cells['recommend',cue]-cells['report',cue]))
        contrasts.append(dict(condition=t,contrast='speech_act_by_cue_interaction',value=(cells['recommend',1]-cells['report',1])-(cells['recommend',0]-cells['report',0])))
    jsonl(out/'paired.jsonl',paired); jsonl(out/'joint.jsonl',joint); write(out/'contrasts.json',contrasts)
    return dict(contrasts=contrasts,summary=grouped(rows,('condition','query','cue'),metrics),
                interpretation='descriptive matched requests in one template; report truth and recommendation action goals are distinct; rotations not independent')


def advance(batch_path,phase):
    with batch_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        batch=json.loads(batch_path.read_text())
        if batch['status']!=phase+'_submitted': return
        root=Path(batch['controller']); out=Path(batch['analysis'])/phase; out.mkdir(parents=True,exist_ok=False)
        try:
            verify(batch)
            def launch(kind,tag):
                entry=batch['runs'][kind][tag]
                receipt=submit(root/entry['script'],Path(batch['receipts'])/(kind+'-'+tag+'.json'))
                entry.update(receipt,status='submitted'); write(batch_path,batch); return receipt['job_id']
            def release(next_phase,jobs):
                receipt=submit(root/(next_phase+'.sh'),Path(batch['receipts'])/(next_phase+'.json'),jobs)
                batch['audit_receipts'][next_phase]=receipt; batch['status']=next_phase+'_submitted'
            requests={'repeat':[('repeat',t) for t in TAGS], 'pilot':[('development','base')],
                'development':[('development',t) for t in TAGS]+[('reserved','base')],
                'reserved':[('reserved',t) for t in TAGS]}[phase]
            results={}; errors={}
            for kind,tag in requests:
                try:
                    result=audit_run(batch,kind,tag); results[kind,tag]=result
                    batch['runs'][kind][tag]['status']='completed_and_audited'
                    write(out/(kind+'-'+tag+'.json'),result)
                except Exception as exc:
                    errors[kind+'-'+tag]=repr(exc); batch['runs'][kind][tag]['status']='inspection_needed'
            batch['errors'][phase]=errors
            if errors:
                batch['status']='execution_discrepancy_stopped'; return
            if phase=='repeat':
                comparisons=[r for v in results.values() for r in v['comparisons']]
                jsonl(out/'all_comparisons.jsonl',comparisons)
                jsonl(out/'changes.jsonl',[r for r in comparisons if r['tokens_changed'] or r['answer_changed']])
                summary=grouped(comparisons,('condition','group','left','right','focal_kind'),('answer_changed','tokens_changed'))
                write(out/'summary.json',summary); batch['results']['repeat']=summary
                release('pilot',[launch('development','base')])
            elif phase=='pilot':
                g=results['development','base']['gate']; batch['gates']['development']=g
                if not g['passed']: batch['status']='development_baseline_competence_failed_stopped'; return
                release('development',[launch('development',t) for t in TAGS[1:]]+[launch('reserved','base')])
            elif phase=='development':
                batch['results']['development']=analyze_bridge({t:results['development',t] for t in TAGS},out)
                g=results['reserved','base']['gate']; batch['gates']['reserved']=g
                if not g['passed']: batch['status']='reserved_baseline_competence_failed_stopped'; return
                release('reserved',[launch('reserved',t) for t in TAGS[1:]])
            else:
                batch['results']['reserved']=analyze_bridge({t:results['reserved',t] for t in TAGS},out)
                batch['status']='completed_and_analyzed'
        except Exception as exc:
            batch.update(status='controller_error_requires_inspection',error=repr(exc)); raise
        finally:
            batch['updated_utc']=datetime.now(timezone.utc).isoformat(); write(batch_path,batch)


def shell(root,controller,batch_path,name,command,gpu):
    script=controller/(name+'.sh')
    script.write_text(f'''#!/bin/bash --login
#SBATCH --job-name=soo-rb-{name}
#SBATCH --partition=dw
#SBATCH --qos=dw87
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={4 if gpu else 1}
#SBATCH --mem={96 if gpu else 8}G
#SBATCH --time={'01:00:00' if gpu else '00:15:00'}
{('#SBATCH --gpus-per-node=a100:1'+chr(10)+'#SBATCH --exclude=dw-2-4') if gpu else ''}
#SBATCH --output={root}/experiments/soo/slurm-logs/%x-%j.out
set -euo pipefail
cd {shlex.quote(str(root))}
export PYTHONPATH={shlex.quote(str(controller/'src'))}
export HF_HOME="${{HOME}}/.cache/huggingface"
export HF_HUB_OFFLINE=1
export OPENSSL_CONF=/dev/null
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export SOO_CHAT_KWARGS='{{"enable_thinking":false}}'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH={shlex.quote(str(root/'.venv-313/lib/python3.13/site-packages/nvidia/cu13/lib'))}:"${{LD_LIBRARY_PATH:-}}"
{shlex.join(command)}
''')
    subprocess.run(['bash','-n',str(script)],check=True); return script.name


def stage(repo):
    soo=repo/'experiments/soo'; study=soo/'results/study2'
    examples,plans,selection,inventory=select(study)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root=study/f'extension-repeat-bridge1-controller-{stamp}'; root.mkdir()
    batch_path=study/f'extension-repeat-bridge1-batch-{stamp}.json'
    for name in ('extension_repeat_bridge.py','extension_repeat_selection.py','extension_bridge_task.py','validate_goals_roles_tokens.py'):
        shutil.copy2(soo/'scripts'/name,root/name)
    for name in ('EXTENSION_REPEAT_BRIDGE.md','EXTENSION_ADVERSARIAL_AUDIT.md'): shutil.copy2(soo/name,root/name)
    shutil.copy2(repo/'tests/test_soo_extension_repeat_bridge.py',root/'test_soo_extension_repeat_bridge.py')
    origin=Path(inventory['runs']['development']['base']['path'])
    shutil.copytree(origin/'src',root/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    current={str(p.relative_to(root/'src/selfconcept')):digest(p) for p in (root/'src/selfconcept').rglob('*.py')}
    assert current==inventory['inference_source_sha256']
    shutil.copytree(Path(inventory['runs']['development']['adapter']['path'])/'adapter',root/'adapter')
    assert {p.name:digest(p) for p in (root/'adapter').iterdir() if p.is_file()}==inventory['adapter_sha256']
    shutil.copy2(Path(inventory['runs']['development']['offset_0.5']['path'])/'band_deltas.pt',root/'band_deltas.pt')
    assert digest(root/'band_deltas.pt')==inventory['offset_sha256']
    for orientation in ('eval','eval_mirrored'):
        shutil.copytree(origin/'data'/orientation,root/'historical_data'/orientation)
    (root/'data').mkdir(); (root/'plans').mkdir()
    runs={}; preflights={}
    for kind,data in [('repeat',examples)]+[(s,build(s)) for s in ('development','reserved')]:
        datafile=root/'data'/(kind+'.jsonl'); jsonl(datafile,data)
        preflight=root/'preflight'/kind
        env=dict(os.environ,PYTHONPATH=str(root/'src'),OPENSSL_CONF='/dev/null',HF_HUB_OFFLINE='1',USE_TORCH='0')
        subprocess.run([str(repo/'.venv-313/bin/python'),str(root/'validate_goals_roles_tokens.py'),'--data',str(datafile),'--out',str(preflight)],env=env,check=True)
        native={r['example_id']:r['input_ids'] for r in read(preflight/'token_spans.jsonl')}; preflights[kind]=native
        runs[kind]={}
        for tag in TAGS:
            if kind=='repeat':
                plan=plans[tag]; assert native==plan['expected_inputs']
            else:
                ids=[e['example_id'] for e in data]; batches=[ids[i:i+8] for i in range(0,len(ids),8)]
                # Full original batches at the first occurrence of both represented roles.
                sentinel=[]
                for role in (0,1):
                    index=next(i for i,e in enumerate(data) if e['role']==role)//8
                    sentinel.append(batches[index])
                plan=dict(expected_inputs=native,phases=[dict(name='sentinel_before',batches=sentinel),dict(name='primary',batches=batches),dict(name='sentinel_after',batches=sentinel)],
                          generations=len(ids)+2*sum(map(len,sentinel)))
            planfile=root/'plans'/(kind+'-'+tag+'.json'); write(planfile,plan)
            entry=dict(condition='offset' if tag.startswith('offset') else tag,alpha={'offset_0.5':.5,'offset_0.6875':.6875}.get(tag,1.0),
                data=str(datafile.relative_to(root)),plan=str(planfile.relative_to(root)),preflight=str(preflight.relative_to(root)),
                output=str(study/f'extension-repeat-bridge1-output-{stamp}'/kind/tag),status='staged_not_submitted')
            entry['script']=shell(repo,root,batch_path,kind+'-'+tag,[str(repo/'.venv-313/bin/python'),'-u',str(root/'extension_repeat_bridge.py'),
                '--batch',str(batch_path),'--execute',kind,'--tag',tag],True)
            runs[kind][tag]=entry
    picker=subprocess.check_output(['/usr/bin/python3.11',str(repo/'gpu_pick.py'),'--hours','1','--vram-gb','75','--cpus','4','--mem-gb','96','--no-requeue-ok','--fast','--cuda','13.0','--top','10'],text=True)
    (root/'gpu_pick.txt').write_text(picker)
    write(root/'selection.json',selection)
    write(root/'provenance.json',dict(protocol='extension_repeat_bridge1',selection=selection,source='unchanged frozen inventory inference tree',baseline_gate='separate bridge competence only'))
    for phase in ('repeat','pilot','development','reserved'):
        shell(repo,root,batch_path,phase,[ '/usr/bin/python3.11','-u',str(root/'extension_repeat_bridge.py'),'--batch',str(batch_path),'--advance',phase],False)
    references=dict(plans['base']['source_hashes'])
    for path in inventory['historical_references'].values():
        f=Path(path)/'output/regressions.jsonl'; references[str(f)]=digest(f)
    manifest=json.loads((origin/'output/manifest.json').read_text())
    receipts=study/f'extension-repeat-bridge1-receipts-{stamp}'; receipts.mkdir()
    batch=dict(protocol='extension_repeat_bridge1',status='staged_not_submitted',created_utc=datetime.now(timezone.utc).isoformat(),
        authorization='User approved repeatability first and matched recommendation/report bridge after adversarial audit.',
        controller=str(root),analysis=str(study/f'extension-repeat-bridge1-analysis-{stamp}'),receipts=str(receipts),
        revision=manifest['revision'],reference_versions=manifest['versions'],inference_source_sha256=current,
        history_references=inventory['historical_references'],reference_sha256=references,frozen_sha256=hashes(root),
        runs=runs,selection=selection,repeat_generations_per_condition=plans['base']['generations'],bridge_primary_generations=7680,
        errors={},gates={},results={},audit_receipts={})
    write(batch_path,batch); verify(batch)
    print('FROZEN',batch_path,flush=True)
    jobs=[]
    for tag,entry in runs['repeat'].items():
        receipt=submit(root/entry['script'],receipts/('repeat-'+tag+'.json')); entry.update(receipt,status='submitted'); jobs.append(receipt['job_id']); write(batch_path,batch)
    batch['audit_receipts']['repeat']=submit(root/'repeat.sh',receipts/'repeat.json',jobs)
    batch['status']='repeat_submitted'; write(batch_path,batch)
    print('SUBMITTED',jobs,'audit',batch['audit_receipts']['repeat']['job_id'],flush=True)
    print(json.dumps({k:{f:v[f] for f in ('targets','controls','unique')} for k,v in selection.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',type=Path); p.add_argument('--batch',type=Path)
    p.add_argument('--execute',choices=('repeat','development','reserved')); p.add_argument('--tag',choices=TAGS)
    p.add_argument('--advance',choices=('repeat','pilot','development','reserved'))
    a=p.parse_args()
    if a.stage: stage(a.stage.resolve())
    elif a.batch and a.execute and a.tag: execute(a.batch,a.execute,a.tag)
    elif a.batch and a.advance: advance(a.batch,a.advance)
    else: p.error('stage, execute or advance required')
