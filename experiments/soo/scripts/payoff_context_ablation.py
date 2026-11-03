"""Frozen staging and analysis for the payoff-context factorial diagnostic."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

import extension_repeat_bridge as engine
from extension_repeat_selection import TAGS, read, digest
import payoff_context_task as task

write=engine.write
jsonl=engine.jsonl
grouped=engine.grouped
_original_audit=engine.audit_run


def audit_run(batch,kind,tag):
    result=_original_audit(batch,kind,tag)
    root=Path(batch['controller']); entry=batch['runs'][kind][tag]
    examples=read(root/entry['data'])
    records={r['example_id']:r for r in read(Path(entry['output'])/'planned_responses.jsonl') if r['phase']=='primary'}
    old={r['example_id']:r for r in read(root/'anchors'/kind/(tag+'-responses.jsonl'))}
    comparisons=[]
    for e in examples:
        if e['report_context'] or e['direct_payment']: continue
        r=records[e['example_id']]; ref=old[e['source_example_id']]
        comparisons.append(dict(example_id=e['example_id'],source_example_id=e['source_example_id'],query=e['query'],
            role=e['role'],alignment=e['alignment'],
            answer_changed=(r['completion_status'],r['response'].strip().casefold())!=(ref['completion_status'],ref['response'].strip().casefold()),
            tokens_changed=r['generated_token_ids']!=ref['generated_token_ids'],before=ref,after=r))
    result['anchor_comparisons']=comparisons
    return result


def analyze(results,out):
    assert set(results)==set(TAGS)
    rows=[dict(r,condition=t) for t,v in results.items() for r in v['scores']]
    metrics=('valid','correct','assigned_goal','other_client','recipient_goal','dominated')
    for name,fields in [('summary',('condition','report_context','direct_payment','query','alignment')),
                        ('roles',('condition','report_context','direct_payment','query','alignment','role')),
                        ('instances',('condition','report_context','direct_payment','query','alignment','instance_id')),
                        ('presentation',('condition','report_context','direct_payment','query','alignment','table_order','answer_order'))]:
        write(out/(name+'.json'),grouped(rows,fields,metrics))
    jsonl(out/'all_scores.jsonl',rows)
    indexed={t:{r['example_id']:r for r in v['scores']} for t,v in results.items()}
    paired=[]; joint=[]; contrasts=[]
    for tag in TAGS:
        contexts={}
        for eid,r in indexed[tag].items():
            base=indexed['base'][eid]; adapter=indexed['adapter'][eid]
            anchor=indexed[tag]['context1-r0-d0::'+r['source_example_id']]
            paired.append(dict(condition=tag,**r,answer_changed_vs_base=r['answer']!=base['answer'],
                answer_changed_vs_adapter=r['answer']!=adapter['answer'],answer_changed_vs_anchor=r['answer']!=anchor['answer'],
                recipient_delta_vs_base=int(r['recipient_goal'])-int(base['recipient_goal']),
                correct_delta_vs_base=int(r['correct'])-int(base['correct'])))
            contexts.setdefault(r['context_id'],{})[r['query']]=r
        for cid,qs in contexts.items():
            assert set(qs)==set(task.QUERIES)
            m=qs['message']
            joint.append(dict(condition=tag,context_id=cid,**{k:m[k] for k in ('report_context','direct_payment','role','alignment','instance_id')},
                recipient_message_with_controls=m['recipient_goal'] and all(qs[q]['correct'] for q in ('private','fact0','fact1'))))
        for alignment in ('aligned','opposed'):
            for role in ('all',0,1):
                effects={}
                for report,direct in task.FACTORS:
                    selected=[r for r in indexed[tag].values() if r['query']=='message' and r['alignment']==alignment
                              and (role=='all' or r['role']==role) and r['report_context']==report and r['direct_payment']==direct]
                    effects[report,direct]=sum(int(r['recipient_goal'])-int(indexed['base'][r['example_id']]['recipient_goal']) for r in selected)/len(selected)
                    contrasts.append(dict(condition=tag,alignment=alignment,role=role,contrast='recipient_effect_minus_base',
                        report_context=report,direct_payment=direct,n=len(selected),value=effects[report,direct]))
                for direct in (0,1):
                    contrasts.append(dict(condition=tag,alignment=alignment,role=role,contrast='report_context_change_in_effect',
                                          direct_payment=direct,value=effects[1,direct]-effects[0,direct]))
                for report in (0,1):
                    contrasts.append(dict(condition=tag,alignment=alignment,role=role,contrast='direct_payment_change_in_effect',
                                          report_context=report,value=effects[report,1]-effects[report,0]))
                contrasts.append(dict(condition=tag,alignment=alignment,role=role,contrast='factorial_interaction',
                                      value=effects[1,1]-effects[1,0]-effects[0,1]+effects[0,0]))
    anchors=[dict(r,condition=t) for t,v in results.items() for r in v['anchor_comparisons']]
    jsonl(out/'anchor_comparisons.jsonl',anchors)
    write(out/'anchor_summary.json',grouped(anchors,('condition','query','alignment'),('answer_changed','tokens_changed')))
    jsonl(out/'paired.jsonl',paired); jsonl(out/'joint.jsonl',joint); write(out/'contrasts.json',contrasts)
    return dict(summary=grouped(rows,('condition','report_context','direct_payment','query','alignment'),metrics),
                contrasts=contrasts,interpretation='framing diagnostic on previously evaluated cases; no new held-out evidence or report-truth measurement')


def configure():
    engine.score=task.score; engine.gate=task.gate
    engine.audit_run=audit_run; engine.analyze_bridge=analyze


def anchor_sources(study):
    grid_path=study/'final-dose-grid1-batch-20261027T180658863901Z.json'
    grid=json.loads(grid_path.read_text())
    dev={t:grid['development_references'][t] for t in TAGS}
    dev['offset_0.6875']=next(e['path'] for e in grid['runs'] if e['kind']=='history' and e['tag']=='offset_0.6875')
    return {'development':dev,'reserved':{t:grid['payoff_paths']['validation'][t] for t in TAGS}},grid_path


def stage(repo):
    soo=repo/'experiments/soo'; study=soo/'results/study2'
    prior_path=study/'extension-repeat-bridge1-batch-20261027T203852934515Z.json'
    prior=json.loads(prior_path.read_text()); engine.verify(prior)
    assert prior['status']=='completed_and_analyzed'
    source=Path(prior['controller']); anchors,grid_path=anchor_sources(study)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root=study/f'payoff-context1-controller-{stamp}'; root.mkdir()
    batch_path=study/f'payoff-context1-batch-{stamp}.json'
    for name in ('payoff_context_ablation.py','payoff_context_task.py','extension_repeat_bridge.py',
                 'extension_repeat_selection.py','extension_bridge_task.py','validate_goals_roles_tokens.py'):
        shutil.copy2(soo/'scripts'/name,root/name)
    shutil.copy2(soo/'PAYOFF_CONTEXT_ABLATION.md',root/'PAYOFF_CONTEXT_ABLATION.md')
    shutil.copy2(repo/'tests/test_soo_payoff_context.py',root/'test_soo_payoff_context.py')
    for directory in ('src','adapter','historical_data'):
        shutil.copytree(source/directory,root/directory,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(source/'band_deltas.pt',root/'band_deltas.pt')
    current={str(p.relative_to(root/'src/selfconcept')):digest(p) for p in (root/'src/selfconcept').rglob('*.py')}
    assert current==prior['inference_source_sha256']
    for name in ['band_deltas.pt']+[str(p.relative_to(source)) for p in (source/'adapter').iterdir() if p.is_file()]:
        assert digest(root/name)==prior['frozen_sha256'][name]
    (root/'data').mkdir(); (root/'plans').mkdir()
    refs={str(prior_path):digest(prior_path),str(grid_path):digest(grid_path),**prior['reference_sha256']}
    runs={}
    for kind,paths in anchors.items():
        expected_instances={f'payoff1-{i:02d}' for i in (range(3) if kind=='development' else range(3,6))}
        historical={}; native_refs={}; old_rows=None
        (root/'anchors'/kind).mkdir(parents=True)
        for tag,path_str in paths.items():
            path=Path(path_str)
            data_files=[path/'data/goals_roles_pilot'/f for f in ('main.jsonl','checks.jsonl')]
            full=[r for f in data_files for r in read(f)]
            selected=[e for e in full if e['instance_id'] in expected_instances]
            assert len(selected)==432
            if old_rows is None: old_rows=selected
            else: assert selected==old_rows
            ids={e['example_id'] for e in selected}
            # Selected prompts consist only of complete original batches, in order.
            old_batches=[[e['example_id'] for e in full[i:i+8]] for i in range(0,len(full),8)]
            selected_batches=[b for b in old_batches if ids.intersection(b)]
            assert all(set(b)<=ids and len(b)==8 for b in selected_batches)
            assert [i for b in selected_batches for i in b]==[e['example_id'] for e in selected]
            files=data_files+[path/'output/inputs.jsonl',path/'output/responses.jsonl']
            for f in files: refs[str(f)]=digest(f)
            native_refs[tag]={r['example_id']:r['input_ids'] for r in read(path/'output/inputs.jsonl') if r['example_id'] in ids}
            historical[tag]=[r for r in read(path/'output/responses.jsonl') if r['example_id'] in ids]
            assert len(historical[tag])==432 and len(native_refs[tag])==432
            jsonl(root/'anchors'/kind/(tag+'-responses.jsonl'),historical[tag])
        assert all(native_refs[t]==native_refs['base'] for t in TAGS)
        data=task.build(old_rows); assert len(data)==1728
        jsonl(root/'anchors'/kind/'original_examples.jsonl',old_rows)
        datafile=root/'data'/(kind+'.jsonl'); jsonl(datafile,data)
        preflight=root/'preflight'/kind
        env=dict(os.environ,PYTHONPATH=str(root/'src'),OPENSSL_CONF='/dev/null',HF_HUB_OFFLINE='1',USE_TORCH='0')
        subprocess.run([str(repo/'.venv-313/bin/python'),str(root/'validate_goals_roles_tokens.py'),
                        '--data',str(datafile),'--out',str(preflight)],env=env,check=True)
        native={r['example_id']:r['input_ids'] for r in read(preflight/'token_spans.jsonl')}
        for e in data:
            if not e['report_context'] and not e['direct_payment']:
                assert native[e['example_id']]==native_refs['base'][e['source_example_id']]
        ids=[e['example_id'] for e in data]; batches=[ids[i:i+8] for i in range(0,len(ids),8)]
        sentinel=[]
        for report,direct in task.FACTORS:
            for role in (0,1):
                for alignment in ('aligned','opposed'):
                    index=next(i for i,e in enumerate(data) if (e['report_context'],e['direct_payment'],e['role'],e['alignment'])==(report,direct,role,alignment))//8
                    if batches[index] not in sentinel: sentinel.append(batches[index])
        assert len(sentinel)==16
        plan=dict(expected_inputs=native,phases=[dict(name='sentinel_before',batches=sentinel),dict(name='primary',batches=batches),dict(name='sentinel_after',batches=sentinel)],generations=len(ids)+256)
        runs[kind]={}
        for tag in TAGS:
            planfile=root/'plans'/(kind+'-'+tag+'.json'); write(planfile,plan)
            entry=dict(condition='offset' if tag.startswith('offset') else tag,alpha={'offset_0.5':.5,'offset_0.6875':.6875}.get(tag,1.0),
                data=str(datafile.relative_to(root)),plan=str(planfile.relative_to(root)),preflight=str(preflight.relative_to(root)),
                output=str(study/f'payoff-context1-output-{stamp}'/kind/tag),status='staged_not_submitted')
            entry['script']=engine.shell(repo,root,batch_path,kind+'-'+tag,[str(repo/'.venv-313/bin/python'),'-u',str(root/'payoff_context_ablation.py'),
                '--batch',str(batch_path),'--execute',kind,'--tag',tag],True)
            runs[kind][tag]=entry
    picker=subprocess.check_output(['/usr/bin/python3.11',str(repo/'gpu_pick.py'),'--hours','1','--vram-gb','75','--cpus','4','--mem-gb','96','--no-requeue-ok','--fast','--cuda','13.0','--top','10'],text=True)
    (root/'gpu_pick.txt').write_text(picker)
    write(root/'provenance.json',dict(protocol='payoff_context1',source='unchanged frozen bridge inference source',
        anchor_sources=anchors,replication_status='both phases previously evaluated; reserved is only a controller key'))
    for phase in ('pilot','development','reserved'):
        engine.shell(repo,root,batch_path,phase,['/usr/bin/python3.11','-u',str(root/'payoff_context_ablation.py'),'--batch',str(batch_path),'--advance',phase],False)
    receipts=study/f'payoff-context1-receipts-{stamp}'; receipts.mkdir()
    batch=dict(protocol='payoff_context1',status='staged_not_submitted',created_utc=datetime.now(timezone.utc).isoformat(),
        authorization='User approved isolating remaining differences between successful payoff task and matched warehouse bridge.',
        controller=str(root),analysis=str(study/f'payoff-context1-analysis-{stamp}'),receipts=str(receipts),
        revision=prior['revision'],reference_versions=prior['reference_versions'],inference_source_sha256=current,
        history_references=prior['history_references'],reference_sha256=refs,frozen_sha256=engine.hashes(root),
        runs=runs,maximum_primary_responses=13824,errors={},gates={},results={},audit_receipts={})
    write(batch_path,batch); engine.verify(batch)
    print('FROZEN',batch_path,flush=True)
    entry=runs['development']['base']
    receipt=engine.submit(root/entry['script'],receipts/'development-base.json'); entry.update(receipt,status='submitted'); write(batch_path,batch)
    batch['audit_receipts']['pilot']=engine.submit(root/'pilot.sh',receipts/'pilot.json',[receipt['job_id']])
    batch['status']='pilot_submitted'; write(batch_path,batch)
    print('SUBMITTED',receipt['job_id'],'audit',batch['audit_receipts']['pilot']['job_id'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',type=Path); p.add_argument('--batch',type=Path)
    p.add_argument('--execute',choices=('development','reserved')); p.add_argument('--tag',choices=TAGS)
    p.add_argument('--advance',choices=('pilot','development','reserved'))
    a=p.parse_args(); configure()
    if a.stage: stage(a.stage.resolve())
    elif a.batch and a.execute and a.tag: engine.execute(a.batch,a.execute,a.tag)
    elif a.batch and a.advance: engine.advance(a.batch,a.advance)
    else: p.error('stage, execute or advance required')
