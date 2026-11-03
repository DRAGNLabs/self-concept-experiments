"""Freeze, submit and audit the report payment and objective-scope comparison."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

import extension_repeat_bridge as engine
from extension_repeat_selection import read, digest
import report_goal_task as task

TAGS=('base','adapter')
ANCHORS={'original':'components1-original::','full':'components1-full::',
         'paid_none':'components1-omit_report_objective::','unpaid_scoped':'components1-unpaid_report::'}
PAIRS=(('full','paid_none'),('unpaid_scoped','unpaid_none'),
       ('paid_reminder','paid_none'),('unpaid_reminder','unpaid_none'),
       ('full','paid_reminder'),('unpaid_scoped','unpaid_reminder'),
       ('full','unpaid_scoped'),('paid_none','unpaid_none'),('paid_reminder','unpaid_reminder'),
       ('original_reminder','original'),('paid_reminder','original_reminder'),
       ('unpaid_reminder','original_reminder'),('full','original'),
       ('paid_none','original'),('unpaid_none','original'))
INTERACTIONS=(('payment_by_scoped_objective',('full','paid_none','unpaid_scoped','unpaid_none')),
              ('payment_by_unscoped_reminder',('paid_reminder','paid_none','unpaid_reminder','unpaid_none')),
              ('payment_by_scope_wording',('full','paid_reminder','unpaid_scoped','unpaid_reminder')))
write=engine.write
jsonl=engine.jsonl
grouped=engine.grouped
_original_audit=engine.audit_run


def audit_run(batch,kind,tag):
    result=_original_audit(batch,kind,tag)
    root=Path(batch['controller']); entry=batch['runs'][kind][tag]
    records={r['example_id']:r for r in read(Path(entry['output'])/'planned_responses.jsonl') if r['phase']=='primary'}
    old={r['example_id']:r for r in read(root/'anchors'/kind/(tag+'-responses.jsonl'))}
    comparisons=[]
    for e in read(root/entry['data']):
        if e['variant'] not in ANCHORS: continue
        r=records[e['example_id']]; ref=old[ANCHORS[e['variant']]+e['source_example_id']]
        comparisons.append(dict(example_id=e['example_id'],source_example_id=e['source_example_id'],variant=e['variant'],query=e['query'],
            role=e['role'],alignment=e['alignment'],answer_changed=(r['completion_status'],r['response'].strip().casefold())!=(ref['completion_status'],ref['response'].strip().casefold()),
            tokens_changed=r['generated_token_ids']!=ref['generated_token_ids'],before=ref,after=r))
    result['anchor_comparisons']=comparisons
    return result


def analyze(results,out):
    assert set(results)==set(TAGS)
    rows=[dict(r,condition=t) for t,v in results.items() for r in v['scores']]
    metrics=('valid','correct','assigned_goal','other_client','recipient_goal','dominated')
    for name,fields in [('summary',('condition','variant','query','alignment')),
                        ('roles',('condition','variant','query','alignment','role')),
                        ('instances',('condition','variant','query','alignment','instance_id')),
                        ('presentation',('condition','variant','query','alignment','table_order','answer_order'))]:
        write(out/(name+'.json'),grouped(rows,fields,metrics))
    jsonl(out/'all_scores.jsonl',rows)
    indexed={t:{r['example_id']:r for r in v['scores']} for t,v in results.items()}
    paired=[]; joint=[]; contrasts=[]
    for tag in TAGS:
        contexts={}
        for eid,r in indexed[tag].items():
            base=indexed['base'][eid]
            original=indexed[tag]['reportgoal1-original::'+r['source_example_id']]
            full=indexed[tag]['reportgoal1-full::'+r['source_example_id']]
            paired.append(dict(condition=tag,**r,answer_changed_vs_base=r['answer']!=base['answer'],
                answer_changed_vs_original=r['answer']!=original['answer'],answer_changed_vs_full=r['answer']!=full['answer'],
                recipient_delta_vs_base=int(r['recipient_goal'])-int(base['recipient_goal']),correct_delta_vs_base=int(r['correct'])-int(base['correct'])))
            contexts.setdefault(r['context_id'],{})[r['query']]=r
        for cid,qs in contexts.items():
            assert set(qs)==set(task.QUERIES)
            m=qs['message']
            joint.append(dict(condition=tag,context_id=cid,**{k:m[k] for k in ('variant','role','alignment','instance_id')},
                recipient_message_with_controls=m['recipient_goal'] and all(qs[q]['correct'] for q in ('private','fact0','fact1'))))
    for alignment in ('aligned','opposed'):
        for role in ('all',0,1):
            cells={}
            for variant in task.VARIANTS:
                counts={}
                for tag in TAGS:
                    selected=[r for r in indexed[tag].values() if r['query']=='message' and r['alignment']==alignment and r['variant']==variant and (role=='all' or r['role']==role)]
                    assert selected
                    counts[tag]=sum(r['recipient_goal'] for r in selected)/len(selected)
                cells[variant]=counts
                contrasts.append(dict(alignment=alignment,role=role,contrast='adapter_minus_base',variant=variant,n=len(selected),
                    base_rate=counts['base'],adapter_rate=counts['adapter'],value=counts['adapter']-counts['base']))
            for left,right in PAIRS:
                base_change=cells[left]['base']-cells[right]['base']
                adapter_change=cells[left]['adapter']-cells[right]['adapter']
                contrasts.append(dict(alignment=alignment,role=role,contrast=left+'_minus_'+right,left=left,right=right,
                                      base_change=base_change,adapter_change=adapter_change,value=adapter_change-base_change))
            for name,(a,b,c,d) in INTERACTIONS:
                values={t:cells[a][t]-cells[b][t]-cells[c][t]+cells[d][t] for t in TAGS}
                contrasts.append(dict(alignment=alignment,role=role,contrast=name,
                    base_change=values['base'],adapter_change=values['adapter'],value=values['adapter']-values['base']))
    anchors=[dict(r,condition=t) for t,v in results.items() for r in v['anchor_comparisons']]
    jsonl(out/'anchor_comparisons.jsonl',anchors)
    write(out/'anchor_summary.json',grouped(anchors,('condition','variant','query','alignment'),('answer_changed','tokens_changed')))
    jsonl(out/'paired.jsonl',paired);jsonl(out/'joint.jsonl',joint);write(out/'contrasts.json',contrasts)
    return dict(summary=grouped(rows,('condition','variant','query','alignment'),metrics),contrasts=contrasts,
                interpretation='payment by objective-wording factorial and reminder controls on previously evaluated cases; not mechanism proof or honesty measurement')


def configure():
    engine.TAGS=TAGS; engine.score=task.score; engine.gate=task.gate
    engine.audit_run=audit_run; engine.analyze_bridge=analyze


def stage(repo):
    soo=repo/'experiments/soo';study=soo/'results/study2'
    parent_path=study/'report-components1-batch-20261027T220002622496Z.json'
    parent=json.loads(parent_path.read_text());engine.verify(parent)
    assert parent['status']=='completed_and_analyzed'
    source=Path(parent['controller'])
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root=study/f'report-goal-scope1-controller-{stamp}';root.mkdir()
    batch_path=study/f'report-goal-scope1-batch-{stamp}.json'
    for name in ('report_goal_scope.py','report_goal_task.py','report_components_task.py','payoff_context_task.py','extension_repeat_bridge.py',
                 'extension_repeat_selection.py','extension_bridge_task.py','validate_goals_roles_tokens.py'):
        shutil.copy2(soo/'scripts'/name,root/name)
    # Parent prompt constructor and execution machinery must remain byte-identical.
    for name in ('report_components_task.py','payoff_context_task.py','extension_repeat_bridge.py','extension_repeat_selection.py','extension_bridge_task.py','validate_goals_roles_tokens.py'):
        assert digest(root/name)==parent['frozen_sha256'][name]
    shutil.copy2(soo/'REPORT_GOAL_SCOPE.md',root/'REPORT_GOAL_SCOPE.md')
    shutil.copy2(repo/'tests/test_soo_report_goal_scope.py',root/'test_soo_report_goal_scope.py')
    for directory in ('src','adapter','historical_data'):
        shutil.copytree(source/directory,root/directory,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(source/'band_deltas.pt',root/'band_deltas.pt')
    current={str(p.relative_to(root/'src/selfconcept')):digest(p) for p in (root/'src/selfconcept').rglob('*.py')}
    assert current==parent['inference_source_sha256']
    for name in ['band_deltas.pt']+[str(p.relative_to(source)) for p in (source/'adapter').iterdir() if p.is_file()]:
        assert digest(root/name)==parent['frozen_sha256'][name]
    (root/'data').mkdir();(root/'plans').mkdir()
    refs={str(parent_path):digest(parent_path),**parent['reference_sha256']};runs={}
    for kind in ('development','reserved'):
        original_file=source/'anchors'/kind/'original_examples.jsonl'
        originals=read(original_file);assert len(originals)==432
        refs[str(original_file)]=digest(original_file)
        data=task.build(originals);assert len(data)==3456
        (root/'anchors'/kind).mkdir(parents=True)
        jsonl(root/'anchors'/kind/'original_examples.jsonl',originals)
        datafile=root/'data'/(kind+'.jsonl');jsonl(datafile,data)
        native_refs=None
        for tag in TAGS:
            pe=parent['runs'][kind][tag]
            oldplan=source/pe['plan'];olddata=source/pe['data'];responses=Path(pe['output'])/'planned_responses.jsonl'
            for f in (oldplan,olddata,responses):refs[str(f)]=digest(f)
            pdata={e['example_id']:e for e in read(olddata)}
            plan=json.loads(oldplan.read_text())
            if native_refs is None:native_refs=plan['expected_inputs']
            else:assert native_refs==plan['expected_inputs']
            wanted=[]
            for variant,prefix in ANCHORS.items():
                selected=[e for e in data if e['variant']==variant]
                ids=[prefix+e['source_example_id'] for e in selected];wanted+=ids
                for e,eid in zip(selected,ids):assert e['prompt']==pdata[eid]['prompt']
                old_batches=next(p['batches'] for p in plan['phases'] if p['name']=='primary')
                chosen=[b for b in old_batches if set(ids).intersection(b)]
                assert all(set(b)<=set(ids) and len(b)==8 for b in chosen)
                assert [eid for b in chosen for eid in b]==ids
            records=[r for r in read(responses) if r['phase']=='primary' and r['example_id'] in set(wanted)]
            assert len(records)==1728 and len({r['example_id'] for r in records})==1728
            jsonl(root/'anchors'/kind/(tag+'-responses.jsonl'),records)
        preflight=root/'preflight'/kind
        env=dict(os.environ,PYTHONPATH=str(root/'src'),OPENSSL_CONF='/dev/null',HF_HUB_OFFLINE='1',USE_TORCH='0')
        subprocess.run([str(repo/'.venv-313/bin/python'),str(root/'validate_goals_roles_tokens.py'),'--data',str(datafile),'--out',str(preflight)],env=env,check=True)
        native={r['example_id']:r['input_ids'] for r in read(preflight/'token_spans.jsonl')}
        for e in data:
            if e['variant'] in ANCHORS:assert native[e['example_id']]==native_refs[ANCHORS[e['variant']]+e['source_example_id']]
        ids=[e['example_id'] for e in data];batches=[ids[i:i+8] for i in range(0,len(ids),8)]
        sentinel=[]
        for variant in task.VARIANTS:
            for role in (0,1):
                for alignment in ('aligned','opposed'):
                    index=next(i for i,e in enumerate(data) if (e['variant'],e['role'],e['alignment'])==(variant,role,alignment))//8
                    if batches[index] not in sentinel:sentinel.append(batches[index])
        assert len(sentinel)==32
        plan=dict(expected_inputs=native,phases=[dict(name='sentinel_before',batches=sentinel),dict(name='primary',batches=batches),dict(name='sentinel_after',batches=sentinel)],generations=len(ids)+512)
        runs[kind]={}
        for tag in TAGS:
            planfile=root/'plans'/(kind+'-'+tag+'.json');write(planfile,plan)
            entry=dict(condition=tag,alpha=1.0,data=str(datafile.relative_to(root)),plan=str(planfile.relative_to(root)),preflight=str(preflight.relative_to(root)),
                output=str(study/f'report-goal-scope1-output-{stamp}'/kind/tag),status='staged_not_submitted')
            entry['script']=engine.shell(repo,root,batch_path,kind+'-'+tag,[str(repo/'.venv-313/bin/python'),'-u',str(root/'report_goal_scope.py'),'--batch',str(batch_path),'--execute',kind,'--tag',tag],True)
            runs[kind][tag]=entry
    picker=subprocess.check_output(['/usr/bin/python3.11',str(repo/'gpu_pick.py'),'--hours','1','--vram-gb','75','--cpus','4','--mem-gb','96','--no-requeue-ok','--fast','--cuda','13.0','--top','10'],text=True)
    (root/'gpu_pick.txt').write_text(picker)
    write(root/'provenance.json',dict(protocol='report_goal_scope1',parent=str(parent_path),variants=task.VARIANTS,
        source='unchanged frozen report-components inference source',replication_status='both phases previously evaluated; reserved only a controller key'))
    for phase in ('pilot','development','reserved'):
        engine.shell(repo,root,batch_path,phase,['/usr/bin/python3.11','-u',str(root/'report_goal_scope.py'),'--batch',str(batch_path),'--advance',phase],False)
    receipts=study/f'report-goal-scope1-receipts-{stamp}';receipts.mkdir()
    batch=dict(protocol='report_goal_scope1',status='staged_not_submitted',created_utc=datetime.now(timezone.utc).isoformat(),
        authorization='User approved recording all four next steps and proceeding with the bounded payment/objective factorial and general-reminder comparison.',
        controller=str(root),analysis=str(study/f'report-goal-scope1-analysis-{stamp}'),receipts=str(receipts),
        revision=parent['revision'],reference_versions=parent['reference_versions'],inference_source_sha256=current,
        history_references=parent['history_references'],reference_sha256=refs,frozen_sha256=engine.hashes(root),runs=runs,
        maximum_primary_responses=13824,errors={},gates={},results={},audit_receipts={})
    write(batch_path,batch);engine.verify(batch);print('FROZEN',batch_path,flush=True)
    entry=runs['development']['base'];receipt=engine.submit(root/entry['script'],receipts/'development-base.json')
    entry.update(receipt,status='submitted');write(batch_path,batch)
    batch['audit_receipts']['pilot']=engine.submit(root/'pilot.sh',receipts/'pilot.json',[receipt['job_id']])
    batch['status']='pilot_submitted';write(batch_path,batch)
    print('SUBMITTED',receipt['job_id'],'audit',batch['audit_receipts']['pilot']['job_id'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',type=Path);p.add_argument('--batch',type=Path)
    p.add_argument('--execute',choices=('development','reserved'));p.add_argument('--tag',choices=TAGS)
    p.add_argument('--advance',choices=('pilot','development','reserved'))
    a=p.parse_args();configure()
    if a.stage:stage(a.stage.resolve())
    elif a.batch and a.execute and a.tag:engine.execute(a.batch,a.execute,a.tag)
    elif a.batch and a.advance:engine.advance(a.batch,a.advance)
    else:p.error('stage, execute or advance required')
