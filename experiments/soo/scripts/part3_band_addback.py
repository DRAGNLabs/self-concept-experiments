"""Frozen crossed q/outside-v add-back diagnostic. BYU ORC policy applies.

Preparation/tests/analysis use CPU Slurm; inference uses allocated A100s.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

PARENT = 'part3-band1-batch-20261029T200515439449Z.json'
CONDITIONS = ('full', 'band', 'add_q', 'add_v')
SEEDS = ('seed0', 'seed1', 'seed2')


def modules(adapter):
    with (adapter/'adapter_model.safetensors').open('rb') as f:
        size, = struct.unpack('<Q', f.read(8)); header = json.loads(f.read(size))
    return sorted(n.removesuffix('.lora_A.weight') for n in header if n.endswith('.lora_A.weight'))


def selection(names, condition):
    assert condition in CONDITIONS
    selected = []
    for n in names:
        layer = int(re.search(r'\.layers\.(\d+)\.', n).group(1))
        leaf = n.rsplit('.', 1)[-1]; assert leaf in ('q_proj', 'v_proj')
        band = leaf == 'v_proj' and 20 <= layer <= 31
        if condition == 'full' or band or (condition == 'add_q' and leaf == 'q_proj') or (condition == 'add_v' and leaf == 'v_proj'):
            selected.append(n)
    return sorted(selected)


def restrict(model, all_names, expected, condition):
    from peft.tuners.lora.layer import LoraLayer
    found = [(n, m) for n, m in model.named_modules() if isinstance(m, LoraLayer)]
    assert sorted(n for n, _ in found) == all_names
    wanted = set(expected)
    for n, m in found:
        if n not in wanted:
            parent, _, leaf = n.rpartition('.')
            setattr(model.get_submodule(parent), leaf, m.get_base_layer())
    actual = sorted(n for n, m in model.named_modules() if isinstance(m, LoraLayer))
    assert actual == expected
    return dict(factorial_condition=condition, actual_module_names=actual,
                found=len(found), kept=len(actual), removed=len(found)-len(actual),
                unmerged=True)


def self_test():
    import torch
    from torch import nn
    from peft import LoraConfig, get_peft_model
    from peft.tuners.lora.layer import LoraLayer
    torch.set_num_threads(1)
    counts = {}
    for condition in CONDITIONS:
        torch.manual_seed(702)
        m = nn.Module(); m.model = nn.Module(); m.model.layers = nn.ModuleList([nn.Module() for _ in range(34)])
        for i, layer in enumerate(m.model.layers):
            layer.q_proj = nn.Linear(7, 5, bias=False)
            if i not in (23, 29): layer.v_proj = nn.Linear(7, 5, bias=False)
        m = get_peft_model(m, LoraConfig(r=2, lora_alpha=4, lora_dropout=.1,
                                       target_modules=['q_proj', 'v_proj'])).eval()
        found = [(n, x) for n, x in m.named_modules() if isinstance(x, LoraLayer)]
        names = sorted(n for n, _ in found); keep = selection(names, condition)
        x = torch.randn(3, 7); expected = {}
        with torch.no_grad():
            for n, layer in found:
                layer.lora_B['default'].weight.normal_()
                expected[n] = layer(x).clone() if n in keep else layer.get_base_layer()(x).clone()
            result = restrict(m, names, keep, condition)
            for n, value in expected.items(): assert torch.equal(m.get_submodule(n)(x), value), (condition, n)
        counts[condition] = result['kept']
    assert counts == dict(full=66, band=10, add_q=44, add_v=32)
    return dict(passed=True, exact_projection_checks=264, retained_counts=counts)


def stage(repo):
    study = repo/'experiments/soo/results/study2'
    parent_path = study/PARENT; parent = json.loads(parent_path.read_text())
    sys.path.insert(0, parent['controller'])
    import part3_band as prior
    engine = prior.engine
    engine.verify(parent); assert parent['status'] == 'completed_band_not_qualified'
    checks = self_test(); print('TESTS', json.dumps(checks), flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root = study/f'part3-addback1-controller-{stamp}'; root.mkdir()
    path = study/f'part3-addback1-batch-{stamp}.json'; source = Path(parent['controller'])
    for p in source.glob('*.py'):
        if not p.name.startswith('test_'): shutil.copy2(p, root/p.name)
    shutil.copy2(__file__, root/'part3_band_addback.py')
    shutil.copy2(repo/'experiments/soo/PART3_ADDBACK.md', root/'PART3_ADDBACK.md')
    shutil.copytree(source/'src', root/'src', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    engine.write(root/'preparation_tests.json', checks)
    examples = []; native = {}; batches = []; sentinels = []
    for split in prior.SPLITS:
        rows = [e for e in prior.read(source/'data'/split/'band.jsonl') if e['cohort']=='anchor']
        assert len(rows)==864; examples.extend(rows); ids = {e['example_id'] for e in rows}
        plan = json.loads((source/'data'/split/'band-plan.json').read_text())
        for phase in plan['phases']:
            if phase['name'] not in ('primary','sentinel_before'): continue
            chosen = [b for b in phase['batches'] if ids.intersection(b)]
            assert all(len(b)==8 and set(b)<=ids for b in chosen)
            (batches if phase['name']=='primary' else sentinels).extend(chosen)
        native.update({eid:v for eid,v in plan['expected_inputs'].items() if eid in ids})
    assert [eid for b in batches for eid in b] == [e['example_id'] for e in examples]
    engine.jsonl(root/'historical.jsonl', examples)
    env = dict(os.environ, PYTHONPATH=str(root/'src'), USE_TORCH='0', HF_HUB_OFFLINE='1', OPENSSL_CONF='/dev/null')
    subprocess.run([str(repo/'.venv-313/bin/python'),str(root/'validate_goals_roles_tokens.py'),
        '--data',str(root/'historical.jsonl'),'--out',str(root/'preflight')],check=True,env=env)
    assert {e['example_id']:e['input_ids'] for e in prior.read(root/'preflight/token_spans.jsonl')} == native
    engine.write(root/'plan.json', prior.plan_for(batches,sentinels,native))
    parent_scores = prior.read(source/'parent_scores.jsonl')
    engine.jsonl(root/'parent_scores.jsonl',[r for r in parent_scores if r['cohort']=='anchor'])
    original = json.loads(Path(parent['parent']).read_text())
    children = {}; runs = {}
    for seed in SEEDS:
        old = json.loads(Path(parent['children'][seed]).read_text()); oldroot = Path(old['controller'])
        childroot = root/seed; childroot.mkdir()
        for name in ('adapter','historical_data'): shutil.copytree(oldroot/name,childroot/name)
        prior.verify_adapter(childroot/'adapter',old['adapter_sha256'],int(seed[-1]))
        names = modules(childroot/'adapter'); assert len(names)==110
        expected = {c:selection(names,c) for c in CONDITIONS}
        assert {c:len(n) for c,n in expected.items()} == dict(full=110,band=10,add_q=70,add_v=50)
        engine.write(childroot/'provenance.json',dict(protocol='part3_addback1',parent=str(parent_path),
            seed=int(seed[-1]),all_historical_cases=True,expected_module_names=expected))
        entries = {}
        for condition in CONDITIONS:
            entries[condition] = dict(adapter=dict(condition='adapter' if condition=='full' else 'band',alpha=1.0,
                data=str(root/'historical.jsonl'),plan=str(root/'plan.json'),preflight=str(root/'preflight'),
                output=str(study/f'part3-addback1-output-{stamp}'/seed/condition)))
            if condition not in ('full','band'): continue
            refs = childroot/'references'/condition; refs.mkdir(parents=True)
            raw = []; regressions = None
            for split in prior.SPLITS:
                previous = Path(original['runs'][seed+'-'+split]['output']) if condition=='full' else Path(old['runs']['band-'+split]['adapter']['output'])
                wanted = {e['example_id'] for e in examples if e['split']==split}
                raw.extend(engine.raw_only(r) for r in prior.read(previous/'planned_responses.jsonl')
                           if r['phase']=='primary' and r['example_id'] in wanted)
                current = prior.read(previous/'regressions.jsonl')
                if regressions is not None: assert regressions==current
                regressions=current
            assert len(raw)==1728
            engine.jsonl(refs/'primary.jsonl',raw); engine.jsonl(refs/'regressions.jsonl',regressions)
        child = dict(controller=str(childroot),seed=int(seed[-1]),revision=old['revision'],
            reference_versions=old['reference_versions'],inference_source_sha256=old['inference_source_sha256'],
            adapter_sha256=old['adapter_sha256'],all_modules=names,expected_modules=expected,
            reference_sha256={},frozen_sha256=engine.hashes(childroot),runs=entries)
        cp = root/(seed+'-batch.json'); engine.write(cp,child); children[seed]=str(cp)
        script = prior.scheduling.shell(repo,root,path,seed,[str(repo/'.venv-313/bin/python'),'-u',
            str(root/'part3_band_addback.py'),'--batch',str(path),'--execute',seed],True)
        p=root/script; p.write_text(p.read_text().replace('soo-tb2-','soo-p3ab-'))
        runs[seed]=dict(script=script,status='staged_not_submitted')
    script=prior.scheduling.shell(repo,root,path,'final',['/usr/bin/python3.11','-u',str(root/'part3_band_addback.py'),
        '--batch',str(path),'--finish'],False)
    p=root/script; p.write_text(p.read_text().replace('soo-tb2-','soo-p3ab-'))
    receipts=study/f'part3-addback1-receipts-{stamp}';receipts.mkdir()
    batch=dict(protocol='part3_addback1',controller=str(root),parent=str(parent_path),
        authorization='Existing-data audit and Part 3; user requested investigation of failed qualification.',
        status='staged_not_submitted',children=children,runs=runs,receipts=str(receipts),
        analysis=str(study/f'part3-addback1-analysis-{stamp}'),errors={},results={},
        primary=20736,new_condition_primary=10368,reference_primary=10368,
        frozen_sha256=engine.hashes(root),reference_sha256={str(parent_path):prior.digest(parent_path)})
    engine.write(path,batch);engine.verify(batch)
    print('FROZEN',path,flush=True)
    return path


def execute_kind(path, seed, condition):
    import part3_band as prior
    batch=json.loads(path.read_text());prior.engine.verify(batch)
    cp=Path(batch['children'][seed]);child=json.loads(cp.read_text())
    from selfconcept.soo import lora_subset
    def subset(model, layers, modules):
        assert (layers,modules)==('range:20-31','v_proj')
        return restrict(model,child['all_modules'],child['expected_modules'][condition],condition)
    lora_subset.apply_adapter_subset=subset
    prior.engine.TAGS=('adapter',)
    prior.engine.execute(cp,condition,'adapter')
    manifest_path=Path(child['runs'][condition]['adapter']['output'])/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    manifest['factorial_condition']=condition
    manifest['expected_adapter_modules']=child['expected_modules'][condition]
    prior.engine.write(manifest_path,manifest)


def audit_run(child, condition):
    import part3_band as prior
    # This is the inherited audit with explicit factorial-subset assertions.
    engine=prior.engine;read=prior.read;digest=prior.digest
    engine.verify(child);entry=child['runs'][condition]['adapter'];out=Path(entry['output'])
    plan=json.loads(Path(entry['plan']).read_text());examples=read(entry['data'])
    assert json.loads((out/'execution_complete.json').read_text())==dict(phases=3,generations=plan['generations'])
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['factorial_condition']==condition
    assert manifest['expected_adapter_modules']==child['expected_modules'][condition]
    assert manifest['source_sha256']==child['inference_source_sha256']
    assert manifest['condition']==entry['condition'] and manifest['offset_alpha']==1
    assert manifest['revision']==child['revision'] and manifest['model']=='google/gemma-4-31B-it'
    assert manifest['versions']==child['reference_versions'] and 'A100' in manifest['hardware']['gpu']
    assert manifest['dtype']=='bfloat16' and manifest['attention']=='sdpa'
    assert manifest['chat_kwargs']=={'enable_thinking':False} and manifest['do_sample'] is False
    assert manifest['batch_size']==8 and manifest['max_new_tokens']==32
    assert manifest['example_ids']==[e['example_id'] for e in examples]
    for p,v in manifest['artifact_sha256'].items():assert digest(p)==v
    assert manifest['adapter']==str(Path(child['controller'])/'adapter')
    inputs=read(out/'inputs.jsonl');assert inputs==read(Path(entry['preflight'])/'token_spans.jsonl')
    native={r['example_id']:r['input_ids'] for r in inputs};assert native==plan['expected_inputs']
    assert manifest['prompt_ids_sha256']==prior.hashlib.sha256(json.dumps([r['input_ids'] for r in inputs]).encode()).hexdigest()
    intervention=json.loads((out/'intervention.json').read_text())
    if condition=='full':assert intervention['subset'] is None
    else:
        s=intervention['subset'];expected=child['expected_modules'][condition]
        assert s==dict(factorial_condition=condition,actual_module_names=expected,
            found=len(child['all_modules']),kept=len(expected),removed=len(child['all_modules'])-len(expected),unmerged=True)
    expected=[(p['name'],i,ids,eid) for p in plan['phases'] for i,ids in enumerate(p['batches']) for eid in ids]
    responses=read(out/'planned_responses.jsonl');assert len(responses)==len(expected)==plan['generations']
    phases={p['name']:{} for p in plan['phases']}
    for r,(phase,i,ids,eid) in zip(responses,expected):
        assert (r['phase'],r['batch_index'],r['batch_members'],r['example_id'])==(phase,i,ids,eid)
        assert r['padded_width']==max(len(native[e]) for e in ids)
        assert r['prompt_tokens']==len(native[eid]) and r['generated_tokens']==len(r['generated_token_ids'])<=32
        assert r['completion_status'] in ('complete','truncated') and eid not in phases[phase]
        phases[phase][eid]=engine.raw_only(r)
    for phase in ('sentinel_before','sentinel_after'):assert all(r==phases['primary'][eid] for eid,r in phases[phase].items())
    regressions=read(out/'regressions.jsonl')
    assert len(regressions)==32 and len({(r['orientation'],r['scenario'],r['example_id']) for r in regressions})==32
    if condition in ('full','band'):
        refs=Path(child['controller'])/'references'/condition
        assert list(phases['primary'].values())==read(refs/'primary.jsonl'),'parent replay changed'
        assert regressions==read(refs/'regressions.jsonl')
    return dict(checks_passed=True,scores=prior.task.score(examples,list(phases['primary'].values())),
                manifest=manifest,intervention=intervention,regressions=regressions)


def finish(path):
    import part3_band as prior
    batch=json.loads(path.read_text());prior.engine.verify(batch)
    out=Path(batch['analysis']);out.mkdir(exist_ok=True);rows=[];gates={}
    base=[r for r in prior.read(Path(batch['controller'])/'parent_scores.jsonl') if r['condition']=='base']
    try:
        for seed,p in batch['children'].items():
            child=json.loads(Path(p).read_text());gates[seed]={}
            for condition in CONDITIONS:
                result=audit_run(child,condition)
                prior.engine.write(out/(seed+'-'+condition+'-audit.json'),result)
                scores=result['scores'];gates[seed][condition]=prior.qualification(scores,base)
                rows.extend(dict(r,seed=seed,condition=condition) for r in scores)
        assert len(rows)==20736
        prior.engine.jsonl(out/'scores.jsonl',rows)
        fields=('seed','condition','split','variant','query','alignment')
        metrics=('valid','correct','recipient_goal','assigned_goal','dominated')
        summary=prior.existing.aggregate(rows,fields,metrics)
        prior.engine.write(out/'summary.json',summary)
        prior.engine.write(out/'role_worlds.json',prior.existing.aggregate(rows,fields+('role','instance_id'),metrics))
        index={(r['seed'],r['condition'],r['example_id']):r for r in rows};paired=[]
        for r in rows:
            if r['condition']!='band':continue
            cells={c:index[r['seed'],c,r['example_id']] for c in CONDITIONS}
            paired.append(dict(seed=r['seed'],example_id=r['example_id'],split=r['split'],variant=r['variant'],
                query=r['query'],alignment=r['alignment'],role=r['role'],instance_id=r['instance_id'],
                recipient_interaction=int(cells['full']['recipient_goal'])-int(cells['add_q']['recipient_goal'])-
                    int(cells['add_v']['recipient_goal'])+int(cells['band']['recipient_goal']),
                **{c+'_answer':s['answer'] for c,s in cells.items()},
                **{c+'_matches_full':s['answer']==cells['full']['answer'] for c,s in cells.items()},
                **{c+'_correct':s['correct'] for c,s in cells.items()}))
        prior.engine.jsonl(out/'paired.jsonl',paired)
        prior.engine.write(out/'paired_summary.json',prior.existing.aggregate(paired,
            ('seed','split','variant','query','alignment','role'),('recipient_interaction',)+
            tuple(c+'_matches_full' for c in CONDITIONS)+tuple(c+'_correct' for c in CONDITIONS)))
        batch['status']='completed_and_analyzed'
        batch['results']=dict(primary=len(rows),valid=sum(r['valid'] for r in rows),gates=gates,
            all_seed_qualified={c:all(gates[s][c]['passed'] for s in SEEDS) for c in CONDITIONS},patches_submitted=False)
        prior.engine.write(out/'qualification.json',batch['results'])
    except Exception as exc:
        batch['status']='audit_failed_stopped';batch['errors']['final']=repr(exc);prior.engine.write(path,batch);raise
    prior.engine.write(path,batch);print(json.dumps(batch['results'],indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path);p.add_argument('--batch',type=Path)
    p.add_argument('--submit',action='store_true');p.add_argument('--execute',choices=SEEDS);p.add_argument('--kind',choices=CONDITIONS);p.add_argument('--finish',action='store_true')
    args=p.parse_args()
    if args.stage:
        path=stage(args.stage.resolve())
        if args.submit:
            import part3_band as prior
            prior.submit(path)
    elif args.execute and args.kind:execute_kind(args.batch,args.execute,args.kind)
    elif args.execute:
        import part3_band as prior
        for c in CONDITIONS:
            subprocess.run([sys.executable,'-u',__file__,'--batch',str(args.batch),'--execute',args.execute,'--kind',c],check=True)
            child=json.loads(Path(json.loads(args.batch.read_text())['children'][args.execute]).read_text())
            audit_run(child,c)
    elif args.finish:finish(args.batch)
    else:p.error('stage, execute or finish required')
