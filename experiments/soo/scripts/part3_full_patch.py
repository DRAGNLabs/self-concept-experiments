"""Frozen full-adapter prompt patches. BYU_ORC_AGENTS.md applies; run in Slurm."""
import argparse
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import copy
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PARENT = 'part3-addback1-batch-20261029T213251426612Z.json'
SEEDS = ('seed0', 'seed1', 'seed2')
MASKS = ('role', 'goal', 'facts', 'later', 'role_control', 'goal_control', 'all_prompt')
SPECS = [(direction+'__'+mask, direction, mask, False)
         for direction in ('restore', 'reverse') for mask in MASKS]
SPECS += [(direction+'__random_'+mask, direction, mask, True)
          for direction in ('restore', 'reverse') for mask in ('role', 'goal')]


def read(path):
    with Path(path).open() as f: return [json.loads(line) for line in f if line.strip()]


def write(path, obj):
    p = Path(path); tmp = p.with_suffix('.tmp'); tmp.write_text(json.dumps(obj, indent=2)+'\n'); tmp.replace(p)


def masks_for(example, encoded, tokenizer):
    text = encoded['rendered_prompt']; prompt = example['prompt']; start = text.index(prompt)
    tokens = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    assert tokens['input_ids'] == encoded['input_ids']
    offsets = tokens['offset_mapping']; special = set(tokenizer.all_special_ids)
    boundaries = {}
    def span(name, lo, hi):
        indices = [i for i,(a,b) in enumerate(offsets) if b>a and a<hi and b>lo and tokens['input_ids'][i] not in special]
        assert indices, name
        boundaries[name] = [i for i in indices if offsets[i][0]<lo or offsets[i][1]>hi]
        return indices
    first = example['semantic_spans']['role'][0]
    assert first['start']==0 and first['text'].startswith('You represent ')
    role_end = prompt.index('.')+1
    goal_start = prompt.index('your objective', role_end)
    result = dict(role=span('role',start,start+role_end),
                  goal=span('initial_goal',start+goal_start,start+first['end']))
    reminder = example['semantic_spans'].get('added_reminder', [])
    assert bool(reminder) == (example['variant']=='anchor_reminder')
    for i,r in enumerate(reminder): result['goal'] += span('reminder'+str(i),start+r['start'],start+r['end'])
    result['goal'] = sorted(set(result['goal']))
    facts = example['semantic_spans']['payoffs'][0]
    result['facts'] = span('facts',start+facts['start'],start+facts['end'])
    content = span('user_content',start,start+len(prompt))
    result['later'] = content[-8:]; assert len(result['later'])==8
    clause=example['semantic_spans']['consequences'][0]
    controls=span('consequences',start+clause['start'],start+clause['end'])
    for kind in ('role','goal'):
        assert len(controls)>=len(result[kind]); result[kind+'_control']=controls[:len(result[kind])]
        assert not set(result[kind]) & set(result[kind+'_control'])
    result['all_prompt']=list(range(len(encoded['input_ids'])))
    assert not set(result['role']) & set(result['goal'])
    assert set(result)==set(MASKS)
    return dict(example_id=example['example_id'],masks=result,boundary_crossing=boundaries)


class ProjectionHooks:
    """One prefill per module; never edit decode outputs or padding."""
    def __init__(self, modules, rows, width, masks=None, source=None, random=False,
                 direction='', mask_name='', norm_writer=None, phase='', batch_index=0):
        self.modules=modules; self.rows=rows; self.width=width; self.masks=masks
        self.source=source; self.random=random; self.direction=direction; self.mask_name=mask_name
        self.norm_writer=norm_writer; self.phase=phase; self.batch_index=batch_index
        self.counts=defaultdict(int); self.decode=defaultdict(int); self.cache={}; self.handles=[]
    def __enter__(self):
        for name,module in self.modules.items():
            self.handles.append(module.register_forward_hook(self.hook(name)))
        return self
    def __exit__(self, typ, value, traceback):
        for handle in self.handles: handle.remove()
        if typ is None: assert set(self.counts)==set(self.modules) and all(v==1 for v in self.counts.values())
    def hook(self,name):
        def apply(module, inputs, output):
            import torch
            assert isinstance(output,torch.Tensor) and output.ndim==3
            if self.counts[name]:
                assert output.shape[1]==1, (name,output.shape)
                self.decode[name]+=1
                return output
            assert output.shape[:2]==(len(self.rows),self.width), (name,output.shape,self.width)
            self.counts[name]+=1
            if self.source is None:
                self.cache[name]=output.detach().to('cpu',copy=True)
                return output
            donor=self.source[name]; assert donor.shape==output.shape
            out=output.clone(); selected=torch.zeros(output.shape[:2],device=output.device,dtype=torch.bool)
            norms=[]; desired=[]
            for i,row in enumerate(self.rows):
                idx=self.masks[row['example_id']]
                shift=self.width-len(row['input_ids']); assert shift>=0
                assert idx and len(idx)==len(set(idx)) and min(idx)>=0 and max(idx)<len(row['input_ids'])
                padded=torch.tensor(idx,device=output.device)+shift
                selected[i,padded]=True
                original=output[i,padded].float()
                replacement=donor[i,padded.cpu()].to(device=output.device,dtype=output.dtype)
                delta=replacement.float()-original; target_norm=delta.norm(dim=-1)
                if self.random:
                    key=f'1729/{row["example_id"]}/{name}/{self.direction}/{self.mask_name}'
                    seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')%(2**63-1)
                    generator=torch.Generator(device=output.device).manual_seed(seed)
                    noise=torch.randn(delta.shape,device=output.device,dtype=torch.float32,generator=generator)
                    noise=noise/noise.norm(dim=-1,keepdim=True).clamp_min(1e-20)*target_norm[:,None]
                    replacement=(original+noise).to(output.dtype)
                    # Quantization changes the realized norm. Calibrate only its
                    # scalar magnitude, without consulting model answers.
                    realized=(replacement.float()-original).norm(dim=-1)
                    if not torch.all((realized-target_norm).abs()<=.01*target_norm+1e-5):
                        low=torch.zeros_like(target_norm); high=torch.full_like(target_norm,4.)
                        best=replacement; error=(realized-target_norm).abs()
                        for _ in range(18):
                            scale=(low+high)/2
                            candidate=(original+noise*scale[:,None]).to(output.dtype)
                            size=(candidate.float()-original).norm(dim=-1)
                            better=(size-target_norm).abs()<error
                            best=torch.where(better[:,None],candidate,best)
                            error=torch.minimum(error,(size-target_norm).abs())
                            low=torch.where(size<target_norm,scale,low)
                            high=torch.where(size>=target_norm,scale,high)
                        replacement=best
                out[i,padded]=replacement
                realized=(out[i,padded].float()-original).norm(dim=-1)
                assert torch.isfinite(realized).all() and torch.isfinite(target_norm).all()
                if self.random:
                    assert torch.all((realized-target_norm).abs() <= .01*target_norm+1e-5), 'bf16 random norm mismatch'
                else: assert torch.equal(out[i,padded],replacement)
                if self.norm_writer:
                    norms.append(realized.cpu().tolist()); desired.append(target_norm.cpu().tolist())
            assert torch.equal(out[~selected],output[~selected]), 'unselected or padding changed'
            if self.norm_writer:
                self.norm_writer.write(json.dumps(dict(phase=self.phase,batch_index=self.batch_index,
                    condition=self.direction+'__'+('random_' if self.random else '')+self.mask_name,
                    module=name,example_ids=[r['example_id'] for r in self.rows],
                    token_indices=[self.masks[r['example_id']] for r in self.rows],
                    realized=norms,desired=desired),separators=(',',':'))+'\n')
            return out
        return apply


def hook_tests(device='cpu'):
    import torch
    from torch import nn
    torch.manual_seed(71)
    module=nn.Linear(6,5,bias=False).to(device)
    rows=[dict(example_id='a',input_ids=[1,2,3]),dict(example_id='b',input_ids=[4,5])]
    x=torch.randn(2,3,6,device=device); modules={'projection':module}
    with torch.inference_mode():
        expected=module(x)
        with ProjectionHooks(modules,rows,3) as h:
            assert torch.equal(module(x),expected); module(x[:,:1])
        assert h.decode['projection']==1
        with ProjectionHooks(modules,rows,3,{'a':[0,1,2],'b':[0,1]},h.cache) as identity:
            assert torch.equal(module(x),expected)
            assert torch.equal(module(x[:,:1]),module.forward(x[:,:1]))
        donor={n:v+1 for n,v in h.cache.items()}; masks={'a':[1],'b':[0]}
        with ProjectionHooks(modules,rows,3,masks,donor):
            changed=module(x)
        target=expected.clone(); target[:,1]+=1; assert torch.equal(changed,target)
        outcomes=[]
        for _ in range(2):
            with ProjectionHooks(modules,rows,3,masks,donor,True,'restore','role'):
                outcomes.append(module(x))
        assert torch.equal(*outcomes)
        assert torch.allclose((outcomes[0]-expected).norm(dim=-1)[:,1],torch.full((2,),5**.5,device=device),atol=1e-5)
        try:
            with ProjectionHooks(modules,rows,3,masks,donor): raise RuntimeError('cleanup probe')
        except RuntimeError: pass
        assert not module._forward_hooks
    return dict(passed=True,device=device,checks=['capture','left_padding','replacement','identity','decode_unchanged','unselected_unchanged','random_norm','random_determinism','exception_cleanup'])


def stage(repo, protocol):
    study=repo/'experiments/soo/results/study2'; parent_path=study/PARENT
    parent=json.loads(parent_path.read_text()); source=Path(parent['controller']); sys.path.insert(0,str(source))
    import part3_band as prior
    import part3_band_addback as addback
    engine=prior.engine
    engine.verify(parent); assert parent['status']=='completed_and_analyzed' and not parent['errors']
    assert parent['results']['all_seed_qualified']==dict(full=True,band=False,add_q=False,add_v=False)
    checks=hook_tests(); print('CPU_HOOK_TESTS',checks,flush=True)
    for seed in SEEDS: addback.audit_run(json.loads(Path(parent['children'][seed]).read_text()),'full')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root=study/f'part3-fullpatch1-controller-{stamp}';root.mkdir()
    path=study/f'part3-fullpatch1-batch-{stamp}.json'
    for p in source.glob('*.py'): shutil.copy2(p,root/p.name)
    shutil.copy2(__file__,root/'part3_full_patch.py');shutil.copy2(protocol,root/'PART3_FULL_PATCH.md')
    shutil.copytree(source/'src',root/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('historical.jsonl','plan.json','parent_scores.jsonl'):shutil.copy2(source/name,root/name)
    shutil.copytree(source/'preflight',root/'preflight')
    from transformers import AutoTokenizer
    from selfconcept.soo.goals_roles import tokenize_example
    first=json.loads(Path(parent['children']['seed0']).read_text())
    tokenizer=AutoTokenizer.from_pretrained('google/gemma-4-31B-it',revision=first['revision'],local_files_only=True,padding_side='left')
    examples=read(root/'historical.jsonl'); encoded=read(root/'preflight/token_spans.jsonl')
    assert len(examples)==len(encoded)==1728
    masks=[]
    for e,r in zip(examples,encoded):
        assert tokenize_example(e,tokenizer,{'enable_thinking':False})==r
        masks.append(masks_for(e,r,tokenizer))
    engine.jsonl(root/'masks.jsonl',masks)
    checks['native_inputs']=1728;checks['mask_counts']={k:sorted({len(r['masks'][k]) for r in masks}) for k in MASKS}
    engine.write(root/'preparation_tests.json',checks)
    print('MASKS',checks['mask_counts'],flush=True)
    # Historical base references from complete unchanged parent batches.
    band=json.loads(Path(parent['parent']).read_text()); original=json.loads(Path(band['parent']).read_text())
    base=[]; base_reg=None; ids={e['example_id'] for e in examples}
    for split in prior.SPLITS:
        previous=Path(original['runs']['base-'+split]['output'])
        base.extend(engine.raw_only(r) for r in read(previous/'planned_responses.jsonl') if r['phase']=='primary' and r['example_id'] in ids)
        regs=read(previous/'regressions.jsonl')
        if base_reg is not None: assert regs==base_reg
        base_reg=regs
    assert len(base)==1728;engine.jsonl(root/'base-reference.jsonl',base);engine.jsonl(root/'base-regressions.jsonl',base_reg)
    children={}
    for seed in SEEDS:
        old=json.loads(Path(parent['children'][seed]).read_text()); oldroot=Path(old['controller']); dest=root/seed;dest.mkdir()
        for name in ('adapter','historical_data'):shutil.copytree(oldroot/name,dest/name)
        for name in ('primary.jsonl','regressions.jsonl'):shutil.copy2(oldroot/'references/full'/name,dest/('full-'+name))
        engine.write(dest/'provenance.json',dict(protocol='part3_fullpatch1',seed=seed,parent=str(parent_path)))
        children[seed]=dict(seed=old['seed'],revision=old['revision'],reference_versions=old['reference_versions'],
            inference_source_sha256=old['inference_source_sha256'],adapter_sha256=old['adapter_sha256'],all_modules=old['all_modules'])
    receipts=study/f'part3-fullpatch1-receipts-{stamp}';receipts.mkdir()
    batch=dict(protocol='part3_fullpatch1',controller=str(root),parent=str(parent_path),children=children,
        status='staged',receipts=str(receipts),output=str(study/f'part3-fullpatch1-output-{stamp}'),
        analysis=str(study/f'part3-fullpatch1-analysis-{stamp}'),errors={},results={},runs={},
        authorization='User approved full-adapter patch recommendation; all seeds, complete historical cohort.',
        scientific_conditions=[s[0] for s in SPECS],primary_per_condition=1728,scientific_primary=93312,
        reference_sha256={str(parent_path):prior.digest(parent_path)})
    for phase in ('mechanics','science'):
        for seed in SEEDS:
            name=phase+'-'+seed
            script=prior.scheduling.shell(repo,root,path,name,[str(repo/'.venv-313/bin/python'),'-u',str(root/'part3_full_patch.py'),
                '--batch',str(path),'--execute',seed,'--phase',phase],True)
            p=root/script;t=p.read_text().replace('soo-tb2-','soo-p3fp-')
            if phase=='science':t=t.replace('01:00:00','03:00:00')
            p.write_text(t);batch['runs'][name]=dict(script=script,status='staged')
    for stage_name in ('release','final'):
        script=prior.scheduling.shell(repo,root,path,stage_name,['/usr/bin/python3.11','-u',str(root/'part3_full_patch.py'),
            '--batch',str(path),'--'+stage_name],False)
        p=root/script;p.write_text(p.read_text().replace('soo-tb2-','soo-p3fp-').replace('--mem=8G','--mem=16G'))
    for p in root.glob('*.sh'):subprocess.run(['bash','-n',str(p)],check=True)
    batch['frozen_sha256']=engine.hashes(root);engine.write(path,batch);engine.verify(batch)
    print('FROZEN',path,flush=True);return path


def execute(path,seed,phase):
    import torch
    from selfconcept.soo import run_goals_roles as runner, goals_roles_repeatability
    import part3_band as prior
    from peft.tuners.lora.layer import LoraLayer
    batch=json.loads(path.read_text());prior.engine.verify(batch)
    if phase=='science':
        release=json.loads((Path(batch['receipts'])/'mechanics-approved.json').read_text())
        assert release['all_seeds_passed'] and release['frozen_sha256']==batch['frozen_sha256']
    root=Path(batch['controller']);child=batch['children'][seed];out=Path(batch['output'])/phase/seed
    out.mkdir(parents=True,exist_ok=False)
    checks=hook_tests('cuda');write(out/'gpu_tests.json',checks)
    examples=read(root/'historical.jsonl');plan=json.loads((root/'plan.json').read_text())
    frozen_inputs=read(root/'preflight/token_spans.jsonl');masks={r['example_id']:r['masks'] for r in read(root/'masks.jsonl')}
    refs={'base':{r['example_id']:r for r in read(root/'base-reference.jsonl')},
          'full':{r['example_id']:r for r in read(root/seed/'full-primary.jsonl')}}
    args=runner.parser().parse_args(['--revision',child['revision'],'--data',str(root/'historical.jsonl'),
        '--out',str(out),'--provenance',str(root/seed/'provenance.json'),'--condition','adapter',
        '--adapter',str(root/seed/'adapter'),'--repeat-plan',str(root/'plan.json'),
        '--regression-data',str(root/seed/'historical_data')])
    def callback(plan_path,output,actual_examples,encoded,generate):
        assert actual_examples==examples and encoded==frozen_inputs
        assert plan['expected_inputs']=={r['example_id']:r['input_ids'] for r in encoded}
        env=dict(zip(generate.__code__.co_freevars,[c.cell_contents for c in generate.__closure__]))
        model=env['model']; tokenizer=env['tokenizer']
        modules={n:m for n,m in model.named_modules() if isinstance(m,LoraLayer)}
        assert sorted(modules)==child['all_modules'] and len(modules)==110
        assert all(not m.merged for m in modules.values())
        write(out/'modules.json',sorted(modules))
        # Full regressions were run by the unchanged inherited runner.
        assert read(out/'regressions.jsonl')==read(root/seed/'full-regressions.jsonl')
        baseout=out/'base-regression';baseout.mkdir()
        baseargs=copy.copy(args);baseargs.out=baseout;baseargs.condition='base'
        with model.disable_adapter():runner.run_regressions(baseargs,tokenizer,{'enable_thinking':False},generate)
        assert read(baseout/'regressions.jsonl')==read(root/'base-regressions.jsonl')
        index={r['example_id']:r for r in encoded};started=time.monotonic();generations=0
        with (out/'responses.jsonl').open('x',buffering=1024*1024) as records, gzip.open(out/'norms.jsonl.gz','xt',compresslevel=1) as norms:
            def save(raw,condition,pname,bi,ids):
                nonlocal generations
                assert [r['example_id'] for r in raw]==ids
                for r in raw:records.write(json.dumps(dict(r,condition=condition,phase=pname,batch_index=bi,batch_members=ids))+'\n')
                generations+=len(raw)
            for p in plan['phases']:
                for bi,ids in enumerate(p['batches']):
                    rows=[index[eid] for eid in ids];width=max(len(r['input_ids']) for r in rows);captures={}
                    for mode in ('base','full'):
                        with (model.disable_adapter() if mode=='base' else nullcontext()), ProjectionHooks(modules,rows,width) as h:
                            raw=generate(rows)
                        assert raw==[refs[mode][eid] for eid in ids], (seed,phase,mode,p['name'],bi,'parent replay changed')
                        captures[mode]=h.cache;save(raw,mode,p['name'],bi,ids)
                    if phase=='mechanics':
                        for mode in ('base','full'):
                            with (model.disable_adapter() if mode=='base' else nullcontext()), ProjectionHooks(modules,rows,width,
                                  {eid:masks[eid]['all_prompt'] for eid in ids},captures[mode]) as h:
                                raw=generate(rows)
                            assert raw==[refs[mode][eid] for eid in ids], ('identity changed',seed,mode,bi)
                            save(raw,mode+'_identity',p['name'],bi,ids)
                    else:
                        for condition,direction,mask,random in SPECS:
                            source,target=('base','full') if direction=='restore' else ('full','base')
                            with (model.disable_adapter() if target=='base' else nullcontext()), ProjectionHooks(modules,rows,width,
                                  {eid:masks[eid][mask] for eid in ids},captures[source],random,direction,mask,norms,p['name'],bi) as h:
                                raw=generate(rows)
                            save(raw,condition,p['name'],bi,ids)
                    del captures
                    records.flush()
                    if bi%16==0:print(phase,seed,p['name'],bi+1,'/',len(p['batches']),'elapsed',round(time.monotonic()-started,1),flush=True)
        expected=plan['generations']*(4 if phase=='mechanics' else 20);assert generations==expected
        write(out/'complete.json',dict(passed=True,generations=generations,elapsed_seconds=time.monotonic()-started,
            max_gpu_bytes=torch.cuda.max_memory_allocated(),modules=110,phase=phase,
            artifacts={n:prior.digest(out/n) for n in ('responses.jsonl','norms.jsonl.gz','modules.json','gpu_tests.json','regressions.jsonl','base-regression/regressions.jsonl')}))
    goals_roles_repeatability.execute_plan=callback
    with (out/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);runner.run(args,examples)


def audit(batch,seed,phase):
    import part3_band as prior
    root=Path(batch['controller']);child=batch['children'][seed];out=Path(batch['output'])/phase/seed
    complete=json.loads((out/'complete.json').read_text());assert complete['passed'] and complete['phase']==phase
    for name,expected in complete['artifacts'].items():assert prior.digest(out/name)==expected,name
    assert json.loads((out/'gpu_tests.json').read_text())['passed']
    assert json.loads((out/'modules.json').read_text())==child['all_modules']
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['source_sha256']==child['inference_source_sha256']
    assert manifest['revision']==child['revision'] and manifest['model']=='google/gemma-4-31B-it'
    assert manifest['condition']=='adapter' and manifest['adapter']==str(root/seed/'adapter')
    assert manifest['versions']==child['reference_versions'] and 'A100' in manifest['hardware']['gpu']
    assert manifest['dtype']=='bfloat16' and manifest['attention']=='sdpa' and manifest['chat_kwargs']=={'enable_thinking':False}
    assert manifest['batch_size']==8 and manifest['max_new_tokens']==32 and manifest['do_sample'] is False
    for name,expected in manifest['artifact_sha256'].items():assert prior.digest(name)==expected
    assert read(out/'inputs.jsonl')==read(root/'preflight/token_spans.jsonl')
    examples=read(root/'historical.jsonl');plan=json.loads((root/'plan.json').read_text())
    assert manifest['example_ids']==[e['example_id'] for e in examples]
    assert manifest['prompt_ids_sha256']==hashlib.sha256(json.dumps([r['input_ids'] for r in read(out/'inputs.jsonl')]).encode()).hexdigest()
    for name,expected in child['adapter_sha256'].items():assert prior.digest(root/seed/'adapter'/name)==expected
    assert read(out/'regressions.jsonl')==read(root/seed/'full-regressions.jsonl')
    assert read(out/'base-regression/regressions.jsonl')==read(root/'base-regressions.jsonl')
    conditions=['base','full']+(['base_identity','full_identity'] if phase=='mechanics' else [s[0] for s in SPECS])
    expected=[(p['name'],i,ids,c,eid) for p in plan['phases'] for i,ids in enumerate(p['batches']) for c in conditions for eid in ids]
    responses=read(out/'responses.jsonl');assert len(responses)==len(expected)==complete['generations']
    by=defaultdict(dict)
    for r,(p,i,ids,c,eid) in zip(responses,expected):
        assert (r['phase'],r['batch_index'],r['batch_members'],r['condition'],r['example_id'])==(p,i,ids,c,eid)
        assert r['prompt_tokens']==len(plan['expected_inputs'][eid])
        assert r['completion_status'] in ('complete','truncated') and r['generated_tokens']==len(r['generated_token_ids'])<=32
        assert eid not in by[p,c];by[p,c][eid]=prior.engine.raw_only(r)
    refs={'base':read(root/'base-reference.jsonl'),'full':read(root/seed/'full-primary.jsonl')}
    for mode in ('base','full'):
        assert list(by['primary',mode].values())==refs[mode]
        if phase=='mechanics':assert by['primary',mode+'_identity']==by['primary',mode]
    for c in conditions:
        for p in ('sentinel_before','sentinel_after'):
            assert all(r==by['primary',c][eid] for eid,r in by[p,c].items()),('sentinel changed',c,p)
    scores=[dict(r,seed=seed,condition=c) for c in conditions for r in prior.task.score(examples,list(by['primary',c].values()))]
    return dict(passed=True,seed=seed,phase=phase,generations=len(responses),complete=complete),scores


def placement(batch,hours):
    root=Path(batch['controller'])
    command=['/usr/bin/python3.11',str(root/'gpu_pick.py'),'--hours',str(hours),'--vram-gb','75','--cpus','4','--mem-gb','96',
        '--no-requeue-ok','--fast','--arch','x86_64','--cuda','13.0','--top','10']
    for gpu in ('p100','l40s','h100','gh200','h200','b200'):command+=['--exclude-gpu',gpu]
    output=subprocess.check_output(command,text=True);selected=None
    for line in output.splitlines():
        cols=line.split()
        if cols and cols[0]=='1' and len(cols)>4:
            assert cols[2]=='a100' and cols[3]=='1';selected=dict(partition=cols[1],qos=cols[4]);break
    evidence=Path(batch['receipts'])/('gpu-pick-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    write(evidence,dict(command=command,output=output,selected=selected,rationale='Best compatible A100 option preserving exact historical replays. No preemption.'))
    if selected is None:raise RuntimeError('No compatible A100 placement')
    return dict(selected,evidence=str(evidence))


def submit(path,phase):
    import part3_band as prior
    batch=json.loads(path.read_text());prior.engine.verify(batch);root=Path(batch['controller']);receipts=Path(batch['receipts'])
    with (receipts/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        selected=None;jobs=[]
        for seed in SEEDS:
            name=phase+'-'+seed;entry=batch['runs'][name];script=root/entry['script'];receipt=receipts/(name+'.json')
            if receipt.exists():r=json.loads(receipt.read_text());assert r['script_sha256']==prior.digest(script)
            else:
                if selected is None:selected=placement(batch,1 if phase=='mechanics' else 3)
                command=['/apps/slurm/latest/bin/sbatch','--parsable','--partition='+selected['partition'],'--qos='+selected['qos'],'--no-requeue',str(script)]
                job=subprocess.check_output(command,text=True).strip().split(';')[0]
                r=dict(job_id=job,command=command,placement=selected,script_sha256=prior.digest(script));write(receipt,r)
            entry.update(r,status='submitted');jobs.append(r['job_id']);write(path,batch)
        name='release' if phase=='mechanics' else 'final';script=root/(name+'.sh');receipt=receipts/(name+'.json')
        if receipt.exists():r=json.loads(receipt.read_text());assert r['script_sha256']==prior.digest(script)
        else:
            command=['/apps/slurm/latest/bin/sbatch','--parsable','--dependency=afterany:'+':'.join(jobs),str(script)]
            r=dict(job_id=subprocess.check_output(command,text=True).strip().split(';')[0],command=command,dependencies=jobs,script_sha256=prior.digest(script));write(receipt,r)
        batch[name+'_job']=r;batch['status']=phase+'_submitted';write(path,batch)
    print('SUBMITTED',phase,jobs,name,r['job_id'],flush=True)


def release(path):
    import part3_band as prior
    batch=json.loads(path.read_text());prior.engine.verify(batch);receipts=Path(batch['receipts']);audits=[]
    try:
        for seed in SEEDS:
            result,scores=audit(batch,seed,'mechanics');audits.append(result)
            full=[r for r in scores if r['condition']=='full'];base=[r for r in scores if r['condition']=='base']
            assert prior.qualification(full,base)['passed']
        write(receipts/'mechanics-approved.json',dict(all_seeds_passed=True,audits=audits,frozen_sha256=batch['frozen_sha256']))
    except Exception as exc:
        batch['status']='mechanics_failed_stopped';batch['errors']['mechanics']=repr(exc);write(path,batch);raise
    submit(path,'science')


def finish(path):
    import part3_band as prior
    batch=json.loads(path.read_text());prior.engine.verify(batch);out=Path(batch['analysis']);out.mkdir(exist_ok=True)
    try:
        scores=[];audits=[]
        for seed in SEEDS:
            result,rows=audit(batch,seed,'science');audits.append(result);scores.extend(rows)
        prior.engine.jsonl(out/'scores.jsonl',scores);write(out/'audits.json',audits)
        fields=('seed','condition','split','variant','query','alignment');metrics=('valid','correct','assigned_goal','recipient_goal','dominated')
        write(out/'summary.json',prior.existing.aggregate(scores,fields,metrics))
        write(out/'world_role.json',prior.existing.aggregate(scores,fields+('instance_id','role'),metrics))
        indexed={(r['seed'],r['condition'],r['example_id']):r for r in scores};paired=[];joint=[]
        for r in scores:
            if r['condition'] in ('base','full'):continue
            target='full' if r['condition'].startswith('restore') else 'base'
            ref=indexed[r['seed'],target,r['example_id']]
            paired.append(dict(**{k:r[k] for k in fields},example_id=r['example_id'],
                assigned_change=int(r['assigned_goal'])-int(ref['assigned_goal']),
                recipient_change=int(r['recipient_goal'])-int(ref['recipient_goal']),
                correct_change=int(r['correct'])-int(ref['correct']),answer_changed=r['answer']!=ref['answer']))
        for r in scores:
            if r['query']!='message':continue
            stem=r['example_id'].rsplit('-',1)[0]
            controls=[indexed[r['seed'],r['condition'],stem+'-'+q] for q in ('private','fact0','fact1')]
            joint.append(dict(**{k:r[k] for k in fields if k!='query'},instance_id=r['instance_id'],role=r['role'],
                controls_correct=all(x['correct'] for x in controls),joint_assigned=r['assigned_goal'] and all(x['correct'] for x in controls),
                joint_recipient=r['recipient_goal'] and all(x['correct'] for x in controls)))
        prior.engine.jsonl(out/'paired.jsonl',paired)
        write(out/'paired_summary.json',prior.existing.aggregate(paired,fields,('assigned_change','recipient_change','correct_change','answer_changed')))
        write(out/'joint_summary.json',prior.existing.aggregate(joint,('seed','condition','split','variant','alignment'),('controls_correct','joint_assigned','joint_recipient')))
        # Independently audit norm coverage and selected token indices, streaming compressed records.
        root=Path(batch['controller']);plan=json.loads((root/'plan.json').read_text());masks={r['example_id']:r['masks'] for r in read(root/'masks.jsonl')}
        stats={}
        for seed in SEEDS:
            stats[seed]={};expected_records=0
            with gzip.open(Path(batch['output'])/'science'/seed/'norms.jsonl.gz','rt') as f:
                for p in plan['phases']:
                    for bi,ids in enumerate(p['batches']):
                        for condition,direction,mask,random in SPECS:
                            seen=set()
                            for _ in batch['children'][seed]['all_modules']:
                                r=json.loads(next(f));expected_records+=1
                                assert (r['phase'],r['batch_index'],r['condition'],r['example_ids'])==(p['name'],bi,condition,ids)
                                assert r['module'] in batch['children'][seed]['all_modules'] and r['module'] not in seen;seen.add(r['module'])
                                assert r['token_indices']==[masks[eid][mask] for eid in ids]
                                assert len(r['realized'])==len(r['desired'])==8
                                key=condition+'|'+r['module'];s=stats[seed].setdefault(key,dict(tokens=0,sum_l2=0.,sum_squared_l2=0.,max_l2=0.))
                                for inds,actual,desired in zip(r['token_indices'],r['realized'],r['desired']):
                                    assert len(inds)==len(actual)==len(desired)
                                    for a,d in zip(actual,desired):
                                        assert 0<=a<float('inf') and 0<=d<float('inf')
                                        assert abs(a-d)<=.01*d+1e-5 if random else a==d
                                        if p['name']=='primary':s['tokens']+=1;s['sum_l2']+=a;s['sum_squared_l2']+=a*a;s['max_l2']=max(s['max_l2'],a)
                assert not f.read().strip()
        write(out/'norm_summary.json',stats)
        batch['status']='completed_and_analyzed';batch['results']=dict(primary=len(scores),scientific_primary=93312,valid=sum(r['valid'] for r in scores),all_integrity_passed=True)
        write(path,batch);print(json.dumps(batch['results']),flush=True)
    except Exception as exc:
        batch['status']='audit_failed_stopped';batch['errors']['final']=repr(exc);write(path,batch);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',type=Path);parser.add_argument('--protocol',type=Path);parser.add_argument('--batch',type=Path)
    parser.add_argument('--execute',choices=SEEDS);parser.add_argument('--phase',choices=('mechanics','science'))
    parser.add_argument('--submit',action='store_true');parser.add_argument('--release',action='store_true');parser.add_argument('--final',action='store_true')
    args=parser.parse_args()
    if args.stage:
        path=stage(args.stage.resolve(),args.protocol.resolve())
        if args.submit:submit(path,'mechanics')
    elif args.execute:execute(args.batch,args.execute,args.phase)
    elif args.release:release(args.batch)
    elif args.final:finish(args.batch)
    elif args.submit:submit(args.batch,'mechanics')
    else:parser.error('stage, execute, release, final or submit required')
