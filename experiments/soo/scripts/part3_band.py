"""Frozen, all-seed bridge from the qualified Part 1 contrast to the v_proj band.

BYU ORC policy: repository BYU_ORC_AGENTS.md. Stage/audit in CPU Slurm;
inference only on allocated GPUs. No automatic activation-patch release.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess

import extension_repeat_bridge as engine
from extension_repeat_selection import digest, read
import goal_identity_task as task
import goal_identity as parent_flow
import part3_existing_audit as existing
import truth_beneficiary_revision as scheduling

VARIANTS = ('anchor_original', 'anchor_reminder', 'original', 'name_late', 'goal_late')
SPLITS = ('development', 'reserved')
SEEDS = ('seed0', 'seed1', 'seed2')


def select_batches(examples, plan):
    wanted = {e['example_id'] for e in examples if e['variant'] in VARIANTS}
    primary = next(p['batches'] for p in plan['phases'] if p['name'] == 'primary')
    selected = [b for b in primary if wanted.intersection(b)]
    assert all(len(b) == 8 and set(b) <= wanted for b in selected)
    assert [eid for b in selected for eid in b] == [e['example_id'] for e in examples if e['example_id'] in wanted]
    index = {e['example_id']: e for e in examples}
    seen = set(); references = []
    for batch in selected:
        keys = {tuple(index[eid][f] for f in ('template', 'variant', 'role', 'alignment', 'query')) for eid in batch}
        if not keys <= seen:
            references.append(batch); seen.update(keys)
    return selected, references


def plan_for(batches, sentinels, native):
    ids = [eid for b in batches for eid in b]
    assert len(ids) == len(set(ids))
    return dict(expected_inputs={eid: native[eid] for eid in ids}, phases=[
        dict(name='sentinel_before', batches=sentinels), dict(name='primary', batches=batches),
        dict(name='sentinel_after', batches=sentinels)],
        generations=len(ids)+2*sum(map(len, sentinels)))


def module_names(adapter):
    # Read the safetensors header only; no tensor load on the CPU preparation node.
    with (adapter/'adapter_model.safetensors').open('rb') as stream:
        size, = struct.unpack('<Q', stream.read(8))
        header = json.loads(stream.read(size))
    result = []
    for name in header:
        match = re.search(r'\.layers\.(\d+)\.self_attn\.v_proj\.lora_A\.weight$', name)
        if match and 20 <= int(match.group(1)) <= 31:
            result.append(name.rsplit('.lora_A.weight', 1)[0])
    assert result
    return sorted(result)


def verify_adapter(adapter, expected, seed):
    # Parent provenance hashes model artifacts; README/log files are also copied
    # and frozen below, but must not change that provenance key set.
    assert set(expected) == {'adapter_config.json', 'adapter_model.safetensors'}
    assert all(digest(adapter/name) == value for name, value in expected.items())
    assert json.loads((adapter/'train_log.json').read_text())['seed'] == seed


def stage(repo, audit_path):
    soo = repo/'experiments/soo'; study = soo/'results/study2'
    assert json.loads((audit_path/'complete.json').read_text()) == dict(passed=True, primary=69120)
    provenance = json.loads((audit_path/'provenance.json').read_text())
    assert provenance['inputs_independently_keyed'] == 17280
    parent_path = study/existing.PARENT; parent = json.loads(parent_path.read_text())
    engine.verify(parent)
    assert parent['status'] == 'completed_and_analyzed'
    source = Path(parent['controller'])
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root = study/f'part3-band1-controller-{stamp}'; root.mkdir()
    path = study/f'part3-band1-batch-{stamp}.json'
    for f in source.glob('*.py'):
        if not f.name.startswith('test_'): shutil.copy2(f, root/f.name)
    for name in ('part3_band.py', 'part3_existing_audit.py', 'truth_beneficiary_revision.py',
                 'truth_beneficiary_revision_task.py', 'truth_beneficiary_task.py'):
        shutil.copy2(soo/'scripts'/name, root/name)
    shutil.copy2(soo/'PART3_CAUSAL.md', root/'PART3_CAUSAL.md')
    shutil.copy2(repo/'tests/test_soo_part3.py', root/'test_soo_part3.py')
    shutil.copy2(repo/'tests/test_soo_goal_identity.py', root/'test_soo_goal_identity.py')
    shutil.copy2(repo/'gpu_pick.py', root/'gpu_pick.py')
    shutil.copytree(source/'src', root/'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(audit_path, root/'existing_audit')
    refs = {str(parent_path): digest(parent_path)}
    for f in audit_path.glob('*.json'): refs[str(f)] = digest(f)
    parent_scores = []
    for label in ('base',)+SEEDS:
        for split in SPLITS:
            audit_file = Path(parent['analysis'])/'final'/(label+'-'+split+'-audit.json')
            prior = json.loads(audit_file.read_text()); assert prior['checks_passed']
            refs[str(audit_file)] = digest(audit_file)
            parent_scores.extend(dict(r, condition=label) for r in prior['scores'] if r['variant'] in VARIANTS)
    engine.jsonl(root/'parent_scores.jsonl', parent_scores)
    data = {}; total_band = 0
    for split in SPLITS:
        examples = read(source/'data'/f'{split}.jsonl')
        oldplan = json.loads((source/'plans'/f'{split}.json').read_text())
        batches, reference = select_batches(examples, oldplan)
        assert sum(map(len, batches)) == 3456
        # Re-run native-token/span preflight rather than trust old annotations alone.
        selected_ids = {eid for b in batches for eid in b}
        selected = [e for e in examples if e['example_id'] in selected_ids]
        folder = root/'data'/split; folder.mkdir(parents=True)
        engine.jsonl(folder/'band.jsonl', selected)
        env = dict(os.environ, PYTHONPATH=str(root/'src'), OPENSSL_CONF='/dev/null', HF_HUB_OFFLINE='1', USE_TORCH='0')
        subprocess.run([str(repo/'.venv-313/bin/python'), str(root/'validate_goals_roles_tokens.py'),
            '--data', str(folder/'band.jsonl'), '--out', str(folder/'band-preflight')], env=env, check=True)
        encoded = read(folder/'band-preflight/token_spans.jsonl')
        native = {r['example_id']: r['input_ids'] for r in encoded}
        assert native == {eid: oldplan['expected_inputs'][eid] for eid in selected_ids}
        ref_ids = {eid for b in reference for eid in b}
        engine.jsonl(folder/'reference.jsonl', [e for e in selected if e['example_id'] in ref_ids])
        (folder/'reference-preflight').mkdir()
        engine.jsonl(folder/'reference-preflight/token_spans.jsonl', [e for e in encoded if e['example_id'] in ref_ids])
        engine.write(folder/'band-plan.json', plan_for(batches, reference, native))
        engine.write(folder/'reference-plan.json', plan_for(reference, reference[:1]+reference[-1:], native))
        data[split] = dict(folder=str(folder), band=len(selected), reference=len(ref_ids))
        total_band += len(selected)*3
    children = {}; runs = {}; weights = []
    for label in SEEDS:
        old = json.loads(Path(parent['children'][label]).read_text()); oldroot = Path(old['controller'])
        childroot = root/label; childroot.mkdir()
        shutil.copytree(oldroot/'adapter', childroot/'adapter')
        verify_adapter(childroot/'adapter', old['adapter_sha256'], int(label[-1]))
        weights.append(old['adapter_sha256']['adapter_model.safetensors'])
        modules = module_names(childroot/'adapter')
        shutil.copytree(oldroot/'historical_data', childroot/'historical_data')
        engine.write(childroot/'provenance.json', dict(protocol='part3_band1', seed=int(label[-1]),
            parent=str(parent_path), primary_cohort='all_historical_anchors', fresh_cohort='all_selected_variants',
            expected_retained_modules=modules, no_outcome_filtering=True))
        current = {str(p.relative_to(root/'src/selfconcept')): digest(p) for p in (root/'src/selfconcept').rglob('*.py')}
        assert current == old['inference_source_sha256']
        entries = {}
        for split in SPLITS:
            folder = Path(data[split]['folder'])
            for condition in ('base', 'adapter', 'band'):
                kind = condition+'-'+split; tag = 'base' if condition == 'base' else 'adapter'
                key = 'band' if condition == 'band' else 'reference'
                output = study/f'part3-band1-output-{stamp}'/label/kind
                entries[kind] = {tag: dict(condition=condition, alpha=1.0,
                    data=str(folder/(key+'.jsonl')), plan=str(folder/(key+'-plan.json')),
                    preflight=str(folder/(key+'-preflight')), output=str(output))}
                if condition != 'band':
                    old_label = 'base' if condition == 'base' else label
                    parent_out = Path(parent['runs'][old_label+'-'+split]['output'])
                    ids = {e['example_id'] for e in read(folder/'reference.jsonl')}
                    reference_dir = childroot/'references'/kind; reference_dir.mkdir(parents=True)
                    engine.jsonl(reference_dir/'primary.jsonl', [engine.raw_only(r)
                        for r in read(parent_out/'planned_responses.jsonl')
                        if r['phase'] == 'primary' and r['example_id'] in ids])
                    shutil.copy2(parent_out/'regressions.jsonl', reference_dir/'regressions.jsonl')
        child = dict(controller=str(childroot), seed=int(label[-1]), revision=old['revision'],
            reference_versions=old['reference_versions'], inference_source_sha256=current,
            adapter_sha256=old['adapter_sha256'], expected_retained_modules=modules,
            reference_sha256={}, frozen_sha256=engine.hashes(childroot), runs=entries)
        childpath = root/(label+'-batch.json'); engine.write(childpath, child)
        children[label] = str(childpath)
        script = scheduling.shell(repo, root, path, label,
            [str(repo/'.venv-313/bin/python'), '-u', str(root/'part3_band.py'), '--batch', str(path), '--execute', label], True)
        scriptpath = root/script
        scriptpath.write_text(scriptpath.read_text().replace('soo-tb2-', 'soo-p3-'))
        runs[label] = dict(script=script, status='staged_not_submitted')
    assert len(set(weights)) == 3 and total_band == 20736
    final_script = scheduling.shell(repo, root, path, 'final', ['/usr/bin/python3.11', '-u',
        str(root/'part3_band.py'), '--batch', str(path), '--finish'], False)
    p = root/final_script; p.write_text(p.read_text().replace('soo-tb2-', 'soo-p3-'))
    receipts = study/f'part3-band1-receipts-{stamp}'; receipts.mkdir()
    batch = dict(protocol='part3_band1', controller=str(root), parent=str(parent_path),
        authorization='User authorized existing-data audit and Part 3 on 2026-10-29.',
        status='staged_not_submitted', created_utc=datetime.now(timezone.utc).isoformat(),
        analysis=str(study/f'part3-band1-analysis-{stamp}'), receipts=str(receipts),
        children=children, runs=runs, data=data, band_primary=total_band,
        errors={}, results={}, reference_sha256=refs, frozen_sha256=engine.hashes(root))
    engine.write(path, batch); engine.verify(batch)
    print('FROZEN', path, flush=True)
    return path


def execute_kind(path, label, kind):
    batch = json.loads(path.read_text()); engine.verify(batch)
    childpath = Path(batch['children'][label]); child = json.loads(childpath.read_text())
    from selfconcept.soo import lora_subset
    from peft.tuners.lora.layer import LoraLayer
    original_subset = lora_subset.apply_adapter_subset
    def record_subset(model, layers, modules):
        result = original_subset(model, layers, modules)
        if result is not None:
            result['actual_module_names'] = sorted(name for name, module in model.named_modules() if isinstance(module, LoraLayer))
            assert result['actual_module_names'] == child['expected_retained_modules']
        return result
    lora_subset.apply_adapter_subset = record_subset
    engine.TAGS = ('base', 'adapter')
    engine.execute(childpath, kind, 'base' if kind.startswith('base-') else 'adapter')


def audit_run(child, kind):
    engine.verify(child)
    tag = 'base' if kind.startswith('base-') else 'adapter'
    entry = child['runs'][kind][tag]; out = Path(entry['output'])
    plan = json.loads(Path(entry['plan']).read_text()); examples = read(entry['data'])
    assert json.loads((out/'execution_complete.json').read_text()) == dict(phases=3, generations=plan['generations'])
    manifest = json.loads((out/'manifest.json').read_text())
    assert manifest['source_sha256'] == child['inference_source_sha256']
    assert manifest['condition'] == entry['condition'] and manifest['offset_alpha'] == 1
    assert manifest['revision'] == child['revision'] and manifest['model'] == 'google/gemma-4-31B-it'
    assert manifest['versions'] == child['reference_versions'] and 'A100' in manifest['hardware']['gpu']
    assert manifest['dtype'] == 'bfloat16' and manifest['attention'] == 'sdpa'
    assert manifest['chat_kwargs'] == {'enable_thinking': False} and manifest['do_sample'] is False
    assert manifest['batch_size'] == 8 and manifest['max_new_tokens'] == 32
    assert manifest['example_ids'] == [e['example_id'] for e in examples]
    for p, value in manifest['artifact_sha256'].items(): assert digest(p) == value
    assert manifest['adapter'] == (None if tag == 'base' else str(Path(child['controller'])/'adapter'))
    inputs = read(out/'inputs.jsonl')
    assert inputs == read(Path(entry['preflight'])/'token_spans.jsonl')
    native = {r['example_id']: r['input_ids'] for r in inputs}
    assert native == plan['expected_inputs']
    assert manifest['prompt_ids_sha256'] == hashlib.sha256(json.dumps([r['input_ids'] for r in inputs]).encode()).hexdigest()
    intervention = json.loads((out/'intervention.json').read_text())
    if tag == 'base': assert intervention['inactive_hook_check']['passed']
    elif entry['condition'] == 'adapter': assert intervention['subset'] is None
    else:
        subset = intervention['subset']
        assert subset['actual_module_names'] == child['expected_retained_modules']
        assert subset['kept'] == len(child['expected_retained_modules'])
        assert subset['layer_spec'] == 'range:20-31' and subset['module_spec'] == 'v_proj'
        assert subset['kept_modules'] == ['v_proj'] and subset['removed'] > 0
    expected = [(phase['name'], i, ids, eid) for phase in plan['phases']
                for i, ids in enumerate(phase['batches']) for eid in ids]
    responses = read(out/'planned_responses.jsonl'); assert len(responses) == len(expected) == plan['generations']
    phases = defaultdict(dict)
    for r, (phase, i, ids, eid) in zip(responses, expected):
        assert (r['phase'], r['batch_index'], r['batch_members'], r['example_id']) == (phase, i, ids, eid)
        assert r['padded_width'] == max(len(native[e]) for e in ids)
        assert r['prompt_tokens'] == len(native[eid]) and r['generated_tokens'] == len(r['generated_token_ids'])
        assert r['completion_status'] in ('complete', 'truncated') and eid not in phases[phase]
        phases[phase][eid] = engine.raw_only(r)
    for phase in ('sentinel_before', 'sentinel_after'):
        assert all(r == phases['primary'][eid] for eid, r in phases[phase].items())
    regressions = read(out/'regressions.jsonl')
    assert len(regressions) == 32 and len({(r['orientation'], r['scenario'], r['example_id']) for r in regressions}) == 32
    if entry['condition'] != 'band':
        reference = Path(child['controller'])/'references'/kind
        assert regressions == read(reference/'regressions.jsonl')
        assert list(phases['primary'].values()) == read(reference/'primary.jsonl'), 'exact parent replay changed'
    # A band is a new condition. Its regression results are reported, not
    # falsely required to equal full-adapter outputs.
    scores = task.score(examples, list(phases['primary'].values()))
    return dict(checks_passed=True, scores=scores, manifest=manifest, intervention=intervention, regressions=regressions)


def qualification(scores, base_scores):
    checks = []
    for split in SPLITS:
        rows = [r for r in scores if r['split'] == split and r['cohort'] == 'anchor']
        base = [r for r in base_scores if r['split'] == split and r['cohort'] == 'anchor']
        assert len(rows) == len(base) == 864
        controls = [r for r in rows if r['query'] != 'message']
        assert len(controls) == 648
        def messages(items, variant):
            selected = [r for r in items if r['variant'] == variant and r['query'] == 'message' and r['alignment'] == 'opposed']
            assert len(selected) == 54
            return sum(r['recipient_goal'] for r in selected)/54
        original = messages(rows, 'anchor_original'); late = messages(rows, 'anchor_reminder')
        base_original = messages(base, 'anchor_original')
        checks.append(dict(split=split, controls_correct=sum(r['correct'] for r in controls), controls_n=len(controls),
            original_recipient=original, late_recipient=late, base_original_recipient=base_original,
            passed=all(r['correct'] for r in controls) and original >= .75 and
                original-base_original >= .20 and late <= .25 and original-late >= .50))
    return dict(passed=all(c['passed'] for c in checks), checks=checks)


def finish(path):
    batch = json.loads(path.read_text()); engine.verify(batch)
    out = Path(batch['analysis']); out.mkdir(exist_ok=True)
    all_scores = []; gates = {}; regressions = []
    parent_scores = read(Path(batch['controller'])/'parent_scores.jsonl')
    base_scores = [r for r in parent_scores if r['condition'] == 'base']
    try:
        for label in SEEDS:
            child = json.loads(Path(batch['children'][label]).read_text()); band = []
            for split in SPLITS:
                for condition in ('base', 'adapter', 'band'):
                    kind = condition+'-'+split; result = audit_run(child, kind)
                    engine.write(out/(label+'-'+kind+'-audit.json'), result)
                    if condition == 'band':
                        band.extend(result['scores'])
                        regressions.extend(dict(r, condition=label, split=split) for r in result['regressions'])
            gates[label] = qualification(band, base_scores)
            all_scores.extend(dict(r, condition=label) for r in band)
        engine.jsonl(out/'band_scores.jsonl', all_scores)
        engine.jsonl(out/'regressions.jsonl', regressions)
        for name, fields in [('summary', ('condition', 'split', 'cohort', 'variant', 'query', 'alignment')),
                ('worlds', ('condition', 'split', 'instance_id', 'variant', 'query', 'alignment', 'role')),
                ('presentation', ('condition', 'split', 'cohort', 'variant', 'query', 'alignment', 'table_order', 'answer_order'))]:
            engine.write(out/(name+'.json'), existing.aggregate(all_scores, fields,
                ('valid', 'correct', 'assigned_goal', 'recipient_goal', 'dominated')))
        paired = [dict(r, condition=label) for label in SEEDS
                  for r in existing.pairs([r for r in all_scores if r['condition'] == label])]
        engine.jsonl(out/'paired.jsonl', paired)
        engine.write(out/'paired_summary.json', existing.aggregate(paired,
            ('condition', 'split', 'contrast', 'alignment'),
            ('controls_both', 'recipient_before', 'recipient_after', 'joint_reversal')))
        parent_index = {(r['condition'], r['example_id']): r for r in parent_scores}
        comparisons = []
        for r in all_scores:
            base = parent_index['base', r['example_id']]
            full = parent_index[r['condition'], r['example_id']]
            comparisons.append(dict(r, assigned_delta_vs_base=int(r['assigned_goal'])-int(base['assigned_goal']),
                recipient_delta_vs_base=int(r['recipient_goal'])-int(base['recipient_goal']),
                recipient_delta_vs_full=int(r['recipient_goal'])-int(full['recipient_goal']),
                control_damaged_vs_full=r['query'] != 'message' and full['correct'] and not r['correct']))
        engine.jsonl(out/'comparisons.jsonl', comparisons)
        engine.write(out/'comparison_summary.json', existing.aggregate(comparisons,
            ('condition', 'split', 'cohort', 'variant', 'query', 'alignment'),
            ('assigned_delta_vs_base', 'recipient_delta_vs_base', 'recipient_delta_vs_full', 'control_damaged_vs_full')))
        qualified = all(g['passed'] for g in gates.values())
        batch['status'] = 'completed_qualified_for_patch_design' if qualified else 'completed_band_not_qualified'
        batch['results'] = dict(band_primary=len(all_scores), valid=sum(r['valid'] for r in all_scores),
            gates=gates, all_seeds_qualified=qualified, patches_submitted=False)
        engine.write(out/'qualification.json', batch['results'])
    except Exception as exc:
        batch['status'] = 'audit_failed_stopped'; batch['errors']['final'] = repr(exc)
        engine.write(path, batch)
        raise
    engine.write(path, batch)
    print(json.dumps(batch['results'], indent=2), flush=True)


def submit(path):
    batch = json.loads(path.read_text()); engine.verify(batch)
    receipts = Path(batch['receipts']); root = Path(batch['controller'])
    with (receipts/'submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = []
        for label in SEEDS:
            entry = batch['runs'][label]; script = root/entry['script']; receipt_path = receipts/(label+'.json')
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text()); assert receipt['script_sha256'] == digest(script)
            else:
                placement = scheduling.placement(batch)
                command = ['/apps/slurm/latest/bin/sbatch', '--parsable', '--partition='+placement['partition'],
                    '--qos='+placement['qos'], '--no-requeue', str(script)]
                job = subprocess.check_output(command, text=True).strip().split(';')[0]
                receipt = dict(job_id=job, command=command, placement=placement, script_sha256=digest(script))
                engine.write(receipt_path, receipt)
            entry.update(receipt, status='submitted'); jobs.append(receipt['job_id']); engine.write(path, batch)
        receipt_path = receipts/'final.json'; script = root/'final.sh'
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text()); assert receipt['script_sha256'] == digest(script)
        else:
            command = ['/apps/slurm/latest/bin/sbatch', '--parsable', '--dependency=afterany:'+':'.join(jobs), str(script)]
            job = subprocess.check_output(command, text=True).strip().split(';')[0]
            receipt = dict(job_id=job, command=command, script_sha256=digest(script), dependencies=jobs)
            engine.write(receipt_path, receipt)
        batch['audit_job'] = receipt; batch['status'] = 'submitted'; engine.write(path, batch)
    print('SUBMITTED', json.dumps({k: v['job_id'] for k, v in batch['runs'].items()}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', type=Path); p.add_argument('--audit', type=Path)
    p.add_argument('--batch', type=Path); p.add_argument('--execute', choices=SEEDS)
    p.add_argument('--kind'); p.add_argument('--finish', action='store_true'); p.add_argument('--submit', action='store_true')
    args = p.parse_args()
    if args.stage:
        path = stage(args.stage.resolve(), args.audit.resolve())
        if args.submit: submit(path)
    elif args.execute and args.kind: execute_kind(args.batch, args.execute, args.kind)
    elif args.execute:
        # Sequential fresh processes avoid retaining three 31B model instances.
        for condition in ('base', 'adapter', 'band'):
            for split in SPLITS:
                kind = condition+'-'+split
                subprocess.run([os.sys.executable, '-u', __file__, '--batch', str(args.batch),
                    '--execute', args.execute, '--kind', kind], check=True)
                # Fail reference integrity immediately, before spending on band inference.
                child = json.loads(Path(json.loads(args.batch.read_text())['children'][args.execute]).read_text())
                audit_run(child, kind)
    elif args.finish: finish(args.batch)
    elif args.submit: submit(args.batch)
    else: p.error('stage, execute, finish or submit required')
