"""Freeze and audit the unchanged report-goal comparison on both additional seeds."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import extension_repeat_bridge as engine
import report_goal_scope as parent_workflow
import report_goal_task as task
from extension_repeat_selection import digest, read

SEEDS = (1, 2)
SPLITS = ('development', 'reserved')
PARENT = 'report-goal-scope1-batch-20261027T230343834417Z.json'
write = engine.write


def adapter_binding(batch, manifest):
    root = Path(batch['controller'])
    assert manifest['adapter'] == str(root / 'adapter'), 'wrong adapter path'
    actual = manifest['artifact_sha256']
    for name, expected in batch['adapter_sha256'].items():
        path = root / 'adapter' / name
        assert actual.get(str(path)) == expected == digest(path), 'wrong adapter hash'
    assert json.loads((root / 'adapter/train_log.json').read_text())['seed'] == batch['seed']


def regression_comparisons(current, reference):
    key = lambda r: (r['orientation'], r['scenario'], r['example_id'])
    assert len(current) == len(reference) == 32
    assert len({key(r) for r in current}) == 32
    assert [key(r) for r in current] == [key(r) for r in reference]
    comparisons = []
    for r, ref in zip(current, reference):
        assert r['prompt_tokens'] == ref['prompt_tokens']
        assert r['completion_status'] in ('complete', 'truncated')
        assert r['generated_tokens'] == len(r['generated_token_ids'])
        comparisons.append(dict(orientation=r['orientation'], scenario=r['scenario'],
            example_id=r['example_id'], label=r['label'], seed0_label=ref['label'],
            answer_changed=(r['completion_status'], r['response'].strip().casefold()) !=
                           (ref['completion_status'], ref['response'].strip().casefold()),
            tokens_changed=r['generated_token_ids'] != ref['generated_token_ids']))
    return comparisons


def audit_seed(batch, split, parent):
    """Parent integrity checks, with explicit intended-seed and historical handling."""
    engine.verify(batch)
    entry = batch['runs'][split]['adapter']
    root = Path(batch['controller']); out = Path(entry['output'])
    plan = json.loads((root / entry['plan']).read_text())
    examples = read(root / entry['data'])
    assert json.loads((out / 'execution_complete.json').read_text()) == dict(
        phases=len(plan['phases']), generations=plan['generations'])
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['source_sha256'] == batch['inference_source_sha256']
    assert manifest['condition'] == 'adapter' and manifest['offset_alpha'] == 1.0
    assert manifest['revision'] == batch['revision'] and manifest['model'] == 'google/gemma-4-31B-it'
    assert manifest['batch_size'] == 8 and manifest['max_new_tokens'] == 32 and manifest['do_sample'] is False
    assert manifest['dtype'] == 'bfloat16' and manifest['attention'] == 'sdpa'
    assert manifest['chat_kwargs'] == {'enable_thinking': False}
    assert manifest['versions'] == batch['reference_versions'] and 'A100' in manifest['hardware']['gpu']
    adapter_binding(batch, manifest)
    for path, expected in manifest['artifact_sha256'].items():
        assert digest(path) == expected
    assert manifest['example_ids'] == [e['example_id'] for e in examples]
    inputs = read(out / 'inputs.jsonl')
    assert inputs == read(root / entry['preflight'] / 'token_spans.jsonl')
    assert manifest['prompt_ids_sha256'] == hashlib.sha256(json.dumps([r['input_ids'] for r in inputs]).encode()).hexdigest()
    assert {r['example_id']: r['input_ids'] for r in inputs} == plan['expected_inputs']
    responses = read(out / 'planned_responses.jsonl')
    expected = [(p['name'], i, ids, eid) for p in plan['phases']
                for i, ids in enumerate(p['batches']) for eid in ids]
    assert len(responses) == len(expected) == plan['generations']
    by_phase = defaultdict(dict)
    for r, (phase, index, ids, eid) in zip(responses, expected):
        assert (r['phase'], r['batch_index'], r['batch_members'], r['example_id']) == (phase, index, ids, eid)
        assert r['padded_width'] == max(len(plan['expected_inputs'][i]) for i in ids)
        assert r['prompt_tokens'] == len(plan['expected_inputs'][eid])
        assert r['generated_tokens'] == len(r['generated_token_ids'])
        assert r['completion_status'] in ('complete', 'truncated')
        assert eid not in by_phase[phase]
        by_phase[phase][eid] = r
    for phase in ('sentinel_before', 'sentinel_after'):
        assert all(engine.raw_only(r) == engine.raw_only(by_phase['primary'][eid])
                   for eid, r in by_phase[phase].items()), 'sentinel changed'
    seed0_out = Path(parent['runs'][split]['adapter']['output'])
    historical = read(out / 'regressions.jsonl')
    historical_comparisons = regression_comparisons(historical, read(seed0_out / 'regressions.jsonl'))
    seed0 = {r['example_id']: r for r in read(seed0_out / 'planned_responses.jsonl') if r['phase'] == 'primary'}
    comparisons = []
    for e in examples:
        r = by_phase['primary'][e['example_id']]; ref = seed0[e['example_id']]
        comparisons.append(dict(**{k: e[k] for k in ('example_id', 'variant', 'query', 'role', 'alignment', 'instance_id')},
            answer_changed=(r['completion_status'], r['response'].strip().casefold()) !=
                           (ref['completion_status'], ref['response'].strip().casefold()),
            tokens_changed=r['generated_token_ids'] != ref['generated_token_ids']))
    return dict(checks_passed=True, scores=task.score(examples, list(by_phase['primary'].values())),
        anchor_comparisons=[], seed0_comparisons=comparisons,
        historical_comparisons=historical_comparisons, historical=historical, manifest=manifest)


def parent_audits(parent):
    engine.verify(parent)
    assert parent['status'] == 'completed_and_analyzed'
    assert not any(parent['errors'].values())
    results = {}
    for split in SPLITS:
        results[split] = {}
        for tag in ('base', 'adapter'):
            result = parent_workflow._original_audit(parent, split, tag)
            result['anchor_comparisons'] = []
            results[split][tag] = result
        assert results[split]['base']['gate']['passed']
        assert len(results[split]['base']['gate']['checks']) == 952
    return results


def summarize_all(parent, results):
    """Every seed and cell is retained; no sign or competence-based selection."""
    summaries = []; targets = []; effects = []
    for split in SPLITS:
        base = parent['results'][split]['summary']
        for seed in (0, 1, 2):
            summary = base if seed == 0 else results[str(seed)][split]['summary']
            summaries.extend(dict(r, seed=seed, split=split) for r in summary if r['condition'] == 'adapter')
            for variant in ('original', 'original_reminder'):
                rows = [r for r in summary if r['query'] == 'message' and r['alignment'] == 'opposed' and r['variant'] == variant]
                assert {r['condition'] for r in rows} == {'base', 'adapter'}
                counts = {r['condition']: r for r in rows}
                targets.append(dict(seed=seed, split=split, variant=variant,
                    n=counts['adapter']['n'], base=counts['base']['recipient_goal'], adapter=counts['adapter']['recipient_goal']))
            original, reminder = targets[-2:]
            assert original['n'] == reminder['n']
            base_change = (reminder['base'] - original['base']) / original['n']
            adapter_change = (reminder['adapter'] - original['adapter']) / original['n']
            effects.append(dict(seed=seed, split=split, base_change=base_change,
                adapter_change=adapter_change, baseline_adjusted_change=adapter_change-base_change))
    return dict(seed_summaries=summaries, primary_counts=targets, primary_effects=effects)


def advance(path):
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        batch = json.loads(path.read_text())
        if batch['status'] == 'completed_and_analyzed': return
        try:
            engine.verify(batch)
            parent = json.loads(Path(batch['parent']).read_text())
            audited_parent = parent_audits(parent)
            out = Path(batch['analysis']); out.mkdir(exist_ok=True)
            audited = {}; errors = {}
            for seed in SEEDS:
                child = json.loads(Path(batch['children'][str(seed)]).read_text())
                for split in SPLITS:
                    key = f'seed{seed}-{split}'
                    try:
                        result = audit_seed(child, split, parent)
                        audited[seed, split] = result
                        write(out / (key + '-audit.json'), result)
                        batch['runs'][key]['status'] = 'completed_and_audited'
                    except Exception as exc:
                        errors[key] = repr(exc)
                        batch['runs'][key]['status'] = 'inspection_needed'
                if all((seed, split) in audited for split in SPLITS):
                    if audited[seed, 'development']['historical'] != audited[seed, 'reserved']['historical']:
                        errors[f'seed{seed}-historical-repeat'] = 'historical outputs changed between runs'
            batch['errors'] = errors
            if errors:
                batch['status'] = 'execution_discrepancy_stopped'; return
            results = {}
            for seed in SEEDS:
                results[str(seed)] = {}
                for split in SPLITS:
                    dest = out / f'seed{seed}' / split; dest.mkdir(parents=True, exist_ok=True)
                    result = audited[seed, split]
                    results[str(seed)][split] = parent_workflow.analyze(
                        dict(base=audited_parent[split]['base'], adapter=result), dest)
                    engine.jsonl(dest / 'seed0_comparisons.jsonl', result['seed0_comparisons'])
                    write(dest / 'seed0_summary.json', engine.grouped(result['seed0_comparisons'],
                        ('variant', 'query', 'alignment'), ('answer_changed', 'tokens_changed')))
                    write(dest / 'historical_vs_seed0.json', result['historical_comparisons'])
            batch['results'] = results
            write(out / 'all_seeds.json', summarize_all(parent, results))
            batch['status'] = 'completed_and_analyzed'
        except Exception as exc:
            batch.update(status='controller_error_requires_inspection', error=repr(exc))
            raise
        finally:
            batch['updated_utc'] = datetime.now(timezone.utc).isoformat()
            write(path, batch)


def stage(repo):
    soo = repo / 'experiments/soo'; study = soo / 'results/study2'
    parent_path = study / PARENT; parent = json.loads(parent_path.read_text())
    parent_workflow.configure()
    parent_audits(parent)
    source = Path(parent['controller'])
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root = study / f'report-goal-seeds1-controller-{stamp}'; root.mkdir()
    path = study / f'report-goal-seeds1-batch-{stamp}.json'
    refs = {str(parent_path): digest(parent_path), **parent['reference_sha256']}
    for split in SPLITS:
        for tag in ('base', 'adapter'):
            for f in Path(parent['runs'][split][tag]['output']).iterdir():
                if f.is_file() and f.name != '.lock': refs[str(f)] = digest(f)
    for name in ('REPORT_GOAL_SEEDS.md',): shutil.copy2(soo / name, root / name)
    shutil.copy2(repo / 'tests/test_soo_report_goal_seeds.py', root / 'test_soo_report_goal_seeds.py')
    names = ('report_goal_scope.py', 'report_goal_task.py', 'report_components_task.py',
             'payoff_context_task.py', 'extension_repeat_bridge.py',
             'extension_repeat_selection.py', 'extension_bridge_task.py')
    shutil.copy2(soo / 'scripts/report_goal_seeds.py', root / 'report_goal_seeds.py')
    for name in names:
        shutil.copy2(source / name, root / name)
        assert digest(root / name) == parent['frozen_sha256'][name]
    shutil.copytree(source / 'src', root / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    adapters = soo / 'results/checkpoints/gemma4-31b-L32'
    training = json.loads((adapters / 'seed0/train_log.json').read_text())['config']
    assert digest(adapters / 'seed0/adapter_model.safetensors') == digest(source / 'adapter/adapter_model.safetensors')
    configs = json.loads((source / 'adapter/adapter_config.json').read_text())
    weights = [digest(source / 'adapter/adapter_model.safetensors')]
    runs = {}; children = {}
    for seed in SEEDS:
        child = root / f'seed{seed}'; child.mkdir()
        for directory in ('src', 'data', 'plans', 'preflight', 'historical_data'):
            shutil.copytree(source / directory, child / directory, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        adapter = adapters / f'seed{seed}'
        assert json.loads((adapter / 'adapter_config.json').read_text()) == configs
        log = json.loads((adapter / 'train_log.json').read_text())
        assert log['seed'] == seed and log['config'] == training
        shutil.copytree(adapter, child / 'adapter')
        weights.append(digest(child / 'adapter/adapter_model.safetensors'))
        for f in adapter.iterdir():
            if f.is_file(): refs[str(f)] = digest(f)
        write(child / 'provenance.json', dict(protocol='report_goal_seeds1', seed=seed,
            parent=str(parent_path), unchanged_prompts=True, new_holdout=False))
        current = {str(p.relative_to(child / 'src/selfconcept')): digest(p)
                   for p in (child / 'src/selfconcept').rglob('*.py')}
        assert current == parent['inference_source_sha256']
        child_runs = {}
        for split in SPLITS:
            old = parent['runs'][split]['adapter']
            for field in ('data', 'plan'):
                assert digest(child / old[field]) == digest(source / old[field])
            assert digest(child / old['preflight'] / 'token_spans.jsonl') == digest(source / old['preflight'] / 'token_spans.jsonl')
            entry = {k: old[k] for k in ('condition', 'alpha', 'data', 'plan', 'preflight')}
            entry.update(output=str(study / f'report-goal-seeds1-output-{stamp}' / f'seed{seed}' / split), status='staged_not_submitted')
            child_runs[split] = {'adapter': entry}
            key = f'seed{seed}-{split}'
            script = engine.shell(repo, root, path, key,
                [str(repo / '.venv-313/bin/python'), '-u', str(root / 'report_goal_seeds.py'),
                 '--batch', str(path), '--execute', split, '--seed', str(seed)], True)
            runs[key] = dict(seed=seed, split=split, script=script, output=entry['output'], status='staged_not_submitted')
        child_batch = dict(controller=str(child), seed=seed, revision=parent['revision'],
            reference_versions=parent['reference_versions'], inference_source_sha256=current,
            adapter_sha256={n: digest(child / 'adapter' / n) for n in ('adapter_config.json', 'adapter_model.safetensors')},
            reference_sha256={}, frozen_sha256=engine.hashes(child), runs=child_runs)
        child_path = root / f'seed{seed}-batch.json'; write(child_path, child_batch)
        children[str(seed)] = str(child_path)
    assert len(set(weights)) == 3, 'adapters must have distinct weights'
    picker = subprocess.check_output(['/usr/bin/python3.11', str(repo / 'gpu_pick.py'), '--hours', '1',
        '--vram-gb', '75', '--cpus', '4', '--mem-gb', '96', '--no-requeue-ok', '--fast', '--cuda', '13.0', '--top', '10'], text=True)
    (root / 'gpu_pick.txt').write_text(picker)
    engine.shell(repo, root, path, 'final-audit', ['/usr/bin/python3.11', '-u', str(root / 'report_goal_seeds.py'), '--batch', str(path), '--advance'], False)
    receipts = study / f'report-goal-seeds1-receipts-{stamp}'; receipts.mkdir()
    batch = dict(protocol='report_goal_seeds1', status='staged_not_submitted',
        authorization='User asked to keep digging after the goal-reminder finding; bounded replication on both preselected existing adapters.',
        created_utc=datetime.now(timezone.utc).isoformat(), controller=str(root), parent=str(parent_path),
        analysis=str(study / f'report-goal-seeds1-analysis-{stamp}'), receipts=str(receipts), children=children,
        runs=runs, maximum_new_primary_responses=13824, reference_sha256=refs,
        frozen_sha256=engine.hashes(root), errors={}, results={}, audit_receipts={})
    write(path, batch); engine.verify(batch)
    print('FROZEN', path, flush=True)


def submit(path):
    batch = json.loads(path.read_text()); engine.verify(batch)
    assert batch['status'] in ('staged_not_submitted', 'submitting'), batch['status']
    root = Path(batch['controller']); receipts = Path(batch['receipts'])
    batch['status'] = 'submitting'; write(path, batch)
    jobs = []
    for key, entry in batch['runs'].items():
        receipt = engine.submit(root / entry['script'], receipts / (key + '.json'))
        entry.update(receipt, status='submitted'); jobs.append(receipt['job_id'])
        write(path, batch)
    batch['audit_receipts']['final'] = engine.submit(root / 'final-audit.sh', receipts / 'final-audit.json', jobs)
    batch['status'] = 'submitted'; write(path, batch)
    print('SUBMITTED', jobs, 'audit', batch['audit_receipts']['final']['job_id'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', type=Path); p.add_argument('--batch', type=Path)
    p.add_argument('--submit', action='store_true'); p.add_argument('--advance', action='store_true')
    p.add_argument('--execute', choices=SPLITS); p.add_argument('--seed', type=int, choices=SEEDS)
    args = p.parse_args(); parent_workflow.configure()
    if args.stage: stage(args.stage.resolve())
    elif args.batch and args.submit: submit(args.batch)
    elif args.batch and args.advance: advance(args.batch)
    elif args.batch and args.execute and args.seed:
        batch = json.loads(args.batch.read_text()); engine.verify(batch)
        engine.execute(Path(batch['children'][str(args.seed)]), args.execute, 'adapter')
    else: p.error('stage, submit, advance, or execute with seed required')
