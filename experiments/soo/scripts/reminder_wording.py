"""Freeze, gate and analyze wording/position controls across all three SOO seeds."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess

import extension_repeat_bridge as engine
import report_goal_scope as prior
import report_goal_seeds as seeds
import reminder_wording_task as task
from extension_repeat_selection import digest, read

LABELS = ('base', 'seed0', 'seed1', 'seed2')
SPLITS = ('development', 'reserved')
PARENT = 'report-goal-scope1-batch-20261027T230343834417Z.json'
SEED_PARENT = 'report-goal-seeds1-batch-20261028T001051474765Z.json'
write = engine.write


def configure():
    engine.TAGS = ('base', 'adapter'); engine.score = task.score; engine.gate = task.gate


def audit(batch, label, split):
    child = json.loads(Path(batch['children'][label]).read_text())
    engine.verify(child)
    tag = 'base' if label == 'base' else 'adapter'
    result = prior._original_audit(child, split, tag)
    if tag == 'adapter': seeds.adapter_binding(child, result['manifest'])
    else: assert result['manifest']['adapter'] is None
    entry = child['runs'][split][tag]
    records = {r['example_id']: r for r in read(Path(entry['output']) / 'planned_responses.jsonl') if r['phase'] == 'primary'}
    old = {r['example_id']: r for r in read(Path(batch['controller']) / 'anchors' / label / split / 'responses.jsonl')}
    comparisons = []
    for e in read(entry['data']):
        if e['variant'] not in task.ANCHORS: continue
        r = records[e['example_id']]
        ref = old['reportgoal1-' + task.ANCHORS[e['variant']] + '::' + e['source_example_id']]
        comparisons.append(dict(**{k: e[k] for k in ('example_id', 'variant', 'query', 'role', 'alignment', 'instance_id')},
            answer_changed=(r['completion_status'], r['response'].strip().casefold()) !=
                           (ref['completion_status'], ref['response'].strip().casefold()),
            tokens_changed=r['generated_token_ids'] != ref['generated_token_ids']))
    result['anchor_comparisons'] = comparisons
    return result


def analyze(results, out):
    assert set(results) == set(LABELS)
    rows = [dict(r, condition=label) for label, result in results.items() for r in result['scores']]
    metrics = ('valid', 'correct', 'assigned_goal', 'other_client', 'recipient_goal', 'dominated')
    for name, fields in [('summary', ('condition', 'variant', 'query', 'alignment')),
                         ('roles', ('condition', 'variant', 'query', 'alignment', 'role')),
                         ('instances', ('condition', 'variant', 'query', 'alignment', 'instance_id')),
                         ('presentation', ('condition', 'variant', 'query', 'alignment', 'table_order', 'answer_order'))]:
        write(out / (name + '.json'), engine.grouped(rows, fields, metrics))
    engine.jsonl(out / 'all_scores.jsonl', rows)
    indexed = {label: {r['example_id']: r for r in result['scores']} for label, result in results.items()}
    paired = []; joint = []; contrasts = []
    for label in LABELS:
        contexts = {}
        assert set(indexed[label]) == set(indexed['base'])
        for eid, r in indexed[label].items():
            base = indexed['base'][eid]
            original = indexed[label]['reminderword1-original::' + r['source_example_id']]
            paired.append(dict(r, condition=label, answer_changed_vs_base=r['answer'] != base['answer'],
                answer_changed_vs_original=r['answer'] != original['answer'],
                recipient_delta_vs_base=int(r['recipient_goal'])-int(base['recipient_goal']),
                correct_delta_vs_base=int(r['correct'])-int(base['correct'])))
            contexts.setdefault(r['context_id'], {})[r['query']] = r
        for cid, qs in contexts.items():
            assert set(qs) == set(task.QUERIES)
            m = qs['message']
            joint.append(dict(condition=label, context_id=cid,
                **{k: m[k] for k in ('variant', 'role', 'alignment', 'instance_id')},
                recipient_message_with_controls=m['recipient_goal'] and all(qs[q]['correct'] for q in ('private', 'fact0', 'fact1'))))
    for alignment in ('aligned', 'opposed'):
        for role in ('all', 0, 1):
            cells = {}
            for label in LABELS:
                cells[label] = {}
                for variant in task.VARIANTS:
                    selected = [r for r in indexed[label].values() if r['query'] == 'message' and r['alignment'] == alignment
                                and r['variant'] == variant and (role == 'all' or r['role'] == role)]
                    assert selected
                    cells[label][variant] = sum(r['recipient_goal'] for r in selected) / len(selected)
            for label in LABELS:
                for variant in task.VARIANTS:
                    contrasts.append(dict(condition=label, alignment=alignment, role=role,
                        contrast='condition_minus_base', variant=variant, rate=cells[label][variant],
                        base_rate=cells['base'][variant], value=cells[label][variant]-cells['base'][variant]))
                for left, right in task.PAIRS:
                    change = cells[label][left]-cells[label][right]
                    base_change = cells['base'][left]-cells['base'][right]
                    contrasts.append(dict(condition=label, alignment=alignment, role=role,
                        contrast=left+'_minus_'+right, left=left, right=right,
                        change=change, base_change=base_change, baseline_adjusted_change=change-base_change))
    anchors = [dict(r, condition=label) for label, result in results.items() for r in result['anchor_comparisons']]
    engine.jsonl(out / 'paired.jsonl', paired); engine.jsonl(out / 'joint.jsonl', joint)
    engine.jsonl(out / 'anchor_comparisons.jsonl', anchors)
    write(out / 'anchor_summary.json', engine.grouped(anchors, ('condition', 'variant', 'query', 'alignment'), ('answer_changed', 'tokens_changed')))
    write(out / 'contrasts.json', contrasts)
    return dict(summary=engine.grouped(rows, ('condition', 'variant', 'query', 'alignment'), metrics), contrasts=contrasts)


def launch(batch, path, label, split):
    key = label+'-'+split; entry = batch['runs'][key]
    receipt = engine.submit(Path(batch['controller']) / entry['script'], Path(batch['receipts']) / (key+'.json'))
    entry.update(receipt, status='submitted'); write(path, batch)
    return receipt['job_id']


def release(batch, phase, jobs):
    batch['audit_receipts'][phase] = engine.submit(Path(batch['controller']) / (phase+'.sh'),
        Path(batch['receipts']) / (phase+'.json'), jobs)
    batch['status'] = phase+'_submitted'


def advance(path, phase):
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        batch = json.loads(path.read_text())
        if batch['status'] == 'completed_and_analyzed' or (phase == 'baseline' and batch['status'] == 'final_submitted'): return
        try:
            engine.verify(batch)
            out = Path(batch['analysis']) / phase; out.mkdir(parents=True, exist_ok=True)
            labels = ('base',) if phase == 'baseline' else LABELS
            results = {}; errors = {}
            for label in labels:
                for split in SPLITS:
                    key = label+'-'+split
                    try:
                        r = audit(batch, label, split); results[label, split] = r
                        write(out / (key+'-audit.json'), r)
                        batch['runs'][key]['status'] = 'completed_and_audited'
                    except Exception as exc:
                        errors[key] = repr(exc); batch['runs'][key]['status'] = 'inspection_needed'
            batch['errors'][phase] = errors
            if errors: batch['status'] = 'execution_discrepancy_stopped'; return
            for split in SPLITS:
                batch['gates'][split] = results['base', split]['gate']
            if not all(g['passed'] for g in batch['gates'].values()):
                batch['status'] = 'baseline_competence_failed_stopped'; return
            if phase == 'baseline':
                release(batch, 'final', [launch(batch, path, label, split) for label in LABELS[1:] for split in SPLITS])
            else:
                for split in SPLITS:
                    dest = out / split; dest.mkdir(exist_ok=True)
                    batch['results'][split] = analyze({label: results[label, split] for label in LABELS}, dest)
                batch['status'] = 'completed_and_analyzed'
        except Exception as exc:
            batch.update(status='controller_error_requires_inspection', error=repr(exc)); raise
        finally:
            batch['updated_utc'] = datetime.now(timezone.utc).isoformat(); write(path, batch)


def stage(repo):
    soo = repo / 'experiments/soo'; study = soo / 'results/study2'
    parent_path = study / PARENT; seed_path = study / SEED_PARENT
    parent = json.loads(parent_path.read_text()); seed_parent = json.loads(seed_path.read_text())
    prior.configure(); seeds.parent_audits(parent)
    engine.verify(seed_parent)
    assert seed_parent['status'] == 'completed_and_analyzed' and not seed_parent['errors']
    for seed in (1, 2):
        child = json.loads(Path(seed_parent['children'][str(seed)]).read_text())
        for split in SPLITS: seeds.audit_seed(child, split, parent)
    configure()
    source = Path(parent['controller']); seed_source = Path(seed_parent['controller'])
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root = study / f'reminder-wording1-controller-{stamp}'; root.mkdir()
    path = study / f'reminder-wording1-batch-{stamp}.json'
    for name in ('reminder_wording.py', 'reminder_wording_task.py'):
        shutil.copy2(soo / 'scripts' / name, root / name)
    for name in ('report_goal_scope.py', 'report_goal_task.py', 'report_components_task.py', 'payoff_context_task.py',
                 'extension_repeat_bridge.py', 'extension_repeat_selection.py', 'extension_bridge_task.py', 'validate_goals_roles_tokens.py'):
        shutil.copy2(source / name, root / name)
        assert digest(root / name) == parent['frozen_sha256'][name]
    shutil.copy2(seed_source / 'report_goal_seeds.py', root / 'report_goal_seeds.py')
    assert digest(root / 'report_goal_seeds.py') == seed_parent['frozen_sha256']['report_goal_seeds.py']
    shutil.copy2(soo / 'REMINDER_WORDING_POSITION.md', root / 'REMINDER_WORDING_POSITION.md')
    shutil.copy2(repo / 'tests/test_soo_reminder_wording.py', root / 'test_soo_reminder_wording.py')
    for directory in ('src', 'historical_data'):
        shutil.copytree(source / directory, root / directory, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    current = {str(p.relative_to(root / 'src/selfconcept')): digest(p) for p in (root / 'src/selfconcept').rglob('*.py')}
    assert current == parent['inference_source_sha256']
    refs = {str(parent_path): digest(parent_path), str(seed_path): digest(seed_path), **parent['reference_sha256'], **seed_parent['reference_sha256']}
    (root / 'data').mkdir(); (root / 'plans').mkdir()
    data_by_split = {}; common_refs = {}
    for split in SPLITS:
        original_path = source / 'anchors' / split / 'original_examples.jsonl'
        originals = read(original_path); assert len(originals) == 432
        refs[str(original_path)] = digest(original_path)
        rows = task.build(originals); assert len(rows) == 3024; data_by_split[split] = rows
        datafile = root / 'data' / (split+'.jsonl'); engine.jsonl(datafile, rows)
        preflight = root / 'preflight' / split
        env = dict(os.environ, PYTHONPATH=str(root / 'src'), OPENSSL_CONF='/dev/null', HF_HUB_OFFLINE='1', USE_TORCH='0')
        subprocess.run([str(repo / '.venv-313/bin/python'), str(root / 'validate_goals_roles_tokens.py'), '--data', str(datafile), '--out', str(preflight)], env=env, check=True)
        native = {r['example_id']: r['input_ids'] for r in read(preflight / 'token_spans.jsonl')}
        oldplan_path = source / parent['runs'][split]['base']['plan']
        oldplan = json.loads(oldplan_path.read_text()); refs[str(oldplan_path)] = digest(oldplan_path)
        for e in rows:
            if e['variant'] in task.ANCHORS:
                eid = 'reportgoal1-'+task.ANCHORS[e['variant']]+'::'+e['source_example_id']
                assert native[e['example_id']] == oldplan['expected_inputs'][eid]
        ids = [e['example_id'] for e in rows]; batches = [ids[i:i+8] for i in range(0, len(ids), 8)]
        for variant, old_variant in task.ANCHORS.items():
            selected = [e for e in rows if e['variant'] == variant]
            old_ids = ['reportgoal1-'+old_variant+'::'+e['source_example_id'] for e in selected]
            wanted = set(old_ids)
            old_batches = next(p['batches'] for p in oldplan['phases'] if p['name'] == 'primary')
            chosen = [b for b in old_batches if wanted.intersection(b)]
            assert all(len(b) == 8 and set(b) <= wanted for b in chosen)
            assert [eid for b in chosen for eid in b] == old_ids
        sentinel = []
        for variant in task.VARIANTS:
            for role in (0, 1):
                for alignment in ('aligned', 'opposed'):
                    i = next(i for i, e in enumerate(rows) if (e['variant'], e['role'], e['alignment']) == (variant, role, alignment)) // 8
                    if batches[i] not in sentinel: sentinel.append(batches[i])
        assert len(sentinel) == 28
        planfile = root / 'plans' / (split+'.json')
        write(planfile, dict(expected_inputs=native, phases=[dict(name='sentinel_before', batches=sentinel),
            dict(name='primary', batches=batches), dict(name='sentinel_after', batches=sentinel)], generations=3472))
        for f in (datafile, planfile, preflight / 'token_spans.jsonl'): common_refs[str(f)] = digest(f)
    runs = {}; children = {}
    for label in LABELS:
        tag = 'base' if label == 'base' else 'adapter'
        child_root = root / label; child_root.mkdir()
        seed = None if label == 'base' else int(label[-1])
        adapter_hashes = {}
        if seed is not None:
            adapter_source = source / 'adapter' if seed == 0 else seed_source / label / 'adapter'
            shutil.copytree(adapter_source, child_root / 'adapter')
            assert json.loads((child_root / 'adapter/train_log.json').read_text())['seed'] == seed
            adapter_hashes = {name: digest(child_root / 'adapter' / name) for name in ('adapter_config.json', 'adapter_model.safetensors')}
        # The inherited runner reads regression data relative to its controller.
        shutil.copytree(root / 'historical_data', child_root / 'historical_data')
        write(child_root / 'provenance.json', dict(protocol='reminder_wording1', condition=label,
            parent=str(parent_path), seed_parent=str(seed_path), new_holdout=False))
        historical_reference = child_root / 'history_reference'; (historical_reference / 'output').mkdir(parents=True)
        child_runs = {}; historical_rows = None
        for split in SPLITS:
            old_entry = parent['runs'][split][tag] if label in ('base', 'seed0') else seed_parent['runs'][label+'-'+split]
            old_out = Path(old_entry['output'])
            for f in old_out.iterdir():
                if f.is_file() and f.name != '.lock': refs[str(f)] = digest(f)
            old_history = read(old_out / 'regressions.jsonl')
            if historical_rows is None: historical_rows = old_history
            else: assert historical_rows == old_history
            anchor_path = root / 'anchors' / label / split; anchor_path.mkdir(parents=True)
            prefixes = tuple('reportgoal1-'+v+'::' for v in task.ANCHORS.values())
            records = [r for r in read(old_out / 'planned_responses.jsonl') if r['phase'] == 'primary' and r['example_id'].startswith(prefixes)]
            assert len(records) == 864 and len({r['example_id'] for r in records}) == 864
            engine.jsonl(anchor_path / 'responses.jsonl', records)
            entry = dict(condition=tag, alpha=1.0, data=str(root / 'data' / (split+'.jsonl')),
                plan=str(root / 'plans' / (split+'.json')), preflight=str(root / 'preflight' / split),
                output=str(study / f'reminder-wording1-output-{stamp}' / label / split), status='staged_not_submitted')
            child_runs[split] = {tag: entry}
            key = label+'-'+split
            script = engine.shell(repo, root, path, key, [str(repo / '.venv-313/bin/python'), '-u', str(root / 'reminder_wording.py'),
                '--batch', str(path), '--execute', split, '--condition', label], True)
            runs[key] = dict(condition=label, split=split, script=script, output=entry['output'], status='staged_not_submitted')
        engine.jsonl(historical_reference / 'output/regressions.jsonl', historical_rows)
        child = dict(controller=str(child_root), seed=seed, revision=parent['revision'], reference_versions=parent['reference_versions'],
            inference_source_sha256=current, adapter_sha256=adapter_hashes, history_references={tag: str(historical_reference)},
            reference_sha256=common_refs, frozen_sha256=engine.hashes(child_root), runs=child_runs)
        child_path = root / (label+'-batch.json'); write(child_path, child); children[label] = str(child_path)
    picker = subprocess.check_output(['/usr/bin/python3.11', str(repo / 'gpu_pick.py'), '--hours', '1', '--vram-gb', '75', '--cpus', '4',
        '--mem-gb', '96', '--no-requeue-ok', '--fast', '--cuda', '13.0', '--top', '10'], text=True)
    (root / 'gpu_pick.txt').write_text(picker)
    for phase in ('baseline', 'final'):
        engine.shell(repo, root, path, phase, ['/usr/bin/python3.11', '-u', str(root / 'reminder_wording.py'), '--batch', str(path), '--advance', phase], False)
    receipts = study / f'reminder-wording1-receipts-{stamp}'; receipts.mkdir()
    batch = dict(protocol='reminder_wording1', status='staged_not_submitted',
        authorization='User asked to keep digging following the proposed bounded reminder wording/position comparison.',
        created_utc=datetime.now(timezone.utc).isoformat(), controller=str(root), parent=str(parent_path), seed_parent=str(seed_path),
        analysis=str(study / f'reminder-wording1-analysis-{stamp}'), receipts=str(receipts), children=children, runs=runs,
        maximum_primary_responses=24192, reference_sha256=refs, frozen_sha256=engine.hashes(root), errors={}, gates={}, results={}, audit_receipts={})
    write(path, batch); engine.verify(batch); print('FROZEN', path, flush=True)


def submit(path):
    batch = json.loads(path.read_text()); engine.verify(batch)
    assert batch['status'] in ('staged_not_submitted', 'submitting')
    batch['status'] = 'submitting'; write(path, batch)
    release(batch, 'baseline', [launch(batch, path, 'base', split) for split in SPLITS])
    write(path, batch)
    print('SUBMITTED', {k: v.get('job_id') for k, v in batch['runs'].items()}, 'AUDIT', batch['audit_receipts'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', type=Path); p.add_argument('--batch', type=Path); p.add_argument('--submit', action='store_true')
    p.add_argument('--advance', choices=('baseline', 'final')); p.add_argument('--execute', choices=SPLITS)
    p.add_argument('--condition', choices=LABELS)
    args = p.parse_args(); configure()
    if args.stage: stage(args.stage.resolve())
    elif args.batch and args.submit: submit(args.batch)
    elif args.batch and args.advance: advance(args.batch, args.advance)
    elif args.batch and args.execute and args.condition:
        batch = json.loads(args.batch.read_text()); engine.verify(batch)
        engine.execute(Path(batch['children'][args.condition]), args.execute, 'base' if args.condition == 'base' else 'adapter')
    else: p.error('stage, submit, advance or execute/condition required')
