"""Bounded base-only diagnosis of the failed truth-versus-beneficiary baseline."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import extension_repeat_bridge as engine
from extension_repeat_selection import digest, read
import truth_beneficiary_revision as parent_flow
import truth_beneficiary_revision_task as task
from selfconcept.soo.goals_roles import PromptBuilder

import truth_revision_diagnostic_task as probes
from truth_revision_diagnostic_task import build, score, independent_key

PARENT = 'truth-beneficiary2-batch-20261029T174757566050Z.json'
KINDS = probes.KINDS
METRICS = ('valid', 'correct', 'truthful', 'assigned_goal', 'other_client', 'recipient_goal', 'compromise')


def configure():
    engine.TAGS = ('base',); engine.score = score; engine.gate = lambda rows: None


def diagnose_parent(batch):
    parent_flow.configure()
    results = {}
    for split in parent_flow.SPLITS:
        result = parent_flow.audit(batch, 'base', split)
        assert result['checks_passed'] and not result['gate']['passed']
        results[split] = dict(integrity_passed=True, competence_passed=False,
            failed_checks=sum(not c['passed'] for c in result['gate']['checks']))
    configure()
    return results


def anchors(parent):
    rows = []; batches = []; raw = {}; native = {}
    for split in parent_flow.SPLITS:
        root = Path(parent['controller'])
        examples = read(root/'data'/f'{split}.jsonl'); indexed = {e['example_id']: e for e in examples}
        plan = json.loads((root/'plans'/f'{split}.json').read_text())
        primary = next(p['batches'] for p in plan['phases'] if p['name'] == 'primary')
        locations = {eid: i for i, batch in enumerate(primary) for eid in batch}
        selected = set()
        for template in sorted({e['template'] for e in examples}):
            for variant, role in itertools.product(task.VARIANTS, (0, 1)):
                e = next(e for e in examples if e['template'] == template and e['variant'] == variant
                         and e['role'] == role and e['alignment'] == 'opposed' and e['truth_relation'] == 'other'
                         and e['query'] == 'fact_state' and e['table_order'] == e['answer_order'] == 0)
                selected.add(locations[e['example_id']])
        for i in sorted(selected):
            batches.append(primary[i])
            for eid in primary[i]:
                e = deepcopy(indexed[eid]); e.update(kind='anchor', diagnostic_arm='anchor', encoding='table', column_order=0,
                    substitution='none', parent_identical=True, parent_example_id=eid)
                rows.append(e); native[eid] = plan['expected_inputs'][eid]
        source = {r['example_id']: r for r in read(Path(parent['runs']['base-'+split]['output'])/'planned_responses.jsonl')
                  if r['phase'] == 'primary'}
        raw.update({eid: engine.raw_only(source[eid]) for batch in batches for eid in batch if eid in source})
    assert len(rows) == 128 and len(batches) == 16 and len(raw) == 128
    return rows, batches, raw, native


def make_plan(rows, native, anchor_batches):
    main = [e for e in rows if e['diagnostic_arm'] != 'anchor']
    ids = [e['example_id'] for e in main]
    batches = [ids[i:i+8] for i in range(0, len(ids), 8)] + anchor_batches
    assert all(len(b) == 8 for b in batches)
    assert [eid for b in batches for eid in b] == [e['example_id'] for e in rows]
    seen = set(); selected = set()
    for i, e in enumerate(rows):
        key = tuple(e[k] for k in ('diagnostic_arm', 'template', 'encoding', 'column_order', 'substitution', 'variant', 'role', 'query'))
        if key not in seen: seen.add(key); selected.add(i//8)
    sentinels = [batches[i] for i in sorted(selected)]
    return dict(expected_inputs=native, phases=[dict(name='sentinel_before', batches=sentinels),
        dict(name='primary', batches=batches), dict(name='sentinel_after', batches=sentinels)],
        generations=len(rows)+2*sum(map(len, sentinels)))


def analyze(rows, out):
    main = [r for r in rows if r['diagnostic_arm'] != 'anchor']
    fields = ('kind', 'diagnostic_arm', 'template', 'encoding', 'column_order', 'substitution', 'variant', 'query')
    engine.jsonl(out/'all_scores.jsonl', rows)
    for name, extra in [('summary', ()), ('worlds', ('instance_id',)),
                        ('roles', ('role', 'alignment', 'truth_relation')),
                        ('presentation', ('instance_id', 'table_order', 'answer_order'))]:
        engine.write(out/(name+'.json'), engine.grouped(main, fields+extra, METRICS))
    # Every probe is paired to the unchanged full-context condition for the
    # SAME parent case. Renamed labels compare semantic row slots, not strings.
    reference = {r['parent_example_id']: r for r in main
                 if r['parent_identical'] and r['kind'] in ('table', 'contexts')}
    paired = []
    for r in main:
        base = reference[r['parent_example_id']]
        paired.append(dict(r, reference_id=base['example_id'],
            answer_changed=r['answer_slot'] != base['answer_slot'],
            correct_gain=int(r['correct'])-int(base['correct']),
            rescued=not base['correct'] and r['correct'],
            damaged=base['correct'] and not r['correct']))
    engine.jsonl(out/'paired.jsonl', paired)
    for name, extra in [('paired_summary', ()), ('paired_worlds', ('instance_id',)),
                        ('paired_presentation', ('instance_id', 'table_order', 'answer_order'))]:
        engine.write(out/(name+'.json'), engine.grouped(paired, fields+extra,
            ('answer_changed', 'correct_gain', 'rescued', 'damaged')))
    # Explicit reminder contrasts, including nominal repeats when a removed
    # goal no longer exists. Such repeats are not independent conditions.
    indexed = {r['example_id']: r for r in main}; transitions = []
    for r in main:
        if r['variant'] == 'goal_late':
            before = indexed[r['example_id'].replace('truthbenefit2-goal_late::', 'truthbenefit2-original::')]
            transitions.append(dict(r, rescued=not before['correct'] and r['correct'],
                damaged=before['correct'] and not r['correct']))
    engine.write(out/'reminder_transitions.json', engine.grouped(transitions,
        tuple(k for k in fields if k != 'variant')+('instance_id',), ('rescued', 'damaged')))
    tracking = defaultdict(list)
    for r in main:
        if r['query'] == 'fact_state':
            key = tuple(r[k] for k in fields+('instance_id', 'role', 'alignment', 'table_order', 'answer_order'))
            tracking[key].append(r)
    tracking_rows = []
    for key, members in tracking.items():
        assert len(members) == 3 and {r['truth_relation'] for r in members} == set(task.RELATIONS)
        tracking_rows.append(dict(zip(fields+('instance_id', 'role', 'alignment', 'table_order', 'answer_order'), key),
            all_three_correct=all(r['correct'] for r in members),
            all_three_assigned=all(r['assigned_goal'] for r in members)))
    engine.write(out/'tracking.json', tracking_rows)
    engine.write(out/'tracking_summary.json', engine.grouped(tracking_rows, fields,
        ('all_three_correct', 'all_three_assigned')))
    return dict(diagnostic_responses=len(main), primary_with_anchors=len(rows),
        valid=sum(r['valid'] for r in main), output=str(out), no_adapter_release=True)


def compare_rebatched_parent(batch, rows, out):
    parent = json.loads(Path(batch['parent']).read_text())
    old = {}
    for split in parent_flow.SPLITS:
        old.update({r['example_id']: r for r in read(Path(parent['runs']['base-'+split]['output'])/'planned_responses.jsonl')
                    if r['phase'] == 'primary'})
    current = {}
    for kind in KINDS:
        current.update({r['example_id']: r for r in read(Path(batch['runs'][kind]['base']['output'])/'planned_responses.jsonl')
                        if r['phase'] == 'primary' and r['example_id'] not in old})
    comparisons = []
    for r in rows:
        if r['diagnostic_arm'] == 'anchor' or not r['parent_identical']: continue
        before = old[r['parent_example_id']]; after = current[r['example_id']]
        comparisons.append(dict(**{k: r[k] for k in ('example_id', 'parent_example_id', 'kind', 'template', 'variant', 'query')},
            answer_changed=(before['completion_status'], before['response'].strip().casefold()) !=
                           (after['completion_status'], after['response'].strip().casefold()),
            tokens_changed=before['generated_token_ids'] != after['generated_token_ids']))
    assert len(comparisons) == 7128
    engine.jsonl(out/'rebatched_parent_comparisons.jsonl', comparisons)
    engine.write(out/'rebatched_parent_summary.json', engine.grouped(comparisons,
        ('kind', 'template', 'variant', 'query'), ('answer_changed', 'tokens_changed')))
    return dict(n=len(comparisons), answer_changed=sum(r['answer_changed'] for r in comparisons),
                tokens_changed=sum(r['tokens_changed'] for r in comparisons))


def stage(repo):
    soo = repo/'experiments/soo'; study = soo/'results/study2'
    parent_path = study/PARENT; parent = json.loads(parent_path.read_text()); engine.verify(parent)
    assert parent['status'] == 'baseline_competence_failed_stopped' and not any(parent['errors'].values())
    diagnosis = diagnose_parent(parent)
    anchor_rows, anchor_batches, anchor_raw, anchor_native = anchors(parent)
    old = json.loads(Path(parent['children']['base']).read_text()); old_root = Path(old['controller'])
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root = study/f'truth-revision-diagnostic1-controller-{stamp}'; root.mkdir()
    path = study/f'truth-revision-diagnostic1-batch-{stamp}.json'
    source = Path(parent['controller'])
    for f in source.glob('*.py'):
        if f.name.startswith('test_'): continue
        shutil.copy2(f, root/f.name); assert digest(root/f.name) == parent['frozen_sha256'][f.name]
    shutil.copy2(soo/'scripts/truth_revision_diagnostic.py', root/'truth_revision_diagnostic.py')
    shutil.copy2(soo/'scripts/truth_revision_diagnostic_task.py', root/'truth_revision_diagnostic_task.py')
    shutil.copy2(repo/'gpu_pick.py', root/'gpu_pick.py')
    shutil.copy2(soo/'TRUTH_REVISION_DIAGNOSTIC.md', root/'TRUTH_REVISION_DIAGNOSTIC.md')
    shutil.copy2(repo/'tests/test_soo_truth_revision_diagnostic.py', root/'test_soo_truth_revision_diagnostic.py')
    shutil.copytree(source/'src', root/'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    current = {str(p.relative_to(root/'src/selfconcept')): digest(p) for p in (root/'src/selfconcept').rglob('*.py')}
    assert current == old['inference_source_sha256']
    shutil.copytree(old_root/'historical_data', root/'historical_data')
    shutil.copytree(old_root/'history_reference', root/'history_reference')
    engine.write(root/'parent_diagnosis.json', diagnosis)
    engine.write(root/'anchor_reference.json', anchor_raw)
    engine.write(root/'provenance.json', dict(protocol='truth_revision_diagnostic1', parent=str(parent_path),
        diagnostic_only=True, all_worlds_previously_exposed=True))
    refs = {str(parent_path): digest(parent_path), str(Path(parent['children']['base'])): digest(parent['children']['base'])}
    for split in parent_flow.SPLITS:
        for name in ('planned_responses.jsonl', 'regressions.jsonl'):
            f = Path(parent['runs']['base-'+split]['output'])/name; refs[str(f)] = digest(f)
    (root/'data').mkdir(); (root/'plans').mkdir(); runs = {}; counts = {}
    parent_native = {}
    for split in parent_flow.SPLITS:
        parent_native.update(json.loads((source/'plans'/f'{split}.json').read_text())['expected_inputs'])
    for kind in KINDS:
        rows = build(kind) + deepcopy(anchor_rows)
        for e in rows: assert independent_key(e) == e['expected_answer']
        datafile = root/'data'/f'{kind}.jsonl'; engine.jsonl(datafile, rows)
        preflight = root/'preflight'/kind
        env = dict(os.environ, PYTHONPATH=str(root/'src'), OPENSSL_CONF='/dev/null', HF_HUB_OFFLINE='1', USE_TORCH='0')
        subprocess.run([str(repo/'.venv-313/bin/python'), str(root/'validate_goals_roles_tokens.py'),
            '--data', str(datafile), '--out', str(preflight)], env=env, check=True)
        native = {r['example_id']: r['input_ids'] for r in read(preflight/'token_spans.jsonl')}
        for e in rows:
            if e['diagnostic_arm'] == 'anchor': assert native[e['example_id']] == anchor_native[e['example_id']]
            if e['parent_identical']:
                assert native[e['example_id']] == parent_native[e['parent_example_id']]
        plan = make_plan(rows, native, anchor_batches); planfile = root/'plans'/f'{kind}.json'; engine.write(planfile, plan)
        output = study/f'truth-revision-diagnostic1-output-{stamp}'/kind
        script = parent_flow.shell(repo, root, path, 'td-'+kind,
            [str(repo/'.venv-313/bin/python'), '-u', str(root/'truth_revision_diagnostic.py'),
             '--batch', str(path), '--execute', kind], True)
        runs[kind] = {'base': dict(condition='base', alpha=1.0, data=str(datafile), plan=str(planfile),
            preflight=str(preflight), output=str(output), script=script, status='staged_not_submitted')}
        counts[kind] = dict(diagnostic=len(rows)-len(anchor_rows), anchors=len(anchor_rows),
            primary=len(rows), sentinels_each=sum(len(b) for b in plan['phases'][0]['batches']),
            generations_including_regressions=plan['generations']+32)
    parent_flow.shell(repo, root, path, 'td-analysis', ['/usr/bin/python3.11', '-u', str(root/'truth_revision_diagnostic.py'),
        '--batch', str(path), '--analyze'], False)
    receipts = study/f'truth-revision-diagnostic1-receipts-{stamp}'; receipts.mkdir()
    batch = dict(protocol='truth_revision_diagnostic1', authorization='User: let\'s do it, after the revised-batch postmortem and diagnostic proposal',
        status='staged_not_submitted', created_utc=datetime.now(timezone.utc).isoformat(), controller=str(root),
        parent=str(parent_path), revision=old['revision'], reference_versions=old['reference_versions'],
        inference_source_sha256=current, history_references={'base': str(root/'history_reference')},
        reference_sha256=refs, frozen_sha256=engine.hashes(root), runs=runs, counts=counts,
        receipts=str(receipts), analysis=str(study/f'truth-revision-diagnostic1-analysis-{stamp}'), errors={})
    engine.write(path, batch); engine.verify(batch)
    print('FROZEN', path, json.dumps(counts), flush=True)
    return path


def submit_gpu(batch, kind, entry):
    receipt_path = Path(batch['receipts'])/(kind+'.json')
    script = Path(batch['controller'])/entry['script']
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        assert receipt['script_sha256'] == digest(script)
        return receipt
    selected = parent_flow.placement(batch)
    command = ['/apps/slurm/latest/bin/sbatch', '--parsable', '--partition='+selected['partition'],
               '--qos='+selected['qos'], '--no-requeue', str(script)]
    job = subprocess.check_output(command, text=True).strip().split(';')[0]
    receipt = dict(job_id=job, script_sha256=digest(script), dependencies=[], placement=selected,
                   command=command, submitted_utc=datetime.now(timezone.utc).isoformat())
    engine.write(receipt_path, receipt)
    return receipt


def submit(path):
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        batch = json.loads(path.read_text()); engine.verify(batch)
        assert batch['status'] in ('staged_not_submitted', 'submitting')
        batch['status'] = 'submitting'; engine.write(path, batch); jobs = []
        for kind in KINDS:
            entry = batch['runs'][kind]['base']
            receipt = submit_gpu(batch, kind, entry)
            entry.update(receipt, status='submitted'); engine.write(path, batch); jobs.append(receipt['job_id'])
        batch['audit_receipt'] = engine.submit(Path(batch['controller'])/'td-analysis.sh',
            Path(batch['receipts'])/'analysis.json', jobs)
        batch.update(status='diagnostic_submitted', updated_utc=datetime.now(timezone.utc).isoformat())
        engine.write(path, batch); print('SUBMITTED', jobs, batch['audit_receipt']['job_id'], flush=True)


def finish(path):
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        batch = json.loads(path.read_text())
        if batch['status'] == 'completed_and_analyzed': return
        configure(); out = Path(batch['analysis']); out.mkdir(exist_ok=True)
        try:
            engine.verify(batch); rows = []; anchor_ref = json.loads((Path(batch['controller'])/'anchor_reference.json').read_text())
            for kind in KINDS:
                try:
                    result = engine.audit_run(batch, kind, 'base'); assert result['manifest']['adapter'] is None
                    raw = {r['example_id']: r for r in read(Path(batch['runs'][kind]['base']['output'])/'planned_responses.jsonl')
                           if r['phase'] == 'primary'}
                    assert all(engine.raw_only(raw[eid]) == ref for eid, ref in anchor_ref.items()), 'parent anchor changed'
                    engine.write(out/(kind+'-audit.json'), result); rows.extend(result['scores'])
                    batch['runs'][kind]['base']['status'] = 'completed_and_audited'
                except Exception as exc:
                    batch['errors'][kind] = repr(exc); batch['runs'][kind]['base']['status'] = 'inspection_needed'
            if batch['errors']:
                batch['status'] = 'execution_discrepancy_stopped'
            else:
                batch['results'] = analyze(rows, out)
                batch['results']['rebatched_parent'] = compare_rebatched_parent(batch, rows, out)
                batch['results']['exact_batch_parent_anchors'] = dict(n=128*len(KINDS), answer_changed=0, tokens_changed=0)
                batch['status'] = 'completed_and_analyzed'
        except Exception as exc:
            batch.update(status='controller_error_requires_inspection', error=repr(exc)); raise
        finally:
            batch['updated_utc'] = datetime.now(timezone.utc).isoformat(); engine.write(path, batch)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', type=Path); p.add_argument('--batch', type=Path)
    p.add_argument('--submit', action='store_true'); p.add_argument('--execute', choices=KINDS)
    p.add_argument('--analyze', action='store_true'); args = p.parse_args()
    if args.stage:
        path = stage(args.stage.resolve())
        if args.submit:
            frozen = Path(json.loads(path.read_text())['controller'])/'truth_revision_diagnostic.py'
            subprocess.run(['/usr/bin/python3.11', str(frozen), '--batch', str(path), '--submit'], check=True)
    elif args.batch and args.submit: submit(args.batch)
    elif args.batch and args.execute:
        configure(); engine.execute(args.batch, args.execute, 'base')
    elif args.batch and args.analyze: finish(args.batch)
    else: p.error('stage, or batch with submit/execute/analyze required')
