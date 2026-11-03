"""Freeze and execute the bounded goal-identity/placement follow-up."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

import extension_repeat_bridge as engine
import reminder_wording as flow
import report_goal_seeds as seeds
import goal_identity_task as task
from extension_repeat_selection import digest, read

LABELS = flow.LABELS
SPLITS = flow.SPLITS
PARENT = 'reminder-wording1-batch-20261028T005925736592Z.json'
parent_audit = flow.audit
write = engine.write


def configure():
    engine.TAGS = ('base', 'adapter'); engine.score = task.score; engine.gate = task.gate
    flow.audit = audit; flow.analyze = analyze


def audit(batch, label, split):
    child = json.loads(Path(batch['children'][label]).read_text()); engine.verify(child)
    tag = 'base' if label == 'base' else 'adapter'
    result = engine.audit_run(child, split, tag)
    if tag == 'adapter': seeds.adapter_binding(child, result['manifest'])
    else: assert result['manifest']['adapter'] is None
    entry = child['runs'][split][tag]
    records = {r['example_id']: r for r in read(Path(entry['output']) / 'planned_responses.jsonl') if r['phase'] == 'primary'}
    old = {r['example_id']: r for r in read(Path(batch['controller']) / 'anchors' / label / split / 'responses.jsonl')}
    comparisons = []
    for e in read(entry['data']):
        if e['cohort'] != 'anchor': continue
        r = records[e['example_id']]; ref = old[e['parent_example_id']]
        comparisons.append(dict(**{k: e[k] for k in ('example_id', 'variant', 'query', 'role', 'alignment', 'instance_id')},
            answer_changed=(r['completion_status'], r['response'].strip().casefold()) !=
                           (ref['completion_status'], ref['response'].strip().casefold()),
            tokens_changed=r['generated_token_ids'] != ref['generated_token_ids']))
    result['anchor_comparisons'] = comparisons
    return result


def analyze(results, out):
    assert set(results) == set(LABELS)
    rows = [dict(r, condition=label) for label, result in results.items() for r in result['scores']]
    metrics = ('valid', 'correct', 'assigned_goal', 'original_goal', 'other_original_goal', 'recipient_goal', 'dominated')
    for name, fields in (
        ('summary', ('condition', 'cohort', 'variant', 'query', 'alignment', 'effective_alignment')),
        ('templates', ('condition', 'template', 'variant', 'query', 'alignment')),
        ('roles', ('condition', 'cohort', 'variant', 'query', 'alignment', 'role')),
        ('instances', ('condition', 'instance_id', 'variant', 'query', 'alignment')),
        ('presentation', ('condition', 'cohort', 'variant', 'query', 'alignment', 'table_order', 'answer_order'))):
        write(out / (name+'.json'), engine.grouped(rows, fields, metrics))
    engine.jsonl(out / 'all_scores.jsonl', rows)
    indexed = {label: {r['example_id']: r for r in result['scores']} for label, result in results.items()}
    paired = []; joint = []; tracking = []
    for label in LABELS:
        assert set(indexed[label]) == set(indexed['base'])
        contexts = defaultdict(dict)
        for eid, r in indexed[label].items():
            base = indexed['base'][eid]
            original_variant = 'original' if r['cohort'] == 'fresh' else 'anchor_original'
            original = indexed[label]['goalidentity1-'+original_variant+'::'+r['source_example_id']]
            paired.append(dict(r, condition=label, answer_changed_vs_base=r['answer'] != base['answer'],
                answer_changed_vs_original=r['answer'] != original['answer'],
                recipient_delta_vs_base=int(r['recipient_goal'])-int(base['recipient_goal']),
                original_goal_delta_vs_base=int(r['original_goal'])-int(base['original_goal'])))
            contexts[r['context_id']][r['query']] = r
        for cid, qs in contexts.items():
            assert set(qs) == set(task.QUERIES)
            m = qs['message']; controls = all(qs[q]['correct'] for q in ('private', 'fact0', 'fact1'))
            joint.append(dict(condition=label, context_id=cid,
                **{k: m[k] for k in ('cohort', 'template', 'variant', 'role', 'alignment', 'instance_id')},
                recipient_message_with_controls=m['recipient_goal'] and controls,
                assigned_message_with_controls=m['assigned_goal'] and controls))
            if m['cohort'] != 'fresh' or m['wording'] != 'assign_same': continue
            other_cid = cid.replace('-assign_same_', '-assign_other_', 1)
            other = contexts[other_cid]
            for q in ('private', 'message'):
                tracking.append(dict(condition=label, context_id=cid, query=q,
                    **{k: m[k] for k in ('template', 'position', 'role', 'alignment', 'instance_id')},
                    both_assignments_correct=qs[q]['correct'] and other[q]['correct'],
                    both_assignments_correct_with_controls=qs[q]['correct'] and other[q]['correct'] and controls
                        and all(other[c]['correct'] for c in ('private', 'fact0', 'fact1')),
                    answer_changes=qs[q]['answer'] != other[q]['answer']))
    # Pool presentation rotations only descriptively. Also save every world's
    # contrast separately; worlds/templates, not rotations, are inferential units.
    fresh = [r for r in rows if r['cohort'] == 'fresh']
    cells = defaultdict(list)
    for r in fresh:
        for unit, value in [('all', 'all'), ('template', r['template']), ('instance', r['instance_id'])]:
            for role in ('all', r['role']):
                cells[unit, value, r['condition'], r['alignment'], role, r['query'], r['variant']].append(r)
    contrasts = []
    units = [('all', 'all')] + [('template', t) for t in sorted({r['template'] for r in fresh})] + [
        ('instance', i) for i in sorted({r['instance_id'] for r in fresh})]
    for unit, value in units:
        for alignment in ('aligned', 'opposed'):
            for role in ('all', 0, 1):
                for query in ('private', 'message'):
                    for metric in ('original_goal', 'recipient_goal', 'correct'):
                        rates = {}
                        for label in LABELS:
                            rates[label] = {}
                            for variant in task.VARIANTS:
                                selected = cells[unit, value, label, alignment, role, query, variant]
                                assert selected
                                rates[label][variant] = sum(r[metric] for r in selected) / len(selected)
                        for label in LABELS:
                            for left, right in task.PAIRS:
                                change = rates[label][left] - rates[label][right]
                                base_change = rates['base'][left] - rates['base'][right]
                                contrasts.append(dict(unit=unit, unit_id=value, condition=label,
                                    alignment=alignment, role=role, query=query, metric=metric,
                                    contrast=left+'_minus_'+right, left=left, right=right,
                                    change=change, base_change=base_change, baseline_adjusted_change=change-base_change))
    anchors = [dict(r, condition=label) for label, result in results.items() for r in result['anchor_comparisons']]
    for name, data in [('paired', paired), ('joint', joint), ('assignment_tracking', tracking), ('anchor_comparisons', anchors)]:
        engine.jsonl(out / (name+'.jsonl'), data)
    write(out / 'anchor_summary.json', engine.grouped(anchors,
        ('condition', 'variant', 'query', 'alignment'), ('answer_changed', 'tokens_changed')))
    tracking_summary = engine.grouped(tracking, ('condition', 'template', 'position', 'query', 'alignment'),
        ('both_assignments_correct', 'both_assignments_correct_with_controls', 'answer_changes'))
    write(out / 'assignment_summary.json', tracking_summary)
    write(out / 'contrasts.json', contrasts)
    return dict(summary=engine.grouped(rows,
        ('condition', 'cohort', 'variant', 'query', 'alignment'), metrics),
        assignment_tracking=tracking_summary,
        contrasts=[c for c in contrasts if c['unit'] == 'all'],
        interpretation='Fresh frozen allocation templates; anchors separate; reassignment changes keys; no independent-prompt inference or mechanism claim.')


def build_plan(rows, native):
    ids = [e['example_id'] for e in rows]
    batches = [ids[i:i+8] for i in range(0, len(ids), 8)]
    assert all(len(b) == 8 for b in batches)
    selected = set()
    seen = set()
    for i, e in enumerate(rows):
        key = tuple(e[k] for k in ('template', 'variant', 'role', 'alignment', 'query'))
        if key not in seen: selected.add(i // 8); seen.add(key)
    sentinel = [batches[i] for i in sorted(selected)]
    return dict(expected_inputs=native, phases=[dict(name='sentinel_before', batches=sentinel),
        dict(name='primary', batches=batches), dict(name='sentinel_after', batches=sentinel)],
        generations=len(ids)+2*sum(map(len, sentinel)))


def matching_report(rows, native):
    by_id = {e['example_id']: e for e in rows}
    comparisons = []
    for e in rows:
        if e['cohort'] != 'fresh' or e['wording'] not in ('goal', 'assign_same'): continue
        other_variant = e['variant'].replace('goal_', 'name_') if e['wording'] == 'goal' else e['variant'].replace('assign_same_', 'assign_other_')
        other = by_id['goalidentity1-'+other_variant+'::'+e['source_example_id']]
        comparisons.append(dict(example_id=e['example_id'], other_example_id=other['example_id'],
            comparison='goal_name' if e['wording'] == 'goal' else 'assignment_identity',
            template=e['template'], role=e['role'], position=e['position'],
            token_difference=len(native[e['example_id']])-len(native[other['example_id']]),
            first_tokens=len(native[e['example_id']]), second_tokens=len(native[other['example_id']])) )
    return comparisons


def stage(repo):
    soo = repo / 'experiments/soo'; study = soo / 'results/study2'
    parent_path = study / PARENT; parent = json.loads(parent_path.read_text())
    engine.verify(parent)
    assert parent['status'] == 'completed_and_analyzed' and not any(parent['errors'].values())
    flow.configure()
    for label in LABELS:
        for split in SPLITS:
            result = parent_audit(parent, label, split)
            assert result['checks_passed']
            if label == 'base': assert result['gate']['passed']
    configure()
    source = Path(parent['controller'])
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root = study / f'goal-identity1-controller-{stamp}'; root.mkdir()
    path = study / f'goal-identity1-batch-{stamp}.json'
    for name in ('goal_identity.py', 'goal_identity_task.py'):
        shutil.copy2(soo / 'scripts' / name, root / name)
    # All inherited inference, audit and submission code stays frozen.
    for name in ('reminder_wording.py', 'reminder_wording_task.py', 'report_goal_scope.py',
                 'report_goal_task.py', 'report_components_task.py', 'payoff_context_task.py',
                 'report_goal_seeds.py', 'extension_repeat_bridge.py', 'extension_repeat_selection.py',
                 'extension_bridge_task.py', 'validate_goals_roles_tokens.py'):
        shutil.copy2(source / name, root / name)
        assert digest(root / name) == parent['frozen_sha256'][name]
    shutil.copy2(soo / 'GOAL_IDENTITY_PLACEMENT.md', root / 'GOAL_IDENTITY_PLACEMENT.md')
    shutil.copy2(repo / 'tests/test_soo_goal_identity.py', root / 'test_soo_goal_identity.py')
    shutil.copytree(source / 'src', root / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    refs = {str(parent_path): digest(parent_path)}
    (root / 'data').mkdir(); (root / 'plans').mkdir()
    common_refs = {}; matching = []
    for split in SPLITS:
        original_path = source / 'data' / (split+'.jsonl')
        rows = task.build(split, read(original_path)); refs[str(original_path)] = digest(original_path)
        datafile = root / 'data' / (split+'.jsonl'); engine.jsonl(datafile, rows)
        preflight = root / 'preflight' / split
        env = dict(os.environ, PYTHONPATH=str(root / 'src'), OPENSSL_CONF='/dev/null', HF_HUB_OFFLINE='1', USE_TORCH='0')
        subprocess.run([str(repo / '.venv-313/bin/python'), str(root / 'validate_goals_roles_tokens.py'),
            '--data', str(datafile), '--out', str(preflight)], env=env, check=True)
        native = {r['example_id']: r['input_ids'] for r in read(preflight / 'token_spans.jsonl')}
        oldplan_path = source / 'plans' / (split+'.json'); oldplan = json.loads(oldplan_path.read_text())
        refs[str(oldplan_path)] = digest(oldplan_path)
        for e in rows:
            if e['cohort'] == 'anchor': assert native[e['example_id']] == oldplan['expected_inputs'][e['parent_example_id']]
        plan = build_plan(rows, native)
        for variant in task.ANCHORS:
            selected = [e for e in rows if e['variant'] == variant]; wanted = {e['parent_example_id'] for e in selected}
            old_batches = next(p['batches'] for p in oldplan['phases'] if p['name'] == 'primary')
            chosen = [b for b in old_batches if wanted.intersection(b)]
            assert all(len(b) == 8 and set(b) <= wanted for b in chosen)
            assert [eid for b in chosen for eid in b] == [e['parent_example_id'] for e in selected]
        planfile = root / 'plans' / (split+'.json'); write(planfile, plan)
        matching += [dict(r, split=split) for r in matching_report(rows, native)]
        for f in (datafile, planfile, preflight / 'token_spans.jsonl'): common_refs[str(f)] = digest(f)
    engine.jsonl(root / 'length_matching.jsonl', matching)
    write(root / 'length_matching_summary.json', engine.grouped(matching,
        ('split', 'comparison', 'template', 'token_difference'), ()))
    assert all(r['token_difference'] == 0 for r in matching), 'matched contrasts must have equal full input lengths'
    runs = {}; children = {}; weights = []
    for label in LABELS:
        tag = 'base' if label == 'base' else 'adapter'
        old_child_path = Path(parent['children'][label]); old_child = json.loads(old_child_path.read_text())
        old_root = Path(old_child['controller']); refs[str(old_child_path)] = digest(old_child_path)
        child_root = root / label; child_root.mkdir()
        seed = None if label == 'base' else int(label[-1])
        if seed is not None:
            shutil.copytree(old_root / 'adapter', child_root / 'adapter')
            for name, sha in old_child['adapter_sha256'].items(): assert digest(child_root / 'adapter' / name) == sha
            weights.append(old_child['adapter_sha256']['adapter_model.safetensors'])
        shutil.copytree(old_root / 'historical_data', child_root / 'historical_data')
        current = {str(p.relative_to(root / 'src/selfconcept')): digest(p) for p in (root / 'src/selfconcept').rglob('*.py')}
        assert current == old_child['inference_source_sha256']
        write(child_root / 'provenance.json', dict(protocol='goal_identity1', condition=label,
            parent=str(parent_path), new_worlds_and_templates=True, historical_anchors_separate=True))
        history_root = child_root / 'history_reference'; (history_root / 'output').mkdir(parents=True)
        history = None; child_runs = {}
        for split in SPLITS:
            old_out = Path(parent['runs'][label+'-'+split]['output'])
            history_file = old_out / 'regressions.jsonl'; old_history = read(history_file)
            refs[str(history_file)] = digest(history_file)
            if history is None: history = old_history
            else: assert history == old_history
            records_file = old_out / 'planned_responses.jsonl'; refs[str(records_file)] = digest(records_file)
            prefixes = tuple('reminderword1-'+v+'::' for v in task.ANCHORS.values())
            records = [r for r in read(records_file) if r['phase'] == 'primary' and r['example_id'].startswith(prefixes)]
            assert len(records) == 864
            anchor_dir = root / 'anchors' / label / split; anchor_dir.mkdir(parents=True)
            engine.jsonl(anchor_dir / 'responses.jsonl', records)
            entry = dict(condition=tag, alpha=1.0, data=str(root / 'data' / (split+'.jsonl')),
                plan=str(root / 'plans' / (split+'.json')), preflight=str(root / 'preflight' / split),
                output=str(study / f'goal-identity1-output-{stamp}' / label / split), status='staged_not_submitted')
            child_runs[split] = {tag: entry}
            key = label+'-'+split
            script = engine.shell(repo, root, path, key, [str(repo / '.venv-313/bin/python'), '-u', str(root / 'goal_identity.py'),
                '--batch', str(path), '--execute', split, '--condition', label], True)
            runs[key] = dict(condition=label, split=split, script=script, output=entry['output'], status='staged_not_submitted')
        engine.jsonl(history_root / 'output/regressions.jsonl', history)
        child = dict(controller=str(child_root), seed=seed, revision=old_child['revision'],
            reference_versions=old_child['reference_versions'], inference_source_sha256=current,
            adapter_sha256=old_child['adapter_sha256'], history_references={tag: str(history_root)},
            reference_sha256=common_refs, frozen_sha256=engine.hashes(child_root), runs=child_runs)
        child_path = root / (label+'-batch.json'); write(child_path, child); children[label] = str(child_path)
    assert len(set(weights)) == 3
    picker = subprocess.check_output(['/usr/bin/python3.11', str(repo / 'gpu_pick.py'), '--hours', '1', '--vram-gb', '75',
        '--cpus', '4', '--mem-gb', '96', '--no-requeue-ok', '--fast', '--cuda', '13.0', '--top', '10'], text=True)
    (root / 'gpu_pick.txt').write_text(picker)
    for phase in ('baseline', 'final'):
        engine.shell(repo, root, path, phase, ['/usr/bin/python3.11', '-u', str(root / 'goal_identity.py'), '--batch', str(path), '--advance', phase], False)
    receipts = study / f'goal-identity1-receipts-{stamp}'; receipts.mkdir()
    batch = dict(protocol='goal_identity1', status='staged_not_submitted',
        authorization='User requested recording the four routes and proceeding with route 1: goal identity and placement with fresh-template confirmation.',
        created_utc=datetime.now(timezone.utc).isoformat(), controller=str(root), parent=str(parent_path),
        analysis=str(study / f'goal-identity1-analysis-{stamp}'), receipts=str(receipts), children=children, runs=runs,
        maximum_primary_responses=69120, reference_sha256=refs, frozen_sha256=engine.hashes(root),
        errors={}, gates={}, results={}, audit_receipts={})
    write(path, batch); engine.verify(batch); print('FROZEN', path, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', type=Path); p.add_argument('--batch', type=Path); p.add_argument('--submit', action='store_true')
    p.add_argument('--advance', choices=('baseline', 'final')); p.add_argument('--execute', choices=SPLITS)
    p.add_argument('--condition', choices=LABELS)
    args = p.parse_args()
    if args.stage: stage(args.stage.resolve())
    else:
        configure()
        if args.batch and args.submit: flow.submit(args.batch)
        elif args.batch and args.advance: flow.advance(args.batch, args.advance)
        elif args.batch and args.execute and args.condition:
            batch = json.loads(args.batch.read_text()); engine.verify(batch)
            engine.execute(Path(batch['children'][args.condition]), args.execute, 'base' if args.condition == 'base' else 'adapter')
        else: p.error('stage, submit, advance or execute/condition required')
