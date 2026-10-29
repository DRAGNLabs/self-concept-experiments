"""Freeze and execute route 2: truth versus beneficiary, with all three SOO seeds."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import shlex

import extension_repeat_bridge as engine
import reminder_wording as flow
import goal_identity as prior
import report_goal_seeds as seeds
import truth_beneficiary_revision_task as task
from extension_repeat_selection import digest, read

LABELS = flow.LABELS
SPLITS = flow.SPLITS
PARENT = 'goal-identity1-batch-20261028T172154803111Z.json'
write = engine.write
METRICS = ('valid', 'correct', 'truthful', 'assigned_goal', 'other_client',
           'recipient_goal', 'compromise', 'total_optimal', 'maximin', 'false_report')


def configure():
    engine.TAGS = ('base', 'adapter'); engine.score = task.score; engine.gate = task.gate
    flow.audit = audit; flow.analyze = analyze; flow.launch = launch


def audit(batch, label, split):
    child = json.loads(Path(batch['children'][label]).read_text()); engine.verify(child)
    tag = 'base' if label == 'base' else 'adapter'
    result = engine.audit_run(child, split, tag)
    if tag == 'adapter': seeds.adapter_binding(child, result['manifest'])
    else: assert result['manifest']['adapter'] is None
    return result


def analyze(results, out):
    assert set(results) == set(LABELS)
    rows = [dict(r, condition=label) for label, result in results.items() for r in result['scores']]
    fields = ('condition', 'variant', 'query', 'alignment', 'truth_relation')
    summary = engine.grouped(rows, fields, METRICS)
    write(out / 'summary.json', summary)
    for name, extra in [('templates', ('template',)), ('instances', ('instance_id',)),
                        ('roles', ('role',)), ('presentation', ('table_order', 'answer_order'))]:
        write(out / (name+'.json'), engine.grouped(rows, fields+extra, METRICS))
    engine.jsonl(out / 'all_scores.jsonl', rows)
    indexed = {label: {r['example_id']: r for r in result['scores']} for label, result in results.items()}
    paired = []; joint = []; tracking = []
    for label in LABELS:
        assert set(indexed[label]) == set(indexed['base'])
        contexts = defaultdict(dict)
        for eid, r in indexed[label].items():
            base = indexed['base'][eid]
            original = indexed[label]['truthbenefit2-original::'+r['source_example_id']]
            paired.append(dict(r, condition=label, answer_changed_vs_base=r['answer'] != base['answer'],
                answer_changed_vs_original=r['answer'] != original['answer'],
                truth_delta_vs_base=int(r['truthful'])-int(base['truthful']),
                assigned_delta_vs_base=int(r['assigned_goal'])-int(base['assigned_goal'])))
            contexts[r['context_id']][r['query']] = r
        state_groups = defaultdict(list)
        for cid, qs in contexts.items():
            assert set(qs) == set(task.QUERIES)
            r = qs['report']; controls = all(qs[q]['correct'] for q in task.CONTROLS)
            record = dict(condition=label, context_id=cid,
                **{k: r[k] for k in ('template', 'variant', 'role', 'alignment', 'truth_relation',
                                    'instance_id', 'table_order', 'answer_order', 'actual')},
                controls_correct=controls, truthful_with_controls=r['truthful'] and controls,
                assigned_with_controls=r['assigned_goal'] and controls,
                recipient_with_controls=r['recipient_goal'] and controls,
                compromise_with_controls=r['compromise'] and controls)
            joint.append(record)
            key = tuple(r[k] for k in ('template', 'instance_id', 'variant', 'role', 'alignment', 'table_order', 'answer_order'))
            state_groups[key].append((qs, record))
        for group, members in state_groups.items():
            assert len(members) == 3 and {qs['report']['truth_relation'] for qs, _ in members} == set(task.RELATIONS)
            r = members[0][0]['report']
            tracking.append(dict(condition=label,
                **{k: r[k] for k in ('template', 'instance_id', 'variant', 'role', 'alignment', 'table_order', 'answer_order')},
                all_three_truthful=all(qs['report']['truthful'] for qs, _ in members),
                all_three_assigned=all(qs['report']['assigned_goal'] for qs, _ in members),
                all_three_recipient=all(qs['report']['recipient_goal'] for qs, _ in members),
                all_three_compromise=all(qs['report']['compromise'] for qs, _ in members),
                all_three_truthful_with_controls=all(j['truthful_with_controls'] for _, j in members)))
    joint_metrics = ('controls_correct', 'truthful_with_controls', 'assigned_with_controls',
                     'recipient_with_controls', 'compromise_with_controls')
    for name, data in [('paired', paired), ('joint', joint), ('truth_tracking', tracking)]:
        engine.jsonl(out / (name+'.jsonl'), data)
    joint_summary = engine.grouped(joint, ('condition', 'variant', 'alignment', 'truth_relation'), joint_metrics)
    write(out / 'joint_summary.json', joint_summary)
    write(out / 'joint_templates.json', engine.grouped(joint,
        ('condition', 'template', 'variant', 'alignment', 'truth_relation'), joint_metrics))
    tracking_summary = engine.grouped(tracking, ('condition', 'template', 'variant', 'alignment'),
        ('all_three_truthful', 'all_three_assigned', 'all_three_recipient',
         'all_three_compromise', 'all_three_truthful_with_controls'))
    write(out / 'truth_tracking_summary.json', tracking_summary)
    # Contrast raw outcomes and unconditioned joint outcomes; do not drop errors.
    contrast_rows = [dict(r, readout=r['query']) for r in rows]
    contrast_rows += [dict(r, readout='joint') for r in joint]
    cells = defaultdict(list)
    for r in contrast_rows:
        for unit, value in [('all', 'all'), ('template', r['template']), ('instance', r['instance_id'])]:
            for role in ('all', r['role']):
                cells[unit, value, r['condition'], r['alignment'], r['truth_relation'], role, r['readout'], r['variant']].append(r)
    units = [('all', 'all')] + [('template', x) for x in sorted({r['template'] for r in rows})] + [
        ('instance', x) for x in sorted({r['instance_id'] for r in rows})]
    contrasts = []
    for unit, value in units:
        for alignment in ('aligned', 'opposed'):
            for relation in task.RELATIONS:
                for role in ('all', 0, 1):
                    for readout in (*task.QUERIES, 'joint'):
                        metrics = joint_metrics if readout == 'joint' else ('truthful', 'assigned_goal', 'recipient_goal', 'compromise', 'correct')
                        for metric in metrics:
                            rates = {}
                            for label in LABELS:
                                rates[label] = {}
                                for variant in task.VARIANTS:
                                    selected = cells[unit, value, label, alignment, relation, role, readout, variant]
                                    assert selected
                                    rates[label][variant] = sum(r[metric] for r in selected) / len(selected)
                            for label in LABELS:
                                info = dict(unit=unit, unit_id=value, condition=label, alignment=alignment,
                                    truth_relation=relation, role=role, readout=readout, metric=metric)
                                for variant in task.VARIANTS:
                                    contrasts.append(dict(info, contrast='condition_minus_base', variant=variant,
                                        rate=rates[label][variant], base_rate=rates['base'][variant],
                                        value=rates[label][variant]-rates['base'][variant]))
                                change = rates[label]['goal_late']-rates[label]['original']
                                base_change = rates['base']['goal_late']-rates['base']['original']
                                contrasts.append(dict(info, contrast='goal_late_minus_original',
                                    change=change, base_change=base_change, baseline_adjusted_change=change-base_change))
    write(out / 'contrasts.json', contrasts)
    return dict(summary=summary, joint_summary=joint_summary, truth_tracking=tracking_summary,
        contrasts=[c for c in contrasts if c['unit'] == 'all' and c['role'] == 'all' and c['readout'] in ('report', 'joint')],
        interpretation='Truth varies independently of fixed payoffs; all seeds/templates retained; raw and unconditioned joint counts; no prompt-level inference or general honesty claim.')


def build_plan(rows, native):
    ids = [e['example_id'] for e in rows]
    batches = [ids[i:i+8] for i in range(0, len(ids), 8)]
    assert all(len(b) == 8 for b in batches)
    selected = set(); seen = set()
    for i, e in enumerate(rows):
        key = tuple(e[k] for k in ('template', 'variant', 'role', 'alignment', 'truth_relation', 'query'))
        if key not in seen: selected.add(i // 8); seen.add(key)
    sentinel = [batches[i] for i in sorted(selected)]
    return dict(expected_inputs=native, phases=[dict(name='sentinel_before', batches=sentinel),
        dict(name='primary', batches=batches), dict(name='sentinel_after', batches=sentinel)],
        generations=len(ids)+2*sum(map(len, sentinel)))


_placement = None


def placement(batch):
    global _placement
    if _placement is not None: return _placement
    root = Path(batch['controller'])
    command = ['/usr/bin/python3.11', str(root/'gpu_pick.py'), '--hours', '1', '--vram-gb', '75',
        '--cpus', '4', '--mem-gb', '96', '--no-requeue-ok', '--fast', '--arch', 'x86_64', '--cuda', '13.0', '--top', '10']
    for gpu in ('p100', 'l40s', 'h100', 'gh200', 'h200', 'b200'): command += ['--exclude-gpu', gpu]
    output = subprocess.check_output(command, text=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    evidence = Path(batch['receipts']) / ('gpu-pick-'+stamp+'.json')
    selected = None
    for line in output.splitlines():
        columns = line.split()
        if columns and columns[0] == '1' and len(columns) > 4:
            assert columns[2] == 'a100' and columns[3] == '1'
            selected = dict(partition=columns[1], qos=columns[4]); break
    write(evidence, dict(command=command, output=output, selected=selected,
        rationale='Best compatible A100 option for frozen historical replay; no preemption. No GPU is requested for CPU audits.'))
    if selected is None: raise RuntimeError('No compatible A100 placement; submission stopped without fallback or duplicate jobs.')
    _placement = dict(selected, evidence=str(evidence))
    return _placement


def launch(batch, path, label, split):
    key = label+'-'+split; entry = batch['runs'][key]
    receipt_path = Path(batch['receipts']) / (key+'.json')
    script = Path(batch['controller']) / entry['script']
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        assert receipt['script_sha256'] == digest(script)
    else:
        selected = placement(batch)
        command = ['/apps/slurm/latest/bin/sbatch', '--parsable', '--partition='+selected['partition'],
                   '--qos='+selected['qos'], '--no-requeue', str(script)]
        job = subprocess.check_output(command, text=True).strip().split(';')[0]
        receipt = dict(job_id=job, script_sha256=digest(script), dependencies=[],
            submitted_utc=datetime.now(timezone.utc).isoformat(), placement=selected, command=command)
        write(receipt_path, receipt)
    entry.update(receipt, status='submitted'); write(path, batch)
    return receipt['job_id']


def shell(repo, root, path, name, command, gpu):
    name = engine.shell(repo, root, path, name, command, gpu)
    script = root/name
    text = script.read_text().replace('#SBATCH --partition=dw\n', '').replace('#SBATCH --qos=dw87\n', '')
    text = text.replace('#SBATCH --nodes=1', '#SBATCH --no-requeue\n'+('' if gpu else '#SBATCH --qos=normal\n')+'#SBATCH --nodes=1')
    text = text.replace('#SBATCH --job-name=soo-rb-', '#SBATCH --job-name=soo-tb2-')
    if not gpu: text = text.replace('export OMP_NUM_THREADS=4', 'export OMP_NUM_THREADS=1')
    script.write_text(text)
    subprocess.run(['bash', '-n', str(script)], check=True)
    return name


def stage(repo):
    soo = repo / 'experiments/soo'; study = soo / 'results/study2'
    parent_path = study / PARENT; parent = json.loads(parent_path.read_text()); engine.verify(parent)
    assert parent['status'] == 'completed_and_analyzed' and not any(parent['errors'].values())
    prior.configure()
    for label in LABELS:
        for split in SPLITS:
            result = prior.audit(parent, label, split)
            assert result['checks_passed']
            if label == 'base': assert result['gate']['passed']
    configure()
    diagnostic_path = study/'truth-wording-diagnostic1-batch-20261028T204429164410Z.json'
    diagnostic = json.loads(diagnostic_path.read_text()); engine.verify(diagnostic)
    assert diagnostic['status'] == 'completed_and_analyzed' and not diagnostic['errors']
    assert diagnostic['results']['diagnostic_responses'] == 18720
    source = Path(parent['controller'])
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root = study / f'truth-beneficiary2-controller-{stamp}'; root.mkdir()
    path = study / f'truth-beneficiary2-batch-{stamp}.json'
    for name in ('truth_beneficiary_revision.py', 'truth_beneficiary_revision_task.py', 'truth_beneficiary_task.py'):
        shutil.copy2(soo / 'scripts' / name, root / name)
    for f in source.glob('*.py'):
        if f.name.startswith('test_'): continue
        shutil.copy2(f, root / f.name)
        assert digest(root / f.name) == parent['frozen_sha256'][f.name]
    shutil.copy2(soo / 'TRUTH_BENEFICIARY_REVISION.md', root / 'TRUTH_BENEFICIARY_REVISION.md')
    shutil.copy2(repo / 'tests/test_soo_truth_beneficiary_revision.py', root / 'test_soo_truth_beneficiary_revision.py')
    shutil.copytree(source / 'src', root / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(repo/'gpu_pick.py', root/'gpu_pick.py')
    refs = {str(parent_path): digest(parent_path), str(diagnostic_path): digest(diagnostic_path)}
    (root / 'data').mkdir(); (root / 'plans').mkdir()
    common_refs = {}
    for split in SPLITS:
        rows = task.build(split); datafile = root / 'data' / (split+'.jsonl'); engine.jsonl(datafile, rows)
        preflight = root / 'preflight' / split
        env = dict(os.environ, PYTHONPATH=str(root / 'src'), OPENSSL_CONF='/dev/null', HF_HUB_OFFLINE='1', USE_TORCH='0')
        subprocess.run([str(repo / '.venv-313/bin/python'), str(root / 'validate_goals_roles_tokens.py'),
            '--data', str(datafile), '--out', str(preflight)], env=env, check=True)
        native = {r['example_id']: r['input_ids'] for r in read(preflight / 'token_spans.jsonl')}
        planfile = root / 'plans' / (split+'.json'); write(planfile, build_plan(rows, native))
        for f in (datafile, planfile, preflight / 'token_spans.jsonl'): common_refs[str(f)] = digest(f)
    runs = {}; children = {}; weights = []
    for label in LABELS:
        tag = 'base' if label == 'base' else 'adapter'
        old_path = Path(parent['children'][label]); old = json.loads(old_path.read_text())
        old_root = Path(old['controller']); refs[str(old_path)] = digest(old_path)
        child_root = root / label; child_root.mkdir()
        seed = None if label == 'base' else int(label[-1])
        if seed is not None:
            shutil.copytree(old_root / 'adapter', child_root / 'adapter')
            for name, sha in old['adapter_sha256'].items(): assert digest(child_root / 'adapter' / name) == sha
            assert json.loads((child_root / 'adapter/train_log.json').read_text())['seed'] == seed
            weights.append(old['adapter_sha256']['adapter_model.safetensors'])
        shutil.copytree(old_root / 'historical_data', child_root / 'historical_data')
        current = {str(p.relative_to(root / 'src/selfconcept')): digest(p) for p in (root / 'src/selfconcept').rglob('*.py')}
        assert current == old['inference_source_sha256']
        write(child_root / 'provenance.json', dict(protocol='truth_beneficiary2', condition=label,
            parent=str(parent_path), new_worlds_and_templates=True, independently_crossed_truth=True))
        history_root = child_root / 'history_reference'; (history_root / 'output').mkdir(parents=True)
        history = None; child_runs = {}
        for split in SPLITS:
            history_file = Path(parent['runs'][label+'-'+split]['output']) / 'regressions.jsonl'
            old_history = read(history_file); refs[str(history_file)] = digest(history_file)
            if history is None: history = old_history
            else: assert history == old_history
            entry = dict(condition=tag, alpha=1.0, data=str(root / 'data' / (split+'.jsonl')),
                plan=str(root / 'plans' / (split+'.json')), preflight=str(root / 'preflight' / split),
                output=str(study / f'truth-beneficiary2-output-{stamp}' / label / split), status='staged_not_submitted')
            child_runs[split] = {tag: entry}; key = label+'-'+split
            script = shell(repo, root, path, key, [str(repo / '.venv-313/bin/python'), '-u', str(root / 'truth_beneficiary_revision.py'),
                '--batch', str(path), '--execute', split, '--condition', label], True)
            runs[key] = dict(condition=label, split=split, script=script, output=entry['output'], status='staged_not_submitted')
        engine.jsonl(history_root / 'output/regressions.jsonl', history)
        child = dict(controller=str(child_root), seed=seed, revision=old['revision'],
            reference_versions=old['reference_versions'], inference_source_sha256=current,
            adapter_sha256=old['adapter_sha256'], history_references={tag: str(history_root)},
            reference_sha256=common_refs, frozen_sha256=engine.hashes(child_root), runs=child_runs)
        child_path = root / (label+'-batch.json'); write(child_path, child); children[label] = str(child_path)
    assert len(set(weights)) == 3
    for phase in ('baseline', 'final'):
        shell(repo, root, path, phase, ['/usr/bin/python3.11', '-u', str(root / 'truth_beneficiary_revision.py'), '--batch', str(path), '--advance', phase], False)
    receipts = study / f'truth-beneficiary2-receipts-{stamp}'; receipts.mkdir()
    batch = dict(protocol='truth_beneficiary2', status='staged_not_submitted',
        authorization='User said lets proceed with that on 2026-10-29 after recommending a separately frozen revision with fresh worlds, multiple wordings, all controls and baseline gates.',
        created_utc=datetime.now(timezone.utc).isoformat(), controller=str(root), parent=str(parent_path),
        analysis=str(study / f'truth-beneficiary2-analysis-{stamp}'), receipts=str(receipts), children=children, runs=runs,
        maximum_primary_responses=51840, reference_sha256=refs, frozen_sha256=engine.hashes(root),
        errors={}, gates={}, results={}, audit_receipts={})
    write(path, batch); engine.verify(batch); print('FROZEN', path, flush=True)
    return path


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', type=Path); p.add_argument('--batch', type=Path); p.add_argument('--submit', action='store_true')
    p.add_argument('--advance', choices=('baseline', 'final')); p.add_argument('--execute', choices=SPLITS)
    p.add_argument('--condition', choices=LABELS)
    args = p.parse_args()
    if args.stage:
        path = stage(args.stage.resolve())
        if args.submit:
            frozen = Path(json.loads(path.read_text())['controller'])/'truth_beneficiary_revision.py'
            subprocess.run(['/usr/bin/python3.11', str(frozen), '--batch', str(path), '--submit'], check=True)
    else:
        configure()
        if args.batch and args.submit: flow.submit(args.batch)
        elif args.batch and args.advance: flow.advance(args.batch, args.advance)
        elif args.batch and args.execute and args.condition:
            batch = json.loads(args.batch.read_text()); engine.verify(batch)
            engine.execute(Path(batch['children'][args.condition]), args.execute, 'base' if args.condition == 'base' else 'adapter')
        else: p.error('stage, submit, advance or execute/condition required')
