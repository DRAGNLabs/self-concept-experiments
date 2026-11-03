"""Freeze and execute one gated inventory-reporting comparison; no retuning."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import difflib
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

from analyze_explicit_payoff import digest, read, verify_run, write_json
from continue_explicit_payoff import submit
from review_payoff_behavior import write_csv
from selfconcept.soo.inventory_reporting import build_dataset, score_records, summarize, baseline_gate, write_dataset

TAGS = ('base', 'adapter', 'offset_0.5', 'offset_0.6875')
SIZES = {'development': 512, 'reserved': 1024}
FIELDS = ('model', 'revision', 'dtype', 'attention', 'chat_kwargs', 'batch_size',
          'max_new_tokens', 'do_sample', 'versions', 'batches', 'source_sha256', 'task')


def extend_runner(original):
    replacements = {
        'choices=("goals_roles", "explicit_payoff"),': 'choices=("goals_roles", "explicit_payoff", "inventory_reporting"),',
        'if args.task == "explicit_payoff" and (args.resume or args.repeat_plan):':
            'if args.task in ("explicit_payoff", "inventory_reporting") and (args.resume or args.repeat_plan):',
        'payoff = getattr(args, "task", "goals_roles") == "explicit_payoff"\n    if payoff:':
            'payoff = getattr(args, "task", "goals_roles") in ("explicit_payoff", "inventory_reporting")\n    if getattr(args, "task", "goals_roles") == "inventory_reporting":\n        from .inventory_reporting import score_records as task_score, summarize as task_summarize, baseline_gate as task_gate\n    elif payoff:',
        'manifest["task"] = "explicit_payoff"': 'manifest["task"] = args.task',
    }
    result = original
    for old, new in replacements.items():
        assert result.count(old) == 1, 'unexpected inherited runner'
        result = result.replace(old, new)
    return result


def source_hashes(source):
    return {str(p.relative_to(source/'selfconcept')): digest(p) for p in sorted((source/'selfconcept').rglob('*.py'))}


def prepare_source(original, target, module, expected):
    assert source_hashes(original) == expected
    shutil.copytree(original, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    runner = target/'selfconcept/soo/run_goals_roles.py'
    old = runner.read_text(); new = extend_runner(old)
    runner.write_text(new)
    shutil.copy2(module, target/'selfconcept/soo/inventory_reporting.py')
    current = source_hashes(target)
    assert set(current) - set(expected) == {'soo/inventory_reporting.py'}
    assert not set(expected) - set(current)
    assert {k for k in expected if current[k] != expected[k]} == {'soo/run_goals_roles.py'}
    (target.parent/'inference_source.diff').write_text(''.join(difflib.unified_diff(
        old.splitlines(True), new.splitlines(True), fromfile='previous/run_goals_roles.py', tofile='inventory/run_goals_roles.py')))
    return current


def audit(entry, split, batch, out):
    path = Path(entry['path']); verify_run(path)
    out.mkdir(parents=True, exist_ok=False)
    data = read(path/'data/goals_roles_pilot/main.jsonl') + read(path/'data/goals_roles_pilot/checks.jsonl')
    assert data == build_dataset(split) and len(data) == SIZES[split], 'dataset differs from frozen factorial'
    manifest = json.loads((path/'output/manifest.json').read_text())
    responses = read(path/'output/responses.jsonl'); inputs = read(path/'output/inputs.jsonl')
    ids = [e['example_id'] for e in data]
    assert len(set(ids)) == len(ids)
    assert ids == manifest['example_ids'] == [r['example_id'] for r in responses] == [r['example_id'] for r in inputs]
    assert manifest['task'] == 'inventory_reporting' and manifest['condition'] == entry['condition']
    assert manifest['offset_alpha'] == entry['alpha'] and manifest['batch_size'] == 8
    assert manifest['source_sha256'] == batch['inference_source_sha256'] == source_hashes(path/'src')
    for name, expected in manifest['artifact_sha256'].items(): assert digest(path/name) == expected
    assert manifest['prompt_ids_sha256'] == hashlib.sha256(json.dumps([r['input_ids'] for r in inputs]).encode()).hexdigest()
    if entry['condition'] == 'offset': assert digest(path/'band_deltas.pt') == batch['offset_sha256']
    if entry['condition'] == 'adapter':
        assert {p.name: digest(p) for p in (path/'adapter').iterdir() if p.is_file()} == batch['adapter_sha256']
    if entry['condition'] == 'base':
        assert json.loads((path/'output/intervention.json').read_text())['inactive_hook_check']['passed']
    assert len(manifest['batches']) == len(data)//8
    for j, recorded in enumerate(manifest['batches']):
        rows = inputs[j*8:(j+1)*8]
        assert recorded == dict(batch_index=j, example_ids=[r['example_id'] for r in rows], padded_width=max(len(r['input_ids']) for r in rows))
    for e, encoded in zip(data, inputs):
        # Independent table maximum and explicit audit checks, including column association.
        col = (1, 0) if e['column_order'] else (0, 1)
        assert 'Quantity | '+' | '.join(e['instance']['clients'][c] for c in col) in e['prompt']
        assert f"exactly {e['actual']} crates" in e['prompt']
        for row in e['instance']['table']:
            assert row['quantity']+' | '+' | '.join(str(row['payoffs'][c]) for c in col) in e['prompt']
        winners = [r['quantity'] for r in e['instance']['table'] if all(r['payoffs'][e['role']] >= t['payoffs'][e['role']] for t in e['instance']['table'])]
        assert winners == [e['assigned_goal_answer']]
        assert e['expected_answer'] == (e['actual'] if e['query'] in ('report', 'fact_stock') else winners[0])
        assert encoded['rendered_prompt'].count(e['prompt']) == 1
    # Native tokenizer preflight is independent of model inference.
    assert inputs == read(Path(batch['preflight'])/split/'token_spans.jsonl')
    scores = score_records(data, responses)
    assert scores == read(path/'output/scored.jsonl')
    summary = summarize(scores)
    saved = json.loads((path/'output/summary.json').read_text())
    assert summary['overall'] == saved['overall'] and summary['by_query'] == saved['by_query']
    repeat = json.loads((path/'output/sentinel_repeat.json').read_text())
    assert repeat['example_ids'] == ids[:8]
    assert repeat['before'] == repeat['primary'] == responses[:8] == repeat['after'] and repeat['passed']
    history = read(path/'output/regressions.jsonl')
    reference = Path(batch['historical_references'][entry['tag']])
    assert digest(reference/'output/regressions.jsonl') == batch['historical_sha256'][entry['tag']]
    assert len(history) == 32 and history == read(reference/'output/regressions.jsonl'), 'same-condition historical records changed'
    gate = baseline_gate(scores) if entry['condition'] == 'base' else None
    if gate is not None: assert gate == saved['baseline_gate']
    result = dict(checks_passed=True, summary=summary, gate=gate, n=len(data), historical_exact=True, sentinel_exact=True)
    write_json(out/'audit.json', result)
    (out/'errors.jsonl').write_text(''.join(json.dumps(dict(example=e, score=s, response=r))+'\n'
        for e, s, r in zip(data, scores, responses) if not s['correct']))
    return result, data, inputs, scores, manifest, responses


def audit_many(entries, split, batch, out):
    results, errors = {}, {}
    for tag, entry in entries.items():
        try:
            results[tag] = audit(entry, split, batch, out/tag)
            entry['status'] = 'completed_and_audited'
        except Exception as error:
            errors[tag] = repr(error)
            entry['status'] = 'inspection_needed'
    if 'base' in results:
        reference = results['base']
        for tag, result in results.items():
            try:
                assert result[1:3] == reference[1:3], 'native input mismatch'
                for field in FIELDS: assert result[4][field] == reference[4][field], field
            except Exception as error:
                errors[tag] = repr(error)
                entries[tag]['status'] = 'inspection_needed'
    return results, errors


def analysis(runs, out):
    out.mkdir(parents=True, exist_ok=False)
    assert set(runs) == set(TAGS)
    rows, comparisons, disagreements, joint = [], [], [], []
    metrics = ('valid_answer', 'correct', 'matches_stock', 'assigned_goal_optimal', 'other_client_optimal', 'false_report')
    for tag, result in runs.items():
        rows += [dict(s, condition=tag) for s in result[3]]
        contexts = defaultdict(dict)
        for e, s, response in zip(result[1], result[3], result[5]):
            contexts[e['context_id']][e['query']] = s
        for context, qs in contexts.items():
            assert set(qs) == {'private', 'report', 'fact_stock', 'fact_goal'}
            joint.append(dict(condition=tag, context_id=context, instance_id=qs['private']['instance_id'],
                role=qs['private']['role'], alignment=qs['private']['alignment'],
                truthful_report=qs['report']['correct'], private_correct=qs['private']['correct'],
                stock_correct=qs['fact_stock']['correct'], goal_correct=qs['fact_goal']['correct'],
                all_four_correct=all(s['correct'] for s in qs.values())))
        for reference_tag in ('base', 'adapter'):
            for e, s, r, ref, ref_response in zip(result[1], result[3], result[5], runs[reference_tag][3], runs[reference_tag][5]):
                comparisons.append(dict(condition=tag, reference=reference_tag, example_id=e['example_id'], query=e['query'],
                    alignment=e['alignment'], answer_changed=s['answer'] != ref['answer'],
                    tokens_changed=r['generated_token_ids'] != ref_response['generated_token_ids'],
                    gained=s['correct'] and not ref['correct'], lost=ref['correct'] and not s['correct']))
                if s['answer'] != ref['answer']:
                    disagreements.append(dict(condition=tag, reference=reference_tag, example=e,
                        score=s, reference_score=ref, response=r, reference_response=ref_response))
    write_csv(out/'all_trials.csv', rows); write_csv(out/'paired_trials.csv', comparisons)
    write_csv(out/'matched_contexts.csv', joint)
    (out/'disagreements.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in disagreements))
    tables = {}
    for filename, fields in (
        ('summary', ('condition', 'query', 'alignment')),
        ('instance_role', ('condition', 'query', 'alignment', 'instance_id', 'role')),
        ('presentation', ('condition', 'query', 'alignment', 'table_order', 'column_order', 'answer_order'))):
        groups = defaultdict(list)
        for r in rows: groups[tuple(r[k] for k in fields)].append(r)
        table = [{**dict(zip(fields, key)), 'n': len(rs), **{m: sum(r[m] for r in rs) for m in metrics}} for key, rs in groups.items()]
        write_csv(out/(filename+'.csv'), table); tables[filename] = table
    summary = dict(summary=tables['summary'], joint=[dict(condition=tag, alignment=alignment,
        n=sum(r['condition']==tag and r['alignment']==alignment for r in joint),
        all_four_correct=sum(r['condition']==tag and r['alignment']==alignment and r['all_four_correct'] for r in joint))
        for tag in TAGS for alignment in ('aligned', 'conflict')],
        interpretation='descriptive new inventory template; rotations are not independent; no equivalence or mechanism claim')
    write_json(out/'analysis.json', summary)
    return summary


def gate_decision(result):
    if not result['checks_passed']: return 'execution_discrepancy_stopped'
    if not result['gate']['competence_passed']: return 'baseline_competence_failed_stopped'
    if not result['gate']['opportunity_passed']: return 'baseline_opportunity_failed_stopped'
    return 'pass'


def advance(batch_path, phase):
    with batch_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        batch = json.loads(batch_path.read_text())
        if batch['status'] != phase+'_submitted': return
        controller = Path(batch['controller'])
        out = Path(batch['analysis'])/phase
        out.mkdir(parents=True, exist_ok=False)
        try:
            for name, expected in batch['controller_sha256'].items(): assert digest(controller/name) == expected
            for path, expected in batch['source_ledger_sha256'].items(): assert digest(Path(path)) == expected
            for path, expected in batch['preflight_sha256'].items(): assert digest(Path(path)) == expected
            def launch(split, tag):
                entry = batch['runs'][split][tag]; path = Path(entry['path'])
                verify_run(path)
                assert digest(path/'run.sh') == entry['script_sha256'] and not (path/'output').exists()
                receipt = submit(path/'run.sh', controller/'receipts'/f'{split}-{tag}.json')
                entry.update(receipt, status='submitted'); write_json(batch_path, batch)
                return receipt['job_id']
            def release(next_phase, jobs):
                receipt = submit(controller/(next_phase+'.sh'), controller/'receipts'/(next_phase+'.json'), jobs)
                batch['audit_receipts'][next_phase] = receipt
                batch['status'] = next_phase+'_submitted'
            if phase == 'pilot':
                results, errors = audit_many({'base': batch['runs']['development']['base']}, 'development', batch, out/'development')
                batch['audits'][phase] = {k: v[0] for k,v in results.items()}
                batch['errors'][phase] = errors
                decision = gate_decision(results['base'][0]) if not errors else 'execution_discrepancy_stopped'
                batch['status'] = decision
                if decision == 'pass':
                    jobs = [launch('development', tag) for tag in TAGS[1:]]
                    jobs.append(launch('reserved', 'base'))
                    release('development', jobs)
            elif phase == 'development':
                dev, errors = audit_many(batch['runs']['development'], 'development', batch, out/'development')
                reserved, reserved_errors = audit_many({'base': batch['runs']['reserved']['base']}, 'reserved', batch, out/'reserved')
                batch['audits'][phase] = {k: v[0] for k,v in dev.items()}
                batch['audits']['reserved_base'] = {k: v[0] for k,v in reserved.items()}
                batch['errors'][phase] = dict(errors, **{'reserved_'+k:v for k,v in reserved_errors.items()})
                if not errors: batch['results']['development'] = analysis(dev, out/'paired')
                decision = gate_decision(reserved['base'][0]) if not errors and not reserved_errors else 'execution_discrepancy_stopped'
                batch['status'] = 'reserved_'+decision
                if decision == 'pass': release('reserved', [launch('reserved', tag) for tag in TAGS[1:]])
            elif phase == 'reserved':
                results, errors = audit_many(batch['runs']['reserved'], 'reserved', batch, out/'reserved')
                batch['audits'][phase] = {k:v[0] for k,v in results.items()}; batch['errors'][phase] = errors
                if not errors:
                    batch['results']['reserved'] = analysis(results, out/'paired')
                    batch['status'] = 'completed_and_analyzed'
                else: batch['status'] = 'execution_discrepancy_stopped'
            else: raise ValueError(phase)
        except Exception as error:
            batch.update(status='controller_error_requires_inspection', error=repr(error))
            raise
        finally:
            batch['updated_utc'] = datetime.now(timezone.utc).isoformat()
            write_json(batch_path, batch)


def stage(grid_path):
    from launch_goals_roles import ROOT, SOO, ADAPTER, freeze
    grid = json.loads(grid_path.read_text())
    assert grid['status'] == 'completed_and_analyzed' and not grid['errors']
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root = SOO/'results/study2'; controller = root/f'inventory-reporting1-controller-{stamp}'; controller.mkdir()
    for name in ('inventory_reporting_workflow.py', 'analyze_explicit_payoff.py', 'continue_explicit_payoff.py', 'review_payoff_behavior.py', 'validate_goals_roles_tokens.py'):
        shutil.copy2(SOO/'scripts'/name, controller/name)
    for name in ('INVENTORY_REPORTING.md', 'PLAN.md', 'FINDINGS.md'): shutil.copy2(SOO/name, controller/name)
    shutil.copy2(ROOT/'tests/test_soo_inventory_reporting.py', controller/'test_soo_inventory_reporting.py')
    source = controller/'src'
    hashes = prepare_source(Path(grid['inherited_source_directory']), source, ROOT/'src/selfconcept/soo/inventory_reporting.py', grid['inference_source_sha256'])
    preflight = controller/'preflight'; preflight.mkdir()
    for split in SIZES:
        data = SOO/f'data/inventory_reporting1_{split}'; write_dataset(data, split)
        command = [str(ROOT/'.venv-313/bin/python'), str(controller/'validate_goals_roles_tokens.py'),
                   '--data', str(data/'main.jsonl'), str(data/'checks.jsonl'), '--out', str(preflight/split)]
        import os
        env = dict(os.environ, PYTHONPATH=str(source), OPENSSL_CONF='/dev/null', HF_HUB_OFFLINE='1', USE_TORCH='0')
        subprocess.run(command, env=env, check=True)
    picker = subprocess.check_output(['/usr/bin/python3.11', str(ROOT/'gpu_pick.py'), '--hours', '0.5', '--vram-gb', '75', '--cpus', '4', '--mem-gb', '96', '--no-requeue-ok', '--fast', '--cuda', '13.0', '--top', '10'], text=True)
    (controller/'gpu_pick.txt').write_text(picker)
    runs = {}
    for split in SIZES:
        runs[split] = {}
        for tag in TAGS:
            condition = 'offset' if tag.startswith('offset_') else tag
            alpha = float(tag.split('_')[1]) if condition=='offset' else 1.0
            path = freeze(condition, data_dir=SOO/f'data/inventory_reporting1_{split}', offset_alpha=alpha,
                          followup_protocol='inventory_reporting1', source_dir=source)
            assert source_hashes(path/'src') == hashes
            if condition=='offset': assert digest(path/'band_deltas.pt') == grid['offset_sha256']
            runs[split][tag] = dict(tag=tag, condition=condition, alpha=alpha, path=str(path),
                script_sha256=digest(path/'run.sh'), status='staged_not_submitted')
    batch_path = root/f'inventory-reporting1-batch-{stamp}.json'
    batch = dict(protocol='inventory_reporting1', status='staged_not_submitted', created_utc=datetime.now(timezone.utc).isoformat(),
        authorization='User explicitly approved the bounded new inventory task, baseline competence check and fixed-intervention comparison on reserved cases.',
        maximum_new_scored_responses=6144, dose_refinement_closed=True, controller=str(controller), runs=runs,
        analysis=str(root/f'inventory-reporting1-analysis-{stamp}'), preflight=str(preflight),
        preflight_sha256={str(p):digest(p) for p in preflight.rglob('*') if p.is_file()},
        inference_source_sha256=hashes, source_ledger_sha256={str(grid_path.resolve()):digest(grid_path)},
        offset_sha256=grid['offset_sha256'], adapter_sha256={p.name:digest(p) for p in ADAPTER.iterdir() if p.is_file()},
        historical_references={tag:grid['history_paths'][tag] for tag in TAGS}, audits={}, errors={}, results={}, audit_receipts={})
    batch['historical_sha256'] = {tag:digest(Path(path)/'output/regressions.jsonl') for tag,path in batch['historical_references'].items()}
    for phase in ('pilot', 'development', 'reserved'):
        script = controller/(phase+'.sh')
        command = ['/usr/bin/python3.11', '-u', str(controller/'inventory_reporting_workflow.py'), '--batch', str(batch_path), '--phase', phase]
        script.write_text(f'''#!/bin/bash --login
#SBATCH --job-name=soo-inventory-{phase}-audit
#SBATCH --partition=dw
#SBATCH --qos=dw87
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output={SOO}/slurm-logs/%x-%j.out
set -euo pipefail
export PYTHONPATH={shlex.quote(str(source))}
export OPENSSL_CONF=/dev/null
{shlex.join(command)}
''')
        subprocess.run(['bash', '-n', str(script)], check=True)
    (controller/'receipts').mkdir()
    batch['controller_sha256'] = {str(p.relative_to(controller)):digest(p) for p in controller.rglob('*')
        if p.is_file() and '__pycache__' not in p.parts}
    write_json(batch_path, batch); print(batch_path, flush=True)
    for entries in runs.values():
        for entry in entries.values(): verify_run(Path(entry['path']))
    for name, expected in batch['controller_sha256'].items(): assert digest(controller/name) == expected
    entry = runs['development']['base']
    receipt = submit(Path(entry['path'])/'run.sh', controller/'receipts/development-base.json')
    entry.update(receipt, status='submitted'); write_json(batch_path, batch)
    batch['audit_receipts']['pilot'] = submit(controller/'pilot.sh', controller/'receipts/pilot.json', [receipt['job_id']])
    batch.update(status='pilot_submitted', updated_utc=datetime.now(timezone.utc).isoformat()); write_json(batch_path, batch)
    print('baseline', receipt['job_id'], 'audit', batch['audit_receipts']['pilot']['job_id'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path); parser.add_argument('--batch', type=Path)
    parser.add_argument('--phase', choices=('pilot', 'development', 'reserved'))
    args = parser.parse_args()
    if args.stage: stage(args.stage)
    elif args.batch and args.phase: advance(args.batch, args.phase)
    else: parser.error('provide --stage SOURCE_LEDGER or --batch LEDGER --phase PHASE')
