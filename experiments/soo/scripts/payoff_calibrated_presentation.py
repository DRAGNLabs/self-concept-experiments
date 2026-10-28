"""Fixed selected-dose transfer to the existing name/column factorial."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import subprocess

from analyze_explicit_payoff import digest, read, verify_run, write_json
from analyze_payoff_followups import diagnostic_analysis
from continue_explicit_payoff import submit
from payoff_preferences_followup import SETTING_FIELDS
from review_payoff_behavior import write_csv

DOSES = (0.5, 0.6875)
PROTOCOL = 'payoff_calibrated_presentation1'


def match_presentation_inputs(first, second):
    assert first[1] == second[1] and first[2] == second[2], 'prompt/data mismatch'
    for field in SETTING_FIELDS:
        assert first[4][field] == second[4][field], field
    new, old = first[4]['source_sha256'], second[4]['source_sha256']
    if new != old:
        # Presentation2 already included payoff_diagnostic.py; only the numerical
        # module was added afterward, alongside the previously bridged alpha arg.
        assert set(new) - set(old) == {'soo/payoff_preferences.py'}
        assert not set(old) - set(new)
        assert {k for k in old if old[k] != new[k]} == {'soo/run_goals_roles.py'}


def load_run(path):
    return (None,
            read(path/'data/goals_roles_pilot/main.jsonl') + read(path/'data/goals_roles_pilot/checks.jsonl'),
            read(path/'output/inputs.jsonl'), read(path/'output/scored.jsonl'),
            json.loads((path/'output/manifest.json').read_text()))


def fixed_dose_checks(path, historical, alpha, summary, source, offset_hash):
    assert alpha in DOSES
    manifest = json.loads((path/'output/manifest.json').read_text())
    previous = json.loads((historical/'output/manifest.json').read_text())
    assert manifest['condition'] == previous['condition'] == 'offset'
    assert manifest['offset_alpha'] == previous['offset_alpha'] == alpha, 'wrong dose'
    assert manifest['source_sha256'] == previous['source_sha256'] == source, 'source changed'
    assert digest(path/'band_deltas.pt') == digest(historical/'band_deltas.pt') == offset_hash
    assert summary['sentinel_repeat_passed'], 'sentinel mismatch'
    assert summary['historical_n'] == 432
    assert summary['historical_answer_changes'] == summary['historical_token_changes'] == 0, 'unchanged cell differs'
    history = read(path/'output/regressions.jsonl')
    assert len(history) == 32 and history == read(historical/'output/regressions.jsonl'), 'same-dose history mismatch'
    return {'passed': True, 'unchanged_cell_n': 432, 'exact_unchanged_answers_and_tokens': True,
            'exact_same_dose_historical_records': True,
            'generic_full_strength_history_expectation': summary['regressions_passed']}


def audit_conditions(entries, batch, out):
    results, runs, errors = {}, {}, {}
    for tag, entry in entries.items():
        try:
            path, historical = Path(entry['path']), Path(entry['historical'])
            summary = diagnostic_analysis(path, historical, out/tag, entry['condition'])
            if entry.get('alpha') in DOSES:
                summary['fixed_dose_checks'] = fixed_dose_checks(
                    path, historical, entry['alpha'], summary,
                    batch['inference_source_sha256'], batch['offset_sha256'])
            else:
                assert summary['sentinel_repeat_passed'] and summary['regressions_passed']
                assert summary['historical_n'] == 432
                assert summary['historical_answer_changes'] == summary['historical_token_changes'] == 0
            result = load_run(path)
            assert result[4]['condition'] == entry['condition']
            if entry['condition'] == 'offset':
                assert result[4].get('offset_alpha', 1.0) == entry['alpha']
                assert digest(path/'band_deltas.pt') == batch['offset_sha256']
            results[tag], runs[tag] = summary, result
        except Exception as error:
            errors[tag] = repr(error)
    return results, runs, errors


def tables(runs, out):
    rows = []
    for tag, result in runs.items():
        for e, score in zip(result[1], result[3]):
            rows.append(dict(score, condition=tag, column_swap=e['column_swap'],
                             name_swap=e['name_swap'], canonical_example_id=e['canonical_example_id']))
    write_csv(out/'all_trials.csv', rows)
    metrics = ('correct', 'valid_answer', 'recipient_optimal', 'other_client_optimal', 'dominated')
    for fields, filename in (
        (('condition', 'column_swap', 'name_swap', 'query', 'alignment'), 'cells.csv'),
        (('condition', 'column_swap', 'name_swap', 'query', 'instance_id', 'role', 'alignment'), 'instance_cells.csv'),
        (('condition', 'column_swap', 'name_swap', 'query', 'table_order', 'answer_order'), 'order_cells.csv')):
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row[k] for k in fields)].append(row)
        write_csv(out/filename, [{**dict(zip(fields, key)), 'n': len(group),
                                 **{m: sum(r[m] for r in group) for m in metrics}}
                                for key, group in groups.items()])
    table = []
    for tag in runs:
        for column in (0, 1):
            for name in (0, 1):
                cell = [r for r in rows if (r['condition'], r['column_swap'], r['name_swap']) == (tag, column, name)]
                groups = {
                    'private': ([r for r in cell if r['query'] == 'private'], 'correct'),
                    'facts': ([r for r in cell if r['query'].startswith('fact')], 'correct'),
                    'opposed_recipient': ([r for r in cell if r['query'] == 'message' and r['alignment'] == 'opposed'], 'recipient_optimal'),
                    'aligned': ([r for r in cell if r['query'] == 'message' and r['alignment'] == 'aligned'], 'correct')}
                row = {'condition': tag, 'column_swap': column, 'name_swap': name}
                for key, (group, metric) in groups.items():
                    row[key] = sum(r[metric] for r in group)
                    row[key+'_n'] = len(group)
                table.append(row)
    lookup = {(r['condition'], r['column_swap'], r['name_swap']): r for r in table}
    for row in table:
        original = lookup[row['condition'], 0, 0]
        adapter = lookup['adapter', row['column_swap'], row['name_swap']]
        for metric in ('private', 'facts', 'opposed_recipient', 'aligned'):
            row[metric+'_change_from_original'] = row[metric] - original[metric]
            row[metric+'_minus_adapter'] = row[metric] - adapter[metric]
    write_csv(out/'presentation_table.csv', table)
    pairs, disagreements, contexts = [], [], defaultdict(dict)
    adapter = runs['adapter'][3]
    for tag, result in runs.items():
        for e, score, reference in zip(result[1], result[3], adapter):
            assert score['example_id'] == reference['example_id']
            pair = {'condition': tag, 'example_id': score['example_id'], 'query': score['query'],
                    'column_swap': e['column_swap'], 'name_swap': e['name_swap'],
                    'answer_differs': score['answer'] != reference['answer'],
                    'condition_correct': score['correct'], 'adapter_correct': reference['correct']}
            pairs.append(pair)
            if pair['answer_differs']:
                disagreements.append({'condition': tag, 'example': e, 'condition_score': score, 'adapter_score': reference})
            contexts[tag, score['context_id']][score['query']] = score
    write_csv(out/'adapter_pairs.csv', pairs)
    (out/'adapter_disagreements.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in disagreements))
    joint = []
    for (tag, context), qs in contexts.items():
        assert set(qs) == {'private', 'message', 'fact0', 'fact1'}
        joint.append({'condition': tag, 'context_id': context, 'alignment': qs['private']['alignment'],
                      'recipient_message': qs['message']['recipient_optimal'],
                      'private_and_facts_correct': all(qs[q]['correct'] for q in ('private', 'fact0', 'fact1'))})
    write_csv(out/'matched_contexts.csv', joint)
    return table


def analyze(batch_path):
    batch = json.loads(batch_path.read_text())
    out = Path(batch['analysis']); out.mkdir(parents=True, exist_ok=False)
    try:
        assert batch['status'] == 'submitted'
        for path, expected in batch['source_ledgers_sha256'].items():
            assert digest(Path(path)) == expected
        for name, expected in batch['controller_sha256'].items():
            assert digest(Path(batch['controller'])/name) == expected
        for entry in batch['runs'].values():
            if 'script_sha256' in entry:
                assert digest(Path(entry['path'])/'run.sh') == entry['script_sha256']
        results, runs, errors = audit_conditions(batch['runs'], batch, out/'audits')
        reference = runs.get('offset_0.5')
        if reference is None:
            errors['input_comparison'] = 'missing audited alpha 0.5 run'
        else:
            for tag, result in runs.items():
                try:
                    match_presentation_inputs(reference, result)
                except Exception as error:
                    errors[tag] = repr(error)
        result = {'conditions': results, 'errors': errors, 'all_checks_passed': not errors,
                  'interpretation': 'fixed-dose robustness on inspected templates; no new strength, holdout, equivalence or mechanism claim'}
        if not errors:
            result['table'] = tables(runs, out)
        write_json(out/'analysis.json', result)
        batch.update(status='completed_and_analyzed' if not errors else 'discrepancy_requires_inspection', result=result)
        for tag, entry in batch['runs'].items():
            if entry.get('alpha') in DOSES:
                entry['status'] = 'completed_and_audited' if tag not in errors else 'inspection_needed'
    except Exception as error:
        batch.update(status='analysis_failed_requires_inspection', error=repr(error))
        raise
    finally:
        batch['updated_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(batch_path, batch)


def stage(presentation_path, grid_path):
    from launch_goals_roles import ROOT, SOO, freeze
    presentation = json.loads(presentation_path.read_text())
    grid = json.loads(grid_path.read_text())
    assert presentation['status'] == grid['status'] == 'completed_and_analyzed'
    assert presentation['result']['all_checks_passed'] and not grid['errors']
    source = Path(grid['inherited_source_directory'])
    expected = grid['inference_source_sha256']
    assert {str(p.relative_to(source/'selfconcept')): digest(p) for p in (source/'selfconcept').rglob('*.py')} == expected
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root = SOO/'results/study2'
    controller = root/f'payoff-calibrated-presentation1-controller-{stamp}'; controller.mkdir()
    for name in ('payoff_calibrated_presentation.py', 'analyze_explicit_payoff.py', 'analyze_payoff_followups.py',
                 'continue_explicit_payoff.py', 'payoff_preferences_followup.py', 'review_payoff_behavior.py'):
        shutil.copy2(SOO/'scripts'/name, controller/name)
    for name in ('PAYOFF_CALIBRATED_PRESENTATION.md', 'PLAN.md', 'FINDINGS.md'):
        shutil.copy2(SOO/name, controller/name)
    shutil.copy2(ROOT/'tests/test_soo_calibrated_presentation.py', controller/'test_soo_calibrated_presentation.py')
    picker = subprocess.check_output(['/usr/bin/python3.11', str(ROOT/'gpu_pick.py'), '--hours', '0.5',
        '--vram-gb', '75', '--cpus', '4', '--mem-gb', '96', '--no-requeue-ok', '--fast', '--cuda', '13.0', '--top', '10'], text=True)
    (controller/'gpu_pick.txt').write_text(picker)
    runs = {}
    for condition, entry in presentation['runs'].items():
        tag = 'offset_1' if condition == 'offset' else condition
        runs[tag] = dict(path=entry['path'], historical=entry['historical'], condition=condition,
                         status='reused_completed')
        if condition == 'offset': runs[tag]['alpha'] = 1.0
    data_dir = Path(runs['base']['path'])/'data/goals_roles_pilot'
    for alpha in DOSES:
        tag = f'offset_{alpha:g}'
        path = freeze('offset', data_dir=data_dir, offset_alpha=alpha, followup_protocol=PROTOCOL, source_dir=source)
        assert {str(p.relative_to(path/'src/selfconcept')): digest(p) for p in (path/'src/selfconcept').rglob('*.py')} == expected
        assert digest(path/'band_deltas.pt') == grid['offset_sha256']
        runs[tag] = dict(path=str(path), historical=grid['history_paths'][tag], condition='offset', alpha=alpha,
                         status='staged_not_submitted', script_sha256=digest(path/'run.sh'))
    batch_path = root/f'payoff-calibrated-presentation1-batch-{stamp}.json'
    batch = {'protocol': PROTOCOL, 'status': 'staged_not_submitted',
             'authorization': 'User requested continued testing; fixed previously selected strengths on existing name/column factorial, no retuning.',
             'created_utc': datetime.now(timezone.utc).isoformat(), 'doses': DOSES,
             'maximum_new_scored_generations': 3456, 'dose_refinement_closed': True,
             'source_ledgers_sha256': {str(p.resolve()): digest(p) for p in (presentation_path, grid_path)},
             'inherited_source_directory': str(source), 'inference_source_sha256': expected,
             'offset_sha256': grid['offset_sha256'], 'runs': runs, 'controller': str(controller),
             'analysis': str(root/f'payoff-calibrated-presentation1-analysis-{stamp}')}
    command = ['/usr/bin/python3.11', '-u', str(controller/'payoff_calibrated_presentation.py'), '--analyze', str(batch_path)]
    script = controller/'analyze.sh'
    script.write_text(f'''#!/bin/bash --login
#SBATCH --job-name=soo-calibrated-presentation-audit
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
    batch['controller_sha256'] = {p.name: digest(p) for p in controller.iterdir() if p.is_file()}
    write_json(batch_path, batch); print(batch_path, flush=True)
    for entry in runs.values(): verify_run(Path(entry['path']))
    for name, expected_hash in batch['controller_sha256'].items(): assert digest(controller/name) == expected_hash
    jobs = []
    for alpha in DOSES:
        tag = f'offset_{alpha:g}'; entry = runs[tag]; path = Path(entry['path'])
        assert digest(path/'run.sh') == entry['script_sha256'] and not (path/'output').exists()
        receipt = submit(path/'run.sh', controller/'receipts'/(tag+'.json'))
        entry.update(receipt, status='submitted'); jobs.append(receipt['job_id'])
        write_json(batch_path, batch); print(tag, receipt['job_id'], flush=True)
    batch['analysis_receipt'] = submit(script, controller/'receipts/analysis.json', jobs)
    batch.update(status='submitted', updated_utc=datetime.now(timezone.utc).isoformat())
    write_json(batch_path, batch); print('analysis', batch['analysis_receipt']['job_id'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--presentation-batch', type=Path)
    parser.add_argument('--grid-batch', type=Path)
    parser.add_argument('--analyze', type=Path)
    args = parser.parse_args()
    if args.analyze:
        analyze(args.analyze)
    else:
        if not args.presentation_batch or not args.grid_batch:
            parser.error('both source batches are required')
        stage(args.presentation_batch, args.grid_batch)
