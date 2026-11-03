"""Frozen batching diagnostic, using the ordinary runner's unchanged generate call."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

from .score_goals_roles import read_jsonl, score_record


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source, crossed, out):
    rows = [r for name in ('main.jsonl', 'checks.jsonl')
            for r in read_jsonl(source / 'data/goals_roles_pilot' / name)]
    by_id = {r['example_id']: r for r in rows}
    scores = read_jsonl(source / 'output/scored.jsonl')
    failed = [r['example_id'] for r in scores if r['query'] == 'prediction'
              and r['parsed_answer'] != r['expected_answer']]
    partners = []
    for eid in failed:
        f = by_id[eid]['order_factors']
        partners.append(eid.rsplit('-f', 1)[0] + f"-f{f['facts']}b{1-f['beliefs']}a{f['answers']}")
    controls = [r['example_id'] for r in rows if r['query'] == 'prediction'
                and not r['world']['belief_swapped'] and r['world']['target0_option'] == 0
                and r['order_factors'] == {'facts': 0, 'beliefs': 0, 'answers': 0}]
    targets = sorted(set(failed + partners + controls))
    assert len(failed) == 52 and len(controls) == 24
    original_batches = [[r['example_id'] for r in rows[i:i+8]] for i in range(0, len(rows), 8)
                        if any(r['example_id'] in targets for r in rows[i:i+8])]
    selected = [eid for batch in original_batches for eid in batch]
    shuffled = list(selected)
    random.Random(20261027).shuffle(shuffled)
    shuffled_batches = [shuffled[i:i+8] for i in range(0, len(shuffled), 8)]
    phases = [{'name': f'{name}_{rep}', 'batches': batches}
              for name, batches in (('fixed', original_batches), ('shuffled', shuffled_batches))
              for rep in (1, 2)]
    phases.append({'name': 'singleton', 'batches': [[eid] for eid in targets]})
    repeats = json.loads((source / 'order_analysis/repeated_contexts.json').read_text())
    changed = [r for r in repeats if not r['same_answer']]
    assert len(changed) == 2 and all(r['example_id'] in targets for r in changed)
    crossed_rows = [r for name in ('main.jsonl', 'checks.jsonl')
                    for r in read_jsonl(crossed / 'data/goals_roles_pilot' / name)]
    crossed_targets = {r['source_id'] for r in changed}
    cross_batches = [[r['example_id'] for r in crossed_rows[i:i+8]]
                     for i in range(0, len(crossed_rows), 8)
                     if any(r['example_id'] in crossed_targets for r in crossed_rows[i:i+8])]
    extra_ids = {eid for batch in cross_batches for eid in batch}
    selected_rows = [by_id[eid] for eid in selected] + [r for r in crossed_rows if r['example_id'] in extra_ids]
    phases += [{'name': f'crossed_fixed_{rep}', 'batches': cross_batches} for rep in (1, 2)]
    sources = {}
    expected_tokens, historical = {}, {}
    for run in (source, crossed):
        for name in ('inputs.jsonl', 'responses.jsonl'):
            p = run / 'output' / name
            sources[str(p)] = digest(p)
        expected_tokens.update({r['example_id']: r['input_ids'] for r in read_jsonl(run / 'output/inputs.jsonl')})
        historical.update({r['example_id']: r for r in read_jsonl(run / 'output/responses.jsonl')})
    ids = [r['example_id'] for r in selected_rows]
    plan = {'dataset_id': 'repeatability1', 'targets': targets, 'failed_predictions': failed,
            'belief_order_pairs': list(zip(failed, partners)), 'accurate_belief_controls': controls,
            'historical_discrepancies': changed, 'phases': phases, 'source_sha256': sources,
            'expected_input_ids': {eid: expected_tokens[eid] for eid in ids},
            'historical_responses': {eid: historical[eid] for eid in ids},
            'interpretation': 'selected development diagnostic; fixed/shuffled repeats share one loaded model; no acceptance gate'}
    out.mkdir(parents=True, exist_ok=False)
    payloads = {'main.jsonl': ''.join(json.dumps(r, sort_keys=True)+'\n' for r in selected_rows),
                'checks.jsonl': '', 'repeat_plan.json': json.dumps(plan, indent=2)+'\n'}
    for name, value in payloads.items():
        (out/name).write_text(value)
    manifest = {'dataset_id': 'repeatability1', 'template_version': 2,
                'counts': {'unique_inputs': len(ids), 'targets': len(targets),
                           'generations': sum(len(b) for p in phases for b in p['batches'])},
                'sha256': {name: digest(out/name) for name in payloads}}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def compare(a, b, ids):
    return {'n': len(ids), 'answer_changes': [eid for eid in ids if a[eid]['parsed_answer'] != b[eid]['parsed_answer']],
            'token_changes': [eid for eid in ids if a[eid]['generated_token_ids'] != b[eid]['generated_token_ids']]}


def execute_plan(path, out, examples, encoded, generate):
    from .run_goals_roles import append_record
    plan = json.loads(path.read_text())
    inputs = {r['example_id']: r for r in encoded}
    by_id = {r['example_id']: r for r in examples}
    assert set(inputs) == set(plan['expected_input_ids'])
    assert all(inputs[eid]['input_ids'] == ids for eid, ids in plan['expected_input_ids'].items())
    phases = {}
    with (out/'repeat_responses.jsonl').open('x') as handle:
        for phase in plan['phases']:
            results = {}
            for index, ids in enumerate(phase['batches']):
                assert not set(ids) & results.keys()
                raw = generate([inputs[eid] for eid in ids])
                assert [r['example_id'] for r in raw] == ids
                for record in raw:
                    scored = score_record(by_id[record['example_id']], record)
                    row = {**record, 'parsed_answer': scored['parsed_answer'], 'valid_answer': scored['valid_answer'],
                           'correct': scored['valid_answer'] and scored['parsed_answer'] == scored['expected_answer'],
                           'phase': phase['name'], 'batch_index': index, 'batch_members': ids,
                           'padded_width': max(len(inputs[eid]['input_ids']) for eid in ids)}
                    results[record['example_id']] = row
                    append_record(handle, row)
            phases[phase['name']] = results
            print(f"Repeatability {phase['name']}: {len(results)} responses", flush=True)
    historical = {eid: {**raw, 'parsed_answer': score_record(by_id[eid], raw)['parsed_answer']}
                  for eid, raw in plan['historical_responses'].items()}
    comparisons = {}
    for a, b in (('fixed_1', 'fixed_2'), ('shuffled_1', 'shuffled_2'), ('fixed_1', 'shuffled_1'),
                 ('fixed_1', 'singleton'), ('shuffled_1', 'singleton')):
        comparisons[f'{a}_vs_{b}'] = compare(phases[a], phases[b], plan['targets'])
    comparisons['historical_vs_fixed'] = compare(historical, phases['fixed_1'], plan['targets'])
    cross_ids = list(phases['crossed_fixed_1'])
    comparisons['crossed_fixed_repeats'] = compare(phases['crossed_fixed_1'], phases['crossed_fixed_2'], cross_ids)
    comparisons['crossed_historical_vs_fixed'] = compare(historical, phases['crossed_fixed_1'], cross_ids)
    effects = {}
    for name, records in phases.items():
        if name.startswith('crossed'):
            continue
        effects[name] = dict(Counter((str(records[a]['correct'])+'->'+str(records[b]['correct']))
                                    for a, b in plan['belief_order_pairs']))
    result = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'status': 'completed_diagnostic',
              'targets': len(plan['targets']), 'generations': sum(map(len, phases.values())),
              'all_valid': all(r['valid_answer'] for p in phases.values() for r in p.values()),
              'comparisons': comparisons, 'belief_order_pairs_correctness': effects,
              'phase_correct': {name: {'n': len(p), 'correct': sum(r['correct'] for r in p.values())}
                                for name, p in phases.items()},
              'historical_discrepancies': [{**d, 'answers': {name: p[eid]['parsed_answer']
                    for name, p in phases.items() for eid in (d['example_id'], d['source_id']) if eid in p}}
                    for d in plan['historical_discrepancies']],
              'interpretation': 'Batch-related changes do not by themselves identify a numerical/internal mechanism.'}
    (out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('comparisons', 'historical_discrepancies')}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--crossed', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(prepare(args.source, args.crossed, args.out), indent=2))
