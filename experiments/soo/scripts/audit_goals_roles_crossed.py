"""Rescore one completed crossed run and audit global and factorial-cell gates.

No model loading. Baseline gate failures are reported, never removed or relabeled.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path

from selfconcept.soo.run_goals_roles import baseline_gate
from selfconcept.soo.score_goals_roles import read_jsonl, score_records, summarize
from analyze_goals_roles_pairs import independent_key


def audit(run, out, original_run):
    manifest = json.loads((run / 'output/manifest.json').read_text())
    paths = [run / 'data/goals_roles_pilot' / name for name in ('main.jsonl', 'checks.jsonl')]
    examples = [r for p in paths for r in read_jsonl(p)]
    data_manifest = json.loads((run / 'data/goals_roles_pilot/manifest.json').read_text())
    expected_n = 1792 if data_manifest['dataset_id'] == 'scopepilot1' else 5376
    by_id = {r['example_id']: r for r in examples}
    assert len(examples) == len(by_id) == expected_n
    assert all(independent_key(e) == e['expected_answer'] for e in examples)
    raw = read_jsonl(run / 'output/responses.jsonl')
    assert len(raw) == expected_n
    scores = score_records(examples, raw)
    saved = read_jsonl(run / 'output/scored.jsonl')
    assert scores == saved, 'saved scoring differs'
    inputs = {r['example_id']: r for r in read_jsonl(run / 'output/inputs.jsonl')}
    assert inputs.keys() == by_id.keys()
    assert all(e['prompt'] in inputs[e['example_id']]['rendered_prompt'] for e in examples)
    correct = lambda r: r['valid_answer'] and r['parsed_answer'] == r['expected_answer']
    lookup = {(r['world_id'], r['role'], r['alignment'], r['query'], r['orientation']): r for r in scores}
    errors = []
    for r in scores:
        if correct(r):
            continue
        e = by_id[r['example_id']]
        related = {q: lookup[r['world_id'], r['role'], r['alignment'], q, r['orientation']]
                   for q in ('fact', 'prediction', 'message', 'private', 'role_check', 'goal_check', 'belief_check')}
        mirror = lookup[r['world_id'], r['role'], r['alignment'], r['query'],
                        'mirrored' if r['orientation'] == 'original' else 'original']
        errors.append({**r, 'base_world_id': e['base_world_id'], 'factors': e['factors'],
                       'prompt': e['prompt'], 'outcomes': e['outcomes'],
                       'first_option': r['parsed_answer'] == e['choices'][0],
                       'mirror_id': mirror['example_id'], 'mirror_correct': correct(mirror),
                       'related': {q: {'answer': v['parsed_answer'], 'correct': correct(v)} for q, v in related.items()}})
    result = {'run': str(run), 'condition': manifest['condition'], 'n': len(scores),
              'overall': summarize(scores)['overall'],
              'error_counts': dict(Counter(r['query'] for r in errors)), 'cells': []}
    thresholds = manifest['provenance']['baseline_gate']
    if manifest['condition'] == 'base':
        result['global_gate'] = baseline_gate(scores, thresholds)
    for wording, belief, layout in itertools.product((0, 1), repeat=3):
        selected = [r for r in scores if by_id[r['example_id']]['factors'] ==
                    {'wording': wording, 'belief_swapped': bool(belief), 'target0_option': layout}]
        assert len(selected) == expected_n // 8
        cell = {'wording': wording, 'belief_swapped': bool(belief), 'target0_option': layout,
                'summary': summarize(selected)}
        if manifest['condition'] == 'base':
            cell['gate'] = baseline_gate(selected, thresholds)
        result['cells'].append(cell)
    if manifest['condition'] == 'base':
        result['all_gates_passed'] = result['global_gate']['passed'] and all(c['gate']['passed'] for c in result['cells'])
    if original_run:
        old = {r['example_id']: r for r in read_jsonl(original_run / 'output/responses.jsonl')}
        old_inputs = {r['example_id']: r for r in read_jsonl(original_run / 'output/inputs.jsonl')}
        new = {r['example_id']: r for r in raw}
        differences = []
        for e in examples:
            if not e['original_factor_cell']:
                continue
            eid, oid = e['example_id'], e['reference_v2_example_id']
            assert inputs[eid]['input_ids'] == old_inputs[oid]['input_ids']
            if new[eid]['generated_token_ids'] != old[oid]['generated_token_ids']:
                differences.append({'example_id': eid, 'original_id': oid,
                                    'old': old[oid]['response'], 'new': new[eid]['response']})
        result['original_cell_reproducibility'] = {'n': 672, 'differences': differences,
                                                   'note': 'identical input tokens; batches may differ'}
    provenance_paths = [*paths, run / 'output/responses.jsonl', run / 'output/scored.jsonl',
                        run / 'output/manifest.json', Path(__file__)]
    result['sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in provenance_paths}
    out.mkdir(parents=True, exist_ok=False)
    (out / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'errors.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in errors))
    print(json.dumps({'run': str(run), 'condition': result['condition'], 'errors': result['error_counts'],
                      'all_gates_passed': result.get('all_gates_passed'),
                      'original_cell_differences': len(result.get('original_cell_reproducibility', {}).get('differences', []))}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--original-run', type=Path)
    args = p.parse_args()
    audit(args.run, args.out, args.original_run)
