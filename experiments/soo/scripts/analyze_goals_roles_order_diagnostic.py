"""Audit independent order toggles, including matched answer-position effects.

Every source context is retained. This analysis cannot approve the crossed study.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

from selfconcept.soo.score_goals_roles import read_jsonl, score_records
from analyze_goals_roles_pairs import independent_key


def correct(r):
    return bool(r['valid_answer'] and r['parsed_answer'] == r['expected_answer'])


def paired_orders(rows):
    pairs = []
    for factor in ('facts', 'beliefs', 'answers'):
        others = [f for f in ('facts', 'beliefs', 'answers') if f != factor]
        groups = defaultdict(dict)
        for r in rows:
            k = (r['world_id'], r['role'], r['alignment'], r['query'], *(r[f] for f in others))
            groups[k][r[factor]] = r
        for k, values in sorted(groups.items()):
            assert set(values) == {0, 1}
            a, b = values[0], values[1]
            p = {'factor': factor, 'world_id': a['world_id'], 'base_world_id': a['base_world_id'],
                 'role': a['role'], 'alignment': a['alignment'], 'query': a['query'],
                 'belief_swapped': a['belief_swapped'], 'target0_option': a['target0_option'],
                 'held_fixed': {f: a[f] for f in others}, 'zero_id': a['example_id'], 'one_id': b['example_id'],
                 'zero_correct': a['correct'], 'one_correct': b['correct'],
                 'answer_changed': a['answer'] != b['answer'],
                 'lost': a['correct'] and not b['correct'], 'gained': not a['correct'] and b['correct']}
            if factor == 'answers':
                first, second = (a, b) if a['expected_first'] else (b, a)
                assert first['expected_first'] and not second['expected_first']
                p.update(expected_first_correct=first['correct'], expected_second_correct=second['correct'])
            pairs.append(p)
    return pairs


def analyze(run, source_run, out):
    paths = [run / 'data/goals_roles_pilot' / f for f in ('main.jsonl', 'checks.jsonl')]
    examples = {r['example_id']: r for p in paths for r in read_jsonl(p)}
    assert len(examples) == 3072
    assert all(independent_key(e) == e['expected_answer'] for e in examples.values())
    raw = read_jsonl(run / 'output/responses.jsonl')
    assert len(raw) == 3072
    scores = score_records(list(examples.values()), raw)
    assert scores == read_jsonl(run / 'output/scored.jsonl')
    source_data = {r['example_id']: r for f in ('main.jsonl', 'checks.jsonl')
                   for r in read_jsonl(source_run / 'data/goals_roles_pilot' / f)}
    source_scores = {r['example_id']: r for r in read_jsonl(source_run / 'output/scored.jsonl')}
    source_inputs = {r['example_id']: r for r in read_jsonl(source_run / 'output/inputs.jsonl')}
    inputs = {r['example_id']: r for r in read_jsonl(run / 'output/inputs.jsonl')}
    old_manifest = json.loads((source_run / 'output/manifest.json').read_text())
    manifest = json.loads((run / 'output/manifest.json').read_text())
    for f in ('model', 'revision', 'dtype', 'attention', 'chat_kwargs', 'batch_size', 'max_new_tokens', 'do_sample'):
        assert manifest[f] == old_manifest[f]
    trials, errors, repeated = [], [], []
    for s in scores:
        e = examples[s['example_id']]
        r = {f: e[f] for f in ('example_id', 'world_id', 'base_world_id', 'role', 'alignment', 'query')}
        r.update(**e['factors'], **e['order_factors'], answer=s['parsed_answer'],
                 expected=e['expected_answer'], correct=correct(s), valid=s['valid_answer'],
                 expected_first=e['expected_answer'] == e['choices'][0],
                 chose_first=s['parsed_answer'] == e['choices'][0])
        trials.append(r)
        if not correct(s):
            errors.append({**r, 'prompt': e['prompt'], 'raw_response': s['response']})
        oid = e['reference_crossed_example_id']
        if oid:
            assert e['prompt'] == source_data[oid]['prompt']
            assert inputs[e['example_id']]['input_ids'] == source_inputs[oid]['input_ids']
            repeated.append({'example_id': e['example_id'], 'source_id': oid,
                             'source_answer': source_scores[oid]['parsed_answer'], 'answer': s['parsed_answer'],
                             'same_answer': source_scores[oid]['parsed_answer'] == s['parsed_answer']})
    pairs = paired_orders(trials)
    groups = defaultdict(list)
    for r in trials:
        groups[(r['query'], r['belief_swapped'], r['target0_option'], r['facts'], r['beliefs'], r['answers'])].append(r)
    cells = [{'query': k[0], 'belief_swapped': k[1], 'target0_option': k[2],
              'facts': k[3], 'beliefs': k[4], 'answers': k[5], 'n': len(rs),
              'correct': sum(r['correct'] for r in rs), 'valid': sum(r['valid'] for r in rs),
              'first_choice': sum(r['chose_first'] for r in rs)} for k, rs in sorted(groups.items())]
    summary = []
    for factor in ('facts', 'beliefs', 'answers'):
        for query in ('prediction', 'belief_check', 'message', 'private'):
            for belief in (False, True):
                ps = [p for p in pairs if p['factor'] == factor and p['query'] == query and p['belief_swapped'] == belief]
                row = {'factor': factor, 'query': query, 'belief_swapped': belief, 'pairs': len(ps),
                       'answers_changed': sum(p['answer_changed'] for p in ps),
                       'lost_0_to_1': sum(p['lost'] for p in ps), 'gained_0_to_1': sum(p['gained'] for p in ps)}
                if factor == 'answers':
                    row.update(first_only_correct=sum(p['expected_first_correct'] and not p['expected_second_correct'] for p in ps),
                               second_only_correct=sum(not p['expected_first_correct'] and p['expected_second_correct'] for p in ps))
                summary.append(row)
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'trials.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(trials[0])); w.writeheader(); w.writerows(trials)
    for name, rows in [('paired_order_toggles.jsonl', pairs), ('errors.jsonl', errors)]:
        (out/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    for name, value in [('cells.json', cells), ('order_effects.json', summary), ('repeated_contexts.json', repeated)]:
        (out/name).write_text(json.dumps(value, indent=2)+'\n')
    audit = {'run': str(run), 'source_run': str(source_run), 'n': len(trials),
             'valid': sum(r['valid'] for r in trials), 'error_counts': dict(Counter(r['query'] for r in errors)),
             'repeated_contexts': len(repeated), 'repeated_answer_changes': sum(not r['same_answer'] for r in repeated),
             'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        [*paths, run/'output/responses.jsonl', run/'output/scored.jsonl', Path(__file__)]},
             'interpretation': 'diagnostic development analysis only; no gate override or automatic intervention release'}
    (out/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='sha256'}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    analyze(args.run, args.source_run, args.out)
