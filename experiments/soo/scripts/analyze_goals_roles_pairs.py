"""Audit frozen goal/role runs and export descriptive matched comparisons.

Uses only the standard library. Counts are descriptive: the 12 development worlds
are not hundreds of independent observations. No prompt filtering or inference.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def index(rows):
    result = {r['example_id']: r for r in rows}
    assert len(result) == len(rows), 'duplicate IDs'
    return result


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def csv_out(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def correct(row):
    return row['valid_answer'] and row['parsed_answer'] == row['expected_answer']


def key(e, query=None, orientation=None):
    return (e['world_id'], e['role'], e['alignment'],
            query or e['query'], orientation or e['orientation'])


def independent_key(e):
    """Reconstruct utilities and beliefs from world metadata, not stored keys."""
    w = e['world']
    target_by_option = [0 if i == w['target0_option'] else 1 for i in range(2)]
    own = target_by_option.index(1 - e['role'] if e['family'] == 'burglar' else e['role'])
    recipient = own if e['alignment'] == 'aligned' else 1 - own
    believed = 1 - recipient if w['belief_swapped'] else recipient
    answers = {'fact': w['options'][target_by_option.index(w['fact_target'])],
               'prediction': w['options'][believed], 'belief_check': w['options'][believed],
               'role_check': w['clients'][e['role']], 'goal_check': w['clients'][e['role']],
               'private': w['options'][own], 'message': w['options'][own]}
    return answers[e['query']]


def analyze(batch_path, out):
    batch = json.loads(batch_path.read_text())
    runs = {'base': Path(batch['baseline'])}
    runs.update({r['condition'] if r['random_seed'] is None else f"random{r['random_seed']}":
                 Path(r['path']) for r in batch['runs']})
    base_path = runs['base']
    data_files = [base_path / 'data/goals_roles_pilot' / f for f in ('main.jsonl', 'checks.jsonl')]
    examples = index([e for p in data_files for e in read(p)])
    assert len(examples) == 672
    for e in examples.values():
        assert independent_key(e) == e['expected_answer'], e['example_id']
    source = base_path / 'src/selfconcept/soo/score_goals_roles.py'
    spec = importlib.util.spec_from_file_location('frozen_scorer', source)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [batch_path, Path(__file__), source, *data_files]}
    rows, lookups, raw = {}, {}, {}
    base_inputs = index(read(base_path / 'output/inputs.jsonl'))
    base_manifest = json.loads((base_path / 'output/manifest.json').read_text())
    config_fields = ('model', 'revision', 'dtype', 'attention', 'chat_kwargs', 'batch_size',
                     'max_new_tokens', 'do_sample')
    for condition, path in runs.items():
        for p in data_files:
            other = path / 'data/goals_roles_pilot' / p.name
            assert p.read_bytes() == other.read_bytes(), (condition, p.name)
        assert source.read_bytes() == (path / 'src/selfconcept/soo/score_goals_roles.py').read_bytes()
        manifest = json.loads((path / 'output/manifest.json').read_text())
        assert all(manifest[f] == base_manifest[f] for f in config_fields)
        inputs = index(read(path / 'output/inputs.jsonl'))
        assert inputs == base_inputs, condition
        raw[condition] = index(read(path / 'output/responses.jsonl'))
        rows[condition] = index(read(path / 'output/scored.jsonl'))
        assert set(rows[condition]) == set(raw[condition]) == set(examples)
        rescored = index(scorer.score_records(list(examples.values()), list(raw[condition].values())))
        assert rescored == rows[condition], condition
        assert all(r['valid_answer'] for r in rows[condition].values())
        lookups[condition] = {key(r): r for r in rows[condition].values()}
        for name in ('inputs.jsonl', 'responses.jsonl', 'scored.jsonl', 'manifest.json'):
            p = path / 'output' / name
            hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    out.mkdir(parents=True, exist_ok=False)
    audit = {'runs': {k: str(v) for k, v in runs.items()}, 'n_per_run': 672,
             'checks': ['unique complete ID joins', 'identical frozen data and rendered tokenized inputs',
                        'identical generation settings', 'independent reconstruction of all answer keys',
                        'raw responses rescored exactly match saved scores', 'all completions valid'],
             'source_sha256': hashes,
             'interpretation': 'Descriptive development analysis; six worlds/family, one trained seed. '
                               'No independence assumption for prompt variants, no significance tests.'}
    dump(out / 'audit.json', audit)
    # One row per matched prompt; retain every condition, including baseline.
    paired, errors = [], []
    for condition, records in rows.items():
        for eid, r in records.items():
            e, b = examples[eid], rows['base'][eid]
            mirror = lookups[condition][key(e, orientation='mirrored' if e['orientation'] == 'original' else 'original')]
            related = {q: lookups[condition][key(e, query=q)] for q in
                       ('fact', 'prediction', 'message', 'private', 'role_check', 'goal_check', 'belief_check')}
            row = {f: e[f] for f in ('example_id', 'world_id', 'family', 'query', 'alignment', 'role', 'orientation')}
            row.update(condition=condition, belief_swapped=e['world']['belief_swapped'],
                       wording=e['world']['wording'], baseline_answer=b['parsed_answer'],
                       answer=r['parsed_answer'], expected=e['expected_answer'],
                       baseline_correct=correct(b), correct=correct(r),
                       first_option=r['parsed_answer'] == e['choices'][0],
                       expected_first=e['expected_answer'] == e['choices'][0],
                       mirror_correct=correct(mirror), mirror_same_answer=r['parsed_answer'] == mirror['parsed_answer'])
            paired.append(row)
            if not correct(r):
                errors.append({**row, 'prompt': e['prompt'], 'raw_response': raw[condition][eid],
                               'choices': e['choices'], 'outcomes': e['outcomes'],
                               'mirror_id': mirror['example_id'], 'mirror_answer': mirror['parsed_answer'],
                               'answers_by_condition': {c: rr[eid]['parsed_answer'] for c, rr in rows.items()},
                               'related_queries': {q: {'id': rr['example_id'], 'answer': rr['parsed_answer'],
                                                        'correct': correct(rr)} for q, rr in related.items()},
                               'matches_recipient_actual_best': r['parsed_answer'] == e['truthful_message_answer'],
                               'matches_recipient_believed_best': r['parsed_answer'] == related['prediction']['expected_answer'],
                               'matches_other_role_action': r['parsed_answer'] == e['other_role_answer']})
    csv_out(out / 'paired_trials.csv', paired)
    (out / 'errors.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in errors))
    # Groupwise paired transitions. All four cells remain visible even for near-perfect baseline.
    cells = []
    for group_fields in [('family', 'query', 'alignment'), ('world_id', 'family', 'query', 'alignment'),
                         ('family', 'query', 'alignment', 'orientation'),
                         ('family', 'query', 'alignment', 'role'),
                         ('family', 'query', 'alignment', 'belief_swapped')]:
        groups = defaultdict(list)
        for r in paired:
            groups[(r['condition'], *(r[f] for f in group_fields))].append(r)
        for k, group in sorted(groups.items()):
            counts = Counter((r['baseline_correct'], r['correct']) for r in group)
            cells.append({'grouping': ','.join(group_fields), 'condition': k[0],
                          **dict(zip(group_fields, k[1:])), 'n': len(group),
                          'base_correct': sum(r['baseline_correct'] for r in group),
                          'correct': sum(r['correct'] for r in group),
                          'retained_correct': counts[True, True], 'lost': counts[True, False],
                          'gained': counts[False, True], 'retained_error': counts[False, False],
                          'first_option': sum(r['first_option'] for r in group)})
    dump(out / 'paired_cells.json', cells)
    mirrors, contexts = [], []
    for condition, lookup in lookups.items():
        for e in examples.values():
            if e['orientation'] == 'original':
                a, b = lookup[key(e)], lookup[key(e, orientation='mirrored')]
                mirrors.append({'condition': condition, **{f: e[f] for f in ('world_id', 'family', 'query', 'alignment', 'role')},
                                'both_correct': correct(a) and correct(b),
                                'both_wrong': not correct(a) and not correct(b),
                                'disagree': a['parsed_answer'] != b['parsed_answer']})
            if e['query'] != 'message':
                continue
            qs = {q: lookup[key(e, query=q)] for q in ('message', 'private', 'fact', 'prediction', 'role_check', 'goal_check', 'belief_check')}
            contexts.append({'condition': condition, **{f: e[f] for f in ('world_id', 'family', 'alignment', 'role', 'orientation')},
                             **{q: correct(r) for q, r in qs.items()},
                             'message_private_agree': qs['message']['parsed_answer'] == qs['private']['parsed_answer'],
                             'all_checks_correct': all(correct(qs[q]) for q in ('fact', 'prediction', 'role_check', 'goal_check', 'belief_check'))})
    csv_out(out / 'mirror_pairs.csv', mirrors)
    csv_out(out / 'matched_contexts.csv', contexts)
    overlaps = []
    for a, b in itertools.combinations(rows, 2):
        for family in ('burglar', 'allocation'):
            for q in ('message', 'private', 'prediction', 'fact', 'role_check', 'goal_check', 'belief_check'):
                ids = {eid for eid, e in examples.items() if e['family'] == family and e['query'] == q}
                ea, eb = ({eid for eid in ids if not correct(rows[c][eid])} for c in (a, b))
                overlaps.append({'a': a, 'b': b, 'family': family, 'query': q, 'n': len(ids),
                                 'a_errors': len(ea), 'b_errors': len(eb), 'shared_errors': len(ea & eb),
                                 'only_a': len(ea - eb), 'only_b': len(eb - ea)})
    csv_out(out / 'error_overlap.csv', overlaps)
    print(json.dumps({'out': str(out), 'audit': 'passed', 'errors_by_condition': dict(Counter(e['condition'] for e in errors))}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, default=Path('experiments/soo/results/study2/goals-roles-v2-intervention-batch-20261026.json'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.batch, args.out)
