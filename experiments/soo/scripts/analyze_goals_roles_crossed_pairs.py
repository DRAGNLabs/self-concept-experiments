"""Execute the frozen crossed follow-up contrasts on complete runs.

Each factor difference holds source world, other factors, role, incentives, query,
and orientation fixed. Exports descriptive counts, never prompt-level p-values.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

from selfconcept.soo.score_goals_roles import read_jsonl, score_records
from analyze_goals_roles_pairs import independent_key


def write_csv(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(batch_path, out):
    batch = json.loads(batch_path.read_text())
    paths = {'base': Path(batch['baseline'])}
    paths.update({r['condition'] if r['random_seed'] is None else f"random{r['random_seed']}":
                  Path(r['path']) for r in batch['runs']})
    data_paths = [paths['base'] / 'data/goals_roles_pilot' / f for f in ('main.jsonl', 'checks.jsonl')]
    examples = {e['example_id']: e for p in data_paths for e in read_jsonl(p)}
    assert len(examples) == 5376
    assert all(independent_key(e) == e['expected_answer'] for e in examples.values())
    all_rows, source_hashes = {}, {}
    reference_inputs = read_jsonl(paths['base'] / 'output/inputs.jsonl')
    reference_manifest = json.loads((paths['base'] / 'output/manifest.json').read_text())
    for condition, path in paths.items():
        for p in data_paths:
            assert p.read_bytes() == (path / 'data/goals_roles_pilot' / p.name).read_bytes()
        assert reference_inputs == read_jsonl(path / 'output/inputs.jsonl')
        manifest = json.loads((path / 'output/manifest.json').read_text())
        for f in ('model', 'revision', 'dtype', 'attention', 'chat_kwargs', 'batch_size', 'max_new_tokens', 'do_sample'):
            assert manifest[f] == reference_manifest[f]
        raw = read_jsonl(path / 'output/responses.jsonl')
        assert len(raw) == 5376
        rows = score_records(list(examples.values()), raw)
        assert rows == read_jsonl(path / 'output/scored.jsonl')
        all_rows[condition] = {r['example_id']: r for r in rows}
        for name in ('responses.jsonl', 'scored.jsonl', 'manifest.json'):
            p = path / 'output' / name
            source_hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    correct = lambda r: bool(r['valid_answer'] and r['parsed_answer'] == r['expected_answer'])
    context_key = lambda e, q: (e['world_id'], e['role'], e['alignment'], e['orientation'], q)
    lookups = {c: {context_key(r, r['query']): r for r in rows.values()} for c, rows in all_rows.items()}
    trials = []
    for c, records in all_rows.items():
        for eid, r in records.items():
            e, b = examples[eid], all_rows['base'][eid]
            t = {'condition': c, **{f: e[f] for f in ('example_id', 'base_world_id', 'family', 'query', 'role', 'alignment', 'orientation')},
                 **e['factors'], 'baseline_correct': correct(b), 'correct': correct(r),
                 'effect': int(correct(r))-int(correct(b)),
                 'lost': correct(b) and not correct(r), 'gained': not correct(b) and correct(r),
                 'retained_correct': correct(b) and correct(r), 'retained_error': not correct(b) and not correct(r),
                 'valid': r['valid_answer'], 'switch': False, 'joint_switch': False,
                 'baseline_eligible': False, 'joint_switch_baseline_eligible': False}
            if e['query'] == 'message' and e['alignment'] == 'opposed':
                qs = ('private', 'fact', 'prediction', 'role_check', 'goal_check', 'belief_check')
                t['switch'] = correct(b) and r['valid_answer'] and r['metrics']['truthful_message']
                t['joint_switch'] = t['switch'] and all(correct(lookups[c][context_key(e, q)]) for q in qs)
                t['baseline_eligible'] = correct(b) and all(correct(lookups['base'][context_key(e, q)]) for q in qs)
                t['joint_switch_baseline_eligible'] = t['joint_switch'] and t['baseline_eligible']
            trials.append(t)
    cells = []
    groupsets = [('condition', 'family', 'query', 'alignment'),
                 ('condition', 'family', 'query', 'alignment', 'wording', 'belief_swapped', 'target0_option'),
                 ('condition', 'base_world_id', 'family', 'query', 'alignment'),
                 ('condition', 'family', 'query', 'alignment', 'role', 'orientation')]
    counts = ('baseline_correct', 'correct', 'lost', 'gained', 'retained_correct', 'retained_error',
              'valid', 'switch', 'joint_switch', 'baseline_eligible', 'joint_switch_baseline_eligible')
    for fields in groupsets:
        groups = defaultdict(list)
        for t in trials:
            groups[tuple(t[f] for f in fields)].append(t)
        for k, group in sorted(groups.items()):
            cells.append({'grouping': ','.join(fields), **dict(zip(fields, k)), 'n': len(group),
                          **{f: sum(t[f] for t in group) for f in counts}})
    factors = ('wording', 'belief_swapped', 'target0_option')
    differences = []
    for factor in factors:
        match_fields = ('condition', 'base_world_id', 'family', 'query', 'role', 'alignment', 'orientation',
                        *(f for f in factors if f != factor))
        groups = defaultdict(dict)
        for t in trials:
            groups[tuple(t[f] for f in match_fields)][int(t[factor])] = t
        for k, group in sorted(groups.items()):
            assert set(group) == {0, 1}
            zero, one = group[0], group[1]
            differences.append({'factor': factor, **dict(zip(match_fields, k)),
                                'zero_id': zero['example_id'], 'one_id': one['example_id'],
                                'baseline_correct_delta': int(one['baseline_correct'])-int(zero['baseline_correct']),
                                'correct_delta': int(one['correct'])-int(zero['correct']),
                                'effect_delta': one['effect']-zero['effect'],
                                'switch_delta': int(one['switch'])-int(zero['switch']),
                                'joint_switch_delta': int(one['joint_switch'])-int(zero['joint_switch'])})
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / 'paired_trials.csv', trials)
    (out / 'cells.json').write_text(json.dumps(cells, indent=2)+'\n')
    # Factor-specific CSVs have different held-fixed factor columns.
    for factor in factors:
        write_csv(out / f'{factor}_contrasts.csv', [d for d in differences if d['factor'] == factor])
    for p in [batch_path, Path(__file__), *data_paths]:
        source_hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    (out / 'audit.json').write_text(json.dumps({'n': len(trials), 'conditions': list(paths),
        'checks': 'exact data/input joins, generation settings, independent keys, raw rescoring',
        'sha256': source_hashes, 'interpretation': 'Descriptive only; 12 source worlds, no independent-prompt inference'}, indent=2)+'\n')
    print(json.dumps({'out': str(out), 'trials': len(trials), 'factor_pairs': len(differences)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    analyze(args.batch, args.out)
