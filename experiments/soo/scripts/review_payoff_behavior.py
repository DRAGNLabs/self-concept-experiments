"""Post hoc behavioral error inspection of all completed payoff conditions.

No selection, inference, new acceptance gate, or prompt-level significance test.
Read immutable run outputs; write a separate, reproducible descriptive analysis.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

from analyze_explicit_payoff import digest, read, write_json
from selfconcept.soo.explicit_payoff import score_records


def counts(rows):
    return {'n': len(rows), **{k: sum(r[k] for r in rows) for k in
            ('correct', 'recipient_optimal', 'other_client_optimal', 'dominated')}}


def grouped(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in fields)].append(row)
    return [{**dict(zip(fields, key)), **counts(group)} for key, group in groups.items()]


def write_csv(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inspect_split(entries):
    all_rows, pairs, blocks, exceptions, provenance = [], [], [], [], {}
    reference_data = reference_inputs = reference_manifest = None
    for entry in entries:
        run = Path(entry['path'])
        data = read(run/'data/goals_roles_pilot/main.jsonl') + read(run/'data/goals_roles_pilot/checks.jsonl')
        responses = read(run/'output/responses.jsonl')
        inputs = read(run/'output/inputs.jsonl')
        manifest = json.loads((run/'output/manifest.json').read_text())
        scores = score_records(data, responses)
        assert scores == read(run/'output/scored.jsonl')
        assert [e['example_id'] for e in data] == manifest['example_ids'] == [r['example_id'] for r in responses]
        for name, expected in manifest['artifact_sha256'].items():
            assert digest(run/name) == expected
        if reference_data is None:
            reference_data, reference_inputs, reference_manifest = data, inputs, manifest
        assert data == reference_data and inputs == reference_inputs
        for field in ('batches', 'model', 'revision', 'dtype', 'attention', 'chat_kwargs', 'batch_size', 'max_new_tokens', 'do_sample', 'source_sha256'):
            assert manifest[field] == reference_manifest[field]
        tag = entry['tag']
        provenance[tag] = {'run': str(run), 'sha256': {name: digest(run/'output'/name) for name in
                          ('responses.jsonl', 'scored.jsonl', 'inputs.jsonl', 'manifest.json')}}
        rows, lookup = [], {}
        for e, s, response in zip(data, scores, responses):
            table = e['instance']['table']
            displayed = table[e['table_order']:] + table[:e['table_order']]
            maximum = max(sum(r['payoffs']) for r in table)
            total_best = [r['option'] for r in table if sum(r['payoffs']) == maximum]
            row = {'condition': tag, **s,
                   'assigned_table_position': [r['option'] for r in displayed].index(e['assigned_goal_answer']),
                   'assigned_answer_position': e['choices'].index(e['assigned_goal_answer']),
                   'total_payoff_optimal': s['answer'] in total_best,
                   'total_payoff_unique': len(total_best) == 1,
                   'chosen_client': next((c for c in (0, 1) if s['answer'] == max(table, key=lambda r: r['payoffs'][c])['option']), None)}
            rows.append(row)
            lookup[e['instance_id'], e['role'], e['alignment'], e['query'], e['table_order'], e['answer_order']] = row
            if ((tag == 'offset' and e['query'] == 'private' and not s['correct']) or
                    (tag == 'adapter' and e['query'] == 'message' and not s['recipient_optimal'])):
                exceptions.append({'condition': tag, 'example': e, 'score': row, 'response': response})
        all_rows.extend(rows)
        for r in rows:
            if r['alignment'] == 'aligned':
                other = lookup[r['instance_id'], r['role'], 'opposed', r['query'], r['table_order'], r['answer_order']]
                pairs.append({'condition': tag, 'factor': 'recipient_objective', 'query': r['query'],
                              'instance_id': r['instance_id'], 'role': r['role'],
                              'first': r['example_id'], 'second': other['example_id'],
                              'answer_changed': r['answer'] != other['answer'],
                              'lost_correct': r['correct'] and not other['correct'],
                              'gained_correct': not r['correct'] and other['correct'],
                              'recipient_optimal_both': r['recipient_optimal'] and other['recipient_optimal']})
            if r['query'] == 'private':
                other = lookup[r['instance_id'], r['role'], r['alignment'], 'message', r['table_order'], r['answer_order']]
                pairs.append({'condition': tag, 'factor': 'action_channel', 'query': 'private_to_message',
                              'instance_id': r['instance_id'], 'role': r['role'],
                              'first': r['example_id'], 'second': other['example_id'],
                              'answer_changed': r['answer'] != other['answer'],
                              'lost_correct': r['correct'] and not other['correct'],
                              'gained_correct': not r['correct'] and other['correct'],
                              'recipient_optimal_both': r['recipient_optimal'] and other['recipient_optimal']})
        groups = defaultdict(list)
        for r in rows:
            groups[r['instance_id'], r['role'], r['alignment'], r['query']].append(r)
        for key, group in groups.items():
            assert len(group) == 9
            blocks.append({'condition': tag, **dict(zip(('instance_id', 'role', 'alignment', 'query'), key)),
                           **counts(group), 'distinct_answers': len({r['answer'] for r in group}),
                           'correct_all_orders': all(r['correct'] for r in group),
                           'recipient_all_orders': all(r['recipient_optimal'] for r in group)})
    pair_groups = defaultdict(list)
    for r in pairs:
        pair_groups[r['condition'], r['factor'], r['query']].append(r)
    pair_summary = [{**dict(zip(('condition', 'factor', 'query'), key)), 'n': len(group),
                     **{k: sum(r[k] for r in group) for k in
                        ('answer_changed', 'lost_correct', 'gained_correct', 'recipient_optimal_both')}}
                    for key, group in pair_groups.items()]
    block_groups = defaultdict(list)
    for r in blocks:
        block_groups[r['condition'], r['query'], r['alignment']].append(r)
    block_summary = [{**dict(zip(('condition', 'query', 'alignment'), key)), 'n_blocks': len(group),
                      'variable_across_orders': sum(r['distinct_answers'] > 1 for r in group),
                      'correct_all_orders': sum(r['correct_all_orders'] for r in group),
                      'recipient_all_orders': sum(r['recipient_all_orders'] for r in group)}
                     for key, group in block_groups.items()]
    return all_rows, pairs, blocks, exceptions, {
        'provenance': provenance, 'role_alignment': grouped(all_rows, ('condition', 'query', 'role', 'alignment')),
        'instance': grouped(all_rows, ('condition', 'query', 'instance_id', 'alignment')),
        'positions': grouped(all_rows, ('condition', 'query', 'assigned_table_position', 'assigned_answer_position')),
        'pair_summary': pair_summary, 'order_block_summary': block_summary}


def main(batch_path, out):
    batch = json.loads(batch_path.read_text())
    assert batch['status'] == 'finished'
    out.mkdir(parents=True, exist_ok=False)
    result = {'source_ledger': str(batch_path.resolve()), 'source_ledger_sha256': digest(batch_path),
              'analysis_script_sha256': digest(Path(__file__)),
              'interpretation': 'Post hoc descriptive review; instances and rotations are not independent trials; no internal mechanism identified.',
              'splits': {}}
    for split, entries in batch['runs'].items():
        rows, pairs, blocks, exceptions, report = inspect_split(entries)
        write_csv(out/f'{split}_trials.csv', rows)
        write_csv(out/f'{split}_pairs.csv', pairs)
        write_csv(out/f'{split}_order_blocks.csv', blocks)
        (out/f'{split}_exceptions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in exceptions))
        result['splits'][split] = report
    assert digest(batch_path) == result['source_ledger_sha256']
    write_json(out/'analysis.json', result)
    print(out/'analysis.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    main(args.batch, args.out)
