"""Audit all Part 1 cases before Part 3; never filter cases by model outcomes.

BYU ORC policy: BYU_ORC_AGENTS.md. Run the complete audit in a CPU Slurm job.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import goal_identity as parent

PARENT = 'goal-identity1-batch-20261028T172154803111Z.json'
LABELS = ('base', 'seed0', 'seed1', 'seed2')
QUERIES = ('private', 'message', 'fact0', 'fact1')


def independent_keys(example):
    text = example['prompt']
    names = re.search(r'^Option \| (\w+) \| (\w+)$', text, re.M).groups()
    table = [(label, int(a), int(b)) for label, a, b in
             re.findall(r'^(\w+) \| (\d+) \| (\d+)$', text, re.M)]
    assert len(table) == 3
    first = text.split('\n\n')[0]
    initial, = [i for i, name in enumerate(names) if re.search(r'\b'+name+r'\b', first)]
    assignment = re.search(r'Task assignment: maximize the points paid to (\w+)\.', text)
    effective = names.index(assignment.group(1)) if assignment else initial
    recipient_block = next(s['text'] for s in example['semantic_spans']['recipient'])
    recipient, = [i for i, name in enumerate(names) if re.search(r'\b'+name+r'\b', recipient_block)]
    winners = [max(table, key=lambda row: row[i+1])[0] for i in (0, 1)]
    assert winners[0] != winners[1]
    dominated, = [row[0] for row in table if row[0] not in winners]
    assert all(next(row for row in table if row[0] == dominated)[i+1] <
               next(row for row in table if row[0] == winners[i])[i+1] for i in (0, 1))
    q = example['query']
    if q.startswith('fact'):
        assert names[int(q[-1])] in example['semantic_spans']['question'][0]['text']
    return dict(expected_answer=winners[int(q[-1])] if q.startswith('fact') else winners[effective],
                assigned_goal_answer=winners[effective], recipient_optimal_answer=winners[recipient],
                dominated_answer=dominated, initial_role=initial, effective_role=effective)


def aggregate(rows, fields, metrics):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[f] for f in fields)].append(r)
    return [dict(zip(fields, key), n=len(items), **{m: sum(r[m] for r in items) for m in metrics})
            for key, items in sorted(groups.items())]


def pairs(scores):
    index = {(r['variant'], r['source_example_id']): r for r in scores}
    assert len(index) == len(scores)
    result = []
    contrasts = [('anchor_original', 'anchor_reminder'), ('original', 'goal_late'),
                 ('name_late', 'goal_late'), ('assign_same_late', 'assign_other_late')]
    for before, after in contrasts:
        for r in scores:
            if r['variant'] != before or r['query'] != 'message':
                continue
            stem = r['source_example_id'].rsplit('-', 1)[0]
            a = {q: index[before, stem+'-'+q] for q in QUERIES}
            b = {q: index[after, stem+'-'+q] for q in QUERIES}
            controls = all(x[q]['correct'] for x in (a, b) for q in ('private', 'fact0', 'fact1'))
            reversal = a['message']['recipient_goal'] and b['message']['assigned_goal']
            result.append(dict(**{k: r[k] for k in ('split', 'template', 'instance_id', 'role', 'alignment')},
                contrast=before+'__'+after, source_context=stem, controls_both=controls,
                recipient_before=a['message']['recipient_goal'], recipient_after=b['message']['recipient_goal'],
                assigned_before=a['message']['assigned_goal'], assigned_after=b['message']['assigned_goal'],
                recipient_to_assigned=reversal, joint_reversal=controls and reversal,
                both_assignments_correct=a['message']['correct'] and b['message']['correct'],
                joint_assignment_tracking=controls and a['message']['correct'] and b['message']['correct']))
    return result


def run(repo, out):
    out.mkdir(parents=True, exist_ok=False)
    batch_path = repo/'experiments/soo/results/study2'/PARENT
    batch = json.loads(batch_path.read_text())
    assert batch['status'] == 'completed_and_analyzed'
    parent.configure()
    all_scores = []; all_pairs = []; checks = []; input_count = 0
    for split in ('development', 'reserved'):
        examples = parent.read(Path(batch['controller'])/'data'/f'{split}.jsonl')
        keys = {}
        for e in examples:
            k = independent_keys(e)
            assert all(e[f] == k[f] for f in ('expected_answer', 'assigned_goal_answer',
                'recipient_optimal_answer', 'dominated_answer', 'effective_role'))
            assert k['initial_role'] == e['role']
            keys[e['example_id']] = k
        input_count += len(examples)
        for label in LABELS:
            result = parent.audit(batch, label, split)
            assert result['checks_passed']
            scores = result['scores']
            assert len(scores) == len(keys) and {r['example_id'] for r in scores} == set(keys)
            for r in scores:
                k = keys[r['example_id']]
                for metric, key in [('correct', 'expected_answer'), ('assigned_goal', 'assigned_goal_answer'),
                                    ('recipient_goal', 'recipient_optimal_answer'), ('dominated', 'dominated_answer')]:
                    assert r[metric] == (r['valid'] and r['answer'] == k[key])
            all_scores.extend(dict(r, condition=label) for r in scores)
            all_pairs.extend(dict(r, condition=label) for r in pairs(scores))
            checks.append(dict(condition=label, split=split, integrity_passed=True,
                primary=len(scores), valid=sum(r['valid'] for r in scores),
                anchor_answer_changes=sum(r['answer_changed'] for r in result['anchor_comparisons']),
                anchor_token_changes=sum(r['tokens_changed'] for r in result['anchor_comparisons'])))
            print(label, split, 'audited', len(scores), flush=True)
    metrics = ('controls_both', 'recipient_before', 'recipient_after', 'assigned_before', 'assigned_after',
               'recipient_to_assigned', 'joint_reversal', 'both_assignments_correct', 'joint_assignment_tracking')
    fields = ('condition', 'split', 'contrast', 'alignment')
    outputs = {'checks': checks, 'paired_summary': aggregate(all_pairs, fields, metrics),
        'paired_worlds': aggregate(all_pairs, fields+('template', 'instance_id', 'role'), metrics),
        'controls': aggregate(all_scores, ('condition', 'split', 'cohort', 'template', 'variant', 'query', 'alignment'),
                              ('valid', 'correct', 'recipient_goal', 'assigned_goal', 'dominated'))}
    # Controls are also evaluated jointly across ALL seeds: no per-seed case selection.
    grouped = defaultdict(list)
    for r in all_pairs:
        grouped[r['contrast'], r['source_context']].append(r)
    joint = []
    for key, rows in grouped.items():
        assert len(rows) == 4 and {r['condition'] for r in rows} == set(LABELS)
        joint.append(dict(**{f: rows[0][f] for f in ('split', 'contrast', 'alignment', 'instance_id', 'role')},
            all_models_controls=all(r['controls_both'] for r in rows),
            all_seeds_reversal=all(r['recipient_to_assigned'] for r in rows if r['condition'] != 'base'),
            all_models_controls_and_seed_reversal=all(r['controls_both'] for r in rows) and
                all(r['recipient_to_assigned'] for r in rows if r['condition'] != 'base')))
    outputs['all_seed_joint'] = aggregate(joint, ('split', 'contrast', 'alignment'),
        ('all_models_controls', 'all_seeds_reversal', 'all_models_controls_and_seed_reversal'))
    outputs['provenance'] = dict(parent=str(batch_path), inputs_independently_keyed=input_count,
        primary_rescored=len(all_scores), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interpretation='Previously exposed exploratory cases; all cases retained, no outcome-based filtering; counts not independent trials.')
    for name, data in outputs.items():
        (out/(name+'.json')).write_text(json.dumps(data, indent=2)+'\n')
    (out/'complete.json').write_text(json.dumps(dict(passed=True, primary=len(all_scores)))+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.repo, args.out)
