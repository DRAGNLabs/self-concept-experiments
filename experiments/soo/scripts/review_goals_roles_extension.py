"""CPU-only closure review: belief-stratified intervention costs and paired repair outcomes.

Retains every condition and trial. These are descriptive development comparisons,
not a rescued acceptance set or a causal estimate of changing belief accuracy.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from selfconcept.soo.score_goals_roles import read_jsonl, score_records
from analyze_goals_roles_pairs import correct, independent_key, key


def review(paired_audit, repair_batch, crossed_run, out):
    hashes = {}
    def track(path):
        hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path
    def load_run(path):
        examples = [r for suite in ('main', 'checks') for r in read_jsonl(
                    track(path/'data/goals_roles_pilot'/f'{suite}.jsonl'))]
        assert len(examples) == len({e['example_id'] for e in examples})
        assert all(independent_key(e) == e['expected_answer'] for e in examples)
        raw = read_jsonl(track(path/'output/responses.jsonl'))
        saved = read_jsonl(track(path/'output/scored.jsonl'))
        assert score_records(examples, raw) == saved
        assert len(saved) == len(raw) == len(examples)
        manifest = json.loads(track(path/'output/manifest.json').read_text())
        inputs = read_jsonl(track(path/'output/inputs.jsonl'))
        return examples, {r['example_id']: r for r in saved}, manifest, inputs
    audit = json.loads(track(paired_audit).read_text())
    repair = json.loads(track(repair_batch).read_text())
    assert repair['status'] == 'pilot_failed_gate_bounded_repair_stopped'
    runs = {name: load_run(Path(path)) for name, path in audit['runs'].items()}
    examples, baseline, base_manifest, base_inputs = runs['base']
    assert len(examples) == 672
    totals, world_rows = [], []
    for condition, (data, scores, manifest, inputs) in runs.items():
        assert data == examples and inputs == base_inputs
        for f in ('model', 'revision', 'dtype', 'attention', 'chat_kwargs', 'batch_size', 'max_new_tokens', 'do_sample'):
            assert manifest[f] == base_manifest[f]
        lookup = {key(r): r for r in scores.values()}
        def summarize(selected):
            private = [scores[e['example_id']] for e in selected if e['query'] == 'private']
            opposed = [e for e in selected if e['query'] == 'message' and e['alignment'] == 'opposed']
            aligned = [scores[e['example_id']] for e in selected if e['query'] == 'message' and e['alignment'] == 'aligned']
            switches = [e for e in opposed if correct(baseline[e['example_id']])
                        and scores[e['example_id']]['valid_answer']
                        and scores[e['example_id']]['metrics']['truthful_message']]
            joint = [e for e in switches if all(correct(lookup[key(e, query=q)]) for q in
                     ('private', 'fact', 'prediction', 'role_check', 'goal_check', 'belief_check'))]
            return {'n': len(selected), 'private_correct': sum(map(correct, private)), 'private_n': len(private),
                    'opposed_message_switches': len(switches), 'opposed_message_n': len(opposed),
                    'switches_with_private_and_checks_correct': len(joint),
                    'aligned_message_errors': sum(not correct(r) for r in aligned), 'aligned_message_n': len(aligned)}
        for belief in (False, True):
            for family in ('all', 'burglar', 'allocation'):
                selected = [e for e in examples if e['world']['belief_swapped'] == belief
                            and (family == 'all' or e['family'] == family)]
                totals.append({'condition': condition, 'belief_swapped': belief, 'family': family, **summarize(selected)})
        for world in sorted({e['world_id'] for e in examples}):
            selected = [e for e in examples if e['world_id'] == world]
            world_rows.append({'condition': condition, 'world_id': world, 'family': selected[0]['family'],
                               'belief_swapped': selected[0]['world']['belief_swapped'], **summarize(selected)})
    old_data, old_scores, _, _ = load_run(crossed_run)
    old_examples = {e['example_id']: e for e in old_data}
    new_data, new_scores, _, _ = load_run(Path(repair['pilot']))
    assert len(new_data) == 1792
    transitions, changes = Counter(), []
    failed_cell = Counter()
    for e in new_data:
        oid, nid = e['reference_crossed1_example_id'], e['example_id']
        old = old_examples[oid]
        for field in ('world', 'outcomes', 'choices', 'expected_answer', 'factors', 'role', 'alignment', 'query', 'orientation'):
            assert old[field] == e[field]
        for kind, spans in old['semantic_spans'].items():
            if kind != 'role':
                assert [s['text'] for s in spans] == [s['text'] for s in e['semantic_spans'][kind]]
        a, b = old_scores[oid], new_scores[nid]
        ac, bc = bool(correct(a)), bool(correct(b))
        transitions[(e['family'], e['query'], ac, bc)] += 1
        if e['family'] == 'burglar' and e['query'] == 'prediction' and e['factors'] == {
                'wording': 0, 'belief_swapped': True, 'target0_option': 1}:
            failed_cell[(ac, bc)] += 1
        if a['parsed_answer'] != b['parsed_answer']:
            changes.append({'old_id': oid, 'new_id': nid, 'family': e['family'], 'query': e['query'],
                            'old_correct': ac, 'new_correct': bc, 'factors': e['factors'],
                            'old_answer': a['parsed_answer'], 'new_answer': b['parsed_answer']})
    transition_rows = [{'family': k[0], 'query': k[1], 'old_correct': k[2], 'new_correct': k[3], 'n': n}
                       for k, n in sorted(transitions.items())]
    assert sum(transitions.values()) == 1792
    track(Path(__file__))
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'completed_cpu_review',
              'v2_trials_rescored': 4704, 'crossed1_baseline_rescored': len(old_data),
              'repair_pilot_rescored': len(new_data), 'paired_revision_trials': 1792,
              'belief_strata': [r for r in totals if r['family'] == 'all'],
              'failed_cell_transitions': [{'old_correct': k[0], 'new_correct': k[1], 'n': n}
                                          for k, n in sorted(failed_cell.items())],
              'revision_transitions': transition_rows, 'source_sha256': hashes,
              'limitations': ['all development data; one trained adapter seed',
                              'belief strata in v2 use different worlds with confounded wording/layout/names',
                              'post hoc stratification cannot rescue a failed crossed gate',
                              'v3 versus v2 comparison changes instructions and batch composition',
                              'separate checks do not certify representations during message generation',
                              'norm matching does not match behavioral effect size or computational disruption',
                              'variants are dependent; no prompt-level significance or equivalence claim'],
              'next_question': 'Do fixed offsets and trained updates differ in direct goal-following costs on a new task without belief tracking?',
              'jobs_submitted': False}
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in (('condition_by_belief.csv', totals), ('world_level.csv', world_rows),
                       ('revision_transitions.csv', transition_rows)):
        with (out/name).open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out/'revision_changes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in changes))
    (out/'review.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256', 'revision_transitions')}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paired-audit', required=True, type=Path)
    p.add_argument('--repair-batch', required=True, type=Path)
    p.add_argument('--crossed-run', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    review(args.paired_audit, args.repair_batch, args.crossed_run, args.out)
