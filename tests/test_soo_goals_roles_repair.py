"""Check controlled wording changes, paired keys, and diagnostic execution safeguards."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import patch

from selfconcept.soo.goals_roles_crossed import build_dataset as old_dataset
from selfconcept.soo.goals_roles_repair import build_dataset
from selfconcept.soo.goals_roles_repeatability import execute_plan
from selfconcept.soo.score_goals_roles import score_records

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/soo/scripts'))
from audit_goals_roles_crossed import audit
from continue_goals_roles_repair import gate_passed, main as continue_repair


class RepairTests(unittest.TestCase):
    def test_only_role_scope_changes_and_all_keys_retained(self):
        old, new = old_dataset(), build_dataset()
        self.assertEqual(len(new), 5376)
        for a, b in zip(old, new):
            for field in ('world', 'outcomes', 'choices', 'expected_answer', 'factors',
                          'assigned_goal_answer', 'other_role_answer', 'truthful_message_answer'):
                self.assertEqual(a[field], b[field])
            for kind, spans in a['semantic_spans'].items():
                if kind != 'role':
                    self.assertEqual([s['text'] for s in spans], [s['text'] for s in b['semantic_spans'][kind]])
            self.assertIn('question_scope', b['semantic_spans'])
            for spans in b['semantic_spans'].values():
                for span in spans:
                    self.assertEqual(b['prompt'][span['start']:span['end']], span['text'])

    def test_pilot_is_exact_balanced_subset_of_frozen_revision(self):
        full = {r['reference_crossed1_example_id']: r for r in build_dataset()}
        pilot = build_dataset(True)
        self.assertEqual(len(pilot), 1792)
        self.assertEqual(len({r['base_world_id'] for r in pilot}), 4)
        for row in pilot:
            other = full[row['reference_crossed1_example_id']]
            self.assertEqual(row['prompt'], other['prompt'])
            self.assertEqual(row['expected_answer'], other['expected_answer'])

    def test_repeat_execution_and_token_identity_guard(self):
        rows = build_dataset()[:2]
        encoded = [{'example_id': r['example_id'], 'input_ids': [i, 7]} for i, r in enumerate(rows)]
        ids = [r['example_id'] for r in rows]
        def generate(batch):
            return [{'example_id': r['example_id'], 'response': rows[ids.index(r['example_id'])]['expected_answer'],
                     'completion_status': 'complete', 'generated_token_ids': [42, 1]} for r in batch]
        raw = generate(encoded)
        plan = {'targets': ids, 'expected_input_ids': {r['example_id']: r['input_ids'] for r in encoded},
                'historical_responses': {r['example_id']: r for r in raw}, 'historical_discrepancies': [],
                'belief_order_pairs': [ids],
                'phases': [{'name': name, 'batches': [[i] for i in ids] if name == 'singleton' else [ids]}
                           for name in ('fixed_1', 'fixed_2', 'shuffled_1', 'shuffled_2', 'singleton',
                                        'crossed_fixed_1', 'crossed_fixed_2')]}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'plan.json'
            p.write_text(json.dumps(plan))
            out = Path(d)/'out'
            out.mkdir()
            execute_plan(p, out, rows, encoded, generate)
            summary = json.loads((out/'summary.json').read_text())
            self.assertTrue(summary['all_valid'])
            self.assertEqual(summary['generations'], 14)
            self.assertTrue(all(not c['answer_changes'] for c in summary['comparisons'].values()))
            changed = deepcopy(encoded)
            changed[0]['input_ids'][0] = 99
            with self.assertRaises(AssertionError):
                execute_plan(p, out, rows, changed, generate)

    def test_audit_accepts_complete_oracle_and_rejects_failed_cell(self):
        thresholds = {'valid_answers_each_family': .95,
                      'fact_prediction_private_and_comprehension_each_family': .90,
                      'opposed_message_assigned_goal_each_family': .75}
        for pilot in (True, False):
            rows = build_dataset(pilot)
            with tempfile.TemporaryDirectory() as d:
                run = Path(d)
                data = run/'data/goals_roles_pilot'
                data.mkdir(parents=True)
                (run/'output').mkdir()
                def write(name, value):
                    (run/name).write_text(json.dumps(value))
                def jsonl(name, values):
                    (run/name).write_text(''.join(json.dumps(v)+'\n' for v in values))
                write('data/goals_roles_pilot/manifest.json', {'dataset_id': 'scopepilot1' if pilot else 'crossed2'})
                for suite in ('main', 'checks'):
                    jsonl(f'data/goals_roles_pilot/{suite}.jsonl', [r for r in rows if r['suite'] == suite])
                # Match ordinary runner ordering.
                rows = [r for suite in ('main', 'checks') for r in rows if r['suite'] == suite]
                write('output/manifest.json', {'condition': 'base', 'provenance': {'baseline_gate': thresholds}})
                jsonl('output/inputs.jsonl', [{'example_id': r['example_id'], 'rendered_prompt': r['prompt']} for r in rows])
                for failing in (False, True):
                    responses = [{'example_id': r['example_id'], 'completion_status': 'complete',
                        'response': next(c for c in r['choices'] if c != r['expected_answer'])
                        if failing and r['family'] == 'burglar' and r['query'] == 'prediction'
                        and r['factors'] == {'wording': 0, 'belief_swapped': True, 'target0_option': 1}
                        else r['expected_answer']} for r in rows]
                    jsonl('output/responses.jsonl', responses)
                    jsonl('output/scored.jsonl', score_records(rows, responses))
                    out = run/f'audit-{failing}'
                    audit(run, out, None)
                    result = json.loads((out/'audit.json').read_text())
                    self.assertEqual(gate_passed(result, len(rows)), not failing)
                    self.assertFalse(gate_passed(result, len(rows)+1))

    def test_failed_gate_never_submits(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            (run/'output').mkdir()
            (run/'output/summary.json').write_text('{}')
            (run/'crossed_audit').mkdir()
            (run/'crossed_audit/audit.json').write_text(json.dumps({'all_gates_passed': False, 'error_counts': {'prediction': 8}}))
            batch = run/'batch.json'
            batch.write_text(json.dumps({'pilot': str(run), 'script_sha256': {'pilot': 'test'}}))
            with patch('continue_goals_roles_repair.verify_run'), patch('continue_goals_roles_repair.submit') as submit:
                continue_repair(batch, 'pilot')
                submit.assert_not_called()
            self.assertEqual(json.loads(batch.read_text())['status'], 'pilot_failed_gate_bounded_repair_stopped')


if __name__ == '__main__':
    unittest.main()
