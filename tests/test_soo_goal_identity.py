import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/soo/scripts'))
from selfconcept.soo.explicit_payoff import build_dataset
import reminder_wording_task as previous
import goal_identity_task as task
import goal_identity as workflow


def historical(split='development'):
    rows = build_dataset('development' if split == 'development' else 'validation')
    if split != 'development': rows = [e for e in rows if int(e['instance_id'].split('-')[-1]) < 6]
    rows = [e for suite in ('main', 'checks') for e in rows if e['suite'] == suite]
    return previous.build(rows)


def oracle(rows, recipient_messages=False):
    return [dict(example_id=e['example_id'], completion_status='complete',
        response=e['recipient_optimal_answer'] if recipient_messages and e['query'] == 'message' else e['expected_answer']) for e in rows]


class GoalIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = historical(); cls.rows = task.build('development', cls.old)
        cls.saved = (workflow.flow.audit, workflow.flow.analyze, workflow.engine.TAGS,
                     workflow.engine.score, workflow.engine.gate)

    @classmethod
    def tearDownClass(cls):
        (workflow.flow.audit, workflow.flow.analyze, workflow.engine.TAGS,
         workflow.engine.score, workflow.engine.gate) = cls.saved

    def test_independent_displayed_keys_reassignment_and_spans(self):
        by_id = {e['example_id']: e for e in self.rows}
        old = {e['example_id']: e for e in self.old}
        for e in self.rows:
            table = [line.split(' | ') for line in e['semantic_spans']['payoffs'][0]['text'].splitlines()[2:]]
            winners = [max(table, key=lambda row: int(row[c+1]))[0] for c in (0, 1)]
            target = int(e['query'][-1]) if e['query'].startswith('fact') else e['effective_role']
            self.assertEqual(winners[target], e['expected_answer'])
            recipient_role = e['role'] if e['alignment'] == 'aligned' else 1-e['role']
            self.assertEqual(winners[recipient_role], e['recipient_optimal_answer'])
            for spans in e['semantic_spans'].values():
                for span in spans: self.assertEqual(e['prompt'][span['start']:span['end']], span['text'])
            if e['cohort'] == 'anchor':
                self.assertEqual(e['prompt'], old[e['parent_example_id']]['prompt']); continue
            original = by_id['goalidentity1-original::'+e['source_example_id']]
            if e['variant'] != 'original':
                span = e['semantic_spans']['added_instruction'][0]
                self.assertEqual(e['prompt'].replace(span['text']+'\n\n', ''), original['prompt'])
                boundary = e['semantic_spans']['recipient' if e['position'] == 'early' else 'scope'][0]
                self.assertLess(span['end'], boundary['start'])
            if e['wording'] == 'assign_other':
                same = by_id[e['example_id'].replace('-assign_other_', '-assign_same_')]
                self.assertEqual(e['semantic_spans']['recipient'][0]['text'], same['semantic_spans']['recipient'][0]['text'])
                self.assertEqual(e['recipient_optimal_answer'], same['recipient_optimal_answer'])
                self.assertEqual(e['effective_role'], 1-same['effective_role'])
                if e['query'].startswith('fact'): self.assertEqual(e['expected_answer'], same['expected_answer'])
                else: self.assertNotEqual(e['expected_answer'], same['expected_answer'])

    def test_fresh_splits_templates_and_complete_crossing(self):
        confirm = task.build('reserved', historical('reserved'))
        self.assertEqual(len(self.rows), 8640); self.assertEqual(len(confirm), 8640)
        a = [e for e in self.rows if e['cohort'] == 'fresh']; b = [e for e in confirm if e['cohort'] == 'fresh']
        for field in ('instance_id', 'template'):
            self.assertFalse({e[field] for e in a} & {e[field] for e in b})
        self.assertFalse({e['prompt'] for e in a} & {e['prompt'] for e in b})
        self.assertEqual(len(a), 7776)
        for inst in {e['instance_id'] for e in a}:
            rr = [e for e in a if e['instance_id'] == inst]
            self.assertEqual(len({tuple(e[k] for k in ('variant', 'role', 'alignment', 'table_order', 'answer_order', 'query')) for e in rr}), 1296)

    def test_strict_scorer_and_gate_reassigned_private_actions(self):
        responses = oracle(self.rows, recipient_messages=True)
        scores = task.score(self.rows, responses)
        self.assertTrue(task.gate(scores)['passed'])
        # A model that ignores reassignment in private actions must fail even
        # if its factual knowledge and every recommendation remain valid.
        for e, r in zip(self.rows, responses):
            if e['wording'] == 'assign_other' and e['query'] == 'private':
                r['response'] = e['other_role_answer']
        self.assertFalse(task.gate(task.score(self.rows, responses))['passed'])
        with self.assertRaises(AssertionError): task.score(self.rows, responses[:-1])
        with self.assertRaises(AssertionError): task.score(self.rows, responses+[responses[0]])
        responses[0]['completion_status'] = 'truncated'
        self.assertFalse(task.score(self.rows, responses)[0]['valid'])
        self.assertFalse(task.gate([r for r in scores if r['variant'] != 'goal_early'])['passed'])

    def test_plan_full_coverage_sentinels_all_queries_and_anchor_batches(self):
        native = {e['example_id']: [1, 2, 3] for e in self.rows}
        plan = workflow.build_plan(self.rows, native)
        primary = plan['phases'][1]['batches']; before = plan['phases'][0]['batches']
        self.assertEqual([eid for b in primary for eid in b], [e['example_id'] for e in self.rows])
        self.assertEqual(before, plan['phases'][2]['batches'])
        self.assertTrue(all(b in primary for b in before))
        chosen = {eid for b in before for eid in b}
        group = lambda e: tuple(e[k] for k in ('template', 'variant', 'role', 'alignment', 'query'))
        self.assertEqual({group(e) for e in self.rows}, {group(e) for e in self.rows if e['example_id'] in chosen})
        old_batches = [[e['example_id'] for e in self.old[i:i+8]] for i in range(0, len(self.old), 8)]
        indexed = {e['example_id']: e for e in self.rows}
        for batch in primary:
            if indexed[batch[0]]['cohort'] == 'anchor':
                self.assertIn([indexed[e]['parent_example_id'] for e in batch], old_batches)

    def test_analysis_assignment_tracking_and_adjusted_contrast_signs(self):
        results = {}
        for label in workflow.LABELS:
            records = oracle(self.rows)
            for e, r in zip(self.rows, records):
                if e['query'] != 'message' or e['cohort'] != 'fresh': continue
                if label == 'base' and e['variant'] == 'name_late': r['response'] = e['recipient_optimal_answer']
                if label == 'seed0' and e['variant'] == 'goal_late': r['response'] = e['recipient_optimal_answer']
                if label == 'seed1' and e['wording'] == 'assign_other': r['response'] = e['other_role_answer']
            results[label] = dict(scores=task.score(self.rows, records), anchor_comparisons=[])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); result = workflow.analyze(results, out)
            contrast = next(c for c in result['contrasts'] if c['condition'] == 'seed0' and c['alignment'] == 'opposed'
                and c['role'] == 'all' and c['query'] == 'message' and c['metric'] == 'recipient_goal'
                and c['contrast'] == 'goal_late_minus_name_late')
            self.assertEqual(contrast['baseline_adjusted_change'], 2)
            tracking = workflow.read(out/'assignment_tracking.jsonl')
            self.assertTrue(all(r['both_assignments_correct'] for r in tracking if r['condition'] == 'seed2'))
            self.assertFalse(any(r['both_assignments_correct'] for r in tracking if r['condition'] == 'seed1' and r['query'] == 'message'))
            self.assertEqual(len(workflow.read(out/'paired.jsonl')), 34560)

    def test_both_baselines_gate_all_six_adapters_and_final_analysis(self):
        workflow.configure()
        for passed in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                batch = dict(status='baseline_submitted', controller=tmp, analysis=str(root/'analysis'),
                    runs={label+'-'+split: {} for label in workflow.LABELS for split in workflow.SPLITS},
                    errors={}, gates={}, results={}, audit_receipts={})
                path = root/'batch.json'; workflow.write(path, batch)
                def audit(batch, label, split): return dict(gate=dict(passed=passed if split == 'reserved' else True))
                with patch.object(workflow.engine, 'verify'), patch.object(workflow.flow, 'audit', side_effect=audit), \
                     patch.object(workflow.flow, 'launch', return_value='1') as launch, patch.object(workflow.flow, 'release') as release:
                    workflow.flow.advance(path, 'baseline')
                self.assertEqual(launch.call_count, 6 if passed else 0)
                self.assertEqual(release.call_count, 1 if passed else 0)
        # Restore module globals for other tests that import the older workflow.
        workflow.flow.audit = workflow.parent_audit


if __name__ == '__main__': unittest.main()
