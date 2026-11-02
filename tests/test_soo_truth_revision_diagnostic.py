import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/soo/scripts'))
import truth_revision_diagnostic as diag
import truth_revision_diagnostic_task as probes


class RevisionDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = {k: probes.build(k) for k in probes.KINDS}

    def test_all_rendered_keys_spans_membership(self):
        ids = set()
        for kind, rows in self.rows.items():
            self.assertEqual(len(rows), probes.COUNTS[kind])
            self.assertEqual({r['instance_id'] for r in rows}, {f'truthbenefit2-{i:02d}' for i in (3, 4, 5)})
            for e in rows:
                self.assertNotIn(e['example_id'], ids); ids.add(e['example_id'])
                self.assertEqual(probes.independent_key(e), e['expected_answer'])
                for spans in e['semantic_spans'].values():
                    for span in spans:
                        self.assertEqual(e['prompt'][span['start']:span['end']], span['text'])
        self.assertEqual(len(ids), 28512)

    def test_substitutions_and_presentation_preserve_semantic_keys(self):
        full = {r['parent_example_id']: r for r in self.rows['table'] if r['parent_identical']}
        for rows in (self.rows['table'], self.rows['substitutions']):
            for e in rows:
                base = full[e['parent_example_id']]
                a = {k: v[0]['text'] for k, v in e['semantic_spans'].items()}
                b = {k: v[0]['text'] for k, v in base['semantic_spans'].items()}
                changed = {k for k in a if a[k] != b[k]}
                if e['substitution'] in ('none', 'numbers'):
                    self.assertLessEqual(changed, {'payoffs'})
                before = [r['option'] for r in base['instance']['table']]
                after = [r['option'] for r in e['instance']['table']]
                for key in ('expected_answer', 'truth_answer', 'assigned_goal_answer',
                            'other_client_answer', 'recipient_optimal_answer', 'compromise_answer',
                            'total_optimal_answer', 'maximin_answer'):
                    self.assertEqual(before.index(base[key]), after.index(e[key]))
                if e['substitution'] == 'numbers':
                    self.assertEqual([[v-5 for v in r['payoffs']] for r in base['instance']['table']],
                                     [r['payoffs'] for r in e['instance']['table']])

    def test_reduced_contexts_remove_only_declared_blocks(self):
        expected = {
            'table_only': {'payoffs', 'question', 'format'},
            'observation_only': {'state', 'question', 'format'},
            'allocation_only': {'role', 'payoffs', 'private_rule', 'question', 'format'},
            'goal_only': {'role', 'state', 'payoffs', 'scope', 'question', 'format'}}
        for e in self.rows['contexts']:
            if e['diagnostic_arm'] == 'full': continue
            keys = expected[e['diagnostic_arm']].copy()
            if e['variant'] == 'goal_late' and e['diagnostic_arm'] in ('goal_only', 'allocation_only'):
                keys.add('added_instruction')
            self.assertEqual(set(e['semantic_spans']), keys)

    def test_anchor_batches_and_sentinel_membership(self):
        parent = json.loads((ROOT/'experiments/soo/results/study2'/diag.PARENT).read_text())
        anchors, batches, raw, native = diag.anchors(parent)
        self.assertEqual((len(anchors), len(batches), len(raw)), (128, 16, 128))
        for kind, rows in self.rows.items():
            together = rows + anchors
            plan = diag.make_plan(together, {e['example_id']: [1] for e in together}, batches)
            primary = plan['phases'][1]['batches']; before = plan['phases'][0]['batches']
            self.assertEqual(primary[-16:], batches)
            self.assertTrue(all(len(b) == 8 for b in primary))
            self.assertEqual(before, plan['phases'][2]['batches'])
            self.assertTrue(all(b in primary for b in before))
            self.assertEqual(plan['generations'], len(together) + 16*len(before))

    def test_analysis_rescues_damages_and_semantic_label_pairing(self):
        scored = []
        for rows in self.rows.values():
            responses = []
            for e in rows:
                answer = e['expected_answer']
                if e['diagnostic_arm'] == 'full' and e['encoding'] == 'table':
                    answer = e['compromise_answer']
                if e['diagnostic_arm'] == 'observation_only':
                    answer = e['assigned_goal_answer']
                responses.append(dict(example_id=e['example_id'], response=answer, completion_status='complete'))
            scored.extend(probes.score(rows, responses))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); result = diag.analyze(scored, out)
            self.assertEqual(result['diagnostic_responses'], 28512)
            self.assertTrue(result['no_adapter_release'])
            paired = diag.read(out/'paired.jsonl')
            for r in paired:
                if r['substitution'] == 'labels':
                    self.assertFalse(r['answer_changed'])
                if r['encoding'] == 'sentences' and r['query'] == 'private':
                    self.assertTrue(r['rescued'])
            self.assertTrue(any(r['rescued'] for r in paired))
            self.assertTrue(any(r['damaged'] for r in paired))

    def test_failed_integrity_stops_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root/'batch.json'
            diag.engine.write(root/'anchor_reference.json', {})
            diag.engine.write(path, dict(status='diagnostic_submitted', controller=tmp,
                analysis=str(root/'analysis'), runs={k: {'base': {}} for k in diag.KINDS}, errors={}))
            with (patch.object(diag.engine, 'verify'),
                  patch.object(diag.engine, 'audit_run', side_effect=AssertionError('sentinel')),
                  patch.object(diag, 'analyze') as analyze):
                diag.finish(path)
            analyze.assert_not_called()
            self.assertEqual(json.loads(path.read_text())['status'], 'execution_discrepancy_stopped')

    def test_submission_is_idempotent_and_uses_picker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'job.sh').write_text('echo test\n')
            batch = dict(controller=tmp, receipts=tmp)
            entry = dict(script='job.sh')
            with (patch.object(diag.parent_flow, 'placement', return_value=dict(partition='cs', qos='cs')) as pick,
                  patch.object(diag.subprocess, 'check_output', return_value='123\n') as submit):
                a = diag.submit_gpu(batch, 'test', entry)
                b = diag.submit_gpu(batch, 'test', entry)
            self.assertEqual(a, b); pick.assert_called_once(); submit.assert_called_once()
            self.assertIn('--partition=cs', a['command']); self.assertIn('--no-requeue', a['command'])


if __name__ == '__main__': unittest.main()
