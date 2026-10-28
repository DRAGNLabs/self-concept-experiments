import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/soo/scripts'))
from selfconcept.soo.explicit_payoff import build_dataset
import report_goal_task as prior
import reminder_wording_task as task
import reminder_wording as workflow


def originals():
    rows = build_dataset('development')
    return [e for suite in ('main', 'checks') for e in rows if e['suite'] == suite]


class ReminderWordingTests(unittest.TestCase):
    def test_exact_anchors_and_only_insertion_position_changes(self):
        source = originals(); rows = task.build(source)
        self.assertEqual(len(rows), 3024)
        self.assertEqual(len({r['example_id'] for r in rows}), 3024)
        for e in source:
            variants = {v: task.transform(e, v) for v in task.VARIANTS}
            self.assertEqual(variants['original']['prompt'], e['prompt'])
            self.assertEqual(variants['exact_late']['prompt'], prior.transform(e, 'original_reminder')['prompt'])
            for word in ('exact', 'paraphrase', 'format'):
                early, late = (variants[word+'_'+pos] for pos in ('early', 'late'))
                self.assertEqual(early['semantic_spans']['added_reminder'][0]['text'], late['semantic_spans']['added_reminder'][0]['text'])
                for r in (early, late):
                    s = r['semantic_spans']['added_reminder'][0]
                    self.assertEqual(r['prompt'].replace(s['text']+'\n\n', ''), e['prompt'])
                self.assertLess(early['semantic_spans']['added_reminder'][0]['end'], early['semantic_spans']['recipient'][0]['start'])
                self.assertGreater(late['semantic_spans']['added_reminder'][0]['start'], late['semantic_spans']['consequences'][0]['end'])
                self.assertLess(late['semantic_spans']['added_reminder'][0]['end'], late['semantic_spans']['scope'][0]['start'])
            for r in variants.values():
                for key in e['semantic_spans']:
                    self.assertEqual([s['text'] for s in r['semantic_spans'][key]], [s['text'] for s in e['semantic_spans'][key]])
                for spans in r['semantic_spans'].values():
                    for s in spans: self.assertEqual(r['prompt'][s['start']:s['end']], s['text'])

    def test_displayed_keys_and_competence_gate_excludes_message_choice(self):
        rows = task.build(originals()); responses = []
        for e in rows:
            table = [line.split(' | ') for line in e['semantic_spans']['payoffs'][0]['text'].splitlines()[2:]]
            role = int(e['query'][-1]) if e['query'].startswith('fact') else e['role']
            winner = max(table, key=lambda r: int(r[role+1]))[0]
            self.assertEqual(winner, e['expected_answer'])
            responses.append(dict(example_id=e['example_id'], completion_status='complete',
                response=e['recipient_optimal_answer'] if e['query'] == 'message' else winner))
        scores = task.score(rows, responses); gate = task.gate(scores)
        self.assertTrue(gate['passed']); self.assertEqual(len(gate['checks']), 833)
        self.assertFalse(task.gate([r for r in scores if r['variant'] != 'format_early'])['passed'])
        for r in scores:
            if r['variant'] == 'exact_late' and r['query'] == 'private': r['correct'] = False
        self.assertFalse(task.gate(scores)['passed'])
        with self.assertRaises(AssertionError): task.score(rows, responses[:-1])
        with self.assertRaises(AssertionError): task.score(rows, responses+[responses[0]])
        responses[0]['completion_status'] = 'truncated'
        self.assertFalse(task.score(rows, responses)[0]['correct'])

    def test_goal_format_position_contrasts_include_every_seed(self):
        rows = task.build(originals()); results = {}
        active = {'base': {'format_early'}, 'seed0': {'exact_early'},
                  'seed1': {'exact_late'}, 'seed2': set()}
        for label in workflow.LABELS:
            responses = [dict(example_id=e['example_id'], completion_status='complete',
                response=e['recipient_optimal_answer'] if e['query'] == 'message' and e['variant'] in active[label] else e['expected_answer']) for e in rows]
            results[label] = dict(scores=task.score(rows, responses), anchor_comparisons=[])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); r = workflow.analyze(results, out)
            cs = {(c['condition'], c['contrast']): c for c in r['contrasts'] if c['alignment'] == 'opposed' and c['role'] == 'all' and c['contrast'] != 'condition_minus_base'}
            self.assertEqual(cs['seed0', 'exact_early_minus_format_early']['baseline_adjusted_change'], 2)
            self.assertEqual(cs['seed0', 'exact_late_minus_exact_early']['change'], -1)
            self.assertEqual(cs['seed1', 'exact_late_minus_exact_early']['change'], 1)
            self.assertEqual(cs['seed2', 'exact_late_minus_exact_early']['change'], 0)
            self.assertEqual(len(workflow.read(out / 'paired.jsonl')), 12096)
            self.assertEqual(len(workflow.read(out / 'joint.jsonl')), 3024)

    def test_both_baselines_required_then_all_seeds_released(self):
        for passed in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                batch = dict(status='baseline_submitted', controller=tmp, analysis=str(root/'analysis'),
                    runs={label+'-'+split: {} for label in workflow.LABELS for split in workflow.SPLITS},
                    errors={}, gates={}, results={}, audit_receipts={})
                path = root/'batch.json'; workflow.write(path, batch)
                def audit(batch, label, split):
                    return dict(gate=dict(passed=passed if split == 'reserved' else True))
                with patch.object(workflow.engine, 'verify'), patch.object(workflow, 'audit', side_effect=audit), \
                     patch.object(workflow, 'launch', return_value='1') as launch, patch.object(workflow, 'release') as release:
                    workflow.advance(path, 'baseline')
                if passed:
                    self.assertEqual([(c.args[2], c.args[3]) for c in launch.call_args_list],
                        [(label, split) for label in workflow.LABELS[1:] for split in workflow.SPLITS])
                    self.assertEqual(release.call_args.args[1], 'final')
                else:
                    launch.assert_not_called(); release.assert_not_called()
                    self.assertEqual(json.loads(path.read_text())['status'], 'baseline_competence_failed_stopped')


if __name__ == '__main__': unittest.main()
