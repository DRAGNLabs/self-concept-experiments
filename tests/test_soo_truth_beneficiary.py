import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/soo/scripts'))
import truth_beneficiary_task as task
import truth_beneficiary as workflow


def oracle(rows, report='truth_answer'):
    return [dict(example_id=e['example_id'], completion_status='complete',
        response=e[report] if e['query'] == 'report' else e['expected_answer']) for e in rows]


class TruthBeneficiaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = task.build('development'); cls.confirm = task.build('reserved')

    def test_raw_prompt_keys_nondominance_reminder_and_independence(self):
        for split_rows in (self.rows, self.confirm):
            by_id = {e['example_id']: e for e in split_rows}
            actual_patterns = (r'the token is (\w+)\.', r'establishes that it is (\w+)\.',
                               r'identifies it as (\w+)\.', r'inspection that it is (\w+)\.')
            for e in split_rows:
                # Parse raw prompt text independently of stored keys/spans.
                lines = e['prompt'].splitlines()
                header = next(line for line in lines if line.startswith('Label | ')).split(' | ')
                table = [line.split(' | ') for line in lines if re.fullmatch(r'\w+ \| \d+ \| \d+', line)]
                self.assertEqual(len(table), 3)
                winners = {header[c]: max(table, key=lambda row: int(row[c]))[0] for c in (1, 2)}
                clients = header[1:]
                client = next(name for name in clients if name in e['prompt'].split('\n\n')[0])
                audience_block = e['prompt'].split('\n\n')[1]
                audience_client = next(name for name in clients if name in audience_block)
                found = [m.group(1) for pattern in actual_patterns if (m := re.search(pattern, e['prompt']))]
                self.assertEqual(len(found), 1); actual = found[0]
                expected = actual if e['query'] in ('report', 'fact_state') else winners[
                    clients[int(e['query'][-1])] if e['query'] in ('fact0', 'fact1') else client]
                self.assertEqual(e['expected_answer'], expected)
                self.assertEqual(e['assigned_goal_answer'], winners[client])
                self.assertEqual(e['recipient_optimal_answer'], winners[audience_client])
                self.assertEqual(e['truth_answer'], actual)
                for a in table:
                    self.assertFalse(any(all(int(b[c]) >= int(a[c]) for c in (1, 2)) and
                        any(int(b[c]) > int(a[c]) for c in (1, 2)) for b in table if b is not a))
                if e['alignment'] == 'opposed' and e['truth_relation'] == 'neither':
                    self.assertEqual(len({actual, e['assigned_goal_answer'], e['recipient_optimal_answer']}), 3)
                for spans in e['semantic_spans'].values():
                    for s in spans: self.assertEqual(e['prompt'][s['start']:s['end']], s['text'])
                if e['variant'] == 'goal_late':
                    s = e['semantic_spans']['added_instruction'][0]
                    original = by_id['truthbenefit1-original::'+e['source_example_id']]
                    self.assertEqual(e['prompt'].replace(s['text']+'\n\n', ''), original['prompt'])
                    self.assertLess(e['semantic_spans']['report_rule'][0]['end'], s['start'])
                    self.assertLess(s['end'], e['semantic_spans']['scope'][0]['start'])

    def test_balanced_fresh_splits_and_three_state_crossing(self):
        self.assertEqual(len(self.rows), 6480); self.assertEqual(len(self.confirm), 6480)
        for field in ('instance_id', 'template', 'prompt'):
            self.assertFalse({e[field] for e in self.rows} & {e[field] for e in self.confirm})
        for rows in (self.rows, self.confirm):
            groups = defaultdict(list)
            for e in rows:
                k = tuple(e[f] for f in ('instance_id', 'variant', 'role', 'alignment', 'table_order', 'answer_order', 'query'))
                groups[k].append(e)
            for group in groups.values():
                self.assertEqual(len(group), 3)
                self.assertEqual({e['truth_relation'] for e in group}, set(task.RELATIONS))
                self.assertEqual(len({e['semantic_spans']['payoffs'][0]['text'] for e in group}), 1)
                self.assertEqual(len({e['assigned_goal_answer'] for e in group}), 1)
                self.assertEqual(len({e['recipient_optimal_answer'] for e in group}), 1)
            # Every truth category occupies each table/answer-list position equally.
            balance = defaultdict(lambda: defaultdict(int))
            for e in rows:
                if e['query'] != 'report': continue
                display = e['instance']['table'][e['table_order']:] + e['instance']['table'][:e['table_order']]
                balance[e['truth_relation'], 'table'][[r['option'] for r in display].index(e['actual'])] += 1
                balance[e['truth_relation'], 'answer'][e['choices'].index(e['actual'])] += 1
            self.assertTrue(all(len(set(v.values())) == 1 and len(v) == 3 for v in balance.values()))

    def test_strict_scoring_gate_never_selects_reporting_outcome(self):
        for mode in ('truth_answer', 'assigned_goal_answer', 'recipient_optimal_answer', 'compromise_answer'):
            self.assertTrue(task.gate(task.score(self.rows, oracle(self.rows, mode)))['passed'])
        responses = oracle(self.rows)
        for e, r in zip(self.rows, responses):
            if e['query'] == 'private': r['response'] = e['other_client_answer']
        self.assertFalse(task.gate(task.score(self.rows, responses))['passed'])
        with self.assertRaises(AssertionError): task.score(self.rows, responses[:-1])
        with self.assertRaises(AssertionError): task.score(self.rows, responses+[responses[0]])
        bad = oracle(self.rows); bad[0]['completion_status'] = 'truncated'
        self.assertFalse(task.score(self.rows, bad)[0]['valid'])
        bad = oracle(self.rows); bad[0]['response'] += ' explanation'
        self.assertFalse(task.score(self.rows, bad)[0]['valid'])

    def test_full_batches_and_sentinel_coverage(self):
        native = {e['example_id']: [1, 2] for e in self.rows}
        plan = workflow.build_plan(self.rows, native)
        primary = plan['phases'][1]['batches']; before = plan['phases'][0]['batches']
        self.assertEqual([eid for b in primary for eid in b], [e['example_id'] for e in self.rows])
        self.assertEqual(before, plan['phases'][2]['batches'])
        self.assertTrue(all(len(b) == 8 and b in primary for b in before))
        selected = {eid for b in before for eid in b}
        key = lambda e: tuple(e[k] for k in ('template', 'variant', 'role', 'alignment', 'truth_relation', 'query'))
        self.assertEqual({key(e) for e in self.rows}, {key(e) for e in self.rows if e['example_id'] in selected})
        self.assertEqual(plan['generations'], len(self.rows)+2*len(selected))

    def test_analysis_distinguishes_strategies_and_keeps_control_failures(self):
        results = {}
        for label, mode in zip(workflow.LABELS, ('assigned_goal_answer', 'truth_answer', 'recipient_optimal_answer', 'compromise_answer')):
            records = oracle(self.rows, mode)
            if label == 'base':
                for e, r in zip(self.rows, records):
                    if e['query'] == 'report' and e['variant'] == 'goal_late': r['response'] = e['truth_answer']
            if label == 'seed0':
                for e, r in zip(self.rows, records):
                    if e['query'] == 'private' and e['variant'] == 'original': r['response'] = e['other_client_answer']
            results[label] = dict(scores=task.score(self.rows, records))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); result = workflow.analyze(results, out)
            track = workflow.read(out/'truth_tracking.jsonl')
            self.assertTrue(all(r['all_three_truthful'] for r in track if r['condition'] == 'seed0'))
            self.assertFalse(any(r['all_three_truthful'] for r in track if r['condition'] in ('seed1', 'seed2')))
            self.assertTrue(all(r['all_three_assigned'] for r in track if r['condition'] == 'base' and r['variant'] == 'original'))
            self.assertTrue(all(r['all_three_recipient'] for r in track if r['condition'] == 'seed1'))
            self.assertTrue(all(r['all_three_compromise'] for r in track if r['condition'] == 'seed2'))
            j = next(r for r in result['joint_summary'] if r['condition']=='seed0' and r['variant']=='original'
                and r['alignment']=='opposed' and r['truth_relation']=='neither')
            self.assertGreater(j['n'], 0); self.assertEqual(j['truthful_with_controls'], 0)
            c = next(c for c in result['contrasts'] if c['condition']=='seed0' and c['contrast']=='condition_minus_base'
                and c['variant']=='original' and c['readout']=='report' and c['metric']=='truthful'
                and c['alignment']=='opposed' and c['truth_relation']=='neither')
            self.assertEqual(c['value'], 1)
            c = next(c for c in result['contrasts'] if c['condition']=='seed0' and c['contrast']=='goal_late_minus_original'
                and c['readout']=='joint' and c['metric']=='truthful_with_controls'
                and c['alignment']=='opposed' and c['truth_relation']=='neither')
            self.assertEqual(c['baseline_adjusted_change'], 0)
            c = next(c for c in result['contrasts'] if c['condition']=='seed1' and c['contrast']=='goal_late_minus_original'
                and c['readout']=='report' and c['metric']=='truthful'
                and c['alignment']=='opposed' and c['truth_relation']=='neither')
            self.assertEqual(c['change'], 0); self.assertEqual(c['baseline_adjusted_change'], -1)

    def test_both_baselines_gate_all_seeds_integrity_and_final_analysis(self):
        saved = (workflow.flow.audit, workflow.flow.analyze, workflow.engine.TAGS, workflow.engine.score, workflow.engine.gate)
        try:
            workflow.configure()
            for scenario in ('pass', 'competence_fail', 'integrity_fail'):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    batch = dict(status='baseline_submitted', controller=tmp, analysis=str(root/'analysis'),
                        runs={label+'-'+split: {} for label in workflow.LABELS for split in workflow.SPLITS},
                        errors={}, gates={}, results={}, audit_receipts={})
                    path = root/'batch.json'; workflow.write(path, batch)
                    def audit(batch, label, split):
                        if split == 'reserved' and scenario == 'integrity_fail': raise AssertionError('sentinel mismatch')
                        return dict(gate=dict(passed=not(split == 'reserved' and scenario == 'competence_fail')))
                    with patch.object(workflow.engine, 'verify'), patch.object(workflow.flow, 'audit', side_effect=audit), \
                         patch.object(workflow.flow, 'launch', return_value='1') as launch, patch.object(workflow.flow, 'release') as release:
                        workflow.flow.advance(path, 'baseline')
                    self.assertEqual(launch.call_count, 6 if scenario == 'pass' else 0)
                    self.assertEqual(release.call_count, 1 if scenario == 'pass' else 0)
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); path = root/'batch.json'
                batch = dict(status='final_submitted', controller=tmp, analysis=str(root/'analysis'),
                    runs={label+'-'+split: {} for label in workflow.LABELS for split in workflow.SPLITS},
                    errors={}, gates={}, results={}, audit_receipts={})
                workflow.write(path, batch)
                with patch.object(workflow.engine, 'verify'), \
                     patch.object(workflow.flow, 'audit', return_value={'gate': {'passed': True}}), \
                     patch.object(workflow.flow, 'analyze', return_value={'tested': True}) as analyze:
                    workflow.flow.advance(path, 'final')
                self.assertEqual(analyze.call_count, 2)
                self.assertEqual(json.loads(path.read_text())['status'], 'completed_and_analyzed')
        finally:
            (workflow.flow.audit, workflow.flow.analyze, workflow.engine.TAGS, workflow.engine.score, workflow.engine.gate) = saved


if __name__ == '__main__': unittest.main()
