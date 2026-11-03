import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/soo/scripts'))
from selfconcept.soo.explicit_payoff import build_dataset
import payoff_context_task as task
import payoff_context_ablation as workflow


def originals():
    rows=build_dataset('development')
    return [e for suite in ('main','checks') for e in rows if e['suite']==suite]


class ContextTests(unittest.TestCase):
    def test_anchor_factorial_and_independent_display_keys(self):
        original=originals(); rows=task.build(original); self.assertEqual(len(rows),1728)
        self.assertEqual(len({r['example_id'] for r in rows}),1728)
        by_source={e['example_id']:e for e in original}
        for e in rows:
            old=by_source[e['source_example_id']]
            if not e['report_context'] and not e['direct_payment']:
                self.assertEqual(e['prompt'],old['prompt']); self.assertEqual(e['semantic_spans'],old['semantic_spans'])
            for k in ('question','role','recipient','payoffs','format'):
                self.assertEqual([s['text'] for s in e['semantic_spans'][k]], [s['text'] for s in old['semantic_spans'][k]])
            self.assertEqual(e['semantic_spans']['consequences'][0]['text'], task.DIRECT if e['direct_payment'] else old['semantic_spans']['consequences'][0]['text'])
            self.assertEqual('public_report_context' in e['semantic_spans'],bool(e['report_context']))
            lines=e['semantic_spans']['payoffs'][0]['text'].splitlines()[2:]
            table=[line.split(' | ') for line in lines]
            client=int(e['query'][-1]) if e['query'].startswith('fact') else e['role']
            expected=max(table,key=lambda r:int(r[client+1]))[0]
            self.assertEqual(e['expected_answer'],expected)
            for spans in e['semantic_spans'].values():
                for s in spans: self.assertEqual(e['prompt'][s['start']:s['end']],s['text'])
        for factor in task.FACTORS:
            self.assertEqual([e['source_example_id'] for e in rows if (e['report_context'],e['direct_payment'])==factor],[e['example_id'] for e in original])

    def test_gate_public_independent_private_factor_failure_and_invalid(self):
        examples=task.build(originals())
        responses=[dict(example_id=e['example_id'],response=e['other_role_answer'] if e['query']=='message' else e['expected_answer'],completion_status='complete') for e in examples]
        self.assertTrue(task.gate(task.score(examples,responses))['passed'])
        for e,r in zip(examples,responses):
            if e['report_context']==1 and e['direct_payment']==0 and e['query']=='private': r['response']=e['other_role_answer']
        self.assertFalse(task.gate(task.score(examples,responses))['passed'])
        responses[0].update(response=examples[0]['expected_answer'],completion_status='truncated')
        self.assertFalse(task.score(examples,responses)[0]['correct'])
        with self.assertRaises(AssertionError): task.score(examples,responses[:-1])
        with self.assertRaises(AssertionError): task.score(examples,responses+[responses[0]])

    def test_all_four_cells_contrasts_and_joint_controls(self):
        examples=task.build(originals()); results={}
        for tag in workflow.TAGS:
            responses=[]
            for e in examples:
                answer=e['expected_answer']
                if tag!='base' and e['query']=='message' and e['report_context']==0 and e['direct_payment']==0:
                    answer=e['recipient_optimal_answer']
                responses.append(dict(example_id=e['example_id'],response=answer,completion_status='complete'))
            results[tag]=dict(scores=task.score(examples,responses),anchor_comparisons=[])
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); result=workflow.analyze(results,out)
            cs=[c for c in result['contrasts'] if c['condition']=='adapter' and c['role']=='all' and c['alignment']=='opposed']
            self.assertEqual(next(c['value'] for c in cs if c['contrast']=='factorial_interaction'),1)
            self.assertEqual(next(c['value'] for c in cs if c['contrast']=='report_context_change_in_effect' and c['direct_payment']==0),-1)
            paired=workflow.read(out/'paired.jsonl'); self.assertEqual(len(paired),6912)
            joint=workflow.read(out/'joint.jsonl'); self.assertEqual(len(joint),1728)
            self.assertTrue(all(r['recipient_message_with_controls'] for r in joint if r['condition']=='adapter' and r['report_context']==0 and r['direct_payment']==0))

    def test_correct_high_strength_anchor_reference(self):
        paths,_=workflow.anchor_sources(ROOT/'experiments/soo/results/study2')
        for kind in paths:
            self.assertIn('alpha0.6875', paths[kind]['offset_0.6875'])
            self.assertNotEqual(paths[kind]['offset_0.6875'],paths[kind]['offset_0.5'])

    def test_failed_competence_stops_without_submissions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'analysis').mkdir()
            batch=dict(status='pilot_submitted',controller=str(root),analysis=str(root/'analysis'),
                       runs={'development':{'base':{}}},errors={},gates={})
            path=root/'batch.json'; workflow.write(path,batch)
            with patch.object(workflow.engine,'verify'), patch.object(workflow.engine,'audit_run',return_value={'gate':{'passed':False}}), patch.object(workflow.engine,'submit') as submit:
                workflow.engine.advance(path,'pilot')
            submit.assert_not_called()
            self.assertEqual(json.loads(path.read_text())['status'],'development_baseline_competence_failed_stopped')

if __name__=='__main__': unittest.main()
