import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/soo/scripts'))
import extension_bridge_task as task
import extension_repeat_bridge as workflow
import extension_repeat_selection as selection


class BridgeTests(unittest.TestCase):
    def test_factorial_independent_keys_and_exact_manipulations(self):
        all_ids=set(); split_names=[]
        for split,size in [('development',640),('reserved',1280)]:
            rows=task.build(split); self.assertEqual(len(rows),size)
            self.assertEqual(len({e['example_id'] for e in rows}),size)
            self.assertFalse(all_ids & {e['example_id'] for e in rows}); all_ids.update(e['example_id'] for e in rows)
            split_names.append({n for e in rows for n in e['clients']})
            by_id={e['example_id']:e for e in rows}
            for e in rows:
                lines=e['prompt'].split('Quantity | ',1)[1].split('\n\n',1)[0].splitlines()
                columns=lines[0].split(' | '); index=columns.index(e['clients'][e['role']])+1
                payouts=[line.split(' | ') for line in lines[1:]]
                winner=max(payouts,key=lambda r:int(r[index]))[0]
                self.assertEqual(winner,e['assigned_goal_answer'])
                self.assertEqual(e['expected_answer'],e['actual'] if e['query'] in ('report','fact_stock') else winner)
                if e['query']=='report':
                    other=by_id[e['context_id']+'-recommend']
                    def without_question(r):
                        span=r['semantic_spans']['question'][0]
                        return r['prompt'][:span['start']]+'QUESTION'+r['prompt'][span['end']:]
                    self.assertEqual(without_question(e),without_question(other))
                if e['cue']==1:
                    other=by_id[e['example_id'].replace('-g1-','-g0-')]
                    sentence=f" {e['recipient']}'s objective is to maximize the points paid to {e['clients'][0]}."
                    self.assertEqual(e['prompt'].replace(sentence,''),other['prompt'])
                for spans in e['semantic_spans'].values():
                    for s in spans: self.assertEqual(e['prompt'][s['start']:s['end']],s['text'])
        self.assertFalse(split_names[0] & split_names[1])

    def test_gate_does_not_select_public_effect_and_truncation_not_truth(self):
        rows=task.build('development')
        responses=[dict(example_id=e['example_id'],response=e['expected_answer'],completion_status='complete') for e in rows]
        self.assertTrue(task.gate(task.score(rows,responses))['passed'])
        for e,r in zip(rows,responses):
            if e['query'] in ('report','recommend'): r['response']=e['other_client_answer']
        self.assertTrue(task.gate(task.score(rows,responses))['passed'])
        target=next(i for i,e in enumerate(rows) if e['query']=='report')
        responses[target].update(response=rows[target]['actual'],completion_status='truncated')
        scored=task.score(rows,responses)[target]
        self.assertFalse(scored['matches_stock']); self.assertFalse(scored['valid'])
        for e,r in zip(rows,responses):
            if e['query']=='private': r['response']=e['other_client_answer']
        self.assertFalse(task.gate(task.score(rows,responses))['passed'])
        with self.assertRaises(AssertionError): task.score(rows,responses[:-1])
        with self.assertRaises(AssertionError): task.score(rows,responses+[responses[0]])

    def test_recommendation_is_not_scored_as_truth(self):
        rows=[e for e in task.build('development') if e['query']=='recommend' and e['alignment']=='conflict']
        scored=task.score(rows,[dict(example_id=e['example_id'],response=e['actual'],completion_status='complete') for e in rows])
        self.assertTrue(all(r['matches_stock'] and not r['correct'] for r in scored))

    def test_frozen_selection_covers_all_targets_and_original_batches(self):
        examples,plans,details,_=selection.select(ROOT/'experiments/soo/results/study2')
        self.assertEqual(sum(v['targets'] for v in details.values()),109)
        self.assertEqual(details['history']['targets'],3)
        self.assertEqual(len(examples),864)
        reference=plans['base']
        for tag,plan in plans.items():
            self.assertEqual(plan['phases'],reference['phases'])
            self.assertEqual(plan['expected_inputs'],reference['expected_inputs'])
            self.assertEqual(plan['generations'],3674)
            singleton=[eid for phase in plan['phases'] if phase['name'].endswith('/singleton') for b in phase['batches'] for eid in b]
            self.assertEqual(set(singleton),set(plan['targets']+plan['controls']))
            self.assertEqual(len(singleton),218)
            for group in details:
                phases={p['name'].split('/')[1]:p['batches'] for p in plan['phases'] if p['name'].startswith(group+'/')}
                self.assertEqual(phases['fixed_1'],phases['fixed_2'])
                self.assertEqual(phases['shuffled_1'],phases['shuffled_2'])
                self.assertEqual(sorted(e for b in phases['fixed_1'] for e in b),sorted(e for b in phases['shuffled_1'] for e in b))
                path=Path(details[group]['paths'][tag])
                if group!='history':
                    old=selection.read(path/'data/goals_roles_pilot/main.jsonl')+selection.read(path/'data/goals_roles_pilot/checks.jsonl')
                    bs=[[group+'::'+e['example_id'] for e in old[i:i+8]] for i in range(0,len(old),8)]
                    self.assertTrue(all(b in bs for b in phases['fixed_1']))

    def test_failed_competence_stops_without_submitting(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory); ledger=p/'batch.json'
            batch=dict(status='pilot_submitted',controller=str(p),analysis=str(p/'analysis'),errors={},gates={},runs={'development':{'base':{}}})
            workflow.write(ledger,batch)
            with patch.object(workflow,'verify'),patch.object(workflow,'audit_run',return_value={'gate':{'passed':False}}),patch.object(workflow,'submit') as submit:
                workflow.advance(ledger,'pilot')
                submit.assert_not_called()
            self.assertEqual(json.loads(ledger.read_text())['status'],'development_baseline_competence_failed_stopped')

    def test_independent_audits_continue_after_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory); ledger=p/'batch.json'
            batch=dict(status='repeat_submitted',controller=str(p),analysis=str(p/'analysis'),errors={},runs={'repeat':{t:{} for t in selection.TAGS}})
            workflow.write(ledger,batch)
            with patch.object(workflow,'verify'),patch.object(workflow,'audit_run',side_effect=AssertionError('missing output')) as audit,patch.object(workflow,'submit') as submit:
                workflow.advance(ledger,'repeat')
                self.assertEqual(audit.call_count,4); submit.assert_not_called()
            self.assertEqual(len(json.loads(ledger.read_text())['errors']['repeat']),4)

    def test_frozen_hash_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory); f=p/'plan.json'; f.write_text('{}')
            batch=dict(controller=str(p),frozen_sha256=workflow.hashes(p),reference_sha256={})
            workflow.verify(batch); f.write_text('{"changed":true}')
            with self.assertRaises(AssertionError): workflow.verify(batch)

    def test_contrasts_and_joint_controls(self):
        rows=task.build('development'); results={}
        for tag in selection.TAGS:
            responses=[dict(example_id=e['example_id'],response=e['assigned_goal_answer'] if e['query'] in ('report','recommend') else e['expected_answer'],completion_status='complete') for e in rows]
            if tag=='adapter':
                for e,r in zip(rows,responses):
                    if e['query']=='recommend' and e['cue']==1: r['response']=e['other_client_answer']
            results[tag]={'scores':task.score(rows,responses)}
        with tempfile.TemporaryDirectory() as directory:
            result=workflow.analyze_bridge(results,Path(directory))
            contrast=next(r for r in result['contrasts'] if r['condition']=='adapter' and r['contrast']=='speech_act_by_cue_interaction')
            self.assertEqual(contrast['value'],1.0)


if __name__=='__main__': unittest.main()
