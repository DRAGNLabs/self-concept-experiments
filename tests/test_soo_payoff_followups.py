"""Check post hoc gate scope, exact matched diagnostics and stopping behavior."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from selfconcept.soo.explicit_payoff import build_dataset as original_dataset,score_records,baseline_gate,summarize
from selfconcept.soo.payoff_diagnostic import build_dataset,transform

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/soo/scripts'))
from analyze_payoff_followups import private_gate,private_baseline_decision,diagnostic_analysis
from continue_payoff_followups import main as continue_main


def original():
    rows=original_dataset('development')
    return [r for suite in ('main','checks') for r in rows if r['suite']==suite]


def oracle(rows):
    return [{'example_id':e['example_id'],'response':e['expected_answer'],'completion_status':'complete',
             'generated_token_ids':[e['choices'].index(e['expected_answer'])+10,1]} for e in rows]


def audit(rows,records):
    return {'gate':baseline_gate(score_records(rows,records)),'repeat_passed':True,'regressions_passed':True}

class FollowupTests(unittest.TestCase):
    def test_new_gate_retains_failed_joint_gate_and_rejects_private_fact_failure(self):
        rows=original();responses=oracle(rows)
        for e,r in zip(rows,responses):
            if e['query']=='message':r['response']=e['other_role_answer']
        a=audit(rows,responses)
        self.assertFalse(a['gate']['passed']);self.assertTrue(private_gate(a)['passed'])
        self.assertEqual(private_baseline_decision(a),'pass')
        self.assertFalse(private_gate(a)['historical_combined_gate_passed'])
        for q in ('private','fact0','fact1'):
            failed=deepcopy(responses)
            for e,r in zip(rows,failed):
                if e['query']==q and e['table_order']==0 and e['answer_order']==0:r['response']=e['dominated_answer']
            self.assertNotEqual(private_baseline_decision(audit(rows,failed)),'pass')
        self.assertFalse(private_gate({'gate':{'checks':[],'passed':False}})['passed'])
        a['repeat_passed']=False
        self.assertNotEqual(private_baseline_decision(a),'pass')

    def test_factorial_identity_batches_and_independent_column_values(self):
        source=original();rows=build_dataset(source)
        self.assertEqual(len(rows),1728)
        old_batches={tuple(e['example_id'] for e in source[i:i+8]) for i in range(0,432,8)}
        for i in range(0,len(rows),8):
            batch=rows[i:i+8]
            self.assertEqual(len({(r['column_swap'],r['name_swap']) for r in batch}),1)
            self.assertIn(tuple(r['canonical_example_id'] for r in batch),old_batches)
        source_by_id={e['example_id']:e for e in source}
        for e in rows:
            old=source_by_id[e['canonical_example_id']]
            self.assertEqual(e['expected_answer'],old['expected_answer'])
            self.assertEqual(e['instance']['table'],old['instance']['table'])
            names=old['instance']['clients'][::-1] if e['name_swap'] else old['instance']['clients']
            self.assertEqual(e['instance']['clients'],names)
            if e['column_swap']==e['name_swap']==0:self.assertEqual(e['prompt'],old['prompt'])
            order=(1,0) if e['column_swap'] else (0,1)
            self.assertIn('Option | '+' | '.join(names[c] for c in order),e['prompt'])
            for t in e['instance']['table']:
                self.assertIn(t['option']+' | '+' | '.join(str(t['payoffs'][c]) for c in order),e['prompt'])
            for clauses in e['semantic_spans'].values():
                for c in clauses:self.assertEqual(e['prompt'][c['start']:c['end']],c['text'])

    def test_diagnostic_analysis_known_column_name_and_action_effects(self):
        self.check_diagnostic_analysis('base')

    def test_intervention_diagnostic_without_baseline_gate_or_inactive_hook(self):
        for condition in ('adapter','offset'):
            with self.subTest(condition=condition):
                self.check_diagnostic_analysis(condition)

    def check_diagnostic_analysis(self,condition):
        source=original();old_records=oracle(source);old_scores=score_records(source,old_records)
        data=build_dataset(source);responses=oracle(data)
        for e,r in zip(data,responses):
            if e['name_swap']==1 and e['column_swap']==0 and e['role']==0 and e['alignment']=='opposed' and e['query']=='message':
                r['response']=e['other_role_answer'];r['generated_token_ids']=[99,1]
        scores=score_records(data,responses)
        old_inputs=[{'example_id':e['example_id'],'input_ids':[i+100,2], 'rendered_prompt':e['prompt']} for i,e in enumerate(source)]
        old_input_by_id={r['example_id']:r for r in old_inputs}
        inputs=[{'example_id':e['example_id'],'input_ids':old_input_by_id[e['canonical_example_id']]['input_ids'], 'rendered_prompt':e['prompt']} for e in data]
        def write(path,value):path.write_text(json.dumps(value))
        def lines(path,rows):path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run=root/'run';out=run/'output';out.mkdir(parents=True)
            historical=root/'old';(historical/'output').mkdir(parents=True)
            lines(historical/'output/responses.jsonl',old_records)
            dataset=run/'data/goals_roles_pilot';dataset.mkdir(parents=True)
            for suite in ('main','checks'):lines(dataset/(suite+'.jsonl'),[e for e in data if e['suite']==suite])
            for name,rows in (('responses',responses),('inputs',inputs),('scored',scores)):lines(out/(name+'.jsonl'),rows)
            manifest={'condition':condition,'batch_size':8,'example_ids':[e['example_id'] for e in data],'artifact_sha256':{},
                      'prompt_ids_sha256':hashlib.sha256(json.dumps([e['input_ids'] for e in inputs]).encode()).hexdigest(),
                      'batches':[{'batch_index':i//8,'example_ids':[e['example_id'] for e in data[i:i+8]],'padded_width':2} for i in range(0,1728,8)]}
            write(out/'manifest.json',manifest)
            summary=summarize(scores)
            if condition=='base':summary['baseline_gate']=baseline_gate(scores)
            write(out/'summary.json',summary)
            write(out/'intervention.json',{'inactive_hook_check':{'passed':True}} if condition=='base' else {})
            write(out/'sentinel_repeat.json',{'example_ids':manifest['example_ids'][:8],'primary':responses[:8],'before':responses[:8],'after':responses[:8],'passed':True})
            lines(out/'regressions.jsonl',[{'scenario':q,'label':'deceptive' if q=='main' and condition=='base' else 'honest','completion_status':'complete'} for q in ('main','perspectives') for _ in range(16)])
            with patch('analyze_payoff_followups.verify_run'),patch('analyze_payoff_followups.audit_run',return_value=({'condition':condition},source,old_inputs,old_scores,{})):
                result=diagnostic_analysis(run,historical,root/'analysis',condition)
            self.assertTrue(result['regressions_passed'])
            self.assertEqual(result['error_count'],27)
            self.assertEqual(result['historical_answer_changes'],0)
            self.assertEqual(result['historical_token_changes'],0)
            self.assertEqual(result['paired_effects']['column_swap']['gained_correct'],27)
            self.assertEqual(result['paired_effects']['name_swap']['lost_correct'],27)
            self.assertEqual(result['paired_effects']['private_to_message']['lost_correct'],27)
            self.assertEqual(result['paired_effects']['column_swap']['n'],864)
            self.assertEqual(result['paired_effects']['name_swap']['n'],864)
            self.assertEqual(result['paired_effects']['private_to_message']['n'],432)

    def test_failed_development_never_submits_validation(self):
        a=audit(original(),oracle(original()))
        result={'audits':{'base':a},'primary_adapter_minus_offset':0,'private_by_instance_role':[],
                'progression':{'passed':False},'all_repeats_passed':True,'all_regressions_passed':True}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);old=root/'old.json';old.write_text('{}');out=root/'analysis';out.mkdir()
            batch_path=root/'batch.json'
            batch={'controller':d,'controller_sha256':{},'original_batch':str(old),
                   'original_batch_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),
                   'private_status':'development_six_interventions_submitted','diagnostic_status':'completed_and_analyzed',
                   'runs':{'development':[{'tag':'base','path':d}]},'analysis':{'development':str(out)}}
            batch_path.write_text(json.dumps(batch))
            with patch('continue_payoff_followups.analyze',return_value=result),patch('continue_payoff_followups.submit') as submit:
                continue_main(batch_path,'development_paired');submit.assert_not_called()
            self.assertTrue(json.loads(batch_path.read_text())['private_status'].endswith('_stopped'))
            self.assertEqual(old.read_text(),'{}')

if __name__=='__main__':unittest.main()
