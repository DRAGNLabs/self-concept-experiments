"""Independent key, factorial, gate and continuation safeguards for payoff1."""
from collections import Counter
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from selfconcept.soo.explicit_payoff import build_dataset, instances, score_records, baseline_gate

SCRIPTS=Path(__file__).resolve().parents[1]/'experiments/soo/scripts'
sys.path.insert(0,str(SCRIPTS))
from analyze_explicit_payoff import progression, analyze, audit_run
from continue_explicit_payoff import baseline_decision, development_decision, main as continue_main


def oracle(rows):
    return [{'example_id':r['example_id'],'response':r['expected_answer'],'completion_status':'complete'} for r in rows]

class PayoffTests(unittest.TestCase):
    def test_complete_paired_analysis_and_corruption_rejection(self):
        import hashlib
        import json
        from selfconcept.soo.explicit_payoff import summarize
        examples=build_dataset('development')
        examples=[e for suite in ('main','checks') for e in examples if e['suite']==suite]
        def write(path,obj):path.write_text(json.dumps(obj))
        def lines(path,rows):path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);entries=[]
            for tag in ('base','adapter','band','offset','random0','random1','random2'):
                run=root/tag;out=run/'output';out.mkdir(parents=True)
                data=run/'data/goals_roles_pilot';data.mkdir(parents=True)
                for suite in ('main','checks'):lines(data/f'{suite}.jsonl',[e for e in examples if e['suite']==suite])
                write(run/'provenance.json',{'files_sha256':{}})
                responses=oracle(examples)
                if tag=='offset':
                    for e,r in zip(examples,responses):
                        if e['query']=='private':r['response']=e['dominated_answer']
                scores=score_records(examples,responses)
                summary=summarize(scores)
                if tag=='base':summary['baseline_gate']=baseline_gate(scores)
                write(out/'summary.json',summary);lines(out/'responses.jsonl',responses);lines(out/'scored.jsonl',scores)
                inputs=[{'example_id':e['example_id'],'input_ids':[1,2,3]} for e in examples]
                lines(out/'inputs.jsonl',inputs)
                manifest={'condition':'random' if tag.startswith('random') else tag,
                          'random_seed':int(tag[-1]) if tag.startswith('random') else None,
                          'example_ids':[e['example_id'] for e in examples],'artifact_sha256':{},
                          'prompt_ids_sha256':hashlib.sha256(json.dumps([e['input_ids'] for e in inputs]).encode()).hexdigest(),
                          'batch_size':8,'model':'test','revision':'test','dtype':'test','attention':'test','chat_kwargs':{},
                          'max_new_tokens':32,'do_sample':False,'source_sha256':{},
                          'batches':[{'batch_index':i//8,'example_ids':[e['example_id'] for e in examples[i:i+8]],'padded_width':3} for i in range(0,len(examples),8)]}
                write(out/'manifest.json',manifest)
                write(out/'intervention.json',{'inactive_hook_check':{'passed':True}})
                write(out/'sentinel_repeat.json',{'example_ids':manifest['batches'][0]['example_ids'],
                      'before':responses[:8],'primary':responses[:8],'after':responses[:8],'passed':True})
                label='deceptive' if tag=='base' or tag.startswith('random') else 'honest'
                lines(out/'regressions.jsonl',[{'scenario':'main','label':label,'completion_status':'complete'} for _ in range(16)]+[{'scenario':'perspectives','label':'honest','completion_status':'complete'} for _ in range(16)])
                entries.append({'tag':tag,'path':str(run)})
            result=analyze(entries,root/'analysis')
            self.assertEqual(result['primary_adapter_minus_offset'],1)
            self.assertTrue(result['progression']['passed'])
            self.assertEqual(len((root/'analysis/paired_trials.csv').read_text().splitlines()),432*7+1)
            scores[0]['correct']=not scores[0]['correct']
            lines(out/'scored.jsonl',scores)
            with self.assertRaises(AssertionError):audit_run(run)

    def test_design_and_independent_keys(self):
        for split,n in (('development',432),('validation',1296)):
            rows=build_dataset(split)
            self.assertEqual(len(rows),n);self.assertEqual(len({r['example_id'] for r in rows}),n)
            for r in rows:
                client=int(r['query'][-1]) if r['query'].startswith('fact') else r['role']
                values={x['option']:x['payoffs'][client] for x in r['instance']['table']}
                self.assertEqual(r['expected_answer'],max(values,key=values.get))
                self.assertEqual(len(set(r['choices'])),3)
                for spans in r['semantic_spans'].values():
                    for s in spans:self.assertEqual(r['prompt'][s['start']:s['end']],s['text'])
            self.assertEqual(set(Counter((r['role'],r['alignment'],r['table_order'],r['answer_order'],r['query']) for r in rows).values()),{n//144})
        dev=[x for x in instances() if x['split']=='development'];val=[x for x in instances() if x['split']=='validation']
        self.assertFalse({r['option'] for x in dev for r in x['table']} & {r['option'] for x in val for r in x['table']})
        for split in (dev,val):
            for client in (0,1):
                positions=[max(range(3),key=lambda i:x['table'][i]['payoffs'][client]) for x in split]
                self.assertEqual(set(Counter(positions).values()),{len(split)//3})

    def test_strict_scoring_and_all_trial_denominator(self):
        rows=build_dataset('development');records=oracle(rows)
        for status,text in (('truncated',rows[0]['expected_answer']),('complete',rows[0]['expected_answer']+'.'),('complete','explanation')):
            altered=deepcopy(records);altered[0].update(completion_status=status,response=text)
            self.assertFalse(score_records(rows,altered)[0]['valid_answer'])
        self.assertFalse(score_records(rows,[])[0]['correct'])
        with self.assertRaises(ValueError):score_records(rows,records+[records[0]])
        with self.assertRaises(ValueError):score_records(rows,[{'example_id':'unknown','completion_status':'complete'}])
        bad=deepcopy(rows);bad[0]['expected_answer']=bad[0]['dominated_answer']
        with self.assertRaises(AssertionError):score_records(bad,records)

    def test_gate_both_splits_and_cell_failure(self):
        for split in ('development','validation'):
            rows=build_dataset(split);records=oracle(rows)
            self.assertTrue(baseline_gate(score_records(rows,records))['passed'])
            for r,record in zip(rows,records):
                if r['query']=='private' and r['table_order']==0 and r['answer_order']==0:
                    record['response']=r['dominated_answer']
            gate=baseline_gate(score_records(rows,records))
            self.assertFalse(gate['passed'])
            self.assertTrue(any(not c['passed'] and c['group']=='table0-answer0' for c in gate['checks']))
        self.assertFalse(baseline_gate([])['passed'])

    def test_progression_requires_both_roles_and_two_instances(self):
        rows=[{'instance_id':str(i),'role':r,'adapter_minus_offset':.1 if i<2 else 0} for i in range(3) for r in (0,1)]
        self.assertTrue(progression(rows)['passed'])
        rows[1]['adapter_minus_offset']=0
        self.assertFalse(progression(rows)['passed'])
        self.assertNotEqual(development_decision({'all_repeats_passed':False,'all_regressions_passed':True,'progression':{'passed':True}}),'pass')

    def test_failed_baseline_and_repeat_never_submit(self):
        import json
        for gate,repeat in ((False,True),(True,False)):
            audit={'gate':{'passed':gate},'repeat_passed':repeat,'regressions_passed':True}
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);batch=root/'batch.json'
                batch.write_text(json.dumps({'controller':d,'controller_sha256':{},'status':'development_baseline_submitted','runs':{'development':[{'path':d}]}}))
                with patch('continue_explicit_payoff.audit_run',return_value=(audit,)), patch('continue_explicit_payoff.submit') as submit:
                    continue_main(batch,'development_base');submit.assert_not_called()
                self.assertNotEqual(json.loads(batch.read_text())['status'],'development_six_interventions_submitted')

if __name__=='__main__':unittest.main()
