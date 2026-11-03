"""Independent numerical-key, release and endpoint checks without model loading."""
from collections import Counter
from copy import deepcopy
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from selfconcept.soo.explicit_payoff import build_dataset as original_dataset,score_records,baseline_gate
from selfconcept.soo.payoff_preferences import build_dataset
from selfconcept.soo.run_goals_roles import main as runner_main,parser
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/soo/scripts'))
from payoff_preferences_followup import endpoint_bridge,feasibility,release_allowed,run_stage


def source():
    rows=original_dataset('development')
    return [r for suite in ('main','checks') for r in rows if r['suite']==suite]


def oracle(rows):
    return [{'example_id':r['example_id'],'response':r['expected_answer'],'completion_status':'complete'} for r in rows]


class PreferenceTests(unittest.TestCase):
    def test_feasible_profiles_and_independent_goals_display_and_clauses(self):
        old=source();by_id={r['example_id']:r for r in old};rows=build_dataset(old)
        self.assertEqual(len(rows),2592);self.assertEqual(len({r['example_id'] for r in rows}),2592)
        self.assertEqual(Counter(r['profile'] for r in rows),{p:432 for p in range(6)})
        combinations=set()
        for e in rows:
            original=by_id[e['canonical_example_id']];table=e['instance']['table']
            best=[max(table,key=lambda r:r['payoffs'][c])['option'] for c in (0,1)]
            self.assertNotEqual(*best)
            expected=best[int(e['query'][-1])] if e['query'].startswith('fact') else best[e['role']]
            self.assertEqual(e['expected_answer'],expected);self.assertEqual(expected,original['expected_answer'])
            answers=[]
            for rule,function in [('total',sum),('minimum',min),('largest_single',max)]:
                values=[function(r['payoffs']) for r in table];winner=table[values.index(max(values))]['option']
                self.assertEqual(values.count(max(values)),1)
                self.assertEqual(e['rule_answers'][rule],winner);answers.append(best.index(winner))
            combinations.add(tuple(answers))
            for r in table:self.assertIn(r['option']+' | '+' | '.join(map(str,r['payoffs'])),e['prompt'])
            for key,spans in e['semantic_spans'].items():
                for span in spans:self.assertEqual(e['prompt'][span['start']:span['end']],span['text'])
                if key!='payoffs':self.assertEqual([s['text'] for s in spans],[s['text'] for s in original['semantic_spans'][key]])
        self.assertEqual(combinations,{(0,0,0),(0,0,1),(0,1,0),(1,0,1),(1,1,0),(1,1,1)})
        for i in range(0,len(rows),8):self.assertEqual(len({r['profile'] for r in rows[i:i+8]}),1)

    def test_profile_gate_catches_local_failure_without_using_messages(self):
        rows=build_dataset(source());records=oracle(rows)
        for e,r in zip(rows,records):
            if e['query']=='message':r['response']=e['other_role_answer']
        scores=score_records(rows,records);gate=feasibility({'gate':baseline_gate(scores)},rows,scores)
        self.assertEqual(len(gate['checks']),120);self.assertTrue(gate['passed'])
        changed=0
        for e,r in zip(rows,records):
            if e['profile']==2 and e['query']=='private' and changed<13:
                r['response']=e['other_role_answer'];changed+=1
        scores=score_records(rows,records)
        self.assertTrue(all(c['passed'] for c in baseline_gate(scores)['checks'] if c['query']!='message'))
        self.assertFalse(feasibility({'gate':baseline_gate(scores)},rows,scores)['passed'])

    def test_endpoint_requires_tokens_and_historical_outputs_not_only_answers(self):
        with tempfile.TemporaryDirectory() as temp:
            a=Path(temp)/'a';b=Path(temp)/'b'
            for p in (a,b):
                (p/'output').mkdir(parents=True)
                for name in ('responses.jsonl','regressions.jsonl'):
                    (p/'output'/name).write_text(json.dumps({'response':'cedar','generated_token_ids':[3,1]})+'\n')
            self.assertTrue(endpoint_bridge(a,b)['passed'])
            (a/'output/regressions.jsonl').write_text(json.dumps({'response':'cedar','generated_token_ids':[4,1]})+'\n')
            self.assertFalse(endpoint_bridge(a,b)['passed'])

    def test_any_failed_requirement_stops_release(self):
        self.assertTrue(release_allowed({'all_checks_passed':True},{'passed':True},{'repeat_passed':True,'regressions_passed':True}))
        for failure in range(4):
            vals=[True]*4;vals[failure]=False
            self.assertFalse(release_allowed({'all_checks_passed':vals[0]},{'passed':vals[1]},
                             {'repeat_passed':vals[2],'regressions_passed':vals[3]}))

    def test_controller_does_not_submit_on_endpoint_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);old=root/'old';data=old/'data/goals_roles_pilot';data.mkdir(parents=True)
            for suite in ('main','checks'):(data/(suite+'.jsonl')).write_text('')
            source_ledger=root/'source.json';source_ledger.write_text('{}')
            from analyze_explicit_payoff import digest
            batch_path=root/'batch.json'
            b={'controller':str(root),'controller_sha256':{},'source_ledger':str(source_ledger),
               'source_ledger_sha256':digest(source_ledger),'status':'dose_and_numeric_baseline_submitted',
               'historical':{'base':str(old)},'dose_runs':[],'numeric_runs':[{'path':'unused'}],
               'dose_analysis':str(root/'dose'),'numeric_analysis':str(root/'numeric')}
            (root/'numeric/baseline_audit').mkdir(parents=True);batch_path.write_text(json.dumps(b))
            with patch('payoff_preferences_followup.dose_analysis',return_value={'all_checks_passed':False}),\
                 patch('payoff_preferences_followup.numerical_audit',return_value=({'repeat_passed':True,'regressions_passed':True},[],[],[],{})),\
                 patch('payoff_preferences_followup.feasibility',return_value={'passed':True}),\
                 patch('payoff_preferences_followup.submit') as submit:
                run_stage(batch_path,'check');submit.assert_not_called()
            self.assertEqual(json.loads(batch_path.read_text())['status'],'checks_failed_stopped')
            self.assertEqual(source_ledger.read_text(),'{}')

    def test_alpha_default_and_invalid_values_rejected_before_loading(self):
        args=['--revision','fake','--data','fake','--out','fake','--provenance','fake']
        self.assertEqual(parser().parse_args(args).offset_alpha,1.0)
        for extra in (['--offset-alpha','nan'],['--offset-alpha','-1'],['--offset-alpha','.5']):
            with patch.object(sys,'argv',['runner',*args,*extra]),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as error:
                runner_main()
            self.assertEqual(error.exception.code,2)


if __name__=='__main__':unittest.main()
