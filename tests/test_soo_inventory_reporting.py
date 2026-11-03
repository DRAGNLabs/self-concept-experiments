"""Inventory task oracles, balanced design and fail-closed stage transitions."""
import ast
import hashlib
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from selfconcept.soo.inventory_reporting import build_dataset, score_records, summarize, baseline_gate, write_dataset
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/soo/scripts'))
from inventory_reporting_workflow import advance, audit, audit_many, extend_runner, gate_decision
from launch_goals_roles import freeze


def records(data, goal_reports=True):
    return [dict(example_id=e['example_id'], response=e['assigned_goal_answer'] if e['query']=='report' and goal_reports else e['expected_answer'],
                 completion_status='complete', generated_token_ids=[1, 2]) for e in data]


class InventoryTests(unittest.TestCase):
    def test_audit_rejects_missing_outputs_and_changed_sentinel(self):
        # Synthetic execution fixture; never used as experimental evidence.
        data=build_dataset('development');response=records(data);scores=score_records(data,response)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';out=run/'output';out.mkdir(parents=True)
            (run/'src/selfconcept').mkdir(parents=True)
            write_dataset(run/'data/goals_roles_pilot','development')
            (run/'provenance.json').write_text(json.dumps({'files_sha256':{}}))
            def write(name,value): (out/name).write_text(json.dumps(value))
            def lines(path,rows): path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            inputs=[dict(example_id=e['example_id'],rendered_prompt=e['prompt'],input_ids=[1,2]) for e in data]
            manifest=dict(task='inventory_reporting',condition='base',offset_alpha=1.0,batch_size=8,source_sha256={},artifact_sha256={},
                example_ids=[e['example_id'] for e in data],prompt_ids_sha256=hashlib.sha256(json.dumps([r['input_ids'] for r in inputs]).encode()).hexdigest(),
                batches=[dict(batch_index=i//8,example_ids=[r['example_id'] for r in inputs[i:i+8]],padded_width=2) for i in range(0,len(data),8)])
            write('manifest.json',manifest);write('intervention.json',dict(inactive_hook_check=dict(passed=True)))
            lines(out/'inputs.jsonl',inputs);lines(out/'responses.jsonl',response);lines(out/'scored.jsonl',scores)
            write('summary.json',dict(summarize(scores),baseline_gate=baseline_gate(scores)))
            repeat=dict(example_ids=manifest['example_ids'][:8],before=response[:8],primary=response[:8],after=response[:8],passed=True)
            write('sentinel_repeat.json',repeat)
            history=[dict(example_id=str(i),generated_token_ids=[i,1]) for i in range(32)]
            lines(out/'regressions.jsonl',history)
            preflight=root/'preflight/development';preflight.mkdir(parents=True);lines(preflight/'token_spans.jsonl',inputs)
            batch=dict(inference_source_sha256={},preflight=str(root/'preflight'),historical_references={'base':str(run)},
                historical_sha256={'base':hashlib.sha256((out/'regressions.jsonl').read_bytes()).hexdigest()})
            entry=dict(path=str(run),condition='base',alpha=1.0,tag='base')
            self.assertTrue(audit(entry,'development',batch,root/'audit1')[0]['gate']['passed'])
            lines(out/'responses.jsonl',response[:-1])
            with self.assertRaises(AssertionError):audit(entry,'development',batch,root/'audit2')
            lines(out/'responses.jsonl',response)
            repeat['after']=deepcopy(response[:8]);repeat['after'][0]['generated_token_ids']=[9]
            write('sentinel_repeat.json',repeat)
            with self.assertRaises(AssertionError):audit(entry,'development',batch,root/'audit3')

    def test_factorial_and_independent_numeric_oracles(self):
        datasets = {split: build_dataset(split) for split in ('development', 'reserved')}
        self.assertEqual([len(v) for v in datasets.values()], [512, 1024])
        for split, data in datasets.items():
            self.assertEqual(len({r['example_id'] for r in data}), len(data))
            self.assertEqual(set(Counter((r['query'],r['role'],r['alignment'],r['table_order'],r['column_order'],r['answer_order']) for r in data).values()), {4 if split=='development' else 8})
            for e in data:
                table = e['instance']['table']
                optimum = [r['quantity'] for r in table if all(r['payoffs'][e['role']] > s['payoffs'][e['role']] for s in table if s is not r)]
                self.assertEqual(optimum, [e['assigned_goal_answer']])
                self.assertEqual(e['expected_answer'], e['actual'] if e['query'] in ('report','fact_stock') else optimum[0])
                self.assertEqual(e['alignment']=='aligned', e['actual']==optimum[0])
                col = (1,0) if e['column_order'] else (0,1)
                self.assertIn('Quantity | '+' | '.join(e['instance']['clients'][c] for c in col), e['prompt'])
                for r in table: self.assertIn(r['quantity']+' | '+' | '.join(str(r['payoffs'][c]) for c in col), e['prompt'])
                for clauses in e['semantic_spans'].values():
                    for c in clauses: self.assertEqual(e['prompt'][c['start']:c['end']], c['text'])
        dev, reserved = datasets.values()
        self.assertFalse({tuple(r['instance']['clients']) for r in dev} & {tuple(r['instance']['clients']) for r in reserved})
        self.assertFalse({tuple(sorted(r['choices'])) for r in dev} & {tuple(sorted(r['choices'])) for r in reserved})

    def test_truth_is_not_private_correctness_and_invalid_is_not_honesty(self):
        data = build_dataset('development')
        truth = score_records(data, records(data, goal_reports=False))
        self.assertTrue(all(s['correct'] for s in truth))
        for e, s in zip(data, truth):
            if e['query']=='private' and e['alignment']=='conflict': self.assertFalse(s['matches_stock'])
        response = records(data)
        i = next(i for i,e in enumerate(data) if e['query']=='report' and e['alignment']=='conflict')
        response[i]['response'] = 'I refuse'
        scored = score_records(data, response)[i]
        self.assertFalse(scored['valid_answer']); self.assertFalse(scored['correct']); self.assertFalse(scored['false_report'])
        response[i]['response'] = data[i]['actual']; response[i]['completion_status'] = 'truncated'
        self.assertFalse(score_records(data, response)[i]['correct'])
        with self.assertRaises(ValueError): score_records(data, response+[response[0]])

    def test_competence_and_opportunity_are_separate(self):
        data = build_dataset('development')
        goal = baseline_gate(score_records(data, records(data)))
        self.assertTrue(goal['passed'])
        honest = baseline_gate(score_records(data, records(data, goal_reports=False)))
        self.assertTrue(honest['competence_passed']); self.assertFalse(honest['opportunity_passed'])
        self.assertEqual(gate_decision(dict(checks_passed=True, gate=honest)), 'baseline_opportunity_failed_stopped')
        for query in ('private', 'fact_stock', 'fact_goal'):
            response = records(data)
            for e, r in zip(data, response):
                if e['query']==query and e['role']==1 and e['column_order']==1:
                    r['response'] = next(c for c in e['choices'] if c != e['expected_answer'])
            broken = baseline_gate(score_records(data, response))
            self.assertFalse(broken['competence_passed'])
        self.assertFalse(baseline_gate([])['passed'])

    def test_dataset_writer_cannot_overwrite_changed_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp); write_dataset(path,'development'); write_dataset(path,'development')
            (path/'main.jsonl').write_text('changed')
            with self.assertRaises(FileExistsError): write_dataset(path,'development')

    def test_failed_pilot_never_submits_interventions_or_reserved(self):
        for competent, opportunity in ((False,True),(True,False)):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); controller=root/'controller';controller.mkdir()
                batch=dict(status='pilot_submitted',controller=str(controller),analysis=str(root/'analysis'),
                    controller_sha256={},source_ledger_sha256={},preflight_sha256={},
                    runs={'development':{'base':{}}},audits={},errors={})
                path=root/'batch.json';path.write_text(json.dumps(batch))
                result=dict(checks_passed=True,gate=dict(competence_passed=competent,opportunity_passed=opportunity))
                with patch('inventory_reporting_workflow.audit_many',return_value=({'base':(result,)},{})), patch('inventory_reporting_workflow.submit') as submit:
                    advance(path,'pilot');submit.assert_not_called()
                self.assertTrue(json.loads(path.read_text())['status'].endswith('failed_stopped'))

    def test_independent_audits_continue_after_failure(self):
        with patch('inventory_reporting_workflow.audit',side_effect=RuntimeError('missing output')) as call:
            results,errors=audit_many({'base':{},'adapter':{}},'development',{},Path('unused'))
        self.assertEqual(call.call_count,2);self.assertEqual(set(errors),{'base','adapter'});self.assertFalse(results)

    def test_source_extension_preserves_generation_and_regressions(self):
        path=Path(__file__).resolve().parents[1]/'src/selfconcept/soo/run_goals_roles.py'
        current=path.read_text()
        old=current.replace('choices=("goals_roles", "explicit_payoff", "inventory_reporting"),','choices=("goals_roles", "explicit_payoff"),')
        old=old.replace('if args.task in ("explicit_payoff", "inventory_reporting") and (args.resume or args.repeat_plan):','if args.task == "explicit_payoff" and (args.resume or args.repeat_plan):')
        old=old.replace('payoff = getattr(args, "task", "goals_roles") in ("explicit_payoff", "inventory_reporting")\n    if getattr(args, "task", "goals_roles") == "inventory_reporting":\n        from .inventory_reporting import score_records as task_score, summarize as task_summarize, baseline_gate as task_gate\n    elif payoff:', 'payoff = getattr(args, "task", "goals_roles") == "explicit_payoff"\n    if payoff:')
        old=old.replace('manifest["task"] = args.task','manifest["task"] = "explicit_payoff"')
        self.assertEqual(extend_runner(old), current)
        for function in ('generate','completion','run_regressions','source_hashes'):
            def subtree(s):return ast.dump(next(n for n in ast.walk(ast.parse(s)) if isinstance(n,ast.FunctionDef) and n.name==function))
            self.assertEqual(subtree(old),subtree(current))

    def test_launch_scope_rejects_new_doses_and_missing_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);(path/'manifest.json').write_text(json.dumps(dict(task='inventory_reporting',dataset_id='inventory_reporting1_development',sha256={})))
            for condition,alpha,source,protocol in [('offset',.625,path,'inventory_reporting1'),('band',1,path,'inventory_reporting1'),('offset',.5,None,'inventory_reporting1'),('base',1,path,None)]:
                with self.assertRaises(ValueError):freeze(condition,data_dir=path,offset_alpha=alpha,source_dir=source,followup_protocol=protocol)


if __name__=='__main__':unittest.main()
