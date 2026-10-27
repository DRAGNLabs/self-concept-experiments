"""New authorized follow-up ledger; never resume the failed combined controller."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
from analyze_explicit_payoff import analyze,audit_run,digest,verify_run,write_json
from analyze_payoff_followups import private_gate,private_baseline_decision,diagnostic_analysis
from continue_explicit_payoff import submit,development_decision


def main(batch_path,stage):
    with batch_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        batch=json.loads(batch_path.read_text());controller=Path(batch['controller'])
        for name,expected in batch['controller_sha256'].items():
            assert digest(controller/name)==expected,f'controller changed: {name}'
        assert digest(Path(batch['original_batch']))==batch['original_batch_sha256'],'original stopped ledger changed'
        def save():
            batch['updated_utc']=datetime.now(timezone.utc).isoformat()
            terminal=lambda value: value=='completed_and_analyzed' or value.endswith('_stopped') or 'inspection' in value
            batch['status']='finished' if terminal(batch['private_status']) and terminal(batch['diagnostic_status']) else 'active'
            tmp=batch_path.with_suffix('.tmp');tmp.write_text(json.dumps(batch,indent=2)+'\n');tmp.replace(batch_path)
        def run_submit(entry,label):
            path=Path(entry['path']);verify_run(path)
            assert digest(path/'run.sh')==entry['script_sha256']
            receipt=submit(path/'run.sh',controller/'receipts'/(label+'.json'))
            entry.update(receipt,status='submitted');save();return receipt['job_id']
        def next_stage(name,jobs):
            receipt=submit(controller/(name+'.sh'),controller/'receipts'/(name+'.json'),jobs)
            batch.setdefault('controllers',{})[name]=receipt;save()
        def base_audit(split):
            entry=batch['runs'][split][0]
            audit=audit_run(Path(entry['path']),Path(batch['analysis'][split])/'baseline_audit')[0]
            gate=private_gate(audit)
            write_json(Path(batch['analysis'][split])/'baseline_audit/private_gate.json',gate)
            batch.setdefault('baseline_audits',{})[split]={'private_gate':gate,'combined_gate_passed':audit['gate']['passed'],
                                                        'repeat_passed':audit['repeat_passed'],'regressions_passed':audit['regressions_passed']}
            entry['status']='reused_completed' if split=='development' else 'completed'
            return private_baseline_decision(audit)
        try:
            if stage=='start':
                assert batch['private_status']==batch['diagnostic_status']=='staged_not_submitted','already started'
                for split,entries in batch['runs'].items():
                    for entry in entries:
                        path=Path(entry['path']);verify_run(path)
                        assert digest(path/'run.sh')==entry['script_sha256']
                        if not (split=='development' and entry['tag']=='base'):
                            assert not (path/'output').exists(),f'outputs already exist: {path}'
                diagnostic=batch['diagnostic'];verify_run(Path(diagnostic['path']))
                assert not (Path(diagnostic['path'])/'output').exists()
                # Independent branch: does not tune, gate or select the private comparison.
                job=run_submit(diagnostic,'diagnostic_run')
                batch['diagnostic_status']='submitted';save();next_stage('diagnostic',[job])
                decision=base_audit('development')
                if decision!='pass':
                    batch['private_status']='development_'+decision;save();return
                jobs=[run_submit(e,'development_'+e['tag']) for e in batch['runs']['development'][1:]]
                batch['private_status']='development_six_interventions_submitted';save()
                next_stage('development_paired',jobs)
            elif stage=='diagnostic':
                result=diagnostic_analysis(Path(batch['diagnostic']['path']),Path(batch['runs']['development'][0]['path']),Path(batch['diagnostic_analysis']))
                batch['diagnostic_result']=result;batch['diagnostic']['status']='completed'
                clean=result['sentinel_repeat_passed'] and result['regressions_passed'] and result['historical_token_changes']==0
                batch['diagnostic_status']='completed_and_analyzed' if clean else 'completed_discrepancies_require_inspection'
                save()
            elif stage=='validation_base':
                decision=base_audit('validation')
                if decision!='pass':
                    batch['private_status']='validation_'+decision;save();return
                jobs=[run_submit(e,'validation_'+e['tag']) for e in batch['runs']['validation'][1:]]
                batch['private_status']='validation_six_interventions_submitted';save()
                next_stage('validation_paired',jobs)
            elif stage in ('development_paired','validation_paired'):
                split=stage.split('_')[0]
                result=analyze(batch['runs'][split],Path(batch['analysis'][split]))
                result['protocol']='post_hoc_private_goal_followup1'
                result['baseline_private_gate']=private_gate(result['audits']['base'])
                assert result['baseline_private_gate']['passed']
                result['recommendations_status']='exploratory; original joint protocol failed'
                write_json(Path(batch['analysis'][split])/'analysis.json',result)
                for e in batch['runs'][split]:e['status']='completed'
                batch.setdefault('private_results',{})[split]={'primary_adapter_minus_offset':result['primary_adapter_minus_offset'],
                    'private_by_instance_role':result['private_by_instance_role'],'progression':result['progression']}
                if split=='development':
                    decision=development_decision(result)
                    if decision!='pass':
                        batch['private_status']='development_'+decision;save();return
                    job=run_submit(batch['runs']['validation'][0],'validation_base_run')
                    batch['private_status']='validation_baseline_submitted';save();next_stage('validation_base',[job])
                else:
                    batch['private_status']='completed_and_analyzed' if result['all_repeats_passed'] and result['all_regressions_passed'] else 'validation_discrepancies_require_inspection'
                    save()
            else:raise ValueError(stage)
        except Exception as exc:
            batch['diagnostic_status' if stage=='diagnostic' else 'private_status']='execution_error_requires_inspection'
            batch.setdefault('errors',{})[stage]=f'{type(exc).__name__}: {exc}';save();raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',type=Path,required=True)
    p.add_argument('--stage',choices=('start','diagnostic','development_paired','validation_base','validation_paired'),required=True)
    a=p.parse_args();main(a.batch.resolve(),a.stage)
