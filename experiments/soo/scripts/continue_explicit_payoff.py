"""Advance only frozen explicit-payoff stages; failures and discrepancies stop."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
from analyze_explicit_payoff import analyze, audit_run, digest, verify_run


def submit(script, receipt_path, dependency=None):
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())
        assert receipt['script_sha256']==digest(script)
        return receipt
    command=['sbatch','--parsable']
    if dependency:
        command+=['--dependency=afterany:'+':'.join(dependency)]
    job=subprocess.check_output([*command,str(script)],text=True).strip().split(';')[0]
    receipt={'job_id':job,'script_sha256':digest(script),'dependency_afterany':dependency,
             'submitted_utc':datetime.now(timezone.utc).isoformat()}
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def baseline_decision(audit):
    if not audit['gate']['passed']:
        return 'baseline_failed_gate_stopped'
    if not audit['repeat_passed']:
        return 'repeat_discrepancy_requires_inspection'
    if not audit['regressions_passed']:
        return 'regression_discrepancy_requires_inspection'
    return 'pass'


def development_decision(result):
    if not result['all_repeats_passed']:
        return 'repeat_discrepancy_requires_inspection'
    if not result['all_regressions_passed']:
        return 'regression_discrepancy_requires_inspection'
    if not result['progression']['passed']:
        return 'primary_contrast_did_not_transfer_stopped'
    return 'pass'


def main(batch_path, stage):
    with batch_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        batch=json.loads(batch_path.read_text())
        controller=Path(batch['controller'])
        for name,expected in batch['controller_sha256'].items():
            assert digest(controller/name)==expected, 'controller changed'
        def save():
            batch['updated_utc']=datetime.now(timezone.utc).isoformat()
            temp=batch_path.with_suffix('.tmp');temp.write_text(json.dumps(batch,indent=2)+'\n');temp.replace(batch_path)
        def run_submit(entry):
            path=Path(entry['path']);verify_run(path)
            assert digest(path/'run.sh')==entry['script_sha256']
            entry.update(submit(path/'run.sh',path/'submission.json'),status='submitted');save()
            return entry['job_id']
        def continue_after(next_stage,jobs):
            receipt=submit(controller/f'{next_stage}.sh',controller/f'{next_stage}_submission.json',jobs)
            batch.setdefault('controllers',{})[next_stage]=receipt;save()
        try:
            if stage=='start':
                assert batch['status']=='staged_not_submitted', 'already started'
                # Verify all fourteen runs before the first output.
                for entries in batch['runs'].values():
                    for entry in entries:
                        verify_run(Path(entry['path']))
                        assert digest(Path(entry['path'])/'run.sh')==entry['script_sha256']
                job=run_submit(batch['runs']['development'][0])
                batch['status']='development_baseline_submitted';save()
                continue_after('development_base',[job]);return
            split,phase=stage.rsplit('_',1)
            entries=batch['runs'][split]
            if phase=='base':
                audit=audit_run(Path(entries[0]['path']))[0]
                entries[0]['status']='completed';batch.setdefault('baseline_audits',{})[split]=audit
                decision=baseline_decision(audit)
                if decision!='pass':
                    batch['status']=split+'_'+decision;save();return
                jobs=[run_submit(entry) for entry in entries[1:]]
                batch['status']=split+'_six_interventions_submitted';save()
                continue_after(split+'_paired',jobs)
            elif phase=='paired':
                result=analyze(entries,Path(batch['analysis'][split]))
                for entry in entries:
                    entry['status']='completed'
                batch.setdefault('analysis_results',{})[split]={'primary_adapter_minus_offset':result['primary_adapter_minus_offset'],'progression':result['progression']}
                if split=='development':
                    decision=development_decision(result)
                    if decision!='pass':
                        batch['status']='development_'+decision;save();return
                    job=run_submit(batch['runs']['validation'][0])
                    batch['status']='development_passed_validation_baseline_submitted';save()
                    continue_after('validation_base',[job])
                else:
                    batch['status']='completed_and_analyzed' if result['all_repeats_passed'] and result['all_regressions_passed'] else 'validation_completed_discrepancies_require_inspection'
                    save()
            else:
                raise ValueError(stage)
        except Exception as exc:
            batch['status']='execution_error_requires_inspection'
            batch['error']=f'{type(exc).__name__}: {exc}';save();raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',type=Path,required=True)
    p.add_argument('--stage',choices=('start','development_base','development_paired','validation_base','validation_paired'),required=True)
    args=p.parse_args();main(args.batch.resolve(),args.stage)
