"""Advance one frozen repair stage after successful audits; never waive a gate."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gate_passed(audit, expected_n):
    return (audit.get('condition') == 'base' and audit.get('n') == expected_n
            and audit.get('all_gates_passed') is True
            and audit.get('global_gate', {}).get('passed') is True
            and len(audit.get('cells', [])) == 8
            and all(c.get('gate', {}).get('passed') is True for c in audit['cells']))


def verify_run(run, script_hash):
    assert digest(run/'run.sh') == script_hash, 'submission script changed'
    provenance = json.loads((run/'provenance.json').read_text())
    for name, expected in provenance['files_sha256'].items():
        assert digest(run/name) == expected, f'frozen file changed: {run/name}'


def submit(script, receipt_path, dependency=None):
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        assert receipt['script_sha256'] == digest(script)
        return receipt
    args = ['sbatch', '--parsable']
    if dependency:
        args += ['--dependency=afterany:' + ':'.join(dependency)]
    job = subprocess.check_output([*args, str(script)], text=True).strip().split(';')[0]
    receipt = {'job_id': job, 'submitted_utc': datetime.now(timezone.utc).isoformat(),
               'script_sha256': digest(script), 'dependency_afterany': dependency}
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def main(batch_path, stage):
    with batch_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        batch = json.loads(batch_path.read_text())
        def save():
            batch['updated_utc'] = datetime.now(timezone.utc).isoformat()
            temporary = batch_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(batch, indent=2)+'\n')
            temporary.replace(batch_path)
        def controller(next_stage, jobs):
            directory = Path(batch['controller'])
            receipt = submit(directory/f'{next_stage}.sh', directory/f'{next_stage}_submission.json', jobs)
            batch.setdefault('controllers', {})[next_stage] = receipt
            save()
        try:
            if stage in ('pilot', 'baseline'):
                run = Path(batch[stage])
                verify_run(run, batch['script_sha256'][stage])
                assert (run/'output/summary.json').exists(), f'{stage} did not finish'
                out = run/'crossed_audit'
                if not (out/'audit.json').exists():
                    subprocess.run([sys.executable, str(run/'analysis_scripts/audit_goals_roles_crossed.py'),
                                    '--run', str(run), '--out', str(out)], check=True)
                audit = json.loads((out/'audit.json').read_text())
                batch[stage+'_audit'] = str(out/'audit.json')
                batch[stage+'_error_counts'] = audit['error_counts']
                if not gate_passed(audit, 1792 if stage == 'pilot' else 5376):
                    batch['status'] = stage+'_failed_gate_bounded_repair_stopped'
                    save()
                    print(batch['status'], flush=True)
                    return
                if stage == 'pilot':
                    full = Path(batch['baseline'])
                    verify_run(full, batch['script_sha256']['baseline'])
                    receipt = submit(full/'run.sh', full/'submission.json')
                    batch['baseline_submission'] = receipt
                    batch['status'] = 'pilot_passed_full_baseline_submitted'
                    save()
                    controller('baseline', [receipt['job_id']])
                else:
                    for entry in batch['runs']:
                        path = Path(entry['path'])
                        verify_run(path, entry['script_sha256'])
                        for name in ('main.jsonl', 'checks.jsonl'):
                            assert (path/'data/goals_roles_pilot'/name).read_bytes() == (run/'data/goals_roles_pilot'/name).read_bytes()
                        for name in ('run_goals_roles.py', 'score_goals_roles.py'):
                            assert (path/'src/selfconcept/soo'/name).read_bytes() == (run/'src/selfconcept/soo'/name).read_bytes()
                    for entry in batch['runs']:
                        path = Path(entry['path'])
                        entry.update(submit(path/'run.sh', path/'submission.json'), status='submitted')
                        save()
                    batch['status'] = 'baseline_passed_six_interventions_submitted'
                    save()
                    controller('paired', [entry['job_id'] for entry in batch['runs']])
            elif stage == 'paired':
                for entry in batch['runs']:
                    assert (Path(entry['path'])/'output/summary.json').exists(), f"incomplete: {entry['path']}"
                out = Path(batch['paired_analysis'])
                if not out.exists():
                    subprocess.run([sys.executable, str(Path(batch['controller'])/'analyze_goals_roles_crossed_pairs.py'),
                                    '--batch', str(batch_path), '--out', str(out)], check=True)
                assert (out/'audit.json').exists(), 'paired analysis incomplete'
                batch['status'] = 'completed_and_analyzed'
                for entry in batch['runs']:
                    entry['status'] = 'completed'
                save()
            else:
                raise ValueError(stage)
        except Exception as exc:
            batch['status'] = 'continuation_error_requires_inspection'
            batch['continuation_error'] = f'{type(exc).__name__}: {exc}'
            save()
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch', required=True, type=Path)
    p.add_argument('--stage', required=True, choices=('pilot', 'baseline', 'paired'))
    args = p.parse_args()
    main(args.batch.resolve(), args.stage)
