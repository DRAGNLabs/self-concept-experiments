"""Submit the six frozen interventions only after the completed baseline audit passes.

Run as a small Slurm job dependent on successful baseline completion. No inference
occurs here. Submission receipts make repeated invocation idempotent.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def gate_passed(audit):
    return (audit.get('condition') == 'base' and audit.get('n') == 5376
            and audit.get('all_gates_passed') is True
            and audit.get('global_gate', {}).get('passed') is True
            and len(audit.get('cells', [])) == 8
            and all(c.get('gate', {}).get('passed') is True for c in audit['cells']))


def main(batch_path):
    def save():
        temp = batch_path.with_suffix('.tmp')
        temp.write_text(json.dumps(batch, indent=2) + '\n')
        temp.replace(batch_path)
    with batch_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        batch = json.loads(batch_path.read_text())
        base = Path(batch['baseline'])
        audit_dir = base / 'crossed_audit'
        try:
            assert (base / 'output/summary.json').exists(), 'baseline incomplete'
            if not (audit_dir / 'audit.json').exists():
                subprocess.run([sys.executable, str(base / 'analysis_scripts/audit_goals_roles_crossed.py'),
                    '--run', str(base), '--out', str(audit_dir), '--original-run',
                    str(batch_path.parent / 'goals-roles-base-v2-20261026T220826126905Z')], check=True)
            audit = json.loads((audit_dir / 'audit.json').read_text())
            batch['baseline_audit'] = str(audit_dir / 'audit.json')
            batch['baseline_errors'] = str(audit_dir / 'errors.jsonl')
            batch['baseline_audit_completed_utc'] = datetime.now(timezone.utc).isoformat()
            batch['baseline_error_counts'] = audit['error_counts']
            if not gate_passed(audit):
                batch['status'] = 'baseline_failed_gate_interventions_not_submitted'
                save()
                print('Baseline failed a frozen gate; no interventions submitted.', flush=True)
                return
            for r in batch['runs']:
                path = Path(r['path'])
                provenance = json.loads((path / 'provenance.json').read_text())
                assert digest(path / 'run.sh') == r['script_sha256'], 'submission script changed'
                for name in ('data/goals_roles_pilot/main.jsonl', 'data/goals_roles_pilot/checks.jsonl',
                             'src/selfconcept/soo/run_goals_roles.py', 'src/selfconcept/soo/score_goals_roles.py',
                             'GOALS_ROLES_CROSSED.md'):
                    assert digest(path / name) == provenance['files_sha256'][name], name
                    assert (path / name).read_bytes() == (base / name).read_bytes(), name
                for name, expected in provenance['files_sha256'].items():
                    if name.startswith('adapter/') or name in ('band_deltas.pt', 'band_deltas.json'):
                        assert digest(path / name) == expected, name
            batch['status'] = 'baseline_passed_submitting_interventions'
            save()
            for r in batch['runs']:
                path = Path(r['path'])
                receipt_path = path / 'submission.json'
                if receipt_path.exists():
                    receipt = json.loads(receipt_path.read_text())
                else:
                    job = subprocess.check_output(['sbatch', '--parsable', str(path / 'run.sh')], text=True).strip().split(';')[0]
                    receipt = {'job_id': job, 'submitted_utc': datetime.now(timezone.utc).isoformat(),
                               'script_sha256': digest(path / 'run.sh')}
                    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
                r.update(receipt, status='submitted')
                save()
                print(r['condition'], r['random_seed'], r['job_id'], flush=True)
            batch['status'] = 'interventions_submitted'
            save()
        except Exception as exc:
            batch['status'] = 'continuation_error_requires_inspection'
            batch['continuation_error'] = f'{type(exc).__name__}: {exc}'
            save()
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch', required=True, type=Path)
    args = p.parse_args()
    main(args.batch.resolve())
