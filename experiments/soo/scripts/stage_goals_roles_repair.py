"""Freeze the bounded repair workflow. Does not submit any job."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import subprocess

from launch_goals_roles import ROOT, SOO, digest, freeze


def main(repeat_run):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    result = SOO/'results/study2'
    batch_path = result/f'goals-roles-repair-batch-{stamp}.json'
    controller = result/f'goals-roles-repair-controller-{stamp}'
    controller.mkdir()
    for name in ('continue_goals_roles_repair.py', 'analyze_goals_roles_crossed_pairs.py', 'analyze_goals_roles_pairs.py'):
        shutil.copy2(SOO/'scripts'/name, controller/name)
    pilot = freeze('base', data_dir=SOO/'data/goals_roles_scopepilot1', wall_minutes=30)
    base = freeze('base', data_dir=SOO/'data/goals_roles_crossed2', wall_minutes=45)
    batch = {'dataset_id': 'crossed2', 'status': 'staged_waiting_repeatability_review',
             'created_utc': datetime.now(timezone.utc).isoformat(),
             'repeatability': str(repeat_run.resolve()), 'pilot': str(pilot), 'baseline': str(base),
             'controller': str(controller), 'runs': [],
             'paired_analysis': str(result/f'goals-roles-crossed2-paired-analysis-{stamp}'),
             'script_sha256': {'pilot': digest(pilot/'run.sh'), 'baseline': digest(base/'run.sh')}}
    for condition, seed in (('adapter', None), ('band', None), ('offset', None), ('random', 0), ('random', 1), ('random', 2)):
        run = freeze(condition, seed, SOO/'data/goals_roles_crossed2', 45)
        batch['runs'].append({'condition': condition, 'random_seed': seed, 'path': str(run),
                              'status': 'staged_not_submitted', 'script_sha256': digest(run/'run.sh')})
    for stage in ('pilot', 'baseline', 'paired'):
        command = ['/usr/bin/python3.11', '-u', str(controller/'continue_goals_roles_repair.py'),
                   '--batch', str(batch_path), '--stage', stage]
        script = controller/f'{stage}.sh'
        script.write_text(f'''#!/bin/bash --login
#SBATCH --job-name=soo-repair-{stage}-audit
#SBATCH --partition=dw
#SBATCH --qos=dw87
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output={SOO}/slurm-logs/%x-%j.out
set -euo pipefail
cd {shlex.quote(str(controller))}
export PYTHONPATH={shlex.quote(str(base/'src'))}
{shlex.join(command)}
''')
        script.chmod(0o755)
        subprocess.run(['bash', '-n', str(script)], check=True)
    (controller/'provenance.json').write_text(json.dumps({
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'files_sha256': {p.name: digest(p) for p in controller.iterdir() if p.is_file()}}, indent=2)+'\n')
    batch_path.write_text(json.dumps(batch, indent=2)+'\n')
    print(str(batch_path), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repeat-run', required=True, type=Path)
    args = p.parse_args()
    main(args.repeat_run)
