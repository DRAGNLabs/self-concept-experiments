"""Freeze new protocol/controllers around unchanged payoff1 execution assets."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from launch_goals_roles import ROOT,SOO,digest,freeze


def main(original_path):
    original=json.loads(original_path.read_text())
    assert original['status']=='development_baseline_failed_gate_stopped'
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root=SOO/'results/study2';controller=root/f'payoff-followups1-controller-{stamp}';controller.mkdir()
    for name in ('analyze_explicit_payoff.py','analyze_payoff_followups.py','continue_explicit_payoff.py',
                 'continue_payoff_followups.py','stage_payoff_followups.py'):
        shutil.copy2(SOO/'scripts'/name,controller/name)
    for name in ('PLAN.md','FINDINGS.md','EXPLICIT_PAYOFF.md'):
        shutil.copy2(SOO/name,controller/name)
    shutil.copy2(ROOT/'tests/test_soo_payoff_followups.py',controller/'test_soo_payoff_followups.py')
    preflight=SOO/'results/study2/payoff-followups1-preflight'
    for name in ('validation.json','tokenization.json'):
        shutil.copy2(preflight/name,controller/name)
    picker=subprocess.check_output(['/usr/bin/python3.11',str(ROOT/'gpu_pick.py'),'--hours','0.5','--vram-gb','75','--cpus','4','--mem-gb','96','--no-requeue-ok','--fast','--cuda','13.0','--top','10'],text=True)
    (controller/'gpu_pick.txt').write_text(picker)
    diagnostic=freeze('base',data_dir=SOO/'data/payoff_name_column1',wall_minutes=30,partition='dw',qos='dw87')
    batch_path=root/f'payoff-followups1-batch-{stamp}.json'
    batch={'protocol':'post_hoc_private_goal_followup1','status':'staged_not_submitted',
           'private_status':'staged_not_submitted','diagnostic_status':'staged_not_submitted',
           'created_utc':datetime.now(timezone.utc).isoformat(),
           'original_batch':str(original_path.resolve()),'original_batch_sha256':digest(original_path),
           'controller':str(controller),'runs':deepcopy(original['runs']),
           'analysis':{split:str(root/f'payoff-private1-{split}-analysis-{stamp}') for split in ('development','validation')},
           'diagnostic':{'path':str(diagnostic),'status':'staged_not_submitted','script_sha256':digest(diagnostic/'run.sh')},
           'diagnostic_analysis':str(root/f'payoff-name-column1-analysis-{stamp}'),
           'resources':{'partition':'dw','qos':'dw87','gpu':'a100:1','cpus':4,'host_gb':96},
           'authorization':'User requested recording the narrower private-goal comparison and name/column diagnostic and proceeding; original combined protocol remains failed.'}
    batch['runs']['development'][0]['status']='reused_completed'
    # Freeze the source actually imported by the CPU audits (including new diagnostic generator).
    source=diagnostic/'src'
    for stage in ('diagnostic','development_paired','validation_base','validation_paired'):
        script=controller/(stage+'.sh')
        command=['/usr/bin/python3.11','-u',str(controller/'continue_payoff_followups.py'),'--batch',str(batch_path),'--stage',stage]
        script.write_text(f'''#!/bin/bash --login
#SBATCH --job-name=soo-payfollow-{stage}
#SBATCH --partition=dw
#SBATCH --qos=dw87
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output={SOO}/slurm-logs/%x-%j.out
set -euo pipefail
export PYTHONPATH={shlex.quote(str(source))}
{shlex.join(command)}
''')
        subprocess.run(['bash','-n',str(script)],check=True)
    (controller/'receipts').mkdir()
    batch['controller_sha256']={p.name:digest(p) for p in controller.iterdir() if p.is_file()}
    batch_path.write_text(json.dumps(batch,indent=2)+'\n')
    print(str(batch_path),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--original-batch',type=Path,required=True)
    a=p.parse_args();main(a.original_batch)
