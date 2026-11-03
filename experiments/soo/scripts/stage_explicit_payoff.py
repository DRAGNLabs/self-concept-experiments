"""Freeze both payoff splits and all seven conditions before any inference."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from launch_goals_roles import ROOT, SOO, digest, freeze


def main(partition,qos):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root=SOO/'results/study2'
    controller=root/f'explicit-payoff1-controller-{stamp}';controller.mkdir()
    for name in ('analyze_explicit_payoff.py','continue_explicit_payoff.py','stage_explicit_payoff.py'):
        shutil.copy2(SOO/'scripts'/name,controller/name)
    batch_path=root/f'explicit-payoff1-batch-{stamp}.json'
    batch={'task':'explicit_payoff1','status':'staged_not_submitted','created_utc':datetime.now(timezone.utc).isoformat(),
           'controller':str(controller),'runs':{},'analysis':{},'resources':{'partition':partition,'qos':qos,'gpu':'a100:1','cpus':4,'host_gb':96}}
    picker=subprocess.check_output(['/usr/bin/python3.11',str(ROOT/'gpu_pick.py'),'--hours','0.5','--vram-gb','75','--cpus','4','--mem-gb','96','--no-requeue-ok','--fast','--cuda','13.0','--top','10'],text=True)
    (controller/'gpu_pick.txt').write_text(picker)
    for split in ('development','validation'):
        batch['runs'][split]=[]
        batch['analysis'][split]=str(root/f'explicit-payoff1-{split}-analysis-{stamp}')
        for condition,seed in (('base',None),('adapter',None),('band',None),('offset',None),('random',0),('random',1),('random',2)):
            path=freeze(condition,seed,SOO/f'data/explicit_payoff1_{split}',30,partition,qos)
            batch['runs'][split].append({'tag':condition+(str(seed) if seed is not None else ''),'condition':condition,'random_seed':seed,'path':str(path),'script_sha256':digest(path/'run.sh'),'status':'staged_not_submitted'})
            print(f'frozen {split} {condition} {seed}: {path}',flush=True)
    source=Path(batch['runs']['development'][0]['path'])/'src'
    for stage in ('development_base','development_paired','validation_base','validation_paired'):
        script=controller/f'{stage}.sh'
        command=['/usr/bin/python3.11','-u',str(controller/'continue_explicit_payoff.py'),'--batch',str(batch_path),'--stage',stage]
        script.write_text(f'''#!/bin/bash --login
#SBATCH --job-name=soo-payoff-{stage}
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
    batch['controller_sha256']={p.name:digest(p) for p in controller.iterdir() if p.is_file()}
    batch_path.write_text(json.dumps(batch,indent=2)+'\n')
    print(str(batch_path),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--partition',default='cs');p.add_argument('--qos',default='cs')
    args=p.parse_args();main(args.partition,args.qos)
