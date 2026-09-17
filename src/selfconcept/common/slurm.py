"""Slurm submission through MIRROR's sbatch template, activating a mamba env.

MIRROR's own launcher hardcodes `source .venv/bin/activate` (uv); this repo uses mamba.
"""

import shlex
import subprocess
import sys
from dataclasses import asdict

from jinja2 import Environment, PackageLoader, StrictUndefined
from mirror.slurm_util import SlurmConfig
from mirror.util import is_login_node

from selfconcept.common.paths import REPO_ROOT


def submit_slurm_job(slurm: SlurmConfig, module: str, python_args: list[str], activate_cmd: str) -> None:
    """Submit `python -m <module> <python_args>` and exit, if on a login node with job_type 'compute'.

    Returns normally otherwise, so the caller continues and runs the work in-process.
    Jobs run from the repo root, so relative paths in configs resolve against it.
    """
    if not is_login_node() or slurm.job_type != "compute":
        return

    # sbatch fails silently if the log directory doesn't exist
    (REPO_ROOT / slurm.output).parent.mkdir(parents=True, exist_ok=True)

    env = Environment(
        loader=PackageLoader("mirror", "templates"),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    script = env.get_template("slurm.jinja").render(
        **asdict(slurm),
        chdir=str(REPO_ROOT),
        activate_cmd=activate_cmd,
        run_cmd=f"srun python -m {module} {shlex.join(python_args)}",
    )

    res = subprocess.run(["sbatch"], input=script, text=True, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"sbatch failed (exit {res.returncode}):\n{res.stderr}\n\nGenerated script:\n{script}"
        )
    print(res.stdout.strip())
    sys.exit(0)
