"""Run model-written Python in a throwaway workspace with hard limits.

The reference ImpossibleBench/EvilGenie harnesses execute inside Docker. Slurm
compute nodes here have no Docker, so this wraps the run in bubblewrap when
available (read-only root, private /tmp, no network, workspace the only
writable bind) plus POSIX rlimits on CPU seconds and address space, with a
wall-clock kill as backstop. Without bwrap it falls back to the
rlimits alone and says so once.
"""

import os
import resource
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

MEMORY_LIMIT_BYTES = 4 * 1024**3
CAPTURE_LIMIT_CHARS = 20_000

_bwrap = shutil.which("bwrap")
_warned_no_bwrap = False


@dataclass
class ExecResult:
    success: bool
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False


def _set_rlimits(cpu_seconds: int) -> None:
    # No RLIMIT_NPROC: it counts every process of the uid machine-wide, so a
    # low cap makes bwrap's unshare() fail with EAGAIN on a shared login node.
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _command(workdir: Path, argv: list[str]) -> list[str]:
    global _warned_no_bwrap
    if _bwrap is None:
        if not _warned_no_bwrap:
            print("code_sandbox: bwrap not found; running with rlimits only", file=sys.stderr)
            _warned_no_bwrap = True
        return argv
    return [
        _bwrap,
        "--ro-bind", "/", "/",
        "--dev", "/dev",
        "--proc", "/proc",
        "--tmpfs", "/tmp",
        "--bind", str(workdir), str(workdir),
        "--unshare-all",
        "--die-with-parent",
        "--chdir", str(workdir),
        "--",
        *argv,
    ]


def run_python(
    files: dict[str, str],
    argv: list[str],
    timeout: float,
    stdin: str | None = None,
    workdir: Path | None = None,
) -> ExecResult:
    """Write `files` into a fresh (or given) workdir and run `python argv...` there.

    `timeout` is both the wall-clock limit and (rounded up) the CPU-seconds
    rlimit. Output is truncated to CAPTURE_LIMIT_CHARS per stream.
    """
    tmp = None
    if workdir is None:
        tmp = tempfile.mkdtemp(prefix="soo_code_")
        workdir = Path(tmp)
    workdir = workdir.resolve()
    for name, content in files.items():
        (workdir / name).write_text(content)
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(workdir),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
    }
    cpu_seconds = max(1, int(timeout) + 1)
    try:
        proc = subprocess.run(
            _command(workdir, [sys.executable, *argv]),
            cwd=workdir,
            env=env,
            input=stdin,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            preexec_fn=lambda: _set_rlimits(cpu_seconds),
        )
        result = ExecResult(
            success=proc.returncode == 0,
            returncode=proc.returncode,
            stdout=proc.stdout[:CAPTURE_LIMIT_CHARS],
            stderr=proc.stderr[:CAPTURE_LIMIT_CHARS],
        )
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"")
        err = (e.stderr or b"")
        result = ExecResult(
            success=False,
            returncode=None,
            stdout=(out.decode("utf-8", "replace") if isinstance(out, bytes) else out)[:CAPTURE_LIMIT_CHARS],
            stderr=(err.decode("utf-8", "replace") if isinstance(err, bytes) else err)[:CAPTURE_LIMIT_CHARS]
            + f"\nVerification timed out after {timeout:.0f}s.",
            timed_out=True,
        )
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    return result
