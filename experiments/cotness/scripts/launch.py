"""Freeze and submit the CoT-ness/deception pilot on already-downloaded models."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from selfconcept.cotness.roles import MODELS


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight(keys):
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoTokenizer
    from selfconcept.cotness.roles import ROLES, render_probe
    report = {}
    for key in keys:
        spec = MODELS[key]
        cached_config = try_to_load_from_cache(spec.model, "config.json", revision=spec.revision)
        if not isinstance(cached_config, str):
            raise RuntimeError(f"No cached config for {key}")
        path = Path(cached_config).parent
        indices = list(path.glob("*.safetensors.index.json"))
        if indices:
            weights = set(json.loads(indices[0].read_text())["weight_map"].values())
        else:
            weights = {p.name for p in path.glob("*.safetensors")}
        if not weights or any(not (path / name).is_file() for name in weights):
            raise RuntimeError(f"Incomplete downloaded checkpoint: {key}")
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        checks = {}
        for role in ROLES:
            text, span = render_probe(tok, spec.family, "The river flows through the valley.", role)
            checks[role] = {"span": span, "prefix_tail": text[:span[0]][-100:]}
        corpus = [json.loads(line) for line in (ROOT / "experiments/cotness/data/neutral.jsonl").read_text().splitlines()]
        for row in corpus:
            text = tok.decode(tok(row["text"], add_special_tokens=False)["input_ids"][:192])
            for role in ROLES:
                render_probe(tok, spec.family, text, role)
        report[key] = {**asdict(spec), "snapshot": str(path), "weights": sorted(weights),
                       "config_sha256": sha(path / "config.json"), "template_sha256": hashlib.sha256(tok.chat_template.encode()).hexdigest(),
                       "role_checks": checks, "corpus_role_renderings_checked": len(corpus) * len(ROLES)}
    return report


def batch_header(name, gpus, hours, snapshot):
    return f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --time={hours}:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:{gpus}
#SBATCH --qos=dw87
#SBATCH --exclude=dw-2-4
#SBATCH --mem-per-cpu=24G
#SBATCH --output={snapshot}/logs/%x-%j.out
set -euo pipefail
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH={shlex.quote(str(snapshot / 'src'))}
unset SOO_CHAT_KWARGS
test -n "${{CUDA_VISIBLE_DEVICES:-}}" || {{ echo 'No Slurm-assigned GPU'; exit 1; }}
cd {shlex.quote(str(snapshot))}
PY={shlex.quote(str(ROOT / '.venv/bin/python'))}
echo "Assigned CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--no-submit", action="store_true")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--code-n", type=int, default=4)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    model_report = preflight(args.models)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot = ROOT / "experiments/cotness/results" / f"pilot-{stamp}"
    snapshot.mkdir(parents=True, exist_ok=False)
    (snapshot / "logs").mkdir()
    for rel in ("src", "benchmarks/codebench", "experiments/soo/data", "experiments/soo/scripts", "experiments/cotness/data"):
        shutil.copytree(ROOT / rel, snapshot / rel, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for rel in ("experiments/cotness/PLAN.md", "experiments/cotness/README.md", "experiments/cotness/scripts/launch.py", "pyproject.toml", "uv.lock", "tests/test_cotness.py"):
        target = snapshot / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    for key in args.models:
        spec = MODELS[key]
        script = batch_header(f"cotness-{key}", spec.gpus, 36 if key == "kimi-72b" else 18, snapshot)
        script += f'"$PY" -u -m selfconcept.cotness.run --model {key} --out output/{key} --n {args.n} --code-n {args.code_n}\n'
        script += f'"$PY" -m selfconcept.cotness.analyze output/{key}\n'
        (snapshot / f"{key}.sh").write_text(script)
    judge = batch_header("cotness-judge", 3, 8, snapshot)
    judge += 'export SOO_CHAT_KWARGS=\'{"enable_thinking": false}\'\n'
    for key in args.models:
        directory = f"output/{key}"
        judge += f'if [[ -f {directory}/outcomes.jsonl ]]; then\n'
        judge += f'  "$PY" experiments/soo/scripts/judge_apollo.py --responses {directory}/base_roleplaying_none.jsonl --batch-size 2\n'
        judge += f'  "$PY" experiments/soo/scripts/judge_insider.py --responses {directory}/base_insider_trading_none.jsonl --batch-size 2\n'
        files = " ".join(f"{directory}/base_{s}.jsonl" for s in ("impossible_original", "impossible_oneoff", "impossible_conflicting", "evilgenie"))
        judge += f'  "$PY" -m selfconcept.codebench.judge --model Qwen/Qwen2.5-72B-Instruct --responses {files} --batch-size 1 --skip-existing\n'
        judge += f'  "$PY" -m selfconcept.cotness.analyze {directory}\nfi\n'
    (snapshot / "judge.sh").write_text(judge)
    source_files = {str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob("*") if p.is_file()}
    report = {"created_utc": stamp, "models": model_report, "n": args.n, "code_n": args.code_n,
              "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
              "sha256": source_files, "jobs": {}, "snapshot": str(snapshot)}
    (snapshot / "launch.json").write_text(json.dumps(report, indent=2))
    for path in snapshot.glob("*.sh"):
        subprocess.run(["bash", "-n", str(path)], check=True)
    if not args.no_submit:
        for key in args.models:
            job = subprocess.check_output(["sbatch", "--parsable", str(snapshot / f"{key}.sh")], cwd=ROOT, text=True).strip().split(";")[0]
            report["jobs"][key] = job
            (snapshot / "launch.json").write_text(json.dumps(report, indent=2))
        dependency = "afterok:" + ":".join(report["jobs"].values())
        report["jobs"]["judge"] = subprocess.check_output(["sbatch", "--parsable", f"--dependency={dependency}", str(snapshot / "judge.sh")], cwd=ROOT, text=True).strip().split(";")[0]
        (snapshot / "launch.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"snapshot": str(snapshot), "jobs": report["jobs"]}, indent=2))


if __name__ == "__main__":
    main()
