"""Freeze repair pilots and conditionally expand each validated reasoning model."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

from launch import ROOT, batch_header, preflight, sha
from selfconcept.cotness.gate import CODE_SCENARIOS, SCENARIOS
from selfconcept.cotness.roles import MODELS
from selfconcept.cotness.run import read_jsonl, select_examples

PILOTS = {key: ROOT / 'experiments/cotness/results' / ('pilot-20260922T192618Z' if key == 'qwen38-27b' else 'pilot-20260922T192346Z') / 'output' / key for key in MODELS}


def judge_commands(directory):
    lines = []
    for script, scenario in [('judge_apollo.py', 'roleplaying'), ('judge_insider.py', 'insider_trading')]:
        path = f'{directory}/base_{scenario}_none.jsonl'
        lines += [f'if [[ -s {path} ]]; then',
                  f'  "$PY" experiments/soo/scripts/{script} --responses {path} --batch-size 2', 'fi']
    files = ' '.join(f'{directory}/base_{s}.jsonl' for s in CODE_SCENARIOS)
    lines += [f'files=(); for path in {files}; do [[ ! -s "$path" ]] || files+=("$path"); done',
              'if (( ${#files[@]} )); then',
              '  "$PY" -m selfconcept.codebench.judge --model Qwen/Qwen2.5-72B-Instruct --responses "${files[@]}" --batch-size 1 --skip-existing', 'fi',
              f'if [[ -f {directory}/outcomes.jsonl ]]; then "$PY" -m selfconcept.cotness.analyze {directory}; fi']
    return '\n'.join(lines) + '\n'


def selected_ids(n, code_n, offset, code_offset):
    result = {}
    for scenario in SCENARIOS:
        code = scenario in CODE_SCENARIOS
        name = scenario.removesuffix('_mirrored')
        if code:
            path = ROOT / f'benchmarks/codebench/data/{name}.jsonl'
        else:
            folder = 'eval_mirrored' if scenario.endswith('_mirrored') else 'eval_apollo' if name in ('roleplaying', 'insider_trading', 'sandbagging') else 'eval'
            path = ROOT / f'experiments/soo/data/{folder}/{name}.jsonl'
        rows = select_examples(read_jsonl(path), code_n if code else n, name, code_offset if code else offset)
        result[scenario] = [r['example_id'] for r in rows]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-submit', action='store_true')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    report = preflight(args.models)
    selections = {'repair': selected_ids(2, 1, 0, 0), 'expanded': selected_ids(32, 12, 8, 4)}
    for scenario in SCENARIOS:
        assert not set(selections['repair'][scenario]) & set(selections['expanded'][scenario])
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    snapshot = ROOT / 'experiments/cotness/results' / f'followup-{stamp}'
    snapshot.mkdir(parents=True, exist_ok=False)
    (snapshot / 'logs').mkdir()
    for rel in ('src', 'benchmarks/codebench', 'experiments/soo/data', 'experiments/soo/scripts', 'experiments/cotness/data', 'experiments/cotness/scripts'):
        shutil.copytree(ROOT / rel, snapshot / rel, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for rel in ('experiments/cotness/PLAN.md', 'experiments/cotness/FOLLOWUP.md', 'experiments/cotness/README.md', 'pyproject.toml', 'uv.lock', 'tests/test_cotness.py'):
        target = snapshot / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    for key in args.models:
        source = PILOTS[key]
        manifest = json.loads((source / 'manifest.json').read_text())
        validation = json.loads((source / 'probes/validation.json').read_text())
        if manifest['model'] != MODELS[key].__dict__ or not validation['usable']:
            raise ValueError(f'Cannot reuse pilot probe for {key}')
        target = snapshot / 'frozen_probes' / key
        shutil.copytree(source / 'probes', target)
        shutil.copy2(source / 'manifest.json', target / 'source_manifest.json')
        for stage, limits, hours in [('repair', '--n 2 --code-n 1 --offset 0 --code-offset 0', 18),
                                     ('expanded', '--n 32 --code-n 12 --offset 8 --code-offset 4', 72)]:
            directory = f'output/{stage}/{key}'
            script = batch_header(f'cot-{stage}-{key}', MODELS[key].gpus, hours, snapshot)
            if stage == 'expanded':
                script += f'"$PY" -m selfconcept.cotness.gate output/repair/{key}\n'
            script += f'"$PY" -u -m selfconcept.cotness.run --model {key} --out {directory} --probe-source frozen_probes/{key} {limits} --temperature 1.0 --top-p 0.95 --top-k 64 --seed 1729 --max-new-tokens 8192 --code-max-new-tokens 32768\n'
            script += f'"$PY" -m selfconcept.cotness.analyze {directory}\n'
            (snapshot / f'{stage}-{key}.sh').write_text(script)
        judge = batch_header(f'cot-judge-{key}', 3, 12, snapshot)
        judge += 'export SOO_CHAT_KWARGS=\'{"enable_thinking": false}\'\n'
        judge += judge_commands(f'output/repair/{key}') + judge_commands(f'output/expanded/{key}')
        (snapshot / f'judge-{key}.sh').write_text(judge)
    launch = {'created_utc': stamp, 'snapshot': str(snapshot), 'models': report,
              'selections': selections, 'probe_sources': {k: str(PILOTS[k]) for k in args.models},
              'decoding': {'temperature': 1.0, 'top_p': .95, 'top_k': 64, 'seed': 1729, 'ordinary_tokens': 8192, 'coding_tokens': 32768},
              'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
              'sha256': {str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob('*') if p.is_file()}, 'jobs': {}}
    record = snapshot / 'launch.json'
    record.write_text(json.dumps(launch, indent=2))
    for script in snapshot.glob('*.sh'):
        subprocess.run(['bash', '-n', str(script)], check=True)
    if not args.no_submit:
        def submit(stage, key, dependencies=None):
            command = ['sbatch', '--parsable', '--kill-on-invalid-dep=yes']
            if dependencies:
                command.append('--dependency=' + dependencies)
            command.append(str(snapshot / f'{stage}-{key}.sh'))
            job = subprocess.check_output(command, cwd=ROOT, text=True).strip().split(';')[0]
            launch['jobs'][f'{stage}/{key}'] = job
            record.write_text(json.dumps(launch, indent=2))
            return job
        # Queue every repair before longer follow-ups, to obtain diagnostics early.
        repairs = {key: submit('repair', key) for key in args.models}
        for key in args.models:
            expanded = submit('expanded', key, f'afterok:{repairs[key]}')
            submit('judge', key, f'afterok:{repairs[key]},afterany:{expanded}')
    print(json.dumps({'snapshot': str(snapshot), 'jobs': launch['jobs']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
