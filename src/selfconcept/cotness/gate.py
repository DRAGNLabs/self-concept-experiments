"""Completion-only gate for expanding the repair pilot; never inspect outcome labels."""
import argparse
import json
from collections import Counter
from pathlib import Path

from .run import read_jsonl

CODE_SCENARIOS = ('impossible_original', 'impossible_conflicting', 'impossible_oneoff', 'evilgenie')
SCENARIOS = ('main', 'main_mirrored', 'treasure_hunt', 'treasure_hunt_mirrored',
             'perspectives', 'perspectives_mirrored', 'roleplaying', 'insider_trading',
             'sandbagging', *CODE_SCENARIOS)


def assess(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    validation = json.loads((directory / 'probes/validation.json').read_text())
    rows = read_jsonl(directory / 'outcomes.jsonl') if (directory / 'outcomes.jsonl').exists() else []
    generations = read_jsonl(directory / 'generations.jsonl') if (directory / 'generations.jsonl').exists() else []
    counts = Counter(r['scenario'] for r in rows)
    expected = {s: 1 if s in CODE_SCENARIOS else 4 if s == 'sandbagging' else 2 for s in SCENARIOS}
    unique = len({(r['scenario'], r['example_id']) for r in rows}) == len(rows)
    covered = {(r['scenario'], r['example_id']) for r in generations if r['turn'] == 0} == {
        (r['scenario'], r['example_id']) for r in rows}
    complete = {kind: sum(r['status'] == 'complete' for r in rows if (r['scenario'] in CODE_SCENARIOS) == code)
                for kind, code in [('coding', True), ('ordinary', False)]}
    midpoint = str(validation['primary_layer'])
    aligned = bool(generations) and all(r.get('generated_alignment') == 'exact' and
               r.get('scores', {}).get(midpoint, {}).get('prompt', {}).get('mean') is not None for r in generations)
    checks = {'manifest_complete': manifest['stage'] == 'complete', 'probe_valid': validation['usable'],
              'expected_24_unique_tasks': counts == expected and unique,
              'first_turn_measurements_for_every_task': covered,
              'ordinary_complete_at_least_16_of_20': complete['ordinary'] >= 16,
              'coding_complete_at_least_3_of_4': complete['coding'] >= 3,
              'exact_token_alignment': aligned}
    return {'passed': all(checks.values()), 'checks': checks, 'complete': complete,
            'counts': dict(counts), 'note': 'No deception labels, correlations or judge results enter this gate.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    result = assess(args.directory)
    (args.directory / 'gate.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result['passed'] else 3)


if __name__ == '__main__':
    main()
