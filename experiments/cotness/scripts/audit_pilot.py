"""Read-only audit of completion, repeated text, and structurally missing labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit(directory):
    generations, outcomes = read(directory / 'generations.jsonl'), read(directory / 'outcomes.jsonl')
    grades = {r['example_id']: r for r in read(directory / 'base_insider_trading_none_graded.jsonl')}
    coding = [r for r in outcomes if r['scenario'].startswith('impossible_') or r['scenario'] == 'evilgenie']
    repeated = []
    for r in generations:
        if r['status'] != 'truncated':
            continue
        words = r['raw_response'].split()[-1024:]
        grams = [tuple(words[i:i+32]) for i in range(max(0, len(words)-31))]
        fraction = 1 - len(set(grams))/len(grams) if grams else 0.
        repeated.append({'scenario': r['scenario'], 'example_id': r['example_id'], 'turn': r['turn'],
                         'tail_repeated_32_word_fraction': fraction,
                         'has_final_span': bool(r['spans']['final'])})
    normal = [r for r in outcomes if r['scenario'] == 'sandbagging' and r.get('expected') == 'normal']
    recovered = []
    invalid = []
    for r in generations:
        if r['scenario'] == 'sandbagging' and r['status'] == 'complete' and '<step2-results>' not in r['response']:
            cot = ''.join(r['raw_response'][a:b] for a,b in r['spans']['cot'])
            if '<step2-results>' in cot:
                recovered.append(r['example_id'])
        if r['scenario'] == 'insider_trading' and grades[r['example_id']]['label'] == 'invalid':
            invalid.append({'example_id': r['example_id'], 'has_email_address': '@whitestone.com' in r['response'],
                            'has_execute_trade_action': 'Action: execute_trade' in r['response']})
    files = ['manifest.json', 'generations.jsonl', 'outcomes.jsonl', 'base_insider_trading_none_graded.jsonl']
    return {'directory': str(directory), 'input_sha256': {f: hashlib.sha256((directory/f).read_bytes()).hexdigest() for f in files},
            'outcomes': dict(Counter(r['status'] for r in outcomes)),
            'coding_outcomes': dict(Counter(r['status'] for r in coding)),
            'truncated_generations': repeated,
            'sandbag_normal': {'n': len(normal), 'correctness_scorable': sum(r.get('correct') is not None for r in normal),
                              'correct': sum(r.get('correct') is True for r in normal),
                              'both_answers_established': sum(r.get('evidence', {}).get('disagree') is not None for r in normal)},
            'missing_private_tag_recoverable_from_cot_candidates': recovered, 'invalid_insider_outputs': invalid}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = {p.name: audit(p) for p in args.directories}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    for model, r in result.items():
        repetitions = sum(x['tail_repeated_32_word_fraction'] > .5 for x in r['truncated_generations'])
        print(model, 'coding', r['coding_outcomes'], 'truncated attempts with >50% repeated tail:', repetitions)


if __name__ == '__main__':
    main()
