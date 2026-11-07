"""Validates the subagent sentence pairs: per pair_id, 5 pairs on distinct aspects, 8-20 words, no banned topic words,
and A and B differ in exactly one word, which is the word in A and the antonym in B (punctuation stripped).
Writes the problems found per pair_id (empty list = valid) to the JSON path given as the only argument."""
import json
import re
import sys
from pathlib import Path

DATA = Path(__file__).parents[2] / "data/controls"
BANNED = re.compile(r"\b(ai|assistants?|chatbots?|computers?|code|tests?|exams?|rules?|i|me|my|we|us|our|you|your)\b", re.IGNORECASE)


def words(sentence: str) -> list[str]:
    return [token.strip(".,;:!?\"'") for token in sentence.split()]


def pair_problems(a: str, b: str, word: str, antonym: str) -> list[str]:
    words_a, words_b = words(a), words(b)
    differing = [(x, y) for x, y in zip(words_a, words_b) if x != y]
    return ([] if len(words_a) == len(words_b) and differing == [(word, antonym)] and b in {a[:m.start()] + antonym + a[m.end():] for m in re.finditer(rf"\b{re.escape(word)}\b", a)} else ["not a one-word swap"]) \
        + ([] if 8 <= len(words_a) <= 20 else ["length"]) + (["banned word"] if BANNED.search(a + " " + b) else [])


def file_problems(path: Path, word: str, antonym: str) -> list[str]:
    if not path.exists():
        return ["missing"]
    sentence_pairs = json.loads(path.read_text())["pairs"]
    aspects = [p["aspect"] for p in sentence_pairs]
    return ([] if len(sentence_pairs) == 5 and len(set(aspects)) == 5 else ["aspects or count"]) \
        + [f"{index}: {problem}" for index, p in enumerate(sentence_pairs) for problem in pair_problems(p["a"], p["b"], word, antonym)]


def main() -> None:
    pairs = [json.loads(line) for line in (DATA / "pairs.jsonl").open()]
    problems_by_pair_id = {p["pair_id"]: file_problems(DATA / f"sentences/{p['pair_id']}.json", p["word"], p["antonym"]) for p in pairs}
    Path(sys.argv[1]).write_text(json.dumps(problems_by_pair_id, indent=1))


if __name__ == "__main__":
    main()
