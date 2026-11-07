"""Validates the subagent sentence pairs and combines them into data/controls/sentences.json.

The subagents (subagent_instructions.md) each write <sentences_dir>/<pair_id>.json. Per pair_id: 5 pairs on distinct
aspects, 8-20 words, no banned topic words, and A and B differ in exactly one word, which is the word in A and the
antonym in B (punctuation stripped). Writes the problems per pair_id (empty list = valid) to <problems.json>, and when
every pair_id is valid writes {pair_id: [{aspect, a, b}, ...]} to data/controls/sentences.json.

    python scripts/controls/check_sentences.py <sentences_dir> <problems.json>
"""
import json
import re
import sys
from pathlib import Path
from typing import TypedDict

DATA = Path(__file__).parents[2] / "data/controls"
BANNED = re.compile(r"\b(ai|assistants?|chatbots?|computers?|code|tests?|exams?|rules?|i|me|my|we|us|our|you|your)\b", re.IGNORECASE)
SENTENCE_PAIRS_PER_WORD, MIN_WORDS, MAX_WORDS = 5, 8, 20


class SentencePair(TypedDict):
    aspect: int
    a: str
    b: str


def words(sentence: str) -> list[str]:
    return [token.strip(".,;:!?\"'") for token in sentence.split()]


def pair_problems(pair: SentencePair, word: str, antonym: str) -> list[str]:
    a, b = pair["a"], pair["b"]
    words_a, words_b = words(a), words(b)
    differing = [(x, y) for x, y in zip(words_a, words_b) if x != y]
    swapped = {a[:match.start()] + antonym + a[match.end():] for match in re.finditer(rf"\b{re.escape(word)}\b", a)}
    return ([] if len(words_a) == len(words_b) and differing == [(word, antonym)] and b in swapped else ["not a one-word swap"]) \
        + ([] if MIN_WORDS <= len(words_a) <= MAX_WORDS else ["length"]) + (["banned word"] if BANNED.search(a + " " + b) else [])


def file_problems(sentence_pairs: list[SentencePair] | None, word: str, antonym: str) -> list[str]:
    if sentence_pairs is None:
        return ["missing"]
    aspects = {pair["aspect"] for pair in sentence_pairs}
    return ([] if len(sentence_pairs) == SENTENCE_PAIRS_PER_WORD and len(aspects) == SENTENCE_PAIRS_PER_WORD else ["aspects or count"]) \
        + [f"{index}: {problem}" for index, pair in enumerate(sentence_pairs) for problem in pair_problems(pair, word, antonym)]


def main() -> None:
    sentences_dir, problems_path = Path(sys.argv[1]), Path(sys.argv[2])
    antonym_pairs = [json.loads(line) for line in (DATA / "pairs.jsonl").open()]
    sentence_pairs_by_pair_id: dict[str, list[SentencePair] | None] = {
        pair["pair_id"]: json.loads(path.read_text())["pairs"] if (path := sentences_dir / f"{pair['pair_id']}.json").exists() else None
        for pair in antonym_pairs}
    problems_by_pair_id = {pair["pair_id"]: file_problems(sentence_pairs_by_pair_id[pair["pair_id"]], pair["word"], pair["antonym"]) for pair in antonym_pairs}
    problems_path.write_text(json.dumps(problems_by_pair_id, indent=1))
    if not any(problems_by_pair_id.values()):
        (DATA / "sentences.json").write_text(json.dumps(sentence_pairs_by_pair_id, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
