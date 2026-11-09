"""Samples 100 WordNet antonym pairs for the control directions: single-word lemmas whose SemCor sense count is >= 5 on
both sides (WordNet's lemma counts), each word used in at most one pair. Writes data/controls/pairs.jsonl.

Needs the WordNet corpus (only it; once):
    NLTK_DATA=~/nobackup/autodelete/nltk_data python -m nltk.downloader -d ~/nobackup/autodelete/nltk_data wordnet
and NLTK_DATA=~/nobackup/autodelete/nltk_data when running this."""
import json
import random
from pathlib import Path
from typing import Literal, TypedDict

from nltk.corpus import wordnet

DATA = Path(__file__).parents[2] / "data/controls"
MIN_SENSE_COUNT, PAIRS, SEED = 5, 100, 0
PartOfSpeech = Literal["adjective", "noun", "verb", "adverb"]
PART_OF_SPEECH_NAMES: dict[str, PartOfSpeech] = {"a": "adjective", "s": "adjective", "n": "noun", "v": "verb", "r": "adverb"}


class AntonymPair(TypedDict):
    pair_id: str
    word: str
    antonym: str
    part_of_speech: PartOfSpeech
    gloss_word: str
    gloss_antonym: str


def candidate_pairs() -> list[AntonymPair]:
    return [{"pair_id": "", "word": lemma.name(), "antonym": antonym.name(), "part_of_speech": PART_OF_SPEECH_NAMES[synset.pos()],
             "gloss_word": synset.definition(), "gloss_antonym": antonym.synset().definition()}
            for synset in wordnet.all_synsets() for lemma in synset.lemmas() for antonym in lemma.antonyms()
            if lemma.name() < antonym.name() and min(lemma.count(), antonym.count()) >= MIN_SENSE_COUNT
            and "_" not in lemma.name() + antonym.name()]


def sample_disjoint(candidates: list[AntonymPair]) -> list[AntonymPair]:
    shuffled = random.Random(SEED).sample(candidates, len(candidates))
    chosen: list[AntonymPair] = []
    for pair in shuffled:
        if len(chosen) < PAIRS and not {pair["word"], pair["antonym"]} & {w for c in chosen for w in (c["word"], c["antonym"])}:
            chosen.append({**pair, "pair_id": f"p{len(chosen):03d}"})
    return chosen


def main() -> None:
    pairs = sample_disjoint(candidate_pairs())
    (DATA / "pairs.jsonl").write_text("".join(json.dumps(pair) + "\n" for pair in pairs))


if __name__ == "__main__":
    main()
