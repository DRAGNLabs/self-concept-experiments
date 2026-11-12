"""Sample the in-character judge (assistant-axis 3_judge.py, scores 0-3) for human validation: half the items have a
judge score in --positive-levels and half do not, each half a model-balanced sample of judged role responses (a model
short on a half is filled from the others), shuffled together.

Labelers see what the judge saw (the role, the question, the assistant content) beside the shared rubric: every role's
eval_prompt after its role paragraph, with the question and answer slots pointed at the item. Writes, to --out-dir:
items.jsonl and rubric.md (sent to labelers) and key.jsonl (judge label, model and source; kept back).

    python sample_in_character.py --run gpt=<root> --run muse=<root> --run qwen=<root> --roles-dir data/roles --out-dir <dir>
"""
import argparse
import json
import random
from collections.abc import Iterator
from pathlib import Path

from selfconcept.common.human_validation import Item, JudgedCandidate, KeyRecord, half_positive_sample
from selfconcept.common.jsonl import write_jsonl

LEVELS = ("0", "1", "2", "3")
QUESTION_PLACEHOLDER = "[Question: shown on the right]"
ANSWER_PLACEHOLDER = "[Response: shown on the right]"
GENERIC_ROLE_PARAGRAPH = "You are evaluating whether the model's response displays the role described in the item (shown on the right)."


class Candidate(JudgedCandidate):
    role: str
    response_key: str
    question: str
    answer: str


class InCharacterKeyRecord(KeyRecord):
    role: str
    response_key: str
    run_root: str


def response_key(record: dict) -> str:
    return f"{record['label']}_p{record['prompt_index']}_q{record['question_index']}"


def first_assistant_content(record: dict) -> str:
    return next(turn["content"] for turn in record["conversation"] if turn["role"] == "assistant")


def judged_candidates(model: str, run_root: Path) -> Iterator[Candidate]:
    for scores_path in sorted((run_root / "scores").glob("*.json")):
        role = scores_path.stem
        assert role.endswith("_fic"), f"{run_root}: unexpected judged role {role}"
        score_by_response_key: dict[str, int] = json.loads(scores_path.read_text())
        for line in (run_root / "responses" / f"{role}.jsonl").read_text().splitlines():
            record = json.loads(line)
            key = response_key(record)
            if key in score_by_response_key:
                yield {"model": model, "role": role, "response_key": key, "question": record["question"],
                       "answer": first_assistant_content(record), "judge_label": str(score_by_response_key[key])}


def showable_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Drops empty answers (truncated mid-reasoning) and answers holding a think tag. A model repeating its answer after
    a second </think> does that rarely; a reasoning leak does it everywhere, so more than 1% of a run fails loudly."""
    with_think_tag = [candidate for candidate in candidates if "</think>" in candidate["answer"]]
    assert len(with_think_tag) <= 0.01 * len(candidates), f"{len(with_think_tag)}/{len(candidates)} answers hold </think>: reasoning left in content?"
    return [candidate for candidate in candidates if candidate["answer"].strip() and "</think>" not in candidate["answer"]]


def split_eval_prompt(eval_prompt: str) -> tuple[str, str]:
    role_paragraph, _, shared_rubric = eval_prompt.partition("\n\n")
    return role_paragraph, shared_rubric


def labeler_rubric(shared_rubric: str) -> str:
    return GENERIC_ROLE_PARAGRAPH + "\n\n" + shared_rubric.format(question=QUESTION_PLACEHOLDER, answer=ANSWER_PLACEHOLDER)


def to_item(item_id: str, candidate: Candidate, role_paragraph: str) -> Item:
    return {"item_id": item_id, "sections": [
        {"title": "Role", "text": role_paragraph},
        {"title": "Question", "text": candidate["question"]},
        {"title": "Response", "text": candidate["answer"]},
    ]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="model=run_root (with scores/ and responses/)")
    parser.add_argument("--roles-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--positive-levels", default="3", help="comma-separated judge scores making up the positive half")
    parser.add_argument("--per-half", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    positive_levels: list[str] = args.positive_levels.split(",")
    assert set(positive_levels) < set(LEVELS), f"--positive-levels must be a proper subset of {LEVELS}"
    run_root_by_model = {model: Path(root) for model, root in (run.split("=", 1) for run in args.run)}
    candidates = [candidate for model, root in run_root_by_model.items() for candidate in showable_candidates(list(judged_candidates(model, root)))]
    sample = half_positive_sample(candidates, list(run_root_by_model), positive_levels, args.per_half, random.Random(args.seed))

    role_paragraph_and_rubric_by_role = {
        role: split_eval_prompt(json.loads((args.roles_dir / f"{role}.json").read_text())["eval_prompt"])
        for role in {candidate["role"] for candidate in sample}
    }
    shared_rubrics = {shared_rubric for _, shared_rubric in role_paragraph_and_rubric_by_role.values()}
    assert len(shared_rubrics) == 1, "roles disagree on the rubric after their role paragraph"

    item_ids = [f"item{position:03d}" for position in range(len(sample))]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "rubric.md").write_text(labeler_rubric(shared_rubrics.pop()))
    write_jsonl(args.out_dir / "items.jsonl", [
        to_item(item_id, candidate, role_paragraph_and_rubric_by_role[candidate["role"]][0]) for item_id, candidate in zip(item_ids, sample)
    ])
    key_records: list[InCharacterKeyRecord] = [
        {"item_id": item_id, "judge_label": candidate["judge_label"], "model": candidate["model"], "role": candidate["role"],
         "response_key": candidate["response_key"], "run_root": str(run_root_by_model[candidate["model"]])}
        for item_id, candidate in zip(item_ids, sample)
    ]
    write_jsonl(args.out_dir / "key.jsonl", key_records)


if __name__ == "__main__":
    main()
