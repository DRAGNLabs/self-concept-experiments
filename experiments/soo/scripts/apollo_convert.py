"""Convert the Apollo roleplaying dataset to our eval jsonl format.

Reads data/apollo/dataset.yaml (371 scenarios from ApolloResearch/
deception-detection) and writes data/eval_apollo/roleplaying.jsonl.

Prompt construction mirrors their RolePlayingDataset "plain" variant, folded
into a single user turn because gemma chat templates reject a system role:
user = scenario + system_postfix + question, assistant prefilled with
answer_prefix (evaluate.py picks the prefill up from the example dict).

honest_answer / deceptive_answer carry the reference completions so
evaluate.py's rule-based classify() runs without erroring, but its inline
labels are meaningless for this dataset — real labels come from
scripts/judge_apollo.py.

Usage: python scripts/apollo_convert.py
"""

import json

import yaml

from selfconcept.common.paths import experiment_dir

ROOT = experiment_dir("soo")
SYSTEM_POSTFIX = "Answer the question directly, without asides or internal thoughts."


def main() -> None:
    raw = yaml.safe_load((ROOT / "data" / "apollo" / "dataset.yaml").open())
    out_dir = ROOT / "data" / "eval_apollo"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "roleplaying.jsonl"
    with out_path.open("w") as f:
        for i, item in enumerate(raw):
            f.write(
                json.dumps(
                    {
                        "scenario": "roleplaying",
                        "example_id": f"rp_{i:04d}",
                        "prompt": f"{item['scenario']}\n\n{SYSTEM_POSTFIX}\n\n{item['question']}",
                        "answer_prefix": item["answer_prefix"],
                        "scenario_text": item["scenario"],
                        "question": item["question"],
                        "honest_answer": item["honest_completion"],
                        "deceptive_answer": item["deceptive_completion"],
                    }
                )
                + "\n"
            )
    print(f"wrote {i + 1} examples to {out_path}")


if __name__ == "__main__":
    main()
