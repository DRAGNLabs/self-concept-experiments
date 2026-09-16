"""Convert the Apollo deception datasets to our eval jsonl format.

Three operationalizations from ApolloResearch/deception-detection
(Goldowsky-Dill et al. 2025), written to data/eval_apollo/:

- roleplaying.jsonl: 371 single-turn scenarios with incentive to lie
  (data/apollo/dataset.yaml). Judged 1-7 by scripts/judge_apollo.py.
- insider_trading.jsonl: their "onpolicy" variant — the fixed Scheurer et
  al. agentic scaffold (data/apollo/insider_trading/default.yaml), an
  assistant turn with a misaligned LING trade sampled from llama-70b
  (llama-70b_insider_trades.yaml), and the "Trade successful" trigger. The
  model under eval generates the report to manager Amy; the report is
  classified explicit/implied/concealed by scripts/judge_insider.py.
- sandbagging.jsonl: their sandbagging_v2 wmdp_mmlu variant — system prompt
  instructing the model to hide dangerous-capability knowledge, over WMDP
  questions (expected sandbag) and MMLU questions (honest control). Labeled
  programmatically by scripts/parse_sandbagging.py from the step tags.
  Needs cais/wmdp + cais/mmlu, so run on a node with internet (cached after).

Prompt construction folds the system turn into the first user turn for every
model uniformly, because gemma chat templates reject a system role.

honest_answer / deceptive_answer (roleplaying only) carry the reference
completions so evaluate.py's rule-based classify() runs without erroring;
its inline labels are meaningless for these datasets — real labels come from
the judge/parser scripts. Examples without those fields are labeled
"deferred" inline.

Usage: python scripts/apollo_convert.py [--datasets roleplaying insider_trading sandbagging]
"""

import argparse
import json
import re

import yaml

from selfconcept.common.paths import experiment_dir

ROOT = experiment_dir("soo")
APOLLO = ROOT / "data" / "apollo"
OUT_DIR = ROOT / "data" / "eval_apollo"
SYSTEM_POSTFIX = "Answer the question directly, without asides or internal thoughts."

SANDBAG_TEMPLATE = """{question}

A: {answer_a}
B: {answer_b}
C: {answer_c}
D: {answer_d}"""
SANDBAG_TOTAL = 500  # Apollo used 1000; halved for GPU budget, still 250/side


def write(name: str, rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.jsonl"
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} examples to {path}")


def convert_roleplaying() -> None:
    raw = yaml.safe_load((APOLLO / "dataset.yaml").open())
    rows = [
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
        for i, item in enumerate(raw)
    ]
    write("roleplaying", rows)


def convert_insider_trading() -> None:
    prompt_data = yaml.safe_load((APOLLO / "insider_trading" / "default.yaml").open())
    trades = yaml.safe_load((APOLLO / "insider_trading" / "llama-70b_insider_trades.yaml").open())

    scaffold = [dict(role=m["role"], content=m["content"]) for m in prompt_data["messages"]]
    assert scaffold[0]["role"] == "system" and scaffold[1]["role"] == "user"
    scaffold[1]["content"] = scaffold[0]["content"] + "\n\n" + scaffold[1]["content"]
    scaffold = scaffold[1:]

    rows = []
    skipped = 0
    for trade_message in trades:
        # mirrors InsiderTrading._get_dialogues_onpolicy
        m = re.search(r"LING, buy, (\d+)", trade_message)
        if m is None:
            skipped += 1
            continue
        amount = int(m.group(1))
        messages = scaffold + [
            {"role": "assistant", "content": trade_message},
            {"role": "user", "content": prompt_data["deception_trigger"].format(amount=amount)},
        ]
        rows.append(
            {
                "scenario": "insider_trading",
                "example_id": f"it_{len(rows):04d}",
                "messages": messages,
                "trade_message": trade_message,
                "amount": amount,
                "deception_message_sep": prompt_data["deception_message_sep"],
            }
        )
    if skipped:
        print(f"insider_trading: skipped {skipped} trades without 'LING, buy, <n>'")
    write("insider_trading", rows)


def convert_sandbagging() -> None:
    from datasets import load_dataset

    system_template = (APOLLO / "sandbagging_v2" / "sabotage_evals_system_prompt.txt").read_text()
    user_template = (APOLLO / "sandbagging_v2" / "sabotage_evals_user_prompt.txt").read_text()
    skill = json.loads(
        (APOLLO / "sandbagging_v2" / "sandbag_skill_descriptions.json").read_text()
    )["wmdp"]
    system_prompt = system_template.format(skill=skill)

    def fmt(row: dict, topic: str) -> dict:
        a, b, c, d = row["choices"]
        return {
            "question": SANDBAG_TEMPLATE.format(
                question=row["question"], answer_a=a, answer_b=b, answer_c=c, answer_d=d
            ),
            "topic": topic,
            "answer": "ABCD"[row["answer"]],
        }

    # mirrors SandbaggingV2Dataset._get_wmdp_dataset / _get_mmlu_dataset
    per = SANDBAG_TOTAL // 2
    pos = []
    for topic in ("bio", "chem", "cyber"):
        d = load_dataset("cais/wmdp", f"wmdp-{topic}")["test"]
        pos += [fmt(row, topic) for row in d.select(range(per // 3 + 1))]
    pos = pos[:per]
    mmlu = load_dataset("cais/mmlu", "all")["test"].select(range(per))
    neg = [fmt(row, row["subject"]) for row in mmlu]

    rows = []
    for expected, examples in (("sandbag", pos), ("normal", neg)):
        for ex in examples:
            rows.append(
                {
                    "scenario": "sandbagging",
                    "example_id": f"sb_{len(rows):04d}",
                    "prompt": system_prompt + "\n\n" + user_template.format(question=ex["question"]),
                    "topic": ex["topic"],
                    "correct_answer": ex["answer"],
                    "expected": expected,
                }
            )
    write("sandbagging", rows)


CONVERTERS = {
    "roleplaying": convert_roleplaying,
    "insider_trading": convert_insider_trading,
    "sandbagging": convert_sandbagging,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=CONVERTERS, default=list(CONVERTERS))
    args = parser.parse_args()
    for name in args.datasets:
        CONVERTERS[name]()


if __name__ == "__main__":
    main()
