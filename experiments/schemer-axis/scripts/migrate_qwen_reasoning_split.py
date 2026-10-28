"""Rewrite thinking-on Qwen role responses saved with `reasoning</think>\\n\\nanswer` in content into the
content + reasoning_content shape of QwenThinkGenerationSpecifics, without regenerating.

Each split turn must re-render through the chat template to the generation-time text (prompt + raw completion). The
template trims content, so a re-render may differ in whitespace only; those turns are counted. A completion with no
</think> hit the token cap mid-reasoning; it becomes reasoning only.

    python scripts/migrate_qwen_reasoning_split.py <old>/responses <new>/responses [--model Qwen/Qwen3.8-27B]
"""
import argparse
import json
from pathlib import Path
from typing import TypedDict, cast

from transformers import AutoTokenizer

from selfconcept.common.generation.model_specifics import QwenAssistantTurn, QwenThinkGenerationSpecifics
from selfconcept.common.generation.model_specifics.qwen_think import THINK_CLOSE
from selfconcept.common.hf_strong_types import Conversation, HFTokenizer, configure_apply_chat_template
from selfconcept.transcript_generation.vllm_generation import chat_prompt_text

TURN_END = "<|im_end|>\n"


def split_turn(raw_completion: str) -> QwenAssistantTurn:
    if THINK_CLOSE not in raw_completion:
        return {"role": "assistant", "content": "", "reasoning_content": raw_completion.strip()}
    return QwenThinkGenerationSpecifics().to_assistant_turn(raw_completion)


class MigrationCounts(TypedDict):
    unclosed: int
    whitespace_only_rerender_diff: int


def rerenders_generation_exactly(tokenizer: HFTokenizer, model_name: str, conversation: Conversation, raw_completion: str) -> bool:
    """Raises unless the re-render matches the generation-time text up to whitespace."""
    prompt = chat_prompt_text(tokenizer, model_name, conversation[:-1], True, {})
    rendered = configure_apply_chat_template(tokenizer).tokenize(False)(conversation, enable_thinking=True)
    expected = prompt + raw_completion.rstrip() + TURN_END
    if rendered.split() != expected.split():
        raise ValueError(f"re-render differs from generation-time text:\n{rendered[len(prompt):len(prompt) + 300]!r}")
    return rendered == expected


def migrate_role_file(tokenizer: HFTokenizer, model_name: str, old_path: Path, new_path: Path) -> MigrationCounts:
    counts: MigrationCounts = {"unclosed": 0, "whitespace_only_rerender_diff": 0}
    with old_path.open() as old_file, new_path.open("w") as new_file:
        for line in old_file:
            record = json.loads(line)
            *prompt_turns, assistant_turn = record["conversation"]
            assert assistant_turn["role"] == "assistant" and "reasoning_content" not in assistant_turn
            raw_completion = assistant_turn["content"]
            record["conversation"] = [*prompt_turns, split_turn(raw_completion)]
            if THINK_CLOSE not in raw_completion:
                counts["unclosed"] += 1
            elif not rerenders_generation_exactly(tokenizer, model_name, cast(Conversation, record["conversation"]), raw_completion):
                counts["whitespace_only_rerender_diff"] += 1
            new_file.write(json.dumps(record) + "\n")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("old_responses_dir", type=Path)
    parser.add_argument("new_responses_dir", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    args = parser.parse_args()

    tokenizer = cast(HFTokenizer, AutoTokenizer.from_pretrained(args.model))
    args.new_responses_dir.mkdir(parents=True, exist_ok=False)
    for old_path in sorted(args.old_responses_dir.glob("*.jsonl")):
        print(old_path.name, migrate_role_file(tokenizer, args.model, old_path, args.new_responses_dir / old_path.name))


if __name__ == "__main__":
    main()
