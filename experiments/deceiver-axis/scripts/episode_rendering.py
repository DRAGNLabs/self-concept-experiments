"""Render finished reward-hacking episodes turn by turn exactly as the model saw them, for teacher forcing.

Brought over from Jansen's axis_acts_extract.py (branch schemer-axis): history with the earlier finals only, the chat
template's own rendering, then this turn's reasoning channel and final in the family's raw completion markup.
"""
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from jaxtyping import Int8
from transformers import PreTrainedTokenizerBase

import make_dossiers

Family = Literal["gpt-oss", "gemma4", "muse-glimmer", "qwen3.5"]
OTHER_SEGMENT, REASONING_SEGMENT, ANSWER_SEGMENT = 0, 1, 2

# family -> (reasoning opener, reasoning closer + final opener, end of turn). Qwen's generation prompt already ends in
# "<think>\n" when thinking is on, so its opener is empty.
COMPLETION_MARKUP_BY_FAMILY: dict[Family, tuple[str, str, str]] = {
    "gpt-oss": ("<|channel|>analysis<|message|>", "<|end|><|start|>assistant<|channel|>final<|message|>", "<|return|>"),
    "gemma4": ("<|channel>thought\n", "\n<channel|>", ""),
    "muse-glimmer": (" to=self<|message|>", "<|eom|><|start|>assistant to=user<|message|>", "<|eot|>"),
    "qwen3.5": ("", "\n</think>\n\n", "<|im_end|>"),
}
# Qwen has no entry: its generation prompt has already opened the thought.
DIRECT_ANSWER_OPENER_BY_FAMILY: dict[Family, str] = {"gpt-oss": "<|channel|>final<|message|>", "gemma4": "", "muse-glimmer": " to=user<|message|>"}

type TokenIds = list[int]
type Segments = Int8[np.ndarray, "tokens"]


def render_turn(tokenizer: PreTrainedTokenizerBase, family: Family, history: list[dict[str, str]], thought: str, final: str,
                chat_kwargs: dict[str, Any]) -> tuple[TokenIds, Segments]:
    def tokenize(text: str) -> TokenIds:
        return tokenizer(text, add_special_tokens=False)["input_ids"] if text else []

    opener, closer, end = COMPLETION_MARKUP_BY_FAMILY[family]
    prompt = list(tokenizer.apply_chat_template(history, tokenize=True, add_generation_prompt=True, return_dict=False, **chat_kwargs))
    if not thought and family in DIRECT_ANSWER_OPENER_BY_FAMILY:
        pieces = [(prompt + tokenize(DIRECT_ANSWER_OPENER_BY_FAMILY[family]), OTHER_SEGMENT), (tokenize(final), ANSWER_SEGMENT), (tokenize(end), OTHER_SEGMENT)]
    elif not final:  # the reasoning never closed (stopped or cut off mid-thought)
        pieces = [(prompt + tokenize(opener), OTHER_SEGMENT), (tokenize(thought), REASONING_SEGMENT)]
    else:
        pieces = [(prompt + tokenize(opener), OTHER_SEGMENT), (tokenize(thought), REASONING_SEGMENT), (tokenize(closer), OTHER_SEGMENT),
                  (tokenize(final), ANSWER_SEGMENT), (tokenize(end), OTHER_SEGMENT)]
    ids = [token for piece, _ in pieces for token in piece]
    segments = np.concatenate([np.full(len(piece), segment, dtype=np.int8) for piece, segment in pieces])
    return ids, segments


def episode_turns(tokenizer: PreTrainedTokenizerBase, family: Family, example: dict, record: dict, reasoning_by_turn: dict[int, str],
                  max_length: int, chat_kwargs: dict[str, Any]) -> tuple[list[tuple[TokenIds, Segments, int]], list[int], list[int]]:
    """(capped ids, segments, turn index) per model turn; the turns whose reasoning is missing; the uncapped turn lengths."""
    history: list[dict[str, str]] = []
    turns, missing_reasoning_turns, full_lengths = [], [], []
    for role, text in make_dossiers.conversation(example, record):
        if role == "user":
            history.append({"role": "user", "content": text})
            continue
        turn = int(text)
        final = record["attempts"][turn].get("completion") or ""
        thought = reasoning_by_turn.get(turn)
        if thought is None:
            missing_reasoning_turns.append(turn)
            thought = ""
        ids, segments = render_turn(tokenizer, family, history, thought, final, chat_kwargs)
        full_lengths.append(len(ids))
        turns.append((ids[:max_length], segments[:max_length], turn))
        history.append({"role": "assistant", "content": final})
    return turns, missing_reasoning_turns, full_lengths


def load_reasoning_sidecar(run_dir: Path, tag: str) -> dict[str, dict[int, str]]:
    reasoning_by_turn_by_example: dict[str, dict[int, str]] = {}
    for path in run_dir.glob(f"{tag}_*_reasoning.jsonl"):
        for line in path.open():
            entry = json.loads(line)
            reasoning_by_turn_by_example.setdefault(entry["example_id"], {})[int(entry["turn"])] = entry["reasoning"]
    return reasoning_by_turn_by_example
