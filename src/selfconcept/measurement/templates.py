"""Native reasoning-role boundaries, independent of SOO's thinking-off defaults."""

from dataclasses import dataclass
import re
from collections.abc import Sequence
from typing import Literal, TypedDict

from selfconcept.codebench.vllm_harmony import HARMONY_MESSAGE
from selfconcept.common.hf_strong_types import (
    configure_apply_chat_template, Conversation, HFTokenizer, OffsetMappingPresent)

type ModelFamily = Literal["qwen", "gemma", "muse", "olmo", "kimi", "gpt-oss"]
type CharSpan = tuple[int, int]


class ResponseSpans(TypedDict):
    cot: list[CharSpan]
    final: list[CharSpan]


@dataclass(frozen=True)
class ModelSpec:
    model: str
    revision: str
    family: ModelFamily
    gpus: int


MODEL_SPECS_BY_KEY: dict[str, ModelSpec] = {
    "gemma4-12b": ModelSpec("google/gemma-4-12B-it", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7", "gemma", 1),
    "qwen38-27b": ModelSpec("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", "qwen", 1),
    "gemma4-31b": ModelSpec("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475", "gemma", 2),
    "muse-30b": ModelSpec("meta-models/Muse-Glimmer-30B", "a4e59da52a7bc87ae7251dd5545c0dd437c44b68", "muse", 2),
}
# Kimi-Dev-72B uses a Qwen2 chat template with no native reasoning channel.
# The initial pilot's manually inserted Kimi thinking markers were unsupported;
# its failed probe is retained as an invalid setup, not a model-level result.

REASONING_MARKERS_BY_FAMILY: dict[ModelFamily, tuple[str, str]] = {
    "qwen": ("<think>", "</think>"), "olmo": ("<think>", "</think>"), "kimi": ("◁think▷", "◁/think▷"),
    "gemma": ("<|channel>thought\n", "<channel|>"),
    "muse": ("<|start|>assistant to=self<|message|>", "<|start|>assistant to=user<|message|>"),
}
MUSE_CHANNEL_PATTERN = re.compile(
    r"<\|start\|>assistant to=(self|user)<\|message\|>(.*?)(?=<\|eom\|>|<\|eot\|>|<\|start\|>|$)", re.S)


def reasoning_template_kwargs(family: ModelFamily) -> dict[str, bool | str]:
    if family == "qwen":
        return {"enable_thinking": True, "reasoning_effort": "medium"}
    if family == "gemma":
        return {"enable_thinking": True}
    if family == "muse":
        return {"reasoning_strength": "high", "current_date": "2026-09-22"}
    return {}


def render_generation_prompt(tokenizer: HFTokenizer, family: ModelFamily, messages: Conversation) -> str:
    # Never append answer_prefix to an open thought header: that suppresses or
    # contaminates reasoning. The runner supplies it as a user instruction.
    return configure_apply_chat_template(tokenizer).tokenize(False)(
        messages, add_generation_prompt=True, **reasoning_template_kwargs(family))


def muse_response_spans(prompt: str, raw_response: str) -> ResponseSpans:
    """Muse's generation header ends at 'assistant', before its recipient, so match over prompt and response together."""
    spans: ResponseSpans = {"cot": [], "final": []}
    for match in MUSE_CHANNEL_PATTERN.finditer(prompt + raw_response):
        content_start, content_end = match.span(2)
        if content_end > len(prompt):
            region: Literal["cot", "final"] = "cot" if match[1] == "self" else "final"
            spans[region].append((max(content_start - len(prompt), 0), content_end - len(prompt)))
    return spans


def harmony_response_spans(raw_response: str) -> ResponseSpans:
    spans: ResponseSpans = {"cot": [], "final": []}
    for match in HARMONY_MESSAGE.finditer(raw_response):
        if match[1] == "analysis":
            spans["cot"].append(match.span(2))
        elif match[1] == "final":
            spans["final"].append(match.span(2))
    return spans


def response_spans(prompt: str, raw_response: str, family: ModelFamily) -> ResponseSpans:
    """Character spans in the generated text. Open/unclosed thoughts have no answer.

    Retain special tokens while parsing; they are stripped only after boundaries
    are found.
    """
    if family == "muse":
        return muse_response_spans(prompt, raw_response)
    if family == "gpt-oss":
        return harmony_response_spans(raw_response)
    opening_marker, closing_marker = REASONING_MARKERS_BY_FAMILY[family]
    spans: ResponseSpans = {"cot": [], "final": []}
    # Qwen's <think> is usually in the prompt, not in generated tokens.
    in_thought = prompt.rfind(opening_marker) > prompt.rfind(closing_marker)
    final_start, thought_start = 0, 0
    for marker in re.finditer(f"{re.escape(opening_marker)}|{re.escape(closing_marker)}", raw_response):
        if marker[0] == opening_marker:
            if not in_thought and marker.start() > final_start:
                spans["final"].append((final_start, marker.start()))
            in_thought, thought_start = True, marker.end()
        else:
            if in_thought:
                spans["cot"].append((thought_start, marker.start()))
            in_thought, final_start = False, marker.end()
    if in_thought:
        spans["cot"].append((thought_start, len(raw_response)))
    elif final_start < len(raw_response):
        spans["final"].append((final_start, len(raw_response)))
    return spans


def final_answer_text(raw_response: str, spans: ResponseSpans, tokenizer: HFTokenizer) -> str:
    text = "".join(raw_response[start:end] for start, end in spans["final"])
    for special_token in tokenizer.all_special_tokens:
        text = text.replace(special_token, "")
    return text.strip()


def content_token_indices(offsets: OffsetMappingPresent, spans: list[CharSpan],
                          special_ids: Sequence[int] = (), token_ids: Sequence[int] = ()) -> list[int]:
    """Exclude tags and tokens straddling a content boundary."""
    special = set(special_ids)
    return [index for index, (start, end) in enumerate(offsets)
            if end > start and (not token_ids or token_ids[index] not in special)
            and any(start >= low and end <= high for low, high in spans)]
