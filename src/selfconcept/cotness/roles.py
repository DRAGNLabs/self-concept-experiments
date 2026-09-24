"""Native reasoning-role boundaries, independent of SOO's thinking-off defaults."""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ModelSpec:
    model: str
    revision: str
    family: str
    gpus: int


MODELS = {
    "gemma4-12b": ModelSpec("google/gemma-4-12B-it", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7", "gemma", 1),
    "qwen38-27b": ModelSpec("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", "qwen", 1),
    "gemma4-31b": ModelSpec("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475", "gemma", 2),
    "muse-30b": ModelSpec("meta-models/Muse-Glimmer-30B", "a4e59da52a7bc87ae7251dd5545c0dd437c44b68", "muse", 2),
}
# Kimi-Dev-72B uses a Qwen2 chat template with no native reasoning channel.
# The initial pilot's manually inserted Kimi thinking markers were unsupported;
# its failed probe is retained as an invalid setup, not a model-level result.
ROLES = ("user", "cot", "assistant")


def template_kwargs(family):
    if family == "qwen":
        return {"enable_thinking": True, "reasoning_effort": "medium"}
    if family == "gemma":
        return {"enable_thinking": True}
    if family == "muse":
        return {"reasoning_strength": "high", "current_date": "2026-09-22"}
    return {}


def markers(family):
    return {"qwen": ("<think>", "</think>"), "kimi": ("◁think▷", "◁/think▷"),
            "gemma": ("<|channel>thought\n", "<channel|>"),
            "muse": ("<|start|>assistant to=self<|message|>",
                     "<|start|>assistant to=user<|message|>")}[family]


def render_probe(tokenizer, family, text, role, context="A neutral document follows."):
    """Render identical content in three native roles; return its character span.

    A fixed prior exchange supplies the same context for every role copy. The
    assistant role uses an empty thought block where the native template does.
    """
    if role not in ROLES:
        raise ValueError(role)
    if family == "kimi":
        raise ValueError("Kimi-Dev-72B has no verified native reasoning-role template")
    # Cropping to a token budget can end on whitespace; native templates trim
    # that whitespace. Apply the same normalization to every role copy.
    text = text.strip()
    if not text:
        raise ValueError("Probe content must be nonempty")
    history = [{"role": "user", "content": context}, {"role": "assistant", "content": "Understood."}]
    if role == "user":
        messages = history + [{"role": "user", "content": text}]
    else:
        messages = history + [{"role": "user", "content": "Continue."}]
        message = {"role": "assistant", "content": text if role == "assistant" else ""}
        if role == "cot":
            message["reasoning_content"] = text
        messages.append(message)
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, **template_kwargs(family))
    if rendered.count(text) != 1:
        raise ValueError(f"Template lost or duplicated {role} content")
    start = rendered.index(text)
    return rendered, (start, start + len(text))


def render_generation(tokenizer, family, messages):
    # Never append answer_prefix to an open thought header: that suppresses or
    # contaminates reasoning. The runner supplies it as a user instruction.
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         **template_kwargs(family))


def response_spans(prompt, raw, family):
    """Character spans in the generated text. Open/unclosed thoughts have no answer.

    Retain special tokens while parsing; they are stripped only after boundaries
    are found. Muse's generation header ends at 'assistant', before its recipient.
    """
    op, cl = markers(family)
    if family == "muse":
        combined = prompt + raw
        pattern = re.compile(r"<\|start\|>assistant to=(self|user)<\|message\|>(.*?)(?=<\|eom\|>|<\|eot\|>|<\|start\|>|$)", re.S)
        spans = {"cot": [], "final": []}
        for m in pattern.finditer(combined):
            a, b = m.span(2)
            if b > len(prompt):
                spans["cot" if m[1] == "self" else "final"].append((max(a-len(prompt), 0), b-len(prompt)))
        return spans
    spans = {"cot": [], "final": []}
    # Qwen's <think> is usually in the prompt, not in generated tokens.
    active = prompt.rfind(op) > prompt.rfind(cl)
    cursor, start = 0, 0
    for match in re.finditer(f"{re.escape(op)}|{re.escape(cl)}", raw):
        if match[0] == op:
            if not active and match.start() > cursor:
                spans["final"].append((cursor, match.start()))
            active, start = True, match.end()
        else:
            if active:
                spans["cot"].append((start, match.start()))
            active, cursor = False, match.end()
    if active:
        spans["cot"].append((start, len(raw)))
    elif cursor < len(raw):
        spans["final"].append((cursor, len(raw)))
    return spans


def clean_final(raw, spans, tokenizer):
    text = "".join(raw[a:b] for a, b in spans["final"])
    for token in tokenizer.all_special_tokens:
        text = text.replace(token, "")
    return text.strip()


def content_indices(offsets, spans, special_ids=(), ids=()):
    """Exclude tags and tokens straddling a content boundary."""
    special = set(special_ids)
    return [i for i, (a, b) in enumerate(offsets)
            if b > a and (not ids or ids[i] not in special)
            and any(a >= lo and b <= hi for lo, hi in spans)]
