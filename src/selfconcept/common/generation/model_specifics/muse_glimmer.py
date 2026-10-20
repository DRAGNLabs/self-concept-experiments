"""Muse Glimmer generation specifics: the ``to=self`` reasoning channel (ATEM protocol).

The Muse Glimmer chat template renders an assistant turn as
``<|start|>assistant to=self<|message|>{reasoning}<|eom|><|start|>assistant to=user<|message|>{content}<|eot|>``
(the reasoning turn only when the turn dict carries ``reasoning_content``) and ends the generation prompt at
``<|start|>assistant``; the model then writes `` to=self<|message|>`` itself. The template has no thinking flag:
``reasoning_strength`` (low / medium / high / xhigh, default high; a chat-template kwarg) only sets the depth, so a
direct answer is forced by appending `` to=user<|message|>`` to the prompt (``direct_answer_prefix_ids``), as the
SOO study's ``--force-user-channel`` did. Completions are kept raw (``skip_special_tokens = False``) and split here
into ``reasoning`` (the ``to=self`` bodies) and ``content`` (the last ``to=user`` body), following the rules of
vLLM's ``muse_glimmer`` reasoning parser.
"""
import re

from selfconcept.common.generation.model_specifics.types import GenerationSpecifics, TemplateWithoutThinkingFlag
from selfconcept.common.hf_strong_types import AllRoles, ConversationTurn, HFTokenizer, configure_call

START = "<|start|>"
MESSAGE = "<|message|>"
EOM = "<|eom|>"
EOT = "<|eot|>"
USER_CHANNEL_PREFIX = " to=user<|message|>"
CONTROL_TOKENS = (START, MESSAGE, EOM, EOT, "<|begin_of_text|>", "<|end_of_text|>")

# One channel of an assistant completion: "to=RECIPIENT<|message|>BODY", ended by <|eom|>, <|eot|>, the next
# <|start|>, the next channel header written without an end token (a model-side defect vLLM's parser also
# tolerates), or truncation.
CHANNEL = re.compile(
    r"to=([^\s<]+)<\|message\|>(.*?)(?=<\|eom\|>|<\|eot\|>|<\|start\|>|to=[^\s<]+<\|message\|>|$)",
    re.DOTALL,
)


class MuseGlimmerAssistantTurn(ConversationTurn, total=False):
    reasoning_content: str  # the chat template's field name for the to=self turn


def strip_muse_glimmer_tokens(text: str) -> str:
    for token in CONTROL_TOKENS:
        text = text.replace(token, "")
    return text


def split_muse_glimmer_completion(raw_completion: str) -> tuple[str | None, str]:
    """-> (reasoning, content): the to=self bodies joined (None when the model opened no reasoning channel) and the
    last to=user body. content is "" when the model never reached the user channel (e.g. reasoning truncated).
    A completion without any channel header is taken as a bare answer."""
    channels: list[tuple[str, str]] = CHANNEL.findall(raw_completion)
    if not channels:
        return None, strip_muse_glimmer_tokens(raw_completion).strip()
    reasoning = [body.strip() for recipient, body in channels if recipient == "self"]
    answers = [body for recipient, body in channels if recipient == "user"]
    content = answers[-1].strip() if answers else ""
    return ("\n\n".join(reasoning) if reasoning else None), content


def join_muse_glimmer_completion(reasoning: str | None, content: str) -> str:
    """Inverse of split_muse_glimmer_completion: the text the model writes after the ``<|start|>assistant``
    generation prompt (as the chat template renders a turn with ``reasoning_content``)."""
    if reasoning is None:
        return f"{USER_CHANNEL_PREFIX}{content}{EOT}"
    return f" to=self{MESSAGE}{reasoning}{EOM}{START}assistant{USER_CHANNEL_PREFIX}{content}{EOT}"


class MuseGlimmerGenerationSpecifics(TemplateWithoutThinkingFlag, GenerationSpecifics[AllRoles, MuseGlimmerAssistantTurn]):
    skip_special_tokens = False

    def direct_answer_prefix_ids(self, tokenizer: HFTokenizer) -> list[int]:
        return configure_call(tokenizer)(USER_CHANNEL_PREFIX, add_special_tokens=False)["input_ids"]

    def to_assistant_turn(self, completion: str) -> MuseGlimmerAssistantTurn:
        reasoning, content = split_muse_glimmer_completion(completion)
        turn: MuseGlimmerAssistantTurn = {"role": "assistant", "content": content}
        if reasoning is not None:
            turn["reasoning_content"] = reasoning
        return turn

    def to_completion(self, assistant_turn: MuseGlimmerAssistantTurn) -> str:
        return join_muse_glimmer_completion(assistant_turn.get("reasoning_content"), assistant_turn["content"])
