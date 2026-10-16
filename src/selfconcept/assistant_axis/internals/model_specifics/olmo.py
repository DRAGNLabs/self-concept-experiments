from typing import Any

from selfconcept.assistant_axis.internals.model_specifics._shared import build_turn_spans_chatml, get_response_indices_chatml
from selfconcept.assistant_axis.internals.model_specifics.types import CoverallLayerGetter, ModelSpecifics
from selfconcept.common.generation.model_specifics import ForcedThinkCloseGenerationSpecifics
from selfconcept.common.hf_strong_types import ConversationTurn, HFTokenizer


class OlmoModelSpecifics(CoverallLayerGetter, ForcedThinkCloseGenerationSpecifics, ModelSpecifics):
    """This was built for OLMo 3. OLMo 3 uses a different chat template than earlier versions."""

    def get_response_indices(
        self,
        conversation: list[ConversationTurn],
        tokenizer: HFTokenizer,
        *,
        include_thinking: bool,
        **apply_chat_template_kwargs: Any
    ) -> list[list[int]]:
        return get_response_indices_chatml(conversation, tokenizer, include_thinking=include_thinking, **apply_chat_template_kwargs)

    def build_turn_spans(
        self,
        conversation: list[ConversationTurn],
        tokenizer: HFTokenizer,
        full_ids: list[int],
        *,
        include_thinking: bool,
        **apply_chat_template_kwargs
    ) -> tuple[list[int], list[dict[str, Any]]]:
        return build_turn_spans_chatml(conversation, tokenizer, full_ids, include_thinking=include_thinking, **apply_chat_template_kwargs)

 