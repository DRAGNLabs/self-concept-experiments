from typing import Any

from selfconcept.assistant_axis.internals.conversation_utils import content_only_ids_and_offset_standard, find_subsequence
from selfconcept.assistant_axis.internals.model_specifics._shared import build_turn_spans_chatml, content_only_ids_and_offset_string_search, get_response_indices_chatml
from selfconcept.assistant_axis.internals.model_specifics.types import CoverallLayerGetter, ModelSpecifics
from selfconcept.common.hf_strong_types import AllRoles, Conversation, HFTokenizer, configure_apply_chat_template


class QwenModelSpecifics(CoverallLayerGetter, ModelSpecifics):
    def get_response_indices(self, conversation: Conversation, tokenizer: HFTokenizer, **apply_chat_template_kwargs: Any) -> list[list[int]]:
        return get_response_indices_chatml(conversation, tokenizer, **apply_chat_template_kwargs)

    def build_turn_spans(
        self,
        conversation: Conversation,
        tokenizer: HFTokenizer,
        full_ids: list[int],
        **apply_chat_template_kwargs,
    ) -> tuple[list[int], list[dict[str, Any]]]:
        return build_turn_spans_chatml(conversation, tokenizer, full_ids, **apply_chat_template_kwargs)

    def content_only_ids_and_offset(
        self,
        messages_before: Conversation,
        tokenizer: HFTokenizer,
        role: AllRoles,
        content: str,
        **chat_kwargs,
    ) -> tuple[list[int], int]:
        """Qwen-specific version that handles thinking tokens properly."""
        # For Qwen assistant turns, thinking tokens interfere even when disabled
        return content_only_ids_and_offset_string_search(messages_before, tokenizer, role, content, **chat_kwargs)


    def set_enable_thinking(self, old_chat_kwargs: dict[str, Any], enable_thinking: bool) -> dict[str, Any]:
        return { **old_chat_kwargs, "enable_thinking": enable_thinking }

    def thinking_close_ids(self, tokenizer: HFTokenizer) -> list[int]:
        return [] # handled by set_enable_thinking
