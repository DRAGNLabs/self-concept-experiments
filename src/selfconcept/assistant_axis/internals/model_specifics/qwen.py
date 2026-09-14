from typing import Any

from selfconcept.assistant_axis.internals.conversation_utils import content_only_ids_and_offset_standard, find_subsequence
from selfconcept.assistant_axis.internals.model_specifics._shared import build_turn_spans_chatml, get_response_indices_chatml
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
        if role == "assistant":
            # Find where content appears in the full tokenized conversation
            msgs_full = messages_before + [{"role": role, "content": content}]
            ids_full = configure_apply_chat_template(tokenizer).tokenize(True)(
                msgs_full, add_generation_prompt=False, **chat_kwargs
            )["input_ids"]

            # Find the content tokens in the full sequence
            plain = tokenizer(content, add_special_tokens=False).input_ids
            content_start = find_subsequence(ids_full, plain)

            if content_start != -1:
                # Calculate offset from the beginning of the conversation
                if messages_before:
                    ids_before = configure_apply_chat_template(tokenizer).tokenize(True)(
                        messages_before, add_generation_prompt=False, **chat_kwargs
                    )["input_ids"]
                    prefix_len = len(ids_before)
                else:
                    prefix_len = 0

                start_in_delta = content_start - prefix_len
                return plain, max(0, start_in_delta)

        # Fall back to standard approach for user turns or if assistant approach fails
        return content_only_ids_and_offset_standard(messages_before, tokenizer, role, content, **chat_kwargs)

    def set_enable_thinking(self, old_chat_kwargs: dict[str, Any], enable_thinking: bool) -> dict[str, Any]:
        return { **old_chat_kwargs, "enable_thinking": enable_thinking }
