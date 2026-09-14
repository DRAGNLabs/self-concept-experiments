from typing import Any

from selfconcept.assistant_axis.internals.conversation_utils import content_only_ids_and_offset_standard, longest_common_prefix_len
from selfconcept.assistant_axis.internals.model_specifics.types import CoverallLayerGetter, ModelSpecifics
from selfconcept.common.hf_strong_types import AllRoles, Conversation, HFTokenizer, _Conversation, configure_apply_chat_template, configure_call


class GemmaLlamaModelSpecifics(CoverallLayerGetter, ModelSpecifics): # TODO: technically not quite right for Llama, since llama supports all roles
    def get_response_indices(self, conversation: Conversation, tokenizer: HFTokenizer, **apply_chat_template_kwargs: Any) -> list[list[int]]:
        """Gemma/Llama-specific implementation using offset mapping approach."""
        all_turn_indices = []

        # Process conversation incrementally to find assistant response boundaries
        for i, turn in enumerate(conversation):
            if turn['role'] != 'assistant':
                continue

            # Get conversation up to but not including this assistant turn
            conversation_before = conversation[:i]

            # Get conversation up to and including this assistant turn
            conversation_including = conversation[:i+1]

            # Format and tokenize both versions
            if conversation_before:
                before_formatted = configure_apply_chat_template(tokenizer).tokenize(False)(
                    conversation_before, add_generation_prompt=True, **apply_chat_template_kwargs
                )
                before_tokens = tokenizer(before_formatted, add_special_tokens=False)
                before_length = len(before_tokens['input_ids'])
            else:
                before_length = 0

            including_formatted = configure_apply_chat_template(tokenizer).tokenize(False)(
                conversation_including, add_generation_prompt=False, **apply_chat_template_kwargs
            )
            including_tokens = tokenizer(including_formatted, add_special_tokens=False)
            including_length = len(including_tokens['input_ids'])

            # Find the actual content of this assistant response (excluding formatting tokens)
            assistant_content = turn['content'].strip()

            # Collect indices for this turn
            turn_indices = []

            # Find where the assistant content appears in the formatted text
            content_start_in_formatted = including_formatted.find(assistant_content)
            if content_start_in_formatted != -1:
                content_end_in_formatted = content_start_in_formatted + len(assistant_content)

                # Convert character positions to token indices using offset mapping
                tokens_with_offsets = configure_call(tokenizer).return_offsets_mapping(True)(including_formatted, add_special_tokens=False)
                offset_mapping = tokens_with_offsets['offset_mapping']

                # Find tokens that overlap with the assistant content
                for token_idx, (start_char, end_char) in enumerate(offset_mapping):
                    if (start_char >= content_start_in_formatted and start_char < content_end_in_formatted) or \
                       (end_char > content_start_in_formatted and end_char <= content_end_in_formatted) or \
                       (start_char < content_start_in_formatted and end_char > content_end_in_formatted):
                        turn_indices.append(token_idx)
            else:
                # Fallback to original method if content not found
                assistant_start = before_length
                assistant_end = including_length
                turn_indices.extend(range(assistant_start, assistant_end))

            # Store indices based on per_turn flag
            all_turn_indices.append(turn_indices)

        return all_turn_indices

    def build_turn_spans(
        self,
        conversation: _Conversation,
        tokenizer: HFTokenizer,
        full_ids: list[int],
        **apply_chat_template_kwargs,
    ) -> tuple[list[int], list[dict[str, Any]]]:
        spans = []
        msgs_before = []
        turn_idx = 0

        for msg in conversation:
            role = msg["role"]
            text = msg.get("content", "")

            if role == "system":
                msgs_before.append(msg)
                continue

            content_ids, start_in_delta = self.content_only_ids_and_offset(
                msgs_before, tokenizer, role, text, **apply_chat_template_kwargs
            )

            # Standard approach for non-Qwen models
            # Calculate absolute start based on the empty message template
            msgs_empty_for_this = msgs_before + [{"role": role, "content": ""}]
            ids_empty_full = configure_apply_chat_template(tokenizer).tokenize(True)(
                msgs_empty_for_this, add_generation_prompt=False, **apply_chat_template_kwargs
            )["input_ids"]

            # Find where the content appears in the full sequence
            ids_full_for_this = configure_apply_chat_template(tokenizer).tokenize(True)(
                msgs_before + [{"role": role, "content": text}], add_generation_prompt=False, **apply_chat_template_kwargs
            )["input_ids"]

            pref_len = longest_common_prefix_len(ids_full_for_this, ids_empty_full)
            abs_start = pref_len + start_in_delta
            abs_end = abs_start + len(content_ids)

            spans.append({
                "turn": turn_idx,
                "role": role,
                "start": abs_start,
                "end": abs_end,   # exclusive
                "n_tokens": len(content_ids),
                "text": text,
            })
            msgs_before.append(msg)
            turn_idx += 1

        return full_ids, spans

    def content_only_ids_and_offset(
        self,
        messages_before: _Conversation,
        tokenizer: HFTokenizer,
        role: AllRoles,
        content: str,
        **apply_chat_template_kwargs,
    ) -> tuple[list[int], int]:
        return content_only_ids_and_offset_standard(messages_before, tokenizer, role, content, **apply_chat_template_kwargs)


    def set_enable_thinking(self, old_chat_kwargs: dict[str, Any], enable_thinking: bool) -> dict[str, Any]:
        if enable_thinking:
            raise NotImplementedError("thinking currently not supported for Gemma type models")
        else:
            return old_chat_kwargs
