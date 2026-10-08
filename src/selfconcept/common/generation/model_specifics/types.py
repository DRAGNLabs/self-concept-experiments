from typing import Any, Protocol

from selfconcept.common.hf_strong_types import AllRoles, ConversationTurn, HFTokenizer


class GenerationSpecifics[RoleT: AllRoles = AllRoles, AssistantTurnT: ConversationTurn = ConversationTurn](Protocol):
    skip_special_tokens: bool = True

    def set_thinking_flag(
        self,
        old_chat_kwargs: dict[str, Any],
        enable_thinking: bool,
    ) -> dict[str, Any]:
        """Fills only the chat-template kwargs that the model's template itself needs to turn thinking on or off."""
        ...

    def direct_answer_prefix_ids(self, tokenizer: HFTokenizer[RoleT]) -> list[int]:
        """Appended after the generation prompt so that the next token is the answer itself. Only needed for models
        whose templates cannot turn thinking off themselves (via a flag, or a pre-closed think block)."""
        return []

    def to_assistant_turn(self, completion: str) -> AssistantTurnT: ...

class TemplateWithoutThinkingFlag:
    def set_thinking_flag(self, old_chat_kwargs: dict[str, Any], enable_thinking: bool) -> dict[str, Any]:
        return old_chat_kwargs

class HFStyleAssistantTurn:
    def to_assistant_turn(self, completion: str) -> ConversationTurn:
        return {"role": "assistant", "content": completion}
