from typing import Any

from jaxtyping import Float
import torch

from selfconcept.assistant_axis.internals.conversation_utils import content_only_ids_and_offset_standard, find_subsequence
from selfconcept.assistant_axis.internals.model_specifics._shared import build_turn_spans_chatml, get_response_indices_chatml
from selfconcept.assistant_axis.internals.model_specifics.types import CoverallLayerGetter, ModelSpecifics
from selfconcept.assistant_axis.internals.model_specifics.unembedding import CheckpointTensorLoader, UnembeddingSpecifics
from selfconcept.common.generation.model_specifics import QwenThinkGenerationSpecifics, ReasoningContentAssistantTurn
from selfconcept.common.hf_strong_types import AllRoles, Conversation, HFTokenizer, configure_apply_chat_template


class QwenModelSpecifics(CoverallLayerGetter, QwenThinkGenerationSpecifics, ModelSpecifics[AllRoles, ReasoningContentAssistantTurn]):
    def get_response_indices(self, conversation: Conversation, tokenizer: HFTokenizer, *, include_thinking: bool, **apply_chat_template_kwargs: Any) -> list[list[int]]:
        return get_response_indices_chatml(conversation, tokenizer, include_thinking=include_thinking, **apply_chat_template_kwargs)

    def build_turn_spans(
        self,
        conversation: Conversation,
        tokenizer: HFTokenizer,
        full_ids: list[int],
        *,
        include_thinking: bool,
        **apply_chat_template_kwargs,
    ) -> tuple[list[int], list[dict[str, Any]]]:
        return build_turn_spans_chatml(conversation, tokenizer, full_ids, include_thinking=include_thinking, **apply_chat_template_kwargs)


class Qwen35ModelSpecifics(QwenModelSpecifics, UnembeddingSpecifics):
    """Qwen 3.5-architecture checkpoints (model_type qwen3_5, e.g. Qwen3.8): a multimodal layout and a centered final norm."""

    def gain_scaled_unembedding(self, load_tensor: CheckpointTensorLoader) -> Float[torch.Tensor, "vocab hidden"]:
        """Qwen3_5RMSNorm scales by (1 + weight)."""
        return load_tensor("lm_head.weight") * (1 + load_tensor("model.language_model.norm.weight"))
