from typing import Any, Protocol, cast

import torch.nn as nn

from selfconcept.assistant_axis.internals.conversation_utils import ContentOnlyIdsAndOffsetFn
from selfconcept.common.hf_strong_types import AllRoles, HFTokenizer, _Conversation


class ModelSpecifics[RoleT: AllRoles = AllRoles](ContentOnlyIdsAndOffsetFn[RoleT], Protocol): # TODO: include tokenizer?
    def get_response_indices(self, conversation: _Conversation[RoleT], tokenizer: HFTokenizer[RoleT], **apply_chat_template_kwargs: Any) -> list[list[int]]: ... # TODO: figure out what to do with apply_chat_template_kwargs

    def build_turn_spans(
        self,
        conversation: _Conversation[RoleT],
        tokenizer: HFTokenizer[RoleT],
        full_ids: list[int],
        **apply_chat_template_kwargs,
    ) -> tuple[list[int], list[dict[str, Any]]]: ...

    def set_enable_thinking(
        self,
        old_chat_kwargs: dict[str, Any],
        enable_thinking: bool,
    ) -> dict[str, Any]: ...

    def thinking_close_ids(self, tokenizer: HFTokenizer[RoleT]) -> list[int]: ...

    def get_layers(self, model: Any) -> nn.ModuleList: ...

class CoverallLayerGetter:
    def get_layers(self, model: Any) -> nn.ModuleList:
        """
        Get the transformer layers from the model, handling different architectures.

        Returns:
            The layers object (usually a ModuleList) that can be indexed and has len()

        Raises:
            AttributeError: If no layers can be found with helpful error message
        """
        # Try common paths for transformer layers
        layer_paths = [
            ('model.model.layers', lambda m: m.model.layers),  # Standard language models (Llama, Gemma 2, Qwen, etc.)
            ('model.language_model.layers', lambda m: m.language_model.layers),  # Vision-language models (Gemma 3, LLaVA, etc.)
            ('model.transformer.h', lambda m: m.transformer.h),  # GPT-style models
            ('model.transformer.layers', lambda m: m.transformer.layers),  # Some transformer variants
            ('model.gpt_neox.layers', lambda m: m.gpt_neox.layers),  # GPT-NeoX models
        ]

        for path_name, path_func in layer_paths:
            try:
                layers = path_func(model)
                if layers is not None and hasattr(layers, '__len__') and len(layers) > 0:
                    return cast(nn.ModuleList, layers)
            except AttributeError:
                continue

        # If we get here, no layers were found
        model_class = type(model).__name__
        model_name = getattr(model, 'name_or_path', 'Unknown')

        # Provide specific guidance for known cases
        error_msg = f"Could not find transformer layers for model '{model_name}' (class: {model_class}). "

        if 'gemma' in model_name.lower() and '3' in model_name:
            error_msg += "For Gemma 3 vision models, try loading with Gemma3ForConditionalGeneration instead."
        elif 'llava' in model_name.lower():
            error_msg += "For LLaVA models, layers should be at model.language_model.layers."
        else:
            # Show what paths were tried
            tried_paths = [path_name for path_name, _ in layer_paths]
            error_msg += f"Tried paths: {tried_paths}"

        raise AttributeError(error_msg)
