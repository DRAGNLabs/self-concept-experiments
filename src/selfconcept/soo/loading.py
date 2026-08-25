"""Model loading that tolerates multimodal wrappers.

Text-only models (Mistral, Gemma-2, OLMo-2) load via AutoModelForCausalLM.
Multimodal releases such as Muse-Glimmer-30B register only under
AutoModelForImageTextToText; their ForConditionalGeneration class still
exposes a Llama-like text tower (model.language_model.layers with
q/k/v/o_proj), so the rest of the pipeline works unchanged once loaded.
"""

from transformers import AutoModelForCausalLM


def load_causal_lm(model_id, **kwargs):
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except ValueError:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
