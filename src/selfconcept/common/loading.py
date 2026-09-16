"""Model loading that tolerates multimodal wrappers.

Text-only models (Mistral, Gemma-2, OLMo-2) load via AutoModelForCausalLM.
Multimodal releases such as Muse-Glimmer-30B register only under
AutoModelForImageTextToText; their ForConditionalGeneration class still
exposes a Llama-like text tower (model.language_model.layers with
q/k/v/o_proj), so the rest of the pipeline works unchanged once loaded.

Qwen3.5/3.8 register a *text-only* class under AutoModelForCausalLM whose
parameter names do not match their multimodal checkpoints
(model.language_model.*), so any config with a text tower goes straight to
the image-text class, and loading is refused if weights would be left
randomly initialised (that would corrupt every downstream number silently).
"""

from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText


def load_causal_lm(model_id, **kwargs):
    config = AutoConfig.from_pretrained(model_id)
    cls = AutoModelForImageTextToText if getattr(config, "text_config", None) is not None else AutoModelForCausalLM
    try:
        model, info = cls.from_pretrained(model_id, output_loading_info=True, **kwargs)
    except ValueError:
        if cls is AutoModelForImageTextToText:
            raise
        model, info = AutoModelForImageTextToText.from_pretrained(model_id, output_loading_info=True, **kwargs)
    # A wrong-class load leaves every decoder layer uninitialised; tied
    # embeddings or an unused head are one key and never inside ".layers.".
    missing = [k for k in info.get("missing_keys", []) if ".layers." in k and "visual" not in k and not k.startswith("mtp.")]
    if missing:
        raise RuntimeError(
            f"{model_id}: {len(missing)} weights missing from the checkpoint would be randomly initialised "
            f"(first: {missing[:5]}); refusing to continue"
        )
    unexpected = info.get("unexpected_keys", [])
    if unexpected:
        print(f"load_causal_lm: {len(unexpected)} checkpoint tensors unused (e.g. {unexpected[:3]})", flush=True)
    return model
