# Derived from safety-research/assistant-axis (https://github.com/safety-research/assistant-axis),
# MIT licensed. See the NOTICE file at the repository root for the full license text.
"""Model configuration lookup for computing axes (target layer, short name, ...).

For model loading, use ProbingModel from selfconcept.assistant_axis.internals instead.
"""


# PORT_ASSUMPTION[model-specific]: hardcoded per-model config. Models not listed here
# fall back to the AutoConfig inference in get_config (middle layer, name-substring
# short_name). target_layer is the recommended layer for axis computation (~middle);
# capping_* are only consumed by the (not-yet-ported) capping step.
MODEL_CONFIGS = {
    "google/gemma-2-27b-it": {
        "target_layer": 22,
        "total_layers": 46,
        "short_name": "Gemma",
    },
    "Qwen/Qwen3-32B": {
        "target_layer": 32,
        "total_layers": 64,
        "short_name": "Qwen",
        "capping_config": "qwen-3-32b/capping_config.pt",
        "capping_experiment": "layers_46:54-p0.25",
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        "target_layer": 40,
        "total_layers": 80,
        "short_name": "Llama",
        "capping_config": "llama-3.3-70b/capping_config.pt",
        "capping_experiment": "layers_56:72-p0.25",
    },
}


def get_config(model_name: str) -> dict:
    """
    Get configuration for a model.

    Args:
        model_name: HuggingFace model name

    Returns:
        Dict with target_layer, total_layers, and short_name.
        If model is not in known configs, infers values from model architecture.
    """
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name].copy()

    # Try to infer config from model
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        total_layers = config.num_hidden_layers
        target_layer = total_layers // 2  # Default to middle layer

        # PORT_ASSUMPTION[model-specific]: short_name inferred by substring match on the
        # model name; unknown families fall back to the first token of the repo name.
        model_lower = model_name.lower()
        if "gemma" in model_lower:
            short_name = "Gemma"
        elif "qwen" in model_lower:
            short_name = "Qwen"
        elif "llama" in model_lower:
            short_name = "Llama"
        elif "mistral" in model_lower:
            short_name = "Mistral"
        else:
            short_name = model_name.split("/")[-1].split("-")[0]

        return {
            "target_layer": target_layer,
            "total_layers": total_layers,
            "short_name": short_name,
        }
    except Exception as e:
        raise ValueError(f"Could not infer config for model {model_name}: {e}")
