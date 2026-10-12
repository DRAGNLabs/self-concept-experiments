from typing import Any

import torch
from transformers import CompressedTensorsConfig, PreTrainedModel

from selfconcept.common.loading import load_causal_lm

ATTENTION_IMPLEMENTATION_BY_NAME_SUBSTRING = {"gpt-oss": "kernels-community/vllm-flash-attn3"}
QUANTIZATION_CONFIG_BY_NAME_SUBSTRING = {"glm-4.5-air-fp8": CompressedTensorsConfig(dequantize=True)}


def load_model(model_name: str) -> PreTrainedModel:
    """bf16 across the visible GPUs; gpt-oss keeps MXFP4 and GLM's FP8 checkpoint is dequantised."""
    model_kwargs: dict[str, Any] = {}
    for name_substring, attention_implementation in ATTENTION_IMPLEMENTATION_BY_NAME_SUBSTRING.items():
        if name_substring in model_name.lower():
            model_kwargs["attn_implementation"] = attention_implementation
    for name_substring, quantization_config in QUANTIZATION_CONFIG_BY_NAME_SUBSTRING.items():
        if name_substring in model_name.lower():
            model_kwargs["quantization_config"] = quantization_config
    model: PreTrainedModel = load_causal_lm(model_name, dtype=torch.bfloat16, device_map="auto", **model_kwargs)
    model.eval()
    return model


def peak_gpu_memory_gib_by_device() -> dict[int, float]:
    return {device: round(torch.cuda.max_memory_allocated(device) / 2**30, 1) for device in range(torch.cuda.device_count())}
