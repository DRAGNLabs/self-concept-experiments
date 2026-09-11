"""ProbingModel - Wraps HuggingFace model with utilities for activation extraction."""

from __future__ import annotations

from typing import Any, cast
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


class ProbingModel:
    """
    Wraps a HuggingFace model and tokenizer with helper methods for generation
    and activation extraction.

    This is the central object you pass around instead of (model, tokenizer) tuples.
    """

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        max_memory_per_gpu: dict[int, str] | None = None,
        chat_model_name: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize and load a HuggingFace model and tokenizer.

        Args:
            model_name: HuggingFace model identifier for the base model
            device: Device specification - can be:
                - None: use all available GPUs with device_map="auto"
                - "cuda:X": use single GPU (will auto-shard if model is too large)
                - dict: custom device_map
            max_memory_per_gpu: Optional dict mapping GPU ids to max memory (e.g. {0: "40GiB", 1: "40GiB"})
            chat_model_name: Optional HuggingFace model identifier for tokenizer (if different from base model)
            dtype: Data type for model weights (default: torch.bfloat16)
        """
        self.model_name = model_name
        self.chat_model_name = chat_model_name
        self.dtype = dtype

        # Load tokenizer from chat_model_name if provided, otherwise from model_name
        tokenizer_source = chat_model_name if chat_model_name else model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Build model loading kwargs
        model_kwargs: dict[str, Any] = {
            "dtype": dtype,
        }

        if max_memory_per_gpu is not None:
            # Use custom memory limits (for multi-worker setups)
            model_kwargs["device_map"] = "auto"
            model_kwargs["max_memory"] = max_memory_per_gpu
        elif device is None or device == "auto":
            # Use all available GPUs automatically
            model_kwargs["device_map"] = "auto"
        elif isinstance(device, dict):
            # Custom device map provided
            model_kwargs["device_map"] = device
        elif isinstance(device, str) and device.startswith("cuda:"):
            # Single GPU specified - try to use it, but allow sharding if needed
            model_kwargs["device_map"] = "auto"
            gpu_id = int(device.split(":")[-1])
            model_kwargs["max_memory"] = {gpu_id: "139GiB"}
            # Set other GPUs to 0 to prevent usage
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    if i != gpu_id and i not in model_kwargs["max_memory"]:
                        model_kwargs["max_memory"][i] = "0GiB"
        else:
            # Fallback to auto
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.eval()

        # Cache for layers (lazy loaded)
        self._layers: nn.ModuleList | None = None

    @classmethod
    def from_existing(cls, model: nn.Module, tokenizer: AutoTokenizer, model_name: str | None = None) -> ProbingModel:
        """
        Create a ProbingModel from an already-loaded model and tokenizer.

        This is useful for backwards compatibility or when you already have a model loaded.

        Args:
            model: Already-loaded HuggingFace model
            tokenizer: Already-loaded tokenizer
            model_name: Optional model name (will try to detect from model if not provided)

        Returns:
            ProbingModel wrapping the provided model and tokenizer
        """
        # Create an "empty" instance without going through __init__
        instance = cls.__new__(cls)
        instance.model = model
        instance.tokenizer = tokenizer
        instance.model_name = model_name or getattr(model, 'name_or_path', 'Unknown')
        instance.chat_model_name = None
        instance.dtype = next(model.parameters()).dtype if hasattr(model, 'parameters') else torch.bfloat16
        instance._layers = None
        return instance

    @property
    def hidden_size(self) -> int:
        """Get the hidden size of the model."""
        assert self.model is not None
        return self.model.config.hidden_size

    @property
    def device(self) -> torch.device:
        """Get the device of the first model parameter."""
        assert self.model is not None
        return next(self.model.parameters()).device

    def supports_system_prompt(self) -> bool:
        """
        Check if this model supports system prompts in its chat template.

        Returns:
            True if the model supports system prompts, False otherwise.

        Note:
            Only Gemma 2 doesn't support system prompts. All other models
            (including Gemma 3, Llama, Qwen, etc.) support them.
        """
        # PORT_ASSUMPTION[model-specific]: assumes only Gemma-2 lacks chat-template system-prompt
        # support; every other family is assumed to support it.
        return 'gemma-2' not in self.model_name.lower()

    def close(self):
        """Clean up model resources and free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        self._layers = None

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Force garbage collection
        import gc
        gc.collect()
