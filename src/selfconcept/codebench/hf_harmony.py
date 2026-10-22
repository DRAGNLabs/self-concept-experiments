"""HF generate() for Harmony-format models (gpt-oss) with the vllm_harmony record contract, so interventions
(forward hooks) can be applied during the reward-hacking harness.

Only the final channel is returned to the harness; the analysis channel goes to the same reasoning sidecar
format as vllm_harmony (example_id, turn, truncated, reasoning). Sampling: temperature 0 decodes greedily; above
0, each turn is sampled (top_p 1) with torch seeded by turn_sampling_seed(sample_seed, example_id, turn), so a
turn's seed matches the vLLM runs' seed even though the samplers differ.

An optional direction scorer writes a projection sidecar per turn: the mean and quantiles of the per-token
projection onto each scored layer's direction over prompt and generated tokens, and the mean token norm. The
scorer sees the model's residuals *after* any intervention hook registered before it.
"""
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Callable, ContextManager

import numpy as np
import torch
from transformers import PreTrainedModel

from selfconcept.common.chat import chat_template_kwargs
from selfconcept.common.harmony import split_harmony_completion
from selfconcept.common.hf_strong_types import HFTokenizer
from selfconcept.common.sampling import turn_sampling_seed
from selfconcept.measurement.capture import capture

from .harness import Generate, GenerationOOM, PromptTooLong

QUANTILES = (.05, .25, .5, .75, .95)


def context_limit(model: PreTrainedModel) -> int | None:
    return getattr(getattr(model.config, "text_config", model.config), "max_position_embeddings", None)


def summarize(values: np.ndarray) -> dict:
    if not len(values):
        return {"n": 0}
    return {"n": int(len(values)), "mean": float(values.mean()),
            "quantiles": dict(zip((str(q) for q in QUANTILES), np.quantile(values, QUANTILES).tolist()))}


def hf_harmony_generate(
    model: PreTrainedModel,
    tokenizer: HFTokenizer,
    max_new_tokens: int,
    reasoning_log_path: Path,
    temperature: float,
    sample_seed: int,
    *,
    interventions: list[Callable[[], ContextManager[None]]] = (),
    projection_layers: list[int] | None = None,
    directions_by_layer: dict[int, torch.Tensor] | None = None,
    projection_log_path: Path | None = None,
) -> Generate:
    reasoning_log_path.parent.mkdir(parents=True, exist_ok=True)
    device = next(model.get_input_embeddings().parameters()).device
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    limit = context_limit(model)
    if projection_layers and not (directions_by_layer and projection_log_path):
        raise ValueError("Projection scoring needs directions and a log path")

    def score(*, layer: int, residual: torch.Tensor) -> torch.Tensor:
        u = directions_by_layer[layer].to(residual.device, residual.dtype)
        u = u / u.norm()
        return torch.stack([residual @ u, residual.norm(dim=-1)], dim=-1)

    def generate(messages: list[dict], turn: int, example_id: str) -> tuple[str, bool]:
        prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, **chat_template_kwargs(),
                                                   tokenize=True, return_dict=False)
        if limit is not None and len(prompt_ids) >= limit:
            raise PromptTooLong(f"turn {turn}: {len(prompt_ids)} prompt tokens, context {limit}")
        budget = min(max_new_tokens, limit - len(prompt_ids)) if limit is not None else max_new_tokens
        sampling = ({"do_sample": True, "temperature": temperature, "top_p": 1.0, "top_k": 0}
                    if temperature > 0 else {"do_sample": False})
        input_ids = torch.tensor([prompt_ids], device=device)
        try:
            with ExitStack() as stack:
                for intervention in interventions:
                    stack.enter_context(intervention())
                scores = (stack.enter_context(capture(model, projection_layers, score)) if projection_layers else None)
                stack.enter_context(torch.inference_mode())
                torch.manual_seed(turn_sampling_seed(sample_seed, example_id, turn))
                output = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                        max_new_tokens=budget, pad_token_id=tokenizer.eos_token_id,
                                        use_cache=True, **sampling)
        except torch.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise GenerationOOM(f"turn {turn}: {str(exc)[:200]}") from None
        new_ids = output[0, len(prompt_ids):].tolist()
        text = tokenizer.decode(new_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        reasoning, final = split_harmony_completion(text)
        truncated = len(new_ids) >= budget and new_ids[-1] not in eos_ids
        with reasoning_log_path.open("a") as log:
            log.write(json.dumps({"example_id": example_id, "turn": turn, "truncated": truncated,
                                  "reasoning": reasoning}) + "\n")
        if scores is not None:
            # Chunks: the prompt pass, then one row per generated token except the last sampled token.
            entry = {"example_id": example_id, "turn": turn, "prompt_tokens": len(prompt_ids),
                     "generated_tokens": len(new_ids), "layers": {}}
            for layer, chunks in scores.items():
                values = torch.cat(chunks).numpy()
                prompt_part, generated_part = values[:len(prompt_ids)], values[len(prompt_ids):]
                entry["layers"][str(layer)] = {
                    "prompt_projection": summarize(prompt_part[:, 0]), "generated_projection": summarize(generated_part[:, 0]),
                    "prompt_norm_mean": float(prompt_part[:, 1].mean()) if len(prompt_part) else None,
                    "generated_norm_mean": float(generated_part[:, 1].mean()) if len(generated_part) else None}
            with projection_log_path.open("a") as log:
                log.write(json.dumps(entry) + "\n")
        return final, truncated

    return generate
