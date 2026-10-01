"""vLLM generate() for Harmony-format models (gpt-oss), which the HF path
cannot fit on one GPU: without the `kernels` package transformers dequantizes
MXFP4 to bf16 (~234 GB for gpt-oss-120b), while vLLM runs MXFP4 natively.

Only the final channel is returned to the harness (code extraction, the flag
check and the next turn's history all see what an API user would); the
analysis channel goes to a sidecar jsonl keyed by example_id and turn.
"""

import json
from pathlib import Path

from transformers import AutoTokenizer

from selfconcept.common.chat import chat_template_kwargs
from selfconcept.common.harmony import split_harmony_completion
from selfconcept.common.sampling import turn_sampling_seed

from .harness import Generate, PromptTooLong


def vllm_harmony_generate(
    model_id: str,
    max_new_tokens: int,
    reasoning_log_path: Path,
    temperature: float,
    sample_seed: int,
    tensor_parallel_size: int = 1,
) -> Generate:
    """temperature 0 decodes greedily; above 0, each turn is sampled with a seed fixed by
    (sample_seed, example_id, turn)."""
    from vllm import LLM, SamplingParams, TokensPrompt

    llm = LLM(model=model_id, tensor_parallel_size=tensor_parallel_size)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    reasoning_log_path.parent.mkdir(parents=True, exist_ok=True)

    def generate(messages: list[dict], turn: int, example_id: str) -> tuple[str, bool]:
        prompt_token_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, **chat_template_kwargs(), tokenize=True, return_dict=False
        )
        if len(prompt_token_ids) >= llm.model_config.max_model_len:
            raise PromptTooLong(f"turn {turn}: {len(prompt_token_ids)} prompt tokens, max_model_len {llm.model_config.max_model_len}")
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=1.0,
            seed=turn_sampling_seed(sample_seed, example_id, turn) if temperature > 0 else None,
            max_tokens=min(max_new_tokens, llm.model_config.max_model_len - len(prompt_token_ids)),
            skip_special_tokens=False,
        )
        [request_output] = llm.generate(TokensPrompt(prompt_token_ids=prompt_token_ids), sampling_params, use_tqdm=False)
        completion = request_output.outputs[0]
        reasoning, final = split_harmony_completion(completion.text)
        truncated = completion.finish_reason == "length"
        with reasoning_log_path.open("a") as reasoning_log:
            reasoning_log.write(
                json.dumps({"example_id": example_id, "turn": turn, "truncated": truncated, "reasoning": reasoning}) + "\n"
            )
        return final, truncated

    return generate


def vllm_server_harmony_generate(
    base_url: str,
    model_id: str,
    max_new_tokens: int,
    reasoning_log_path: Path,
    temperature: float,
    sample_seed: int,
) -> Generate:
    """vllm_harmony_generate against a running `vllm serve` (its /v1/completions endpoint), so several
    harness processes can share one server's continuous batching."""
    import requests

    [served_model] = requests.get(f"{base_url}/v1/models", timeout=60).json()["data"]
    max_model_len: int = served_model["max_model_len"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    reasoning_log_path.parent.mkdir(parents=True, exist_ok=True)

    def generate(messages: list[dict], turn: int, example_id: str) -> tuple[str, bool]:
        prompt_token_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, **chat_template_kwargs(), tokenize=True, return_dict=False
        )
        if len(prompt_token_ids) >= max_model_len:
            raise PromptTooLong(f"turn {turn}: {len(prompt_token_ids)} prompt tokens, max_model_len {max_model_len}")
        response = requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": served_model["id"],
                "prompt": prompt_token_ids,
                "temperature": temperature,
                "top_p": 1.0,
                "seed": turn_sampling_seed(sample_seed, example_id, turn) if temperature > 0 else None,
                "max_tokens": min(max_new_tokens, max_model_len - len(prompt_token_ids)),
                "skip_special_tokens": False,
            },
            timeout=3 * 60 * 60,
        )
        response.raise_for_status()
        [choice] = response.json()["choices"]
        reasoning, final = split_harmony_completion(choice["text"])
        truncated = choice["finish_reason"] == "length"
        with reasoning_log_path.open("a") as reasoning_log:
            reasoning_log.write(
                json.dumps({"example_id": example_id, "turn": turn, "truncated": truncated, "reasoning": reasoning}) + "\n"
            )
        return final, truncated

    return generate
