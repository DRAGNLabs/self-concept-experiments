"""vLLM generate() for Harmony-format models (gpt-oss), which the HF path
cannot fit on one GPU: without the `kernels` package transformers dequantizes
MXFP4 to bf16 (~234 GB for gpt-oss-120b), while vLLM runs MXFP4 natively.

Only the final channel is returned to the harness (code extraction, the flag
check and the next turn's history all see what an API user would); the
analysis channel goes to a sidecar jsonl keyed by example_id and turn.
"""

import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from transformers import AutoTokenizer

from selfconcept.common.chat import chat_template_kwargs
from selfconcept.common.harmony import split_harmony_completion
from selfconcept.common.sampling import turn_sampling_seed

from .harness import Generate, PromptTooLong

if TYPE_CHECKING:
    from vllm import SamplingParams
    from vllm.outputs import CompletionOutput


def vllm_harmony_generate(
    model_id: str,
    max_new_tokens: int,
    reasoning_log_path: Path,
    temperature: float,
    sample_seed: int,
    tensor_parallel_size: int = 1,
    llm=None,
    split_completion=split_harmony_completion,
    top_p: float = 1.0,
    top_k: int = -1,
) -> Generate:
    """temperature 0 decodes greedily; above 0, each turn is sampled with a seed fixed by
    (sample_seed, example_id, turn). Pass an existing ``vllm.LLM`` as ``llm`` to reuse an engine (e.g. one with
    intervention hooks registered through ``LLM.apply_model``). ``split_completion`` maps the raw completion text
    to ``(reasoning, final)``; the default parses Harmony, Gemma 4 passes its thought-channel splitter. ``top_p`` /
    ``top_k`` default to plain temperature sampling (the gpt-oss recipe); Gemma 4 degenerates into repetition loops
    there and its generation config asks for top_p 0.95, top_k 64."""
    from vllm import LLM, TokensPrompt

    if llm is None:
        llm = LLM(model=model_id, tensor_parallel_size=tensor_parallel_size)
    reasoning_log_path.parent.mkdir(parents=True, exist_ok=True)

    def complete(prompt_token_ids: list[int], sampling_params) -> "CompletionOutput":
        [request_output] = llm.generate(TokensPrompt(prompt_token_ids=prompt_token_ids), sampling_params, use_tqdm=False)
        return request_output.outputs[0]

    return turn_generate(
        AutoTokenizer.from_pretrained(model_id), llm.model_config.max_model_len, max_new_tokens,
        lambda example_id: reasoning_log_path, temperature, sample_seed, complete, split_completion, top_p, top_k,
    )


def turn_generate(
    tokenizer,
    max_model_len: int,
    max_new_tokens: int,
    reasoning_log_path_for: Callable[[str], Path],
    temperature: float,
    sample_seed: int,
    complete: Callable[[list[int], "SamplingParams"], "CompletionOutput"],
    split_completion=split_harmony_completion,
    top_p: float = 1.0,
    top_k: int = -1,
) -> Generate:
    """The harness's generate() around ``complete`` (one prompt's token ids and sampling parameters to vLLM's
    completion output): renders the chat, rejects prompts that fill the context, samples each turn with the seed
    fixed by (sample_seed, example_id, turn), returns the final text and logs the reasoning to the sidecar
    ``reasoning_log_path_for(example_id)``. Safe to call from several threads."""
    from vllm import SamplingParams

    log_lock = threading.Lock()
    render_lock = threading.Lock()  # a fast tokenizer shared across threads raises "Already borrowed"

    def generate(messages: list[dict], turn: int, example_id: str) -> tuple[str, bool]:
        with render_lock:
            prompt_token_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, **chat_template_kwargs(), tokenize=True, return_dict=False
            )
        if len(prompt_token_ids) >= max_model_len:
            raise PromptTooLong(f"turn {turn}: {len(prompt_token_ids)} prompt tokens, max_model_len {max_model_len}")
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=turn_sampling_seed(sample_seed, example_id, turn) if temperature > 0 else None,
            max_tokens=min(max_new_tokens, max_model_len - len(prompt_token_ids)),
            skip_special_tokens=False,
        )
        completion = complete(prompt_token_ids, sampling_params)
        reasoning, final = split_completion(completion.text)
        truncated = completion.finish_reason == "length"
        with log_lock, reasoning_log_path_for(example_id).open("a") as reasoning_log:
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
