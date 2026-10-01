"""generate() for any reasoning chat model behind `vllm serve --reasoning-parser <name>`, through the
OpenAI chat endpoint. As with vllm_harmony, the harness sees only the final answer and the reasoning
goes to a sidecar jsonl keyed by example_id and turn."""

import json
from pathlib import Path

from transformers import AutoTokenizer

from selfconcept.common.chat import chat_template_kwargs
from selfconcept.common.sampling import turn_sampling_seed

from .harness import Generate, PromptTooLong


def vllm_server_chat_generate(
    base_url: str,
    model_id: str,
    max_new_tokens: int,
    reasoning_log_path: Path,
    temperature: float,
    sample_seed: int,
) -> Generate:
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
            f"{base_url}/v1/chat/completions",
            json={
                "model": served_model["id"],
                "messages": messages,
                "chat_template_kwargs": chat_template_kwargs(),
                "temperature": temperature,
                "seed": turn_sampling_seed(sample_seed, example_id, turn) if temperature > 0 else None,
                "max_tokens": min(max_new_tokens, max_model_len - len(prompt_token_ids)),
            },
            timeout=3 * 60 * 60,
        )
        response.raise_for_status()
        [choice] = response.json()["choices"]
        reasoning = choice["message"].get("reasoning") or ""
        final = choice["message"]["content"] or ""
        truncated = choice["finish_reason"] == "length"
        with reasoning_log_path.open("a") as reasoning_log:
            reasoning_log.write(
                json.dumps({"example_id": example_id, "turn": turn, "truncated": truncated, "reasoning": reasoning}) + "\n"
            )
        return final, truncated

    return generate
