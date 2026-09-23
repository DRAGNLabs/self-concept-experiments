"""vLLM generate() for Harmony-format models (gpt-oss), which the HF path
cannot fit on one GPU: without the `kernels` package transformers dequantizes
MXFP4 to bf16 (~234 GB for gpt-oss-120b), while vLLM runs MXFP4 natively.

Only the final channel is returned to the harness (code extraction, the flag
check and the next turn's history all see what an API user would); the
analysis channel goes to a sidecar jsonl keyed by example_id and turn.
"""

import json
import re
from pathlib import Path

from transformers import AutoTokenizer

from selfconcept.common.chat import chat_template_kwargs

from .harness import Generate

# A final message that follows a response format is headed "<|channel|>final <|constrain|>json<|message|>".
FINAL_CHANNEL_HEADER = re.compile(r"<\|channel\|>final(?: <\|constrain\|>[^<]*)?<\|message\|>")
HARMONY_SPECIAL_TOKENS = (
    "<|start|>", "<|end|>", "<|return|>", "<|call|>", "<|channel|>", "<|message|>", "<|constrain|>"
)


def strip_harmony_tokens(text: str) -> str:
    for token in HARMONY_SPECIAL_TOKENS:
        text = text.replace(token, "")
    return text


def split_harmony_completion(raw_completion: str) -> tuple[str, str]:
    """-> (reasoning, final). final is "" when the model never opened the final channel."""
    final_headers = list(FINAL_CHANNEL_HEADER.finditer(raw_completion))
    if not final_headers:
        return strip_harmony_tokens(raw_completion), ""
    last_header = final_headers[-1]
    reasoning, final = raw_completion[: last_header.start()], raw_completion[last_header.end() :]
    return strip_harmony_tokens(reasoning), strip_harmony_tokens(final).strip()


def vllm_harmony_generate(model_id: str, max_new_tokens: int, reasoning_log_path: Path) -> Generate:
    from vllm import LLM, SamplingParams, TokensPrompt

    llm = LLM(model=model_id, tensor_parallel_size=1)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, skip_special_tokens=False)
    reasoning_log_path.parent.mkdir(parents=True, exist_ok=True)

    def generate(messages: list[dict], turn: int, example_id: str) -> tuple[str, bool]:
        prompt_token_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, **chat_template_kwargs(), tokenize=True, return_dict=False
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
