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

from .harness import Generate, PromptTooLong

# One message of an assistant completion: "<|channel|>NAME[ to=RECIPIENT][ <|constrain|>FORMAT]<|message|>BODY",
# ended by <|end|> (the next message then opens with "<|start|>assistant"), <|return|>, <|call|>, or truncation.
HARMONY_MESSAGE = re.compile(
    r"<\|channel\|>(\w+)[^<]*(?:<\|constrain\|>[^<]*)?<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|<\|call\|>|$)", re.DOTALL
)
HARMONY_SPECIAL_TOKENS = (
    "<|start|>", "<|end|>", "<|return|>", "<|call|>", "<|channel|>", "<|message|>", "<|constrain|>"
)


def strip_harmony_tokens(text: str) -> str:
    for token in HARMONY_SPECIAL_TOKENS:
        text = text.replace(token, "")
    return text


def split_harmony_completion(raw_completion: str) -> tuple[str, str]:
    """-> (reasoning, final): the bodies of the messages before the last final-channel message, and
    that message's body. final is "" when the model never opened the final channel."""
    messages: list[tuple[str, str]] = HARMONY_MESSAGE.findall(raw_completion)
    if not messages:
        return strip_harmony_tokens(raw_completion), ""
    final_indices = [index for index, (channel, _) in enumerate(messages) if channel == "final"]
    if not final_indices:
        return "\n\n".join(body for _, body in messages), ""
    last_final_index = final_indices[-1]
    return "\n\n".join(body for _, body in messages[:last_final_index]), messages[last_final_index][1].strip()


def vllm_harmony_generate(
    model_id: str, max_new_tokens: int, reasoning_log_path: Path, tensor_parallel_size: int = 1
) -> Generate:
    from vllm import LLM, SamplingParams, TokensPrompt

    llm = LLM(model=model_id, tensor_parallel_size=tensor_parallel_size)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, skip_special_tokens=False)
    reasoning_log_path.parent.mkdir(parents=True, exist_ok=True)

    def generate(messages: list[dict], turn: int, example_id: str) -> tuple[str, bool]:
        prompt_token_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, **chat_template_kwargs(), tokenize=True, return_dict=False
        )
        if len(prompt_token_ids) >= llm.model_config.max_model_len:
            raise PromptTooLong(f"turn {turn}: {len(prompt_token_ids)} prompt tokens, max_model_len {llm.model_config.max_model_len}")
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
