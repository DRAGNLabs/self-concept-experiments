"""Load Beam-1.0-501B-A23B-NVFP4 on two GPUs and generate a short reply.

The checkpoint is the NVFP4 copy, about 274 GiB. vLLM's published minimum is
353 GB of GPU memory, which is two B200s and not one. Reasoning effort is
``minimal`` (budget 8). The prompt is the chat template's token ids.

This environment's vLLM 0.28 does not register BeamForCausalLM. The script
stops before loading weights when that class is missing. Beam support is in
vLLM 0.33 and newer.

    python scripts/beam_nvfp4_fit_smoke.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


MODEL = "reflection/Beam-1.0-501B-A23B-NVFP4"
PROMPT = "Reply with the single word: ready"


def gpu_memory() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def require_beam_vllm() -> None:
    try:
        from vllm.models.beam import BeamForCausalLM  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "This vLLM does not provide BeamForCausalLM. Beam needs vLLM 0.33 or newer "
            "before the weights are loaded. "
            f"Import failed: {exc}"
        ) from exc


def chat_prompt_ids(tokenizer) -> list[int]:
    messages = [{"role": "user", "content": PROMPT}]
    ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, reasoning_effort="minimal",
        tokenize=True, return_dict=False,
    )
    return list(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    require_beam_vllm()
    from vllm import LLM, SamplingParams, TokensPrompt

    print(f"before load: {gpu_memory()}", flush=True)
    started = time.perf_counter()
    llm = LLM(
        args.model,
        tensor_parallel_size=2,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.95,
        max_num_seqs=4,
        max_num_batched_tokens=8192,
        kv_cache_dtype="fp8",
    )
    load_seconds = time.perf_counter() - started
    after_load = gpu_memory()
    print(f"after load ({load_seconds:.0f}s): {after_load}", flush=True)

    tokenizer = llm.get_tokenizer()
    prompt_ids = chat_prompt_ids(tokenizer)
    rendered = tokenizer.decode(prompt_ids)
    print(f"prompt_tokens: {len(prompt_ids)}", flush=True)
    print(f"prompt_has_minimal_budget: {'Reasoning effort: 8' in rendered}", flush=True)
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_ids)],
        SamplingParams(temperature=0, max_tokens=args.max_tokens, skip_special_tokens=False),
    )
    completion = outputs[0].outputs[0]
    text = completion.text
    token_ids = list(completion.token_ids)
    print(f"completion: {text!r}", flush=True)
    print(f"completion_token_ids: {token_ids}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": args.model,
        "tensor_parallel_size": 2,
        "reasoning_effort": "minimal",
        "max_model_len": args.max_model_len,
        "load_seconds": round(load_seconds, 1),
        "gpu_after_load": after_load,
        "prompt": PROMPT,
        "prompt_n_tokens": len(prompt_ids),
        "prompt_has_minimal_budget": "Reasoning effort: 8" in rendered,
        "completion": text,
        "completion_token_ids": token_ids,
        "n_tokens": len(token_ids),
    }
    (args.out / "fit_smoke.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
