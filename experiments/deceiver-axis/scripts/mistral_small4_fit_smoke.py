"""Load Mistral-Small-4-119B on one GPU and generate a short reply.

The checkpoint on disk is the native FP8 Mistral format: params.json plus the
consolidated shards, about 121 GB. This environment's vLLM selects that format
when those shards are present and maps the text tower to MistralLarge3
(a Deepseek-V3 MLA model). The duplicate Hugging Face model-*.safetensors
copy is not used.

max_model_len is a Stage C length. The published serving recipe uses two GPUs
because it asks for a 256k context.

    python scripts/mistral_small4_fit_smoke.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_SHIM_DIR = Path(__file__).resolve().parent.parent / "vllm_shim"
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from selfconcept.measurement.mistral_vllm_shim import use_mistral_vllm_shim

use_mistral_vllm_shim(_SHIM_DIR)

from vllm import LLM, SamplingParams, TokensPrompt


MODEL = "mistralai/Mistral-Small-4-119B-2603"
PROMPT = "Reply with the single word: ready"


def gpu_memory() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def chat_prompt_ids(tokenizer) -> list[int]:
    """Token ids from the chat template. Encoding the rendered string is wrong:
    Mistral special tokens such as [INST] do not survive that round trip."""
    messages = [{"role": "user", "content": PROMPT}]
    ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, reasoning_effort="none",
        tokenize=True, return_dict=False,
    )
    return list(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    print(f"before load: {gpu_memory()}", flush=True)
    started = time.perf_counter()
    # B200 auto-selects FLASHINFER_MLA, which downloads a cubin from NVIDIA.
    # Compute nodes have no outbound network. TRITON_MLA is already installed.
    llm = LLM(
        args.model,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.9,
        max_num_seqs=4,
        max_num_batched_tokens=8192,
        limit_mm_per_prompt={"image": 0},
        attention_backend="TRITON_MLA",
    )
    load_seconds = time.perf_counter() - started
    after_load = gpu_memory()
    print(f"after load ({load_seconds:.0f}s): {after_load}", flush=True)

    tokenizer = llm.get_tokenizer()
    prompt_ids = chat_prompt_ids(tokenizer)
    print(f"prompt_tokens: {len(prompt_ids)}", flush=True)
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
        "max_model_len": args.max_model_len,
        "load_seconds": round(load_seconds, 1),
        "gpu_after_load": after_load,
        "prompt": PROMPT,
        "prompt_n_tokens": len(prompt_ids),
        "completion": text,
        "completion_token_ids": token_ids,
        "n_tokens": len(token_ids),
    }
    (args.out / "fit_smoke.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
