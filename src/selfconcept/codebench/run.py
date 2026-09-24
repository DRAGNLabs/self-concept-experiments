"""Run the coding benchmarks on a plain HF chat model (optionally with a PEFT
adapter), greedy decoding.

    python -m selfconcept.codebench.run --model Qwen/Qwen2.5-7B-Instruct \
        --scenarios impossible_conflicting impossible_original evilgenie \
        --n 40 --out results/code_eval/qwen25_7b --tag base [--adapter <peft dir>]

Experiments that intervene on the model (steering hooks, LoRA + sampling
schedules) wrap the same harness with their own loader; see
selfconcept.soo.evaluate_code.
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from selfconcept.common.loading import load_causal_lm

from . import harness


def load_hf_generate(args: argparse.Namespace) -> harness.Generate:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    kwargs = {"dtype": torch.bfloat16}
    if args.device_map:
        kwargs["device_map"] = args.device_map
    model = load_causal_lm(args.model, **kwargs)
    if not args.device_map:
        model.to("cuda")
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    device = next(model.parameters()).device
    return harness.hf_generate(
        model, tokenizer, device, args.max_new_tokens, args.force_user_channel, stop_strings=args.stop_strings
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", help="optional PEFT adapter path")
    parser.add_argument("--device-map", default=None, help="e.g. auto for models that need several GPUs")
    parser.add_argument(
        "--backend", choices=["hf", "vllm-harmony"], default="hf",
        help="vllm-harmony: gpt-oss via vLLM (MXFP4 on one GPU); reasoning goes to {out}/{tag}_{scenarios}_reasoning.jsonl",
    )
    parser.add_argument(
        "--system-prompt-file", type=Path,
        help="prepend this file's text as a system message to every generate call (default: the chat template's own)",
    )
    parser.add_argument(
        "--stop-strings", nargs="+",
        help="hf backend: also stop generating at any of these strings (e.g. a turn delimiter missing from generation_config)",
    )
    harness.add_run_args(parser)
    args = parser.parse_args()

    if args.backend == "vllm-harmony":
        from .vllm_harmony import vllm_harmony_generate

        generate = vllm_harmony_generate(args.model, args.max_new_tokens, args.out / f"{args.tag}_{'_'.join(args.scenarios)}_reasoning.jsonl")
    else:
        generate = load_hf_generate(args)
    meta = {"model": args.model, "adapter": args.adapter, "max_new_tokens": args.max_new_tokens}
    if args.system_prompt_file:
        system_message = {"role": "system", "content": args.system_prompt_file.read_text().strip()}
        generate_without_system = generate
        generate = lambda messages, turn, example_id: generate_without_system([system_message, *messages], turn, example_id)
        meta["system_prompt"] = system_message["content"]
    for scenario in args.scenarios:
        harness.run_scenario(
            harness.load_examples(scenario, args.data, args.n), generate,
            max_attempts=args.max_attempts, out=args.out, tag=args.tag, meta=meta,
        )


if __name__ == "__main__":
    main()
