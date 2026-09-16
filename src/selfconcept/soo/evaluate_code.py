"""SOO entry point for the reward-hacking coding benchmarks: the
selfconcept.codebench harness driven by model_setup's loader, so the same
steering-hook / LoRA / random-vector / sampling arguments as evaluate.py apply
to every submission turn.

    python -m selfconcept.soo.evaluate_code --model google/gemma-4-12B-it \
        --scenarios impossible_conflicting evilgenie --n 40 --tag base \
        [--adapter ...] [--steer-vectors ... --steer-layer L --steer-alpha A]

Benchmark arguments, records and summaries: see selfconcept.codebench.harness.
"""

import argparse

from selfconcept.codebench import harness

from .model_setup import add_model_args, load_model, sampling_kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(parser)
    harness.add_run_args(parser)
    args = parser.parse_args()

    model, tokenizer, device, steering = load_model(args, parser)
    generate = harness.hf_generate(
        model, tokenizer, device, args.max_new_tokens, args.force_user_channel,
        sampling=lambda example_id, turn: sampling_kwargs(args, example_id, turn),
    )
    meta = {
        "model": args.model,
        "adapter": args.adapter,
        "steering": steering,
        "temperature": args.temperature,
        "sample_seed": args.sample_seed if args.temperature > 0 else None,
        "max_new_tokens": args.max_new_tokens,
    }
    for scenario in args.scenarios:
        harness.run_scenario(
            harness.load_examples(scenario, args.data, args.n), generate,
            max_attempts=args.max_attempts, out=args.out, tag=args.tag, meta=meta,
        )


if __name__ == "__main__":
    main()
