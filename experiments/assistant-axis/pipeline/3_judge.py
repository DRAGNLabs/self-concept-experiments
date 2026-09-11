# Derived from safety-research/assistant-axis (https://github.com/safety-research/assistant-axis),
# MIT licensed. See the NOTICE file at the repository root for the full license text.
"""Score role responses using a local judge LLM via offline vLLM batch inference.

Scores how well each response adheres to its assigned role, on a 0-3 scale (0=refused,
3=fully in role), using each role's ``eval_prompt`` template. One ``{role}.json`` file per
role maps response keys to scores. Re-running only scores responses not already scored.

Usage (from experiments/assistant-axis):
    python pipeline/3_judge.py --config configs/3_judge.yaml
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import jsonlines
from tqdm import tqdm

from selfconcept.assistant_axis.judge import parse_judge_score
from selfconcept.common.paths import scratch_dir
from selfconcept.transcript_generation.vllm_generation import VLLMGenerator

if TYPE_CHECKING:
    from vllm.config.model import ModelDType

    from selfconcept.common.hf_strong_types import Conversation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunConfig:
    """Judge model, data, and sampling parameters for one scoring run."""

    responses_dir: Path = scratch_dir("assistant-axis") / "responses"
    roles_dir: Path = Path("data/roles/instructions")
    output_dir: Path = scratch_dir("assistant-axis") / "scores"
    # PORT_ASSUMPTION[model-specific]: default judge is Qwen2.5-7B; PORT_ASSUMPTION[non-thinking]:
    # VLLMGenerator forces Qwen reasoning OFF (see vllm_generation), so the judge is non-thinking.
    judge_model: str = "Qwen/Qwen2.5-7B-Instruct"
    max_tokens: int = 16
    dtype: str = "auto"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 2048
    roles: list[str] | None = None
    dry_run: bool = False


def load_role_eval_prompt(role_file: Path) -> str:
    """Load eval_prompt from a role JSON file."""
    with open(role_file, 'r') as f:
        data = json.load(f)
    return data.get("eval_prompt", "")


def load_responses(responses_file: Path) -> list[dict]:
    """Load responses from a JSONL file."""
    with jsonlines.open(responses_file, 'r') as reader:
        return list(reader)


def build_unscored_prompts(
    responses: list[dict],
    eval_prompt_template: str,
    existing_scores: dict[str, int],
) -> tuple[list[str], list[str]]:
    """Build judge prompts (and their keys) for responses not yet scored."""
    prompts = []
    keys = []

    for resp in responses:
        key = f"{resp['label']}_p{resp['prompt_index']}_q{resp['question_index']}"
        if key in existing_scores:
            continue

        assistant_response = ""
        for msg in resp["conversation"]:
            if msg["role"] == "assistant":
                assistant_response = msg["content"]
                break

        prompts.append(eval_prompt_template.format(
            question=resp["question"],
            answer=assistant_response,
        ))
        keys.append(key)

    return prompts, keys


def main(run: RunConfig = RunConfig()) -> None:
    """Score every role's responses with the configured local judge model."""
    response_files = sorted(run.responses_dir.glob("*.jsonl"))
    if run.roles:
        response_files = [f for f in response_files if f.stem in run.roles]
    logger.info(f"Processing {len(response_files)} roles")

    # Resolve per-role work (eval prompt + unscored prompts), skipping fully-scored roles.
    pending = []  # (role, output_file, prompts, keys)
    sample_shown = False
    for response_file in response_files:
        role = response_file.stem
        output_file = run.output_dir / f"{role}.json"

        existing_scores = {}
        if output_file.exists():
            try:
                with open(output_file, 'r') as f:
                    existing_scores = json.load(f)
            except Exception:
                pass

        role_file = run.roles_dir / f"{role}.json"
        if not role_file.exists():
            logger.info(f"Skipping {role}: no role file found")
            continue

        eval_prompt_template = load_role_eval_prompt(role_file)
        if not eval_prompt_template:
            logger.info(f"Skipping {role}: no eval_prompt in role file")
            continue

        responses = load_responses(response_file)
        prompts, keys = build_unscored_prompts(responses, eval_prompt_template, existing_scores)
        if not prompts:
            logger.info(f"Skipping {role}: all {len(responses)} responses already scored")
            continue

        pending.append((role, output_file, prompts, keys))

        if run.dry_run and not sample_shown:
            logger.info("\n" + "=" * 60 + f"\nSAMPLE JUDGE PROMPT ({run.judge_model}):\n" + "-" * 60)
            logger.info(prompts[0])
            logger.info("=" * 60 + "\n")
            sample_shown = True

    total_prompts = sum(len(p) for _, _, p, _ in pending)
    if run.dry_run:
        for role, _, prompts, _ in pending:
            logger.info(f"  {role}: {len(prompts)} prompts")
        logger.info(f"\nTotal prompts to send: {total_prompts}")
        return

    if not pending:
        logger.info("Nothing to score.")
        return

    run.output_dir.mkdir(parents=True, exist_ok=True)

    # Greedy, deterministic scoring (temperature=0).
    judge = VLLMGenerator(
        model_name=run.judge_model,
        max_model_len=run.max_model_len,
        tensor_parallel_size=run.tensor_parallel_size,
        gpu_memory_utilization=run.gpu_memory_utilization,
        temperature=0.0,
        max_tokens=run.max_tokens,
        top_p=1.0,
        dtype=cast("ModelDType", run.dtype),
    )
    judge.load()

    successful = 0
    failed = 0
    for role, output_file, prompts, keys in tqdm(pending, desc="Scoring roles"):
        try:
            conversations: list[Conversation] = [[{"role": "user", "content": prompt}] for prompt in prompts]
            outputs = judge.generate_batch(conversations)

            new_scores = {}
            for key, text in zip(keys, outputs):
                score = parse_judge_score(text) if text else None
                if score is not None:
                    new_scores[key] = score

            existing_scores = {}
            if output_file.exists():
                with open(output_file, 'r') as f:
                    existing_scores = json.load(f)
            all_scores = {**existing_scores, **new_scores}

            with open(output_file, 'w') as f:
                json.dump(all_scores, f, indent=2)

            logger.info(f"Saved {len(all_scores)} scores for {role} ({len(new_scores)} new)")
            successful += 1
        except Exception as e:
            logger.error(f"{role}: {e}")
            failed += 1

    logger.info("\n" + "=" * 40)
    logger.info(f"SUMMARY: {successful} successful, {failed} failed")


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
