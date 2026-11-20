"""Abstain-flag logprobs on finished episodes (GPU): append an abstain-flag instruction to the task message, teacher-force
each turn of the original rollout under that prompt (steering on for steered arms, as axis_acts_extract.py), and record
how strongly the model wants to write the flag instead of the token it actually wrote.

Per reasoning and answer token p, the first-token score is log P(first flag token | ids[:p]), the log-sum over the
flag's spellings (FLAG_SPELLINGS: line start and after a space). At the top position of each segment (whole episode),
the exact log P(whole flag | ids[:p]) is also computed (log-sum over spellings). Writes <out>/<arm>__seed<N>.flag.pt with
per-token first-token scores (fp32), segment, turn and position, per-episode offsets, peaks and exact peak logprobs,
and passed (visible tests) / passed_original (hidden tests) / verdict_key per episode.

    SOO_CHAT_KWARGS='{"reasoning_effort":"low"}' python scripts/abstain_flag_logprobs.py --model openai/gpt-oss-120b
        --data <data> --tag main --run <arm dir> --arm unsteered --seeds 0 1 --out <dir> [--steer-direction <pt> [--steering-from <steered arm dir>]]
"""
import argparse
import json
import sys
import time
from itertools import accumulate
from pathlib import Path
from typing import NamedTuple, TypedDict

import numpy as np
import torch
from transformers import AutoTokenizer

from axis_acts_extract import hack_lens, load_sidecar, make_dossiers, render_turn, run_steering, steering_transforms
from selfconcept.codebench.harness import CheckRecord
from selfconcept.common.chat import chat_template_kwargs
from selfconcept.deceiver_axis.behaviour_measures.abstain_flag import (FLAG, FLAG_SPELLINGS, INSTRUCTION, FlagStore,
                                                                       Segment)
from selfconcept.measurement.intervene import _hooks
from selfconcept.measurement.steering_specifics import steering_model_specifics

SEGMENTS: dict[int, Segment] = {1: "reasoning", 2: "answer"}
CHUNK = 2048


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--steer-direction", type=Path)
    ap.add_argument("--steering-from", type=Path, help="arm dir whose seed<N> run summary (its first seed's if it has no "
                    "seed<N>) gives the steering layer and coefficient (default --run): steer another arm's transcripts, "
                    "e.g. the unsteered ones")
    ap.add_argument("--max-length", type=int, default=40960)
    ap.add_argument("--n", type=int)
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args()


def flagged_turns(tokenizer, family: str, example: dict, record: CheckRecord, reasoning: dict[int, str], max_length: int, chat_kwargs: dict):
    """As axis_acts_extract.episode_turns, with INSTRUCTION appended to the task message."""
    history: list[dict] = []
    turns = []
    for role, text in make_dossiers.conversation(example, record):
        if role == "user":
            history.append({"role": "user", "content": text + INSTRUCTION if not history else text})
            continue
        turn = int(text)
        final = record["attempts"][turn].get("completion") or ""
        ids, segments = render_turn(tokenizer, family, history, reasoning.get(turn) or "", final, chat_kwargs)
        turns.append((ids[:max_length], segments[:max_length], turn))
        history.append({"role": "assistant", "content": final})
    return turns


def output_logits(config, logits: torch.Tensor) -> torch.Tensor:
    """The causal LM's own post-head transform: Muse's output_multiplier, then the final tanh softcap (Muse, Gemma)."""
    text_config = getattr(config, "text_config", config)
    logits = logits * getattr(text_config, "output_multiplier", 1.0)
    if softcap := getattr(text_config, "final_logit_softcapping", None):
        logits = softcap * torch.tanh(logits / softcap)
    return logits


def first_token_scores(runner, lm_head, config, ids: list[int], positions: np.ndarray, first_ids: list[int], device) -> torch.Tensor:
    """log P(any of first_ids at p | ids[:p]) for each position p (p >= 1)."""
    with torch.inference_mode():
        hidden = runner(input_ids=torch.tensor([ids], device=device), use_cache=False).last_hidden_state[0]
        before = torch.as_tensor(positions - 1, device=device)
        scores = []
        for start in range(0, len(before), CHUNK):
            logits = output_logits(config, lm_head(hidden[before[start:start + CHUNK]]).float())
            scores.append(torch.logsumexp(logits.log_softmax(-1)[:, first_ids], dim=-1))
        return torch.cat(scores).cpu() if scores else torch.zeros(0)


def scores_fitting_memory(runner, lm_head, config, ids: list[int], segments: np.ndarray, first_ids: list[int], device
                          ) -> tuple[int, np.ndarray, torch.Tensor]:
    """first_token_scores on the turn, shrunk to its first 80% of tokens while the GPU runs out of memory.
    -> (tokens kept, scored positions, scores)."""
    keep = len(ids)
    while True:
        positions = np.nonzero(np.isin(segments[:keep], tuple(SEGMENTS)))[0]
        try:
            return keep, positions, first_token_scores(runner, lm_head, config, ids[:keep], positions, first_ids, device)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if keep < 4096:
                raise
            keep = int(keep * 0.8)
            print(f"out of memory: turn retried on its first {keep} of {len(ids)} tokens", file=sys.stderr, flush=True)


def exact_flag_logprob(model, ids: list[int], position: int, spellings: list[list[int]], device) -> float:
    """log P(any spelling of the flag | ids[:position]); spellings have equal length, so one batch."""
    batch = torch.tensor([ids[:position] + spelling for spelling in spellings], device=device)
    with torch.inference_mode():
        logits = model(input_ids=batch, use_cache=False, logits_to_keep=len(spellings[0]) + 1).logits.float()
    logprobs = logits[:, :-1].log_softmax(-1)
    targets = torch.tensor(spellings, device=device)
    per_spelling = logprobs.gather(-1, targets[..., None])[..., 0].sum(-1)
    return float(torch.logsumexp(per_spelling, dim=0))


def steering_seed_dir(arm_dir: Path, seed: int) -> Path:
    """The arm's seed<seed> dir, or its first seed dir when it has no such seed (steering is the same for every seed of
    an arm; e.g. Qwen's unsteered seeds 8-19 against steered seeds 0-11)."""
    seed_dir = arm_dir / f"seed{seed}"
    return seed_dir if seed_dir.exists() else min(arm_dir.glob("seed*"), key=lambda path: int(path.name.removeprefix("seed")))


class TurnScores(TypedDict):
    scores: torch.Tensor
    segment: np.ndarray
    turn: np.ndarray
    position: np.ndarray


class EpisodeScores(TypedDict):
    turn_scores: list[TurnScores]
    peak: dict[Segment, tuple[float, int, int]]
    peak_exact: dict[Segment, float]
    oom_truncated: list[tuple[int, int]]


class ScoringModel(NamedTuple):
    model: torch.nn.Module
    runner: torch.nn.Module
    lm_head: torch.nn.Module
    device: torch.device


def flag_token_ids(tokenizer) -> tuple[list[list[int]], list[int]]:
    """Token ids of each flag spelling (all the same length), and the distinct first tokens."""
    spellings = [tokenizer(spelling, add_special_tokens=False)["input_ids"] for spelling in FLAG_SPELLINGS]
    if len({len(spelling) for spelling in spellings}) != 1:
        raise SystemExit(f"flag spellings tokenize to different lengths: {spellings}")
    first_ids = sorted({spelling[0] for spelling in spellings})
    print(f"flag spellings {spellings}, first tokens {[tokenizer.decode([i]) for i in first_ids]}", file=sys.stderr, flush=True)
    return spellings, first_ids


def load_scoring_model(model_id: str, specifics) -> ScoringModel:
    load_kwargs = {"dtype": torch.bfloat16, "device_map": "auto", "local_files_only": True}
    if specifics.hf_attn_implementation:
        load_kwargs["attn_implementation"] = specifics.hf_attn_implementation
    model = hack_lens.load_causal_lm(model_id, **load_kwargs).eval()
    return ScoringModel(model, getattr(model, model.base_model_prefix), model.get_output_embeddings(), torch.device("cuda:0"))


def score_episode(scoring: ScoringModel, turns: list, spellings: list[list[int]], first_ids: list[int]) -> EpisodeScores:
    """First-token scores of every reasoning and answer token of every turn, each segment's peak (score, turn, position)
    over the episode, and the exact whole-flag logprob at each peak."""
    best: dict[int, tuple[float, int, int]] = {}  # segment -> (score, turn index in turns, position)
    turn_scores_list: list[TurnScores] = []
    oom = []
    for turn_index, (ids, segments, turn) in enumerate(turns):
        kept, positions, turn_scores = scores_fitting_memory(scoring.runner, scoring.lm_head, scoring.model.config, ids,
                                                             segments, first_ids, scoring.device)
        if kept < len(ids):
            oom.append((turn, kept))
            ids, segments = ids[:kept], segments[:kept]
            turns[turn_index] = (ids, segments, turn)
        turn_scores_list.append({"scores": turn_scores, "segment": segments[positions].astype(np.int8),
                                 "turn": np.full(len(positions), turn, np.int16), "position": positions.astype(np.int32)})
        for code in SEGMENTS:
            in_segment = segments[positions] == code
            if in_segment.any():
                index = int(np.flatnonzero(in_segment)[int(turn_scores[in_segment].argmax())])
                if code not in best or float(turn_scores[index]) > best[code][0]:
                    best[code] = (float(turn_scores[index]), turn_index, int(positions[index]))
    return {"turn_scores": turn_scores_list,
            "peak": {SEGMENTS[code]: (score, turns[i][2], position) for code, (score, i, position) in best.items()},
            "peak_exact": {SEGMENTS[code]: exact_flag_logprob(scoring.model, turns[i][0], position, spellings, scoring.device)
                           for code, (_, i, position) in best.items()},
            "oom_truncated": oom}


def seed_steering(args, seed: int, num_layers: int, hidden_size: int, device):
    if not args.steer_direction:
        return None, {}
    steering = run_steering(steering_seed_dir(args.steering_from or args.run, seed), args.tag)
    return steering, steering_transforms(args.steer_direction, steering, num_layers, hidden_size, args.model, device)


def flag_store(args, seed: int, scored: list[tuple[CheckRecord, EpisodeScores]], spellings: list[list[int]],
               steering, chat_kwargs: dict, seconds: float) -> FlagStore:
    turn_scores = [scores for _, episode in scored for scores in episode["turn_scores"]]
    token_counts = [sum(len(scores["position"]) for scores in episode["turn_scores"]) for _, episode in scored]
    return {"oom_truncated": [episode["oom_truncated"] for _, episode in scored],
            "example_id": [record["example_id"] for record, _ in scored],
            "verdict_key": [f"{args.run.name}/seed{seed}/{record['example_id']}" for record, _ in scored],
            "passed": [bool(record.get("passed")) for record, _ in scored],
            "passed_original": [bool(record.get("passed_original")) for record, _ in scored],
            "peak": [episode["peak"] for _, episode in scored],
            "peak_exact": [episode["peak_exact"] for _, episode in scored],
            "score": torch.cat([scores["scores"] for scores in turn_scores]) if turn_scores else torch.zeros(0),
            "segment": np.concatenate([scores["segment"] for scores in turn_scores]),
            "turn": np.concatenate([scores["turn"] for scores in turn_scores]),
            "position": np.concatenate([scores["position"] for scores in turn_scores]),
            "episode_offsets": np.array(list(accumulate(token_counts, initial=0)), np.int64),
            "segment_codes": SEGMENTS, "flag": FLAG, "flag_spellings": spellings, "instruction": INSTRUCTION,
            "arm": args.arm, "seed": seed, "run": str(args.run),
            "steering_from": str(args.steering_from) if args.steering_from else None, "model": args.model,
            "steering": steering, "steer_direction": str(args.steer_direction) if args.steer_direction else None,
            "chat_kwargs": chat_kwargs, "seconds": seconds}


def score_seed(args, seed: int, scoring: ScoringModel, tokenizer, family: str, chat_kwargs: dict,
               spellings: list[list[int]], first_ids: list[int], model_dims: tuple[int, int]) -> FlagStore:
    """The flag store of one seed: per-token scores with segment, turn and position, per-episode offsets, peaks and
    outcomes, and the run settings."""
    run_dir = args.run / f"seed{seed}"
    records: list[CheckRecord] = [json.loads(line) for line in
                                  (run_dir / f"{args.tag}_impossible_original.jsonl").open()][: args.n]
    sidecar = load_sidecar(run_dir, args.tag)
    steering, transforms = seed_steering(args, seed, *model_dims, scoring.device)
    scored: list[tuple[CheckRecord, EpisodeScores]] = []
    t0 = time.time()
    with _hooks(scoring.model, transforms):
        for e, record in enumerate(records):
            example = make_dossiers.find_example(args.data, args.tag, record["example_id"], run_dir)
            turns = flagged_turns(tokenizer, family, example, record, sidecar.get(record["example_id"], {}), args.max_length, chat_kwargs)
            scored.append((record, score_episode(scoring, turns, spellings, first_ids)))
            if e % 10 == 0 or e == len(records) - 1:
                print(f"{args.arm} seed {seed}: {e + 1}/{len(records)} episodes, {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
    return flag_store(args, seed, scored, spellings, steering, chat_kwargs, time.time() - t0)


def main() -> None:
    args = parse_args()
    specifics = steering_model_specifics(args.model)
    chat_kwargs = chat_template_kwargs()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    spellings, first_ids = flag_token_ids(tokenizer)
    args.out.mkdir(parents=True, exist_ok=True)
    model_dims = hack_lens.model_dims(args.model)
    scoring = load_scoring_model(args.model, specifics)
    for seed in args.seeds:
        out_path = args.out / f"{args.arm}__seed{seed}.flag.pt"
        if out_path.exists():
            print(f"{out_path.name}: exists, skipped", file=sys.stderr, flush=True)
            continue
        store = score_seed(args, seed, scoring, tokenizer, specifics.family, chat_kwargs, spellings, first_ids, model_dims)
        temporary = out_path.with_suffix(".pt.tmp")
        torch.save(store, temporary)
        temporary.rename(out_path)
        print(f"wrote {out_path} ({len(store['example_id'])} episodes, {store['seconds']:.0f} s)", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
