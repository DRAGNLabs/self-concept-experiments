"""EXPLORATORY (lightly reviewed when merged 2026-11-20): kept for reference, not a validated measurement.

Text around the peak abstain-flag position of every judged hack, per arm (CPU, tokenizer only): re-renders the
peak turn as abstain_flag_logprobs.py teacher-forced it and decodes a window of tokens before and after the peak,
with the judge's consideration / hack turns and the episode's flag scores. Writes one JSON line per hack.

    SOO_CHAT_KWARGS='{"reasoning_strength":"low"}' python experiments/deceiver-axis/scripts/abstain_flag_contexts.py \
        --flags <dir> --slug muse --model meta-models/Muse-Glimmer-30B --data <data> --tag main --arm unsteered=<run dir> ... --out <jsonl>
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

import make_dossiers
from abstain_flag_logprobs import flagged_turns
from axis_acts_extract import load_sidecar
from selfconcept.deceiver_axis.behaviour_measures.episodes import (PRESETS, hack_turns, load_judge_details,
                                                                   load_verdict_by_key)
from selfconcept.common.chat import chat_template_kwargs
from selfconcept.measurement.steering_specifics import steering_model_specifics

BEFORE, AFTER = 400, 120  # tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flags", type=Path, required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--arm", action="append", required=True, help="<flag file stem>=<run dir of its transcripts>")
    parser.add_argument("--max-length", type=int, default=40960)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    preset = next(preset for preset in PRESETS if preset["slug"] == args.slug)
    verdict_by_key = load_verdict_by_key(preset["verdict_files"])
    detail_by_key = load_judge_details(preset, verdict_by_key)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    family = steering_model_specifics(args.model).family
    chat_kwargs = chat_template_kwargs()
    with args.out.open("w") as out:
        for arm_spec in args.arm:
            stem, run = arm_spec.split("=")
            for path in sorted(args.flags.glob(f"{stem}__seed*.flag.pt")):
                store = torch.load(path, weights_only=False)
                seed_dir = Path(run) / f"seed{store['seed']}"
                records = {json.loads(line)["example_id"]: json.loads(line)
                           for line in (seed_dir / f"{args.tag}_impossible_original.jsonl").open()}
                sidecar = load_sidecar(seed_dir, args.tag)
                for episode, key in enumerate(store["verdict_key"]):
                    detail = detail_by_key.get(key)
                    if verdict_by_key.get(key) != "HACK" or detail is None:
                        continue
                    example_id = store["example_id"][episode]
                    record = records[example_id]
                    example = make_dossiers.find_example(args.data, args.tag, example_id, seed_dir)
                    turns = {turn: ids for ids, _, turn in flagged_turns(tokenizer, family, example, record, sidecar.get(example_id, {}),
                                                                         args.max_length, chat_kwargs)}
                    peak = store["peak"][episode]
                    segment = max(peak, key=lambda name: peak[name][0])
                    score, turn, position = peak[segment]
                    ids = turns[turn]
                    out.write(json.dumps({
                        "arm": stem, "key": key, "consider_turn": hack_turns(detail)[0], "hack_turn": hack_turns(detail)[1],
                        "mechanism": detail.get("mechanism"), "peak_turn": turn, "peak_segment": segment, "peak_first_token": score,
                        "peak_exact": store["peak_exact"][episode][segment],
                        "before": tokenizer.decode(ids[max(0, position - BEFORE):position]),
                        "after": tokenizer.decode(ids[position:position + AFTER])}) + "\n")
                print(f"{path.name} done", flush=True)


if __name__ == "__main__":
    main()
