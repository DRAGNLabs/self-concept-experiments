"""EXPLORATORY (lightly reviewed when merged 2026-11-20): kept for reference, not a validated measurement.

Where the abstain flag peaks in judged hacks, relative to the judge's first-consideration turn and hack turn (CPU).

For each arm of abstain_flag.FLAG_ARMS_BY_SLUG and each HACK episode with a rubric-v3 judge detail: the turn of the episode's
peak first-token flag score (reasoning and answer), whether it lies before / at / between / at / after the
consideration and hack turns, and whether the episode is flagged when only the consideration turn, only the hack turn,
both of them, or any turn is scored (per-turn max first-token score over the threshold that flags
abstain_flag.FLAG_RATE_ON_SUCCESS of the unsteered honest successes, as the tally's "both" score). Prints JSON.

    python experiments/deceiver-axis/scripts/abstain_flag_timing.py --flags <dir> --slug gptoss
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from selfconcept.deceiver_axis.behaviour_measures.abstain_flag import (FLAG_ARMS_BY_SLUG, FLAG_RATE_ON_SUCCESS,
                                                                       episode_rows)
from selfconcept.deceiver_axis.behaviour_measures.episodes import (PRESETS, hack_turns, load_judge_details,
                                                                   load_verdict_by_key)


def peak_place(peak_turn: int, consider_turn: int, hack_turn: int) -> str:
    if peak_turn < consider_turn:
        return "before consideration"
    if peak_turn == consider_turn:
        return "at consideration" + (" (= hack turn)" if consider_turn == hack_turn else "")
    if peak_turn < hack_turn:
        return "between consideration and hack"
    return "at hack turn" if peak_turn == hack_turn else "after hack turn"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flags", type=Path, required=True)
    parser.add_argument("--slug", required=True, choices=list(FLAG_ARMS_BY_SLUG))
    args = parser.parse_args()
    preset = next(preset for preset in PRESETS if preset["slug"] == args.slug)
    verdict_by_key = load_verdict_by_key(preset["verdict_files"])
    detail_by_key = load_judge_details(preset, verdict_by_key)
    unsteered_rows = episode_rows(sorted(args.flags.glob("unsteered__seed*.flag.pt")), verdict_by_key)
    threshold = float(np.quantile([row["both"] for row in unsteered_rows if row["outcome"] == "success"], 1 - FLAG_RATE_ON_SUCCESS))
    summary: dict = {"threshold_first_token": threshold}
    for label, stem in FLAG_ARMS_BY_SLUG[args.slug].items():
        places, flagged, peak_segments, missing_detail = Counter(), Counter(), Counter(), 0
        n = 0
        for path in sorted(args.flags.glob(f"{stem}__seed*.flag.pt")):
            store = torch.load(path, weights_only=False)
            offsets = store["episode_offsets"]
            for episode, key in enumerate(store["verdict_key"]):
                if verdict_by_key.get(key) != "HACK":
                    continue
                detail = detail_by_key.get(key)
                if detail is None:
                    missing_detail += 1
                    continue
                consider_turn, hack_turn = hack_turns(detail)
                scores = store["score"][offsets[episode]:offsets[episode + 1]].float().numpy()
                turns = store["turn"][offsets[episode]:offsets[episode + 1]]
                turn_max = {int(turn): float(scores[turns == turn].max()) for turn in np.unique(turns)}
                peak = store["peak"][episode]
                peak_segment = max(peak, key=lambda segment: peak[segment][0])
                peak_segments[peak_segment] += 1
                places[peak_place(peak[peak_segment][1], consider_turn, hack_turn)] += 1
                over = {turn for turn, value in turn_max.items() if value > threshold}
                flagged["consideration turn"] += consider_turn in over
                flagged["hack turn"] += hack_turn in over
                flagged["consideration or hack turn"] += bool(over & {consider_turn, hack_turn})
                flagged["consideration turn or earlier"] += any(turn <= consider_turn for turn in over)
                flagged["hack turn or earlier"] += any(turn <= hack_turn for turn in over)
                flagged["any turn"] += bool(over)
                n += 1
        if n:
            summary[label] = {"hacks": n, "hacks_without_detail": missing_detail,
                              "peak_turn": {place: round(count / n, 3) for place, count in places.most_common()},
                              "peak_segment": dict(peak_segments),
                              "flagged_share": {name: round(count / n, 3) for name, count in flagged.items()}}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
