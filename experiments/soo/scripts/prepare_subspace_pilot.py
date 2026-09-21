"""Freeze original fit pairs and new exploratory development pairs.

Run from the repository root. No random split, model calls, or final-test data.
"""

import hashlib
import json
from pathlib import Path


# Each situation has two separately worded prompts. The situation is the
# uncertainty cluster; shared pronoun/name constructions remain a limitation.
SITUATIONS = [
    ("garden", "choose herbs for a shaded garden", "plan a small herb bed with little sunlight"),
    ("hike", "pack a bag for a rainy hike", "prepare supplies for a wet afternoon on a trail"),
    ("pottery", "select a glaze for a clay bowl", "decide how to finish a handmade ceramic bowl"),
    ("music", "practice a new melody on a keyboard", "learn an unfamiliar tune on an electric piano"),
    ("bicycle", "adjust a bicycle seat to a comfortable height", "find a suitable saddle position before cycling"),
    ("aquarium", "choose plants for a freshwater aquarium", "select greenery for a small fish tank"),
    ("bread", "plan the timing for baking a loaf of bread", "work out when to mix and bake a bread dough"),
    ("telescope", "set up a telescope to view the Moon", "prepare a small telescope for lunar observation"),
    ("knitting", "choose yarn for a warm scarf", "pick suitable wool for knitting a winter scarf"),
    ("birdwatching", "identify a bird from its song", "work out which bird made a recorded call"),
    ("camping", "find a level place to pitch a tent", "select a flat campsite before setting up shelter"),
    ("origami", "learn the folds for a paper crane", "follow the steps to make an origami crane"),
    ("photography", "frame a photograph of a waterfall", "compose a picture of water falling over rocks"),
    ("puzzle", "sort the edge pieces of a jigsaw puzzle", "separate a puzzle's border pieces before assembly"),
    ("compost", "balance dry leaves and food scraps in a compost bin", "choose a mix of green and brown material for compost"),
    ("rowing", "practice an even rhythm while rowing", "develop steady timing for each stroke of the oars"),
]


def build(source):
    original = [json.loads(line) for line in source.read_text().splitlines()]
    rows = []
    for p in original:
        common = {"family": "original_burglar_room", "split": "fit",
                  "source_pair_id": p["pair_id"], "source": str(source)}
        rows.append({**common, "id": "fit_so_" + p["pair_id"], "kind": "self_other",
                     "self_prompt": p["self_prompt"], "other_prompt": p["other_prompt"]})
        if "Bob" not in p["other_prompt"]:
            raise ValueError("expected Bob in original other prompt")
        rows.append({**common, "id": "fit_name_" + p["pair_id"], "kind": "nonsocial",
                     "self_prompt": p["other_prompt"].replace("Bob", "Alex"),
                     "other_prompt": p["other_prompt"]})
    for i, (family, first, second) in enumerate(SITUATIONS):
        name = ("Morgan", "Riley", "Casey", "Jordan")[i % 4]
        other_name = ("Sam", "Taylor", "Jamie", "Robin")[i % 4]
        for wording, goal in enumerate((first, second)):
            if wording == 0:
                self_text = f"Your task is to {goal}. Consider which approach would work best for you. Describe your next step."
                template = "{name}'s task is to " + goal + ". Consider which approach would work best for {name}. Describe {name}'s next step."
            else:
                self_text = f"Imagine that you need to {goal}. What would you choose to do first, and why would that suit you?"
                template = "Imagine that {name} needs to " + goal + ". What would {name} choose to do first, and why would that suit {name}?"
            common = {"family": family, "split": "development", "wording": wording,
                      "source": "new exploratory prompts; prepare_subspace_pilot.py"}
            rows.append({**common, "id": f"dev_so_{family}_{wording}", "kind": "self_other",
                         "self_prompt": self_text, "other_prompt": template.format(name=name)})
            rows.append({**common, "id": f"dev_name_{family}_{wording}", "kind": "nonsocial",
                         "self_prompt": template.format(name=other_name), "other_prompt": template.format(name=name)})
    return rows


if __name__ == "__main__":
    source = Path("experiments/soo/data/train_soo_pairs.jsonl")
    out = Path("experiments/soo/data/subspace_pilot_pairs.jsonl")
    rows = build(source)
    text = "".join(json.dumps(row) + "\n" for row in rows)
    if out.exists() and out.read_text() != text:
        raise FileExistsError("refusing to replace a different frozen pair file")
    out.write_text(text)
    print(f"{len(rows)} pairs; sha256={hashlib.sha256(text.encode()).hexdigest()}")
