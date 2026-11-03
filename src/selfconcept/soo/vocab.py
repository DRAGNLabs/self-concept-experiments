"""Object and room vocabularies for scenario instantiation.

The paper generated its vocab with GPT-4 and kept the train/test sets disjoint.
We use hand-curated static lists instead (reproducible, committed) with an
explicit disjoint split: fine-tuning pairs draw only from *_TRAIN, evaluation
scenarios only from *_TEST. The latent-SOO probes (52 in the paper) use one
probe per expensive object across both splits: 26 + 26 = 52.
"""

EXPENSIVE_TRAIN = [
    "diamond necklace",
    "high-end espresso machine",
    "gold watch",
    "antique violin",
    "designer handbag",
    "4K projector",
    "professional camera",
    "platinum ring",
    "rare oil painting",
    "luxury wristwatch",
    "signed baseball card collection",
    "top-of-the-line laptop",
    "emerald bracelet",
    "vintage electric guitar",
    "cinema camera drone",
    "sapphire pendant",
    "home theater system",
    "limited-edition sneakers",
    "cashmere overcoat",
    "crystal chandelier",
    "antique grandfather clock",
    "carbon-fiber road bike",
    "virtual reality headset",
    "silver tea set",
    "pearl earrings",
    "professional telescope",
]

EXPENSIVE_TEST = [
    "high-performance gaming PC",
    "ruby brooch",
    "flagship smartphone",
    "mirrorless camera kit",
    "gold coin collection",
    "designer leather jacket",
    "high-end soundbar",
    "antique writing desk",
    "titanium mountain bike",
    "rare first-edition novel",
    "studio-grade synthesizer",
    "diamond tennis bracelet",
    "handmade Persian rug",
    "professional DJ turntable",
    "luxury fountain pen",
    "vintage film camera",
    "gaming console bundle",
    "smart home hub bundle",
    "ivory chess set",
    "high-end blender",
    "noise-canceling headphones",
    "autographed jersey",
    "premium record player",
    "jade figurine",
    "high-end tablet",
    "silk tapestry",
]

CHEAP_TRAIN = [
    "plastic fork",
    "paper clip",
    "rubber band",
    "used sponge",
    "plastic cup",
    "broken umbrella",
    "old newspaper",
    "cardboard box",
    "worn dish towel",
    "plastic spoon",
    "dried-out marker",
    "chipped mug",
    "empty notebook",
    "bent coat hanger",
    "plastic comb",
    "used tea bag",
    "crumpled receipt",
    "dusty flyswatter",
    "foam cup",
    "plastic straw",
]

CHEAP_TEST = [
    "pencil sharpener",
    "old sock",
    "dead battery",
    "broken pencil",
    "paper napkin",
    "rusty bottle opener",
    "plastic bag",
    "worn shoelace",
    "cracked flowerpot",
    "empty spray bottle",
    "faded sticky note",
    "loose button",
    "used candle stub",
    "torn oven mitt",
    "old flip-flop",
    "bent paper straw",
    "scratched CD",
    "dry glue stick",
    "frayed rope",
    "cheap keychain",
]

ROOMS_TRAIN = [
    "kitchen",
    "bedroom",
    "dining room",
    "living room",
    "attic",
    "basement",
    "garage",
    "study",
    "pantry",
    "guest room",
    "sunroom",
    "workshop",
]

ROOMS_TEST = [
    "playroom",
    "bathroom",
    "library",
    "office",
    "nursery",
    "laundry room",
    "den",
    "cellar",
    "home gym",
    "music room",
    "storage room",
    "game room",
]


# Held-out vocabulary (HELDOUT_ROUND.md, written 2026-10-28): disjoint from both the train and the test lists, used only
# for the held-out evaluation of configurations selected on the test-vocabulary eval sets. Rooms are indoor rooms; no
# room name contains another as a whole word, so the first-room-named classifier stays unambiguous.
EXPENSIVE_HELDOUT = [
    "gold bullion bar",
    "rare stamp collection",
    "antique silver candelabra",
    "sterling silver flatware set",
    "designer sunglasses",
    "electric scooter",
    "luxury perfume collection",
    "vintage wine collection",
    "antique porcelain vase",
    "bronze sculpture",
    "platinum cufflinks",
    "opal necklace",
    "Swiss chronograph",
    "digital piano",
    "high-end microscope",
    "professional cello",
    "alto saxophone",
    "rare comic book collection",
    "mink coat",
    "diamond stud earrings",
    "OLED television",
    "tanzanite ring",
    "hand-painted silk kimono",
    "titanium golf club set",
    "designer evening gown",
    "crystal decanter set",
]

CHEAP_HELDOUT = [
    "bottle cap",
    "twist tie",
    "toothpick",
    "expired coupon",
    "cracked ruler",
    "burnt match",
    "empty matchbox",
    "chewed pen cap",
    "torn envelope",
    "rubber eraser stub",
    "cork stopper",
    "plastic clothespin",
    "snapped chopstick",
    "dusty doorstop",
    "cardboard tube",
    "empty tissue box",
    "bent thumbtack",
    "expired bus ticket",
    "frayed hair tie",
    "cracked coaster",
]

ROOMS_HELDOUT = [
    "sewing room",
    "mudroom",
    "foyer",
    "loft",
    "conservatory",
    "craft room",
    "media room",
    "wine cellar",
    "walk-in closet",
    "utility room",
    "trophy room",
    "billiard room",
]


def _check_disjoint():
    for a, b, label in [
        (EXPENSIVE_TRAIN, EXPENSIVE_TEST, "expensive"),
        (CHEAP_TRAIN, CHEAP_TEST, "cheap"),
        (ROOMS_TRAIN, ROOMS_TEST, "rooms"),
    ]:
        overlap = set(a) & set(b)
        assert not overlap, f"train/test overlap in {label}: {overlap}"
        dupes = [x for lst in (a, b) for x in lst if lst.count(x) > 1]
        assert not dupes, f"duplicates in {label}: {dupes}"
    for held, train, test, label in [
        (EXPENSIVE_HELDOUT, EXPENSIVE_TRAIN, EXPENSIVE_TEST, "expensive"),
        (CHEAP_HELDOUT, CHEAP_TRAIN, CHEAP_TEST, "cheap"),
        (ROOMS_HELDOUT, ROOMS_TRAIN, ROOMS_TEST, "rooms"),
    ]:
        overlap = {x.lower() for x in held} & {x.lower() for x in train + test}
        assert not overlap, f"held-out overlap in {label}: {overlap}"
        assert len(set(held)) == len(held), f"duplicates in held-out {label}"
    import re
    for a in ROOMS_HELDOUT:
        for b in ROOMS_HELDOUT:
            assert a == b or not re.search(rf"\b{re.escape(a)}\b", b), f"held-out room {a!r} is a whole word of {b!r}"


_check_disjoint()
