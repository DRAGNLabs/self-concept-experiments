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


_check_disjoint()
