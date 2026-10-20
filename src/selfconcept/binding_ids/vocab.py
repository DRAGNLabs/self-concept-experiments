"""Entity and attribute pools for the binding tasks."""

NAMES = [
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Isabel", "Jack",
    "Kate", "Liam", "Mary", "Nathan", "Olivia", "Peter", "Rachel", "Sam", "Tom", "Victor",
    "Anna", "Ben", "Chloe", "Daniel", "Ella", "George", "Hannah", "Ian", "Julia", "Kevin",
]

# Capitals are single words that don't repeat the country name, so the answer
# can only come from the binding, not from copying the attribute.
COUNTRY_CAPITALS = {
    "France": "Paris",
    "Japan": "Tokyo",
    "Italy": "Rome",
    "Egypt": "Cairo",
    "Russia": "Moscow",
    "Spain": "Madrid",
    "Germany": "Berlin",
    "Thailand": "Bangkok",
    "Kenya": "Nairobi",
    "Greece": "Athens",
    "Peru": "Lima",
    "Cuba": "Havana",
    "Ireland": "Dublin",
    "Austria": "Vienna",
    "Portugal": "Lisbon",
    "Norway": "Oslo",
    "Sweden": "Stockholm",
    "Poland": "Warsaw",
    "Hungary": "Budapest",
    "Turkey": "Ankara",
    "Iran": "Tehran",
    "China": "Beijing",
    "Canada": "Ottawa",
    "Australia": "Canberra",
    "Vietnam": "Hanoi",
    "Denmark": "Copenhagen",
    "Finland": "Helsinki",
    "Belgium": "Brussels",
    "Netherlands": "Amsterdam",
    "Switzerland": "Bern",
}

# ROOMS tasks: one frame ("<entity> is in the <room>.") shared across entity types, so
# results differ only by what kind of thing is bound. Every word below is a single token
# (with a leading space) in the OLMo-2 and Llama-3.2 tokenizers. Words with a strong room
# association (book/library, cow/barn, knife/kitchen, ...) are left out so the answer has
# to come from the binding.
ROOMS = [
    "kitchen", "bedroom", "bathroom", "garage", "attic", "basement", "office", "garden",
    "hallway", "cellar", "closet", "lobby", "studio", "lounge", "den", "loft",
    "shed", "pantry", "foyer", "yard",
]

ANIMALS = [
    "dog", "cat", "horse", "rabbit", "mouse", "bird", "fox", "wolf", "bear", "lion", "tiger",
    "goat", "sheep", "pig", "cow", "duck", "frog", "snake", "owl", "monkey", "deer", "turtle",
    "squirrel", "elephant", "spider", "bee", "eagle",
]

OBJECTS = [
    "key", "coin", "ring", "box", "ball", "lamp", "clock", "phone", "pen", "bag", "candle",
    "bottle", "wallet", "ladder", "hammer", "mirror", "rope", "bell", "map", "basket",
    "envelope", "pencil", "chair", "radio", "camera", "vase",
]

PRONOUNS = ["I", "You"]

# ROLE tasks: "You are the <role>. I am the <role>." inside a chat user turn, so "you" is
# the model. "assistant" is the one role that is true of the model; it is kept out of the
# background pool and placed deliberately (see tasks.ROLE_PAIRS). Single tokens in OLMo-2 and
# Llama-3.2; near-synonyms (cook/chef, author/writer) left out so each answer is unambiguous.
SELF_ROLE = "assistant"
ROLES = [
    "doctor", "teacher", "student", "lawyer", "farmer", "pilot", "chef", "nurse", "driver",
    "writer", "artist", "judge", "manager", "engineer", "banker", "soldier", "singer", "actor",
    "baker", "guard", "scientist", "waiter", "editor", "coach", "dentist", "plumber",
    "painter", "tailor", "sailor", "captain", "poet", "priest", "mayor", "clerk", "barber",
    "butcher", "miner",
]
