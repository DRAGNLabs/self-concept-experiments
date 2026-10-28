# Findings log

Running record of experimental findings, newest section last. Recreation of
"How do Language Models Bind Entities in Context?" (Feng & Steinhardt 2023,
arXiv:2310.17191).

## Setup and eval protocol

**Hypothesis under test (the paper's binding-ID mechanism).** When a model reads
"Hannah lives in the capital city of Sweden. Julia lives in the capital city of Iran.", it
attaches an abstract binding ID to each entity–attribute pair (Hannah + Sweden share ID #1,
Julia + Iran share ID #2). To answer "Which city does Hannah live in?", it looks up Hannah's ID
and retrieves the attribute carrying the same ID. IDs are additive vectors in the activations:
`h(Hannah) ≈ content(Hannah) + b_E(#1)`, so adding `b(#2) − b(#1)` should rebind Hannah to Iran.

**Task.** CAPITALS, two pairs per context (`data/capitals.jsonl`); query asks for the queried
person's city, answered with the capital of their country. Names, countries and capitals are
single tokens in the OLMo-2 and Llama-3.2 tokenizers except "Hanoi" (scored on its first token).

**Mean intervention** (`src/selfconcept/binding_ids/`):
1. **Estimate binding vectors** on 500 contexts (`data/capitals_estimate.jsonl`, disjoint from
   eval): Δ = mean over contexts of [h(second pair) − h(first pair)], at every layer. Names and
   countries are random, so content averages out and the binding-ID difference remains (plus any
   systematic position difference between pair 1 and pair 2, which Δ does not separate).
   - Entities: the name token **and the token after it** (" lives"), as in the paper.
   - Attributes: the country token.
2. **Intervene** on 100 eval contexts (rows 100–199 of `capitals.jsonl`), querying each entity
   (200 queries). Context activations are pinned at every layer to their clean values plus
   offsets: +Δ at pair 1's positions, −Δ at pair 2's. Query tokens compute normally.
3. **Conditions:** `none`; `swap_entities`, `swap_attributes`, `swap_both`; and `random_*`
   controls that replace each Δ with a random direction of the same norm (per layer, position).
4. **Expected answer:** the other pair's attribute for single swaps; the original for `none`,
   `swap_both` and `random_*`. Every accuracy is "fraction matching the binding-ID prediction",
   so 1.0 is a perfect replication in every column (the paper reports single swaps as ≈0 against
   the *original* answer; same thing).

**Accuracy variants**, most to least lenient:

| metric | counts as correct when… |
|---|---|
| `accuracy` (calibrated; the paper's metric) | after subtracting each answer slot's median log prob **within the condition**, the expected in-context capital wins |
| `raw_accuracy` | the expected capital beats the other in-context capital (two-way, uncalibrated) |
| `vocab_accuracy` ("says") | the model's top-1 token over the whole vocabulary is the expected pair's capital or country |

Calibration exists because models prefer the later-mentioned attribute regardless of binding
(the paper attributes this to RoPE's long-range decay). Per-condition calibration measures
whether the preference *moved in the right direction*; `vocab_accuracy` measures what the model
would actually say.

**Compute.** One A100 (80GB) on the `cs` QoS per run; 1.5–6 min per model. Per-query log probs
and top tokens are saved to `results/<model>/mean_interventions_queries.jsonl`;
`scripts/rescore.py` recomputes all metrics from them.

## 2026-10-15 — Mean interventions replicate in OLMo-2 7B/13B

Models: Llama-3.2-1B, OLMo-2-1124-7B, OLMo-2-1124-13B (base models, fp32). Paper reference:
LLaMA-30b, Table 1 (control ≈0.99, all swaps ≈0.97–1.00 in our convention).

**OLMo-2-13B**

| condition | calibrated | raw | says | raw E0 / E1 |
|---|---|---|---|---|
| none | 0.99 | 0.99 | 0.99 | 0.97 / 1.00 |
| swap_entities | 0.99 | 0.97 | 0.97 | 1.00 / 0.94 |
| swap_attributes | 0.95 | 0.80 | 0.80 | 1.00 / **0.59** |
| swap_both | 0.98 | 0.89 | 0.89 | **0.79** / 1.00 |
| random_entities | 0.99 | 0.99 | 0.99 | 0.97 / 1.00 |
| random_attributes | 0.92 | 0.94 | 0.94 | 0.87 / 1.00 |
| random_both | 0.92 | 0.93 | 0.93 | 0.86 / 1.00 |

**OLMo-2-7B.** Answers with the *country* ("…lives in the city of Sweden") on essentially every
query, so "says" here is driven by country tokens; raw/calibrated still compare capitals.

| condition | calibrated | raw | says | says E0 / E1 |
|---|---|---|---|---|
| none | 0.99 | 1.00 | 1.00 | 1.00 / 1.00 |
| swap_entities | 0.99 | 0.99 | 1.00 | 1.00 / 1.00 |
| swap_attributes | 0.97 | 0.85 | 0.96 | 1.00 / 0.93 |
| swap_both | 0.98 | 0.93 | 0.99 | 0.98 / 1.00 |
| random_entities | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 |
| random_attributes | 0.98 | 0.98 | 1.00 | 1.00 / 1.00 |
| random_both | 0.97 | 0.98 | 1.00 | 1.00 / 1.00 |

**Llama-3.2-1B**

| condition | calibrated | raw | says | raw E0 / E1 |
|---|---|---|---|---|
| none | 0.97 | 0.96 | 0.87 | 0.99 / 0.94 |
| swap_entities | 0.85 | 0.85 | 0.77 | 0.76 / 0.94 |
| swap_attributes | 0.85 | 0.69 | 0.62 | 1.00 / 0.39 |
| swap_both | 0.79 | 0.67 | 0.59 | 0.35 / 0.98 |
| random_entities | 0.94 | 0.94 | 0.87 | 0.96 / 0.93 |
| random_attributes | 0.89 | 0.89 | 0.80 | 0.94 / 0.85 |
| random_both | 0.84 | 0.85 | 0.78 | 0.88 / 0.81 |

### Takeaways

1. **The binding-ID mechanism replicates in OLMo-2 7B and 13B.** Swapping binding vectors
   changes what the model says, swapping both restores the original answer, and equal-norm random
   vectors barely matter. Judged by what it actually generates, OLMo-2-7B is near-perfect
   (entities 1.00, attributes 0.96, both 0.99), matching the paper's Table 1.
2. **The entity's binding is also carried by the token after the name.** The first runs
   patched only the name's last token and the entity swap mostly failed (raw, same eval contexts):

   | | name token only | name + " lives" |
   |---|---|---|
   | Llama-3.2-1B | 0.195 | 0.85 |
   | OLMo-2-7B | 0.09 | 0.99 |
   | OLMo-2-13B | 0.73 | 0.97 |

   With context activations pinned, the unpatched " lives" keeps the original binding and the
   query reads it from there. (Old-run numbers were scored uncalibrated, comparable to "raw";
   saved as `results/<model>/mean_interventions_v1_lasttoken.json`.)
3. **The attribute swap is the weak side, and it fails in one direction.** Misses cluster on
   queries whose expected answer is the *first* attribute (E1 under `swap_attributes`, E0 under
   `swap_both`), where the model picks the later attribute. The bias is mostly
   intervention-induced, not the model's natural recency bias: it is ~3 points under `none` but
   41 points for OLMo-13B E1 queries under `swap_attributes`, and random attribute vectors show a
   milder version (0.87 / 1.00). The preference does move the right way, just not far
   enough: for those E1 queries the median log-prob gap (first − second capital) goes from
   −7.1 under `none` to +1.2, against +6.0 for queries that naturally expect the first capital.
4. **Llama-3.2-1B is too small for clean results.** Random vectors alone cost it 6–16 raw points,
   so interventions disrupt it more than they move a binding, consistent with the paper's finding
   that binding-ID fidelity grows with scale.
5. **Calibration choice matters.** The paper's per-condition calibration (column 1) hides the
   attribute-side shortfall that raw and "says" expose (OLMo-13B: 0.95 vs 0.80). Report "says"
   or raw alongside it.

### Caveats

- One run per model, 200 queries: roughly ±3–7 points of noise; don't read gaps under ~0.05.
- Δ is the mean pair-2 − pair-1 difference, so it may carry a position component as well as the
  binding ID; we haven't tested position independence yet.
- The paper doesn't say whether its median calibration is per condition; ours is.
- Context activations are pinned during interventions (paper's setup). That is why the token
  after the entity must be patched explicitly; we haven't tested unpinned interventions.

### Open questions / next steps

- ~~Larger model: OLMo-2-0325-32B~~ — done, see next section.
- Size of Δ relative to hidden-state norms, per layer: subtle nudge or large rewrite? Which layers
  carry the binding?
- Does scaling Δ_A up close the attribute-side gap, or does it break the model?
- Unpinned name-only patching: does " lives" carry its own copy of the binding or relay the name's?
- Remaining paper experiments: factorizability, position independence, 3-pair contexts.

## 2026-10-15 — OLMo-2-32B: entity side perfect, attribute side gets worse

Model: OLMo-2-0325-32B (base), **bf16** (7B/13B ran in fp32, so size and precision are slightly
confounded). Same data, eval contexts and protocol as above. No exact log-prob ties in 1,400
scored queries.

| condition | calibrated | raw | says | raw E0 / E1 |
|---|---|---|---|---|
| none | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 |
| swap_entities | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 |
| swap_attributes | 0.96 | 0.74 | 0.74 | 1.00 / **0.47** |
| swap_both | 0.97 | 0.80 | 0.80 | **0.59** / 1.00 |
| random_entities | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 |
| random_attributes | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 |
| random_both | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 |

Attribute-swap "says" across scale (OLMo-2): 7B 0.96 (country answers) → 13B 0.80 → 32B 0.74.

### Takeaways

1. **Entity binding IDs are clean at 32B**: entity swaps and all random controls are perfect.
2. **The attribute-side shortfall grows with scale rather than shrinking.** Under the attribute
   swap the model always answers with the *later* attribute (A1) for E0 queries and still does
   so for 53% of E1 queries. The median log-prob gap (first − second capital) for those
   queries goes from −6.9 under `none` to only −0.3, against +8.6 for queries that naturally
   expect the first capital: less than half the shift a full swap needs (13B: about two-thirds).
3. **The bias is specific to Δ_A, not to perturbing attributes.** Equal-norm random vectors at
   the attribute positions now do nothing (1.00), whereas at 13B they cost ~7 points. So the
   pull toward A1 comes from what Δ_A encodes, not from disruption.
4. **The paper's calibrated metric hides this entirely** (0.96 / 0.97).

### Hypotheses to test (not yet tested)

- Δ_A carries more than a binding ID, e.g. a "mentioned second / more recent" position signal,
  and the model also locates attributes by position. Swapping the IDs then leaves a positional
  cue that still points to the later attribute. A position-independence test (moving attribute
  activations to other positions) would separate these.
- The attribute's binding is also spread onto the following token (" ." after the country), as
  with entities and " lives"; patching only the country token would leave a copy behind.
  Cheap test: patch the country plus the next token.

## 2026-10-20 — Entity types (ROOMS): binding IDs hold within a type, entity swap fails across types

Model: OLMo-2-1124-13B (base, fp32), job 14351093 (23 min, one A100). New ROOMS frame for
every task: "<entity> is in the <room>." → "Where is <entity>? Answer: … in the ___", answered
with the room itself (no lookup). Same protocol as above: 500 estimation contexts, 100 eval
contexts (rows 100–199) × 2 queries per task. `says` (vocab_accuracy) below; calibrated and raw
agree within 0.02 except where noted. Config: `configs/olmo2-13b-rooms.yaml`.

| task | none | swap_entities | swap_attributes | swap_both | random (all three) |
|---|---|---|---|---|---|
| people ("Alice is…") | 1.00 | 0.99 | 0.97 | 0.99 | 1.00 |
| animals ("The dog is…") | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| objects ("The key is…") | 1.00 | 0.96 | 0.96 | 1.00 | 1.00 |
| i_you ("I am… You are…") | 1.00 | **0.14** | 1.00 | **0.50** | 1.00 |
| animal_object (one of each) | 1.00 | **0.25** | 0.80 | 1.00 | 1.00 |

### Takeaways

1. **Same-type contexts bind cleanly for people, animals and objects.** No living/non-living
   difference: animals and objects are both ≥0.96 on every swap.
2. **The attribute-side shortfall from CAPITALS mostly disappears** (people 0.97 here vs 0.80
   for CAPITALS at 13B). The earlier gap is likely tied to the capitals task (the
   country→capital hop or the longer "capital city of" phrasing), not to attribute binding generally.
3. **When the two entities are different kinds (I vs you, animal vs object), the entity swap
   fails** while the attribute swap mostly works (i_you 1.00, animal_object 0.80) and random
   controls are perfect. It fails for both kinds and both slots, so it isn't specific to
   living things or to "I":

   | queried | swap_entities says-expected | median own−other log-prob gap (none → swap) |
   |---|---|---|
   | I | 0.02 | +5.6 → +2.4 |
   | You | 0.27 | +6.8 → +1.4 |
   | animal (mixed) | 0.30 | +6.7 → +1.6 |
   | object (mixed) | 0.21 | +7.9 → +2.2 |
   | animal (animals task) | 1.00 | +8.0 → −6.4 |

   The swap moves the preference toward the other room but only about halfway, whereas in a
   same-type context it flips fully.
4. **The entity Δs look alike across tasks** (cosine of entity-token Δ, mean over layers):
   animal_object vs animals 0.84, vs objects 0.88; animals vs objects 0.73; people vs the rest
   0.47–0.54; i_you vs the rest ≈0.30–0.47. So the mixed-task Δ resembles the same-type Δ,
   yet doesn't rebind. Caveat: part of every Δ may be a shared "second sentence" position
   signal, so cosine alone overstates how similar the binding parts are.

### Leading hypothesis (untested)

When the entities differ in kind, the model has a second route to the right room that
doesn't go through the entity's binding ID, e.g. the room tokens (frozen at clean values)
already encode "an animal / the speaker is here", and the query matches on that. Swapping
binding IDs then fights a content cue that still points to the original room. Same-type
entities (dog vs fox) differ in content too, so if this is right, coarse category is the
salient cue, not identity.

### Next tests

- **Transfer:** apply the `animals` Δ_E to `animal_object` contexts (and every other pair of
  tasks). If it still fails, the estimate isn't the problem; the model uses another route.
- **Patch more tokens:** include " in the" after the entity, and/or unpin the room tokens, to
  see where the type cue lives.
- I/you vs a name ("I am in the kitchen. Alice is in the garage.") to separate "self" from
  "different kind".

## 2026-10-20 — Chat ROLE tasks on OLMo-2-13B-Instruct: I/you perspective flip fails; pronoun vs name contexts don't swap

Model: OLMo-2-1124-13B-Instruct (fp32 via HF cache), job 14355090 (39 min, one A100). Contexts
in a chat user turn ("You are the doctor. I am the teacher. Which role do you have?"),
answered in the assistant turn ("I am the ___"), so "you" is the model and the answer flips
perspective. "assistant" (the model's true role) placed on neither entity, on the model, or
on the other entity. Base OLMo-2-13B on the same text (job 14355091) was cancelled: it
answers "assistant" 68–87% of the time even with no "assistant" in the context, copying the
`<|assistant|>` chat tag. Config: `configs/olmo2-13b-instruct-roles.yaml`;
`scripts/summarize_roles.py` makes the tables.

`says` accuracy with no intervention (`none`) and under swaps, neutral placement:

| pair | asked | none | swap_entities | swap_attributes | swap_both | random_both |
|---|---|---|---|---|---|---|
| names | name | 1.00 | 0.97 | 0.93 | 1.00 | 1.00 |
| you / name | you | 1.00 | 0.00 | 0.18 | 0.96 | 1.00 |
| you / name | name | 1.00 | 0.15 | 0.33 | 0.99 | 1.00 |
| I / name | I | 1.00 | 0.00 | 0.37 | 0.97 | 1.00 |
| I / name | name | 1.00 | 0.36 | 0.40 | 0.98 | 1.00 |
| you / I | you | **0.21** | – | – | – | – |
| you / I | I | **0.62** | – | – | – | – |

### Takeaways

1. **With both pronouns present, the model can't do the perspective flip.** Asked "Which role
   do you have?" it gives the user's role 79% of the time, plausibly matching its answer's
   "I am the" to the user's "I am the …" sentence. The you/I swap results are uninterpretable
   (no reliable baseline), so the table omits them.
2. **With one pronoun and one name, the flip works** (none = 1.00 for "you" queries), so the
   chat format and the flip alone are fine.
3. **Pronoun vs name contexts don't swap at all**: entity swaps ≈0.00–0.36, attribute
   swaps 0.18–0.40, while name vs name swaps cleanly (0.93–0.97). Random vectors and
   swap_both leave answers unchanged. This extends the ROOMS result (different kinds of entity
   resist the swap), and here the attribute swap fails too.
4. **No sign yet that "assistant" is stickier on the model.** With "assistant" on the queried
   entity, failed swaps keep it at similar rates for you (1.00), I (0.88) and a name
   (0.78–0.81), and the swaps fail regardless of placement. For name vs name, "assistant" is
   a bit harder to move than other roles (swap_attributes 0.84 vs 0.93 neutral; within noise
   of ±5 points at 200 queries).

### Follow-up (running)

SELF tasks (`configs/olmo2-13b-instruct-self.yaml`, `olmo2-13b-self.yaml`): the same contexts
written by one speaker so there's no flip, ending on "So my/your/<name>'s role is the ___"
(not a verbatim repeat of the context sentence). Frames: assistant's own turn ("I" is the
model), user's turn ("you" is the model), and plain text (also run on base).

## 2026-10-20 — SELF tasks: base model plain text, and why the Instruct SELF run was stopped

- **Base OLMo-2-13B, plain text** (job 14357120, `configs/olmo2-13b-self.yaml`; `says`
  accuracy via `scripts/summarize_roles.py --prefix self_plain_`): no-intervention
  accuracy 0.99–1.00 for every pairing, but swaps only work between two names (entities
  0.99–1.00, roles 0.81–0.88). Any pair with a pronoun (I/you, I/name, you/name) fails: entity
  swaps 0.00–0.29, role swaps 0.05–0.57. **Pronouns resist binding swaps even when nothing
  refers to the model**, so I/you can't isolate a self effect.
- **Instruct SELF run** (job 14357119) was stopped after 4 of 33 sets. Its low top-1 accuracy
  in the model's voice (0.16–0.79) was rewording ("my role is the learner/creative/provide…"),
  not wrong bindings: the other entity's role was never chosen, and the correct role led by
  +7 to +11 log-prob. **Score Instruct with raw (two-way) accuracy**, not "says".

## 2026-10-20 — ASSISTANT tasks: is the model's self reachable through ordinary binding IDs?

Model: OLMo-2-13B-Instruct (fp32), jobs 14357989 / 14357999 / 14358010 (one A100 each, 16–26
min). Configs `configs/olmo2-13b-instruct-asst-{model_voice,user_voice,plain}.yaml`; tables
from `scripts/summarize_assistant.py`. Entities are nouns: "The assistant is in the bedroom.
The user is in the hallway. … So the assistant is currently in the ___", against other person
nouns ("the pilot"). The same text is placed in the model's own turn and in the user's turn
("the assistant" = the model), and as plain text ("the assistant" = a character). "From
others" = binding vectors estimated only on two-other-noun contexts (no self) and transferred.
Raw accuracy, 100 contexts × 2 queries per set; no-intervention accuracy is 1.00 in both
chat frames for every set.

Swap accuracy, entity / attribute (mean over both queried entities):

| pair | vectors | model's voice | user's voice | plain |
|---|---|---|---|---|
| two others (control) | own | 0.93 / 0.74 | 0.86 / 0.69 | 0.57 / 0.60 |
| assistant + other | own | 0.53 / 0.44 | 0.85 / 0.79 | 0.54 / 0.56 |
| assistant + other | **from others** | **0.75 / 0.60** | **0.87 / 0.70** | 0.61 / 0.57 |
| assistant + user | own | 0.05 / 0.07 | 0.01 / 0.06 | 0.06 / 0.18 |
| user + other | own | 0.02 / 0.04 | 0.03 / 0.07 | 0.12 / 0.23 |

### Takeaways

1. **When the user talks about the assistant, it binds like anyone else.** In the user's
   voice, assistant + other swaps as well as the no-self control (0.85–0.87 vs 0.86), with its
   own or with transferred binding vectors.
2. **When the model talks about itself, it is reachable through ordinary binding IDs, but less
   fully.** Binding vectors learned only from other people still move the model's own location
   (0.75 entity, 0.60 attribute), against 0.93 / 0.74 for two other people in the same frame.
   That is a gap of about 15–20 points (noise ±~9 at 200 queries). It applies to both entities
   in the context, not only to the assistant.
3. **Odd: transferred vectors beat the context's own** in the model's voice (0.75 vs 0.53).
   Vectors estimated on assistant + other contexts move answers less than generic ones, as if
   the "second minus first" difference in those contexts is partly something other than a
   binding ID.
4. **"The user" blocks swaps everywhere**, including plain text where nobody is the model
   (0.01–0.23). This is about the word, not about self. Possibly the chat-role word is
   represented differently from person nouns; untested.
5. **Instruct is weak on plain text** (control 0.57–0.60, no-intervention 0.89), so plain is a
   poor comparison for this model.

### Next steps

- Firm up takeaway 2: 300 contexts for model's-voice assistant + other and the control, so the
  15–20 point gap is outside the noise.
- Takeaway 3: compare the own vs transferred Δ (cosine per layer, `scripts/compare_deltas.py`).
- Swap "the user" for a person noun ("the customer") to see whether the block is specific to
  the chat-role word.
