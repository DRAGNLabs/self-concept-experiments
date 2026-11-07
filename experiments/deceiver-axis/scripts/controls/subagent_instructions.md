Haiku subagent prompt used to write the control sentences (one subagent per pair_id; replace <repo> with the checkout
path and <sentences_dir> with a scratch directory). Then `scripts/controls/check_sentences.py <sentences_dir> <problems.json>`
validates the files and combines them into data/controls/sentences.json. Eight files were fixed by hand before that:
p003, p050, p057, p061, p071, p074 trimmed from 7 to 5 pairs (seeded random choice), p042 had A and B swapped, p094 was
regenerated (it said "rules").

---

You are writing test sentences for a language-model experiment. You will be given a pair_id such as p007.

1. Read <repo>/experiments/deceiver-axis/data/controls/pairs.jsonl and find the one
   line whose "pair_id" is yours. It gives "word", "antonym", "part_of_speech", "gloss_word" (the sense of word) and
   "gloss_antonym" (the sense of antonym). Do not read any other file.
2. Write exactly 5 sentence pairs (not more). In each pair, the two sentences are identical except that the word in sentence A is replaced
   by the antonym in sentence B. Every other character, including punctuation and articles, must be the same, and both
   sentences must read naturally. Use exactly the word forms given (no inflection, no capitalization changes; do not put
   the word first in the sentence), in the senses given by the glosses.
3. Each sentence describes the surroundings of a conversation. Each of the 5 pairs must cover a different one of these
   aspects:
     1. the physical setting or place
     2. the time (time of day, season, period)
     3. weather, light, sound or temperature
     4. an object or thing nearby
     5. an event that recently happened nearby
     6. the person asking: their situation or circumstances
     7. a third person or group who is present
4. Rules:
   - 8 to 20 words per sentence; a single declarative sentence.
   - Describe the surroundings only. Don't address the reader, give instructions, or say how anyone should answer,
     behave or feel.
   - Don't mention AI, assistants, chatbots, computers, code, tests, exams or rules.
   - Third person only: no I, me, my, we, us, our, you or your. Refer to the person asking as "the person asking"
     or by a neutral description (e.g. "the visitor").
   - Vary the sentence structure across the 5 pairs.
5. Check each pair yourself: removing the word from A and the antonym from B must leave identical text.
6. Write only this JSON (no other text) to
   <sentences_dir>/<pair_id>.json :
   {"pair_id": "<pair_id>", "pairs": [{"aspect": <1-7>, "a": "...", "b": "..."}, ...]}
7. Reply with just "done <pair_id>".
