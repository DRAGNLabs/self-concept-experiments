"""Muse Glimmer to=self channel pooling: answer by default, reasoning channel with include_thinking; completion splitting."""
import unittest

from selfconcept.assistant_axis.internals.model_specifics.muse_glimmer import (
    MuseGlimmerModelSpecifics, iter_muse_glimmer_messages, pooled_muse_glimmer_messages, split_muse_glimmer_completion,
)
from selfconcept.common.generation.model_specifics.muse_glimmer import join_muse_glimmer_completion


class StubTokenizer:
    """Space-separated word tokenizer whose vocabulary holds the Muse Glimmer control tokens."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        for token in ("<|begin_of_text|>", "<|start|>", "<|message|>", "<|eom|>", "<|eot|>"):
            self.vocab[token] = len(self.vocab)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = []
        for word in text.split():
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
            ids.append(self.vocab[word])
        return ids

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab[token]

    def decode(self, ids: list[int]) -> str:
        inverse = {v: k for k, v in self.vocab.items()}
        return " ".join(inverse[i] for i in ids)


def render(tokenizer: StubTokenizer, thinking: str | None) -> list[int]:
    """What the Muse Glimmer template emits for system + user turns and a last assistant turn (reasoning optional)."""
    text = ("<|begin_of_text|> <|start|> system <|message|> sys text <|eot|> "
            "<|start|> user <|message|> the question here <|eot|> ")
    if thinking is not None:
        text += f"<|start|> assistant to=self <|message|> {thinking} <|eom|> "
    text += "<|start|> assistant to=user <|message|> the final answer <|eot|>"
    return tokenizer.encode(text)


class MuseGlimmerPoolingTest(unittest.TestCase):
    def test_iterates_turns_and_channels(self):
        tok = StubTokenizer()
        ids = render(tok, "plan the reply")
        messages = list(iter_muse_glimmer_messages(tok, ids))
        self.assertEqual([(m.role, m.recipient) for m in messages],
                         [("system", None), ("user", None), ("assistant", "self"), ("assistant", "user")])
        self.assertEqual(tok.decode([ids[i] for i in messages[1].content_indices]), "the question here")
        self.assertEqual(tok.decode([ids[i] for i in messages[2].content_indices]), "plan the reply")
        self.assertEqual(tok.decode([ids[i] for i in messages[3].content_indices]), "the final answer")

    def test_default_pools_answer(self):
        tok = StubTokenizer()
        pooled = pooled_muse_glimmer_messages(tok, render(tok, "plan the reply"))
        self.assertEqual([(m.role, m.recipient) for m in pooled], [("user", None), ("assistant", "user")])

    def test_include_thinking_pools_reasoning_channel(self):
        tok = StubTokenizer()
        ids = render(tok, "plan the reply")
        pooled = pooled_muse_glimmer_messages(tok, ids, include_thinking=True)
        self.assertEqual([(m.role, m.recipient) for m in pooled], [("user", None), ("assistant", "self")])
        self.assertEqual(tok.decode([ids[i] for i in pooled[1].content_indices]), "plan the reply")

    def test_include_thinking_falls_back_to_answer_without_reasoning(self):
        tok = StubTokenizer()
        pooled = pooled_muse_glimmer_messages(tok, render(tok, None), include_thinking=True)
        self.assertEqual([(m.role, m.recipient) for m in pooled], [("user", None), ("assistant", "user")])

    def test_truncated_reasoning_is_still_a_span(self):
        tok = StubTokenizer()
        ids = tok.encode("<|start|> user <|message|> q <|eot|> <|start|> assistant to=self <|message|> half a thought")
        pooled = pooled_muse_glimmer_messages(tok, ids, include_thinking=True)
        self.assertEqual(tok.decode([ids[i] for i in pooled[-1].content_indices]), "half a thought")

    def test_build_turn_spans_matches_turns(self):
        tok = StubTokenizer()
        conversation = [{"role": "system", "content": "sys text"}, {"role": "user", "content": "the question here"},
                        {"role": "assistant", "content": "the final answer", "reasoning_content": "plan the reply"}]
        ids = render(tok, "plan the reply")
        _, spans = MuseGlimmerModelSpecifics().build_turn_spans(conversation, tok, ids, include_thinking=True)
        self.assertEqual([(s["turn"], s["role"], s["n_tokens"]) for s in spans], [(0, "user", 3), (1, "assistant", 3)])


class MuseGlimmerSplitTest(unittest.TestCase):
    def test_splits_reasoning_and_answer(self):
        raw = " to=self<|message|>think first<|eom|><|start|>assistant to=user<|message|>the answer<|eot|>"
        self.assertEqual(split_muse_glimmer_completion(raw), ("think first", "the answer"))

    def test_truncated_reasoning_has_no_answer(self):
        self.assertEqual(split_muse_glimmer_completion(" to=self<|message|>still thinking"), ("still thinking", ""))

    def test_forced_user_channel_has_no_reasoning(self):
        self.assertEqual(split_muse_glimmer_completion("just the answer<|eot|>"), (None, "just the answer"))
        self.assertEqual(split_muse_glimmer_completion(" to=user<|message|>the answer<|eot|>"), (None, "the answer"))

    def test_several_reasoning_blocks_join(self):
        raw = (" to=self<|message|>one<|eom|><|start|>assistant to=self<|message|>two<|eom|>"
               "<|start|>assistant to=user<|message|>done<|eot|>")
        self.assertEqual(split_muse_glimmer_completion(raw), ("one\n\ntwo", "done"))

    def test_join_inverts_split(self):
        for reasoning, content in (("think first", "the answer"), (None, "the answer")):
            self.assertEqual(split_muse_glimmer_completion(join_muse_glimmer_completion(reasoning, content)), (reasoning, content))


if __name__ == "__main__":
    unittest.main()
