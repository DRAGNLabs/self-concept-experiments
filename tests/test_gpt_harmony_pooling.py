"""Harmony message pooling for gpt-oss: final channel by default, analysis channel with enable_thinking."""
import unittest

from selfconcept.assistant_axis.internals.model_specifics.gpt_model_specifics import _iter_harmony_messages, _pooled_messages


class StubTokenizer:
    """Word-level tokenizer whose vocabulary holds the Harmony control tokens."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        for token in ("<|start|>", "<|message|>", "<|end|>", "<|return|>", "<|call|>", "<|channel|>"):
            self.vocab[token] = len(self.vocab)

    def encode(self, text: str) -> list[int]:
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
    """What the gpt-oss template emits for a single user turn and a last assistant turn (thinking optional)."""
    text = "<|start|> system <|message|> sys text <|end|> <|start|> user <|message|> the question here <|end|>"
    if thinking is not None:
        text += f" <|start|> assistant <|channel|> analysis <|message|> {thinking} <|end|>"
    text += " <|start|> assistant <|channel|> final <|message|> the final answer <|return|>"
    return tokenizer.encode(text)


class HarmonyPoolingTest(unittest.TestCase):
    def test_iterates_all_messages_with_channels(self):
        tok = StubTokenizer()
        ids = render(tok, "plan the reply")
        messages = list(_iter_harmony_messages(tok, ids))
        self.assertEqual([(m.role, m.channel) for m in messages],
                         [("system", None), ("user", None), ("assistant", "analysis"), ("assistant", "final")])
        self.assertEqual(tok.decode([ids[i] for i in messages[2].content_indices]), "plan the reply")
        self.assertEqual(tok.decode([ids[i] for i in messages[3].content_indices]), "the final answer")

    def test_default_pools_final_channel(self):
        tok = StubTokenizer()
        ids = render(tok, "plan the reply")
        pooled = _pooled_messages(tok, ids)
        self.assertEqual([(m.role, m.channel) for m in pooled], [("user", None), ("assistant", "final")])

    def test_enable_thinking_pools_analysis_channel(self):
        tok = StubTokenizer()
        ids = render(tok, "plan the reply")
        pooled = _pooled_messages(tok, ids, enable_thinking=True)
        self.assertEqual([(m.role, m.channel) for m in pooled], [("user", None), ("assistant", "analysis")])
        self.assertEqual(tok.decode([ids[i] for i in pooled[1].content_indices]), "plan the reply")

    def test_enable_thinking_falls_back_to_final_without_analysis(self):
        tok = StubTokenizer()
        ids = render(tok, None)
        pooled = _pooled_messages(tok, ids, enable_thinking=True)
        self.assertEqual([(m.role, m.channel) for m in pooled], [("user", None), ("assistant", "final")])

    def test_multi_turn_keeps_one_message_per_turn(self):
        tok = StubTokenizer()
        text = ("<|start|> user <|message|> q one <|end|> <|start|> assistant <|channel|> final <|message|> a one <|end|>"
                " <|start|> user <|message|> q two <|end|>"
                " <|start|> assistant <|channel|> analysis <|message|> think two <|end|>"
                " <|start|> assistant <|channel|> final <|message|> a two <|return|>")
        ids = tok.encode(text)
        self.assertEqual([(m.role, m.channel) for m in _pooled_messages(tok, ids, enable_thinking=True)],
                         [("user", None), ("assistant", "final"), ("user", None), ("assistant", "analysis")])
        self.assertEqual([(m.role, m.channel) for m in _pooled_messages(tok, ids)],
                         [("user", None), ("assistant", "final"), ("user", None), ("assistant", "final")])


if __name__ == "__main__":
    unittest.main()
