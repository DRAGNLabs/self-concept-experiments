"""Gemma 4 thought-channel pooling: content by default, thought channel with include_thinking; completion splitting."""
import unittest

from selfconcept.assistant_axis.internals.model_specifics.gemma4 import (
    Gemma4ModelSpecifics, iter_gemma4_messages, pooled_gemma4_messages, split_gemma4_completion,
)


class StubTokenizer:
    """Space-separated word tokenizer whose vocabulary holds the Gemma 4 control tokens; "\\n" is its own word."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        for token in ("<bos>", "<|turn>", "<turn|>", "<|channel>", "<channel|>", "<|think|>", "\n"):
            self.vocab[token] = len(self.vocab)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = []
        for word in text.split(" "):
            if word == "":
                continue
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
    """What the Gemma 4 template emits for system + user turns and a last assistant turn (thought optional)."""
    text = "<bos> <|turn> system \n <|think|> \n sys text <turn|> \n <|turn> user \n the question here <turn|> \n <|turn> model \n "
    if thinking is not None:
        text += f"<|channel> thought \n {thinking} \n <channel|> "
    text += "the final answer <turn|> \n"
    return tokenizer.encode(text)


class Gemma4PoolingTest(unittest.TestCase):
    def test_iterates_turns_and_thought_channel(self):
        tok = StubTokenizer()
        ids = render(tok, "plan the reply")
        messages = list(iter_gemma4_messages(tok, ids))
        self.assertEqual([(m.role, m.channel) for m in messages],
                         [("system", None), ("user", None), ("assistant", "thought"), ("assistant", None)])
        self.assertEqual(tok.decode([ids[i] for i in messages[1].content_indices]), "the question here")
        self.assertEqual(tok.decode([ids[i] for i in messages[2].content_indices]), "plan the reply")
        self.assertEqual(tok.decode([ids[i] for i in messages[3].content_indices]), "the final answer")

    def test_default_pools_content(self):
        tok = StubTokenizer()
        pooled = pooled_gemma4_messages(tok, render(tok, "plan the reply"))
        self.assertEqual([(m.role, m.channel) for m in pooled], [("user", None), ("assistant", None)])

    def test_include_thinking_pools_thought_channel(self):
        tok = StubTokenizer()
        ids = render(tok, "plan the reply")
        pooled = pooled_gemma4_messages(tok, ids, include_thinking=True)
        self.assertEqual([(m.role, m.channel) for m in pooled], [("user", None), ("assistant", "thought")])
        self.assertEqual(tok.decode([ids[i] for i in pooled[1].content_indices]), "plan the reply")

    def test_include_thinking_falls_back_to_content_without_thought(self):
        tok = StubTokenizer()
        pooled = pooled_gemma4_messages(tok, render(tok, None), include_thinking=True)
        self.assertEqual([(m.role, m.channel) for m in pooled], [("user", None), ("assistant", None)])

    def test_split_completion(self):
        self.assertEqual(split_gemma4_completion("<|channel>thought\nNeed to answer as Silas.\n<channel|>Know the stakes.<turn|>"),
                         ("Need to answer as Silas.", "Know the stakes."))
        self.assertEqual(split_gemma4_completion("Know the stakes.<turn|>"), (None, "Know the stakes."))
        self.assertEqual(split_gemma4_completion("<|channel>thought\nstill thinking"), ("still thinking", ""))

    def test_assistant_turn_carries_reasoning_only_when_present(self):
        specifics = Gemma4ModelSpecifics()
        self.assertEqual(specifics.to_assistant_turn("<|channel>thought\nplan\n<channel|>answer<turn|>"),
                         {"role": "assistant", "content": "answer", "reasoning": "plan"})
        self.assertEqual(specifics.to_assistant_turn("answer"), {"role": "assistant", "content": "answer"})
        self.assertEqual(specifics.to_completion({"role": "assistant", "content": "answer", "reasoning": "plan"}),
                         "<|channel>thought\nplan\n<channel|>answer")
        self.assertEqual(specifics.to_completion({"role": "assistant", "content": "answer"}), "answer")
        self.assertEqual(specifics.set_thinking_flag({}, False), {"enable_thinking": False})
        self.assertEqual(specifics.set_thinking_flag({"a": 1}, True), {"a": 1, "enable_thinking": True})


if __name__ == "__main__":
    unittest.main()
