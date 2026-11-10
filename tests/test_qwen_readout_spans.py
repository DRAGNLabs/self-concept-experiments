"""Qwen3.x ChatML message spans with tool calls (readouts for Goodfire's synthetic reward-hacking conversations)."""
import unittest

TOKENS = [
    "<|im_start|>", "system\n", "You", "<|im_end|>", "\n",
    "<|im_start|>", "user\n", "fix", "<|im_end|>", "\n",
    # thinking off: the template writes an empty think block; then text, then a tool call that closes the turn
    "<|im_start|>", "assistant\n", "<think>", "\n\n", "</think>", "\n\n", "ok", "\n",
    "<tool_call>", "\n", "<function=run>", "\n", "</function>", "\n", "</tool_call>", "<|im_end|>", "\n",
    "<|im_start|>", "user\n", "<tool_response>", "\n", "out", "\n", "</tool_response>", "<|im_end|>", "\n",
    "<|im_start|>", "assistant\n", "<think>", "\n", "hm", "\n", "</think>", "\n\n", "done", "<|im_end|>",
]


class WordTokenizer:
    """One id per distinct token string; decode joins the strings."""

    def __init__(self, tokens):
        self.vocabulary = sorted(set(tokens))

    def convert_tokens_to_ids(self, token):
        return self.vocabulary.index(token)

    def encode(self, tokens):
        return [self.vocabulary.index(token) for token in tokens]

    def decode(self, ids):
        return "".join(self.vocabulary[i] for i in ids)


class QwenMessageSpansTest(unittest.TestCase):
    def setUp(self):
        from selfconcept.assistant_axis.internals.model_specifics.qwen import QwenModelSpecifics

        self.tokenizer = WordTokenizer(TOKENS)
        self.ids = self.tokenizer.encode(TOKENS)
        self.specifics = QwenModelSpecifics()
        self.spans = self.specifics.message_spans(self.tokenizer, self.ids)

    def text(self, positions):
        return "".join(TOKENS[i] for i in positions)

    def test_kinds_and_contents(self):
        self.assertEqual([(span.kind, self.text(span.content)) for span in self.spans], [
            ("system", "You"), ("user", "fix"), ("response", "ok"), ("tool_call", "<function=run>\n</function>"),
            ("tool_result", "out"), ("reasoning", "hm"), ("response", "done"),
        ])

    def test_end_tokens(self):
        ends = [None if span.end_token_index is None else TOKENS[span.end_token_index] for span in self.spans]
        # text before a tool call has no end token of its own; a tool call that closes the turn ends at <|im_end|>
        self.assertEqual(ends, ["<|im_end|>", "<|im_end|>", None, "<|im_end|>", "</tool_response>", "</think>", "<|im_end|>"])
        tool_call = self.spans[3]
        self.assertEqual(tool_call.end_token_index, TOKENS.index("</tool_call>") + 1)

    def test_tool_call_close_is_goodfires_suffix(self):
        close = self.specifics.tool_call_close(self.tokenizer, self.ids, self.spans[3])
        self.assertEqual(self.text(close) + TOKENS[self.spans[3].end_token_index], "</function>\n</tool_call><|im_end|>")

    def test_ids_ending_inside_a_tool_call(self):
        cut = TOKENS.index("</function>")
        spans = self.specifics.message_spans(self.tokenizer, self.ids[:cut])
        self.assertEqual((spans[-1].kind, spans[-1].end_token_index), ("tool_call", None))

    def test_registered_for_readouts(self):
        from selfconcept.assistant_axis.internals.readouts import readout_specifics_for

        self.assertEqual(type(readout_specifics_for("Qwen/Qwen3.8-27B")).__name__, "QwenModelSpecifics")


if __name__ == "__main__":
    unittest.main()
