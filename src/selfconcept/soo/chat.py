"""Chat-template kwargs shared by every apply_chat_template call in the pipeline.

Set $SOO_CHAT_KWARGS to a JSON object to pass extra kwargs, e.g.
    SOO_CHAT_KWARGS='{"enable_thinking": false}'
for Qwen3.x models whose template opens a <think> block by default (with
thinking off the template pre-closes an empty one, matching gemma-4's default
and the direct-answer format the SOO pairs and evals assume). An environment
variable rather than a CLI flag so training, extraction, evaluation and the
code harness all agree without threading an argument through each entry point.
"""

import json
import os


def chat_template_kwargs() -> dict:
    raw = os.environ.get("SOO_CHAT_KWARGS")
    return json.loads(raw) if raw else {}
