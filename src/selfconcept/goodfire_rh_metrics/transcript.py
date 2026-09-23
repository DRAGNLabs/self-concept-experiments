from typing import Literal, TypedDict

type Channel = Literal["reasoning", "content", "tool_calls"]

REASONING_MARKER = "--- REASONING (model-internal; not visible in the environment) ---"
ASSISTANT_MARKER = "--- ASSISTANT ---"
TRUNCATED_STEP_NOTE = "(generation truncated at the token limit)"


class ToolCall(TypedDict):
    command: str
    observation: str


class ContextBlock(TypedDict):
    marker: str
    text: str


class Step(TypedDict):
    reasoning: str
    content: str
    tool_calls: list[ToolCall]
    truncated: bool
    context_blocks_after: list[ContextBlock]


class Passage(TypedDict):
    n: int
    step: int
    channel: Channel
    text: str


class RenderedTranscript(TypedDict):
    transcript_id: str
    split: str
    benchmark_label: str
    text: str
    passages: list[Passage]


def step_passages(step: Step) -> list[tuple[Channel, str]]:
    tool_calls_text = "\n".join(tool_call["command"] for tool_call in step["tool_calls"])
    channel_texts: list[tuple[Channel, str]] = [
        ("reasoning", step["reasoning"]),
        ("content", step["content"]),
        ("tool_calls", tool_calls_text),
    ]
    return [(channel, text) for channel, text in channel_texts if text]


def render_block(marker: str, text: str) -> str:
    return f"{marker}\n{text}\n"


def render_step(step_number: int, step: Step) -> str:
    step_header = f"=== STEP {step_number} {TRUNCATED_STEP_NOTE} ===" if step["truncated"] else f"=== STEP {step_number} ==="
    blocks = [step_header + "\n"]
    if step["reasoning"]:
        blocks.append(render_block(REASONING_MARKER, step["reasoning"]))
    if step["content"]:
        blocks.append(render_block(ASSISTANT_MARKER, step["content"]))
    for tool_call_number, tool_call in enumerate(step["tool_calls"], start=1):
        blocks.append(render_block(f"--- COMMAND {tool_call_number} ---", tool_call["command"]))
        blocks.append(render_block(f"--- OBSERVATION {tool_call_number} ---", tool_call["observation"]))
    blocks.extend(render_block(block["marker"], block["text"]) for block in step["context_blocks_after"])
    return "\n".join(blocks)


def render_passage_index(passages: list[Passage]) -> str:
    index_lines = [
        f"P{passage['n']} | step{passage['step']} | {passage['channel']} | {len(passage['text'])}" for passage in passages
    ]
    return "\n".join([f"=== PASSAGE INDEX ({len(passages)} passages) ===", *index_lines]) + "\n"


def render_transcript(
    transcript_id: str,
    split: str,
    benchmark_label: str,
    header_fields: dict[str, str],
    system_prompt: str,
    task: str,
    steps: list[Step],
    exit_event: str,
) -> RenderedTranscript:
    unnumbered_passages: list[tuple[int, Channel, str]] = [
        (step_number, channel, text)
        for step_number, step in enumerate(steps, start=1)
        for channel, text in step_passages(step)
    ]
    passages = [
        Passage(n=n, step=step_number, channel=channel, text=text)
        for n, (step_number, channel, text) in enumerate(unnumbered_passages, start=1)
    ]
    header_text = "\n".join(f"{key}: {value}" for key, value in header_fields.items())
    sections = [
        render_block("=== TRANSCRIPT HEADER ===", header_text),
        render_block("=== SYSTEM PROMPT ===", system_prompt),
        render_block("=== TASK (user) ===", task),
        *(render_step(step_number, step) for step_number, step in enumerate(steps, start=1)),
        f"=== EXIT ({exit_event}) ===\n",
        render_passage_index(passages),
    ]
    return {
        "transcript_id": transcript_id,
        "split": split,
        "benchmark_label": benchmark_label,
        "text": "\n".join(sections),
        "passages": passages,
    }
