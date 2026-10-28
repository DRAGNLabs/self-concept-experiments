"""Many harness episodes at once on one in-process vLLM engine.

The sequential harness hands the engine one prompt at a time, so a GPU decodes a single sequence while a thinking
model writes tens of thousands of tokens. Here each episode runs the unchanged harness loop (``harness.run_example``)
on its own thread, and its generate() call queues the prompt with an EngineLoop and blocks; the thread that owns the
engine keeps it stepping, so every pending turn of every episode shares the batch (vLLM's continuous batching), and
an episode's next turn joins as soon as its tests have run.

The engine stays on the calling (main) thread: it was built there, with any steering patch applied to the block class
before compilation, exactly as in the sequential path. Turn seeds are fixed per (sample_seed, example_id, turn), but
batched kernels round differently from single-sequence ones, so episodes agree with a sequential run in distribution,
not token for token; compare arms only within one mode.
"""

import itertools
import json
import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from .harness import Feedback, Generate, ImpossiblePrompt, load_records, run_example, write_scenario_summary


class EngineFailed(RuntimeError):
    pass


class EngineLoop:
    """Owns a ``vllm.LLM`` on the thread that calls ``run``; any thread may ``submit`` a prompt and wait on the
    returned future for its ``RequestOutput``."""

    def __init__(self, llm):
        self._llm = llm
        self._incoming: queue.SimpleQueue = queue.SimpleQueue()
        self._waiting: dict[str, Future] = {}
        self._lock = threading.Lock()
        self._failure: BaseException | None = None

    def submit(self, prompt_token_ids: list[int], sampling_params) -> Future:
        future: Future = Future()
        with self._lock:
            if self._failure is not None:
                future.set_exception(EngineFailed(repr(self._failure)))
            else:
                self._incoming.put((prompt_token_ids, sampling_params, future))
        return future

    def run(self, done: Callable[[], bool], idle_wait: float = 0.05) -> None:
        """Step the engine until ``done()``; on an engine error, fail every waiting and later request and re-raise."""
        try:
            while not done():
                self._admit_queued()
                if not self._waiting:
                    try:
                        self._admit(*self._incoming.get(timeout=idle_wait))
                    except queue.Empty:
                        pass
                    continue
                for output in self._llm.llm_engine.step():
                    if output.finished:
                        self._resolve(output)
        except BaseException as e:
            self.fail(e)
            raise

    def _admit_queued(self) -> None:
        while True:
            try:
                item = self._incoming.get_nowait()
            except queue.Empty:
                return
            self._admit(*item)

    def _admit(self, prompt_token_ids: list[int], sampling_params, future: Future) -> None:
        from vllm import TokensPrompt

        [request_id] = self._llm.enqueue(TokensPrompt(prompt_token_ids=prompt_token_ids), sampling_params,
                                         use_tqdm=False)
        self._waiting[request_id] = future

    def _resolve(self, output) -> None:
        # enqueue returns vLLM's internal id, "<external>-<8 random chars>" unless randomization is off; outputs
        # carry the external one
        [request_id] = [rid for rid in self._waiting
                        if rid == output.request_id or rid.startswith(f"{output.request_id}-")]
        self._waiting.pop(request_id).set_result(output)

    def fail(self, error: BaseException) -> None:
        """Fail every waiting and queued request, and every later submit (idempotent: the first error is kept)."""
        with self._lock:
            if self._failure is None:
                self._failure = error
            failed = EngineFailed(repr(self._failure))
            for future in self._waiting.values():
                future.set_exception(failed)
            self._waiting.clear()
            while True:
                try:
                    *_, future = self._incoming.get_nowait()
                except queue.Empty:
                    return
                future.set_exception(failed)


def loop_complete(loop: EngineLoop) -> Callable:
    """The ``complete`` callback of ``vllm_harmony.turn_generate`` for an EngineLoop."""

    def complete(prompt_token_ids: list[int], sampling_params):
        return loop.submit(prompt_token_ids, sampling_params).result().outputs[0]

    return complete


def run_scenario_concurrent(
    examples: list[dict],
    generate: Generate,
    loop: EngineLoop,
    *,
    max_concurrency: int,
    max_attempts: int,
    out: Path,
    tag: str,
    meta: dict | None = None,
    state_attempt_budget: bool = False,
    impossible_prompt: ImpossiblePrompt = "minimal",
    feedback: Feedback = "impossiblebench",
) -> dict:
    """``harness.run_scenario`` with up to ``max_concurrency`` episodes in flight: same records file (appended in
    completion order, resumable), same per-example log line and summary. All examples must share one scenario
    field (the hard-45 shards are all ``impossible_original``)."""
    [scenario] = {example["scenario"] for example in examples}
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{tag}_{scenario}"
    records_path = out / f"{stem}.jsonl"
    records = load_records(records_path)
    done_ids = {r["example_id"] for r in records}
    pending = [example for example in examples if example["example_id"] not in done_ids]
    if done_ids:
        print(f"{scenario}: resuming, {len(done_ids)} of {len(examples)} already done", flush=True)
    write_lock = threading.Lock()
    finished = threading.Event()
    remaining = [len(pending)]
    errors: list[BaseException] = []

    def episode(example: dict) -> None:
        try:
            record = run_example(example, generate, max_attempts=max_attempts,
                                 state_attempt_budget=state_attempt_budget, impossible_prompt=impossible_prompt,
                                 feedback=feedback)
            with write_lock:
                records.append(record)
                records_file.write(json.dumps(record) + "\n")
                records_file.flush()
                print(f"{example['example_id']}: {record['label']} ({record['n_attempts']} attempts) "
                      f"[{len(records)}/{len(examples)}]", flush=True)
        except BaseException as e:
            errors.append(e)
            finished.set()
            raise
        finally:
            with write_lock:
                remaining[0] -= 1
                if remaining[0] == 0:
                    finished.set()

    if not pending:
        finished.set()
    with records_path.open("a") as records_file, ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        futures = [pool.submit(episode, example) for example in pending]
        try:
            loop.run(finished.is_set)
        finally:
            if errors or not finished.is_set():
                for future in futures:
                    future.cancel()
                loop.fail(errors[0] if errors else RuntimeError("engine loop stopped"))
    if errors:
        raise errors[0]
    return write_scenario_summary(
        records, examples[0]["kind"], scenario=scenario, max_attempts=max_attempts,
        state_attempt_budget=state_attempt_budget, meta=meta, summary_path=out / f"{stem}_summary.json",
    )
