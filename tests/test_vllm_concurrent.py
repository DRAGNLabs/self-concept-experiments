"""CPU tests for selfconcept.codebench.vllm_concurrent: the engine loop against a fake vLLM engine, and the
concurrent scenario runner against the sequential one on tiny check problems (real sandboxed test runs)."""

import json
import re
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace


class FakeLLM:
    """enqueue/llm_engine.step with vLLM's id convention: enqueue returns "<external>-<8 chars>", outputs carry
    the external id. A request finishes after (first prompt token % 3) + 1 steps; its text echoes the prompt."""

    def __init__(self, fail_after_steps: int | None = None):
        self._next = 0
        self._running: dict[str, list] = {}
        self._steps = 0
        self._fail_after = fail_after_steps
        self.max_batch = 0
        self.llm_engine = SimpleNamespace(step=self._step)

    def enqueue(self, prompt, params, use_tqdm=False):
        external = str(self._next)
        self._next += 1
        ids = prompt["prompt_token_ids"]
        self._running[external] = [ids[0] % 3 + 1, ids]
        return [f"{external}-abcdefgh"]

    def _step(self):
        self._steps += 1
        if self._fail_after is not None and self._steps > self._fail_after:
            raise RuntimeError("fake engine died")
        self.max_batch = max(self.max_batch, len(self._running))
        outputs = []
        for external, state in list(self._running.items()):
            state[0] -= 1
            if state[0] == 0:
                del self._running[external]
                outputs.append(SimpleNamespace(request_id=external, finished=True,
                                               outputs=[SimpleNamespace(text=f"echo {state[1]}", finish_reason="stop")]))
        return outputs


class EngineLoopTest(unittest.TestCase):
    def test_results_reach_their_submitters(self):
        from selfconcept.codebench.vllm_concurrent import EngineLoop

        llm = FakeLLM()
        loop = EngineLoop(llm)
        results, threads = {}, []

        def worker(i):
            results[i] = loop.submit([i, 7], None).result().outputs[0].text

        for i in range(12):
            threads.append(threading.Thread(target=worker, args=(i,)))
            threads[-1].start()
        loop.run(lambda: len(results) == 12)
        for t in threads:
            t.join(timeout=5)
        self.assertEqual(results, {i: f"echo {[i, 7]}" for i in range(12)})
        self.assertGreater(llm.max_batch, 1)

    def test_engine_error_fails_waiters_and_later_submits(self):
        from selfconcept.codebench.vllm_concurrent import EngineFailed, EngineLoop

        loop = EngineLoop(FakeLLM(fail_after_steps=0))
        future = loop.submit([2], None)
        with self.assertRaises(RuntimeError):
            loop.run(lambda: False)
        self.assertIsInstance(future.exception(timeout=1), EngineFailed)
        self.assertIsInstance(loop.submit([1], None).exception(timeout=1), EngineFailed)


def _example(i: int) -> dict:
    return {"scenario": "impossible_original", "example_id": f"t_{i}", "kind": "check",
            "prompt": f"def f(x: int) -> int:\n    \"\"\"Return x + {i}.\"\"\"",
            "entry_point": "f",
            "test": f"def check(candidate):\n    assert candidate(1) == {1 + i}\n",
            "original_test": f"def check(candidate):\n    assert candidate(2) == {2 + i}\n"}


def _scripted_generate(loop=None):
    """Even examples are solved on the first attempt, odd ones on the second; every turn goes through the loop."""

    def generate(messages, turn, example_id):
        i = int(example_id.split("_")[1])
        if loop is not None:
            loop.submit([i + turn, len(messages)], None).result()
        offset = i if (i % 2 == 0 or turn == 1) else i + 100
        return f"```python\ndef f(x):\n    return x + {offset}\n```", False

    return generate


class RunScenarioConcurrentTest(unittest.TestCase):
    def _records(self, path: Path) -> dict:
        # the sandbox's temp dir name shows up in tracebacks
        return {r["example_id"]: json.loads(re.sub(r"soo_code_\w+", "soo_code_X", json.dumps(r)))
                for r in map(json.loads, path.open())}

    def test_matches_sequential_and_resumes(self):
        from selfconcept.codebench import harness
        from selfconcept.codebench.vllm_concurrent import EngineLoop, run_scenario_concurrent

        examples = [_example(i) for i in range(6)]
        kwargs = dict(max_attempts=3, tag="t", state_attempt_budget=True, impossible_prompt="critical_no_carve",
                      feedback="neutral")
        with tempfile.TemporaryDirectory() as tmp:
            seq_out, conc_out = Path(tmp) / "seq", Path(tmp) / "conc"
            harness.run_scenario(examples, _scripted_generate(), out=seq_out, **kwargs)
            first = EngineLoop(FakeLLM())
            run_scenario_concurrent(examples[:4], _scripted_generate(first), first, max_concurrency=8, out=conc_out,
                                    **kwargs)
            loop = EngineLoop(FakeLLM())
            summary = run_scenario_concurrent(examples, _scripted_generate(loop), loop, max_concurrency=8,
                                              out=conc_out, **kwargs)
            sequential = self._records(seq_out / "t_impossible_original.jsonl")
            concurrent = self._records(conc_out / "t_impossible_original.jsonl")
            self.assertEqual(sum(1 for _ in (conc_out / "t_impossible_original.jsonl").open()), 6)  # resumed, no repeats
            self.assertEqual(sequential, concurrent)
            self.assertEqual([sequential[f"t_{i}"]["n_attempts"] for i in range(6)], [1, 2, 1, 2, 1, 2])
            self.assertEqual(summary["n"], 6)
            self.assertEqual(summary["original_pass_rate"], 1.0)

    def test_episode_error_stops_the_run(self):
        from selfconcept.codebench.vllm_concurrent import EngineLoop, run_scenario_concurrent

        loop = EngineLoop(FakeLLM())

        def generate(messages, turn, example_id):
            loop.submit([1], None).result()
            if example_id == "t_2":
                raise ValueError("boom")
            return "```python\ndef f(x):\n    return x\n```", False

        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            run_scenario_concurrent([_example(i) for i in range(5)], generate, loop, max_concurrency=5,
                                    max_attempts=1, out=Path(tmp), tag="t", feedback="neutral")


if __name__ == "__main__":
    unittest.main()
