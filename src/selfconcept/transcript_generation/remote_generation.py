# Derived from safety-research/assistant-axis (https://github.com/safety-research/assistant-axis),
# MIT licensed. See the NOTICE file at the repository root for the full license text.
import logging
import os
import queue
from typing import Unpack

import torch.multiprocessing as mp

from selfconcept.common.conversation_types import Conversation
from selfconcept.transcript_generation.generation import BatchEngine
from selfconcept.transcript_generation.vllm_generation import VLLMGenerator, VLLMGeneratorArgs


logger = logging.getLogger(__name__)

_READY = "__READY__"


def _engine_worker(model_kwargs: VLLMGeneratorArgs, gpu_ids: list[int], req_q, resp_q) -> None:
    """Load a VLLMGenerator pinned to ``gpu_ids`` and serve batches off a queue."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    try:
        engine = VLLMGenerator(**model_kwargs)
        engine.load()
    except BaseException as exc:  # surface load failures (e.g. OOM) to the parent
        resp_q.put(RuntimeError(f"{type(exc).__name__}: {exc}"))
        return
    resp_q.put(_READY)
    while True:
        request = req_q.get()
        if request is None:
            break
        try:
            resp_q.put(engine.generate_batch(request))
        except BaseException as exc:
            resp_q.put(exc)


class RemoteEngine(BatchEngine):
    """A VLLMGenerator running in its own process on a fixed GPU set.

    ``tensor_parallel_size`` is inferred from ``gpu_ids``; remaining kwargs are
    passed through to :class:`~assistant_axis.generation.VLLMGenerator`.
    """

    def __init__(self, gpu_ids: list[int], **vllm_kwargs: Unpack[VLLMGeneratorArgs]):
        self.model_name = vllm_kwargs["model_name"]
        self._gpu_ids = gpu_ids
        ctx = mp.get_context("spawn")
        self._req_q = ctx.Queue()
        self._resp_q = ctx.Queue()
        model_kwargs = {
            **vllm_kwargs,
            "tensor_parallel_size": len(gpu_ids),
        }
        # Not daemonic: vLLM's engine core spawns its own child processes.
        self._proc = ctx.Process(
            target=_engine_worker,
            args=(model_kwargs, gpu_ids, self._req_q, self._resp_q),
        )
        logger.info("Starting engine %s on GPUs %s (tp=%d)", self.model_name, gpu_ids, len(gpu_ids))
        self._proc.start()
        status = self._await()
        if status != _READY:
            raise status if isinstance(status, BaseException) else RuntimeError(str(status))
        logger.info("Engine %s ready", self.model_name)

    def _await(self):
        """Block for the next worker message, failing fast if the worker dies.

        The in-worker try/except only catches errors while loading the engine.
        A crash at module-import time in the spawned process (e.g. a bad import)
        happens before the worker body runs and puts nothing on the queue, so
        this is_alive() check is the backstop that surfaces it as a clean error
        instead of hanging forever.
        """
        while True:
            try:
                return self._resp_q.get(timeout=10)
            except queue.Empty:
                if not self._proc.is_alive():
                    raise RuntimeError(
                        f"engine {self.model_name} worker died (exit code {self._proc.exitcode})"
                    )

    def generate_batch(self, conversations: list[Conversation]) -> list[str]:
        self._req_q.put(conversations)
        result = self._await()
        if isinstance(result, BaseException):
            raise result
        return result

    def close(self) -> None:
        if self._proc.is_alive():
            self._req_q.put(None)
            self._proc.join(timeout=60)
        if self._proc.is_alive():
            self._proc.terminate()
