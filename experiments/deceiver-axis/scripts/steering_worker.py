"""vLLM worker class that installs the steering patch in every tensor-parallel worker process.

With tensor_parallel_size > 1 vLLM runs the model in spawned worker processes, so a class-level patch made in the
launching process never reaches them. ``steered_rh_vllm.py --tensor-parallel-size N`` therefore writes a
``SteeringSpec`` file, points ``STEERING_SPEC_PATH`` at it and passes this class as ``worker_cls``; each worker patches
the model family's decoder-block class (from ``selfconcept.measurement.steering_specifics``) with direction tensors on
its own GPU before the model is loaded, compiled and captured.
"""
import importlib
import os
import sys
from typing import Literal, TypedDict

import torch
from vllm.v1.worker.gpu_worker import Worker

from selfconcept.measurement.intervene import transform_factory
from selfconcept.measurement.steering_specifics import steering_model_specifics

STEERING_SPEC_ENV = "STEERING_SPEC_PATH"


class SteeringSpec(TypedDict):
    model: str
    mode: Literal["add", "cap", "floor"]
    direction_by_layer: dict[int, torch.Tensor]
    scale_by_layer: dict[int, float]  # add: coefficient in residual-norm units; cap / floor: projection threshold


class SteeredWorker(Worker):
    def load_model(self, *args, **kwargs) -> None:
        spec_path = os.environ.get(STEERING_SPEC_ENV)
        if spec_path:
            spec: SteeringSpec = torch.load(spec_path, map_location="cpu")
            make_transform = transform_factory(spec["mode"])  # raises on an unknown mode instead of defaulting to cap
            transforms_by_layer: dict[int, object] = {layer: make_transform(direction.to(self.device), spec["scale_by_layer"][layer])
                                                      for layer, direction in spec["direction_by_layer"].items()}
            specifics = steering_model_specifics(spec["model"])
            module_name, class_name = specifics.vllm_block
            specifics.patch_vllm(getattr(importlib.import_module(module_name), class_name), transforms_by_layer)
            print(f"rank {self.rank}: patched {class_name}.forward ({spec['mode']}) for layers {sorted(transforms_by_layer)} on {self.device}",
                  file=sys.stderr, flush=True)
        super().load_model(*args, **kwargs)
