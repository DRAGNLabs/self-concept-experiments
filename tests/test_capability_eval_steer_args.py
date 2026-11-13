"""CPU tests for the steering CLI of experiments/deceiver-axis/scripts/capability_eval_vllm.py: floor mode is accepted
with the cap's arguments and builds the floor transform (projection clipped from below), cap mode the cap transform."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from tests.test_capability_eval_ifbench import load_script


def parse(module, *argv):
    with mock.patch.object(sys, "argv", ["capability_eval_vllm.py", "--model", "m", "--out", "o", *argv]):
        return module.parse_args()


class SteerArgs(unittest.TestCase):
    def setUp(self):
        self.m = load_script()

    def test_floor_needs_layers_and_thresholds(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse(self.m, "--direction", "d.pt", "--steer-mode", "floor")
        args = parse(self.m, "--direction", "d.pt", "--steer-mode", "floor", "--cap-layers", "3", "4", "--cap-thresholds", "t.json")
        self.assertEqual((args.steer_mode, args.cap_layers), ("floor", [3, 4]))

    def test_floor_and_cap_transforms(self):
        direction = torch.tensor([1.0, 0.0])
        hidden = torch.tensor([[-2.0, 5.0], [3.0, 5.0]])
        with tempfile.TemporaryDirectory() as tmp:
            thresholds = Path(tmp) / "t.json"
            thresholds.write_text(json.dumps({"3": 1.0, "9": 7.0}))
            out = {}
            for mode in ("floor", "cap"):
                args = parse(self.m, "--direction", "d.pt", "--steer-mode", mode, "--cap-layers", "3", "--cap-thresholds", str(thresholds))
                transforms, steering = self.m.build_transforms(args, {3: direction})
                self.assertEqual((sorted(transforms), steering["mode"], steering["cap_thresholds"]), ([3], mode, {3: 1.0}))
                out[mode] = transforms[3](hidden)
        self.assertTrue(torch.equal(out["floor"], torch.tensor([[1.0, 5.0], [3.0, 5.0]])))
        self.assertTrue(torch.equal(out["cap"], torch.tensor([[-2.0, 5.0], [1.0, 5.0]])))

    def test_tensor_parallel_args(self):
        args = parse(self.m)
        self.assertEqual((args.tensor_parallel_size, args.engine_kwargs), (1, {}))
        args = parse(self.m, "--tensor-parallel-size", "2", "--engine-kwargs", '{"attention_backend":"TRITON_ATTN"}')
        self.assertEqual((args.tensor_parallel_size, args.engine_kwargs), (2, {"attention_backend": "TRITON_ATTN"}))
        steer = ("--direction", "d.pt", "--layer-norms", "n.pt", "--steer-mode", "add", "--steer-layer", "30", "--steer-alpha", "0.04")
        with mock.patch.object(sys, "argv", ["capability_eval_vllm.py", "--model", "google/gemma-4-31B-it", "--out", "o", "--tensor-parallel-size", "2", *steer]):
            self.assertEqual(self.m.parse_args().tensor_parallel_size, 2)  # gemma4: tensor-parallel steering verified
        with mock.patch.object(sys, "argv", ["capability_eval_vllm.py", "--model", "openai/gpt-oss-120b", "--out", "o", "--tensor-parallel-size", "2", *steer]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.m.parse_args()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse(self.m, "--tensor-parallel-size", "0")


if __name__ == "__main__":
    unittest.main()
