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


if __name__ == "__main__":
    unittest.main()
