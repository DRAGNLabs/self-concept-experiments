"""Installed by putting this directory first on PYTHONPATH.

vLLM inspects PixtralForConditionalGeneration in a subprocess. Python imports
sitecustomize from PYTHONPATH at startup, so the subprocess gets the same
Transformers aliases as the parent.
"""

import importlib.util
from pathlib import Path

_SHIM = Path(__file__).resolve().parents[3] / "src/selfconcept/measurement/mistral_vllm_shim.py"
_spec = importlib.util.spec_from_file_location("mistral_vllm_shim", _SHIM)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load {_SHIM}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
_module.install_mistral_small4_pixtral_shims()
