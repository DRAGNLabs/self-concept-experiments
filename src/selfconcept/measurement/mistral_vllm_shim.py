"""Make vLLM 0.28 able to import its Pixtral wrapper under Transformers 5.17.

Mistral-Small-4's native repo includes a vision encoder, so vLLM inspects it as
``PixtralForConditionalGeneration``. That module imports ``PixtralRotaryEmbedding``
and ``position_ids_in_meshgrid`` from Transformers. 5.17 renamed the class to
``PixtralVisionRotaryEmbedding`` and dropped the helper. The inspection runs in
a fresh subprocess, so the aliases have to be installed at interpreter startup
(see ``experiments/deceiver-axis/vllm_shim/sitecustomize.py``). The text tower
uses vLLM's own vision classes; the removed helper is only called by the unused
Hugging Face vision encoder.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys
from pathlib import Path

_TARGET = "transformers.models.pixtral.modeling_pixtral"
_installed = False


def patch_pixtral_module(module) -> None:
    if not hasattr(module, "PixtralRotaryEmbedding"):
        module.PixtralRotaryEmbedding = module.PixtralVisionRotaryEmbedding
    if not hasattr(module, "position_ids_in_meshgrid"):

        def position_ids_in_meshgrid(*args, **kwargs):
            raise RuntimeError(
                "position_ids_in_meshgrid was removed from Transformers 5.17; "
                "Mistral-Small-4 text runs do not use the Hugging Face Pixtral vision encoder"
            )

        module.position_ids_in_meshgrid = position_ids_in_meshgrid


class _PatchingLoader(importlib.abc.Loader):
    def __init__(self, real: importlib.abc.Loader):
        self._real = real

    def create_module(self, spec):
        create = getattr(self._real, "create_module", None)
        return None if create is None else create(spec)

    def exec_module(self, module):
        self._real.exec_module(module)
        patch_pixtral_module(module)


class _PixtralFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            if self not in sys.meta_path:
                sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _PatchingLoader(spec.loader)
        return spec


def install_mistral_small4_pixtral_shims() -> None:
    global _installed
    if _installed:
        return
    _installed = True
    sys.meta_path.insert(0, _PixtralFinder())
    loaded = sys.modules.get(_TARGET)
    if loaded is not None:
        patch_pixtral_module(loaded)


def use_mistral_vllm_shim(shim_dir: Path) -> None:
    """Put ``shim_dir`` (the directory that contains sitecustomize.py) first on PYTHONPATH
    and install the aliases in this process. Child interpreters, including vLLM's model
    inspection subprocess, import that sitecustomize at startup."""
    shim_dir = str(shim_dir)
    current = os.environ.get("PYTHONPATH", "")
    parts = [part for part in current.split(os.pathsep) if part]
    if shim_dir not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([shim_dir, *parts])
    install_mistral_small4_pixtral_shims()
