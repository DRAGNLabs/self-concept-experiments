"""Cache importlib.metadata.packages_distributions() to avoid a ~15s NFS stall.

transformers.utils.import_utils calls this stdlib function unconditionally at
import time to build a package-name -> distribution-name map. It walks every
*.dist-info/*.egg-info directory in site-packages; on ORC's NFS-mounted home
each directory read is a network round-trip, so a cold call takes ~15-17s.
The result only changes when packages are installed/removed, which touches
site-packages' mtime, so that mtime is enough to invalidate the cache.
"""
from __future__ import annotations

import importlib.metadata
import logging
import os
import pickle
import site
from pathlib import Path

from selfconcept.common.paths import SCRATCH_DIR

logger = logging.getLogger(__name__)

# This is duplicated from common/paths.py. Duplicated to avoid any 
# risk that something paths.py imports beats this file to the punch
# and we end up with a slow import
REPO_ROOT = Path(__file__).resolve().parents[2]
_CACHE_FILE = Path.home() / "nobackup" / "autodelete" / REPO_ROOT.name / "packages_distributions_cache.pkl"


def _site_packages_dir() -> Path | None:
    dirs = [Path(p) for p in site.getsitepackages() if Path(p).is_dir()]

    if len(dirs) > 1:
        logger.warning("Detected multiple canonical sitepackages. `_speed_up_imports.py` assumes just one.")

    return dirs[0] if dirs else None


def patch() -> None:
    site_packages = _site_packages_dir()
    if site_packages is None:
        return
    # mtime is updatd any time the site_packages directory or any of its direct children is updated,
    # busting this cache. If there are any grandchildren or further down that are modified, the cache
    # won't be busted. I don't know of any case where this would happen though.
    mtime_ns = os.stat(site_packages).st_mtime_ns

    if _CACHE_FILE.exists():
        try:
            cached_mtime_ns, mapping = pickle.loads(_CACHE_FILE.read_bytes())
        except Exception:
            cached_mtime_ns, mapping = None, None
        if cached_mtime_ns == mtime_ns:
            importlib.metadata.packages_distributions = lambda: mapping
            return

    mapping = importlib.metadata.packages_distributions()
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_FILE.write_bytes(pickle.dumps((mtime_ns, mapping)))
    importlib.metadata.packages_distributions = lambda: mapping


patch()
