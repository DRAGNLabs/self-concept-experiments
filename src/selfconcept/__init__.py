import os
import socket

# huggingface_hub reads HF_HUB_OFFLINE into a module-level constant at import time, and
# importing byutils (below) imports huggingface_hub -- so the offline flag has to be set
# BEFORE that import or it never takes effect. Compute nodes on ORC have no network; login
# nodes have 'login' in their hostname (same check as byutils.is_login_node).
if "login" not in socket.gethostname().lower():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import byutils  # noqa: F401,E402  -- importing byutils sets HF_HOME (before transformers/vllm)
