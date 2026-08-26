import os

from byutils import is_login_node  # importing byutils sets HF_HOME; must run before transformers/vllm import

if not is_login_node():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
