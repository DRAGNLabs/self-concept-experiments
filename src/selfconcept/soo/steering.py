"""Steering-vector variant of self-other overlap.

Instead of training LoRA adapters until self/other o_proj activations match,
measure the mean activation difference v = E[a_self - a_other] once on the
base model (scripts/extract_steering.py), then intervene at inference:

    mode 'add':     h <- h - alpha * v
        alpha is in natural units: alpha=1 subtracts the full mean
        self-minus-other difference, pushing self-referencing activations
        toward their other-referencing counterparts.
    mode 'project': h <- h - alpha * (h . v_hat) v_hat
        removes the component along the self/other axis; alpha=1 deletes it
        entirely (directional ablation), fractional alpha attenuates it.

Applied at the same site the SOO loss trained on (one layer's
self_attn.o_proj output, every token position), so results are directly
comparable to the LoRA rounds.

Positional variants (PositionalSteering): a constant offset added at every
token also perturbs the model's *reading* of the prompt, and at the alpha
that flips the SOO task on Qwen3.8-27B any matched-norm offset collapses
multi-turn coding. `positions='response'` adds the vector only from the
current assistant turn's header onward (the generation-prompt suffix of the
chat template plus every generated token, i.e. the positions the 'last'
extraction convention read from); `positions='prompt'` is the complementary
diagnostic (context only, never the model's own turn). Earlier assistant
turns inside a multi-turn prompt count as context.
"""

import atexit
from contextlib import contextmanager
from pathlib import Path

import torch

from .activations import attn_out_proj, get_decoder_layers

TOKEN_MODES = ("last", "mean")
POSITION_MODES = ("all", "response", "prompt")


def load_vectors(path: str | Path) -> dict:
    """Load an extract_steering.py output: metadata plus per-layer vectors."""
    return torch.load(path, map_location="cpu", weights_only=True)


def get_vector(data: dict, layer: int, token_mode: str) -> torch.Tensor:
    """One layer's self-minus-other vector (float32, [hidden_size])."""
    if token_mode not in TOKEN_MODES:
        raise ValueError(f"token_mode must be one of {TOKEN_MODES}, got {token_mode!r}")
    return data[f"vectors_{token_mode}"][layer]


def random_matched_vector(vector: torch.Tensor, seed: int) -> torch.Tensor:
    """Random Gaussian direction scaled to the real vector's norm.

    Damage control: separates "this direction matters" from "any perturbation
    of this magnitude at this layer changes behavior".
    """
    gen = torch.Generator().manual_seed(seed)
    rand = torch.randn(vector.shape, generator=gen, dtype=torch.float32)
    return rand / rand.norm() * vector.norm()


def response_marker(tokenizer, chat_kwargs: dict | None = None) -> list[int]:
    """Token ids the chat template appends for add_generation_prompt=True.

    E.g. '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' for Qwen3.x with
    thinking off, '<|turn>model\\n<|channel>thought\\n<channel|>' for gemma-4.
    The last occurrence in a prompt marks the assistant turn being generated.
    """
    kwargs = chat_kwargs or {}
    messages = [{"role": "user", "content": "x"}]

    def ids(gen: bool) -> list[int]:
        enc = tokenizer.apply_chat_template(
            messages, add_generation_prompt=gen, tokenize=True, return_dict=True, **kwargs
        )
        return list(enc["input_ids"])

    without, with_ = ids(False), ids(True)
    if with_[: len(without)] != without or len(with_) == len(without):
        raise ValueError("chat template's generation prompt is not a suffix of the no-generation rendering")
    return with_[len(without) :]


def _last_marker_start(input_ids: torch.Tensor, marker: list[int]) -> torch.Tensor:
    """Per row, index where the last occurrence of `marker` starts; seq_len if absent."""
    batch, seq = input_ids.shape
    k = len(marker)
    if seq < k:
        return torch.full((batch,), seq, device=input_ids.device, dtype=torch.long)
    m = torch.tensor(marker, device=input_ids.device, dtype=input_ids.dtype)
    hit = (input_ids.unfold(1, k, 1) == m).all(-1)  # [batch, seq-k+1]
    last = (seq - k) - hit.flip(1).int().argmax(1)
    return torch.where(hit.any(1), last, torch.full_like(last, seq))


def _past_length(kwargs: dict) -> int:
    """Tokens already in the KV cache for this forward call (0 for a fresh sequence).

    transformers 5.x generate() passes `past_key_values` (a Cache) and
    `position_ids` to the top-level forward but not `cache_position`, so read
    the cache's length; honor `cache_position` when a caller does pass it.
    """
    cache_position = kwargs.get("cache_position")
    if cache_position is not None:
        return int(cache_position[0])
    cache = kwargs.get("past_key_values")
    if cache is None:
        return 0
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    raise RuntimeError(f"positional steering: cannot read the length of cache type {type(cache).__name__}")


class PositionalSteering:
    """Restrict a steering hook to the model's own turn (or to the context).

    A forward pre-hook on the model records, per forward call, the absolute
    positions of the tokens being processed (cache length + arange(seq);
    arange(seq) for a plain cache-less forward) and, at the start of each
    sequence, where the current assistant turn begins (last occurrence of the
    chat template's generation-prompt suffix; seq_len when absent, i.e.
    nothing is 'response'). The o_proj hook then masks the offset per
    position: 'response' steers positions >= start, 'prompt' positions < start.
    """

    def __init__(self, marker: list[int], mode: str):
        if mode not in POSITION_MODES:
            raise ValueError(f"positions must be one of {POSITION_MODES}, got {mode!r}")
        if not marker:
            raise ValueError("empty response marker")
        self.marker = marker
        self.mode = mode
        self.positions: torch.Tensor | None = None
        self.start: torch.Tensor | None = None
        self.n_sequences = 0
        self.n_missing = 0
        self._handle = None

    def attach(self, model) -> None:
        self._handle = model.register_forward_pre_hook(self._pre_hook, with_kwargs=True)
        atexit.register(self.report)

    def _pre_hook(self, _module, args, kwargs) -> None:
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if input_ids is None or input_ids.dim() != 2:
            self.positions = None
            return
        seq = input_ids.shape[1]
        past = _past_length(kwargs)
        positions = past + torch.arange(seq, device=input_ids.device)
        if past == 0:
            if seq < len(self.marker):
                # generate() always prefills a full chat prompt; a "fresh"
                # 1-token sequence means the cache length was not readable
                # and decode steps would be mis-steered as new prompts.
                raise RuntimeError(f"positional steering: {seq}-token sequence with no past (cache length unreadable?)")
            self.start = _last_marker_start(input_ids, self.marker)
            self.n_sequences += 1
            missing = int((self.start >= seq).sum())
            self.n_missing += missing
            if self.n_sequences == 1:
                print(
                    f"Positional steering ({self.mode}): marker {len(self.marker)} tokens, "
                    f"first sequence start {self.start.tolist()} of {seq}",
                    flush=True,
                )
            if missing and self.n_missing <= 5:
                print(f"Positional steering: marker absent in a {seq}-token sequence (treated as all prompt)", flush=True)
        self.positions = positions

    def mask(self, output: torch.Tensor) -> torch.Tensor:
        """[batch, seq, 1] multiplier for the steering offset."""
        batch, seq = output.shape[0], output.shape[1]
        if self.positions is None or self.start is None or self.positions.shape[0] != seq or self.start.shape[0] != batch:
            raise RuntimeError(
                f"positional steering lost alignment: output {tuple(output.shape)}, "
                f"positions {None if self.positions is None else tuple(self.positions.shape)}, "
                f"start {None if self.start is None else tuple(self.start.shape)}"
            )
        positions = self.positions.to(output.device)
        start = self.start.to(output.device)
        response = positions[None, :] >= start[:, None]  # [batch, seq]
        keep = response if self.mode == "response" else ~response
        return keep.to(output.dtype)[..., None]

    def report(self) -> None:
        print(
            f"Positional steering ({self.mode}): {self.n_sequences} sequences, "
            f"marker missing in {self.n_missing}",
            flush=True,
        )


def steer_o_proj(
    model,
    layer: int,
    vector: torch.Tensor,
    alpha: float,
    mode: str = "add",
    positions: PositionalSteering | None = None,
):
    """Register a steering hook on one layer's o_proj output; returns the handle.

    The hook stays active for the model's lifetime unless the handle is
    removed — use apply_steering() when a scoped intervention is needed.
    With `positions`, the offset is masked per token position (see
    PositionalSteering); its pre-hook is attached to `model` here.
    """
    module = attn_out_proj(get_decoder_layers(model)[layer])
    vector = vector.float()
    unit = vector / vector.norm()
    if positions is not None and positions.mode == "all":
        positions = None
    if positions is not None:
        positions.attach(model)

    def hook(_module, _inputs, output):
        v = vector.to(output.device, output.dtype)
        if mode == "add":
            delta = alpha * v
        elif mode == "project":
            u = unit.to(output.device, output.dtype)
            delta = alpha * (output * u).sum(-1, keepdim=True) * u
        else:
            raise ValueError(f"unknown steering mode {mode!r}")
        if positions is not None:
            delta = delta * positions.mask(output)
        return output - delta

    return module.register_forward_hook(hook)


@contextmanager
def apply_steering(
    model,
    layer: int,
    vector: torch.Tensor,
    alpha: float,
    mode: str = "add",
    positions: PositionalSteering | None = None,
):
    """Context-managed steer_o_proj (removes both hooks on exit)."""
    handle = steer_o_proj(model, layer, vector, alpha, mode, positions)
    try:
        yield
    finally:
        handle.remove()
        if positions is not None and positions._handle is not None:
            positions._handle.remove()
