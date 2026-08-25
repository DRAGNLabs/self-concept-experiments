"""SOO fine-tuning: LoRA training that minimizes MSE between self- and
other-referencing activations at one self_attn.o_proj output.

Usage:
    python -m selfconcept.soo.train --config configs/olmo2-1b.yaml [--seeds 0 1 2 3 4]

Per the paper (section 3.1.1): two forward passes per pair (self prompt, other
prompt), SOO loss = MSE(A_self, A_other) at the configured layer, LoRA adapters
on q_proj/v_proj only, no task loss. Chat template applied with generation
prompt, matching the paper's setup.

The paper does not specify how the MSE handles self/other prompts tokenizing to
different lengths; default mode 'full' right-pads both to a common length and
takes elementwise MSE over the padded tensors, 'last_token' compares only the
final non-pad position (ablation).
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoTokenizer

from .activations import capture_o_proj
from .loading import load_causal_lm
from .evaluate import pick_device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def chat_text(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


def encode_batch(tokenizer, prompts: list[str], max_len: int, device: str):
    """Chat-template each prompt, right-pad to a common max_len."""
    texts = [chat_text(tokenizer, p) for p in prompts]
    return tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        max_length=max_len,
        add_special_tokens=False,
    ).to(device)


def soo_loss(a_self: torch.Tensor, a_other: torch.Tensor, mode: str, self_mask, other_mask):
    if mode == "full":
        return torch.nn.functional.mse_loss(a_self, a_other)
    if mode == "last_token":
        self_last = a_self[torch.arange(len(a_self)), self_mask.sum(1) - 1]
        other_last = a_other[torch.arange(len(a_other)), other_mask.sum(1) - 1]
        return torch.nn.functional.mse_loss(self_last, other_last)
    raise ValueError(mode)


def train_one_seed(config: dict, seed: int, device: str, out_dir: Path) -> dict:
    set_seed(seed)
    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if config.get("quant_4bit"):
        # Match the paper's QLoRA setup (4-bit base, LoRA adapters on top).
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = load_causal_lm(config["model"], quantization_config=bnb, dtype=dtype)
        # No gradient checkpointing: its reentrant mode severs the graph for
        # hook-captured activations (our loss), raising "does not require grad".
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    else:
        model = load_causal_lm(config["model"], dtype=dtype)
    lora = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config.get("target_modules", ["q_proj", "v_proj"]),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    if not config.get("quant_4bit"):
        model.to(device)
    model.train()

    pairs = [json.loads(l) for l in Path(config["train_data"]).open()]
    if config.get("max_pairs"):
        pairs = pairs[: config["max_pairs"]]
    # Common padded length across the whole dataset so every batch is comparable.
    max_len = max(
        len(tokenizer(chat_text(tokenizer, p), add_special_tokens=False)["input_ids"])
        for pair in pairs
        for p in (pair["self_prompt"], pair["other_prompt"])
    )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=config["lr"]
    )
    batch_size = config["batch_size"]
    layer = config["layer"]
    mode = config.get("soo_mode", "full")
    rng = random.Random(seed)

    losses = []
    for epoch in range(config["epochs"]):
        rng.shuffle(pairs)
        epoch_losses = []
        for i in tqdm(range(0, len(pairs), batch_size), desc=f"seed{seed} epoch{epoch}"):
            batch = pairs[i : i + batch_size]
            self_enc = encode_batch(tokenizer, [b["self_prompt"] for b in batch], max_len, device)
            other_enc = encode_batch(tokenizer, [b["other_prompt"] for b in batch], max_len, device)

            store = []
            with capture_o_proj(model, layer, store):
                model(**self_enc)
                model(**other_enc)
            a_self, a_other = store
            loss = soo_loss(
                a_self, a_other, mode, self_enc["attention_mask"], other_enc["attention_mask"]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        mean_loss = float(np.mean(epoch_losses))
        losses.append(mean_loss)
        print(f"seed {seed} epoch {epoch}: soo_loss={mean_loss:.6f}", flush=True)

    seed_dir = out_dir / f"seed{seed}"
    model.save_pretrained(seed_dir)
    with (seed_dir / "train_log.json").open("w") as f:
        json.dump({"config": config, "seed": seed, "epoch_losses": losses}, f, indent=2)
    return {"seed": seed, "final_loss": losses[-1], "adapter": str(seed_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--out", type=Path, default=Path("results/checkpoints"))
    parser.add_argument("--layer", type=int, help="override config layer (layer sweep)")
    parser.add_argument("--epochs", type=int, help="override config epochs (smoke tests)")
    parser.add_argument("--max-pairs", type=int, help="truncate training pairs (smoke tests)")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    if args.layer is not None:
        config["layer"] = args.layer
    if args.epochs:
        config["epochs"] = args.epochs
    if args.max_pairs:
        config["max_pairs"] = args.max_pairs
    device = pick_device()
    stem = args.config.stem + (f"-L{args.layer}" if args.layer is not None else "")
    out_dir = args.out / stem
    print(f"Training {config['model']} on {device}, layer {config['layer']}, seeds {args.seeds}")

    results = [train_one_seed(config, seed, device, out_dir) for seed in args.seeds]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
