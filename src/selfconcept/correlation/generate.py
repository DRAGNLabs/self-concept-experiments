"""Generate while scoring every processed token with one residual-stream measurement."""
import hashlib
import json

import numpy as np
import torch

from selfconcept.codebench import harness
from selfconcept.measurement.capture import capture
from selfconcept.measurement.interface import Measurement
from selfconcept.measurement.templates import clean_final, content_indices, render_generation, response_spans


def append(path, record):
    with path.open("a") as f:
        f.write(json.dumps(record, allow_nan=False) + "\n")
        f.flush()


def score_summary(values, indices):
    selected = values[[i for i in indices if i < len(values)]]
    return {"n": len(selected), "mean": float(selected.mean()) if len(selected) else None,
            "p90": float(np.quantile(selected, .9)) if len(selected) else None}


def token_offsets(tokenizer, ids, text):
    # Single-token decoding is exact for ordinary text, but not split UTF-8.
    pieces = [tokenizer.decode([i], skip_special_tokens=False, clean_up_tokenization_spaces=False) for i in ids]
    if "".join(pieces) == text:
        ends = np.cumsum([len(p) for p in pieces]).tolist()
        return list(zip([0] + ends[:-1], ends))
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if enc["input_ids"] == ids:
        return enc["offset_mapping"]
    return None  # Never assign an activation to the wrong token.


def generation_seed(seed, scenario, example_id, turn):
    return int.from_bytes(hashlib.sha256(f"{seed}:{scenario}:{example_id}:{turn}".encode()).digest()[:4], "big")


class MeasuredGenerator:
    def __init__(self, model, tokenizer, family, measurement: Measurement, out, max_new_tokens,
                 temperature=0.0, top_p=0.95, top_k=64, seed=1729):
        self.model, self.tokenizer, self.family = model, tokenizer, family
        self.out, self.max_new_tokens = out, max_new_tokens
        self.records = []
        self.scenario = ""
        self.sampling = ({"do_sample": True, "temperature": temperature, "top_p": top_p, "top_k": top_k}
                         if temperature > 0 else {"do_sample": False})
        self.seed = seed
        self.measurement = measurement

    def __call__(self, messages, turn, example_id):
        tok = self.tokenizer
        prompt = render_generation(tok, self.family, messages)
        enc = tok(prompt, add_special_tokens=False, return_tensors="pt").to(self.model.get_input_embeddings().weight.device)
        context_limit = getattr(getattr(self.model.config, "text_config", self.model.config), "max_position_embeddings", None)
        if context_limit and enc.input_ids.shape[1] + self.max_new_tokens > context_limit:
            raise ValueError("Prompt plus generation budget exceeds model context; refusing silent truncation")
        try:
            seed = generation_seed(self.seed, self.scenario, example_id, turn)
            torch.manual_seed(seed)
            with torch.inference_mode(), capture(self.model, self.measurement.layers, self.measurement.score) as captured:
                output = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, **self.sampling,
                                             pad_token_id=tok.eos_token_id, use_cache=True)
        except torch.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise harness.GenerationOOM(str(exc)[:200]) from None
        n_prompt = enc.input_ids.shape[1]
        ids = output[0].tolist()
        new_ids = ids[n_prompt:]
        raw = tok.decode(new_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        spans = response_spans(prompt, raw, self.family)
        final = clean_final(raw, spans, tok)
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        truncated = len(new_ids) >= self.max_new_tokens and new_ids[-1] not in eos
        status = "truncated" if truncated else "complete" if final else "no_final"
        prompt_enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
        # Measure only content from the last externally supplied message, excluding
        # assistant history, system instructions and generation-role delimiters.
        # Native templates may trim user content (notably coding feedback's
        # leading newline). Match the shared non-whitespace content exactly.
        last = messages[-1]["content"].strip()
        start = prompt.rfind(last)
        if start < 0 or prompt_enc.input_ids != ids[:n_prompt]:
            raise ValueError("Cannot exactly locate the last message in the rendered prompt")
        p_indices = content_indices(prompt_enc.offset_mapping, [(start, start+len(last))], tok.all_special_ids, ids[:n_prompt])
        offsets = token_offsets(tok, new_ids, raw)
        groups = {"prompt": p_indices}
        if offsets is not None:
            groups.update({role: [n_prompt+i for i in content_indices(offsets, segments, tok.all_special_ids, new_ids)]
                           for role, segments in spans.items()})
        scores, arrays = {}, {"input_ids": np.asarray(ids, dtype=np.int32), "prompt_length": np.array(n_prompt)}
        for layer, chunks in captured.items():
            values = torch.cat(chunks).numpy()
            if len(values) != len(ids)-1:
                raise ValueError(f"Unexpected cached generation alignment: {len(values)} vs {len(ids)-1}")
            scores[str(layer)] = {name: score_summary(values, idx) for name, idx in groups.items()}
            arrays[f"{self.measurement.name.replace('-', '_')}_layer_{layer}"] = values
        for name, indices in groups.items():
            arrays[f"indices_{name}"] = np.asarray([i for i in indices if i < len(ids)-1], dtype=np.int32)
        trace_key = hashlib.sha256(f"{self.scenario}:{example_id}:{turn}".encode()).hexdigest()[:24]
        trace = self.out / "traces" / f"{trace_key}.npz"
        trace.parent.mkdir(exist_ok=True)
        np.savez_compressed(trace, **arrays)
        rec = {"example_id": example_id, "scenario": self.scenario, "turn": turn,
               "response": final, "raw_response": raw, "rendered_prompt": prompt,
               "status": status, "truncated": truncated, "generated_tokens": len(new_ids),
               "sampling": self.sampling, "seed": seed,
               "prompt_tokens": n_prompt, "spans": spans, "scores": scores,
               "generated_alignment": "exact" if offsets is not None else "unavailable",
               "trace": str(trace.relative_to(self.out)), "measurement": self.measurement.name}
        self.records.append(rec)
        append(self.out / "generations.jsonl", rec)
        # A partial final answer/code is never a completed behavioral submission.
        return final if status == "complete" else "", truncated
