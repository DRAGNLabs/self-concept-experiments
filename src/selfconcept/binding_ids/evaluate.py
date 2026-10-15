"""Binding task accuracy, with and without interventions."""

import json
from typing import cast

import torch
from lightning_fabric import Fabric
from mirror.metrics.mirror_metric import MirrorMetric
from mirror.models.inference_model import InferenceModel
from mirror.models.trainable_model import TrainableModel

from selfconcept.common.paths import REPO_ROOT

from .activations import decoder_layers, encode_prompt, run_with_cache, run_with_patched_context
from .binding_vectors import estimate_binding_deltas
from .interventions import CONDITIONS, expected_answer, mean_intervention_offsets, random_like_norm
from .tasks import load_rows


class MeanInterventionMetric(MirrorMetric):
    """Swap the two pairs' binding IDs with mean interventions and check whether answers follow.

    Binding vectors are estimated on `estimate_data_path`; contexts
    `data_path[eval_start : eval_start + n_eval]` are evaluated, querying each entity once.
    `accuracy_<condition>` is the fraction of queries where the model's answer matches what
    the binding-ID account predicts under that condition (original answer for none, swap_both
    and the random_* controls; the other pair's answer for single swaps). Answers are scored
    by each in-context capital's first-token log prob, median-calibrated per answer slot and
    condition as in the paper. Variants:
      - `strict_accuracy_*`: calibrated with the medians from the `none` condition only, so a
        slot bias the intervention itself introduces isn't calibrated away
      - `raw_accuracy_*`: no calibration; the model prefers the expected in-context capital
      - `vocab_accuracy_*`: the top-1 token over the whole vocabulary is the first token of the
        expected pair's capital or country, i.e. what the model would actually generate (some
        models, e.g. OLMo-2-7B, answer "the city of" with the country)
    `_e0`/`_e1` suffixes split by which entity was queried. Per-query log probs and top
    tokens are written next to `output_path` as `<stem>_queries.jsonl`.
    """

    def __init__(
        self,
        data_path: str,
        estimate_data_path: str,
        eval_start: int = 0,
        n_eval: int | None = None,
        seed: int = 0,
        output_path: str | None = None,
    ) -> None:
        self.data_path = data_path
        self.estimate_data_path = estimate_data_path
        self.eval_start = eval_start
        self.n_eval = n_eval
        self.seed = seed
        self.output_path = output_path

    def get_metrics(self, model: TrainableModel, fabric: Fabric) -> dict:
        inference_model = cast(InferenceModel, model)
        hf_model = inference_model.hf_model
        tokenizer = inference_model.formatter.tokenizer
        layers = decoder_layers(hf_model)

        estimate_rows = load_rows(REPO_ROOT / self.estimate_data_path)
        eval_rows = load_rows(REPO_ROOT / self.data_path)[self.eval_start :]
        eval_rows = eval_rows[: self.n_eval] if self.n_eval is not None else eval_rows
        generator = torch.Generator().manual_seed(self.seed)

        records: list[dict] = []  # one per (query, condition)
        n_skipped = 0
        with torch.no_grad():
            delta_entity, delta_attribute = estimate_binding_deltas(
                hf_model, layers, tokenizer, estimate_rows, fabric.device
            )
            for row_index, row in enumerate(eval_rows, start=self.eval_start):
                candidates = [tokenizer.encode(" " + a, add_special_tokens=False)[0] for a in row["answers"]]
                if candidates[0] == candidates[1]:
                    n_skipped += 1  # first-token scoring can't tell these answers apart
                    continue
                for query_index in (0, 1):
                    prompt = encode_prompt(tokenizer, row, query_index, fabric.device)
                    clean_logits, clean = run_with_cache(hf_model, layers, prompt.input_ids)
                    random_entity = random_like_norm(delta_entity, generator)
                    random_attribute = random_like_norm(delta_attribute, generator)
                    for condition in CONDITIONS:
                        if condition == "none":
                            logits = clean_logits
                        else:
                            is_random = condition.startswith("random_")
                            offsets = mean_intervention_offsets(
                                prompt,
                                condition,
                                random_entity if is_random else delta_entity,
                                random_attribute if is_random else delta_attribute,
                            )
                            logits = run_with_patched_context(
                                hf_model, layers, prompt.input_ids, clean, offsets, prompt.context_length
                            )
                        top_token = int(logits.argmax())
                        records.append(
                            {
                                "row": row_index,
                                "query_index": query_index,
                                "condition": condition,
                                "answers": row["answers"],
                                "attributes": row["attributes"],
                                "log_probs": logits.float().log_softmax(-1)[candidates].tolist(),
                                "top_token": tokenizer.decode([top_token]),
                                "top_slot": top_slot(tokenizer, row, top_token),
                            }
                        )

        results = {
            "n_estimate_contexts": len(estimate_rows),
            "n_eval_contexts": len(eval_rows),
            "n_queries": sum(r["condition"] == "none" for r in records),
            "n_skipped_contexts": n_skipped,
            **summarize(records),
        }
        if self.output_path:
            out = REPO_ROOT / self.output_path
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(results, indent=2) + "\n")
            with open(out.with_name(f"{out.stem}_queries.jsonl"), "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in records)
        return results


def top_slot(tokenizer, row: dict, token_id: int) -> int | None:
    """Answer slot whose capital or country starts with `token_id`, if any."""
    for slot in (0, 1):
        for word in (row["answers"][slot], row["attributes"][slot]):
            if tokenizer.encode(" " + word, add_special_tokens=False)[0] == token_id:
                return slot
    return None


def summarize(records: list[dict]) -> dict[str, float]:
    """All accuracy variants for every condition, from per-query records (see the metric docstring)."""
    by_condition = {c: [r for r in records if r["condition"] == c] for c in CONDITIONS}
    baseline_medians = _log_probs(by_condition["none"]).median(dim=0).values
    results = {}
    for condition, rows in by_condition.items():
        query_index = torch.tensor([r["query_index"] for r in rows])
        log_probs = _log_probs(rows)
        expected = torch.tensor([expected_answer(condition, r["query_index"]) for r in rows])
        # Subtracting each slot's median removes the model's prior for an answer slot (e.g. the
        # later-mentioned attribute) independent of binding
        predictions = {
            "accuracy": (log_probs - log_probs.median(dim=0).values).argmax(dim=1),
            "strict_accuracy": (log_probs - baseline_medians).argmax(dim=1),
            "raw_accuracy": log_probs.argmax(dim=1),
            "vocab_accuracy": torch.tensor([-1 if r["top_slot"] is None else r["top_slot"] for r in rows]),
        }
        for prefix, predicted in predictions.items():
            correct = (predicted == expected).float()
            results[f"{prefix}_{condition}"] = correct.mean().item()
            for q in (0, 1):
                results[f"{prefix}_{condition}_e{q}"] = correct[query_index == q].mean().item()
    return results


def _log_probs(rows: list[dict]) -> torch.Tensor:
    return torch.tensor([r["log_probs"] for r in rows])  # (n_queries, 2 answer slots)
