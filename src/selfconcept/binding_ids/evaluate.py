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
    condition as in the paper; `raw_accuracy_<condition>` skips calibration. `_e0`/`_e1`
    suffixes split by which entity was queried.
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

        # Per condition, one (query_index, log prob of slot-0 answer, of slot-1 answer) per query
        scores: dict[str, list[tuple[int, float, float]]] = {c: [] for c in CONDITIONS}
        n_skipped = 0
        with torch.no_grad():
            delta_entity, delta_attribute = estimate_binding_deltas(
                hf_model, layers, tokenizer, estimate_rows, fabric.device
            )
            for row in eval_rows:
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
                        log_probs = logits.float().log_softmax(-1)[candidates].tolist()
                        scores[condition].append((query_index, *log_probs))

        results = {
            "n_estimate_contexts": len(estimate_rows),
            "n_eval_contexts": len(eval_rows),
            "n_queries": len(scores["none"]),
            "n_skipped_contexts": n_skipped,
        }
        for condition in CONDITIONS:
            results.update(_accuracies(condition, scores[condition]))
        if self.output_path:
            out = REPO_ROOT / self.output_path
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(results, indent=2) + "\n")
        return results


def _accuracies(condition: str, scores: list[tuple[int, float, float]]) -> dict[str, float]:
    """Raw and median-calibrated accuracy for one condition, overall and per queried entity."""
    query_index = torch.tensor([s[0] for s in scores])
    log_probs = torch.tensor([s[1:] for s in scores])  # (n_queries, 2 answer slots)
    # Subtracting each slot's median removes the model's prior for an answer slot (e.g. the
    # later-mentioned attribute) independent of binding
    calibrated = log_probs - log_probs.median(dim=0).values
    expected = torch.tensor([expected_answer(condition, int(q)) for q in query_index])

    results = {}
    for prefix, values in (("accuracy", calibrated), ("raw_accuracy", log_probs)):
        correct = (values.argmax(dim=1) == expected).float()
        results[f"{prefix}_{condition}"] = correct.mean().item()
        for q in (0, 1):
            results[f"{prefix}_{condition}_e{q}"] = correct[query_index == q].mean().item()
    return results
