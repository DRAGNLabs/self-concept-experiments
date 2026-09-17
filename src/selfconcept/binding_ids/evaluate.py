"""Binding task accuracy, with and without interventions."""

import json
from pathlib import Path
from typing import cast

import torch
from lightning_fabric import Fabric
from mirror.metrics.mirror_metric import MirrorMetric
from mirror.models.inference_model import InferenceModel
from mirror.models.trainable_model import TrainableModel

from selfconcept.common.paths import REPO_ROOT

from .activations import decoder_layers, encode_prompt, run_with_cache, run_with_patched_context
from .binding_vectors import estimate_binding_deltas
from .interventions import CONDITIONS, expected_answer, mean_intervention_offsets
from .tasks import load_rows


class MeanInterventionMetric(MirrorMetric):
    """Swap the two pairs' binding IDs with mean interventions and check whether answers follow.

    The first `n_estimate` contexts estimate the binding vectors; the rest are evaluated,
    querying each entity once. `accuracy_<condition>` is the fraction of queries where the
    model's answer matches what the binding-ID account predicts under that condition
    (original answer for none/swap_both, the other pair's answer for single swaps).
    Answers are scored by comparing the logits of each in-context capital's first token.
    """

    def __init__(self, data_path: str, n_estimate: int = 100, output_path: str | None = None) -> None:
        self.data_path = data_path
        self.n_estimate = n_estimate
        self.output_path = output_path

    def get_metrics(self, model: TrainableModel, fabric: Fabric) -> dict:
        inference_model = cast(InferenceModel, model)
        hf_model = inference_model.hf_model
        tokenizer = inference_model.formatter.tokenizer
        layers = decoder_layers(hf_model)

        rows = load_rows(REPO_ROOT / self.data_path)
        estimate_rows, eval_rows = rows[: self.n_estimate], rows[self.n_estimate :]

        correct = dict.fromkeys(CONDITIONS, 0)
        n_queries = n_skipped = 0
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
                    for condition in CONDITIONS:
                        if condition == "none":
                            logits = clean_logits
                        else:
                            offsets = mean_intervention_offsets(prompt, condition, delta_entity, delta_attribute)
                            logits = run_with_patched_context(
                                hf_model, layers, prompt.input_ids, clean, offsets, prompt.context_length
                            )
                        predicted = int(logits[candidates].argmax())
                        correct[condition] += predicted == expected_answer(condition, query_index)
                    n_queries += 1

        results = {
            "n_estimate_contexts": len(estimate_rows),
            "n_queries": n_queries,
            "n_skipped_contexts": n_skipped,
            **{f"accuracy_{c}": correct[c] / n_queries for c in CONDITIONS},
        }
        if self.output_path:
            out = REPO_ROOT / self.output_path
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(results, indent=2) + "\n")
        return results
