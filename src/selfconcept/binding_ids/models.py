"""MIRROR model + formatter wrappers for OLMo 2, since MIRROR only ships Llama and GPT-2.

Parametrized on the HF model name so 7B and 13B (and future sizes) share one class.
"""

from typing import cast

import torch
import torch.nn as nn
import torch.optim as optim
from mirror.formatters.formatter_util import collate_tokens, load_hf_tokenizer
from mirror.formatters.infer_friendly_formatter import InferFriendlyFormatter
from mirror.formatters.mirror_formatter import MirrorFormatter
from mirror.models.inference_model import InferenceModel
from mirror.models.model_util import build_causal_lm
from mirror.models.trainable_model import TrainableModel
from mirror.models.whitebox_transformers.whitebox_transformers import WhiteboxTransformerExecutor
from mirror.types import LabeledTokens, Loss, StandardBatch, TextRow, TokenTensor
from mirror.util import _ds_cache_path_context
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from typed_datasets import TypedDataset

from mirror.datasets.mirror_dataset import MirrorDataset


class MirrorOlmoFormatter(
    InferFriendlyFormatter,
    MirrorFormatter[TextRow, LabeledTokens, StandardBatch],
):
    def __init__(self, hf_model_name: str, max_length: int | None = 2048) -> None:
        self._tokenizer: PreTrainedTokenizerBase = load_hf_tokenizer(hf_model_name)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._max_length = max_length

    def format_data(self, data_source: MirrorDataset[TextRow]) -> TypedDataset[LabeledTokens]:
        tokenizer = self._tokenizer
        max_length = self._max_length

        def tokenize(row: TextRow) -> LabeledTokens:
            ids = tokenizer.encode(
                row["text"],
                add_special_tokens=True,
                max_length=max_length,
                truncation=max_length is not None,
            )
            if len(ids) < 2:
                eos = tokenizer.eos_token_id
                ids = [eos, eos] if len(ids) == 0 else [*ids, eos]
            token_ids = cast(TokenTensor, ids)
            return LabeledTokens(input_ids=token_ids, labels=list(token_ids))

        with _ds_cache_path_context():
            return data_source.ds.map(tokenize, remove_columns=list(data_source.ds.columns))

    def collate(self, examples: list[LabeledTokens]) -> StandardBatch:
        return collate_tokens(self._tokenizer, examples)

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    @property
    def pad_token_id(self) -> int:
        return int(self._tokenizer.pad_token_id)  # type: ignore[arg-type]


class MirrorOlmoModel(
    TrainableModel[TextRow, LabeledTokens, StandardBatch],
    InferenceModel[TextRow, LabeledTokens, StandardBatch, torch.Tensor],
):
    def __init__(self, hf_model_name: str = "allenai/OLMo-2-1124-7B") -> None:
        super().__init__()
        self._hf_model = cast(PreTrainedModel, build_causal_lm(hf_model_name, weights="pretrained"))
        self._formatter = MirrorOlmoFormatter(hf_model_name)

    @property
    def hf_model(self) -> PreTrainedModel:
        return self._hf_model

    @property
    def formatter(self) -> MirrorOlmoFormatter:
        return self._formatter

    def forward(self, batch: StandardBatch) -> torch.Tensor:
        input_ids, attention_mask, _ = batch
        return WhiteboxTransformerExecutor.fresh(self).execute((input_ids, attention_mask)).logits

    def training_step(self, batch: StandardBatch) -> Loss:
        input_ids, attention_mask, labels = batch
        return WhiteboxTransformerExecutor.fresh(self).include_loss(labels).execute((input_ids, attention_mask)).loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters())

    def mlp_modules(self) -> list[nn.Module]:
        return [cast(nn.Module, layer.mlp) for layer in self._hf_model.model.layers]
