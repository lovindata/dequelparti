from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
from loguru import logger
from tqdm import tqdm

from src.confs import spacy_conf
from src.modules.llm_prep.llm_row_vo import LLMRowVo
from src.modules.nlp_classifier.prepare_data_svc import cache_svc
from src.modules.vocabulary_prep.vocabulary_vo import VocabularyVo


@dataclass(frozen=True)
class PrepareDataSvc:
    spacy_conf: spacy_conf.SpacyConf = spacy_conf.impl
    cache_svc: cache_svc.CacheSvc = cache_svc.impl

    def prepare_data(
        self, llm_rows: Sequence[LLMRowVo], vocabulary: VocabularyVo
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tensors = self.cache_svc.load_feature_and_label_tensors_from_disk(
            llm_rows, vocabulary
        )
        if tensors:
            logger.info("Feature and label tensors loaded from cache.")
            feature_tensor, label_tensor = tensors
            return feature_tensor, label_tensor
        feature_tensor = self._convert_llm_rows_to_feature_tensor(llm_rows, vocabulary)
        label_tensor = self._convert_llm_rows_to_label_tensor(llm_rows)
        self.cache_svc.save_feature_and_label_tensors_to_disk(
            llm_rows, vocabulary, feature_tensor, label_tensor
        )
        return feature_tensor, label_tensor

    def _convert_llm_rows_to_feature_tensor(
        self, llm_rows: Sequence[LLMRowVo], vocabulary: VocabularyVo
    ) -> torch.Tensor:
        logger.info("Converting feature from LLM rows to tensor.")
        tokens_per_feature = [
            list(spacy_conf.impl.spacy(llm_row.feature)) for llm_row in tqdm(llm_rows)
        ]
        tensor_per_feature = [
            torch.vstack(
                [
                    torch.FloatTensor(vocabulary.root[token.text])
                    for token in tokens
                    if token.text in vocabulary.root
                ]
            ).mean(dim=0)
            for tokens in tqdm(tokens_per_feature)
        ]
        tensor = torch.vstack(tensor_per_feature)
        return tensor

    def _convert_llm_rows_to_label_tensor(
        self,
        llm_rows: Sequence[LLMRowVo],
    ) -> torch.Tensor:
        logger.info("Converting label from LLM rows to tensor.")
        unique_labels = list(set([llm_row.label for llm_row in llm_rows]))
        tensor = torch.zeros(len(llm_rows), dtype=torch.long)
        for i, llm_row in enumerate(tqdm(llm_rows)):
            label_enc_idx = unique_labels.index(llm_row.label)
            tensor[i] = label_enc_idx
        return tensor


impl = PrepareDataSvc()
