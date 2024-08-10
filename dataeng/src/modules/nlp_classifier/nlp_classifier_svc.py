from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import lightning as L
import onnx
import torch
from loguru import logger

from src.confs import envs_conf, spacy_conf
from src.modules.llm_prep.llm_row_vo import LLMRowVo
from src.modules.nlp_classifier.classifier_data_mod import ClassifierDataMod
from src.modules.nlp_classifier.classifier_mod import ClassifierMod
from src.modules.nlp_classifier.prepare_data_svc import prepare_data_svc
from src.modules.vocabulary_prep.vocabulary_vo import VocabularyVo


@dataclass
class NLPClassifierSvc:
    envs_conf: envs_conf.EnvsConf = envs_conf.impl
    spacy_conf: spacy_conf.SpacyConf = spacy_conf.impl
    prepare_data_svc: prepare_data_svc.PrepareDataSvc = prepare_data_svc.impl

    def train_and_export_classifier(
        self, llm_rows: Sequence[LLMRowVo], vocabulary: VocabularyVo
    ) -> None:
        model = self._get_and_train_model(llm_rows, vocabulary)
        self._save_as_onnx(model)

    def _get_and_train_model(
        self, llm_rows: Sequence[LLMRowVo], vocabulary: VocabularyVo
    ) -> ClassifierMod:
        logger.info("Initializing the data and model.")
        data = ClassifierDataMod(
            llm_rows, vocabulary, self.prepare_data_svc.prepare_data
        )
        model = ClassifierMod()
        logger.info("Training the model.")
        model_checkpoint = ClassifierMod.model_checkpoint()
        trainer = L.Trainer(
            default_root_dir=self.envs_conf.nlp_classifier_svc__get_and_train_model_cache_dirpath,
            max_epochs=-1,
            callbacks=[
                ClassifierMod.early_stopping(),
                model_checkpoint,
            ],
            logger=False,
        )
        trainer.fit(model, data)
        logger.info("Loading the checkpointed best model.")
        model = ClassifierMod.load_from_checkpoint(model_checkpoint.best_model_path)
        logger.info("Computing model final validation and test scores.")
        trainer.validate(model, data)
        trainer.test(model, data)
        return model

    def _save_as_onnx(self, model: ClassifierMod) -> None:
        logger.info("Saving classifier model in ONNX format.")
        os.makedirs(os.path.dirname(self.envs_conf.model_filepath), exist_ok=True)
        torch.onnx.export(
            model, torch.randn(1, 300), self.envs_conf.model_filepath, opset_version=20
        )  # https://github.com/onnx/onnx/blob/main/docs/Versioning.md
        onnx.checker.check_model(onnx.load(self.envs_conf.model_filepath))


impl = NLPClassifierSvc()
