from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import lightning as L

from src.confs import envs_conf, spacy_conf
from src.modules.llm_prep.llm_row_vo import LLMRowVo
from src.modules.nlp_classifier.classifier_mod import ClassifierMod
from src.modules.nlp_classifier.prepare_data_svc import prepare_data_svc
from src.modules.vocabulary_prep.vocabulary_vo import VocabularyVo


@dataclass
class NLPClassifierSvc:
    envs_conf: envs_conf.EnvsConf = envs_conf.impl
    spacy_conf: spacy_conf.SpacyConf = spacy_conf.impl
    prepare_data_svc: prepare_data_svc.PrepareDataSvc = prepare_data_svc.impl

    def build_and_train_classifier(
        self, llm_rows: Sequence[LLMRowVo], vocabulary: VocabularyVo
    ) -> ClassifierMod:
        model = ClassifierMod(
            llm_rows,
            vocabulary,
            self.prepare_data_svc.prepare_data,
        )
        """
        # early_stop_callback_f1 = EarlyStopping(
        #     monitor="val_f1",  # Monitor validation F1 score
        #     min_delta=0.00,
        #     patience=3,
        #     verbose=True,
        #     mode="max",  # 'max' because we want to maximize F1 score
        # )
        """
        trainer = L.Trainer(max_epochs=-1, logger=False, enable_checkpointing=False)
        trainer.fit(model)
        return model


impl = NLPClassifierSvc()
