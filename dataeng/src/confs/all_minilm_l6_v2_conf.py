from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray
from transformers import AutoTokenizer
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

from src.confs import envs_conf


@dataclass(frozen=True)
class AllMiniLML6V2Conf:
    envs_conf: envs_conf.EnvsConf = envs_conf.impl

    _tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained(
        envs_conf.embedding_model_dirpath,
        clean_up_tokenization_spaces=True,
    )  # type: ignore
    _ort_sess: ort.InferenceSession = ort.InferenceSession(
        os.path.join(envs_conf.embedding_model_dirpath, "onnx/model.onnx"),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        texts_tokenized = self._tokenizer(
            list(texts),
            return_tensors="np",
            max_length=128,
            stride=64,
            return_overflowing_tokens=True,
        )
        input_ids: NDArray[np.int64] = texts_tokenized["input_ids"]  # type: ignore
        print(input_ids.shape, input_ids.shape)
        attention_mask: NDArray[np.int64] = texts_tokenized["attention_mask"]  # type: ignore
        sentence_embedding: NDArray[np.float32] = self._ort_sess.run(
            ["sentence_embedding"],
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )
        return sentence_embedding


impl = AllMiniLML6V2Conf()
