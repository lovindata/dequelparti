from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import onnxruntime as ort
from loguru import logger
from numpy.typing import NDArray
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

from src.confs import envs_conf
from src.modules.shared.atomic_vos.atoms.decoded_embedding_vo import DecodedEmbeddingVo
from src.modules.shared.atomic_vos.atoms.id_vo import IdVo
from src.modules.shared.atomic_vos.atoms.vector_embedding_vo import VectorEmbeddingVo
from src.modules.shared.atomic_vos.molecules.embedding_row_vo import EmbeddingRowVo
from src.modules.shared.atomic_vos.molecules.pdf_vo import PdfVo
from src.modules.shared.atomic_vos.organisms.embedding_table_vo import EmbeddingTableVo
from src.modules.vector_database import vector_database_rep


@dataclass(frozen=True)
class VectorDatabaseSvc:
    envs_conf: envs_conf.EnvsConf = envs_conf.impl
    vector_database_rep: vector_database_rep.VectorDatabaseRep = (
        vector_database_rep.impl
    )

    _tokenizer: (
        BertTokenizerFast
    ) = AutoTokenizer.from_pretrained(  # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
        envs_conf.embedding_model_dirpath,
        clean_up_tokenization_spaces=True,
    )  # type: ignore
    _ort_sess: ort.InferenceSession = ort.InferenceSession(
        os.path.join(envs_conf.embedding_model_dirpath, "onnx/model.onnx"),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    def build_vector_database(self, pdfs: Sequence[PdfVo]) -> None:
        logger.info("Building vector database.")
        embedding_table: EmbeddingTableVo = []
        id: IdVo = 0
        for pdf in pdfs:
            for page in tqdm(pdf.pages):
                for decoded_embedding, vector_embedding in self._embed(page):
                    embedding_table.append(
                        EmbeddingRowVo(
                            id, decoded_embedding, vector_embedding, pdf.name
                        )
                    )
                    id += 1
        self.vector_database_rep.save(embedding_table)

    def _embed(self, text: str) -> List[Tuple[DecodedEmbeddingVo, VectorEmbeddingVo]]:
        text_tokenized = self._tokenizer(
            text,
            return_tensors="np",
            return_overflowing_tokens=True,
            truncation=True,
            padding=True,
            max_length=self.envs_conf.embedding_model_max_length,
            stride=self.envs_conf.embedding_model_stride,
        )
        input_ids: NDArray[np.int64] = text_tokenized["input_ids"]  # type: ignore
        attention_mask: NDArray[np.int64] = text_tokenized["attention_mask"]  # type: ignore
        sentence_embedding: NDArray[np.float32] = self._ort_sess.run(
            ["sentence_embedding"],
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )[0]
        vector_embeddings: List[List[float]] = sentence_embedding.tolist()
        decoded_embeddings = self._tokenizer.batch_decode(input_ids)
        return list(zip(decoded_embeddings, vector_embeddings, strict=True))


impl = VectorDatabaseSvc()
