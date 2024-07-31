from __future__ import annotations

import os
import random
from dataclasses import dataclass
from hashlib import sha3_512
from typing import Dict, List, Literal, Sequence

from loguru import logger
from tqdm import tqdm

from src.confs import envs_conf, ollama_conf, spacy_conf
from src.modules.file_system import file_system_svc
from src.modules.file_system.pdf_vo.pdf_vo import PdfVo
from src.modules.llm_prep.llm_rows_vo import LLMRowsVo


@dataclass(frozen=True)
class LLMPrepSvc:
    envs_conf: envs_conf.EnvsConf = envs_conf.impl
    ollama_conf: ollama_conf.OllamaConf = ollama_conf.impl
    spacy_conf: spacy_conf.SpacyConf = spacy_conf.impl
    file_system_svc: file_system_svc.FileSystemSvc = file_system_svc.impl

    def compute_llm_rows(self, pdfs: Sequence[PdfVo]) -> List[LLMRowsVo]:
        texts_per_pdf = [
            [page.text for page in pdf.pages if self._has_enough_words(page.text)]
            for pdf in pdfs
        ]
        logger.info("Computing positive consequences for all pages of each PDF.")
        positive_csqs_per_pdf = [
            LLMRowsVo.concat_all(
                [self._compute_csqs_via_text(text, "positive") for text in tqdm(texts)]
            )
            for texts in texts_per_pdf
        ]
        logger.info("Computing negative consequences for all pages of each PDF.")
        negative_csqs_per_pdf = [
            LLMRowsVo.concat_all(
                [self._compute_csqs_via_text(text, "negative") for text in tqdm(texts)]
            )
            for texts in texts_per_pdf
        ]
        logger.info("Computing reformulations for all consequences of each PDF.")
        csqs_per_pdf = [
            LLMRowsVo.concat_all([positive_csqs, negative_csqs])
            for positive_csqs, negative_csqs in zip(
                positive_csqs_per_pdf, negative_csqs_per_pdf, strict=True
            )
        ]
        rfms_per_pdf = [
            LLMRowsVo.concat_all(
                [self._compute_rfms_via_text(csq) for csq in tqdm(csqs.root)]
            )
            for csqs in csqs_per_pdf
        ]
        logger.info("Cleaning up rows with no word embeddings.")
        rows_per_pdf = [
            LLMRowsVo(
                [
                    row
                    for row in tqdm(rows.root)
                    if self._has_at_least_one_token_with_vector(row)
                ]
            )
            for rows in rfms_per_pdf
        ]
        self._save_as_json_dataset_for_display(rows_per_pdf, pdfs)
        return rfms_per_pdf

    def _has_enough_words(self, text: str) -> bool:
        tokens = list(self.spacy_conf.spacy(text))
        return len(tokens) > 50

    def _compute_csqs_via_text(
        self, text: str, kind: Literal["positive", "negative"]
    ) -> LLMRowsVo:
        command = f"""{text}

For the above text, in the context of data augmentation to train a dense neural network:

- Generate 25 {kind} consequences of these measures
- An answer EXACTLY FORMATTED "- consequence1\n- consequence2\n- consequence3 ... \n- consequence25"
- No headers or other artifacts in your response, you REALLY NEED TO FOLLOW THE FORMAT "- consequence1\n- consequence2\n- consequence3 ... \n- consequence25"
- In French
"""
        with self.ollama_conf.get_prediction(command) as raw_csqs:
            return LLMRowsVo.from_raw(raw_csqs)

    def _compute_rfms_via_text(self, text: str) -> LLMRowsVo:
        command = f"""{text}

For the above text, in the context of data augmentation to train a dense neural network:

- Generate 25 rephrasings WITHOUT USING THE SAME WORDS for the text
- An answer EXACTLY FORMATTED "- rephrasing1\n- rephrasing2\n- rephrasing3 ... \n- rephrasing25"
- No headers or other artifacts in your response, you REALLY NEED TO FOLLOW THE FORMAT "- rephrasing1\n- rephrasing2\n- rephrasing3 ... \n- rephrasing25"
- In French
"""
        with self.ollama_conf.get_prediction(command) as raw_rfms:
            return LLMRowsVo.from_raw(raw_rfms)

    def _has_at_least_one_token_with_vector(self, text: str) -> bool:
        return True in [token.has_vector for token in list(self.spacy_conf.spacy(text))]

    def _save_as_json_dataset_for_display(
        self, rfms_per_pdf: Sequence[LLMRowsVo], pdfs: Sequence[PdfVo]
    ) -> None:
        out_rows: List[Dict[str, str]] = []
        for rfms, pdf in zip(rfms_per_pdf, pdfs, strict=True):
            rows = [{"data": row, "label": pdf.name} for row in rfms.root]
            out_rows.extend(rows)
        filename = sha3_512(random.random().__str__().encode()).hexdigest()
        filepath = os.path.join(
            envs_conf.impl.llm_prep_svc_cache_dirpath, f"{filename}.json"
        )
        logger.info(f"Saving as JSON dataset for display at '{filepath}'.")
        self.file_system_svc.write_as_json(out_rows, filepath)


impl = LLMPrepSvc()
