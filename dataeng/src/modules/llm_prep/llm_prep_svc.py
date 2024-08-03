from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Sequence

from loguru import logger
from tqdm import tqdm

from src.confs import envs_conf, ollama_conf, spacy_conf
from src.modules.file_system import file_system_svc
from src.modules.file_system.pdf_vo.pdf_vo import PdfVo
from src.modules.llm_prep import cache_svc
from src.modules.llm_prep.llm_row_vo import LLMRowVo
from src.modules.vocabulary_prep.vocabulary_vo import VocabularyVo


@dataclass(frozen=True)
class LLMPrepSvc:
    envs_conf: envs_conf.EnvsConf = envs_conf.impl
    ollama_conf: ollama_conf.OllamaConf = ollama_conf.impl
    spacy_conf: spacy_conf.SpacyConf = spacy_conf.impl
    file_system_svc: file_system_svc.FileSystemSvc = file_system_svc.impl
    cache_svc: cache_svc.CacheSvc = cache_svc.impl

    def compute_llm_rows(
        self, pdfs: Sequence[PdfVo], vocabulary: VocabularyVo
    ) -> List[LLMRowVo]:
        if llm_rows_per_pdf := self.cache_svc.load_cache_from_disk(pdfs):
            logger.info("Loading LLM rows from cache.")
            return llm_rows_per_pdf
        texts_per_pdf = [
            [page.text for page in pdf.pages if self._has_enough_words(page.text)]
            for pdf in pdfs
        ]
        logger.info("Computing positive consequences for all pages of each PDF.")
        positive_csq_llm_rows = [
            llm_row
            for texts, pdf in zip(texts_per_pdf, pdfs, strict=True)
            for text in tqdm(texts)
            for llm_row in self._compute_csqs_via_text(text, "positive", pdf.name)
        ]
        logger.info("Computing negative consequences for all pages of each PDF.")
        negative_csq_llm_rows = [
            llm_row
            for texts, pdf in zip(texts_per_pdf, pdfs, strict=True)
            for text in tqdm(texts)
            for llm_row in self._compute_csqs_via_text(text, "negative", pdf.name)
        ]
        logger.info("Computing reformulations for all computed consequences.")
        csq_llm_rows = positive_csq_llm_rows
        csq_llm_rows.extend(negative_csq_llm_rows)
        rfm_llm_rows = [
            rfm_llm_row
            for csq_llm_row in tqdm(csq_llm_rows)
            for rfm_llm_row in self._compute_rfms_via_text(
                csq_llm_row.feature, csq_llm_row.label
            )
        ]
        logger.info("Cleaning up rows with no words in the vocabulary.")
        out_llm_rows = [
            rfm_llm_row
            for rfm_llm_row in tqdm(rfm_llm_rows)
            if self._has_at_least_one_token_with_vector(rfm_llm_row.feature, vocabulary)
        ]
        self.cache_svc.cache_to_disk(pdfs, out_llm_rows)
        return out_llm_rows

    def _has_enough_words(self, text: str) -> bool:
        tokens = list(self.spacy_conf.spacy(text))
        return len(tokens) > 50

    def _compute_csqs_via_text(
        self, text: str, kind: Literal["positive", "negative"], label: str
    ) -> List[LLMRowVo]:
        command = f"""{text}

For the above text, in the context of data augmentation to train a dense neural network:

- Generate 25 {kind} consequences of these measures
- An answer EXACTLY FORMATTED "- consequence1\n- consequence2\n- consequence3 ... \n- consequence25"
- No headers or other artifacts in your response, you REALLY NEED TO FOLLOW THE FORMAT "- consequence1\n- consequence2\n- consequence3 ... \n- consequence25"
- In French
"""
        with self.ollama_conf.get_prediction(command) as raw_csqs:
            return self._convert_raw_to_llm_rows(raw_csqs, label)

    def _compute_rfms_via_text(self, text: str, label: str) -> List[LLMRowVo]:
        command = f"""{text}

For the above text, in the context of data augmentation to train a dense neural network:

- Generate 25 rephrasings WITHOUT USING THE SAME WORDS for the text
- An answer EXACTLY FORMATTED "- rephrasing1\n- rephrasing2\n- rephrasing3 ... \n- rephrasing25"
- No headers or other artifacts in your response, you REALLY NEED TO FOLLOW THE FORMAT "- rephrasing1\n- rephrasing2\n- rephrasing3 ... \n- rephrasing25"
- In French
"""
        with self.ollama_conf.get_prediction(command) as raw_rfms:
            return self._convert_raw_to_llm_rows(raw_rfms, label)

    def _convert_raw_to_llm_rows(self, raw: str, label: str) -> List[LLMRowVo]:
        out_rows = raw.split("- ")
        out_rows = [csq.strip() for csq in out_rows]
        out_rows = [
            LLMRowVo(feature=out_row, label=label)
            for out_row in out_rows
            if out_row != ""
        ]
        return out_rows

    def _has_at_least_one_token_with_vector(
        self, text: str, vocabulary: VocabularyVo
    ) -> bool:
        return [
            token
            for token in list(self.spacy_conf.spacy(text))
            if token.text in vocabulary.root
        ] != []


impl = LLMPrepSvc()
