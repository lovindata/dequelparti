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


@dataclass
class LLMPrepSvc:
    envs_conf = envs_conf.impl
    ollama_conf = ollama_conf.impl
    spacy_conf = spacy_conf.impl
    file_system_svc = file_system_svc.impl

    def compute_llm_rows(self, pdfs: Sequence[PdfVo]) -> List[LLMRowsVo]:
        texts_per_pdf = [
            [page.text for page in pdf.pages if self._has_enough_words(page.text)]
            for pdf in pdfs
        ]
        logger.info("Computing positive consequences for all pages of each PDF.")
        positive_csqs_per_pdf = [
            LLMRowsVo.concat_all(
                [self._compute_csqs_via_text(text, "positives") for text in tqdm(texts)]
            )
            for texts in texts_per_pdf
        ]
        logger.info("Computing negative consequences for all pages of each PDF.")
        negative_csqs_per_pdf = [
            LLMRowsVo.concat_all(
                [self._compute_csqs_via_text(text, "negatives") for text in tqdm(texts)]
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
        self, text: str, kind: Literal["positives", "negatives"]
    ) -> LLMRowsVo:
        command = f"""{text}

Pour le texte ci-dessus, dans le cadre de l'augmentation de données pour entraîner un réseau de neurones dense:
- Génère environ 25 conséquences {kind} de ces mesures
- Une réponse EXACTEMENT FORMATTER "- conséquence0\n- conséquence1\n- ..."
- Pas d'entêtes ou autres artéfacts dans ta réponse, il faut VRAIMENT RESPECTER LE FROMAT "- conséquence0\n- conséquence1\n- ..."
"""
        with self.ollama_conf.get_prediction(command) as raw_csqs:
            return LLMRowsVo.from_raw(raw_csqs)

    def _compute_rfms_via_text(self, text: str) -> LLMRowsVo:
        command = f"""{text}

Pour le texte ci-dessus, dans le cadre de l'augmentation de données pour entraîner un réseau de neurones dense:
- Génère environ 25 reformulations SANS USAGE DES MËMES MOTS pour ce texte
- Une réponse EXACTEMENT FORMATTER "- reformulation0\n- reformulation1\n ... - reformulation25\n"
- Pas d'entêtes ou autres artéfacts dans ta réponse, il faut VRAIMENT RESPECTER LE FROMAT "- reformulation0\n- reformulation1\n ... - reformulation25\n"
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
