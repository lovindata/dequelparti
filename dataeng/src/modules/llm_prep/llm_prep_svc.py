from dataclasses import dataclass
from typing import List, Literal, Sequence

from loguru import logger
from tqdm import tqdm

from src.confs import envs_conf, ollama_conf, spacy_conf
from src.modules.file_system import file_system_svc
from src.modules.file_system.pdf_vo.pdf_vo import PdfVo
from src.modules.llm_prep.llm_prep_rows_vo import LLMPrepRowsVo


@dataclass
class LLMPrepSvc:
    envs_conf = envs_conf.impl
    ollama_conf = ollama_conf.impl
    spacy_conf = spacy_conf.impl
    file_system_svc = file_system_svc.impl

    def compute_llm_prep_rows(self, pdfs: Sequence[PdfVo]) -> List[LLMPrepRowsVo]:
        texts_per_pdf = [
            [page.text for page in pdf.pages if self._has_enough_words(page.text)]
            for pdf in pdfs
        ]
        logger.info("Computing positive consequences for each page of each PDF.")
        positive_csqs_per_pdf = [
            LLMPrepRowsVo.concat_all(
                [self._compute_csqs_via_text(text, "positives") for text in tqdm(texts)]
            )
            for texts in texts_per_pdf
        ]
        logger.info("Computing negative consequences for each page of each PDF.")
        negative_csqs_per_pdf = [
            LLMPrepRowsVo.concat_all(
                [self._compute_csqs_via_text(text, "negatives") for text in tqdm(texts)]
            )
            for texts in texts_per_pdf
        ]
        logger.info(
            "Computing reformulations for each consequence from each page of each PDF."
        )
        csqs_per_pdf = positive_csqs_per_pdf
        csqs_per_pdf.extend(negative_csqs_per_pdf)
        rfms_per_pdf = [
            LLMPrepRowsVo.concat_all(
                [self._compute_rfms_via_text(csq) for csq in tqdm(csqs.rows)]
            )
            for csqs in csqs_per_pdf
        ]
        return rfms_per_pdf

    def _has_enough_words(self, text: str) -> bool:
        tokens = self.spacy_conf.predict(text)
        return len(tokens) > 50

    def _compute_csqs_via_text(
        self, text: str, kind: Literal["positives", "negatives"]
    ) -> LLMPrepRowsVo:
        command = f"""{text}

Pour le texte ci-dessus, dans le cadre de l'augmentation de données pour entraîner un réseau de neurones dense:
- Génère environ 25 conséquences {kind} de ces mesures
- Une réponse EXACTEMENT FORMATTER "- conséquence0\n- conséquence1\n- ..."
- Pas d'entêtes ou autres artéfacts dans ta réponse, il faut VRAIMENT RESPECTER LE FROMAT "- conséquence0\n- conséquence1\n- ..."
"""
        with self.ollama_conf.get_prediction(command) as raw_csqs:
            return LLMPrepRowsVo.from_raw(raw_csqs)

    def _compute_rfms_via_text(self, text: str) -> LLMPrepRowsVo:
        command = f"""{text}

Pour le texte ci-dessus, dans le cadre de l'augmentation de données pour entraîner un réseau de neurones dense:
- Génère environ 25 reformulations SANS USAGE DES MËMES MOTS pour ce texte
- Une réponse EXACTEMENT FORMATTER "- reformulation0\n- reformulation1\n ... - reformulation25\n"
- Pas d'entêtes ou autres artéfacts dans ta réponse, il faut VRAIMENT RESPECTER LE FROMAT "- reformulation0\n- reformulation1\n ... - reformulation25\n"
"""
        with self.ollama_conf.get_prediction(command) as raw_rfms:
            return LLMPrepRowsVo.from_raw(raw_rfms)


impl = LLMPrepSvc()
