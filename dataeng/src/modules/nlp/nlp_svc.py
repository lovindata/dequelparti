from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import spacy
from loguru import logger
from spacy.tokens import Token
from spellchecker import SpellChecker
from tqdm import tqdm

from src.confs import envs_conf
from src.modules.file_system.pdf_vo.pdf_page_vo import PdfPageVo
from src.modules.file_system.pdf_vo.pdf_vo import PdfVo
from src.modules.nlp.lemma_embeddings_vo import LemmaEmbeddingsVo
from src.modules.nlp.slinding_avg_lemmas_vo import SlindingAvgLemmasVo
from src.modules.nlp.vocabulary_vo import VocabularyVo


@dataclass
class NLPSvc:
    envs_conf = envs_conf.impl
    _nlp = spacy.load("fr_core_news_lg")
    _spell_checker = SpellChecker(language="fr")
    _manual_stop_words = {"l", "etc."}

    def compute_nlp_value_objects(
        self, pdfs: Sequence[PdfVo]
    ) -> Tuple[List[SlindingAvgLemmasVo], VocabularyVo, LemmaEmbeddingsVo]:
        logger.info(f"Extracting tokens from {len(pdfs)} PDFs.")
        tokens_per_pdf = [
            [
                token
                for pdf_page in pdf.pages
                for token in self._extract_tokens(pdf_page)
            ]
            for pdf in pdfs
        ]
        vocabulary = self._build_vocabulary(tokens_per_pdf)
        lemma_embeddings = self._build_lemma_embeddings(vocabulary)
        all_sliding_avg_lemmas = self._build_all_sliding_avg_lemmas(
            tokens_per_pdf, vocabulary=vocabulary
        )
        self._log_stats(
            all_sliding_avg_lemmas,
            vocabulary=vocabulary,
            lemma_embeddings=lemma_embeddings,
        )
        return all_sliding_avg_lemmas, vocabulary, lemma_embeddings

    def _extract_tokens(self, page: PdfPageVo) -> List[Token]:
        doc = self._nlp(page.text)
        tokens = [
            token
            for token in doc
            if token.is_stop is False
            and token.is_punct is False
            and token.is_space is False
        ]
        return tokens

    def _build_vocabulary(
        self, tokens_per_pdf: Sequence[Sequence[Token]]
    ) -> VocabularyVo:
        logger.info(f"Building vocabulary.")
        vocabulary: Dict[str, str] = {}
        for tokens in tokens_per_pdf:
            for token in tokens:
                if token.text not in vocabulary:
                    vocabulary[token.text] = token.lemma_
        known_words = self._spell_checker.known(vocabulary.keys())
        has_vector = lambda word: self._nlp(word).has_vector
        vocabulary = {
            word: vocabulary[word]
            for word in tqdm(vocabulary)
            if word in known_words
            and word not in self._manual_stop_words
            and has_vector(vocabulary[word])  # Lemma has a vector
        }
        vocabulary = dict(sorted(vocabulary.items()))
        return VocabularyVo(vocabulary)

    def _build_all_sliding_avg_lemmas(
        self, batch_of_tokens: Sequence[Sequence[Token]], vocabulary: VocabularyVo
    ) -> List[SlindingAvgLemmasVo]:
        logger.info(f"Lemmatizing tokens.")
        several_lemmas = [
            [vocabulary[token.text] for token in tokens if token.text in vocabulary]
            for tokens in batch_of_tokens
        ]
        return several_lemmas

    def _build_lemma_embeddings(self, vocabulary: VocabularyVo) -> LemmaEmbeddingsVo:
        logger.info(f"Building word embeddings.")
        lemmas = list(sorted(set(vocabulary.root.values())))
        word_embedding_of_lemmas: Dict[str, List[float]] = {}
        for lemma in tqdm(lemmas):
            word_embedding_of_lemmas[lemma] = list(
                [float(x) for x in self._nlp(lemma)[0].vector]
            )
        return LemmaEmbeddingsVo(word_embedding_of_lemmas)

    def _log_stats(
        self,
        all_sliding_avg_lemmas: Sequence[SlindingAvgLemmasVo],
        vocabulary: VocabularyVo,
        lemma_embeddings: LemmaEmbeddingsVo,
    ) -> None:
        logger.info(
            f"Vocabulary of size {vocabulary.size()} with {vocabulary.nb_unique_lemmas()} unique lemmas."
        )
        logger.info(
            f"Word embedding of lemmas have registered {lemma_embeddings.size()} entries with vectors of size {lemma_embeddings.vector_size()}."
        )
        for i, lemmas in enumerate(all_sliding_avg_lemmas):
            logger.info(
                f"{len(lemmas)} lemmas for the batch '{i}'."
            )  # TODO James continue here!


impl = NLPSvc()
