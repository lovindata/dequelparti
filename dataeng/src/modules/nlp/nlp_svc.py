from dataclasses import dataclass
from typing import Dict, List, Tuple

import spacy
from loguru import logger
from spacy.lang.fr.stop_words import STOP_WORDS
from spacy.tokens import Token
from spellchecker import SpellChecker


@dataclass
class NLPSvc:
    _nlp = spacy.load("fr_dep_news_trf")
    _spell_checker = SpellChecker(language="fr")

    def compute_lemmas(
        self, batch_of_texts: List[List[str]]
    ) -> Tuple[List[List[str]], Dict[str, str]]:
        logger.info(f"Extracting tokens from {len(batch_of_texts)} batch of texts.")
        batch_of_tokens = [
            [token for text in texts for token in self._extract_tokens(text)]
            for texts in batch_of_texts
        ]
        vocabulary = self._build_vocabulary(batch_of_tokens)
        batch_of_lemmas = self._lemmatize_tokens(batch_of_tokens, vocabulary=vocabulary)
        self._log_stats(batch_of_lemmas, vocabulary=vocabulary)
        return batch_of_lemmas, vocabulary

    def _extract_tokens(self, text: str) -> List[Token]:
        doc = self._nlp(text)
        tokens = [
            token
            for token in doc
            if not token.is_stop and not token.is_punct and not token.is_space
        ]
        return tokens

    def _build_vocabulary(self, batch_of_tokens: List[List[Token]]) -> Dict[str, str]:
        vocabulary: Dict[str, str] = {}
        for tokens in batch_of_tokens:
            for token in tokens:
                vocabulary[token.text] = token.lemma_
        vocabulary = {
            word: vocabulary[word]
            for word in vocabulary
            # if self._spell_checker.word_usage_frequency(word) > 0
            # and
            if self._spell_checker.correction(word) is None
            if token.text not in STOP_WORDS
        }
        vocabulary = dict(sorted(vocabulary.items()))
        return vocabulary

    def _lemmatize_tokens(
        self, batch_of_tokens: List[List[Token]], vocabulary: Dict[str, str]
    ) -> List[List[str]]:
        several_lemmas = [
            [token.text for token in tokens if token.text in vocabulary]
            for tokens in batch_of_tokens
        ]
        return several_lemmas

    def _log_stats(
        self, batch_of_lemmas: List[List[str]], vocabulary: Dict[str, str]
    ) -> None:
        logger.info(f"Vocabulary of size {len(vocabulary)}.")
        for i, lemmas in enumerate(batch_of_lemmas):
            logger.info(f"{len(lemmas)} lemmas for the batch '{i}'.")


impl = NLPSvc()
