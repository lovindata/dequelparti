from dataclasses import dataclass
from typing import List

import spacy
from loguru import logger
from spacy.lang.fr.stop_words import STOP_WORDS
from spellchecker import SpellChecker


@dataclass
class NLPSvc:
    _NLP = spacy.load("fr_dep_news_trf")

    def compute_tokens(self, text: str) -> List[str]:
        tokens = self._extract_tokens(text)
        output_tokens = self._remove_incorrect_spellchecked_tokens(tokens)
        output_tokens = self._remove_stop_words_tokens(output_tokens)
        self._log_stats(output_tokens)
        return output_tokens

    def _extract_tokens(self, text: str) -> List[str]:
        doc = self._NLP(text)
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct and not token.is_space
        ]
        return tokens

    def _remove_incorrect_spellchecked_tokens(self, tokens: List[str]) -> List[str]:
        fr = SpellChecker(language="fr")
        unique_tokens = set(tokens)
        incorrect_tokens = {
            token for token in unique_tokens if fr.word_usage_frequency(token) == 0
        }
        return [token for token in tokens if token not in incorrect_tokens]

    def _remove_stop_words_tokens(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if token not in STOP_WORDS]

    def _log_stats(self, tokens: List[str]) -> None:
        logger.info(f"{len(tokens)} tokens with {len(set(tokens))} unique tokens.")


impl = NLPSvc()
