from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import List

import spacy
from loguru import logger
from spacy.lang.fr.stop_words import STOP_WORDS


@dataclass
class NLPSvc:
    _NLP = spacy.load("fr_dep_news_trf")

    def compute_tokens(self, texts: List[str]) -> List[str]:
        with Pool(processes=cpu_count()) as pool:
            tokens_by_text = pool.map(self._extract_tokens, texts)
        output_tokens = [token for tokens in tokens_by_text for token in tokens]
        self._display_stats(output_tokens)
        return output_tokens

    def _extract_tokens(self, text: str) -> List[str]:
        doc = self._NLP(text)
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct and not token.is_space
        ]
        return tokens

    def _remove_stop_words_tokens(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if token not in STOP_WORDS]

    def _display_stats(self, tokens: List[str]) -> None:
        logger.info(f"{len(tokens)} tokens with {len(set(tokens))} unique tokens.")


impl = NLPSvc()
