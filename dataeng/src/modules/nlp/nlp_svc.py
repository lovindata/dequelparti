from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import spacy
from loguru import logger
from spacy.tokens import Token
from spellchecker import SpellChecker
from tqdm import tqdm


@dataclass
class NLPSvc:
    _nlp = spacy.load("fr_core_news_lg")
    _spell_checker = SpellChecker(language="fr")
    _manual_stop_words = {"l", "etc."}

    def compute_nlp_entities(
        self, batch_of_texts: Sequence[Sequence[str]]
    ) -> Tuple[List[List[str]], Dict[str, str], Dict[str, List[float]]]:
        logger.info(f"Extracting tokens from {len(batch_of_texts)} batch of texts.")
        batch_of_tokens = [
            [token for text in texts for token in self._extract_tokens(text)]
            for texts in batch_of_texts
        ]
        vocabulary = self._build_vocabulary(batch_of_tokens)
        word_embedding_of_lemmas = self._build_word_embeddings(vocabulary)
        batch_of_lemmas = self._lemmatize_tokens(batch_of_tokens, vocabulary=vocabulary)
        self._log_stats(
            batch_of_lemmas,
            vocabulary=vocabulary,
            word_embedding_of_lemmas=word_embedding_of_lemmas,
        )
        return batch_of_lemmas, vocabulary, word_embedding_of_lemmas

    def _extract_tokens(self, text: str) -> List[Token]:
        doc = self._nlp(text)
        tokens = [
            token
            for token in doc
            if token.is_stop is False
            and token.is_punct is False
            and token.is_space is False
        ]
        return tokens

    def _build_vocabulary(
        self, batch_of_tokens: Sequence[Sequence[Token]]
    ) -> Dict[str, str]:
        logger.info(f"Building vocabulary.")
        vocabulary: Dict[str, str] = {}
        for tokens in batch_of_tokens:
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
        return vocabulary

    def _lemmatize_tokens(
        self, batch_of_tokens: Sequence[Sequence[Token]], vocabulary: Mapping[str, str]
    ) -> List[List[str]]:
        logger.info(f"Lemmatizing tokens.")
        several_lemmas = [
            [token.lemma_ for token in tokens if token.text in vocabulary]
            for tokens in batch_of_tokens
        ]
        return several_lemmas

    def _build_word_embeddings(
        self, vocabulary: Mapping[str, str]
    ) -> Dict[str, List[float]]:
        logger.info(f"Building word embeddings.")
        lemmas = list(sorted(set(vocabulary.values())))
        word_embedding_of_lemmas: Dict[str, List[float]] = {}
        for lemma in tqdm(lemmas):
            word_embedding_of_lemmas[lemma] = list(
                [float(x) for x in self._nlp(lemma)[0].vector]
            )
        return word_embedding_of_lemmas

    def _log_stats(
        self,
        batch_of_lemmas: Sequence[Sequence[str]],
        vocabulary: Mapping[str, str],
        word_embedding_of_lemmas: Mapping[str, List[float]],
    ) -> None:
        logger.info(
            f"Vocabulary of size {len(vocabulary)} with {len(set(vocabulary.values()))} unique lemmas."
        )
        logger.info(
            f"Word embedding of lemmas have registered {len(word_embedding_of_lemmas)} entries with vectors of size {len(list(word_embedding_of_lemmas.values())[0])}."
        )
        for i, lemmas in enumerate(batch_of_lemmas):
            logger.info(f"{len(lemmas)} lemmas for the batch '{i}'.")


impl = NLPSvc()
