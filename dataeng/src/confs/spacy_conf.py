from dataclasses import dataclass
from typing import List

import spacy
from spacy.tokens import Token
from spellchecker import SpellChecker


@dataclass
class SpacyConf:
    _spacy = spacy.load("fr_core_news_lg")
    _spell_checker = SpellChecker(language="fr")

    def predict(self, text: str) -> List[Token]:
        tokens = list(self._spacy(text))
        return tokens


impl = SpacyConf()
