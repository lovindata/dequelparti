from dataclasses import dataclass

import spacy
from spellchecker import SpellChecker


@dataclass
class SpacyConf:
    spacy = spacy.load("fr_core_news_lg")
    spell_checker = SpellChecker(language="fr")


impl = SpacyConf()
