from importlib.resources import files
from typing import Dict, List, Literal, Set

import spacy
from PyPDF2 import PdfReader
from spacy.lang.fr.stop_words import STOP_WORDS
from spellchecker import SpellChecker

import src.modules.ensemble.resources
import src.modules.nouveau_front_populaire.resources
import src.modules.rassemblement_national.resources


def load_pages(
    prgm: Literal[
        "prgm_rassemblement_national", "prgm_ensemble", "prgm_nouveau_front_populaire"
    ]
) -> List[str]:
    match prgm:
        case "prgm_rassemblement_national":
            pdf_path = str(
                files(src.modules.rassemblement_national.resources) / f"{prgm}.pdf"
            )
        case "prgm_ensemble":
            pdf_path = str(files(src.modules.ensemble.resources) / f"{prgm}.pdf")
        case "prgm_nouveau_front_populaire":
            pdf_path = str(
                files(src.modules.nouveau_front_populaire.resources) / f"{prgm}.pdf"
            )
    pages = PdfReader(pdf_path).pages
    pages = [page.extract_text() for page in pages]
    return pages


NLP = spacy.load("fr_dep_news_trf")


def fix_page_from_pdf_text_extract(page: str) -> str:
    fixes = {"-\n": "", "\n": " "}
    for fix in fixes:
        page = page.replace(fix, fixes[fix])
    return page


def extract_tokens(page: str) -> List[str]:
    doc = NLP(page)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]
    return tokens


def remove_incorrect_spellchecked_tokens(tokens: List[str]) -> List[str]:
    fr = SpellChecker(language="fr")
    unique_tokens = set(tokens)
    incorrect_tokens = {
        token for token in unique_tokens if fr.word_usage_frequency(token) == 0
    }
    return [token for token in tokens if token not in incorrect_tokens]


def remove_or_correct_incorrect_spellchecked_tokens(tokens: List[str]) -> List[str]:
    fr = SpellChecker(language="fr")
    unique_tokens = set(tokens)
    fixes: Dict[str, str | None] = {}
    for token in unique_tokens:
        fixed_token = fr.correction(token)
        if token != fixed_token:
            fixes[token] = fixed_token

    output_tokens: List[str] = []
    for token in tokens:
        if token in fixes:
            fixed_token = fixes[token]
            if fixed_token:
                output_tokens.append(fixed_token)
        else:
            output_tokens.append(token)
    return output_tokens


def remove_stop_words_tokens(tokens: List[str]) -> List[str]:
    return [token for token in tokens if token not in STOP_WORDS]


def display_stats(tokens: List[str]) -> None:
    print(tokens)
    print(len(tokens), "tokens")
    print(len(set(tokens)), "unique tokens")


pages = load_pages("prgm_nouveau_front_populaire")
tokens = []
for page in pages:
    page = fix_page_from_pdf_text_extract(page)
    tokens.extend(extract_tokens(page))
# tokens = remove_or_correct_incorrect_spellchecked_tokens(tokens)
tokens = remove_incorrect_spellchecked_tokens(tokens)
tokens = remove_stop_words_tokens(tokens)
display_stats(tokens)
