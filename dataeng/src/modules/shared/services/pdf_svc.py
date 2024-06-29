import json
import os
from dataclasses import dataclass
from typing import List

from loguru import logger
from PyPDF2 import PageObject, PdfReader

from src.modules.shared.services import nlp_svc


@dataclass
class PDFSvc:
    nlp_svc = nlp_svc.impl

    def extract_tokens_and_save_as_json(
        self, input_pdf_filepath: str, output_tokens_dirpath: str
    ) -> List[str]:
        pages = self._load_pages(input_pdf_filepath)
        tokens = self.nlp_svc.compute_tokens(pages)
        tokens_as_json = json.dumps(tokens, indent=2)
        basename = os.path.basename(input_pdf_filepath)
        output_path = os.path.join(output_tokens_dirpath, basename)
        with open(output_path, "w+") as f:
            f.write(tokens_as_json)
            logger.info(f"'{basename}' tokens extraction success.")
        return tokens

    def _load_pages(self, path: str) -> List[str]:
        pages = PdfReader(path).pages
        pages = [self._extract_text(page) for page in pages]
        logger.info(f"Loaded PDF '{path}'.")
        return pages

    def _extract_text(self, page: PageObject) -> str:
        text = page.extract_text()
        fixes = {"-\n": "", "\n": " "}
        for fix in fixes:
            text = text.replace(fix, fixes[fix])
        return text


impl = PDFSvc()
