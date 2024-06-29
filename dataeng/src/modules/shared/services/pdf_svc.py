import os
from dataclasses import dataclass
from typing import List

from loguru import logger
from PyPDF2 import PageObject, PdfReader

from src.modules.shared.services import nlp_svc, write_json_svc


@dataclass
class PDFSvc:
    nlp_svc = nlp_svc.impl
    write_json_svc = write_json_svc.impl

    def extract_tokens_and_save_as_json(
        self, input_pdf_filepath: str, output_tokens_dirpath: str
    ) -> List[str]:
        logger.info(f"Loading PDF '{input_pdf_filepath}'.")
        pages = self._load_pages(input_pdf_filepath)

        logger.info(f"Extracting tokens from PDF '{input_pdf_filepath}'.")
        tokens = self.nlp_svc.compute_tokens(pages)

        output_filename = os.path.basename(input_pdf_filepath).split(".")[0] + ".json"
        output_filepath = os.path.join(output_tokens_dirpath, output_filename)
        logger.info(f"Saving tokens at '{output_filepath}'.")
        self.write_json_svc.write_as_json(tokens, output_filepath)
        return tokens

    def _load_pages(self, path: str) -> List[str]:
        pages = PdfReader(path).pages
        pages = [self._extract_text(page) for page in pages]
        return pages

    def _extract_text(self, page: PageObject) -> str:
        text = page.extract_text()
        fixes = {"-\n": "", "\n": " "}
        for fix in fixes:
            text = text.replace(fix, fixes[fix])
        return text


impl = PDFSvc()
