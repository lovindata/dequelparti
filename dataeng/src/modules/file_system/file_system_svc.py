import json
import os
from dataclasses import dataclass
from typing import Any, List

from loguru import logger
from PyPDF2 import PageObject, PdfReader


@dataclass
class FileSystemSvc:
    def read_pdf_pages(self, path: str) -> List[str]:
        def extract_text(page: PageObject) -> str:
            text = page.extract_text()
            fixes = {"-\n": "", "\n": " "}
            for fix in fixes:
                text = text.replace(fix, fixes[fix])
            return text

        logger.info(f"Loading PDF '{path}' pages.")
        pages = PdfReader(path).pages
        pages = [extract_text(page) for page in pages]
        return pages

    def write_as_json(self, object: Any, output_json_filepath: str) -> None:
        logger.info(f"Saving json at '{output_json_filepath}'.")
        object_as_json = json.dumps(object, indent=2, ensure_ascii=False)
        dirpath = os.path.dirname(output_json_filepath)
        os.makedirs(dirpath, exist_ok=True)
        with open(output_json_filepath, "w+") as f:
            f.write(object_as_json)


impl = FileSystemSvc()
