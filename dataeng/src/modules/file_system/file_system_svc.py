import json
import os
from dataclasses import dataclass
from typing import Any, List

from loguru import logger
from PyPDF2 import PdfReader

from src.modules.file_system.pdf_vo.pdf_page_vo import PdfPageVo
from src.modules.file_system.pdf_vo.pdf_vo import PdfVo


@dataclass
class FileSystemSvc:
    def read_pdfs(self, dirpath: str) -> List[PdfVo]:
        filenames = os.listdir(dirpath)
        filepaths = [os.path.join(dirpath, filename) for filename in filenames]
        logger.info(f"Loading PDF(s) '{", ".join(filepaths)}'.")
        pdfs = [self._read_pdf(filepath) for filepath in filepaths]
        return pdfs

    def _read_pdf(self, filepath: str) -> PdfVo:
        pages = PdfReader(filepath).pages
        pages = [page.extract_text() for page in pages]
        pages = [PdfPageVo(page) for page in pages]
        filename = os.path.basename(filepath)
        return PdfVo(filename, pages)

    def write_as_json(self, data: Any, output_json_filepath: str) -> None:
        logger.info(f"Saving json at '{output_json_filepath}'.")
        data_as_json = json.dumps(data, ensure_ascii=False)
        dirpath = os.path.dirname(output_json_filepath)
        os.makedirs(dirpath, exist_ok=True)
        with open(output_json_filepath, "w+") as f:
            f.write(data_as_json)


impl = FileSystemSvc()
