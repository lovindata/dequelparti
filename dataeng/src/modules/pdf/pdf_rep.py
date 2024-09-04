from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List

from loguru import logger
from PyPDF2 import PdfReader
from tqdm import tqdm

from src.confs import envs_conf
from src.modules.shared.atomic_vos.molecules.pdf_vo import PdfVo


@dataclass(frozen=True)
class PdfRep:
    envs_conf: envs_conf.EnvsConf = envs_conf.impl

    def read_pdfs(self) -> List[PdfVo]:
        pdfnames = [
            filename
            for filename in os.listdir(self.envs_conf.prgms_dirpath)
            if filename.endswith(".pdf")
        ]
        pdfpaths = [
            os.path.join(self.envs_conf.prgms_dirpath, pdfname) for pdfname in pdfnames
        ]
        logger.info(f"PDF files found: {pdfnames}.")
        logger.info(f"Loading PDF files.")
        pdfs = [self._read_pdf(filepath) for filepath in tqdm(pdfpaths)]
        return pdfs

    def _read_pdf(self, filepath: str) -> PdfVo:
        pages = PdfReader(filepath).pages
        pages = [page.extract_text() for page in pages]
        pages = [page for page in pages if page.strip() != ""]
        name = os.path.basename(filepath).removesuffix(".pdf")
        return PdfVo(name, pages)


impl = PdfRep()
