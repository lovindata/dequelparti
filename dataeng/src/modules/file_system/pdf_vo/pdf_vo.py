from dataclasses import dataclass
from typing import Sequence

from src.modules.file_system.pdf_vo.pdf_page_vo import PdfPageVo


@dataclass
class PdfVo:
    name: str
    pages: Sequence[PdfPageVo]
