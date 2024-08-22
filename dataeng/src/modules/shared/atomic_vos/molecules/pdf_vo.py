from dataclasses import dataclass
from typing import Sequence

from src.modules.shared.atomic_vos.atoms.pdf_page_vo import PdfPageVo


@dataclass
class PdfVo:
    name: str
    pages: Sequence[PdfPageVo]
