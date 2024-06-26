from dataclasses import dataclass
from importlib.resources import files
from typing import List

from PyPDF2 import PdfReader

from src.confs.envs_conf import envs_conf_impl
from src.modules.nouveau_front_populaire import resources


@dataclass
class NouveauFrontPopulaireSvc:
    envs_conf = envs_conf_impl

    def extract_and_save_as_json(self) -> List[str]:
        pdf_path = str(files(resources) / f"nouveau_front_populaire_programme.pdf")
        pages = PdfReader(pdf_path).pages
        for page in pages:
            text = page.extract_text()
            points = text.split("\n• ")
            print(points)

        raise NotImplementedError()


nouveau_front_populaire_svc_impl = NouveauFrontPopulaireSvc()
