from dataclasses import dataclass
from importlib.resources import files
from typing import List

from PyPDF2 import PdfReader

from src.confs.envs_conf import envs_conf_impl
from src.modules.rassemblement_national import resources


@dataclass
class RassemblementNationalSvc:
    envs_conf = envs_conf_impl

    def extract_and_save_as_json(self) -> List[str]:
        pdf_path = str(files(resources) / f"rassemblement_national_programme.pdf")
        pages = PdfReader(pdf_path).pages
        print(pages[5])
        # for page in pages:
        #     text = page.extract_text()
        #     points = text.split("\n• ")
        #     print(points)

        raise NotImplementedError()


rassemblement_national_svc_impl = RassemblementNationalSvc()
