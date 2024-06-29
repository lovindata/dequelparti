import os.path
from dataclasses import dataclass
from typing import List

from src.confs import envs_conf
from src.modules.shared.services import nlp_svc, pdf_svc


@dataclass
class EnsembleSvc:
    envs_conf = envs_conf.impl
    pdf_svc = pdf_svc.impl
    nlp_svc = nlp_svc.impl

    def extract_and_save_as_json(self) -> List[str]:
        input_pdf_filepath = os.path.abspath(self.envs_conf.input_prgm_ensemble)
        output_dir_path = os.path.join(
            self.envs_conf.output_data_path, os.path.basename(input_pdf_filepath)
        )
        tokens = self.pdf_svc.extract_tokens_and_save_as_json(
            input_pdf_filepath, output_dir_path
        )
        return tokens


impl = EnsembleSvc()
