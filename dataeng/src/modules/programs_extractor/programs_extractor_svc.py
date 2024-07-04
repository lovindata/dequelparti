import os
from dataclasses import dataclass
from typing import List

from src.confs import envs_conf
from src.modules.file_system import file_system_svc
from src.modules.nlp import nlp_svc


@dataclass
class ProgramsExtractorSvc:
    envs_conf = envs_conf.impl
    nlp_svc = nlp_svc.impl
    file_system_svc = file_system_svc.impl

    def run(self) -> None:
        def build_input_filepaths() -> List[str]:
            filenames = os.listdir(self.envs_conf.input_dirpath)
            input_filepaths = [
                os.path.join(self.envs_conf.input_dirpath, filename)
                for filename in filenames
            ]
            return input_filepaths

        def build_output_filepaths(
            input_filepaths: List[str], output_dirpath: str
        ) -> List[str]:
            output_filepaths = []
            for input_filepath in input_filepaths:
                output_filename = (
                    os.path.basename(input_filepath).split(".")[0] + ".json"
                )
                output_filepath = os.path.join(output_dirpath, output_filename)
                output_filepaths.append(output_filepath)
            return output_filepaths

        def save_batch_of_lemmas(
            batch_of_lemmas: List[List[str]], output_filepaths: List[str]
        ) -> None:
            for lemmas, output_filepath in zip(batch_of_lemmas, output_filepaths):
                self.file_system_svc.write_as_json(lemmas, output_filepath)

        input_filepaths = build_input_filepaths()
        output_filepaths = build_output_filepaths(
            input_filepaths, self.envs_conf.output_dirpath
        )
        output_vocabulary_path = os.path.join(
            self.envs_conf.output_dirpath, "vocabulary.json"
        )
        batch_of_pages = [
            self.file_system_svc.read_pdf_pages(input_filepath)
            for input_filepath in input_filepaths
        ]
        batch_of_lemmas, vocabulary = self.nlp_svc.compute_lemmas(batch_of_pages)
        self.file_system_svc.write_as_json(vocabulary, output_vocabulary_path)
        save_batch_of_lemmas(batch_of_lemmas, output_filepaths)


impl = ProgramsExtractorSvc()
