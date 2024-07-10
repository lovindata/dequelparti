import os
from dataclasses import dataclass
from typing import List

from src.confs import envs_conf
from src.modules.file_system import file_system_svc
from src.modules.nlp import nlp_svc
from src.modules.nlp.slinding_avg_lemmas_vo import SlindingAvgLemmasVo


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
                input_filename = os.path.basename(input_filepath).split(".")[0]
                output_filename = f"{input_filename}_sliding_avg_lemmas.json"
                output_filepath = os.path.join(output_dirpath, output_filename)
                output_filepaths.append(output_filepath)
            return output_filepaths

        def save_all_sliding_avg_lemmas(
            all_sliding_avg_lemmas: List[SlindingAvgLemmasVo],
            output_filepaths: List[str],
        ) -> None:
            for lemmas, output_filepath in zip(
                all_sliding_avg_lemmas, output_filepaths
            ):
                self.file_system_svc.write_as_json(lemmas, output_filepath)

        input_filepaths = build_input_filepaths()
        output_filepaths = build_output_filepaths(
            input_filepaths, self.envs_conf.output_dirpath
        )
        output_vocabulary_path = os.path.join(
            self.envs_conf.output_dirpath, "vocabulary.json"
        )
        output_sim_matrix_path = os.path.join(
            self.envs_conf.output_dirpath, "lemma_embeddings.json"
        )
        pdfs = [
            self.file_system_svc.read_pdf(input_filepath)
            for input_filepath in input_filepaths
        ]
        all_sliding_avg_lemmas, vocabulary, lemma_embeddings = (
            self.nlp_svc.compute_nlp_value_objects(pdfs)
        )
        self.file_system_svc.write_as_json(vocabulary, output_vocabulary_path)
        self.file_system_svc.write_as_json(lemma_embeddings, output_sim_matrix_path)
        save_all_sliding_avg_lemmas(all_sliding_avg_lemmas, output_filepaths)


impl = ProgramsExtractorSvc()
