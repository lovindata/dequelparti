import os
from dataclasses import dataclass
from typing import List, Set, Tuple

from loguru import logger

from src.confs import envs_conf
from src.modules.file_system import file_system_svc
from src.modules.nlp import nlp_svc


@dataclass
class ProgramsExtractorSvc:
    envs_conf = envs_conf.impl
    nlp_svc = nlp_svc.impl
    file_system_svc = file_system_svc.impl

    def run(self) -> None:  # TODO James: Cleanup code and fix this method
        def build_input_and_output_filepaths() -> Tuple[List[str], str]:
            filenames = os.listdir(self.envs_conf.dirpath_input)
            input_filepaths = [
                os.path.join(self.envs_conf.dirpath_input, filename)
                for filename in filenames
            ]
            output_filepath = os.path.join(
                self.envs_conf.dirpath_output, "vocabulary.json"
            )
            return input_filepaths, output_filepath

        input_filepaths, output_filepath = build_input_and_output_filepaths()

        vocabulary: Set[str] = set()
        for input_filepath in input_filepaths:
            vocabulary.update(
                self._extract_tokens_and_save_as_json(
                    input_filepath, self.envs_conf.dirpath_output
                )
            )
        logger.info(f"Vocabulary size is {len(vocabulary)}.")
        output_filepath = os.path.join(envs_conf.impl.dirpath_output, "vocabulary.json")
        logger.info(f"Saving vocabulary at '{output_filepath}'.")
        file_system_svc.impl.write_as_json(sorted(list(vocabulary)), output_filepath)

    def _extract_tokens_and_save_as_json(
        self, input_pdf_filepath: str, output_tokens_dirpath: str
    ) -> List[str]:
        logger.info(f"Loading PDF '{input_pdf_filepath}'.")
        pages = self.file_system_svc.read_pdf_pages(input_pdf_filepath)

        logger.info(f"Extracting tokens from PDF '{input_pdf_filepath}'.")
        tokens = [
            token for page in pages for token in self.nlp_svc.compute_tokens(page)
        ]

        output_filename = os.path.basename(input_pdf_filepath).split(".")[0] + ".json"
        output_filepath = os.path.join(output_tokens_dirpath, output_filename)
        logger.info(f"Saving tokens at '{output_filepath}'.")
        self.file_system_svc.write_as_json(tokens, output_filepath)
        return tokens


impl = ProgramsExtractorSvc()
