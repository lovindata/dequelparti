import os
from typing import Set

from loguru import logger

from src.confs import envs_conf
from src.modules.shared.services import pdf_svc, write_json_svc


@logger.catch
def main() -> None:
    filenames = os.listdir(envs_conf.impl.dirpath_input)
    input_filepaths = [
        os.path.join(envs_conf.impl.dirpath_input, filename) for filename in filenames
    ]
    vocabulary: Set[str] = set()
    for input_filepath in input_filepaths:
        vocabulary.update(
            pdf_svc.impl.extract_tokens_and_save_as_json(
                input_filepath, envs_conf.impl.dirpath_output
            )
        )
    logger.info(f"Vocabulary size is {len(vocabulary)}.")
    output_filepath = os.path.join(envs_conf.impl.dirpath_output, "vocabulary.json")
    logger.info(f"Saving vocabulary at '{output_filepath}'.")
    write_json_svc.impl.write_as_json(sorted(list(vocabulary)), output_filepath)


if __name__ == "__main__":
    main()
