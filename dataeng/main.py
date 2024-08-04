from loguru import logger

from src.confs import envs_conf
from src.modules.file_system import file_system_svc
from src.modules.llm_prep import llm_prep_svc
from src.modules.vocabulary_prep import vocabulary_prep_svc


@logger.catch
def main() -> None:
    vocabulary = vocabulary_prep_svc.impl.compute_vocabulary()
    pdfs = file_system_svc.impl.read_pdfs(envs_conf.impl.input_dirpath)
    llm_prep_rows_per_pdf = llm_prep_svc.impl.compute_llm_rows(pdfs, vocabulary)


if __name__ == "__main__":
    main()
