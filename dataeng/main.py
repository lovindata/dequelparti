from loguru import logger

from src.confs import envs_conf
from src.modules.file_system import file_system_svc
from src.modules.llm_prep import llm_prep_svc
from src.modules.nlp_classifier import nlp_classifier_svc
from src.modules.vocabulary_prep import vocabulary_prep_svc


@logger.catch
def main() -> None:
    vocabulary = vocabulary_prep_svc.impl.compute_vocabulary()
    pdfs = file_system_svc.impl.read_pdfs(envs_conf.impl.input_dirpath)
    llm_rows = llm_prep_svc.impl.compute_llm_rows(pdfs, vocabulary)
    nlp_classifier_svc.impl.train_and_export_classifier(llm_rows, vocabulary)


if __name__ == "__main__":
    main()
