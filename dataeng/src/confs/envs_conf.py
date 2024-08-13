import os
from dataclasses import dataclass


@dataclass(frozen=True)
class EnvsConf:
    input_dirpath: str = os.path.abspath(
        os.getenv("DEQUELPARTI_INPUT_DIRPATH", "../frontend/public/programs")
    )
    embedding_model_dirpath = os.path.abspath(
        os.getenv(
            "DEQUELPARTI_EMBEDDING_MODEL_DIRPATH",
            "../frontend/public/artifacts/all-MiniLM-L6-v2",
        )
    )
    ollama_ip: str = os.getenv("DEQUELPARTI_OLLAMA_IP", "localhost")
    ollama_port: int = int(os.getenv("DEQUELPARTI_OLLAMA_PORT", "11434"))
    ollama_conf_get_prediction_cache_dirpath: str = os.path.abspath(
        os.getenv(
            "DEQUELPARTI_OLLAMA_CONF_GET_PREDICTION_CACHE_DIRPATH",
            "./data/src/confs/ollama_conf/get_prediction/cache",
        )
    )
    llm_prep_svc_compute_llm_rows_cache_dirpath: str = os.path.abspath(
        os.getenv(
            "DEQUELPARTI_LLM_PREP_SVC_COMPUTE_LLM_ROWS_CACHE_DIRPATH",
            "./data/src/modules/llm_prep_svc/compute_llm_rows/cache",
        )
    )
    prepare_data_svc_prepare_data_cache_dirpath: str = os.path.abspath(
        os.getenv(
            "DEQUELPARTI_PREPARE_DATA_SVC_PREPARE_DATA_CACHE_DIRPATH",
            "./data/src/modules/prepare_data_svc/prepare_data/cache",
        )
    )
    nlp_classifier_svc__get_and_train_model_cache_dirpath: str = os.path.abspath(
        os.getenv(
            "DEQUELPARTI_NLP_CLASSIFIER_SVC_GET__AND_TRAIN_MODEL_CACHE_DIRPATH",
            "./data/src/modules/nlp_classifier_svc/_get_and_train_model/cache",
        )
    )
    vocabulary_dirpath: str = os.path.abspath(
        os.getenv(
            "DEQUELPARTI_VOCABULARY_DIRPATH",
            "../frontend/public/artifacts/dataeng/vocabulary",
        )
    )
    model_filepath: str = os.path.abspath(
        os.getenv(
            "DEQUELPARTI_MODEL_DIRPATH",
            "../frontend/public/artifacts/dataeng/model.onnx",
        )
    )


impl = EnvsConf()
