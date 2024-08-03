import os
from dataclasses import dataclass


@dataclass(frozen=True)
class EnvsConf:
    vocabulary_dirpath: str = os.path.abspath(
        os.getenv("DEQUELPARTI_VOCABULARY_DIRPATH", "../frontend/data/vocabulary")
    )

    input_dirpath: str = os.path.abspath(
        os.getenv("DEQUELPARTI_INPUT_DIRPATH", "../frontend/public/programs")
    )
    output_dirpath: str = os.path.abspath(
        os.getenv("DEQUELPARTI_OUTPUT_DIRPATH", "../frontend/data")
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

    sliding_avg_lemmas_window: int = int(
        os.getenv("DEQUELPARTI_SLIDING_AVG_LEMMAS_WINDOW", "50")
    )
    sliding_avg_lemmas_stride: int = int(
        os.getenv("DEQUELPARTI_SLIDING_AVG_LEMMAS_STRIDE", "10")
    )


impl = EnvsConf()
