import os
from dataclasses import dataclass


@dataclass
class EnvsConf:
    vocabulary_dirpath = os.path.abspath(
        os.getenv("DEQUELPARTI_VOCABULARY_DIRPATH", "../frontend/data/vocabulary")
    )

    input_dirpath = os.path.abspath(
        os.getenv("DEQUELPARTI_INPUT_DIRPATH", "../frontend/public/programs")
    )
    output_dirpath = os.path.abspath(
        os.getenv("DEQUELPARTI_OUTPUT_DIRPATH", "../frontend/data")
    )

    ollama_ip = os.getenv("DEQUELPARTI_OLLAMA_IP", "localhost")
    ollama_port = int(os.getenv("DEQUELPARTI_OLLAMA_PORT", "11434"))
    ollama_cache_dirpath = os.path.abspath(
        os.getenv(
            "DEQUELPARTI_OLLAMA_CACHE_DIRPATH", "./data/src/confs/ollama_conf/cache"
        )
    )

    sliding_avg_lemmas_window = int(
        os.getenv("DEQUELPARTI_SLIDING_AVG_LEMMAS_WINDOW", "50")
    )
    sliding_avg_lemmas_stride = int(
        os.getenv("DEQUELPARTI_SLIDING_AVG_LEMMAS_STRIDE", "10")
    )


impl = EnvsConf()
