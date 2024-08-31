from dataclasses import dataclass


@dataclass(frozen=True)
class EnvsConf:
    prgms_dirpath: str = "../frontend/public/prgms"
    embedding_model_dirpath: str = "../frontend/public/artifacts/all-MiniLM-L6-v2"
    embedding_model_max_length: int = 32
    embedding_model_stride: int = 16
    vector_database_filepath: str = "../frontend/public/artifacts/vector-database.json"


impl = EnvsConf()
