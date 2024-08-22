from dataclasses import dataclass


@dataclass(frozen=True)
class EnvsConf:
    prgms_dirpath: str = "../frontend/public/prgms"
    embedding_model_dirpath = "../frontend/public/artifacts/all-MiniLM-L6-v2"
    vector_database_filepath = "../frontend/public/artifacts/vector-database.json"


impl = EnvsConf()
