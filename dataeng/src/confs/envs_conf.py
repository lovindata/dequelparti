from dataclasses import dataclass


@dataclass(frozen=True)
class EnvsConf:
    input_dirpath: str = "../frontend/public/programs"
    embedding_model_dirpath = "../frontend/public/artifacts/all-MiniLM-L6-v2"


impl = EnvsConf()
