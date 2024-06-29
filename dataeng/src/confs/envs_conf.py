import os
from dataclasses import dataclass


@dataclass
class EnvsConf:
    input_prgm_ensemble = os.getenv(
        "DEQUELPARTI_PRGM_ENSEMBLE", "../frontend/public/prgm_ensemble.pdf"
    )
    input_prgm_nouveau_front_populaire = os.getenv(
        "DEQUELPARTI_PRGM_NOUVEAU_FRONT_POPULAIRE",
        "../frontend/public/prgm_nouveau_front_populaire.pdf",
    )
    input_prgm_rassemblement_national = os.getenv(
        "DEQUELPARTI_PRGM_RASSEMBLEMENT_NATIONAL",
        "../frontend/public/prgm_rassemblement_national.pdf",
    )
    output_data_path = os.getenv("DEQUELPARTI_OUTPUT_PATH", "../frontend/data")


impl = EnvsConf()
