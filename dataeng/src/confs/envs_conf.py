import os
from dataclasses import dataclass


@dataclass
class EnvsConf:
    input_dirpath = os.path.abspath(
        os.getenv("DEQUELPARTI_INPUT_DIRPATH", "../frontend/public/programs")
    )
    output_dirpath = os.path.abspath(
        os.getenv("DEQUELPARTI_OUTPUT_DIRPATH", "../frontend/data")
    )


impl = EnvsConf()
