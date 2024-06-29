import os
from dataclasses import dataclass


@dataclass
class EnvsConf:
    dirpath_input = os.path.abspath(
        os.getenv("DEQUELPARTI_INPUT_DIRPATH", "../frontend/public/programs")
    )
    dirpath_output = os.path.abspath(
        os.getenv("DEQUELPARTI_OUTPUT_DIRPATH", "../frontend/data")
    )


impl = EnvsConf()
